#!/usr/bin/env python
"""Phase 4: SSM-internal quantization ablations for AuM (brief section 7:
"which internal tensor -- Δ, A/Ā, B, C, recurrent state h -- drives
degradation?"). AuM only -- AST has no SSM internals, nothing to ablate
there. Counterpart to scripts/phase3_aum_eval.py, but quantizing the SSM
scan's own tensors (via src/ssmquant/models/aum_quantized_scan.py)
instead of ordinary Linear-layer weights/activations. Linear layers stay
full precision throughout this script, per the Phase 4 plan's explicit
choice -- isolates the SSM-internal effect cleanly, matching the brief's
H1 (Phase 3, Linears)/H2 (Phase 4, SSM tensors) split.

Conditions: 4 targets (delta, A, BC, state) x 3 bit-widths (8, 6, 4),
matching configs/quant/ssm_{delta,A,BC,state}.yaml's bits sweep exactly.
Calibration (delta/BC/state; "A" needs none, see aum_quantized_scan.py)
reuses Phase 3's existing 256-clip/seed train-split calibration set
(results/phase3_calibration_seed{0,1,2}.json) -- calibrated once per
condition at native length, then re-applied by layer position to each
length group's freshly-built model (same AuM-specific wrinkle as Phase 3:
FlexiPatchEmbed/FlexiPosEmbed need a fresh model instance per length).

MUST pass scripts/phase4_validate_quantized_scan.py first -- if the
reference-scan reimplementation doesn't match the real fused-kernel
forward at full precision, nothing built on it can be trusted.

For a small fixed subset of clips per length, also logs the per-timestep
state-divergence trace (||h_quantized - h_full_precision|| at every
timestep) -- the metric that keeps working even where Phase 3 found
AuM's top-1 accuracy already at its chance floor. This is the SLOW path
(doubles the scan's per-step compute AND only applies when explicitly
requested), so it's deliberately limited to PHASE4_DIVERGENCE_CLIPS clips
per length, not the full eval sample.

Run inside the AuM venv (same as Phase 0/1/2/3), from
third_party/Audio-Mamba-AuM/:
    cd third_party/Audio-Mamba-AuM
    cp ../../scripts/phase4_aum_ablation_eval.py .
    python3 phase4_aum_ablation_eval.py
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

sys.path.append(".")  # third_party/Audio-Mamba-AuM, for `import src.models`
import src.models as models

from ssmquant.models.aum_quantized_mamba import unpatch_model
from ssmquant.models.aum_quantized_scan import (
    apply_ssm_tensor_scale_by_index,
    calibrate_ssm_tensor_scale_by_index,
    patch_model_for_ssm_ablation,
)

ROOT = Path(__file__).resolve().parent  # third_party/Audio-Mamba-AuM when copied there
PROJECT_ROOT = ROOT.parent.parent
CHECKPOINT_PATH = "exps/speechcommands/models/aum-base_audioset-spc_v2.pth"
LABEL_CSV = "exps/speechcommands/data/speechcommands_class_labels_indices.csv"
_MANIFEST_SUFFIX = "" if "PHASE1_CLIPS_PER_CLASS" not in os.environ else f"_{os.environ['PHASE1_CLIPS_PER_CLASS']}pc"
MANIFEST_PATH = PROJECT_ROOT / "results" / f"phase1_eval_manifest{_MANIFEST_SUFFIX}.json"
RESULTS_PATH = PROJECT_ROOT / "results" / f"phase4_aum_ablation{_MANIFEST_SUFFIX}.jsonl"
DIVERGENCE_PATH = PROJECT_ROOT / "results" / f"phase4_state_divergence{_MANIFEST_SUFFIX}.json"

NUM_MEL_BINS = 128
DATASET_MEAN = -6.845978  # from AuM's exps/speechcommands/aum_eval.sh (Phase 1)
DATASET_STD = 5.5654526
NATIVE_PAD_FRAMES = 128

TARGETS = ("delta", "A", "BC", "state")
BITS = (8, 6, 4)
CALIBRATION_SEED = int(os.environ.get("PHASE4_CALIBRATION_SEED", 0))
# Small by design -- divergence tracking doubles the scan's per-step
# compute AND this is new, unvalidated-at-scale code (see the gate
# script's timing output before changing this).
N_DIVERGENCE_CLIPS = int(os.environ.get("PHASE4_DIVERGENCE_CLIPS", 2))

CONDITIONS = [{"name": f"{target}_b{bits}", "target": target, "bits": bits}
              for target in TARGETS for bits in BITS]


def mid_to_word():
    m = {}
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            m[row["mid"]] = row["display_name"].strip('"')
    return m


def compute_fbank(wav_path, pad_to_frames=None):
    waveform, sr = torchaudio.load(str(PROJECT_ROOT / wav_path))
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0, frame_shift=10,
    )
    if pad_to_frames is not None:
        n = fbank.shape[0]
        if n < pad_to_frames:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_to_frames - n))(fbank)
        elif n > pad_to_frames:
            fbank = fbank[:pad_to_frames, :]
    fbank = (fbank - DATASET_MEAN) / (DATASET_STD * 2)
    return fbank


def build_model(target_length, n_classes, device):
    model = models.AudioMamba(
        spectrogram_size=(NUM_MEL_BINS, target_length),
        patch_size=(16, 16),
        strides=(16, 16),
        embed_dim=768,
        num_classes=n_classes,
        imagenet_pretrain=False,
        imagenet_pretrain_path=None,
        aum_pretrain=True,
        aum_pretrain_path=CHECKPOINT_PATH,
        bimamba_type="v1",
    )
    model.to(device).eval()
    return model


def load_calibration_fbanks(seed, device):
    records = json.load(open(PROJECT_ROOT / "results" / f"phase3_calibration_seed{seed}.json"))
    return [compute_fbank(r["wav"], pad_to_frames=NATIVE_PAD_FRAMES).unsqueeze(0).to(device) for r in records]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("FATAL: needs CUDA (causal_conv1d_fn requires it).")
        raise SystemExit(1)

    mid2word = mid_to_word()
    word2idx = {}
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            word2idx[row["display_name"].strip('"')] = int(row["index"])
    idx2word = {i: w for w, i in word2idx.items()}
    n_classes = len(word2idx)

    manifest = json.load(open(MANIFEST_PATH))["data"]
    by_length = defaultdict(list)
    for item in manifest:
        by_length[item["length_sec"]].append(item)
    # Subsample to a fixed count per length group, independent of the
    # manifest's own per-class clip count -- the step-loop scan's real
    # per-clip cost (measured by the Phase 4 gate: 85.68s/clip at 160s)
    # makes the manifest's full clip counts (35-105/length) take ~56hr
    # for the full 12-condition grid. Deliberately not stratified by
    # class like Phase 1/3's manifests -- this is a cost-scoped accuracy
    # read plus divergence curves, not a class-balanced evaluation.
    clips_per_length = int(os.environ.get("PHASE4_CLIPS_PER_LENGTH", 5))
    by_length = {length: items[:clips_per_length] for length, items in by_length.items()}
    length_order = sorted(by_length.items(), key=lambda kv: (kv[0] != "native", kv[0]))

    calib_fbanks = load_calibration_fbanks(CALIBRATION_SEED, device)

    results = []
    divergence_records = []
    for condition in CONDITIONS:
        target, bits = condition["target"], condition["bits"]
        label = condition["name"]
        t_cond0 = time.time()

        calib_model = build_model(NATIVE_PAD_FRAMES, n_classes, device)
        scales_by_index = calibrate_ssm_tensor_scale_by_index(calib_model, calib_fbanks, target, bits)
        del calib_model
        torch.cuda.empty_cache()

        print(f"=== {label} ===")
        for length_sec, items in length_order:
            t0 = time.time()
            pad_to = NATIVE_PAD_FRAMES if length_sec == "native" else None
            first_fbank = compute_fbank(items[0]["wav"], pad_to_frames=pad_to)
            target_length = first_fbank.shape[0]

            model = build_model(target_length, n_classes, device)
            scales = apply_ssm_tensor_scale_by_index(model, scales_by_index)
            patched, _ = patch_model_for_ssm_ablation(model, target, bits, calibration_scales=scales)

            correct = 0
            with torch.no_grad():
                for item in items:
                    fbank = compute_fbank(item["wav"], pad_to_frames=pad_to)
                    inputs = fbank.unsqueeze(0).to(device)
                    logits = torch.sigmoid(model(inputs))
                    pred_idx = int(logits.argmax(-1).item())
                    pred_word = idx2word[pred_idx]
                    true_word = mid2word[item["labels"]]
                    is_correct = pred_word == true_word
                    correct += int(is_correct)
                    results.append({
                        "condition": label, "target": target, "bits": bits,
                        "length_sec": length_sec, "position": item["position"],
                        "wav": item["wav"], "true_word": true_word,
                        "pred_word": pred_word, "correct": is_correct,
                    })

            unpatch_model(patched)

            # State-divergence trace for a small fixed subset of clips --
            # separate patch call (track_divergence=True) so the accuracy
            # loop above stays at its normal (cheaper, non-doubled) cost.
            div_patched, div_log = patch_model_for_ssm_ablation(
                model, target, bits, calibration_scales=scales, track_divergence=True
            )
            try:
                for item in items[:N_DIVERGENCE_CLIPS]:
                    div_log.clear()
                    fbank = compute_fbank(item["wav"], pad_to_frames=pad_to)
                    inputs = fbank.unsqueeze(0).to(device)
                    with torch.no_grad():
                        model(inputs)
                    # div_log now holds one {"fwd": [...], "bwd": [...]}
                    # dict per Mamba layer, in layer order, for this clip.
                    divergence_records.append({
                        "condition": label, "target": target, "bits": bits,
                        "length_sec": length_sec, "wav": item["wav"],
                        "per_layer": div_log.copy(),
                    })
            finally:
                unpatch_model(div_patched)

            acc = correct / len(items)
            elapsed = time.time() - t0
            print(f"  length={length_sec} n={len(items)} frames={target_length} "
                  f"acc={acc:.4f} elapsed_s={elapsed:.1f}")
            del model
            torch.cuda.empty_cache()

        print(f"  ({label} total: {time.time() - t_cond0:.1f}s)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")

    json.dump(divergence_records, open(DIVERGENCE_PATH, "w"))
    print(f"Wrote {len(divergence_records)} divergence records to {DIVERGENCE_PATH}")


if __name__ == "__main__":
    main()
