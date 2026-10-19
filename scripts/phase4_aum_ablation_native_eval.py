#!/usr/bin/env python
"""Phase 4 re-run: SSM-internal quantization ablations for AuM, NATIVE
LENGTH ONLY, with a real class-balanced sample.

The first completed Phase 4 run (scripts/phase4_aum_ablation_eval.py, see
DECISIONS.md's 2026-10-13 entry) found the recurrent state h is the
fragile tensor and that divergence doesn't compound with length -- but it
used only 5 clips per length, and because the 1pc manifest lists each
source word's variants together, every extended length covered just 2
distinct words. That entry's own "not decided" section flagged the fix:
a bigger, class-balanced sample at NATIVE length only -- extended lengths
are already at AuM's full-precision chance floor (Phase 1/3), so more
samples there adds no information; native is the one length with any
accuracy headroom at all, and native clips are cheap (~0.3-0.9s/clip).

Uses results/phase1_eval_manifest.json (the full 4-clips/class design,
NOT the _1pc/_3pc budget-scoped variants) filtered to length_sec=="native"
-- 140 clips, exactly 4 per word across all 35 classes, already built and
committed, no new sampling logic needed. This directly fixes the
"only 2 distinct words" problem without touching the general-purpose
phase4_aum_ablation_eval.py (which stays as-is for any future length-grid
re-run).

Same 12 conditions (delta/A/BC/state x 8/6/4-bit) as the first run, same
calibration (Phase 3's existing 256-clip/seed train-split set). State
divergence is logged for every clip here (not a small subset) since
native-length clips are cheap enough that this isn't the bottleneck the
way it was at 160s.

MUST pass scripts/phase4_validate_quantized_scan.py first in this
environment -- same discipline as every Phase 4 run.

Run inside the AuM venv, from third_party/Audio-Mamba-AuM/:
    cd third_party/Audio-Mamba-AuM
    cp ../../scripts/phase4_aum_ablation_native_eval.py .
    python3 phase4_aum_ablation_native_eval.py
"""
import csv
import json
import os
import sys
import time
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
MANIFEST_PATH = PROJECT_ROOT / "results" / "phase1_eval_manifest.json"  # full design, NOT _1pc/_3pc
RESULTS_PATH = PROJECT_ROOT / "results" / "phase4_aum_ablation_native.jsonl"
DIVERGENCE_PATH = PROJECT_ROOT / "results" / "phase4_state_divergence_native.json"

NUM_MEL_BINS = 128
DATASET_MEAN = -6.845978
DATASET_STD = 5.5654526
NATIVE_PAD_FRAMES = 128

TARGETS = ("delta", "A", "BC", "state")
BITS = (8, 6, 4)
CALIBRATION_SEED = int(os.environ.get("PHASE4_CALIBRATION_SEED", 0))

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
    native_items = [item for item in manifest if item["length_sec"] == "native"]
    distinct_words = sorted(set(mid2word[item["labels"]] for item in native_items))
    print(f"Native-length clips: {len(native_items)}, distinct words: {len(distinct_words)} "
          f"(expect 140 clips, 35 words -- 4/class)")
    assert len(distinct_words) == 35, (
        f"expected all 35 classes represented at native length, got {len(distinct_words)} -- "
        "check results/phase1_eval_manifest.json wasn't swapped for a budget-scoped variant"
    )

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

        model = build_model(NATIVE_PAD_FRAMES, n_classes, device)
        scales = apply_ssm_tensor_scale_by_index(model, scales_by_index)
        patched, _ = patch_model_for_ssm_ablation(model, target, bits, calibration_scales=scales)

        correct = 0
        with torch.no_grad():
            for item in native_items:
                fbank = compute_fbank(item["wav"], pad_to_frames=NATIVE_PAD_FRAMES)
                inputs = fbank.unsqueeze(0).to(device)
                logits = torch.sigmoid(model(inputs))
                pred_idx = int(logits.argmax(-1).item())
                pred_word = idx2word[pred_idx]
                true_word = mid2word[item["labels"]]
                is_correct = pred_word == true_word
                correct += int(is_correct)
                results.append({
                    "condition": label, "target": target, "bits": bits,
                    "wav": item["wav"], "true_word": true_word,
                    "pred_word": pred_word, "correct": is_correct,
                })
        unpatch_model(patched)

        # Full divergence logging (every clip) -- native length is cheap
        # enough that this isn't the bottleneck the 160s length was.
        div_patched, div_log = patch_model_for_ssm_ablation(
            model, target, bits, calibration_scales=scales, track_divergence=True
        )
        try:
            for item in native_items:
                div_log.clear()
                fbank = compute_fbank(item["wav"], pad_to_frames=NATIVE_PAD_FRAMES)
                inputs = fbank.unsqueeze(0).to(device)
                with torch.no_grad():
                    model(inputs)
                divergence_records.append({
                    "condition": label, "target": target, "bits": bits,
                    "wav": item["wav"], "per_layer": div_log.copy(),
                })
        finally:
            unpatch_model(div_patched)

        acc = correct / len(native_items)
        elapsed = time.time() - t_cond0
        print(f"=== {label} === acc={acc:.4f} n={len(native_items)} elapsed_s={elapsed:.1f}")
        del model
        torch.cuda.empty_cache()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")

    json.dump(divergence_records, open(DIVERGENCE_PATH, "w"))
    print(f"Wrote {len(divergence_records)} divergence records to {DIVERGENCE_PATH}")


if __name__ == "__main__":
    main()
