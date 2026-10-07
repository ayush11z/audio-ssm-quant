#!/usr/bin/env python
"""Phase 3: standard quantization grid for AuM (brief section 6/Phase 3:
W8A16, W4A16, W8A8-per-tensor, W8A8-per-token x both models x all Phase 1
lengths x 3 calibration seeds). Counterpart: scripts/phase3_ast_eval.py.

Reuses Phase 1's AuM model loading/fbank preprocessing unchanged (see
scripts/phase1_aum_eval.py). Activation quantization goes through
src/ssmquant/models/aum_quantized_mamba.py's custom forward (gate-
validated bit-exact against AuM's real fused BiMambaInnerFn forward pass,
see DECISIONS.md) instead of src/ssmquant/quant/apply.py's
forward_pre_hook-based ActivationFakeQuantHooks, since AuM's Mamba blocks
bypass nn.Linear.forward() entirely. The one real nn.Linear AuM calls
normally -- model.head, the classification head -- still goes through the
ordinary hook-based path, since it's a real nn.Linear.forward() call like
any AST layer.

AUM-SPECIFIC WRINKLE THIS SCRIPT HANDLES: AuM's own architecture
(FlexiPatchEmbed/FlexiPosEmbed) needs a FRESH model instance built at each
input length (confirmed in Phase 1 -- unlike AST, which reuses one model
and interpolates its position embeddings in place). That means, within one
(condition, seed), every length group gets its own newly-constructed model
with its own new Mamba-block Python objects -- so a per_tensor calibration
pass run on one model instance can't be looked up by module identity on a
different instance's modules. Calibration therefore runs ONCE per
(condition, seed) on a model built at native length (matching AST's
design: one static scale from native-length clips, reused UNCHANGED across
every extended eval length -- whether that native-calibrated scale holds
up at 20x the training length is the thing being measured here, not
something to paper over by recalibrating per length), then the resulting
per-layer scales are re-applied by position to each length group's freshly
built model via apply_bimamba_v1_scales_by_index / a plain scalar re-key
for the single head module.

Per-condition weight-only runs (W8A16/W4A16) and the per_token activation
run (dynamic, no calibration) only need one seed each, same reasoning as
scripts/phase3_ast_eval.py.

Run inside the AuM venv (same as Phase 1 / the custom-forward validation
gate), from third_party/Audio-Mamba-AuM/ so its relative imports resolve:

    cd third_party/Audio-Mamba-AuM
    cp ../../scripts/phase3_aum_eval.py .
    python3 phase3_aum_eval.py
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

from ssmquant.models import (
    apply_bimamba_v1_scales_by_index,
    calibrate_bimamba_v1_scales_by_index,
    patch_model_for_quantized_forward,
    unpatch_model,
)
from ssmquant.quant import ActivationFakeQuantHooks, calibrate_activation_scales, quantize_linear_weights_

ROOT = Path(__file__).resolve().parent  # third_party/Audio-Mamba-AuM when copied there, per docstring
PROJECT_ROOT = ROOT.parent.parent  # back to the repo root, where results/ live
CHECKPOINT_PATH = "exps/speechcommands/models/aum-base_audioset-spc_v2.pth"
LABEL_CSV = "exps/speechcommands/data/speechcommands_class_labels_indices.csv"
_MANIFEST_SUFFIX = "" if "PHASE1_CLIPS_PER_CLASS" not in os.environ else f"_{os.environ['PHASE1_CLIPS_PER_CLASS']}pc"
MANIFEST_PATH = PROJECT_ROOT / "results" / f"phase1_eval_manifest{_MANIFEST_SUFFIX}.json"
RESULTS_PATH = PROJECT_ROOT / "results" / f"phase3_aum_quantized{_MANIFEST_SUFFIX}.jsonl"

NUM_MEL_BINS = 128
DATASET_MEAN = -6.845978  # from AuM's exps/speechcommands/aum_eval.sh (Phase 1)
DATASET_STD = 5.5654526
NATIVE_PAD_FRAMES = 128  # matches phase1_aum_eval.py / AuM's training convention

W8 = {"bits": 8, "granularity": "per_channel", "symmetric": True}
W4 = {"bits": 4, "granularity": "per_channel", "symmetric": True}
A8_PER_TENSOR = {"bits": 8, "granularity": "per_tensor", "symmetric": True}
A8_PER_TOKEN = {"bits": 8, "granularity": "per_token", "symmetric": True}
A_NONE = {"bits": None}

CONDITIONS = [
    {"name": "w8a16", "weights": W8, "activations": A_NONE, "seeds": [None]},
    {"name": "w4a16", "weights": W4, "activations": A_NONE, "seeds": [None]},
    {"name": "w8a8_per_tensor", "weights": W8, "activations": A8_PER_TENSOR, "seeds": [0, 1, 2]},
    {"name": "w8a8_per_token", "weights": W8, "activations": A8_PER_TOKEN, "seeds": [None]},
]


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
        bimamba_type="v1",  # Fo-Bi
    )
    model.to(device).eval()
    return model


def load_calibration_fbanks(seed, device):
    records = json.load(open(PROJECT_ROOT / "results" / f"phase3_calibration_seed{seed}.json"))
    fbanks = [compute_fbank(r["wav"], pad_to_frames=NATIVE_PAD_FRAMES).unsqueeze(0).to(device) for r in records]
    return fbanks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("FATAL: needs CUDA (causal_conv1d_fn / selective_scan_fn require it).")
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
    length_order = sorted(by_length.items(), key=lambda kv: (kv[0] != "native", kv[0]))

    results = []
    for condition in CONDITIONS:
        for seed in condition["seeds"]:
            t_cond0 = time.time()
            label = f"{condition['name']}" + (f"_seed{seed}" if seed is not None else "")

            mamba_scales_by_index = None
            head_scale = None
            if condition["activations"].get("granularity") == "per_tensor":
                calib_model = build_model(NATIVE_PAD_FRAMES, n_classes, device)
                quantize_linear_weights_(calib_model, condition["weights"])
                calib_fbanks = load_calibration_fbanks(seed, device)
                mamba_scales_by_index = calibrate_bimamba_v1_scales_by_index(
                    calib_model, calib_fbanks, bits=condition["activations"]["bits"]
                )
                head_calib = calibrate_activation_scales(
                    calib_model, calib_fbanks, bits=condition["activations"]["bits"]
                )
                # Every Mamba-internal Linear's forward_pre_hook never fires
                # (bypassed, same reason this whole module exists) -- only
                # model.head actually calls nn.Linear.forward(). Exactly one
                # entry confirms that's still true and nothing else in AuM's
                # architecture started routing through a real nn.Linear call.
                assert len(head_calib) == 1, (
                    f"expected exactly 1 calibrated nn.Linear (the head); got {len(head_calib)} "
                    "-- did AuM's architecture change?"
                )
                head_scale = next(iter(head_calib.values()))
                del calib_model
                torch.cuda.empty_cache()

            print(f"=== {label} (weights={condition['weights']}, activations={condition['activations']}) ===")
            for length_sec, items in length_order:
                t0 = time.time()
                pad_to = NATIVE_PAD_FRAMES if length_sec == "native" else None
                first_fbank = compute_fbank(items[0]["wav"], pad_to_frames=pad_to)
                target_length = first_fbank.shape[0]

                model = build_model(target_length, n_classes, device)
                n_quantized = quantize_linear_weights_(model, condition["weights"])

                mamba_scales = (
                    apply_bimamba_v1_scales_by_index(model, mamba_scales_by_index)
                    if mamba_scales_by_index is not None else None
                )
                mamba_patch = patch_model_for_quantized_forward(
                    model, condition["weights"], condition["activations"], calibration_scales=mamba_scales
                )
                head_scales = {model.head: head_scale} if head_scale is not None else None

                correct = 0
                with ActivationFakeQuantHooks(model, condition["activations"], calibration_scales=head_scales):
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
                                "condition": condition["name"],
                                "seed": seed,
                                "length_sec": length_sec,
                                "position": item["position"],
                                "wav": item["wav"],
                                "true_word": true_word,
                                "pred_word": pred_word,
                                "correct": is_correct,
                            })

                unpatch_model(mamba_patch)
                acc = correct / len(items)
                elapsed = time.time() - t0
                print(f"  length={length_sec} n={len(items)} frames={target_length} "
                      f"quantized_linear={n_quantized} acc={acc:.4f} elapsed_s={elapsed:.1f}")
                del model
                torch.cuda.empty_cache()

            print(f"  ({label} total: {time.time() - t_cond0:.1f}s)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
