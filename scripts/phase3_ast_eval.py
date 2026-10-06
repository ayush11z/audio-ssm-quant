#!/usr/bin/env python
"""Phase 3: standard quantization grid for AST (brief section 6/Phase 3:
W8A16, W4A16, W8A8-per-tensor, W8A8-per-token x both models x all Phase 1
lengths x 3 calibration seeds). Counterpart: scripts/phase3_aum_eval.py.

Reuses Phase 1's model loading, fbank preprocessing, and AST position-
embedding interpolation unchanged (see scripts/phase1_ast_eval.py and
DECISIONS.md) -- quantization is an orthogonal axis applied on top.

Conditions with a per_tensor activation (static, calibrated) component
need their own calibration run (brief: 3 seeds, 256 train-split clips,
scripts/phase3_build_calibration_set.py). Weight-only conditions
(W8A16/W4A16) and the per_token condition (dynamic, no calibration) only
need one run each -- running them 3x under different "seed" labels would
be identical, wasted GPU time (see src/ssmquant/quant/apply.py docstring
for why per_token doesn't need calibration at all).

The model is reloaded from the checkpoint fresh for every condition, not
quantized-in-place repeatedly -- repeated quantization of already-
quantized weights would compound error in a way that doesn't correspond
to anything in the brief's experimental design.

Run on a GPU instance (same AST venv as Phase 1):
    python3 scripts/phase3_ast_eval.py
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from ssmquant.quant import ActivationFakeQuantHooks, calibrate_activation_scales, quantize_linear_weights_

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = "MIT/ast-finetuned-speech-commands-v2"
_MANIFEST_SUFFIX = "" if "PHASE1_CLIPS_PER_CLASS" not in os.environ else f"_{os.environ['PHASE1_CLIPS_PER_CLASS']}pc"
MANIFEST_PATH = ROOT / "results" / f"phase1_eval_manifest{_MANIFEST_SUFFIX}.json"
LABEL_CSV = ROOT / "third_party" / "Audio-Mamba-AuM" / "exps" / "speechcommands" / "data" / "speechcommands_class_labels_indices.csv"
RESULTS_PATH = ROOT / "results" / f"phase3_ast_quantized{_MANIFEST_SUFFIX}.jsonl"

NUM_MEL_BINS = 128
PATCH_SIZE = None
FREQ_STRIDE = None
TIME_STRIDE = None

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
    import csv

    m = {}
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            m[row["mid"]] = row["display_name"].strip('"')
    return m


def compute_fbank(wav_path, extractor_mean, extractor_std, pad_to_frames=None):
    audio, sr = sf.read(str(ROOT / wav_path), dtype="float32")
    waveform = torch.from_numpy(audio).unsqueeze(0)
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
    fbank = (fbank - extractor_mean) / (extractor_std * 2)
    return fbank


def time_out_dim(n_frames):
    return (n_frames - PATCH_SIZE) // TIME_STRIDE + 1


def interpolated_position_embeddings(base_pos_embed, base_time_dim, freq_dim, new_time_dim):
    hidden = base_pos_embed.shape[-1]
    prefix = base_pos_embed[:, :2, :]
    patch_pos = base_pos_embed[:, 2:, :]
    grid = patch_pos.reshape(1, freq_dim, base_time_dim, hidden).permute(0, 3, 1, 2)
    new_grid = F.interpolate(grid, size=(freq_dim, new_time_dim), mode="bicubic", align_corners=False)
    new_patch_pos = new_grid.permute(0, 2, 3, 1).reshape(1, freq_dim * new_time_dim, hidden)
    return torch.cat([prefix, new_patch_pos], dim=1)


def forward_logits(model, inputs):
    return model.classifier(model.audio_spectrogram_transformer(inputs).pooler_output)


def load_calibration_fbanks(seed, extractor_mean, extractor_std, pad_to_frames, device):
    records = json.load(open(ROOT / "results" / f"phase3_calibration_seed{seed}.json"))
    fbanks = []
    for r in records:
        fbank = compute_fbank(r["wav"], extractor_mean, extractor_std, pad_to_frames=pad_to_frames)
        fbanks.append(fbank.unsqueeze(0).to(device))
    return fbanks


def main():
    global PATCH_SIZE, FREQ_STRIDE, TIME_STRIDE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    ext_mean, ext_std = extractor.mean, extractor.std
    mid2word = mid_to_word()

    manifest = json.load(open(MANIFEST_PATH))["data"]
    by_length = defaultdict(list)
    for item in manifest:
        by_length[item["length_sec"]].append(item)
    length_order = sorted(by_length.items(), key=lambda kv: (kv[0] != "native", kv[0]))

    results = []
    for condition in CONDITIONS:
        for seed in condition["seeds"]:
            t_cond0 = time.time()
            model = ASTForAudioClassification.from_pretrained(CHECKPOINT).to(device).eval()
            PATCH_SIZE = model.config.patch_size
            FREQ_STRIDE = model.config.frequency_stride
            TIME_STRIDE = model.config.time_stride
            id2label = {str(k): v for k, v in model.config.id2label.items()}
            base_freq_dim = (NUM_MEL_BINS - PATCH_SIZE) // FREQ_STRIDE + 1
            base_time_dim = (model.config.max_length - PATCH_SIZE) // TIME_STRIDE + 1
            base_pos_embed = model.audio_spectrogram_transformer.embeddings.position_embeddings.data.clone()
            embeddings_module = model.audio_spectrogram_transformer.embeddings

            n_quantized = quantize_linear_weights_(model, condition["weights"])

            calibration_scales = None
            if condition["activations"].get("granularity") == "per_tensor":
                calib_fbanks = load_calibration_fbanks(seed, ext_mean, ext_std, model.config.max_length, device)
                calibration_scales = calibrate_activation_scales(
                    model, calib_fbanks, bits=condition["activations"]["bits"], forward_fn=forward_logits
                )

            label = f"{condition['name']}" + (f"_seed{seed}" if seed is not None else "")
            print(f"=== {label}: quantized {n_quantized} Linear layers "
                  f"(weights={condition['weights']}, activations={condition['activations']}) ===")

            with ActivationFakeQuantHooks(model, condition["activations"], calibration_scales=calibration_scales):
                for length_sec, items in length_order:
                    t0 = time.time()
                    correct = 0
                    n_frames_seen = None
                    pad_to = model.config.max_length if length_sec == "native" else None
                    for item in items:
                        fbank = compute_fbank(item["wav"], ext_mean, ext_std, pad_to_frames=pad_to)
                        n_frames = fbank.shape[0]
                        if n_frames_seen is None:
                            n_frames_seen = n_frames
                            new_time_dim = time_out_dim(n_frames)
                            if new_time_dim != base_time_dim:
                                new_pos_embed = interpolated_position_embeddings(
                                    base_pos_embed, base_time_dim, base_freq_dim, new_time_dim
                                )
                            else:
                                new_pos_embed = base_pos_embed
                            embeddings_module.position_embeddings = torch.nn.Parameter(new_pos_embed.to(device))

                        inputs = fbank.unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = forward_logits(model, inputs)
                        pred_id = int(logits.argmax(-1).item())
                        pred_word = id2label[str(pred_id)]
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
                    acc = correct / len(items)
                    elapsed = time.time() - t0
                    print(f"  length={length_sec} n={len(items)} frames={n_frames_seen} "
                          f"acc={acc:.4f} elapsed_s={elapsed:.1f}")

            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  ({label} total: {time.time() - t_cond0:.1f}s)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
