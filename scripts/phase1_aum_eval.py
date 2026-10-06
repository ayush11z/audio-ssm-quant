#!/usr/bin/env python
"""Phase 1: full-precision AuM baseline across length-extended Speech
Commands V2 clips. Counterpart to scripts/phase1_ast_eval.py.

Unlike AST, AuM's own codebase already handles variable input length via
FlexiPatchEmbed/FlexiPosEmbed (confirmed working in the Phase 0 AuM run --
see its log lines "Initializing FlexiPatchEmbed..." / "Loading position
embedding!"). This just needs spectrogram_size set correctly per length
group at model construction time, no hand-rolled interpolation needed.

MUST run inside the AuM venv with the environment from
scripts/phase0_setup_env.sh (Python 3.10, torch 2.1.1+cu118,
mamba_ssm==1.1.3.post1 + ViM bidirectional patch). Run from
third_party/Audio-Mamba-AuM/ so its relative imports resolve:

    cd third_party/Audio-Mamba-AuM
    cp ../../scripts/phase1_aum_eval.py .
    python3 phase1_aum_eval.py
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

sys.path.append(".")
import src.models as models

ROOT = Path(__file__).resolve().parent  # third_party/Audio-Mamba-AuM when copied there, per docstring
PROJECT_ROOT = ROOT.parent.parent  # back to the repo root, where results/ and data/ live
# Base AudioSet-pretrained, Speech Commands V2-finetuned (94.82% reported
# acc), downloaded via gdown from the README's table -- same "Base
# AudioSet" variant used for VGGSound in Phase 0, for methodology
# consistency across phases.
CHECKPOINT_PATH = "exps/speechcommands/models/aum-base_audioset-spc_v2.pth"
LABEL_CSV = "exps/speechcommands/data/speechcommands_class_labels_indices.csv"
MANIFEST_PATH = PROJECT_ROOT / "results" / "phase1_eval_manifest.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "phase1_aum_full_precision.jsonl"

NUM_MEL_BINS = 128
DATASET_MEAN = -6.845978  # from AuM's exps/speechcommands/aum_eval.sh, NOT the AudioSet stats used in Phase 0
DATASET_STD = 5.5654526


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
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=NUM_MEL_BINS,
        dither=0.0,
        frame_shift=10,
    )
    # Native (~1s) clips vary slightly in raw duration and naturally produce
    # ~98 frames, not the 128 frames AuM's own eval script (audio_length=128)
    # trained/evaluated with (fixed-length, zero-padded). Only the native
    # baseline needs this -- extended lengths are already exactly the
    # target duration by construction.
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
    model.to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mid2word = mid_to_word()
    word2idx = {}
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            word2idx[row["display_name"].strip('"')] = int(row["index"])
    n_classes = len(word2idx)

    manifest = json.load(open(MANIFEST_PATH))["data"]
    by_length = defaultdict(list)
    for item in manifest:
        by_length[item["length_sec"]].append(item)

    results = []
    for length_sec, items in sorted(by_length.items(), key=lambda kv: (kv[0] != "native", kv[0])):
        t0 = time.time()
        # Determine this length group's frame count from its first clip
        # (all clips at a given length_sec have identical duration, so
        # identical frame count). Native group pads/crops to 128 frames,
        # matching AuM's own training/eval convention for this checkpoint.
        pad_to = 128 if length_sec == "native" else None
        first_fbank = compute_fbank(items[0]["wav"], pad_to_frames=pad_to)
        target_length = first_fbank.shape[0]
        model = build_model(target_length, n_classes, device)
        print(f"length={length_sec} target_length_frames={target_length} model built")

        correct = 0
        with torch.no_grad():
            for item in items:
                fbank = compute_fbank(item["wav"], pad_to_frames=pad_to)
                inputs = fbank.unsqueeze(0).to(device)
                logits = torch.sigmoid(model(inputs))
                pred_idx = int(logits.argmax(-1).item())
                pred_word = [w for w, i in word2idx.items() if i == pred_idx][0]
                true_word = mid2word[item["labels"]]
                is_correct = pred_word == true_word
                correct += int(is_correct)
                results.append({
                    "length_sec": length_sec,
                    "position": item["position"],
                    "wav": item["wav"],
                    "true_word": true_word,
                    "pred_word": pred_word,
                    "correct": is_correct,
                })
        acc = correct / len(items)
        elapsed = time.time() - t0
        print(f"length={length_sec} n={len(items)} acc={acc:.4f} elapsed_s={elapsed:.1f}")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
