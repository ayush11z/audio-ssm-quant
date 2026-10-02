#!/usr/bin/env python
"""Phase 0, AST/ESC-50 half: loads a community ESC-50 fine-tune of the exact
checkpoint named in the brief (MIT/ast-finetuned-audioset-10-10-0.4593) and
runs it over the full ashraq/esc50 dataset.

Ran successfully on a free Lightning.ai T4 Studio (2026-10-01):
    FINAL full-dataset accuracy: 0.993  n=2000  elapsed_s=181.48

IMPORTANT CAVEAT (do not treat this as a clean Phase 0 pass): the checkpoint
author (bioamla/ast-esc50) reports 92.75% accuracy on their own eval split,
but does not document which ESC-50 fold(s) were held out during fine-tuning.
ashraq/esc50 on Hugging Face only ships a single "train" split (no separate
test split), so this script evaluates on the FULL 2000-clip dataset -- which
almost certainly includes the clips the checkpoint was fine-tuned on. The
measured 99.3% is therefore inflated by train/eval leakage, not a real
apples-to-apples reproduction of their reported 92.75%. See DECISIONS.md.

What this DOES establish: the environment, checkpoint loading, and inference
pipeline all work correctly end-to-end on a T4 (predictions are sensible,
not garbage). Getting an honest few-point reproduction requires either
contacting the checkpoint author for their held-out fold, or fine-tuning a
fresh checkpoint ourselves with a documented fold split.
"""
import io
import json
import time

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification

REPO = "bioamla/ast-esc50"  # community fine-tune of MIT/ast-finetuned-audioset-10-10-0.4593 on ESC-50
DATASET = "ashraq/esc50"
BATCH_SIZE = 16


def decode(raw, target_sr=16000):
    arr, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa

        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASTForAudioClassification.from_pretrained(REPO).to(device).eval()
    extractor = ASTFeatureExtractor.from_pretrained(REPO)
    label2id = model.config.label2id

    ds = load_dataset(DATASET, split="train")
    # decode=False: pull raw audio bytes ourselves via soundfile, bypassing
    # the `datasets` library's torchcodec-based Audio decoding (which needs
    # system FFmpeg/NVIDIA NPP shared libraries not present on this Studio).
    ds = ds.cast_column("audio", Audio(decode=False))
    print("dataset size", len(ds))

    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(ds), BATCH_SIZE):
            batch = ds[i : i + BATCH_SIZE]
            arrays = [decode(a) for a in batch["audio"]]
            labels = [label2id[c] for c in batch["category"]]
            inputs = extractor(arrays, sampling_rate=16000, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(-1).cpu().tolist()
            correct += sum(p == l for p, l in zip(preds, labels))
            total += len(labels)
            if i % 320 == 0:
                print(i, "/", len(ds), "acc so far", correct / total)

    acc = correct / total
    elapsed = time.time() - t0
    print("FINAL full-dataset accuracy:", acc, "n=", total, "elapsed_s=", elapsed)
    json.dump(
        {
            "model": REPO,
            "dataset": f"{DATASET} (full train split, no documented held-out fold)",
            "n": total,
            "accuracy": acc,
            "elapsed_s": elapsed,
            "caveat": "inflated by likely train/eval leakage; not a clean reproduction of the reported 0.9275",
        },
        open("ast_esc50_result.json", "w"),
    )


if __name__ == "__main__":
    main()
