#!/usr/bin/env python
"""Phase 1: full-precision AST baseline across length-extended Speech
Commands V2 clips (brief Phase 1: "Run both models at full precision
across all lengths. Gate: report how much each model degrades with length").

AST confound (brief section 5): the checkpoint's positional embeddings are
sized for its native ~1s (128 mel-frame) input. Decision (DECISIONS.md):
interpolate the position-embedding patch grid along the time axis for
longer inputs, bicubic, same trick used to fine-tune ViT/DeiT at a
different resolution than pretrained. AST has no built-in support for this
(checked transformers 4.57.1 source directly -- position_embeddings is a
plain fixed nn.Parameter), so it's hand-rolled here.

Run on a GPU instance (CUDA not required but recommended for speed with
160s clips): `python scripts/phase1_ast_eval.py`
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

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = "MIT/ast-finetuned-speech-commands-v2"
_MANIFEST_SUFFIX = "" if "PHASE1_CLIPS_PER_CLASS" not in os.environ else f"_{os.environ['PHASE1_CLIPS_PER_CLASS']}pc"
MANIFEST_PATH = ROOT / "results" / f"phase1_eval_manifest{_MANIFEST_SUFFIX}.json"
LABEL_CSV = ROOT / "third_party" / "Audio-Mamba-AuM" / "exps" / "speechcommands" / "data" / "speechcommands_class_labels_indices.csv"
RESULTS_PATH = ROOT / "results" / f"phase1_ast_full_precision{_MANIFEST_SUFFIX}.jsonl"

NUM_MEL_BINS = 128
PATCH_SIZE = 16
STRIDE = 16


def mid_to_word():
    import csv

    m = {}
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            m[row["mid"]] = row["display_name"].strip('"')
    return m


def compute_fbank(wav_path, extractor_mean, extractor_std, pad_to_frames=None):
    waveform, sr = torchaudio.load(str(ROOT / wav_path))
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
    # ~98 frames, not the 128 frames both checkpoints were trained/evaluated
    # with (fixed-length, zero-padded). Only the native/in-distribution
    # baseline needs this -- extended lengths are already exactly the
    # target duration by construction (that's the whole point of Phase 1),
    # so they're left as their natural frame count.
    if pad_to_frames is not None:
        n = fbank.shape[0]
        if n < pad_to_frames:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, pad_to_frames - n))(fbank)
        elif n > pad_to_frames:
            fbank = fbank[:pad_to_frames, :]
    fbank = (fbank - extractor_mean) / (extractor_std * 2)
    return fbank  # (n_frames, num_mel_bins)


def time_out_dim(n_frames):
    return (n_frames - PATCH_SIZE) // STRIDE + 1


def interpolated_position_embeddings(base_pos_embed, base_time_dim, freq_dim, new_time_dim):
    """base_pos_embed: (1, freq_dim*base_time_dim + 2, hidden). Returns a
    new (1, freq_dim*new_time_dim + 2, hidden) tensor with the patch-grid
    portion bicubic-interpolated along time, cls/distillation tokens kept
    as-is (brief section 5's chosen strategy)."""
    hidden = base_pos_embed.shape[-1]
    prefix = base_pos_embed[:, :2, :]  # cls + distillation token positions
    patch_pos = base_pos_embed[:, 2:, :]  # (1, freq_dim*base_time_dim, hidden)
    grid = patch_pos.reshape(1, freq_dim, base_time_dim, hidden).permute(0, 3, 1, 2)  # (1, hidden, freq, time)
    new_grid = F.interpolate(grid, size=(freq_dim, new_time_dim), mode="bicubic", align_corners=False)
    new_patch_pos = new_grid.permute(0, 2, 3, 1).reshape(1, freq_dim * new_time_dim, hidden)
    return torch.cat([prefix, new_patch_pos], dim=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASTForAudioClassification.from_pretrained(CHECKPOINT).to(device).eval()
    extractor = ASTFeatureExtractor.from_pretrained(CHECKPOINT)
    ext_mean, ext_std = extractor.mean, extractor.std

    id2label = model.config.id2label
    label2id = {v: int(k) for k, v in id2label.items()}
    mid2word = mid_to_word()

    base_freq_dim = (NUM_MEL_BINS - PATCH_SIZE) // STRIDE + 1
    base_time_dim = (model.config.max_length - PATCH_SIZE) // STRIDE + 1
    base_pos_embed = model.audio_spectrogram_transformer.embeddings.position_embeddings.data.clone()
    print(f"Base grid: freq={base_freq_dim} time={base_time_dim} "
          f"(native max_length={model.config.max_length})")

    manifest = json.load(open(MANIFEST_PATH))["data"]
    # Group by length_sec so position embeddings are only recomputed once
    # per distinct sequence length, not once per clip.
    by_length = defaultdict(list)
    for item in manifest:
        by_length[item["length_sec"]].append(item)

    results = []
    embeddings_module = model.audio_spectrogram_transformer.embeddings
    for length_sec, items in sorted(by_length.items(), key=lambda kv: (kv[0] != "native", kv[0])):
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
                # max_length controls ASTEmbeddings.get_shape() at init only;
                # forward() itself just adds position_embeddings to patch
                # embeddings, so no further config change needed per length.

            inputs = fbank.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model.audio_spectrogram_transformer(inputs).pooler_output
                logits = model.classifier(logits)
            pred_id = int(logits.argmax(-1).item())
            pred_word = id2label[str(pred_id)]
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
        print(f"length={length_sec} n={len(items)} frames={n_frames_seen} "
              f"acc={acc:.4f} elapsed_s={elapsed:.1f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
