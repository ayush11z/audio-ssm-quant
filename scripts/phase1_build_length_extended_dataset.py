#!/usr/bin/env python
"""Phase 1: build length-extended Speech Commands V2 eval clips.

Params mirror configs/experiment/phase1_length_extended.yaml (not Hydra-
wired yet -- kept as plain constants here, same style as the Phase 0
scripts, so this runs with nothing but stdlib + numpy + soundfile).

For each of a stratified sample of native (~1s) test clips, generates:
  - 1 native-length copy (baseline, no background)
  - 4 extended lengths (20/40/80/160s) x 3 positions (start/middle/end)
    = 12 variants, each the original clip embedded in low-level background
    noise drawn from the official archive's own _background_noise_ folder

Output: data/processed/phase1_speech_commands/<variant_id>.wav +
results/phase1_eval_manifest.json (brief section 5's "all lengths x
positions" eval set).

Run locally -- this is pure audio processing (numpy/soundfile), no GPU,
no torch needed.
"""
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "SpeechCommands" / "speech_commands_v0.02"
OUT_DIR = ROOT / "data" / "processed" / "phase1_speech_commands"
# Separate manifest filename when CLIPS_PER_CLASS is overridden via env var,
# so a budget-constrained run (see DECISIONS.md) doesn't silently clobber
# the full-design manifest.
_MANIFEST_SUFFIX = "" if "PHASE1_CLIPS_PER_CLASS" not in os.environ else f"_{os.environ['PHASE1_CLIPS_PER_CLASS']}pc"
MANIFEST_PATH = ROOT / "results" / f"phase1_eval_manifest{_MANIFEST_SUFFIX}.json"
LABEL_CSV = ROOT / "third_party" / "Audio-Mamba-AuM" / "exps" / "speechcommands" / "data" / "speechcommands_class_labels_indices.csv"

SR = 16000
NATIVE_LENGTH_SEC = 1.0
EXTENDED_LENGTHS_SEC = [20, 40, 80, 160]
POSITIONS = ["start", "middle", "end"]
CLIPS_PER_CLASS = int(os.environ.get("PHASE1_CLIPS_PER_CLASS", 4))
BACKGROUND_SCALE = 0.1
SEED = 0


def load_label_map():
    word_to_mid = {}
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            word_to_mid[row["display_name"].strip('"')] = row["mid"]
    return word_to_mid


def load_background_noise():
    bg_dir = RAW / "_background_noise_"
    tracks = []
    for wav_path in sorted(bg_dir.glob("*.wav")):
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == SR, f"unexpected sample rate {sr} in {wav_path}"
        tracks.append(audio)
    assert tracks, f"no background noise files found in {bg_dir}"
    return tracks


def sample_background(tracks, n_samples, rng):
    """Build n_samples of continuous background noise by tiling/cropping a
    random background track, scaled to a low level per the brief's
    'low-level noise... never clips containing other labeled events'."""
    track = tracks[rng.randrange(len(tracks))]
    if len(track) >= n_samples:
        start = rng.randrange(len(track) - n_samples + 1)
        seg = track[start : start + n_samples]
    else:
        reps = n_samples // len(track) + 1
        tiled = np.tile(track, reps)
        start = rng.randrange(len(tiled) - n_samples + 1)
        seg = tiled[start : start + n_samples]
    return seg * BACKGROUND_SCALE


def embed_clip(clip, total_length_sec, position, bg_tracks, rng):
    total_n = int(round(total_length_sec * SR))
    clip_n = len(clip)
    assert clip_n <= total_n, "clip longer than target total length"

    background = sample_background(bg_tracks, total_n, rng)

    if position == "start":
        offset = 0
    elif position == "end":
        offset = total_n - clip_n
    elif position == "middle":
        offset = (total_n - clip_n) // 2
    else:
        raise ValueError(position)

    # Splice: replace the background at [offset, offset+clip_n) with the
    # real clip at its native energy (not additively mixed), with a short
    # linear crossfade to avoid audible clicks at the splice boundary.
    out = background.copy()
    fade_n = min(160, clip_n // 4)  # ~10ms fade at 16kHz
    segment = clip.copy()
    if fade_n > 0:
        fade_in = np.linspace(0, 1, fade_n)
        fade_out = np.linspace(1, 0, fade_n)
        segment[:fade_n] = segment[:fade_n] * fade_in + out[offset : offset + fade_n] * (1 - fade_in)
        segment[-fade_n:] = segment[-fade_n:] * fade_out + out[offset + clip_n - fade_n : offset + clip_n] * (1 - fade_out)
    out[offset : offset + clip_n] = segment
    return out.astype(np.float32)


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    word_to_mid = load_label_map()
    bg_tracks = load_background_noise()

    with open(RAW / "testing_list.txt") as f:
        test_files = [line.strip() for line in f if line.strip()]

    by_class = {}
    for rel_path in test_files:
        word = rel_path.split("/")[0]
        if word not in word_to_mid:
            continue  # skip anything outside the 35-word vocabulary
        by_class.setdefault(word, []).append(rel_path)

    selected = []
    for word, files in sorted(by_class.items()):
        rng.shuffle(files)
        selected.extend(files[:CLIPS_PER_CLASS])
    rng.shuffle(selected)

    print(f"Selected {len(selected)} native clips across {len(by_class)} classes "
          f"({CLIPS_PER_CLASS}/class target)")

    manifest = []
    for rel_path in selected:
        word = rel_path.split("/")[0]
        mid = word_to_mid[word]
        src_path = RAW / rel_path
        clip, sr = sf.read(src_path, dtype="float32")
        assert sr == SR, f"unexpected sample rate {sr} in {src_path}"
        clip_id = rel_path.replace("/", "__").replace(".wav", "")

        # Native-length baseline (as-is, no synthetic background).
        native_out = OUT_DIR / f"{clip_id}__native.wav"
        if not native_out.exists():
            sf.write(native_out, clip, SR)
        manifest.append({
            "wav": str(native_out.relative_to(ROOT)),
            "labels": mid,
            "word": word,
            "source_clip": rel_path,
            "length_sec": "native",
            "position": None,
        })

        # Extended lengths x positions.
        for length_sec in EXTENDED_LENGTHS_SEC:
            for position in POSITIONS:
                out_path = OUT_DIR / f"{clip_id}__{length_sec}s_{position}.wav"
                if not out_path.exists():
                    extended = embed_clip(clip, length_sec, position, bg_tracks, rng)
                    sf.write(out_path, extended, SR)
                manifest.append({
                    "wav": str(out_path.relative_to(ROOT)),
                    "labels": mid,
                    "word": word,
                    "source_clip": rel_path,
                    "length_sec": length_sec,
                    "position": position,
                })

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"data": manifest}, f, indent=1)

    print(f"Wrote {len(manifest)} manifest entries to {MANIFEST_PATH}")
    print(f"({len(selected)} native clips x (1 native + "
          f"{len(EXTENDED_LENGTHS_SEC)}x{len(POSITIONS)} extended) = {len(manifest)})")


if __name__ == "__main__":
    main()
