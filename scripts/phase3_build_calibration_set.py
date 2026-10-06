#!/usr/bin/env python
"""Phase 3: build the calibration clip sets (brief section 6: "256 in-
distribution clips at training length, with 3 calibration seeds").

Draws from the TRAIN split specifically (derived as all-files minus
validation_list.txt minus testing_list.txt, same logic as AuM's own
exps/speechcommands/prep_sc.py), deliberately kept separate from Phase 1's
eval manifest (which samples the TEST split) -- calibration data must not
overlap with eval data, or the "calibration gap" question Phase 6 asks
about becomes meaningless.

These are plain native-length clips, no length-extension/background
splicing (calibration is specifically "at training length" per the brief,
not across the length grid -- the resulting scale is then reused at every
eval length, which is the whole point of testing whether a fixed
calibration generalizes).

Only data/raw/speech_commands_v0.02.tar.gz, testing_list.txt and
validation_list.txt need to already be present locally -- the word-folder
audio itself does NOT need prior extraction. This script lists the
archive's contents directly (`tar -tzf`), samples train-split filenames
from that list, then extracts only the ~768 sampled files (not the full
~85k-file train split) directly from the archive.

Output: results/phase3_calibration_seed{0,1,2}.json (list of
{"wav": relative_path, "word": ...}) + the extracted wav files themselves
under data/raw/SpeechCommands/speech_commands_v0.02/<word>/. Pure CPU, no
GPU needed.
"""
import csv
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "SpeechCommands" / "speech_commands_v0.02"
ARCHIVE = ROOT / "data" / "raw" / "speech_commands_v0.02.tar.gz"
LABEL_CSV = ROOT / "third_party" / "Audio-Mamba-AuM" / "exps" / "speechcommands" / "data" / "speechcommands_class_labels_indices.csv"

N_CLIPS = 256
SEEDS = [0, 1, 2]


def load_words():
    words = set()
    with open(LABEL_CSV) as f:
        for row in csv.DictReader(f):
            words.add(row["display_name"].strip('"'))
    return words


def list_archive_wavs():
    """Returns {relpath_without_leading_./: archive_entry_name_as_listed}."""
    out = subprocess.run(["tar", "-tzf", str(ARCHIVE)], capture_output=True, text=True, check=True).stdout
    mapping = {}
    for line in out.splitlines():
        if not line.endswith(".wav"):
            continue
        rel = line[2:] if line.startswith("./") else line  # strip leading "./"
        mapping[rel] = line
    return mapping


def main():
    words = load_words()
    archive_wavs = list_archive_wavs()
    print(f"archive contains {len(archive_wavs)} total .wav entries")

    with open(RAW / "validation_list.txt") as f:
        val_set = {line.strip() for line in f if line.strip()}
    with open(RAW / "testing_list.txt") as f:
        test_set = {line.strip() for line in f if line.strip()}
    excluded = val_set | test_set

    train_files = [
        rel for rel in archive_wavs
        if rel.split("/")[0] in words and rel not in excluded
    ]
    print(f"train split: {len(train_files)} clips across {len(words)} classes "
          f"(excludes {len(excluded)} val/test clips)")

    all_sampled = set()
    seed_samples = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        sample = rng.sample(train_files, N_CLIPS)
        seed_samples[seed] = sample
        all_sampled.update(sample)

    # Extract only the sampled files (not the full train split) directly
    # from the archive in one pass.
    to_extract = [rel for rel in all_sampled if not (RAW / rel).exists()]
    if to_extract:
        filelist_path = ROOT / "data" / "raw" / "_calibration_extract_list.txt"
        with open(filelist_path, "w") as f:
            f.writelines(archive_wavs[rel] + "\n" for rel in to_extract)
        subprocess.run(
            ["tar", "-xzf", str(ARCHIVE), "-C", str(RAW), "--strip-components=1", "-T", str(filelist_path)],
            check=True,
        )
        filelist_path.unlink()
        print(f"extracted {len(to_extract)} new calibration clips")
    else:
        print("all sampled calibration clips already extracted")

    for seed in SEEDS:
        records = [
            {"wav": f"data/raw/SpeechCommands/speech_commands_v0.02/{rel}", "word": rel.split("/")[0]}
            for rel in seed_samples[seed]
        ]
        out_path = ROOT / "results" / f"phase3_calibration_seed{seed}.json"
        json.dump(records, open(out_path, "w"), indent=1)
        print(f"seed {seed}: wrote {len(records)} calibration clips to {out_path}")


if __name__ == "__main__":
    main()
