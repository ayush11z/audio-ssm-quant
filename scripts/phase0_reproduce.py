#!/usr/bin/env python
"""Phase 0 gate: reproduce AuM's and AST's reported benchmark numbers to
within ~1 point. Run on the A100 Nautilus pod after phase0_setup_env.sh.

This script does NOT fabricate or estimate results (brief section 12): the
`reported_score` fields below are left as None until someone reads the
actual number off the AuM / AST paper's table or README and fills them in.
Do not guess a plausible-looking number here.

Usage:
    python scripts/phase0_reproduce.py --model aum --task esc50
    python scripts/phase0_reproduce.py --model ast --task esc50
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase0_reproduction.jsonl"

# Fill these in only after reading the actual number from the paper/README —
# cite the source (table number, URL) in the comment next to it.
REPORTED_SCORES = {
    ("aum", "esc50"): None,      # TODO: read off AuM paper/README
    ("aum", "audioset"): None,   # TODO
    ("ast", "esc50"): None,      # TODO: read off AST paper/README
    ("ast", "audioset"): None,   # TODO (e.g. the 0.4593 mAP in the
                                  # checkpoint name itself, verify it's the
                                  # same eval split/protocol before trusting it)
}

TOLERANCE_POINTS = 1.0


def load_model(name: str):
    if name == "aum":
        # TODO: import from third_party/Audio-Mamba-AuM once cloned;
        # verify which checkpoint is actually released (brief section 4).
        raise NotImplementedError(
            "AuM loading not implemented yet — clone third_party/Audio-Mamba-AuM "
            "first (scripts/phase0_setup_env.sh) and check its README for "
            "released checkpoints before wiring this up."
        )
    elif name == "ast":
        from transformers import ASTForAudioClassification, ASTFeatureExtractor

        ckpt = "MIT/ast-finetuned-audioset-10-10-0.4593"
        model = ASTForAudioClassification.from_pretrained(ckpt)
        extractor = ASTFeatureExtractor.from_pretrained(ckpt)
        return model, extractor
    else:
        raise ValueError(name)


def run_eval(model_name: str, task_name: str) -> float:
    """Returns the measured score (accuracy for classification). Must
    actually run inference over the task's eval set — never return a
    placeholder number from here."""
    raise NotImplementedError(
        f"Eval loop for ({model_name}, {task_name}) not implemented yet. "
        "Needs the dataset config filled in (configs/task/*.yaml has ??? "
        "fields) and a real forward pass, not a stand-in number."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["aum", "ast"])
    ap.add_argument("--task", required=True, choices=["esc50", "audioset"])
    args = ap.parse_args()

    reported = REPORTED_SCORES.get((args.model, args.task))
    if reported is None:
        print(
            f"BLOCKED: no reported score recorded for ({args.model}, {args.task}). "
            "Fill in REPORTED_SCORES from the actual paper/README table before "
            "running this — do not guess a number.",
            file=sys.stderr,
        )
        sys.exit(1)

    measured = run_eval(args.model, args.task)
    gap = abs(measured - reported)
    passed = gap <= TOLERANCE_POINTS

    row = {
        "model": args.model,
        "task": args.task,
        "reported_score": reported,
        "measured_score": measured,
        "gap_points": gap,
        "tolerance_points": TOLERANCE_POINTS,
        "gate_passed": passed,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")

    print(json.dumps(row, indent=2))
    if not passed:
        print(
            "Phase 0 GATE FAILED: reproduction outside tolerance. "
            "Stop and report options per the brief — do not proceed to Phase 1.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
