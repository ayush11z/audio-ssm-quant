#!/usr/bin/env python
"""Phase 4 native-length re-run: summarize the raw divergence traces into a
small, committable file, and print the accuracy + divergence tables
DECISIONS.md cites (see its 2026-10-19 entry).

Inputs (written by scripts/phase4_aum_ablation_native_eval.py):
  results/phase4_aum_ablation_native.jsonl      per-clip predictions (committed)
  results/phase4_state_divergence_native.json   per-timestep ||h_q - h_fp||
                                                traces, ~96MB -- gitignored,
                                                kept only on the machine that
                                                ran the grid (this time: the
                                                Modal volume, downloaded
                                                locally but not pushed)

Output: results/phase4_state_divergence_native_summary.json -- per
condition, flattened across every clip x layer x direction x timestep:
n values, median, mean, p90, max. Enough to re-derive every divergence
number in DECISIONS.md's 2026-10-19 entry without the raw file.

Run locally -- pure Python, no GPU, no model loading:
    python3 scripts/phase4_summarize_native.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ABLATION_PATH = RESULTS / "phase4_aum_ablation_native.jsonl"
DIVERGENCE_PATH = RESULTS / "phase4_state_divergence_native.json"
OUT_PATH = RESULTS / "phase4_state_divergence_native_summary.json"

TARGETS = ("delta", "A", "BC", "state")
BITS = (8, 6, 4)


def main():
    rows = [json.loads(line) for line in open(ABLATION_PATH)]
    accuracy = []
    for target in TARGETS:
        for bits in BITS:
            cond = f"{target}_b{bits}"
            cell = [r for r in rows if r["condition"] == cond]
            accuracy.append({
                "condition": cond, "n": len(cell),
                "correct": sum(r["correct"] for r in cell),
                "accuracy": sum(r["correct"] for r in cell) / len(cell),
            })

    by_cond = defaultdict(list)
    for rec in json.load(open(DIVERGENCE_PATH)):
        cond = rec["condition"]
        for layer in rec["per_layer"]:
            by_cond[cond].extend(layer["fwd"])
            by_cond[cond].extend(layer["bwd"])

    divergence = []
    for target in TARGETS:
        for bits in BITS:
            cond = f"{target}_b{bits}"
            vals = sorted(by_cond[cond])
            n = len(vals)
            divergence.append({
                "condition": cond, "n_values": n,
                "median": st.median(vals),
                "mean": sum(vals) / n,
                "p90": vals[int(0.9 * n)],
                "max": vals[-1],
            })

    out = {"accuracy": accuracy, "divergence": divergence}
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}")

    print(f"\n{'condition':12s} {'acc':>8s} {'n':>5s}")
    for a in accuracy:
        print(f"{a['condition']:12s} {a['accuracy']:>8.4f} {a['n']:>5d}")

    print(f"\n{'condition':12s} {'median':>10s} {'mean':>10s} {'p90':>10s} {'max':>10s}")
    for d in divergence:
        print(f"{d['condition']:12s} {d['median']:>10.4f} {d['mean']:>10.4f} {d['p90']:>10.4f} {d['max']:>10.4f}")


if __name__ == "__main__":
    main()
