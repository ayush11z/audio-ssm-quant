#!/usr/bin/env python
"""Phase 4: summarize the SSM-internal ablation grid's REAL outputs into a
small, committable file, and print the tables DECISIONS.md cites.

Inputs (written by scripts/phase4_aum_ablation_eval.py):
  results/phase4_aum_ablation_1pc.jsonl       per-clip predictions (committed)
  results/phase4_state_divergence_1pc.json    per-timestep ||h_q - h_fp||
                                              traces, ~318MB -- too large for
                                              GitHub, gitignored, kept only on
                                              the machine that ran the grid
  results/phase1_aum_full_precision_1pc.jsonl full-precision per-clip rows for
                                              the SAME wav files (Phase 4 has no
                                              unquantized condition of its own)

Output: results/phase4_state_divergence_1pc_summary.json -- accuracy per
condition x length (plus the matched full-precision count), and for every
divergence trace (condition x length x clip x layer x direction) its
length, mean, first-10%/last-10% means, max and final value. Enough to
re-derive every divergence number in DECISIONS.md without the raw file.

A trace that is exactly zero at every timestep is flagged, not dropped:
these are real properties of the checkpoint, not a bug (see DECISIONS.md
2026-10-13 -- layers 2-4 have ~1e-38 B/C projection weights so their state
never leaves zero; layers 7/22/23 have every A entry exactly -1, which
per-tensor quantization represents exactly). The aggregate table excludes
them so they don't drag medians toward zero.

Run locally -- pure Python, no GPU, no model loading:
    python3 scripts/phase4_summarize.py
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ABLATION_PATH = RESULTS / "phase4_aum_ablation_1pc.jsonl"
DIVERGENCE_PATH = RESULTS / "phase4_state_divergence_1pc.json"
FULL_PRECISION_PATH = RESULTS / "phase1_aum_full_precision_1pc.jsonl"
OUT_PATH = RESULTS / "phase4_state_divergence_1pc_summary.json"

LENGTHS = ["native", 20, 40, 80, 160]


def trace_stats(trace):
    k = max(1, len(trace) // 10)
    return {
        "n_steps": len(trace),
        "mean": st.mean(trace),
        "early10": st.mean(trace[:k]),
        "late10": st.mean(trace[-k:]),
        "max": max(trace),
        "final": trace[-1],
        "all_zero": max(trace) == 0.0,
    }


def main():
    rows = [json.loads(line) for line in open(ABLATION_PATH)]
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    fp_correct = {}
    for line in open(FULL_PRECISION_PATH):
        d = json.loads(line)
        fp_correct[d["wav"]] = d["correct"]

    accuracy = []
    for c in conditions:
        for length in LENGTHS:
            cell = [r for r in rows if r["condition"] == c and r["length_sec"] == length]
            accuracy.append({
                "condition": c, "length_sec": length, "n": len(cell),
                "correct": sum(r["correct"] for r in cell),
                "full_precision_correct_same_wavs": sum(bool(fp_correct.get(r["wav"])) for r in cell),
                "distinct_words": sorted({r["true_word"] for r in cell}),
            })

    traces = []
    for rec in json.load(open(DIVERGENCE_PATH)):
        for layer_idx, layer in enumerate(rec["per_layer"]):
            for direction in ("fwd", "bwd"):
                traces.append({
                    "condition": rec["condition"], "length_sec": rec["length_sec"],
                    "wav": rec["wav"], "layer": layer_idx, "direction": direction,
                    **trace_stats(layer[direction]),
                })

    aggregate = []
    for c in conditions:
        for length in LENGTHS:
            live = [t for t in traces if t["condition"] == c and t["length_sec"] == length and not t["all_zero"]]
            early = st.median(t["early10"] for t in live)
            late = st.median(t["late10"] for t in live)
            aggregate.append({
                "condition": c, "length_sec": length, "n_nonzero_traces": len(live),
                "median_mean": st.median(t["mean"] for t in live),
                "median_early10": early, "median_late10": late,
                "late_over_early": late / early if early else None,
            })

    json.dump({"accuracy": accuracy, "divergence_aggregate": aggregate, "divergence_traces": traces},
              open(OUT_PATH, "w"), indent=1)

    print("Accuracy (correct/n; full precision on the same wavs in brackets)")
    print("condition   " + "  ".join(f"{str(L):>9}" for L in LENGTHS))
    for c in conditions:
        cells = {a["length_sec"]: a for a in accuracy if a["condition"] == c}
        print(f"{c:<11} " + "  ".join(
            f"{cells[L]['correct']}/{cells[L]['n']} [{cells[L]['full_precision_correct_same_wavs']}]".rjust(9)
            for L in LENGTHS))
    print("\nDivergence ||h_q - h_fp||, median over non-zero traces: mean | early10 -> late10 (late/early)")
    for a in aggregate:
        print(f"{a['condition']:<11} {str(a['length_sec']):<7} {a['median_mean']:.3e} | "
              f"{a['median_early10']:.3e} -> {a['median_late10']:.3e} ({a['late_over_early']:.2f})")
    zero = defaultdict(set)
    for t in traces:
        if t["all_zero"]:
            zero[t["condition"]].add(t["layer"])
    print("\nAll-zero layers per condition:", {c: sorted(v) for c, v in zero.items()})
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
