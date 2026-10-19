#!/usr/bin/env python
"""Phase 6 extension: bootstrap CIs + a figure for Phase 4's native-length
SSM-internal ablation grid (see DECISIONS.md's 2026-10-19 entry). Phase 6's
original script (scripts/phase6_bootstrap_slopes.py) only covered Phase
1/3 since Phase 4 hadn't landed yet -- its own docstring flagged this to
be filled in once it did.

Unlike Phase 1/3 (accuracy vs. length, slope fits), Phase 4 is a single
length (native) x 4 targets x 3 bit-widths grid -- no length axis, so no
slope fit here. What IS real: `results/phase4_aum_ablation_native.jsonl`
has full per-clip rows for all 12 conditions (140 clips each), so every
condition gets a real nonparametric bootstrap, not a Wilson fallback.

Run locally -- pure numpy/scipy/matplotlib, no GPU, no model loading:
    python3 scripts/phase6_phase4_bootstrap.py
"""
import json
from collections import defaultdict
from pathlib import Path

from ssmquant.analysis.bootstrap import bootstrap_accuracy_ci
from ssmquant.analysis.figures import plot_accuracy_vs_bits

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_PATH = RESULTS / "phase6_phase4_bootstrap.json"

TARGETS = ("delta", "A", "BC", "state")
BITS = (8, 6, 4)
N_RESAMPLES = 10000
SEED = 0


def main():
    rows = [json.loads(line) for line in open(RESULTS / "phase4_aum_ablation_native.jsonl")]
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r["correct"])

    out = {}
    for target in TARGETS:
        for bits in BITS:
            cond = f"{target}_b{bits}"
            flags = by_condition[cond]
            point, lo, hi, n = bootstrap_accuracy_ci(flags, n_resamples=N_RESAMPLES, seed=SEED)
            out[cond] = {"target": target, "bits": bits, "accuracy": point, "ci_low": lo, "ci_high": hi, "n": n}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"Wrote {OUT_PATH}")

    print(f"\n{'condition':12s} {'acc':>8s} {'ci_low':>8s} {'ci_high':>8s} {'n':>5s}")
    for target in TARGETS:
        for bits in BITS:
            cond = f"{target}_b{bits}"
            d = out[cond]
            print(f"{cond:12s} {d['accuracy']:>8.4f} {d['ci_low']:>8.4f} {d['ci_high']:>8.4f} {d['n']:>5d}")

    series = []
    for target in TARGETS:
        series.append({
            "target": target,
            "bits": list(BITS),
            "accuracy": [out[f"{target}_b{b}"]["accuracy"] for b in BITS],
            "ci_low": [out[f"{target}_b{b}"]["ci_low"] for b in BITS],
            "ci_high": [out[f"{target}_b{b}"]["ci_high"] for b in BITS],
        })
    out_path = FIGURES / "phase6_phase4_accuracy_vs_bits.pdf"
    plot_accuracy_vs_bits(
        series,
        title="AuM native-length: accuracy vs. SSM-internal quantization bit-width (n=140, 95% CI)",
        out_path=out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
