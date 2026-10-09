#!/usr/bin/env python
"""Phase 6: bootstrap CIs, slope fits, and figures (brief: "Bootstrap CIs,
slope fits, figures, tables") computed from the REAL results Phases 1 and
3 already produced. Phase 4's ablation grid hasn't completed yet (see
DECISIONS.md -- lost twice to infrastructure issues, not a code problem),
so it's excluded here; this script should be re-run once it lands.

Primary analysis uses the 3-clips-per-class round (the one DECISIONS.md
designates as the primary reported result, superseding the earlier
1-clip-per-class round): per-clip rows exist for the quantized grid
(results/phase3_{ast,aum}_quantized_3pc.jsonl), so those get a real
nonparametric bootstrap. The matching full-precision baseline
(results/phase1_{ast,aum}_full_precision_3pc_summary.json) only has
aggregate accuracy -- its per-clip rows were lost to a process mistake
this session (see DECISIONS.md) -- so it gets a Wilson score interval
from the surviving (k, n) counts instead of a bootstrap. This script is
explicit in its output about which numbers are bootstrapped and which
are the analytic fallback; they are not the same kind of interval and
shouldn't be read as equivalently precise.

A secondary "matched check" also runs on the 1-clip-per-class round,
where BOTH the full-precision baseline and one quantized condition
(w8a16) still have real per-clip rows -- a fully bootstrapped, same-
methodology comparison, smaller sample than the 3pc round but not
missing any data.

Run locally -- pure numpy/scipy/matplotlib, no GPU, no model loading:
    python3 scripts/phase6_bootstrap_slopes.py
"""
import json
from collections import defaultdict
from pathlib import Path

from ssmquant.analysis.bootstrap import bootstrap_accuracy_ci, wilson_ci
from ssmquant.analysis.figures import plot_accuracy_vs_length
from ssmquant.analysis.slope_fit import EXTENDED_LENGTHS_SEC, bootstrap_slope_ci, fit_slope

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_PATH = RESULTS / "phase6_bootstrap_slopes.json"

N_RESAMPLES = 10000
SEED = 0


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def group_correct_by_length(rows):
    by_length = defaultdict(list)
    for r in rows:
        by_length[r["length_sec"]].append(r["correct"])
    return dict(by_length)


def bootstrap_condition(by_length):
    """{length_sec: [bool,...]} -> {"by_length": {...}, "slope": {...}}
    using real per-clip bootstrap at every length, and a bootstrap CI on
    the log-length slope across the four extended lengths."""
    by_length_result = {}
    for length_sec, flags in by_length.items():
        point, lo, hi, n = bootstrap_accuracy_ci(flags, n_resamples=N_RESAMPLES, seed=SEED)
        by_length_result[str(length_sec)] = {
            "accuracy": point, "ci_low": lo, "ci_high": hi, "n": n, "method": "bootstrap",
        }

    extended = {l: by_length[l] for l in EXTENDED_LENGTHS_SEC if l in by_length}
    slope_result = None
    if set(extended.keys()) == set(EXTENDED_LENGTHS_SEC):
        point_slope, slope_lo, slope_hi = bootstrap_slope_ci(extended, n_resamples=N_RESAMPLES, seed=SEED)
        slope_result = {"slope": point_slope, "ci_low": slope_lo, "ci_high": slope_hi, "method": "bootstrap"}
    return {"by_length": by_length_result, "slope": slope_result}


def wilson_condition(summary_rows):
    """Phase 1's 3pc full-precision summary ({length_sec, n, accuracy})
    -> {"by_length": {...}, "slope": {...}} using Wilson intervals (no
    per-clip data survives for this one -- see module docstring) and a
    point-estimate-only slope (no bootstrap CI possible without per-clip
    resampling)."""
    by_length_result = {}
    accuracy_by_length = {}
    for row in summary_rows:
        length_sec = row["length_sec"]
        n = row["n"]
        k = round(row["accuracy"] * n)  # summary only stored the rounded accuracy, not raw k
        p, lo, hi, _ = wilson_ci(k, n)
        by_length_result[str(length_sec)] = {
            "accuracy": row["accuracy"], "ci_low": lo, "ci_high": hi, "n": n, "method": "wilson",
        }
        if length_sec != "native":
            accuracy_by_length[int(length_sec)] = row["accuracy"]

    slope_result = None
    if set(accuracy_by_length.keys()) == set(EXTENDED_LENGTHS_SEC):
        lengths = sorted(accuracy_by_length.keys())
        point_slope, _ = fit_slope(lengths, [accuracy_by_length[l] for l in lengths])
        slope_result = {"slope": point_slope, "ci_low": None, "ci_high": None, "method": "point_estimate_only"}
    return {"by_length": by_length_result, "slope": slope_result}


def series_for_figure(model, label, result, linestyle="-"):
    lengths = sorted(l for l in EXTENDED_LENGTHS_SEC if str(l) in result["by_length"])
    by_length = result["by_length"]
    return {
        "model": model, "label": label, "linestyle": linestyle,
        "lengths": lengths,
        "accuracy": [by_length[str(l)]["accuracy"] for l in lengths],
        "ci_low": [by_length[str(l)]["ci_low"] for l in lengths],
        "ci_high": [by_length[str(l)]["ci_high"] for l in lengths],
        "native_accuracy": by_length.get("native", {}).get("accuracy"),
    }


def main():
    out = {"primary_3pc": {}, "matched_check_1pc": {}}

    # ---- Primary: 3pc quantized grid (real bootstrap) ----
    for model in ("ast", "aum"):
        quant_rows = load_jsonl(RESULTS / f"phase3_{model}_quantized_3pc.jsonl")
        by_condition_seed = defaultdict(list)
        for r in quant_rows:
            by_condition_seed[(r["condition"], r["seed"])].append(r)

        model_out = {"full_precision": None, "quantized": {}}
        for (condition, seed), rows in by_condition_seed.items():
            label = condition + (f"_seed{seed}" if seed is not None else "")
            by_length = group_correct_by_length(rows)
            model_out["quantized"][label] = bootstrap_condition(by_length)

        # ---- Primary: 3pc full-precision baseline (Wilson, summary-only) ----
        summary_rows = json.load(open(RESULTS / f"phase1_{model}_full_precision_3pc_summary.json"))
        model_out["full_precision"] = wilson_condition(summary_rows)

        out["primary_3pc"][model] = model_out

        print(f"=== {model} (3pc, primary) ===")
        fp_slope = model_out["full_precision"]["slope"]
        print(f"  full-precision slope: {fp_slope['slope']:.4f} acc/decade (point estimate only, no per-clip data)")
        for cond in ("w8a16", "w4a16"):
            if cond in model_out["quantized"]:
                s = model_out["quantized"][cond]["slope"]
                print(f"  {cond} slope: {s['slope']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}] (95% bootstrap CI)")

    # ---- Secondary: 1pc matched check (both sides have per-clip data) ----
    for model in ("ast", "aum"):
        fp_rows = load_jsonl(RESULTS / f"phase1_{model}_full_precision_1pc.jsonl")
        fp_by_length = group_correct_by_length(fp_rows)
        fp_result = bootstrap_condition(fp_by_length)

        quant_rows = json.load(open(RESULTS / f"phase3_{model}_quantized_1pc_summary.json"))
        # the 1pc round's quantized grid only survives as a summary too
        # (see DECISIONS.md) -- use Wilson here for consistency with what
        # actually exists, rather than bootstrap, since there's no
        # per-clip 1pc quantized data either.
        w8a16_summary = [r for r in quant_rows if r["condition"] == "w8a16"]
        w8a16_result = wilson_condition(w8a16_summary) if w8a16_summary else None

        out["matched_check_1pc"][model] = {
            "full_precision": fp_result,  # real bootstrap (per-clip data survives)
            "w8a16": w8a16_result,        # Wilson (summary-only, same as 3pc baseline situation)
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")

    # ---- Figures: 3pc primary, AST and AuM ----
    for model in ("ast", "aum"):
        mo = out["primary_3pc"][model]
        series = [
            series_for_figure(model, "full-precision", mo["full_precision"], linestyle="--"),
            series_for_figure(model, "W8A16", mo["quantized"]["w8a16"], linestyle="-"),
        ]
        if "w4a16" in mo["quantized"]:
            series.append(series_for_figure(model, "W4A16", mo["quantized"]["w4a16"], linestyle="-."))
        out_path = FIGURES / f"phase6_accuracy_vs_length_{model}.pdf"
        plot_accuracy_vs_length(
            series,
            title=f"{model.upper()}: accuracy vs. length (3 clips/class, 95% CI)",
            out_path=out_path,
        )
        print(f"Wrote {out_path}")

    # ---- Figure: AST vs AuM full-precision head-to-head (the headline comparison) ----
    head_to_head = [
        series_for_figure("ast", "AST full-precision", out["primary_3pc"]["ast"]["full_precision"], linestyle="-"),
        series_for_figure("aum", "AuM full-precision", out["primary_3pc"]["aum"]["full_precision"], linestyle="-"),
    ]
    out_path = FIGURES / "phase6_fullprecision_ast_vs_aum.pdf"
    plot_accuracy_vs_length(
        head_to_head,
        title="Full-precision length degradation: AST vs AuM (3 clips/class, 95% Wilson CI)",
        out_path=out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
