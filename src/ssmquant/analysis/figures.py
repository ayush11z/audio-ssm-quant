"""Phase 6: publication-style figures (vector PDF) for the length-vs-
accuracy degradation curves -- the paper's central comparison."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # no display available when run from a script/CI

EXTENDED_LENGTHS_SEC = (20, 40, 80, 160)

MODEL_COLORS = {"ast": "#1f77b4", "aum": "#d62728"}
MODEL_LABELS = {"ast": "AST (Transformer)", "aum": "AuM (Mamba SSM)"}


def plot_accuracy_vs_length(series, title, out_path):
    """series: list of dicts, each:
      {"model": "ast"|"aum", "label": str, "linestyle": str,
       "lengths": [20,40,80,160], "accuracy": [...], "ci_low": [...] or None,
       "ci_high": [...] or None, "native_accuracy": float or None}
    Draws one line per series, shaded CI band where ci_low/ci_high given,
    native plotted as a separate marker at the left edge (not connected
    by a line to the extended points -- see slope_fit.py's docstring for
    why native is excluded from the length trend itself).
    Saves a vector PDF to out_path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for s in series:
        color = MODEL_COLORS.get(s["model"], "#333333")
        ax.plot(s["lengths"], s["accuracy"], marker="o", color=color,
                linestyle=s.get("linestyle", "-"), label=s["label"], linewidth=2)
        if s.get("ci_low") is not None and s.get("ci_high") is not None:
            ax.fill_between(s["lengths"], s["ci_low"], s["ci_high"], color=color, alpha=0.15)
        if s.get("native_accuracy") is not None:
            ax.scatter([1], [s["native_accuracy"]], marker="s", color=color, s=40, zorder=5)

    chance = 1 / 35
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1, label=f"chance (1/35 = {chance:.3f})")

    ax.set_xscale("log")
    ax.set_xticks([1] + list(EXTENDED_LENGTHS_SEC))
    ax.set_xticklabels(["native\n(~1s)"] + [f"{l}s" for l in EXTENDED_LENGTHS_SEC])
    ax.set_xlabel("Input length")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


TARGET_COLORS = {"delta": "#ff7f0e", "A": "#2ca02c", "BC": "#9467bd", "state": "#d62728"}
TARGET_LABELS = {"delta": "Δ", "A": "A", "BC": "B,C", "state": "state h"}


def plot_accuracy_vs_bits(series, title, out_path, n_classes=35):
    """Phase 4's SSM-internal ablation: accuracy vs. quantization bit-width,
    one line per target tensor.

    series: list of dicts, each:
      {"target": "delta"|"A"|"BC"|"state", "bits": [8,6,4],
       "accuracy": [...], "ci_low": [...] or None, "ci_high": [...] or None}
    Bit-width decreases left to right (8 -> 4, more aggressive quantization
    rightward), matching the length-vs-accuracy figure's convention of
    "more stress -> rightward."
    Saves a vector PDF to out_path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for s in series:
        color = TARGET_COLORS.get(s["target"], "#333333")
        label = TARGET_LABELS.get(s["target"], s["target"])
        ax.plot(s["bits"], s["accuracy"], marker="o", color=color, linewidth=2, label=label)
        if s.get("ci_low") is not None and s.get("ci_high") is not None:
            ax.fill_between(s["bits"], s["ci_low"], s["ci_high"], color=color, alpha=0.15)

    chance = 1 / n_classes
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1, label=f"chance (1/{n_classes} = {chance:.3f})")

    ax.set_xticks([8, 6, 4])
    ax.invert_xaxis()
    ax.set_xlabel("Quantization bit-width (lower = more aggressive)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
