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
