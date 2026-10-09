"""Phase 6: slope fits for "how fast does accuracy degrade with length".

Fits accuracy vs log10(length_sec) with ordinary least squares -- log
length rather than raw length because the real degradation curves in this
project (see DECISIONS.md's Phase 1/3 tables) visibly flatten out at long
lengths rather than falling linearly, closer to a log-linear shape, and a
log-x slope gives one comparable number ("accuracy points lost per decade
of length") across both models despite their very different absolute
degradation rates.

Deliberately excludes the "native" condition from every fit here: native
clips are the original ~1s recordings with no synthetic background noise
spliced in, not just "the shortest length" -- mixing it into a length
regression would conflate "no synthetic noise" with "short length". Every
fit here uses only the four extended lengths (20/40/80/160s), which share
the same construction (original clip + background noise, see
scripts/phase1_build_length_extended_dataset.py).
"""
import numpy as np

__all__ = ["fit_slope", "bootstrap_slope_ci"]

EXTENDED_LENGTHS_SEC = (20, 40, 80, 160)


def fit_slope(lengths_sec, accuracies):
    """Ordinary least squares fit of accuracy ~ log10(length_sec).

    lengths_sec, accuracies: equal-length sequences, one point per length
    group (not per clip -- this fits the aggregate accuracy curve).
    Returns (slope, intercept) where slope is in accuracy-points per
    decade of length (e.g. slope=-0.3 means accuracy drops ~0.3 for every
    10x increase in length).
    """
    x = np.log10(np.asarray(lengths_sec, dtype=np.float64))
    y = np.asarray(accuracies, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("fit_slope needs at least 2 length points")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def bootstrap_slope_ci(per_length_correct_flags, n_resamples=10000, ci=0.95, seed=0):
    """Stratified bootstrap CI for the log-length slope.

    per_length_correct_flags: {length_sec (int): array-like of bool},
    covering exactly the four extended lengths -- one array of per-clip
    correct/incorrect outcomes per length. Each bootstrap iteration
    resamples WITHIN each length group independently (stratified, so
    every length keeps its own real sample size each time), recomputes
    that resample's per-length accuracy, and refits the slope -- this is
    what makes the CI reflect the actual per-clip sampling noise at each
    length, not just noise in a single pooled resample.

    Returns (point_slope, ci_low, ci_high).
    """
    lengths = sorted(per_length_correct_flags.keys())
    if set(lengths) != set(EXTENDED_LENGTHS_SEC):
        raise ValueError(
            f"expected exactly the four extended lengths {EXTENDED_LENGTHS_SEC}, got {lengths}"
        )
    arrays = [np.asarray(list(per_length_correct_flags[length_sec]), dtype=np.float64) for length_sec in lengths]
    point_accuracies = [a.mean() for a in arrays]
    point_slope, _ = fit_slope(lengths, point_accuracies)

    rng = np.random.default_rng(seed)
    slopes = np.empty(n_resamples)
    for i in range(n_resamples):
        resampled_accuracies = []
        for a in arrays:
            idx = rng.integers(0, len(a), size=len(a))
            resampled_accuracies.append(a[idx].mean())
        slope, _ = fit_slope(lengths, resampled_accuracies)
        slopes[i] = slope

    alpha = 1 - ci
    lo, hi = np.percentile(slopes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point_slope, float(lo), float(hi)
