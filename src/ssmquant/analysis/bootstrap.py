"""Phase 6: bootstrap confidence intervals on classification accuracy.

Two paths, used depending on what data survives for a given result:
- `bootstrap_accuracy_ci`: a real nonparametric bootstrap over per-clip
  correct/incorrect outcomes -- resample clips with replacement, recompute
  accuracy each time, take percentiles of the resulting distribution. This
  needs the actual per-clip rows (not just an aggregate accuracy number).
- `wilson_ci`: an analytic approximation (Wilson score interval) from just
  (k correct, n total) -- used where per-clip rows are unavailable (e.g.
  this project's Phase 1 3pc full-precision baselines, whose per-clip
  files were lost to a process mistake and only survive as aggregate
  accuracy -- see DECISIONS.md). Deliberately NOT a plain normal
  approximation (which breaks down near p=0 or p=1, exactly the regime
  several of this project's real numbers sit in, e.g. AuM's near-chance
  accuracy at extended lengths) -- Wilson stays well-behaved there.
"""
import numpy as np

__all__ = ["bootstrap_accuracy_ci", "wilson_ci"]


def bootstrap_accuracy_ci(correct_flags, n_resamples=10000, ci=0.95, seed=0):
    """Nonparametric bootstrap CI for accuracy from per-clip outcomes.

    correct_flags: iterable of bool (or 0/1), one per clip.
    Returns (point_estimate, ci_low, ci_high, n).
    """
    x = np.asarray(list(correct_flags), dtype=np.float64)
    n = len(x)
    if n == 0:
        raise ValueError("bootstrap_accuracy_ci needs at least one clip")
    point = x.mean()
    if n == 1:
        # A single resampled point is always that one point -- bootstrap
        # degenerates to a point mass, not a meaningful interval.
        return point, point, point, n

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = x[idx].mean(axis=1)

    alpha = 1 - ci
    lo, hi = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), n


def wilson_ci(k, n, ci=0.95):
    """Wilson score interval for a binomial proportion from aggregate
    counts alone (k successes out of n trials) -- no per-clip data needed.
    Stays well-behaved at p near 0 or 1, unlike a plain normal
    approximation (relevant here: several real results in this project
    sit at or near the chance floor, e.g. AuM's extended-length accuracy).
    """
    if n <= 0:
        raise ValueError("wilson_ci needs n > 0")
    from scipy.stats import norm

    p = k / n
    alpha = 1 - ci
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return p, float(lo), float(hi), n
