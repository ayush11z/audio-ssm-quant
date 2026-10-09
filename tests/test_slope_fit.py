"""Phase 6 gate: unit tests for src/ssmquant/analysis/slope_fit.py --
recovers a known trend exactly on noiseless synthetic data, and the
bootstrap CI behaves sensibly, before trusting any slope computed from
real project data."""
import numpy as np
import pytest

from ssmquant.analysis.slope_fit import bootstrap_slope_ci, fit_slope


def test_fit_slope_recovers_exact_noiseless_trend():
    # y = 1.0 - 0.2 * log10(x), evaluated at the project's real lengths
    lengths = [20, 40, 80, 160]
    true_slope, true_intercept = -0.2, 1.0
    accuracies = [true_intercept + true_slope * np.log10(l) for l in lengths]
    slope, intercept = fit_slope(lengths, accuracies)
    assert abs(slope - true_slope) < 1e-9
    assert abs(intercept - true_intercept) < 1e-9


def test_fit_slope_needs_at_least_two_points():
    with pytest.raises(ValueError):
        fit_slope([20], [0.5])


def test_bootstrap_slope_ci_detects_clear_negative_trend():
    """A strong, unambiguous degradation (high accuracy at 20s, near-zero
    at 160s, large n at every length) should give a CI entirely below
    zero -- this is the real shape of this project's AST/AuM data."""
    rng = np.random.default_rng(0)
    true_acc = {20: 0.7, 40: 0.4, 80: 0.15, 160: 0.03}
    flags = {length: (rng.random(1000) < acc) for length, acc in true_acc.items()}
    point, lo, hi = bootstrap_slope_ci(flags, n_resamples=1000, seed=1)
    assert point < 0
    assert hi < 0  # CI doesn't straddle zero -- a real, detectable trend


def test_bootstrap_slope_ci_no_trend_straddles_zero():
    """Flat accuracy across all four lengths (no real trend) should give
    a CI that includes zero -- confirms the test above isn't just always
    reporting a negative slope regardless of the data."""
    rng = np.random.default_rng(0)
    flags = {length: (rng.random(1000) < 0.3) for length in (20, 40, 80, 160)}
    point, lo, hi = bootstrap_slope_ci(flags, n_resamples=1000, seed=1)
    assert lo < 0 < hi


def test_bootstrap_slope_ci_rejects_wrong_lengths():
    with pytest.raises(ValueError):
        bootstrap_slope_ci({20: [True, False], 40: [True]})  # missing 80, 160
