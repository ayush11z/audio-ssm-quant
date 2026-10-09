"""Phase 6 gate: unit tests for src/ssmquant/analysis/bootstrap.py --
correct behavior on degenerate inputs (all-correct, n=1, p near 0/1)
before trusting any CI computed from real project data."""
import numpy as np
import pytest

from ssmquant.analysis.bootstrap import bootstrap_accuracy_ci, wilson_ci


def test_bootstrap_all_correct_is_degenerate_at_one():
    point, lo, hi, n = bootstrap_accuracy_ci([True] * 50, n_resamples=500, seed=0)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0
    assert n == 50


def test_bootstrap_all_wrong_is_degenerate_at_zero():
    point, lo, hi, n = bootstrap_accuracy_ci([False] * 50, n_resamples=500, seed=0)
    assert point == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_single_clip_is_a_point_not_an_interval():
    """n=1 can't produce a meaningful resampled distribution -- every
    resample just reselects that one clip -- so the CI should collapse
    to the point estimate rather than claiming false precision."""
    point, lo, hi, n = bootstrap_accuracy_ci([True], n_resamples=500, seed=0)
    assert point == lo == hi == 1.0
    assert n == 1


def test_bootstrap_ci_contains_true_rate_for_large_balanced_sample():
    rng = np.random.default_rng(42)
    flags = rng.random(2000) < 0.5
    point, lo, hi, n = bootstrap_accuracy_ci(flags, n_resamples=2000, seed=1)
    assert lo < 0.5 < hi
    assert abs(point - 0.5) < 0.05


def test_bootstrap_ci_narrows_with_larger_sample():
    """Same true rate, more clips -> tighter CI -- a basic sanity check
    that the interval actually reflects sample size, not a fixed width."""
    rng = np.random.default_rng(0)
    small = rng.random(50) < 0.3
    large = rng.random(5000) < 0.3
    _, lo_small, hi_small, _ = bootstrap_accuracy_ci(small, n_resamples=2000, seed=2)
    _, lo_large, hi_large, _ = bootstrap_accuracy_ci(large, n_resamples=2000, seed=2)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_accuracy_ci([])


def test_wilson_ci_matches_known_reference_value():
    """k=50, n=100 (p=0.5) at 95% CI: textbook Wilson interval is
    approximately (0.404, 0.596) -- checked to 2 decimal places, not
    exact equality, since this is a closed-form formula not a stochastic
    estimate."""
    p, lo, hi, n = wilson_ci(50, 100, ci=0.95)
    assert p == 0.5
    assert abs(lo - 0.404) < 0.01
    assert abs(hi - 0.596) < 0.01


def test_wilson_ci_stays_in_bounds_near_zero():
    """p near 0 (relevant: this project's real AuM extended-length
    accuracy sits near chance, ~0.03-0.09) -- Wilson must not produce a
    negative lower bound, unlike a naive normal approximation would."""
    p, lo, hi, n = wilson_ci(3, 315, ci=0.95)
    assert lo >= 0.0
    assert lo < p < hi

    p0, lo0, hi0, _ = wilson_ci(0, 315, ci=0.95)
    assert p0 == 0.0
    assert lo0 == 0.0
    assert hi0 > 0.0


def test_wilson_ci_stays_in_bounds_near_one():
    p, lo, hi, n = wilson_ci(314, 315, ci=0.95)
    assert hi <= 1.0

    p1, lo1, hi1, _ = wilson_ci(315, 315, ci=0.95)
    assert p1 == 1.0
    assert hi1 == pytest.approx(1.0)
    assert lo1 < 1.0


def test_wilson_ci_zero_n_raises():
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
