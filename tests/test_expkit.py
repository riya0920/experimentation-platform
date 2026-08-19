"""Tests for the assignment and statistics layers.

These are the fast checks that run on every commit. The slow, definitive
validation (1,000 A/A experiments) lives in `expkit.validation` and is run
explicitly -- but the properties it depends on are asserted here cheaply.
"""
import numpy as np
import pytest
from scipy import stats as sps

from expkit.assignment import assign, bucket, in_experiment
from expkit.generator import Effect, ProductConfig, simulate, user_level
from expkit.stats import (
    benjamini_hochberg,
    mde_for_sample_size,
    power_for_sample_size,
    proportion_test,
    sample_size_for_proportion,
    welch_ttest,
)
from expkit.validation import binomial_ci


def test_assignment_is_deterministic():
    a = [assign("u1", "exp_a") for _ in range(5)]
    assert len(set(a)) == 1


def test_assignment_is_uniform_chi_square():
    """Uniformity is verified, not assumed."""
    units = ["u%d" % i for i in range(20_000)]
    counts = {"control": 0, "treatment": 0}
    for u in units:
        counts[assign(u, "exp_uniform")] += 1
    expected = len(units) / 2
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    assert sps.chi2.sf(chi2, df=1) > 0.01, "assignment is not uniform: %r" % counts


def test_different_experiments_assign_independently():
    """Without a per-experiment salt, the same users land in treatment forever."""
    units = ["u%d" % i for i in range(5_000)]
    a = np.array([assign(u, "exp_a") == "treatment" for u in units])
    b = np.array([assign(u, "exp_b") == "treatment" for u in units])
    agreement = (a == b).mean()
    # independent coin flips agree ~50% of the time; a shared salt gives 100%
    assert 0.45 < agreement < 0.55, "experiments are correlated (agreement=%.3f)" % agreement


def test_exposure_ramp_is_independent_of_variant():
    """The ramp must not correlate with the split, or arms are unbalanced."""
    units = ["u%d" % i for i in range(20_000)]
    exposed = [u for u in units if in_experiment(u, "exp_ramp", 0.10)]
    assert 0.08 < len(exposed) / len(units) < 0.12
    treat = sum(assign(u, "exp_ramp") == "treatment" for u in exposed)
    assert 0.45 < treat / len(exposed) < 0.55


def test_weighted_assignment_respects_weights():
    units = ["u%d" % i for i in range(20_000)]
    counts = {"a": 0, "b": 0, "c": 0}
    for u in units:
        counts[assign(u, "exp_w", ("a", "b", "c"), [0.5, 0.3, 0.2])] += 1
    assert abs(counts["a"] / len(units) - 0.5) < 0.02
    assert abs(counts["b"] / len(units) - 0.3) < 0.02
    assert abs(counts["c"] / len(units) - 0.2) < 0.02


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        assign("u1", "e", ("a", "b"), [0.6, 0.6])


def test_bucket_is_in_range():
    assert all(0 <= bucket("u%d" % i, "s", 100) < 100 for i in range(500))


def test_welch_finds_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 3, 4000)
    b_same = rng.normal(10, 3, 4000)
    b_diff = rng.normal(11, 3, 4000)
    assert not welch_ttest(a, b_same).significant
    assert welch_ttest(a, b_diff).significant


def test_welch_handles_unequal_variances():
    """The reason Welch is used instead of Student's t."""
    rng = np.random.default_rng(1)
    a = rng.normal(10, 1, 2000)
    b = rng.normal(10, 8, 2000)   # same mean, wildly different variance
    r = welch_ttest(a, b)
    assert not r.significant


def test_proportion_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(2)
    c = (rng.random(5000) < 0.10).astype(int)
    t = (rng.random(5000) < 0.13).astype(int)
    r = proportion_test(c, t)
    assert r.ci_low < r.absolute_effect < r.ci_high
    assert r.significant


def test_ci_and_pvalue_agree_on_direction():
    rng = np.random.default_rng(3)
    c = (rng.random(8000) < 0.20).astype(int)
    t = (rng.random(8000) < 0.20).astype(int)
    r = proportion_test(c, t)
    # a non-significant result must have a CI containing zero
    if not r.significant:
        assert r.ci_low <= 0 <= r.ci_high


def test_sample_size_shrinks_as_mde_grows():
    big = sample_size_for_proportion(0.10, 0.02)
    small = sample_size_for_proportion(0.10, 0.20)
    assert big > small * 10


def test_mde_and_sample_size_are_inverses():
    n = sample_size_for_proportion(0.10, 0.05)
    recovered = mde_for_sample_size(0.10, n)
    assert abs(recovered - 0.05) < 0.005


def test_power_at_the_designed_sample_size_is_the_designed_power():
    n = sample_size_for_proportion(0.10, 0.10, alpha=0.05, power=0.80)
    assert abs(power_for_sample_size(0.10, 0.10, n) - 0.80) < 0.02


def test_benjamini_hochberg_is_less_conservative_than_bonferroni():
    p = [0.001, 0.008, 0.02, 0.04, 0.3, 0.7]
    rejected, _ = benjamini_hochberg(p, fdr=0.05)
    bonferroni = np.array(p) < 0.05 / len(p)
    assert rejected.sum() >= bonferroni.sum()
    assert rejected[0]
    assert not rejected[-1]


def test_bh_rejects_nothing_when_all_null():
    rng = np.random.default_rng(4)
    rejected, _ = benjamini_hochberg(rng.random(50), fdr=0.05)
    assert rejected.sum() <= 3


def test_wilson_interval_brackets_the_rate():
    lo, hi = binomial_ci(50, 1000)
    assert lo < 0.05 < hi
    assert hi - lo < 0.03


def test_generator_bakes_in_the_true_effect():
    cfg = ProductConfig(n_users=6000, days=5, seed=11)
    null = user_level(simulate(cfg, Effect(), "e_null", with_pre_period=False))
    lifted = user_level(simulate(cfg, Effect(conversion_lift=0.40), "e_lift", with_pre_period=False))

    def gap(u):
        return u[u["variant"] == "treatment"]["converted"].mean() - u[u["variant"] == "control"]["converted"].mean()

    assert gap(lifted) > gap(null)
    assert gap(lifted) > 0.02


def test_ground_truth_is_recorded_on_the_frame():
    df = simulate(ProductConfig(n_users=500, days=2), Effect(conversion_lift=0.1), "e", with_pre_period=False)
    assert df.attrs["ground_truth"]["conversion_lift"] == 0.1
    assert df.attrs["ground_truth"]["is_null"] is False


def test_pre_period_has_no_treatment_effect():
    """CUPED depends on this: the pre-period must be untouched by the treatment."""
    cfg = ProductConfig(n_users=8000, days=6, seed=5)
    df = simulate(cfg, Effect(conversion_lift=0.50), "e_pre", with_pre_period=True)
    pre = user_level(df, "pre")
    c = pre[pre["variant"] == "control"]["converted"].mean()
    t = pre[pre["variant"] == "treatment"]["converted"].mean()
    assert abs(t - c) < 0.03, "treatment leaked into the pre-period"


# --------------------------------------------------------------------------
# CUPED
# --------------------------------------------------------------------------

def test_cuped_reduces_variance_when_covariate_is_predictive():
    from expkit.cuped import apply_cuped

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = 3 * x + rng.normal(0, 1, 5000)      # strongly correlated
    adj, theta = apply_cuped(y, x)
    assert adj.var(ddof=1) < y.var(ddof=1) * 0.2
    assert abs(theta - 3.0) < 0.1


def test_cuped_does_not_shift_the_mean():
    """Unbiasedness: the adjustment subtracts a mean-zero term."""
    from expkit.cuped import apply_cuped

    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, 4000)
    y = 2 * x + rng.normal(0, 3, 4000)
    adj, _ = apply_cuped(y, x)
    assert abs(adj.mean() - y.mean()) < 1e-9


def test_cuped_is_a_noop_when_the_covariate_is_uncorrelated():
    """Failure mode 1: no correlation means no reduction, not a free lunch."""
    from expkit.cuped import apply_cuped

    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 4000)
    y = rng.normal(0, 1, 4000)
    adj, theta = apply_cuped(y, x)
    assert abs(theta) < 0.1
    assert adj.var(ddof=1) > y.var(ddof=1) * 0.9


def test_cuped_handles_a_constant_covariate_without_dividing_by_zero():
    from expkit.cuped import apply_cuped

    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([7.0, 7.0, 7.0, 7.0])
    adj, theta = apply_cuped(y, x)
    assert theta == 0.0
    assert np.allclose(adj, y)


def test_cuped_variance_reduction_equals_r_squared():
    """The expectation-setting fact: reduction is r^2, not r."""
    from expkit.cuped import apply_cuped, theoretical_reduction

    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20000)
    y = 0.5 * x + rng.normal(0, 1, 20000)
    r = float(np.corrcoef(y, x)[0, 1])
    adj, _ = apply_cuped(y, x)
    measured = 100.0 * (1 - adj.var(ddof=1) / y.var(ddof=1))
    assert abs(measured - theoretical_reduction(r)) < 1.5


# --------------------------------------------------------------------------
# sequential testing
# --------------------------------------------------------------------------

def test_msprt_lambda_grows_with_a_larger_effect():
    from expkit.sequential import msprt_lambda

    small = msprt_lambda(effect=0.001, se=0.01, tau=0.02)
    large = msprt_lambda(effect=0.050, se=0.01, tau=0.02)
    assert large > small


def test_always_valid_p_is_monotonically_non_increasing():
    """The running minimum is what makes a stopping rule on it valid."""
    from expkit.sequential import SequentialState, update

    state = SequentialState(tau=0.02)
    ps = []
    for effect in (0.05, 0.001, 0.0005, 0.03):
        update(state, effect, se=0.01, n=1000)
        ps.append(state.always_valid_p)
    assert ps == sorted(ps, reverse=True)


def test_sequential_does_not_fire_on_pure_noise():
    from expkit.sequential import SequentialState, update

    rng = np.random.default_rng(7)
    fired = 0
    for _ in range(200):
        state = SequentialState(tau=0.02, alpha=0.05)
        for step in range(1, 15):
            n = 500 * step
            se = 0.5 / np.sqrt(n)
            update(state, rng.normal(0, se), se, n)
        fired += int(state.crossed)
    # Ville's inequality bounds this by alpha over ALL looks; it is typically
    # far more conservative than the nominal rate.
    assert fired / 200 <= 0.05


def test_confidence_sequence_is_wider_than_a_fixed_horizon_ci():
    """The price of being allowed to look whenever you like."""
    from expkit.sequential import confidence_sequence

    effect, se = 0.02, 0.005
    lo, hi = confidence_sequence(effect, se, tau=0.02, alpha=0.05)
    fixed_half_width = 1.959963985 * se
    assert (hi - lo) / 2 > fixed_half_width
    assert lo < effect < hi
