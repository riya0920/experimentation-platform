"""The statistics engine: power/MDE up front, then a fixed-horizon readout.

Everything here is validated against known-truth data in `validation.py`. That
is the difference between a stats engine and a pile of scipy calls: the FPR and
the power this module *claims* are checked against what it actually delivers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Readout:
    metric: str
    control_n: int
    treatment_n: int
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    alpha: float
    method: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05, metric: str = "metric") -> Readout:
    """Welch's t-test: does NOT assume equal variances between arms.

    Student's t assumes equal variance, and a treatment that changes revenue
    almost always changes its variance too -- so the equal-variance assumption is
    wrong in exactly the case you care about. Welch costs nothing when variances
    happen to be equal, so there is no reason to use anything else here.
    """
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    nc, nt = len(c), len(t)
    mc, mt = c.mean(), t.mean()
    vc, vt = c.var(ddof=1), t.var(ddof=1)
    se = math.sqrt(vc / nc + vt / nt)

    if se == 0:
        return Readout(metric, nc, nt, mc, mt, mt - mc, 0.0, 0.0, 0.0, 1.0, False, alpha, "welch_t")

    # Welch-Satterthwaite degrees of freedom
    dof = (vc / nc + vt / nt) ** 2 / ((vc / nc) ** 2 / (nc - 1) + (vt / nt) ** 2 / (nt - 1))
    tstat = (mt - mc) / se
    p = 2 * stats.t.sf(abs(tstat), dof)
    crit = stats.t.ppf(1 - alpha / 2, dof)
    diff = mt - mc
    return Readout(
        metric=metric, control_n=nc, treatment_n=nt, control_mean=mc, treatment_mean=mt,
        absolute_effect=diff, relative_effect=(diff / mc if mc else float("nan")),
        ci_low=diff - crit * se, ci_high=diff + crit * se, p_value=p,
        significant=p < alpha, alpha=alpha, method="welch_t",
    )


def proportion_test(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05,
                    metric: str = "conversion") -> Readout:
    """Two-proportion z-test with an unpooled SE for the confidence interval.

    Deliberate subtlety: the p-value uses the POOLED standard error (correct under
    the null hypothesis being tested), while the confidence interval uses the
    UNPOOLED standard error (correct for estimating the true difference). Using
    one for both is a real and common inconsistency -- it produces intervals that
    disagree with their own p-value near the boundary.
    """
    c, t = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    nc, nt = len(c), len(t)
    pc, pt = c.mean(), t.mean()
    diff = pt - pc

    pooled = (c.sum() + t.sum()) / (nc + nt)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / nc + 1 / nt))
    se_unpooled = math.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)

    if se_pooled == 0:
        return Readout(metric, nc, nt, pc, pt, diff, 0.0, 0.0, 0.0, 1.0, False, alpha, "two_proportion_z")

    z = diff / se_pooled
    p = 2 * stats.norm.sf(abs(z))
    crit = stats.norm.ppf(1 - alpha / 2)
    return Readout(
        metric=metric, control_n=nc, treatment_n=nt, control_mean=pc, treatment_mean=pt,
        absolute_effect=diff, relative_effect=(diff / pc if pc else float("nan")),
        ci_low=diff - crit * se_unpooled, ci_high=diff + crit * se_unpooled,
        p_value=p, significant=p < alpha, alpha=alpha, method="two_proportion_z",
    )


def sample_size_for_proportion(baseline: float, mde_relative: float, alpha: float = 0.05,
                               power: float = 0.80, two_sided: bool = True) -> int:
    """Users PER ARM needed to detect a relative lift with the stated power."""
    p1 = baseline
    p2 = baseline * (1 + mde_relative)
    # A relative lift on a high baseline can imply a probability above 1, which is
    # not a small numerical annoyance -- it is a meaningless question. Reject it
    # explicitly rather than returning a number from a sqrt of a negative.
    if not 0.0 < p1 < 1.0:
        raise ValueError("baseline must be in (0, 1), got %r" % baseline)
    if not 0.0 < p2 < 1.0:
        raise ValueError(
            "baseline %.4f with a %+.1f%% relative lift implies a rate of %.4f, which is outside (0, 1). "
            "On a high baseline, ask for an absolute effect instead." % (p1, 100 * mde_relative, p2)
        )
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p2 - p1) ** 2))


def mde_for_sample_size(baseline: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """The relative lift you can detect with the sample you actually have.

    This is the number to state BEFORE running. A test powered to detect +10%
    that returns "not significant" has not shown the feature does nothing; it has
    shown the effect is probably under 10%, which is a completely different claim.
    """
    # Upper bound is set by the baseline: a relative lift larger than
    # (1 - baseline) / baseline would push the treatment rate past 1.0. On a
    # baseline of 0.75 the largest meaningful relative lift is +33%.
    hi_cap = (0.999 - baseline) / baseline
    if hi_cap <= 1e-6:
        return float("nan")
    lo, hi = 1e-6, min(5.0, hi_cap)
    for _ in range(200):
        mid = (lo + hi) / 2
        if sample_size_for_proportion(baseline, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def power_for_sample_size(baseline: float, mde_relative: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """Achieved power -- reported on every readout, not just planned for."""
    p1 = baseline
    p2 = min(max(baseline * (1 + mde_relative), 1e-9), 1 - 1e-9)
    se = math.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    if se == 0:
        return float("nan")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z_alpha - abs(p2 - p1) / se) + stats.norm.cdf(-z_alpha - abs(p2 - p1) / se))


def benjamini_hochberg(p_values, fdr: float = 0.05):
    """Multiple-comparison control across the metrics in one readout.

    BH controls the false DISCOVERY rate rather than the family-wise error rate.
    That is the right trade for a metrics dashboard: Bonferroni over 15 guardrail
    metrics is so conservative that nothing is ever significant, and the cost of
    an occasional false discovery among many true ones is lower than the cost of
    being blind. FWER control is the right choice for the single decision metric.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = fdr * (np.arange(1, n + 1)) / n
    passed = ranked <= thresholds
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    rejected = np.zeros(n, dtype=bool)
    if k:
        rejected[order[:k]] = True
    return rejected, thresholds[order.argsort()]
