"""CUPED: variance reduction using pre-experiment data.

The idea in one line: a user's pre-period behaviour predicts their post-period
behaviour, and prediction that does not depend on the treatment can be subtracted
off without biasing the estimate.

    Y_adjusted = Y - theta * (X - E[X])        theta = Cov(Y, X) / Var(X)

`theta` is exactly the OLS slope of Y on X, which is why this is sometimes
described as regression adjustment. Because X is measured **before** the
treatment exists, it cannot be affected by it, so subtracting it removes variance
without moving the expected difference between arms.

Two implementation details that decide whether it works:

* **theta is estimated on the POOLED data** (both arms together), not per arm.
  Estimating it separately per arm lets the treatment influence its own
  adjustment and reintroduces bias -- which is the failure mode that makes CUPED
  "not work" in practice.
* **E[X] is the pooled mean.** Any constant works for unbiasedness, but the
  pooled mean keeps the adjusted metric on the same scale as the original, which
  matters when a PM reads the number.

**When CUPED fails**, stated up front because it is the interview question:
  1. the covariate is uncorrelated with the metric -> theta ~ 0, no reduction,
     and you have added complexity for nothing
  2. new users have no pre-period at all -> no covariate exists; they must be
     handled explicitly (fall back to unadjusted, or use a segment-level mean)
  3. the covariate is affected by the treatment (e.g. measured after exposure)
     -> the adjustment is no longer independent of the treatment and the estimate
     becomes biased. This is the dangerous one, because it fails silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CupedResult:
    theta: float
    correlation: float
    variance_before: float
    variance_after: float
    variance_reduction_pct: float
    effective_sample_multiplier: float
    n_with_covariate: int
    n_without_covariate: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def estimate_theta(y: np.ndarray, x: np.ndarray) -> float:
    """theta = Cov(Y, X) / Var(X), estimated on the pooled sample."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    var_x = x.var(ddof=1)
    if var_x <= 0:
        # A constant covariate carries no information; theta is undefined and
        # zero is the honest answer (no adjustment) rather than a division blow-up.
        return 0.0
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def apply_cuped(y: np.ndarray, x: np.ndarray, theta: float | None = None):
    """Return (adjusted_y, theta). X must be pre-treatment."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if theta is None:
        theta = estimate_theta(y, x)
    return y - theta * (x - x.mean()), theta


def cuped_frame(post: pd.DataFrame, pre: pd.DataFrame, metric: str = "revenue",
                unit: str = "user_id") -> tuple:
    """Join a pre-period covariate onto the post-period frame and adjust.

    Users with no pre-period row get the pooled pre-period mean, which makes
    their adjustment term exactly zero -- i.e. they fall back to unadjusted,
    which is the correct behaviour rather than dropping them (dropping new users
    would change the population the experiment is about).
    """
    pre_agg = pre.groupby(unit, as_index=False)[metric].sum().rename(columns={metric: "pre_metric"})
    merged = post.merge(pre_agg, on=unit, how="left")
    n_missing = int(merged["pre_metric"].isna().sum())
    merged["pre_metric"] = merged["pre_metric"].fillna(merged["pre_metric"].mean())

    y = merged[metric].to_numpy(dtype=float)
    x = merged["pre_metric"].to_numpy(dtype=float)
    adjusted, theta = apply_cuped(y, x)
    merged[metric + "_cuped"] = adjusted

    var_before = float(np.var(y, ddof=1))
    var_after = float(np.var(adjusted, ddof=1))
    corr = float(np.corrcoef(y, x)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    reduction = 100.0 * (1 - var_after / var_before) if var_before > 0 else 0.0

    result = CupedResult(
        theta=theta,
        correlation=corr,
        variance_before=var_before,
        variance_after=var_after,
        variance_reduction_pct=reduction,
        # Sample size needed scales with variance, so a 1 - r^2 variance ratio
        # means the same power at that fraction of the users.
        effective_sample_multiplier=(var_before / var_after) if var_after > 0 else float("inf"),
        n_with_covariate=int(len(merged) - n_missing),
        n_without_covariate=n_missing,
    )
    return merged, result


def theoretical_reduction(correlation: float) -> float:
    """CUPED's variance reduction is exactly r^2, in percent.

    Worth stating because it sets expectations: a covariate correlated 0.3 with
    the metric buys 9% variance reduction, not 30%. Teams routinely expect the
    former number and are disappointed by the latter.
    """
    return 100.0 * correlation ** 2
