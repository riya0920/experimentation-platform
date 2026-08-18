"""Simulated product with a KNOWN ground truth.

This is the foundation of the whole platform: because the true effect is an input
rather than something to be estimated, every statistical claim the platform makes
can be checked against reality. A platform validated only on real data cannot be
validated at all -- you never learn whether the answer was right.

The generator produces the messiness that actually breaks naive analyses:
  * heavy-tailed revenue (a few whales dominate the variance)
  * per-user heterogeneity that persists across the pre-period -> makes CUPED work
  * day-of-week seasonality -> makes fixed-horizon peeking look "significant"
  * varying sessions per user -> makes the unit of analysis matter
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assignment import assign, in_experiment


@dataclass
class ProductConfig:
    n_users: int = 20_000
    days: int = 14
    base_conversion: float = 0.12
    base_revenue_mean: float = 18.0
    revenue_sigma: float = 1.1          # lognormal sigma -> heavy tail
    user_quality_sd: float = 0.45       # persistent per-user effect; CUPED exploits this
    weekend_lift: float = 0.15          # seasonality
    sessions_lambda: float = 2.4
    seed: int = 0


@dataclass
class Effect:
    """The ground truth. `conversion_lift` is RELATIVE (0.05 = +5%)."""
    conversion_lift: float = 0.0
    revenue_lift: float = 0.0
    latency_ms_delta: float = 0.0       # guardrail: positive = slower = worse


def simulate(cfg: ProductConfig, effect: Effect, experiment: str = "exp_001",
             exposure_rate: float = 1.0, with_pre_period: bool = True) -> pd.DataFrame:
    """One experiment's worth of user-day data with the true effect baked in."""
    rng = np.random.default_rng(cfg.seed)

    user_ids = np.array(["u%07d" % i for i in range(cfg.n_users)])
    # Persistent per-user quality: the same user converts and spends above or
    # below average consistently. This is what makes a pre-period covariate
    # predictive, and therefore what makes CUPED reduce variance.
    quality = rng.normal(0, cfg.user_quality_sd, size=cfg.n_users)

    exposed = np.array([in_experiment(u, experiment, exposure_rate) for u in user_ids])
    variant = np.array([assign(u, experiment) if e else "excluded" for u, e in zip(user_ids, exposed)])

    rows = []
    n_pre = cfg.days if with_pre_period else 0
    for day in range(-n_pre, cfg.days):
        is_pre = day < 0
        weekend = (day % 7) in (5, 6)
        season = (1.0 + cfg.weekend_lift) if weekend else 1.0

        sessions = rng.poisson(cfg.sessions_lambda, size=cfg.n_users)
        active = sessions > 0

        # Treatment effect applies only in the post-period, only to treated users.
        treated = (variant == "treatment") & (not is_pre)
        conv_p = np.clip(
            cfg.base_conversion * season * np.exp(quality) * (1.0 + np.where(treated, effect.conversion_lift, 0.0)),
            0.0, 0.98,
        )
        converted = rng.random(cfg.n_users) < conv_p

        rev_mu = np.log(cfg.base_revenue_mean) + quality + np.where(treated, np.log1p(effect.revenue_lift), 0.0)
        revenue = np.where(converted, rng.lognormal(rev_mu, cfg.revenue_sigma), 0.0)

        latency = rng.gamma(shape=9.0, scale=12.0, size=cfg.n_users) + np.where(treated, effect.latency_ms_delta, 0.0)

        rows.append(
            pd.DataFrame(
                {
                    "user_id": user_ids,
                    "day": day,
                    "period": np.where(is_pre, "pre", "post"),
                    "variant": variant,
                    "exposed": exposed,
                    "sessions": sessions,
                    "converted": converted.astype(int),
                    "revenue": revenue,
                    "latency_ms": latency,
                }
            )[active]
        )

    df = pd.concat(rows, ignore_index=True)
    df.attrs["ground_truth"] = {
        "conversion_lift": effect.conversion_lift,
        "revenue_lift": effect.revenue_lift,
        "latency_ms_delta": effect.latency_ms_delta,
        "is_null": effect.conversion_lift == 0 and effect.revenue_lift == 0,
    }
    return df


def user_level(df: pd.DataFrame, period: str = "post") -> pd.DataFrame:
    """Aggregate to one row per user -- the randomisation unit.

    Analysing at the session or event level when randomisation happened at the
    user level is the single most common way an A/B analysis produces a
    confidently wrong p-value: the rows are not independent, the effective sample
    size is the number of USERS, and the variance is understated by roughly the
    average sessions-per-user. The platform aggregates here so no analysis can
    accidentally do otherwise.
    """
    sub = df[(df["period"] == period) & (df["variant"] != "excluded")]
    out = sub.groupby(["user_id", "variant"], as_index=False).agg(
        sessions=("sessions", "sum"),
        converted=("converted", "max"),
        revenue=("revenue", "sum"),
        latency_ms=("latency_ms", "mean"),
    )
    return out
