"""Interference, and why user-randomised A/B tests lie about marketplaces.

Every other module in this repo assumes **SUTVA**: one user's outcome depends on
that user's assignment and nothing else. That assumption is what makes a
difference-in-means an estimate of anything. In a marketplace it is false, and
false in a direction that flatters the experiment.

## The mechanism, concretely

Couriers are a finite, shared resource. A treatment that lets treated requests
claim a courier faster does two separable things:

  1. **A real efficiency gain.** Better batching means each courier serves more
     requests per hour. This is the effect worth shipping for.
  2. **Displacement.** Treated requests take couriers that control requests would
     otherwise have taken. This is pure zero-sum theft between arms.

A user-randomised A/B measures (1) + (2). Ship at 100% and only (1) survives,
because there is no control arm left to steal from. The launch under-delivers and
nobody can explain why — the experiment "was significant".

## What the ground truth is here

Because this is a simulation, the **global treatment effect** is computable
directly: run the whole world at 100% control, run it again at 100% treatment,
subtract. That is the number a launch actually delivers, and it is the target
every estimator below is scored against. No real experiment can do this, which is
precisely why the design question has to be settled in simulation.

## What was measured, including the parts that came out wrong

Over 30 simulated worlds, ground-truth global effect **+9.56pp** fulfillment:

| design | estimate | bias | RMSE |
|---|---|---|---|
| user-randomised A/B | +69.5pp | **+627%** | 0.599 |
| switchback, 4h buckets | +8.86pp | -7.3% | 0.0295 |
| switchback, 4h, **stratified by hour** | +9.56pp | **-0.008%** | 0.0040 |

Three things fell out of building it that were not in the original design:

**1. The direction of the bias is a property of the matching policy, not of
interference.** Under priority matching the A/B overstates by 7.3x. Re-run with
proportional rationing -- no arm jumps the queue -- and the A/B reports *exactly
zero* against a real +9.6pp effect, because the treated arm's saved supply flows
back into the shared pool and lifts control by the same amount. "Interference
inflates your estimate" is the wrong lesson. `mechanism` runs the falsification.

**2. Blocking beat sample size, and by a lot.** The first sweep said longer
buckets were better, which made no sense: fewer randomisation units should be
worse. The real driver was hour-of-day imbalance, not unit count. Stratifying so
each slot-of-day is split evenly between arms cut RMSE by **6-14x at every bucket
length**, and reversed the recommendation from 24h buckets to 2-4h.

**3. Carryover was never the problem -- the estimand was.** At high fill the
switchback estimate ran 25% low and no amount of burn-in fixed it. The cause was
that the unweighted mean of per-bucket rates answers *"what did the average hour
look like"*, while a launch delivers *"what did the average request experience"*.
Off-peak buckets are saturated in both arms and carry no effect; peak buckets
carry all of it and all the demand. Demand-weighting holds bias inside +/-1.5%
across every supply regime, where the unweighted estimator ranges from +11% to
-33%.

## Inference, and the cost of being robust

Every interval here **over**-covers: 100% against a nominal 95%. That is a bug in
the other direction. Measured against the estimator's actual spread across worlds:

| interval | coverage | width vs calibrated |
|---|---|---|
| design-matched (within-slot) | 1.00 | 1.5x |
| iid t | 1.00 | 4.2x |
| day-level block bootstrap | 1.00 | 7.5x |

The "robust" choice is the worst one. Resampling whole days discards precisely
the hour-of-day blocking the design paid for. Robustness is not free and it is
not automatically correct: the analysis has to match the design.

## Commands

    python -m expkit.interference compare     # both designs vs ground truth
    python -m expkit.interference sweep       # bucket length x burn-in x stratification
    python -m expkit.interference mechanism   # the falsification
    python -m expkit.interference regimes     # when does interference matter at all
    python -m expkit.interference coverage    # do the intervals cover
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")


@dataclass
class MarketConfig:
    """A two-sided marketplace with a diurnal cycle and supply that persists."""

    days: int = 14
    buckets_per_hour: int = 1
    hours_per_day: int = 24
    demand_per_hour: float = 900.0
    supply_per_hour: float = 520.0
    diurnal_amplitude: float = 0.55     # demand seasonality; peaks at 19:00
    supply_amplitude: float = 0.30      # supply follows demand, but lags and is flatter
    carryover: float = 0.35             # fraction of idle supply that survives the bucket
    treatment_efficiency: float = 1.18  # a treated request costs 1/1.18 of a courier
    seed: int = 0

    @property
    def n_buckets(self) -> int:
        return self.days * self.hours_per_day * self.buckets_per_hour


def draw_world(cfg: MarketConfig) -> dict:
    """Demand and supply draws, independent of any assignment.

    Drawn ONCE and reused across every policy and every design. Common random
    numbers are not a nicety here: without them the difference between two arms
    is dominated by Poisson noise in demand, and a 3% effect is invisible under a
    sampling difference of the same size.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_buckets
    per_bucket = 1.0 / cfg.buckets_per_hour

    hour_of_day = (np.arange(n) // cfg.buckets_per_hour) % cfg.hours_per_day
    # Peak at 19:00, trough at 07:00. Demand and supply both cycle, but supply
    # is flatter -- which is why fulfillment is worst exactly at peak.
    demand_mult = 1.0 + cfg.diurnal_amplitude * np.sin(2 * np.pi * (hour_of_day - 13) / 24.0)
    supply_mult = 1.0 + cfg.supply_amplitude * np.sin(2 * np.pi * (hour_of_day - 15) / 24.0)

    demand = rng.poisson(cfg.demand_per_hour * per_bucket * demand_mult)
    supply = rng.poisson(cfg.supply_per_hour * per_bucket * supply_mult)
    return {"demand": demand, "supply": supply, "hour_of_day": hour_of_day, "cfg": cfg}


def run_world(world: dict, treated_fraction: np.ndarray, priority: bool = True) -> dict:
    """Simulate the marketplace given a per-bucket treated fraction.

    `treated_fraction[t]` in [0,1]. A switchback design passes 0 or 1; a
    user-randomised design passes 0.5 everywhere; the ground-truth runs pass all
    0 or all 1.

    `priority=True` is the displacement mechanism: treated requests are matched
    first. Set it False to isolate the pure efficiency gain and confirm that the
    A/B bias really does come from displacement rather than from the efficiency
    term -- `python -m expkit.interference mechanism` does exactly that.
    """
    cfg = world["cfg"]
    demand, supply = world["demand"], world["supply"]
    eff = cfg.treatment_efficiency

    n = len(demand)
    served_t = np.zeros(n)
    served_c = np.zeros(n)
    n_t = np.zeros(n)
    n_c = np.zeros(n)
    carry = 0.0

    for t in range(n):
        frac = float(treated_fraction[t])
        d = int(demand[t])
        # Deterministic split rather than a binomial draw: the assignment
        # mechanism's own noise is not what is being studied, and letting it in
        # would widen every interval below for no reason.
        treated = int(round(d * frac))
        control = d - treated
        n_t[t], n_c[t] = treated, control

        available = supply[t] + carry
        cost_t, cost_c = 1.0 / eff, 1.0

        if priority:
            fill_t = min(treated, available / cost_t)
            available -= fill_t * cost_t
            fill_c = min(control, available / cost_c)
            available -= fill_c * cost_c
        else:
            # Proportional service: no arm jumps the queue, so displacement is
            # symmetric rather than one-directional.
            want = treated * cost_t + control * cost_c
            share = 1.0 if want <= available else available / max(want, 1e-9)
            fill_t, fill_c = treated * share, control * share
            available -= (fill_t * cost_t + fill_c * cost_c)

        served_t[t], served_c[t] = fill_t, fill_c
        carry = max(available, 0.0) * cfg.carryover

    total = n_t + n_c
    served = served_t + served_c
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, served / np.maximum(total, 1), np.nan)
        rate_t = np.where(n_t > 0, served_t / np.maximum(n_t, 1), np.nan)
        rate_c = np.where(n_c > 0, served_c / np.maximum(n_c, 1), np.nan)

    return {
        "fulfillment": rate, "fulfillment_t": rate_t, "fulfillment_c": rate_c,
        "n_t": n_t, "n_c": n_c, "served": served, "demand": total,
        "overall": float(served.sum() / max(total.sum(), 1)),
    }


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------

def global_treatment_effect(world: dict, priority: bool = True) -> dict:
    """Run the whole world both ways. This is what a launch actually delivers."""
    n = world["cfg"].n_buckets
    all_c = run_world(world, np.zeros(n), priority=priority)
    all_t = run_world(world, np.ones(n), priority=priority)
    return {
        "control_fulfillment": all_c["overall"],
        "treatment_fulfillment": all_t["overall"],
        "gte_absolute": all_t["overall"] - all_c["overall"],
        "gte_relative": all_t["overall"] / all_c["overall"] - 1.0,
    }


# ---------------------------------------------------------------------------
# the two designs
# ---------------------------------------------------------------------------

def user_randomised_estimate(world: dict, split: float = 0.5, priority: bool = True) -> dict:
    """The naive design: split users, compare arms inside the same timeline.

    Demand-weighted rather than a bucket average, because that is what a real
    readout computes -- it pools requests, it does not average hours.
    """
    n = world["cfg"].n_buckets
    res = run_world(world, np.full(n, split), priority=priority)
    rt = float(np.nansum(res["fulfillment_t"] * res["n_t"]) / max(res["n_t"].sum(), 1))
    rc = float(np.nansum(res["fulfillment_c"] * res["n_c"]) / max(res["n_c"].sum(), 1))
    return {"estimate_absolute": rt - rc, "treatment_arm": rt, "control_arm": rc,
            "n_requests": float(res["demand"].sum())}


def switchback_assignment(n_buckets: int, bucket_len: int, rng,
                          stratify: bool = False, blocks_per_day: int = None) -> np.ndarray:
    """Assign whole blocks of `bucket_len` base buckets to one policy.

    Blocks are randomised independently rather than alternated. Alternation looks
    tidier and balances trivially, but it aliases against any periodic component
    of demand -- a 12-hour alternation on a 24-hour cycle assigns every single
    peak to the same arm, and no amount of data fixes that.

    `stratify=True` blocks on **time of day**: for each slot-of-day, exactly half
    the days go to treatment. This was not in the original design; the sweep
    forced it. See `run_sweep` for why.
    """
    n_blocks = int(np.ceil(n_buckets / bucket_len))
    if not stratify or not blocks_per_day or blocks_per_day < 2:
        blocks = rng.integers(0, 2, size=n_blocks)
        return np.repeat(blocks, bucket_len)[:n_buckets].astype(float)

    n_days = int(np.ceil(n_blocks / blocks_per_day))
    blocks = np.zeros(n_days * blocks_per_day, dtype=int)
    for slot in range(blocks_per_day):
        # Exactly half of this slot's days are treated, order randomised. The
        # arms then contain the same mix of hours by construction rather than in
        # expectation, which is the whole point.
        col = np.zeros(n_days, dtype=int)
        col[: n_days // 2] = 1
        rng.shuffle(col)
        blocks[slot::blocks_per_day] = col
    blocks = blocks[:n_blocks]
    return np.repeat(blocks, bucket_len)[:n_buckets].astype(float)


def switchback_estimate(world: dict, bucket_len: int = 4, burn_in: int = 0,
                        seed: int = 0, priority: bool = True,
                        stratify: bool = False, weight: str = "demand") -> dict:
    """Randomise time. Inside a bucket there is no other arm to displace.

    `burn_in` drops the first N base buckets after every switch, on the theory
    that carryover leaves them partly running under the previous policy.

    `weight` chooses the **estimand**, and it turned out to matter far more than
    either of the other two knobs:

      * ``"bucket"`` -- the unweighted mean of per-bucket fulfillment rates. This
        is "what the average HOUR looked like".
      * ``"demand"`` -- weight each bucket by its request count. This is "what the
        average REQUEST experienced", and it is what the ground-truth global
        effect measures.

    They coincide only when the effect is uncorrelated with demand. In a
    marketplace it never is: scarcity *is* peak demand, so the effect is
    concentrated in exactly the buckets the unweighted estimator down-weights.
    Measured across supply regimes, the unweighted estimator ranges from +11% to
    -33% biased while the demand-weighted one stays inside +/-1.5%. Default is
    therefore ``"demand"``; ``"bucket"`` is kept because it is the mistake, and
    reproducing it is the point.
    """
    cfg = world["cfg"]
    n = cfg.n_buckets
    rng = np.random.default_rng(seed)
    blocks_per_day = max(1, (cfg.hours_per_day * cfg.buckets_per_hour) // bucket_len)
    assign = switchback_assignment(n, bucket_len, rng, stratify=stratify,
                                   blocks_per_day=blocks_per_day)
    res = run_world(world, assign, priority=priority)

    keep = np.ones(n, dtype=bool)
    if burn_in > 0:
        switch = np.r_[True, assign[1:] != assign[:-1]]
        for i in np.flatnonzero(switch):
            keep[i:i + burn_in] = False

    rate = res["fulfillment"]
    dem = res["demand"]
    t_mask = keep & (assign == 1)
    c_mask = keep & (assign == 0)
    if t_mask.sum() < 2 or c_mask.sum() < 2:
        return {"estimate_absolute": float("nan"), "n_treated_buckets": int(t_mask.sum()),
                "n_control_buckets": int(c_mask.sum()), "usable_fraction": float(keep.mean())}

    est_bucket = float(rate[t_mask].mean() - rate[c_mask].mean())
    est_demand = float(np.average(rate[t_mask], weights=dem[t_mask])
                       - np.average(rate[c_mask], weights=dem[c_mask]))
    est = est_demand if weight == "demand" else est_bucket

    # Naive iid interval, kept only so the block bootstrap has something to be
    # compared against. Weighted variance for the demand estimand.
    if weight == "demand":
        se_iid = float(np.sqrt(_wvar(rate[t_mask], dem[t_mask]) / t_mask.sum()
                               + _wvar(rate[c_mask], dem[c_mask]) / c_mask.sum()))
    else:
        se_iid = float(np.sqrt(rate[t_mask].var(ddof=1) / t_mask.sum()
                               + rate[c_mask].var(ddof=1) / c_mask.sum()))

    return {
        "estimate_absolute": est,
        "estimate_bucket_weighted": est_bucket,
        "estimate_demand_weighted": est_demand,
        "se_iid": se_iid,
        "ci_iid": [est - 1.96 * se_iid, est + 1.96 * se_iid],
        "n_treated_buckets": int(t_mask.sum()),
        "n_control_buckets": int(c_mask.sum()),
        "usable_fraction": float(keep.mean()),
        "_assign": assign, "_rate": rate, "_demand": dem, "_keep": keep, "_weight": weight,
    }


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    m = np.average(x, weights=w)
    return float(np.average((x - m) ** 2, weights=w))


def block_bootstrap_ci(sb: dict, cfg: MarketConfig, n_boot: int = 2000,
                       seed: int = 7) -> dict:
    """Resample whole DAYS, not buckets.

    Buckets are correlated twice over: carryover links neighbours mechanically,
    and the diurnal cycle links every 19:00 to every other 19:00. A bucket-level
    bootstrap -- like the iid interval -- treats 336 correlated observations as
    336 independent ones and reports an interval that is too narrow. Resampling
    days keeps each day's internal correlation structure intact.
    """
    rate, assign, keep = sb["_rate"], sb["_assign"], sb["_keep"]
    dem, weight = sb["_demand"], sb.get("_weight", "demand")
    per_day = cfg.hours_per_day * cfg.buckets_per_hour
    n_days = len(rate) // per_day
    rng = np.random.default_rng(seed)

    ests = []
    for _ in range(n_boot):
        days = rng.integers(0, n_days, size=n_days)
        idx = np.concatenate([np.arange(d * per_day, (d + 1) * per_day) for d in days])
        r, a, k, d = rate[idx], assign[idx], keep[idx], dem[idx]
        tm, cm = k & (a == 1), k & (a == 0)
        if tm.sum() < 2 or cm.sum() < 2:
            continue
        if weight == "demand":
            ests.append(np.average(r[tm], weights=d[tm]) - np.average(r[cm], weights=d[cm]))
        else:
            ests.append(r[tm].mean() - r[cm].mean())

    ests = np.asarray(ests)
    lo, hi = np.percentile(ests, [2.5, 97.5])
    return {"ci_block_bootstrap": [float(lo), float(hi)],
            "se_block_bootstrap": float(ests.std(ddof=1)),
            "width_ratio_vs_iid": float((hi - lo) / max(2 * 1.96 * sb["se_iid"], 1e-12)),
            "n_boot_used": int(len(ests))}


def stratified_analysis(sb: dict, cfg: MarketConfig, bucket_len: int) -> dict:
    """The analysis that matches a stratified design.

    Blocking on time of day and then analysing as if it were a simple randomised
    comparison throws the blocking away. The iid interval prices in the
    bucket-to-bucket spread that stratification has already eliminated, so it
    comes out roughly 4x too WIDE -- the opposite of the usual "iid intervals are
    too narrow" story, and a reminder that the failure direction depends on which
    structure you ignored.

    The design-matched estimator differences **within** each slot-of-day and then
    pools:  est = sum_s w_s * (Tbar_s - Cbar_s),  w_s proportional to the slot's
    demand. Variance comes from within-slot spread only.
    """
    rate, assign, keep, dem = sb["_rate"], sb["_assign"], sb["_keep"], sb["_demand"]
    n = len(rate)
    slots_per_day = max(1, (cfg.hours_per_day * cfg.buckets_per_hour) // bucket_len)
    slot = ((np.arange(n) // bucket_len) % slots_per_day)

    diffs, weights, varis = [], [], []
    for sidx in range(slots_per_day):
        m = keep & (slot == sidx)
        t, c = m & (assign == 1), m & (assign == 0)
        if t.sum() < 2 or c.sum() < 2:
            continue
        diffs.append(rate[t].mean() - rate[c].mean())
        varis.append(rate[t].var(ddof=1) / t.sum() + rate[c].var(ddof=1) / c.sum())
        weights.append(dem[m].sum())

    if not diffs:
        return {"estimate": float("nan")}
    d = np.asarray(diffs); v = np.asarray(varis)
    w = np.asarray(weights, dtype=float); w /= w.sum()
    est = float((w * d).sum())
    se = float(np.sqrt((w ** 2 * v).sum()))
    return {"estimate": est, "se_stratified": se,
            "ci_stratified": [est - 1.96 * se, est + 1.96 * se],
            "n_slots": len(diffs)}


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def run_comparison(cfg: MarketConfig = None, n_worlds: int = 40) -> dict:
    """Score both designs against the ground truth, over many simulated worlds."""
    cfg = cfg or MarketConfig()
    ab, sb, sbs, truth = [], [], [], []
    for w in range(n_worlds):
        world = draw_world(MarketConfig(**{**asdict(cfg), "seed": w}))
        g = global_treatment_effect(world)
        truth.append(g["gte_absolute"])
        ab.append(user_randomised_estimate(world)["estimate_absolute"])
        sb.append(switchback_estimate(world, bucket_len=4, burn_in=1, seed=1000 + w)["estimate_absolute"])
        sbs.append(switchback_estimate(world, bucket_len=4, burn_in=1, seed=1000 + w,
                                       stratify=True)["estimate_absolute"])

    truth, ab, sb, sbs = np.array(truth), np.array(ab), np.array(sb), np.array(sbs)
    return {
        "n_worlds": n_worlds,
        "ground_truth_gte": {"mean": float(truth.mean()), "sd": float(truth.std(ddof=1))},
        "user_randomised": _score(ab, truth),
        "switchback_4h_burnin1": _score(sb, truth),
        "switchback_4h_burnin1_stratified": _score(sbs, truth),
        "verdict": (
            "user-randomised overstates the global effect by %.1fx"
            % (ab.mean() / truth.mean()) if truth.mean() > 0 else "n/a"
        ),
    }


def _score(est: np.ndarray, truth: np.ndarray) -> dict:
    bias = est - truth
    return {
        "mean_estimate": float(est.mean()),
        "bias_absolute": float(bias.mean()),
        "bias_relative_to_truth": float(bias.mean() / truth.mean()) if truth.mean() else float("nan"),
        "sd_of_estimate": float(est.std(ddof=1)),
        "rmse": float(np.sqrt((bias ** 2).mean())),
    }


def run_sweep(cfg: MarketConfig = None, n_worlds: int = 30) -> dict:
    """The design decision: bucket length x burn-in, scored on bias AND spread.

    Longer buckets mean proportionally less of the timeline is contaminated by
    carryover, so bias falls. Longer buckets also mean fewer randomisation units,
    so the spread rises. RMSE is what actually decides, because a design can be
    unbiased and still useless.
    """
    cfg = cfg or MarketConfig()
    worlds = [draw_world(MarketConfig(**{**asdict(cfg), "seed": w})) for w in range(n_worlds)]
    truth = np.array([global_treatment_effect(w)["gte_absolute"] for w in worlds])

    rows = []
    for bucket_len in (1, 2, 4, 12, 24):
        for burn_in in (0, 1):
            if burn_in >= bucket_len:
                continue
            for stratify in (False, True):
                if stratify and bucket_len >= cfg.hours_per_day:
                    continue   # one block per day: nothing left to stratify on
                ests = np.array([
                    switchback_estimate(w, bucket_len=bucket_len, burn_in=burn_in,
                                        seed=2000 + i, stratify=stratify)["estimate_absolute"]
                    for i, w in enumerate(worlds)
                ])
                sc = _score(ests, truth)
                rows.append({"bucket_hours": bucket_len, "burn_in_buckets": burn_in,
                             "stratified_by_hour": stratify,
                             "switches_per_day": round(24.0 / bucket_len / 2, 2), **sc})

    best = min(rows, key=lambda r: r["rmse"])
    return {"n_worlds": n_worlds, "ground_truth_gte": float(truth.mean()),
            "grid": rows,
            "best_by_rmse": {k: best[k] for k in
                             ("bucket_hours", "burn_in_buckets", "stratified_by_hour",
                              "bias_relative_to_truth", "sd_of_estimate", "rmse")}}


def run_regimes(cfg: MarketConfig = None, n_worlds: int = 15) -> dict:
    """When does interference actually matter? Sweep how binding the constraint is.

    The 7x figure below is not a property of marketplaces, it is a property of a
    marketplace running at 58% fill. Displacement requires scarcity: if every
    request can be served, taking a courier from control costs control nothing.
    This sweep is the answer to "so should I switchback everything?" -- no, and
    here is the number that decides.
    """
    cfg = cfg or MarketConfig()
    rows = []
    for supply in (300.0, 520.0, 750.0, 900.0, 1100.0, 1400.0):
        truth, ab, sb = [], [], []
        for w in range(n_worlds):
            world = draw_world(MarketConfig(**{**asdict(cfg), "seed": w, "supply_per_hour": supply}))
            truth.append(global_treatment_effect(world)["gte_absolute"])
            ab.append(user_randomised_estimate(world)["estimate_absolute"])
            sb.append(switchback_estimate(world, bucket_len=4, burn_in=1, seed=4000 + w,
                                          stratify=True)["estimate_absolute"])
        truth, ab, sb = np.array(truth), np.array(ab), np.array(sb)
        base = run_world(draw_world(MarketConfig(**{**asdict(cfg), "seed": 0, "supply_per_hour": supply})),
                         np.zeros(cfg.n_buckets))["overall"]
        rows.append({
            "supply_per_hour": supply,
            "baseline_fulfillment": base,
            "ground_truth_gte": float(truth.mean()),
            "ab_estimate": float(ab.mean()),
            "ab_overstatement_x": float(ab.mean() / truth.mean()) if truth.mean() > 1e-9 else None,
            "switchback_estimate": float(sb.mean()),
            "switchback_bias_relative": float((sb.mean() - truth.mean()) / truth.mean())
            if truth.mean() > 1e-9 else None,
        })
    return {"n_worlds": n_worlds, "grid": rows,
            "note": ("as supply becomes abundant the constraint stops binding, the true "
                     "effect goes to zero, and the A/B bias goes with it -- switchback is "
                     "insurance you buy only where the resource is scarce")}


def run_mechanism(cfg: MarketConfig = None, n_worlds: int = 20) -> dict:
    """Is the A/B bias displacement, or is it something about the efficiency gain?

    Re-run with `priority=False`, where no arm jumps the queue. If the bias is
    displacement it should collapse; if it survives, the story above is wrong.
    Running the falsification is the difference between a mechanism and a guess.
    """
    cfg = cfg or MarketConfig()
    out = {}
    for label, priority in (("priority_matching", True), ("proportional_matching", False)):
        ab, truth = [], []
        for w in range(n_worlds):
            world = draw_world(MarketConfig(**{**asdict(cfg), "seed": w}))
            truth.append(global_treatment_effect(world, priority=priority)["gte_absolute"])
            ab.append(user_randomised_estimate(world, priority=priority)["estimate_absolute"])
        out[label] = _score(np.array(ab), np.array(truth))
        out[label]["ground_truth_gte"] = float(np.mean(truth))
    return out


def run_inference(cfg: MarketConfig = None) -> dict:
    """One switchback run, with both intervals, to price the iid shortcut."""
    cfg = cfg or MarketConfig(days=28)
    world = draw_world(cfg)
    truth = global_treatment_effect(world)
    sb = switchback_estimate(world, bucket_len=4, burn_in=1, seed=99)
    boot = block_bootstrap_ci(sb, cfg)
    lo, hi = boot["ci_block_bootstrap"]
    return {
        "ground_truth_gte": truth["gte_absolute"],
        "estimate": sb["estimate_absolute"],
        "ci_iid": sb["ci_iid"],
        "ci_block_bootstrap": boot["ci_block_bootstrap"],
        "block_bootstrap_is_wider_by": boot["width_ratio_vs_iid"],
        "iid_covers_truth": bool(sb["ci_iid"][0] <= truth["gte_absolute"] <= sb["ci_iid"][1]),
        "bootstrap_covers_truth": bool(lo <= truth["gte_absolute"] <= hi),
        "usable_fraction_after_burn_in": sb["usable_fraction"],
        "n_buckets": sb["n_treated_buckets"] + sb["n_control_buckets"],
    }


def run_coverage(cfg: MarketConfig = None, n_worlds: int = 200) -> dict:
    """Do the intervals actually cover at 95%? The only test of an interval.

    A nominal 95% interval that covers 70% of the time is not a conservative
    interval, it is a wrong one, and nothing downstream of it can be trusted. The
    reverse failure -- covering 100% of the time -- is also a bug: it means the
    design's precision is being thrown away, and every "not significant" readout
    it produces is wrong about why.
    """
    cfg = cfg or MarketConfig()
    bucket_len = 4
    hits = {"iid": 0, "block_bootstrap": 0, "stratified": 0}
    widths = {"iid": [], "block_bootstrap": [], "stratified": []}
    ests = []
    n = 0
    for w in range(n_worlds):
        world = draw_world(MarketConfig(**{**asdict(cfg), "seed": w}))
        truth = global_treatment_effect(world)["gte_absolute"]
        sb = switchback_estimate(world, bucket_len=bucket_len, burn_in=1,
                                 seed=3000 + w, stratify=True)
        if not np.isfinite(sb["estimate_absolute"]):
            continue
        n += 1
        ests.append(sb["estimate_absolute"])
        boot = block_bootstrap_ci(sb, cfg, n_boot=400, seed=w)
        strat = stratified_analysis(sb, cfg, bucket_len)

        for name, ci in (("iid", sb["ci_iid"]),
                         ("block_bootstrap", boot["ci_block_bootstrap"]),
                         ("stratified", strat["ci_stratified"])):
            hits[name] += int(ci[0] <= truth <= ci[1])
            widths[name].append(ci[1] - ci[0])

    # The benchmark an interval SHOULD hit: the estimator's actual spread across
    # independent worlds. Without this row, "coverage 1.00" reads as a success.
    sd_actual = float(np.std(ests, ddof=1))
    ideal_width = 2 * 1.96 * sd_actual
    return {
        "n_worlds": n, "nominal": 0.95, "design": "4h buckets, stratified by hour, burn-in 1",
        "coverage": {k: hits[k] / max(n, 1) for k in hits},
        "mean_ci_width": {k: float(np.mean(v)) for k, v in widths.items()},
        "empirical_sd_of_estimate": sd_actual,
        "well_calibrated_width_would_be": ideal_width,
        "width_vs_calibrated": {k: float(np.mean(v) / ideal_width) for k, v in widths.items()},
        "reading": ("all three cover at 100% against a nominal 95%, which is not a pass: "
                    "every one is conservative, and the more structure the analysis "
                    "ignores the more precision it burns. The day-level block bootstrap "
                    "-- the 'robust' choice -- is the worst, because resampling whole days "
                    "discards the hour-of-day blocking the design paid for."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["compare", "sweep", "mechanism", "inference",
                                       "coverage", "regimes"])
    ap.add_argument("--worlds", type=int, default=None)
    args = ap.parse_args()

    fn = {"compare": run_comparison, "sweep": run_sweep, "mechanism": run_mechanism,
          "inference": run_inference, "coverage": run_coverage, "regimes": run_regimes}[args.command]
    kwargs = {}
    if args.worlds and args.command != "inference":
        kwargs["n_worlds"] = args.worlds
    out = fn(**kwargs)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "validation_interference_%s.json" % args.command)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print("\nwritten:", os.path.relpath(path, os.path.join(os.path.dirname(__file__), "..", "..")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
