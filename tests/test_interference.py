"""Tests for the interference / switchback module.

The load-bearing tests here are not "does the function run". They are:

  * the simulator has a **ground truth** that behaves the way a marketplace does
  * the naive design is **provably** biased, in the direction claimed
  * switchback fixes it, and the *stratified* variant fixes it much harder
  * the estimand choice (bucket-mean vs demand-weighted) is a real bug and the
    test would fail if someone quietly switched the default back
"""
from dataclasses import asdict

import numpy as np
import pytest

from expkit.interference import (
    MarketConfig,
    draw_world,
    global_treatment_effect,
    run_world,
    stratified_analysis,
    switchback_assignment,
    switchback_estimate,
    user_randomised_estimate,
)


@pytest.fixture(scope="module")
def worlds():
    cfg = MarketConfig()
    return [draw_world(MarketConfig(**{**asdict(cfg), "seed": w})) for w in range(12)]


# --- the simulator itself --------------------------------------------------

def test_treatment_helps_when_supply_is_scarce(worlds):
    g = global_treatment_effect(worlds[0])
    assert g["gte_absolute"] > 0.02
    assert 0.0 < g["control_fulfillment"] < 1.0


def test_treatment_does_nothing_when_supply_is_abundant():
    """No scarcity, no effect. A more efficient matcher cannot help when every
    request was already going to be served, and a simulator that shows a gain
    there is measuring an artifact."""
    world = draw_world(MarketConfig(supply_per_hour=2000.0))
    g = global_treatment_effect(world)
    assert g["control_fulfillment"] == pytest.approx(1.0, abs=1e-6)
    assert g["gte_absolute"] == pytest.approx(0.0, abs=1e-6)


def test_fulfillment_never_exceeds_one(worlds):
    res = run_world(worlds[0], np.full(worlds[0]["cfg"].n_buckets, 0.5))
    assert np.nanmax(res["fulfillment"]) <= 1.0 + 1e-9
    assert res["served"].sum() <= res["demand"].sum() + 1e-6


# --- the bias --------------------------------------------------------------

def test_user_randomised_is_biased_upward_under_priority_matching(worlds):
    """The headline claim, asserted rather than narrated."""
    truth = np.mean([global_treatment_effect(w)["gte_absolute"] for w in worlds])
    ab = np.mean([user_randomised_estimate(w)["estimate_absolute"] for w in worlds])
    assert ab > 3 * truth


def test_the_bias_is_displacement_not_the_efficiency_gain(worlds):
    """Falsification: remove the queue-jumping and the inflation must vanish.

    It does more than vanish. Under proportional rationing the treated arm's
    saved supply flows straight back into the shared pool, both arms are rationed
    identically, and the A/B measures *exactly zero* against a real +9.6pp global
    effect. The direction of the bias is a property of the matching policy, not
    of interference in general.
    """
    truth = np.mean([global_treatment_effect(w, priority=False)["gte_absolute"] for w in worlds])
    ab = np.mean([user_randomised_estimate(w, priority=False)["estimate_absolute"] for w in worlds])
    assert truth > 0.02
    assert abs(ab) < 1e-4


# --- switchback ------------------------------------------------------------

def test_switchback_is_far_closer_to_the_truth_than_the_ab(worlds):
    truth = np.array([global_treatment_effect(w)["gte_absolute"] for w in worlds])
    ab = np.array([user_randomised_estimate(w)["estimate_absolute"] for w in worlds])
    sb = np.array([switchback_estimate(w, bucket_len=4, burn_in=1, seed=100 + i, stratify=True)
                   ["estimate_absolute"] for i, w in enumerate(worlds)])
    rmse = lambda e: float(np.sqrt(((e - truth) ** 2).mean()))
    assert rmse(sb) < rmse(ab) / 10


def test_stratifying_on_hour_of_day_cuts_the_spread(worlds):
    """Blocking beats sample size here.

    The unstratified design has the same number of randomisation units; what it
    lacks is balance on the diurnal cycle, and that imbalance -- not the unit
    count -- is the dominant error term.
    """
    plain = np.array([switchback_estimate(w, bucket_len=4, burn_in=0, seed=200 + i)
                      ["estimate_absolute"] for i, w in enumerate(worlds)])
    strat = np.array([switchback_estimate(w, bucket_len=4, burn_in=0, seed=200 + i, stratify=True)
                      ["estimate_absolute"] for i, w in enumerate(worlds)])
    assert strat.std(ddof=1) < plain.std(ddof=1) / 2


def test_stratified_assignment_balances_every_slot_of_day():
    rng = np.random.default_rng(0)
    a = switchback_assignment(14 * 24, bucket_len=4, rng=rng, stratify=True, blocks_per_day=6)
    blocks = a[::4]
    for slot in range(6):
        col = blocks[slot::6]
        assert col.sum() == pytest.approx(len(col) // 2)


def test_unstratified_assignment_does_not_balance_slots():
    """The control for the test above: without stratification, imbalance is real."""
    rng = np.random.default_rng(3)
    a = switchback_assignment(14 * 24, bucket_len=4, rng=rng)
    blocks = a[::4]
    imbalance = max(abs(blocks[s::6].mean() - 0.5) for s in range(6))
    assert imbalance > 0.05


# --- the estimand ----------------------------------------------------------

def test_demand_weighting_matters_when_the_effect_lives_at_peak():
    """The bug that the sweep surfaced.

    At high fill, off-peak buckets are saturated for both arms and carry no
    effect at all, while peak buckets carry all of it. An unweighted mean over
    buckets answers "what did the average hour look like"; the launch delivers
    "what did the average request experience". They differ by ~25% here.
    """
    cfg = MarketConfig(supply_per_hour=900.0)
    ws = [draw_world(MarketConfig(**{**asdict(cfg), "seed": w})) for w in range(10)]
    truth = np.mean([global_treatment_effect(w)["gte_absolute"] for w in ws])
    sbs = [switchback_estimate(w, bucket_len=4, burn_in=1, seed=300 + i, stratify=True)
           for i, w in enumerate(ws)]
    demand = np.mean([s["estimate_demand_weighted"] for s in sbs])
    bucketw = np.mean([s["estimate_bucket_weighted"] for s in sbs])

    assert abs(demand - truth) / truth < 0.06
    assert bucketw < truth * 0.85          # the unweighted estimator attenuates


def test_estimate_absolute_follows_the_weight_argument(worlds):
    sb_d = switchback_estimate(worlds[0], bucket_len=4, seed=1, stratify=True, weight="demand")
    sb_b = switchback_estimate(worlds[0], bucket_len=4, seed=1, stratify=True, weight="bucket")
    assert sb_d["estimate_absolute"] == sb_d["estimate_demand_weighted"]
    assert sb_b["estimate_absolute"] == sb_b["estimate_bucket_weighted"]


# --- inference -------------------------------------------------------------

def test_the_design_matched_interval_is_tighter_than_the_iid_one(worlds):
    """Analysing a blocked design as if it were unblocked throws precision away."""
    cfg = worlds[0]["cfg"]
    sb = switchback_estimate(worlds[0], bucket_len=4, burn_in=1, seed=5, stratify=True)
    strat = stratified_analysis(sb, cfg, bucket_len=4)
    iid_w = sb["ci_iid"][1] - sb["ci_iid"][0]
    strat_w = strat["ci_stratified"][1] - strat["ci_stratified"][0]
    assert strat_w < iid_w


def test_burn_in_discards_data_proportional_to_switch_frequency(worlds):
    short = switchback_estimate(worlds[0], bucket_len=2, burn_in=1, seed=9, stratify=True)
    long_ = switchback_estimate(worlds[0], bucket_len=12, burn_in=1, seed=9, stratify=True)
    assert short["usable_fraction"] < long_["usable_fraction"]
