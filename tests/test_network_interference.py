"""Tests for social-graph interference.

The load-bearing ones pin the two facts that make this module worth having: that
the bias runs the *opposite* way to the marketplace study, and that cluster
randomisation is a real trade rather than a free upgrade. Both were wrong in the
first version and both would look fine in a write-up.
"""
from dataclasses import asdict

import numpy as np
import pytest

from expkit.network_interference import (
    GraphConfig,
    bernoulli_estimate,
    build_graph,
    cluster_estimate,
    exposure_weighted_estimate,
    global_treatment_effect,
    modularity,
    run_sweep,
    treated_fraction_of_neighbours,
)


@pytest.fixture(scope="module")
def graph():
    return build_graph(GraphConfig(n_users=3000, seed=0))


# --- the graph -------------------------------------------------------------

def test_the_graph_has_community_structure():
    """A social graph is not Erdos-Renyi, and the difference is the whole
    subject. Uniform random edges give cluster randomisation nothing to work
    with, and a study on one concludes clustering does not help -- for a graph
    nobody has."""
    g = build_graph(GraphConfig(n_users=3000, seed=0))
    assert modularity(g) > 0.25


def test_modularity_falls_as_the_graph_mixes():
    tight = build_graph(GraphConfig(n_users=3000, p_between=0.00005, seed=0))
    mixed = build_graph(GraphConfig(n_users=3000, p_between=0.006, seed=0))
    assert modularity(tight) > modularity(mixed) + 0.3


def test_exposure_is_the_treated_share_of_neighbours(graph):
    n = len(graph["community"])
    assert treated_fraction_of_neighbours(graph, np.zeros(n)).max() == 0.0
    all_t = treated_fraction_of_neighbours(graph, np.ones(n))
    has_neighbours = np.diff(graph["starts"]) > 0
    assert all_t[has_neighbours].min() == pytest.approx(1.0)


def test_isolated_users_are_kept_not_dropped(graph):
    """They are genuinely unexposed and they are part of the population the
    global effect is defined over. Dropping them would quietly change the
    estimand."""
    n = len(graph["community"])
    exposure = treated_fraction_of_neighbours(graph, np.ones(n))
    assert len(exposure) == n


def test_communities_differ_in_baseline():
    """The term a cluster-randomised design pays for. Without it clustering comes
    out nearly free, which is not a property of clustering -- it is a property of
    pretending every community is the same."""
    g = build_graph(GraphConfig(n_users=3000, community_sd=0.06, seed=0))
    assert g["community_offset"].std() > 0.02
    flat = build_graph(GraphConfig(n_users=3000, community_sd=0.0, seed=0))
    assert flat["community_offset"].std() == 0.0


# --- ground truth ----------------------------------------------------------

def test_no_treatment_effect_and_no_spillover_means_no_gte():
    g = build_graph(GraphConfig(n_users=3000, direct_effect=0.0, spillover=0.0, seed=0))
    assert abs(global_treatment_effect(g)["gte"]) < 0.005


def test_spillover_makes_the_global_effect_larger_than_the_direct_effect():
    """The whole point: at 100% treatment everyone gets both their own effect and
    their neighbours'."""
    g = build_graph(GraphConfig(n_users=3000, direct_effect=0.08, spillover=0.10, seed=0))
    assert global_treatment_effect(g)["gte"] > 0.10


def test_the_ground_truth_is_averaged_because_it_is_itself_an_estimate():
    """The first version used one pair of draws. At 6,000 users that carries a
    standard error near 0.012, which on a small true effect invents a 20% bias
    for a design that has none -- and it did: a spillover-zero row reported +21%.
    """
    g = build_graph(GraphConfig(n_users=3000, seed=0))
    out = global_treatment_effect(g)
    assert out["reps"] > 1
    assert out["gte_se"] < abs(out["gte"]) / 5


# --- the headline: the bias runs the OTHER way -----------------------------

def test_user_randomisation_understates_under_positive_spillover():
    """Opposite sign to the marketplace study, and the more dangerous one: an
    effect that looks smaller than it is reads as a conservative experiment."""
    truths, ests = [], []
    for w in range(6):
        g = build_graph(GraphConfig(n_users=3000, seed=w))
        truths.append(global_treatment_effect(g)["gte"])
        ests.append(bernoulli_estimate(g, seed=100 + w)["estimate"])
    assert np.mean(ests) < np.mean(truths) * 0.7


def test_with_no_spillover_user_randomisation_is_unbiased():
    """The control. Without it, 'user randomisation is biased' is a claim about
    this generator rather than about interference."""
    truths, ests = [], []
    for w in range(6):
        g = build_graph(GraphConfig(n_users=3000, spillover=0.0, seed=w))
        truths.append(global_treatment_effect(g)["gte"])
        ests.append(bernoulli_estimate(g, seed=200 + w)["estimate"])
    assert abs(np.mean(ests) - np.mean(truths)) < 0.02


def test_cluster_randomisation_reduces_the_bias():
    truths, bern, clus = [], [], []
    for w in range(6):
        g = build_graph(GraphConfig(n_users=3000, seed=w))
        truths.append(global_treatment_effect(g)["gte"])
        bern.append(bernoulli_estimate(g, seed=300 + w)["estimate"])
        clus.append(cluster_estimate(g, seed=300 + w)["estimate"])
    truth = np.mean(truths)
    assert abs(np.mean(clus) - truth) < abs(np.mean(bern) - truth)


# --- and it is a trade, not an upgrade -------------------------------------

def test_cluster_randomisation_is_worse_when_there_is_no_interference():
    """The finding the first generator hid. With no spillover to correct,
    clustering buys nothing and costs the between-community variance."""
    out = run_sweep(GraphConfig(n_users=3000), n_worlds=6)
    zero = next(r for r in out["rows"] if r["spillover"] == 0.0)
    assert zero["graph_cluster"]["rmse"] > zero["bernoulli"]["rmse"]
    assert not zero["cluster_wins_on_rmse"]


def test_there_is_a_crossover_and_it_is_not_at_zero():
    out = run_sweep(GraphConfig(n_users=3000), n_worlds=6)
    assert out["crossover_spillover"] is not None
    assert out["crossover_spillover"] > 0.0


def test_the_crossover_test_requires_a_margin_not_a_bare_comparison():
    """Two RMSEs within Monte Carlo error of each other are a tie, and calling a
    tie a win reports a crossover at exactly the setting with nothing to fix."""
    out = run_sweep(GraphConfig(n_users=3000), n_worlds=6)
    for row in out["rows"]:
        if row["tied_within_noise"]:
            assert not row["cluster_wins_on_rmse"]


# --- the exposure-weighted estimator ---------------------------------------

def test_the_exposure_estimator_is_nearly_unbiased_and_still_loses():
    """Unbiasedness is not the goal. It conditions its way to a good point
    estimate and throws away 97% of the sample doing it, so its RMSE is worse
    than the biased cluster design's."""
    truths, expo, clus = [], [], []
    for w in range(8):
        g = build_graph(GraphConfig(n_users=3000, seed=w))
        truths.append(global_treatment_effect(g)["gte"])
        expo.append(exposure_weighted_estimate(g, seed=400 + w)["estimate"])
        clus.append(cluster_estimate(g, seed=400 + w)["estimate"])
    truth = np.mean(truths)
    expo = np.array([e for e in expo if np.isfinite(e)])
    assert len(expo) >= 4
    assert abs(np.mean(expo) - truth) < abs(np.mean(clus) - truth), "should be less biased"
    assert expo.std(ddof=1) > np.std(clus, ddof=1), "and pay for it in variance"


def test_the_exposure_estimator_reports_how_little_data_it_used(graph):
    """How little depends on graph density -- 2.8% at the module's default
    6,000-user config, 13% on this smaller sparser one. The assertion is loose on
    purpose: the claim is that the estimator discards most of the sample, not
    that it discards a particular fraction."""
    out = exposure_weighted_estimate(graph, seed=1)
    assert out["usable_fraction"] < 0.25


def test_the_exposure_estimator_refuses_rather_than_guessing_on_thin_cells():
    """A dense graph leaves no pure-control users at all, and an estimate from
    five of them would be noise with a confident sign.

    This graph is also the one that found the reduceat crash: at 1,500 users the
    trailing users can have no edges at all, and the exposure calculation indexed
    past the end of the neighbour array."""
    g = build_graph(GraphConfig(n_users=1500, p_within=0.30, p_between=0.05, seed=0))
    out = exposure_weighted_estimate(g, seed=1)
    assert not np.isfinite(out["estimate"])
    assert "too few" in out["failed"]
