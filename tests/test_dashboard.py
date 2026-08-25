"""Tests for the SRM gate and the results page.

A dashboard test that only asserts "the file was written" is worthless - the
failure mode of a results page is not a crash, it is rendering a confident number
that should never have been shown. So the tests here check the *gate*: that a
broken assignment produces INVALID rather than a big win, and that the page says
so where a reader will see it.
"""
import numpy as np
import pytest

from expkit.dashboard import _ci_svg, build, render
from expkit.generator import Effect, ProductConfig, simulate, user_level
from expkit.assignment import assign
from expkit.readout import analyse, srm_check


# --- the SRM check itself --------------------------------------------------

def test_srm_passes_on_a_balanced_split():
    r = srm_check(10_000, 10_050)
    assert r["passed"]
    assert r["p_value"] > 0.001


def test_srm_catches_a_small_but_real_imbalance():
    """1.9% missing treatment users at n=39k -- detectable at the 0.001 threshold.

    A 1.5% shortfall at the same n is NOT detectable (p=0.13), which is the more
    useful half of this fact: the check is a floor on how broken assignment can
    be before anyone notices, not a proof that it is fine.
    """
    assert not srm_check(20_000, 19_250)["passed"]
    assert srm_check(20_000, 19_700)["passed"]


def test_srm_is_quiet_on_the_real_assignment_function():
    """The check has to be quiet on healthy experiments or it gets ignored.

    Run against the ACTUAL hash assignment over many experiment salts, not
    against binomial draws. That distinction matters here: assignment is
    deterministic in user_id, so for a FIXED user population and a FIXED
    experiment name the split is not random at all -- it is whatever the hash
    says. Only the salt varies it. Simulating binomial counts would have tested
    numpy rather than the thing that can actually be broken.

    Measured over 1,500 salts at n=40,000: 0 fires at the 0.001 threshold, 3.7%
    at 0.05, median p 0.497.
    """
    users = ["u%07d" % i for i in range(20_000)]
    fired = 0
    for e in range(120):
        v = np.array([assign(u, "srmfpr_%d" % e) for u in users])
        t = int((v == "treatment").sum())
        fired += int(not srm_check(len(v) - t, t)["passed"])
    assert fired <= 2


def test_srm_handles_an_empty_experiment_without_dividing_by_zero():
    r = srm_check(0, 0)
    assert r["passed"] and r["n"] == 0


# --- the gate --------------------------------------------------------------

@pytest.fixture(scope="module")
def healthy():
    cfg = ProductConfig(n_users=40_000, days=7, seed=3)
    frame = simulate(cfg, Effect(conversion_lift=0.06), "t_srm", with_pre_period=False)
    return user_level(frame)


def test_a_healthy_experiment_is_not_flagged(healthy):
    r = analyse(healthy, experiment="t_srm")
    assert r.srm["passed"]
    assert r.decision != "INVALID"


def test_a_broken_redirect_invalidates_the_whole_readout(healthy):
    """Drop 6% of treatment users, which is what a variant-specific redirect
    failure looks like from the warehouse. The primary metric still 'wins' --
    that is exactly the danger -- and the readout must refuse it anyway."""
    rng = np.random.default_rng(5)
    is_t = (healthy["variant"] == "treatment").to_numpy()
    broken = healthy[~(is_t & (rng.random(len(healthy)) < 0.06))]

    r = analyse(broken, experiment="t_srm_broken")
    # The metric still looks like a win. That is the whole danger.
    prow = next(x for x in r.rows if x["is_primary"])
    assert prow["relative_effect"] > 0 and prow["significant"]
    assert not r.srm["passed"]
    assert r.decision == "INVALID"
    assert "SAMPLE RATIO MISMATCH" in r.reasons[0]


def test_the_invalid_decision_short_circuits_before_guardrails(healthy):
    """An INVALID readout must not also announce a guardrail verdict; that would
    invite someone to act on the part they liked."""
    rng = np.random.default_rng(5)
    is_t = (healthy["variant"] == "treatment").to_numpy()
    broken = healthy[~(is_t & (rng.random(len(healthy)) < 0.10))]
    r = analyse(broken, experiment="t_srm_broken")
    assert r.decision == "INVALID"
    assert len(r.reasons) == 1


# --- the page --------------------------------------------------------------

def test_page_is_self_contained():
    out = build(lift=0.03, latency_delta=4.0, users=6_000, days=5)
    page = out["html"]
    assert "http://" not in page and "https://" not in page
    assert "<script" not in page
    assert page.lstrip().startswith("<!doctype html>")


def test_page_shows_the_srm_failure_in_the_band_not_buried():
    out = build(lift=0.03, users=8_000, days=5, drop_treatment=0.08)
    page = out["html"]
    assert out["readout"].decision == "INVALID"
    assert "SRM FAIL" in page
    # The band has to come before the decision block, or a reader scrolling to
    # the big number never sees it.
    assert page.index("SRM FAIL") < page.index("class='decision")


def test_mde_is_always_rendered_even_when_significant():
    out = build(lift=0.10, users=20_000, days=7)
    assert "MDE at this n" in out["html"]


def test_ci_chart_uses_relative_units_so_the_axis_is_shareable():
    """Regression test for the first version of this panel.

    With an absolute axis, a metric measured in milliseconds swamps a metric
    measured in probability and the conversion interval renders as a dot. The
    guard is that conversion's drawn interval is a meaningful fraction of the
    plot even when a latency metric is on the same chart.
    """
    cfg = ProductConfig(n_users=12_000, days=5, seed=4)
    people = user_level(simulate(cfg, Effect(conversion_lift=0.05, latency_ms_delta=30.0),
                                 "t_ci", with_pre_period=False))
    rows = analyse(people, experiment="t_ci").rows
    svg = _ci_svg(rows)

    # Pull every horizontal segment (the interval bars) out of the SVG and check
    # none of them collapsed. Under an absolute shared axis the conversion bar is
    # sub-pixel next to a metric measured in milliseconds, which is the exact
    # regression this guards.
    import re

    widths = [float(m[2]) - float(m[0]) for m in
              re.findall(r"<line x1='([\d.]+)' y1='([\d.]+)' x2='([\d.]+)' y2='([\d.]+)'", svg)
              if m[1] == m[3]]
    assert len(widths) == len(rows), "expected one interval bar per metric"
    assert min(widths) > 4.0, "an interval collapsed to a sliver: %r" % widths
    assert svg.count("<circle") == len(rows)
    assert "0%" in svg


def test_page_renders_with_no_sequence_panel():
    """Short experiments produce too few daily points to plot. The page must
    degrade rather than raise."""
    cfg = ProductConfig(n_users=4_000, days=2, seed=6)
    people = user_level(simulate(cfg, Effect(), "t_short", with_pre_period=False))
    page = render(analyse(people, experiment="t_short"), sequence=[])
    assert "not enough days" in page or "always-valid" not in page
