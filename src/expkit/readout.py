"""Experiment readout: the thing a PM actually reads.

Produces one decision per experiment, not a wall of p-values. The rules that make
it a decision rather than a number:

  * The primary metric is declared BEFORE the run and there is exactly one.
  * Guardrails can veto. A win on the primary metric with a tripped guardrail is
    **DO NOT SHIP**, not "ship it, we'll watch the guardrail".
  * Achieved power is on every readout. "Not significant" from an underpowered
    test is not evidence of no effect, and the readout has to say which one it is.
  * Secondary metrics get Benjamini-Hochberg correction. The primary does not --
    it is one pre-declared test and correcting it would throw away power for
    nothing.

    python -m expkit.readout --lift 0.08 --latency-delta 25
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np

from .generator import Effect, ProductConfig, simulate, user_level
from .stats import benjamini_hochberg, mde_for_sample_size, power_for_sample_size, proportion_test, welch_ttest


@dataclass
class MetricSpec:
    name: str
    column: str
    kind: str                      # "proportion" | "mean"
    direction: str = "increase"    # the direction that counts as good
    guardrail: bool = False
    # For guardrails: how much regression is tolerable before it vetoes. Stated
    # as a business input, because "any regression at all" is not a real policy.
    tolerance_relative: float = 0.0


DEFAULT_METRICS = [
    MetricSpec("conversion_rate", "converted", "proportion", "increase"),
    MetricSpec("revenue_per_user", "revenue", "mean", "increase"),
    MetricSpec("p50_latency_ms", "latency_ms", "mean", "decrease", guardrail=True, tolerance_relative=0.02),
    MetricSpec("sessions_per_user", "sessions", "mean", "increase", guardrail=True, tolerance_relative=0.05),
]


@dataclass
class Readout:
    experiment: str
    primary: str
    rows: list = field(default_factory=list)
    decision: str = ""
    reasons: list = field(default_factory=list)


def analyse(users, metrics=DEFAULT_METRICS, primary: str = "conversion_rate", alpha: float = 0.05,
            experiment: str = "exp_001") -> Readout:
    control = users[users["variant"] == "control"]
    treatment = users[users["variant"] == "treatment"]

    rows = []
    for m in metrics:
        c = control[m.column].to_numpy()
        t = treatment[m.column].to_numpy()
        r = proportion_test(c, t, alpha, m.name) if m.kind == "proportion" else welch_ttest(c, t, alpha, m.name)
        row = r.as_dict()
        row["is_guardrail"] = m.guardrail
        row["is_primary"] = m.name == primary
        row["direction"] = m.direction

        moved_well = (r.absolute_effect > 0) if m.direction == "increase" else (r.absolute_effect < 0)
        row["moved_in_good_direction"] = bool(moved_well)

        if m.guardrail:
            # A guardrail trips on a SIGNIFICANT regression beyond tolerance.
            # Not on any regression: with enough traffic every metric moves, and
            # a policy of "no metric may ever dip" blocks every ship.
            regressed = (not moved_well) and abs(r.relative_effect) > m.tolerance_relative
            row["tripped"] = bool(regressed and r.significant)
            row["tolerance_relative"] = m.tolerance_relative
        rows.append(row)

    # Power, reported for the primary metric using its own observed baseline.
    prow = next(r for r in rows if r["is_primary"])
    n_min = min(prow["control_n"], prow["treatment_n"])
    baseline = prow["control_mean"]
    prow["mde_at_this_n"] = mde_for_sample_size(baseline, n_min, alpha) if baseline > 0 else float("nan")
    prow["achieved_power_at_observed_effect"] = (
        power_for_sample_size(baseline, prow["relative_effect"], n_min, alpha) if baseline > 0 else float("nan")
    )

    # Multiple-comparison control across the non-primary metrics only.
    secondary = [r for r in rows if not r["is_primary"]]
    if secondary:
        rejected, _ = benjamini_hochberg([r["p_value"] for r in secondary], fdr=alpha)
        for r, keep in zip(secondary, rejected):
            r["significant_after_bh"] = bool(keep)

    return _decide(Readout(experiment=experiment, primary=primary, rows=rows), alpha)


def _decide(readout: Readout, alpha: float) -> Readout:
    prow = next(r for r in readout.rows if r["is_primary"])
    tripped = [r for r in readout.rows if r.get("tripped")]

    if tripped:
        readout.decision = "DO NOT SHIP"
        readout.reasons.append(
            "guardrail(s) tripped: %s" % ", ".join("%s %+.1f%%" % (r["metric"], 100 * r["relative_effect"]) for r in tripped)
        )
        if prow["significant"] and prow["moved_in_good_direction"]:
            readout.reasons.append(
                "primary metric won (%+.2f%%) but a guardrail regression vetoes it -- "
                "quantify the trade and escalate, do not ship on the primary alone"
                % (100 * prow["relative_effect"])
            )
        return readout

    if prow["significant"] and prow["moved_in_good_direction"]:
        readout.decision = "SHIP"
        readout.reasons.append("primary %s %+.2f%% (p=%.4f), CI [%+.4f, %+.4f], no guardrail tripped"
                               % (prow["metric"], 100 * prow["relative_effect"], prow["p_value"],
                                  prow["ci_low"], prow["ci_high"]))
        return readout

    if prow["significant"]:
        readout.decision = "DO NOT SHIP"
        readout.reasons.append("primary metric moved significantly in the WRONG direction (%+.2f%%)"
                               % (100 * prow["relative_effect"]))
        return readout

    readout.decision = "INCONCLUSIVE"
    readout.reasons.append(
        "primary not significant (p=%.3f). This test could only detect a %+.1f%% relative change "
        "at 80%% power, so it does not show the feature does nothing -- it shows any effect is "
        "probably smaller than that." % (prow["p_value"], 100 * prow["mde_at_this_n"])
    )
    return readout


def render(readout: Readout) -> str:
    lines = ["# Experiment readout: %s" % readout.experiment, "", "## Decision: %s" % readout.decision, ""]
    for r in readout.reasons:
        lines.append("* %s" % r)
    lines += ["", "| metric | role | control | treatment | rel. effect | 95% CI (abs) | p | significant |",
              "|---|---|---|---|---|---|---|---|"]
    for r in readout.rows:
        role = "PRIMARY" if r["is_primary"] else ("guardrail" if r["is_guardrail"] else "secondary")
        if r.get("tripped"):
            role += " TRIPPED"
        sig = "yes" if r["significant"] else "no"
        if not r["is_primary"] and "significant_after_bh" in r:
            sig += " (BH: %s)" % ("yes" if r["significant_after_bh"] else "no")
        lines.append("| %s | %s | %.4f | %.4f | %+.2f%% | [%+.4f, %+.4f] | %.4f | %s |"
                     % (r["metric"], role, r["control_mean"], r["treatment_mean"], 100 * r["relative_effect"],
                        r["ci_low"], r["ci_high"], r["p_value"], sig))
    prow = next(r for r in readout.rows if r["is_primary"])
    lines += ["", "**Power.** n=%d per arm; minimum detectable relative effect at 80%% power: %+.1f%%."
              % (min(prow["control_n"], prow["treatment_n"]), 100 * prow["mde_at_this_n"])]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", type=float, default=0.08, help="true relative conversion lift")
    ap.add_argument("--latency-delta", type=float, default=0.0, help="true guardrail regression in ms")
    ap.add_argument("--users", type=int, default=40_000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = ProductConfig(n_users=args.users, days=args.days, seed=7)
    effect = Effect(conversion_lift=args.lift, latency_ms_delta=args.latency_delta)
    users = user_level(simulate(cfg, effect, "exp_readout", with_pre_period=False))
    readout = analyse(users, experiment="exp_readout")

    if args.json:
        print(json.dumps({"decision": readout.decision, "reasons": readout.reasons, "rows": readout.rows},
                         indent=2, default=float))
    else:
        print(render(readout))
        print("\n_Ground truth for this simulation: conversion lift %+.1f%%, latency delta %+.0f ms._"
              % (100 * args.lift, args.latency_delta))


if __name__ == "__main__":
    main()
