"""Validate the platform against known truth. This is the centerpiece.

Three experiments on the experiment engine itself:

  1. **A/A**: N simulated experiments with a true effect of exactly zero. The
     empirical false-positive rate must land inside the binomial CI of the
     nominal alpha. If it does not, every p-value the platform emits is a lie.
  2. **Power**: inject a KNOWN non-zero effect and confirm the detection rate
     matches what the power calculation promised. A platform whose power
     calculator is optimistic ships underpowered tests forever.
  3. **Peeking**: the same null data, analysed daily instead of once. Quantifies
     how much the false-positive rate inflates -- the answer to "the PM wants to
     stop on day 3 because it's clearly winning".

    python -m expkit.validation aa --runs 1000
    python -m expkit.validation power --runs 300
    python -m expkit.validation peeking --runs 500
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import stats as sps

from .generator import Effect, ProductConfig, simulate, user_level
from .stats import mde_for_sample_size, power_for_sample_size, proportion_test, welch_ttest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")

# A small, fast product config: the validation runs need HUNDREDS of simulated
# experiments, so each one has to be cheap. Sample size per run is still large
# enough for the normal approximations in stats.py to hold.
FAST = ProductConfig(n_users=4_000, days=7, sessions_lambda=1.6)


def _one_experiment(seed: int, effect: Effect, cfg: ProductConfig = FAST, with_pre: bool = False):
    c = ProductConfig(**{**cfg.__dict__, "seed": seed})
    df = simulate(c, effect, experiment="aa_%d" % seed, with_pre_period=with_pre)
    users = user_level(df, "post")
    control = users[users["variant"] == "control"]
    treatment = users[users["variant"] == "treatment"]
    return control, treatment


def binomial_ci(k: int, n: int, conf: float = 0.95):
    """Wilson interval for the empirical FPR. Normal approx is bad near 0.05."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


def run_aa(runs: int = 1000, alpha: float = 0.05, seed0: int = 10_000) -> dict:
    """N A/A experiments. True effect is zero by construction."""
    null = Effect()
    conv_sig, rev_sig, p_values = 0, 0, []
    t0 = time.perf_counter()
    for i in range(runs):
        control, treatment = _one_experiment(seed0 + i, null)
        r_conv = proportion_test(control["converted"].to_numpy(), treatment["converted"].to_numpy(), alpha)
        r_rev = welch_ttest(control["revenue"].to_numpy(), treatment["revenue"].to_numpy(), alpha, "revenue")
        conv_sig += int(r_conv.significant)
        rev_sig += int(r_rev.significant)
        p_values.append(r_conv.p_value)
        if (i + 1) % 200 == 0:
            print("  %d/%d  conversion FPR so far: %.3f" % (i + 1, runs, conv_sig / (i + 1)))

    p_values = np.asarray(p_values)
    # Under the null, p-values must be UNIFORM on [0,1], not merely "5% below
    # 0.05". A KS test against uniform is the stronger check and catches a
    # miscalibrated test that happens to hit 5% by luck.
    ks_stat, ks_p = sps.kstest(p_values, "uniform")

    out = {
        "runs": runs,
        "nominal_alpha": alpha,
        "conversion": {
            "false_positives": conv_sig,
            "empirical_fpr": conv_sig / runs,
            "ci95": binomial_ci(conv_sig, runs),
            "within_ci": binomial_ci(conv_sig, runs)[0] <= alpha <= binomial_ci(conv_sig, runs)[1],
        },
        "revenue": {
            "false_positives": rev_sig,
            "empirical_fpr": rev_sig / runs,
            "ci95": binomial_ci(rev_sig, runs),
            "within_ci": binomial_ci(rev_sig, runs)[0] <= alpha <= binomial_ci(rev_sig, runs)[1],
        },
        "pvalue_uniformity_ks": {"statistic": float(ks_stat), "p_value": float(ks_p),
                                 "uniform_not_rejected": bool(ks_p > 0.01)},
        "wall_s": time.perf_counter() - t0,
    }
    return out


def run_power(runs: int = 300, true_lift: float = 0.15, alpha: float = 0.05, seed0: int = 50_000) -> dict:
    """Inject a known effect; confirm the detection rate matches the power calc."""
    effect = Effect(conversion_lift=true_lift)
    detected, n_per_arm, baselines, observed_lifts = 0, [], [], []
    for i in range(runs):
        control, treatment = _one_experiment(seed0 + i, effect)
        r = proportion_test(control["converted"].to_numpy(), treatment["converted"].to_numpy(), alpha)
        detected += int(r.significant and r.absolute_effect > 0)
        n_per_arm.append(min(r.control_n, r.treatment_n))
        # Observed CONTROL rate, not the per-day config value: `converted` is a
        # max over the experiment window, so the window-level baseline is much
        # higher than the per-day parameter. Feeding the config value into the
        # power calculator would compare the platform against the wrong promise.
        baselines.append(r.control_mean)
        observed_lifts.append(r.relative_effect)
        if (i + 1) % 100 == 0:
            print("  %d/%d  detection rate so far: %.3f" % (i + 1, runs, detected / (i + 1)))

    n_med = int(np.median(n_per_arm))
    baseline = float(np.mean(baselines))
    observed_lift = float(np.mean(observed_lifts))

    # TWO predictions, because the first run of this suite exposed a real trap.
    #
    # The generator's `conversion_lift` is applied to the PER-DAY conversion
    # probability. The analysed metric is "converted at least once during the
    # window", which saturates: lifting a daily 12% by +15% moves the window-level
    # rate by far less than 15%, because users who would convert anyway mostly
    # still convert. Feeding the per-day lift into the power calculator therefore
    # asks it about an effect that does not exist at the metric's own level, and
    # it duly predicts ~99.9% power against an empirical 85%.
    #
    # That gap was NOT a bug in the power calculator. It was a units error in the
    # question -- the classic "the effect size in the PRD is not the effect size
    # in the metric" mistake. The honest validation compares against the lift
    # actually present in the analysed metric.
    predicted_from_generator_lift = power_for_sample_size(baseline, true_lift, n_med, alpha)
    predicted = power_for_sample_size(baseline, observed_lift, n_med, alpha)
    empirical = detected / runs
    return {
        "runs": runs,
        "true_relative_lift": true_lift,
        "median_n_per_arm": n_med,
        "observed_relative_lift_in_metric": observed_lift,
        "predicted_power_from_metric_level_lift": predicted,
        "predicted_power_from_generator_lift": predicted_from_generator_lift,
        "empirical_power": empirical,
        "ci95": binomial_ci(detected, runs),
        "predicted_within_ci": binomial_ci(detected, runs)[0] <= predicted <= binomial_ci(detected, runs)[1],
        "mde_at_this_n": mde_for_sample_size(baseline, n_med, alpha),
        "observed_baseline_conversion": baseline,
        "note": ("The generator lift is PER-DAY; the metric is converted-at-least-once over the "
                 "window, which saturates. predicted_power_from_generator_lift asks the calculator "
                 "about an effect that does not exist at the metric level and overshoots badly. "
                 "predicted_power_from_metric_level_lift is the correct comparison, and it is the "
                 "one checked against the empirical CI."),
    }


def run_peeking(runs: int = 500, days: int = 7, alpha: float = 0.05, seed0: int = 90_000) -> dict:
    """The peeking demo: same null data, checked every day vs checked once.

    This is the quantified answer to "can we stop early, it's clearly winning?".
    """
    null = Effect()
    fixed_horizon_fp, peeking_fp = 0, 0
    stop_days = []
    for i in range(runs):
        c = ProductConfig(**{**FAST.__dict__, "seed": seed0 + i, "days": days})
        df = simulate(c, null, experiment="peek_%d" % (seed0 + i), with_pre_period=False)
        post = df[df["variant"] != "excluded"]

        stopped = False
        for day in range(1, days + 1):
            cumulative = post[post["day"] < day]
            users = cumulative.groupby(["user_id", "variant"], as_index=False).agg(converted=("converted", "max"))
            control = users[users["variant"] == "control"]["converted"].to_numpy()
            treatment = users[users["variant"] == "treatment"]["converted"].to_numpy()
            if len(control) < 30 or len(treatment) < 30:
                continue
            r = proportion_test(control, treatment, alpha)
            if r.significant and not stopped:
                # A peeking analyst stops HERE and declares a winner.
                peeking_fp += 1
                stop_days.append(day)
                stopped = True
            if day == days:
                # The disciplined analyst only looks at the pre-registered horizon.
                fixed_horizon_fp += int(r.significant)
        if (i + 1) % 100 == 0:
            print("  %d/%d  peeking FPR %.3f vs fixed %.3f"
                  % (i + 1, runs, peeking_fp / (i + 1), fixed_horizon_fp / (i + 1)))

    return {
        "runs": runs,
        "days_checked": days,
        "nominal_alpha": alpha,
        "fixed_horizon_fpr": fixed_horizon_fp / runs,
        "peeking_fpr": peeking_fp / runs,
        "inflation_factor": (peeking_fp / max(fixed_horizon_fp, 1)),
        "peeking_ci95": binomial_ci(peeking_fp, runs),
        "median_false_stop_day": float(np.median(stop_days)) if stop_days else float("nan"),
        "interpretation": ("Every additional look is another chance to cross the threshold. "
                           "The fixed-horizon test spends its whole alpha budget once; daily "
                           "checking spends it %d times." % days),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["aa", "power", "peeking", "cuped", "sequential", "all"])
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    out = {}
    if args.suite in ("aa", "all"):
        print("A/A validation...")
        out["aa"] = run_aa(args.runs or 1000, args.alpha)
    if args.suite in ("power", "all"):
        print("power validation...")
        out["power"] = run_power(args.runs or 300, alpha=args.alpha)
    if args.suite in ("peeking", "all"):
        print("peeking demo...")
        out["peeking"] = run_peeking(args.runs or 500, alpha=args.alpha)
    if args.suite in ("cuped", "all"):
        print("CUPED validation...")
        out["cuped"] = run_cuped(args.runs or 200, alpha=args.alpha)
    if args.suite in ("sequential", "all"):
        print("sequential (mSPRT) validation...")
        out["sequential"] = run_sequential(args.runs or 400, alpha=args.alpha)

    path = os.path.join(RESULTS, "validation_%s.json" % args.suite)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print("\nwrote", path)




# ---------------------------------------------------------------------------
# CUPED and sequential testing, validated the same way as everything else:
# against data whose ground truth is known.
# ---------------------------------------------------------------------------

def run_cuped(runs: int = 200, alpha: float = 0.05, seed0: int = 70_000) -> dict:
    """Measure CUPED's variance reduction and confirm it does not bias the estimate.

    Two things must both hold, and only one of them is what people check:
      * variance falls (the selling point)
      * the ESTIMATE stays unbiased (the thing that makes it safe)
    A CUPED implementation that estimates theta per-arm shrinks variance and
    silently biases the effect, which is why the second check is here.
    """
    from .cuped import cuped_frame, theoretical_reduction

    reductions, thetas, corrs = [], [], []
    unadj_effects, adj_effects = [], []
    unadj_sig, adj_sig = 0, 0

    for i in range(runs):
        cfg = ProductConfig(**{**FAST.__dict__, "seed": seed0 + i})
        df = simulate(cfg, Effect(), experiment="cuped_%d" % i, with_pre_period=True)
        post = user_level(df, "post")
        pre = user_level(df, "pre")

        merged, res = cuped_frame(post, pre, metric="revenue")
        reductions.append(res.variance_reduction_pct)
        thetas.append(res.theta)
        corrs.append(res.correlation)

        c = merged[merged["variant"] == "control"]
        t = merged[merged["variant"] == "treatment"]
        r_unadj = welch_ttest(c["revenue"].to_numpy(), t["revenue"].to_numpy(), alpha, "revenue")
        r_adj = welch_ttest(c["revenue_cuped"].to_numpy(), t["revenue_cuped"].to_numpy(), alpha, "revenue_cuped")
        unadj_effects.append(r_unadj.absolute_effect)
        adj_effects.append(r_adj.absolute_effect)
        unadj_sig += int(r_unadj.significant)
        adj_sig += int(r_adj.significant)
        if (i + 1) % 50 == 0:
            print("  %d/%d  mean variance reduction so far: %.1f%%"
                  % (i + 1, runs, float(np.mean(reductions))))

    mean_corr = float(np.mean(corrs))
    mean_reduction = float(np.mean(reductions))
    return {
        "runs": runs,
        "true_effect": 0.0,
        "mean_theta": float(np.mean(thetas)),
        "mean_correlation_pre_post": mean_corr,
        "mean_variance_reduction_pct": mean_reduction,
        "theoretical_reduction_pct_from_r2": theoretical_reduction(mean_corr),
        "effective_sample_multiplier": 1.0 / (1.0 - mean_reduction / 100.0) if mean_reduction < 100 else float("inf"),
        "mean_effect_unadjusted": float(np.mean(unadj_effects)),
        "mean_effect_cuped": float(np.mean(adj_effects)),
        "unbiased": abs(float(np.mean(adj_effects))) < 3 * float(np.std(adj_effects)) / np.sqrt(runs) + 1e-9,
        "fpr_unadjusted": unadj_sig / runs,
        "fpr_cuped": adj_sig / runs,
        "fpr_cuped_ci95": binomial_ci(adj_sig, runs),
        "note": ("variance reduction should track r^2 closely; both arms are null so the FPR "
                 "must stay near alpha AFTER adjustment -- a CUPED that reduces variance but "
                 "inflates the FPR is biased"),
    }


def run_sequential(runs: int = 400, days: int = 7, alpha: float = 0.05, seed0: int = 80_000) -> dict:
    """The payoff for the peeking demo: peek daily with mSPRT and stay at alpha.

    Same schedule as `run_peeking` -- a look every day -- but the decision rule is
    the always-valid p-value instead of a fixed-horizon one.
    """
    from .sequential import SequentialState, suggested_tau, update

    null = Effect()
    seq_fp, naive_fp = 0, 0
    stop_days = []
    tau = suggested_tau(0.5, 0.05)   # declared before the run, not tuned after

    for i in range(runs):
        c = ProductConfig(**{**FAST.__dict__, "seed": seed0 + i, "days": days})
        df = simulate(c, null, experiment="seq_%d" % (seed0 + i), with_pre_period=False)
        post = df[df["variant"] != "excluded"]

        state = SequentialState(tau=tau, alpha=alpha)
        naive_hit = False
        for day in range(1, days + 1):
            cum = post[post["day"] < day]
            users = cum.groupby(["user_id", "variant"], as_index=False).agg(converted=("converted", "max"))
            control = users[users["variant"] == "control"]["converted"].to_numpy()
            treatment = users[users["variant"] == "treatment"]["converted"].to_numpy()
            if len(control) < 30 or len(treatment) < 30:
                continue
            r = proportion_test(control, treatment, alpha)
            se = (r.ci_high - r.ci_low) / (2 * 1.959963985)
            update(state, r.absolute_effect, se, len(control) + len(treatment))
            if r.significant and not naive_hit:
                naive_hit = True

        seq_fp += int(state.crossed)
        naive_fp += int(naive_hit)
        if state.crossed:
            stop_days.append(state.crossed_at_n)
        if (i + 1) % 100 == 0:
            print("  %d/%d  sequential FPR %.3f vs naive-peeking %.3f"
                  % (i + 1, runs, seq_fp / (i + 1), naive_fp / (i + 1)))

    return {
        "runs": runs,
        "days_checked": days,
        "nominal_alpha": alpha,
        "tau": tau,
        "naive_peeking_fpr": naive_fp / runs,
        "sequential_fpr": seq_fp / runs,
        "sequential_ci95": binomial_ci(seq_fp, runs),
        "controls_error": binomial_ci(seq_fp, runs)[1] <= alpha * 1.5,
        "interpretation": ("the mSPRT likelihood ratio is a martingale under the null, so by "
                           "Ville's inequality the probability of EVER crossing 1/alpha is at "
                           "most alpha -- peeking every day is allowed by construction"),
    }

if __name__ == "__main__":
    main()
