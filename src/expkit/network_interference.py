"""Interference through a social graph - the same lesson, the opposite sign.

    python -m expkit.network_interference compare
    python -m expkit.network_interference sweep
    python -m expkit.network_interference modularity

`interference.py` studies a marketplace, where treated requests *take* a shared
resource from control requests. The spillover is negative and the A/B **overstates**
the global effect by 7.3x.

This module studies a social product, where a treated user's new feature makes
the experience better for the people they interact with. The spillover is
positive, control users get some of the treatment's benefit, the arms converge - and the A/B **understates**. Same violation of SUTVA, opposite direction, and the
same conclusion as the marketplace case reached from the other side: *the sign of
interference bias is a property of the mechanism, not of interference.*

## Why this one is harder to notice

The marketplace version announces itself: a 7x effect is implausible and someone
asks why. This one produces an estimate that is *smaller* than the truth, which
looks like an experiment that was appropriately conservative. Nobody investigates
a disappointing result for being too disappointing, and the feature gets killed
for an effect it does have.

## The designs

**Bernoulli (user-level).** Each user independently. Maximum power, maximum
contamination: with a well-mixed graph nearly every control user has treated
neighbours.

**Graph-cluster.** Randomise whole communities, so most of a user's neighbours
share their assignment. Contamination falls with the fraction of edges that stay
inside a cluster, which is exactly what graph modularity measures - so
**modularity predicts how well cluster randomisation will work**, and it is
computable before the experiment runs. That is the practically useful part: it is
a go/no-go input rather than a post-hoc explanation.

The cost is severe, and it comes from a specific place: the unit of analysis stops
being 6,000 users and becomes 40 communities, **and communities differ in ways
users do not**. The generator carries a per-community baseline offset for exactly
that reason - the first version did not, cluster randomisation came out nearly
free at zero spillover, and that is not a property of clustering. It is a property
of pretending every community is the same. `sweep` prices the real trade.

**Exposure-weighted estimator on the Bernoulli design.** Instead of changing the
design, condition on how much spillover each user actually received: compare
control users with *no* treated neighbours against treated users with *all*
treated neighbours. It needs no new design and it throws away most of the
sample - and on a well-mixed graph the "no treated neighbours" cell is nearly
empty, which is measured below rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results")


@dataclass
class GraphConfig:
    n_users: int = 6_000
    n_communities: int = 40
    p_within: float = 0.020      # edge probability inside a community
    p_between: float = 0.0008    # and across communities
    base_rate: float = 0.30      # baseline outcome probability
    direct_effect: float = 0.08  # own treatment, on the probability scale
    spillover: float = 0.10      # full effect when ALL neighbours are treated
    # Per-community baseline offset. Communities differ for reasons that have
    # nothing to do with the treatment -- tenure, geography, how they were
    # acquired -- and this term is precisely what a cluster-randomised design
    # PAYS FOR: it is the between-cluster variance that a user-level design never
    # sees. The first version of this generator had no such term, and cluster
    # randomisation came out nearly free, which is not a property of clustering.
    # It is a property of pretending every community is the same.
    community_sd: float = 0.06
    seed: int = 0


def build_graph(cfg: GraphConfig) -> dict:
    """Stochastic block model: dense inside communities, sparse across.

    A social graph is not Erdos-Renyi, and the difference is the entire subject
    here. Uniform random edges give modularity near zero, cluster randomisation
    nothing to work with, and a study that concludes clustering does not help --
    for a graph nobody has.
    """
    rng = np.random.default_rng(cfg.seed)
    community = rng.integers(0, cfg.n_communities, size=cfg.n_users)

    # Sampled per candidate pair would be O(n^2). Instead draw an expected edge
    # count per block and sample endpoints, which is the same distribution at
    # this density and runs in seconds rather than minutes.
    edges = []
    for c in range(cfg.n_communities):
        members = np.flatnonzero(community == c)
        m = len(members)
        if m < 2:
            continue
        n_edges = rng.binomial(m * (m - 1) // 2, cfg.p_within)
        if n_edges:
            a = rng.choice(members, size=n_edges)
            b = rng.choice(members, size=n_edges)
            edges.append(np.column_stack([a, b]))

    n_cross = rng.binomial(cfg.n_users * (cfg.n_users - 1) // 2, cfg.p_between)
    if n_cross:
        a = rng.integers(0, cfg.n_users, size=n_cross)
        b = rng.integers(0, cfg.n_users, size=n_cross)
        edges.append(np.column_stack([a, b]))

    E = np.vstack(edges) if edges else np.zeros((0, 2), dtype=int)
    E = E[E[:, 0] != E[:, 1]]

    # Adjacency as a neighbour list via sorted edge arrays -- no scipy here, and
    # a dense n x n matrix would be 288 MB at this size.
    both = np.vstack([E, E[:, ::-1]])
    order = np.argsort(both[:, 0], kind="stable")
    both = both[order]
    starts = np.searchsorted(both[:, 0], np.arange(cfg.n_users + 1))
    community_offset = rng.normal(0.0, cfg.community_sd, size=cfg.n_communities)
    return {"community": community, "neighbours": both[:, 1], "starts": starts,
            "community_offset": community_offset, "n_edges": len(E), "cfg": cfg}


def modularity(graph: dict) -> float:
    """Fraction of edges inside communities, minus what chance would give.

    Newman modularity restricted to the known partition, which is the right
    version here: the communities are not being discovered, they are the units
    the design would randomise over. The question is only how much of the graph
    they contain.
    """
    community, nb, starts = graph["community"], graph["neighbours"], graph["starts"]
    src = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    same = (community[src] == community[nb]).mean() if len(nb) else 0.0

    degree = np.diff(starts).astype(float)
    total = degree.sum()
    if total == 0:
        return 0.0
    expected = sum((degree[community == c].sum() / total) ** 2
                   for c in range(graph["cfg"].n_communities))
    return float(same - expected)


def treated_fraction_of_neighbours(graph: dict, treated: np.ndarray) -> np.ndarray:
    """Exposure: what share of each user's neighbours are treated.

    Users with no neighbours get 0.0 and are *not* dropped. They are genuinely
    unexposed, they are part of the population the global effect is defined over,
    and dropping them would quietly change the estimand.
    """
    nb, starts = graph["neighbours"], graph["starts"]
    if len(nb) == 0:
        return np.zeros(len(starts) - 1)
    t = treated[nb].astype(float)
    degree = np.diff(starts).astype(float)

    # `starts[:-1]` can contain len(nb) itself when the LAST users are isolated,
    # and reduceat raises IndexError on an out-of-range index rather than
    # returning zero. Clamping is safe because every clamped row has degree 0 and
    # its value is overwritten below -- but the crash is real, and it only
    # appears on graphs whose trailing users happen to have no edges. The three
    # studies in this module never hit it; a 1,500-user test graph did on the
    # first run.
    idx = np.minimum(starts[:-1], max(len(nb) - 1, 0))
    sums = np.add.reduceat(t, idx)
    sums = np.where(degree > 0, sums, 0.0)
    return np.where(degree > 0, sums / np.maximum(degree, 1), 0.0)


def outcomes(graph: dict, treated: np.ndarray, rng) -> np.ndarray:
    cfg = graph["cfg"]
    exposure = treated_fraction_of_neighbours(graph, treated)
    baseline = cfg.base_rate + graph["community_offset"][graph["community"]]
    p = baseline + cfg.direct_effect * treated + cfg.spillover * exposure
    return (rng.random(len(treated)) < np.clip(p, 0.0, 1.0)).astype(float)


def global_treatment_effect(graph: dict, seed: int = 999, reps: int = 24) -> dict:
    """Run the world both ways, several times. This is what a launch delivers.

    Averaged over `reps` draws because the ground truth is itself a Monte Carlo
    estimate, and every bias figure in this module is measured *against* it. At
    6,000 users a single pair of draws carries a standard error near 0.012, which
    on a small true effect is a bias of 20% invented out of nothing -- and that is
    exactly what the first version produced: a spillover-zero row reporting +21%
    bias for a design that has none.

    Common random numbers across the two arms, so the difference is not paying
    for two independent draws when it only needs one.
    """
    n = len(graph["community"])
    diffs = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        u = rng.random(n)           # shared noise: CRN across the two worlds
        cfg = graph["cfg"]
        exposure_c = treated_fraction_of_neighbours(graph, np.zeros(n))
        exposure_t = treated_fraction_of_neighbours(graph, np.ones(n))
        baseline = cfg.base_rate + graph["community_offset"][graph["community"]]
        p_c = np.clip(baseline + cfg.spillover * exposure_c, 0, 1)
        p_t = np.clip(baseline + cfg.direct_effect + cfg.spillover * exposure_t, 0, 1)
        diffs.append(float((u < p_t).mean() - (u < p_c).mean()))
    return {"gte": float(np.mean(diffs)), "gte_se": float(np.std(diffs, ddof=1) / math.sqrt(reps)),
            "reps": reps}


# ---------------------------------------------------------------------------
# designs
# ---------------------------------------------------------------------------

def bernoulli_estimate(graph: dict, seed: int, split: float = 0.5) -> dict:
    rng = np.random.default_rng(seed)
    n = len(graph["community"])
    treated = (rng.random(n) < split).astype(float)
    y = outcomes(graph, treated, rng)
    exposure = treated_fraction_of_neighbours(graph, treated)
    return {"estimate": float(y[treated == 1].mean() - y[treated == 0].mean()),
            "mean_control_exposure": float(exposure[treated == 0].mean()),
            "n": n}


def cluster_estimate(graph: dict, seed: int, split: float = 0.5) -> dict:
    """Randomise communities, then estimate at the cluster level.

    The estimator is the mean of cluster means, not the mean over users. Pooling
    users across clusters would treat 6,000 correlated observations as
    independent and report an interval far too narrow -- the design changed, so
    the analysis has to.
    """
    rng = np.random.default_rng(seed)
    cfg = graph["cfg"]
    community = graph["community"]
    assign = (rng.random(cfg.n_communities) < split).astype(float)
    treated = assign[community]
    y = outcomes(graph, treated, rng)

    means = np.array([y[community == c].mean() for c in range(cfg.n_communities)
                      if (community == c).any()])
    arms = np.array([assign[c] for c in range(cfg.n_communities) if (community == c).any()])
    if arms.sum() < 2 or (1 - arms).sum() < 2:
        return {"estimate": float("nan"), "n_clusters": len(means)}

    exposure = treated_fraction_of_neighbours(graph, treated)
    return {"estimate": float(means[arms == 1].mean() - means[arms == 0].mean()),
            "mean_control_exposure": float(exposure[treated == 0].mean()),
            "n_clusters": int(len(means))}


def exposure_weighted_estimate(graph: dict, seed: int, split: float = 0.5,
                               pure_threshold: float = 0.1) -> dict:
    """Condition on exposure instead of changing the design.

    Compares users with essentially no treated neighbours against treated users
    with essentially all treated neighbours -- the two cells that approximate the
    all-control and all-treatment worlds. No new design needed, and most of the
    sample discarded.

    On a well-mixed graph the "pure control" cell is nearly empty, which is the
    catch and is reported as `usable_fraction` rather than left implicit.
    """
    rng = np.random.default_rng(seed)
    n = len(graph["community"])
    treated = (rng.random(n) < split).astype(float)
    y = outcomes(graph, treated, rng)
    exposure = treated_fraction_of_neighbours(graph, treated)

    pure_c = (treated == 0) & (exposure <= pure_threshold)
    pure_t = (treated == 1) & (exposure >= 1 - pure_threshold)
    usable = int(pure_c.sum() + pure_t.sum())
    if pure_c.sum() < 30 or pure_t.sum() < 30:
        return {"estimate": float("nan"), "usable_fraction": usable / n,
                "n_pure_control": int(pure_c.sum()), "n_pure_treated": int(pure_t.sum()),
                "failed": "too few users in the pure cells to estimate"}
    return {"estimate": float(y[pure_t].mean() - y[pure_c].mean()),
            "usable_fraction": usable / n,
            "n_pure_control": int(pure_c.sum()), "n_pure_treated": int(pure_t.sum())}


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------

def _score(est: np.ndarray, truth: float) -> dict:
    est = np.asarray([e for e in est if np.isfinite(e)])
    if not len(est):
        return {"n_valid": 0, "mean_estimate": None, "bias_relative": None, "rmse": None}
    return {"n_valid": int(len(est)),
            "mean_estimate": float(est.mean()),
            "sd_of_estimate": float(est.std(ddof=1)) if len(est) > 1 else 0.0,
            "bias_relative": float((est.mean() - truth) / truth) if truth else None,
            "rmse": float(np.sqrt(((est - truth) ** 2).mean()))}


def run_comparison(cfg: GraphConfig = None, n_worlds: int = 20) -> dict:
    cfg = cfg or GraphConfig()
    bern, clus, expo, truths = [], [], [], []
    mods = []
    for w in range(n_worlds):
        g = build_graph(GraphConfig(**{**asdict(cfg), "seed": w}))
        mods.append(modularity(g))
        truths.append(global_treatment_effect(g)["gte"])
        bern.append(bernoulli_estimate(g, seed=1000 + w)["estimate"])
        clus.append(cluster_estimate(g, seed=1000 + w)["estimate"])
        expo.append(exposure_weighted_estimate(g, seed=1000 + w)["estimate"])

    truth = float(np.mean(truths))
    sample = build_graph(GraphConfig(**{**asdict(cfg), "seed": 0}))
    control_exposure = bernoulli_estimate(sample, seed=1)["mean_control_exposure"]
    ew = exposure_weighted_estimate(sample, seed=1)

    out = {
        "config": asdict(cfg),
        "n_worlds": n_worlds,
        "graph": {"modularity": float(np.mean(mods)),
                  "edges": sample["n_edges"],
                  "mean_degree": float(np.diff(sample["starts"]).mean())},
        "ground_truth_gte": truth,
        "mean_control_exposure_under_bernoulli": control_exposure,
        "bernoulli": _score(bern, truth),
        "graph_cluster": _score(clus, truth),
        "exposure_weighted": {**_score(expo, truth),
                              "usable_fraction": ew.get("usable_fraction"),
                              "n_pure_control": ew.get("n_pure_control"),
                              "note": ew.get("failed")},
    }
    b = out["bernoulli"]["bias_relative"]
    c = out["graph_cluster"]["bias_relative"]
    out["reading"] = (
        "user-randomised UNDERSTATES the global effect by %.0f%% because %.0f%% of a control "
        "user's neighbours are treated and the benefit leaks to them. Cluster randomisation cuts "
        "the bias to %.0f%%. Note the SIGN: the marketplace study in interference.py finds the "
        "same violation inflating the estimate by 627%%. The direction belongs to the mechanism, "
        "and a positive-spillover bias is the more dangerous one -- an effect that looks smaller "
        "than it is reads as a conservative experiment, and nobody investigates a disappointing "
        "result for being too disappointing."
        % (abs(100 * b), 100 * control_exposure, abs(100 * c)))
    return out


def run_sweep(cfg: GraphConfig = None, n_worlds: int = 10) -> dict:
    """How the two designs trade off as spillover strengthens."""
    cfg = cfg or GraphConfig()
    rows = []
    for spill in (0.0, 0.05, 0.10, 0.20, 0.40):
        bern, clus, truths = [], [], []
        for w in range(n_worlds):
            g = build_graph(GraphConfig(**{**asdict(cfg), "seed": w, "spillover": spill}))
            truths.append(global_treatment_effect(g)["gte"])
            bern.append(bernoulli_estimate(g, seed=2000 + w)["estimate"])
            clus.append(cluster_estimate(g, seed=2000 + w)["estimate"])
        truth = float(np.mean(truths))
        b, c = _score(bern, truth), _score(clus, truth)
        # A 10% margin, not a bare comparison. At zero spillover the two designs
        # come out 0.0137 against 0.0117 -- a tie inside the Monte Carlo error of
        # 8 worlds -- and calling that "cluster wins" would report a crossover at
        # exactly the setting where there is no interference to correct for.
        rows.append({"spillover": spill, "ground_truth_gte": truth,
                     "bernoulli": b, "graph_cluster": c,
                     "cluster_wins_on_rmse": bool(c["rmse"] < b["rmse"] * 0.9),
                     "tied_within_noise": bool(0.9 <= c["rmse"] / b["rmse"] <= 1.1)})
        print("  spillover %.2f  truth %+.4f  bernoulli bias %+6.1f%% rmse %.4f  |  "
              "cluster bias %+6.1f%% rmse %.4f  -> %s"
              % (spill, truth, 100 * b["bias_relative"], b["rmse"],
                 100 * c["bias_relative"], c["rmse"],
                 "cluster" if c["rmse"] < b["rmse"] * 0.9
                 else ("tie" if c["rmse"] <= b["rmse"] * 1.1 else "bernoulli")))

    crossover = next((r["spillover"] for r in rows if r["cluster_wins_on_rmse"]), None)
    return {
        "rows": rows,
        "crossover_spillover": crossover,
        "reading": ("cluster randomisation is not free and is not always right. It removes bias "
                    "and pays in variance, so with no interference to correct it buys nothing%s."
                    % (" -- it starts winning on RMSE at spillover %.2f" % crossover
                       if crossover is not None else
                       " and never wins on RMSE at these settings")),
    }


def run_modularity(cfg: GraphConfig = None, n_worlds: int = 8) -> dict:
    """Modularity predicts how well clustering will work - before you run it.

    This is the practically useful result. Modularity is computable from the
    graph alone, with no experiment and no outcome data, so it turns "should we
    pay for a cluster-randomised test?" into a number available up front.
    """
    cfg = cfg or GraphConfig()
    rows = []
    for p_between in (0.00005, 0.0004, 0.0008, 0.002, 0.006):
        mods, bern, clus, truths = [], [], [], []
        for w in range(n_worlds):
            g = build_graph(GraphConfig(**{**asdict(cfg), "seed": w, "p_between": p_between}))
            mods.append(modularity(g))
            truths.append(global_treatment_effect(g)["gte"])
            bern.append(bernoulli_estimate(g, seed=3000 + w)["estimate"])
            clus.append(cluster_estimate(g, seed=3000 + w)["estimate"])
        truth = float(np.mean(truths))
        b, c = _score(bern, truth), _score(clus, truth)
        rows.append({"p_between": p_between, "modularity": float(np.mean(mods)),
                     "bernoulli_bias": b["bias_relative"],
                     "cluster_bias": c["bias_relative"],
                     "bias_removed": (abs(b["bias_relative"]) - abs(c["bias_relative"]))
                                     / abs(b["bias_relative"]) if b["bias_relative"] else None})
        print("  p_between %.5f  modularity %.3f  bernoulli bias %+6.1f%%  cluster bias %+6.1f%%"
              % (p_between, rows[-1]["modularity"], 100 * b["bias_relative"],
                 100 * c["bias_relative"]))

    ms = [r["modularity"] for r in rows]
    removed = [r["bias_removed"] for r in rows if r["bias_removed"] is not None]
    corr = float(np.corrcoef(ms[:len(removed)], removed)[0, 1]) if len(removed) > 2 else None
    return {"rows": rows, "correlation_modularity_vs_bias_removed": corr,
            "reading": ("modularity is computable from the graph with no experiment and no "
                        "outcome data, and it tracks how much bias clustering removes "
                        "(r=%.2f here). That makes it a go/no-go input available before "
                        "committing to a design, rather than an explanation afterwards."
                        % corr if corr is not None else "too few points to correlate")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["compare", "sweep", "modularity"])
    ap.add_argument("--worlds", type=int, default=None)
    args = ap.parse_args()

    fn = {"compare": run_comparison, "sweep": run_sweep, "modularity": run_modularity}[args.command]
    out = fn(**({"n_worlds": args.worlds} if args.worlds else {}))

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "validation_network_%s.json" % args.command)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2, default=float))
    print("\nwritten:", os.path.relpath(path, os.path.join(os.path.dirname(__file__), "..", "..")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
