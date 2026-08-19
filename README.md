# Mini Experimentation Platform

An A/B testing engine that **validates its own statistics against known truth**.
1,000 simulated A/A experiments confirm the false-positive rate is what the
platform claims; injected effects confirm the power calculator is honest; and a
peeking simulation quantifies exactly how much early stopping inflates error.

> **Status: ~80% built.** Assignment, the statistics engine, the validation
> suite, the guardrail-aware readout, **CUPED**, and **sequential testing
> (mSPRT)** are done and **measured against ground truth**. The dbt metrics
> layer, a results UI and a written experiment memo are not — see
> [Roadmap](#roadmap).

## Sequential testing: the fix for the peeking problem

The peeking demo above shows the disease. mSPRT is the cure, validated the same
way as everything else — against data whose truth is known.

**300 null experiments, checked every day for a week:**

| decision rule | false-positive rate |
|---|---|
| naive fixed-horizon test, peeked daily | **19.0%** |
| mSPRT always-valid p-value, peeked daily | **0.67%** (95% CI [0.18%, 2.40%]) |

**Peeking every day is allowed by construction.** The mSPRT likelihood ratio is
a martingale under the null, so by Ville's inequality the probability of *ever*
crossing the `1/alpha` boundary — at any sample size, across every look — is at
most alpha.

**What you give up.** The 0.67% is far below the nominal 5%, and that
conservatism is the cost: an always-valid test protects against every possible
stopping time, not one, so it needs a larger sample for the same power. The
honest framing is that fixed-horizon wins *if you can actually commit to the
horizon*, and sequential wins in practice if the alternative is peeking anyway.

`tau` (the prior scale on the effect) is declared **before** the run via
`suggested_tau`. Tuning it after seeing data voids the guarantee, so it is a
documented decision rather than a knob.

## CUPED: measured variance reduction

150 null experiments with a pre-period covariate:

| | value |
|---|---|
| pre/post correlation (r) | 0.203 |
| **measured variance reduction** | **4.27%** |
| predicted from r² | 4.13% |
| effective sample multiplier | 1.045x |
| estimate biased? | no — mean effect unchanged |

**The measurement matches theory to within 0.14 points**, which is the point of
running it: CUPED's reduction is exactly r², so a covariate correlated 0.20 buys
4%, not 20%. Teams routinely expect the latter. On this simulated product the
persistent per-user component is modest, so the win is real but small — an
honest 4% rather than a headline number.

Two implementation details that decide whether it works at all, both asserted by
tests:

* **theta is estimated on POOLED data**, not per arm. Per-arm estimation lets the
  treatment influence its own adjustment and reintroduces the bias CUPED exists
  to avoid.
* **Users with no pre-period fall back to unadjusted** rather than being dropped.
  Dropping new users would silently change the population the experiment is about.

`test_cuped_variance_reduction_equals_r_squared` pins the r² relationship, and
`test_cuped_is_a_noop_when_the_covariate_is_uncorrelated` pins the first failure
mode.

## The headline result: the platform is validated

**A/A validation — 1,000 experiments with a true effect of exactly zero:**

| metric | nominal α | false positives | empirical FPR | 95% CI | contains α? |
|---|---|---|---|---|---|
| conversion | 0.05 | 42 / 1000 | **4.2%** | [3.12%, 5.63%] | **yes** |
| revenue | 0.05 | 58 / 1000 | **5.8%** | [4.51%, 7.42%] | **yes** |

Hitting 5% is necessary but not sufficient, so the suite also checks that the
p-values are **uniform on [0,1]** under the null — a miscalibrated test can land
on 5% by luck while being wrong everywhere else. Kolmogorov–Smirnov against
uniform: **D = 0.0279, p = 0.41**, uniformity not rejected.

```bash
make validate-aa    # ~3 minutes, 1000 experiments
```

## The peeking demo

The same null data, analysed once at the pre-registered horizon versus checked
every day. This is the quantified answer to *"the PM wants to stop on day 3
because it's clearly winning."*

| analysis discipline | false-positive rate |
|---|---|
| fixed horizon, one look | **5.0%** (exactly nominal) |
| peeking daily for 7 days | **17.0%** (95% CI [13.6%, 21.0%]) |

**A 3.4x inflation**, and the median false stop happens on **day 2** — early,
when the estimate is noisiest and looks most exciting. Every additional look is
another chance to cross the threshold: the fixed-horizon test spends its whole
alpha budget once, daily checking spends it seven times.

## The power validation caught a real error — mine

Injecting a known +15% lift, the first run reported:

* predicted power: **99.9%**
* empirical power: **85.0%**

The calculator was not broken. The **question** was wrong. The generator applies
its lift to the *per-day* conversion probability, while the analysed metric is
"converted at least once during the window" — which **saturates**. A +15% daily
lift shows up as only **+9.0%** at the metric level, because users who would
convert anyway mostly still convert.

Comparing against the lift that actually exists in the analysed metric:

| | value |
|---|---|
| generator lift (per-day) | +15.0% |
| **observed lift in the metric** | **+9.0%** |
| predicted power from the per-day lift | 99.9% ← the wrong question |
| **predicted power from the metric-level lift** | **85.8%** |
| **empirical power** | **85.0%** (95% CI [80.5%, 88.6%]) |

Prediction now lands inside the empirical CI. This is the classic "the effect
size in the PRD is not the effect size in the metric" trap, and the validation
suite is what surfaced it — which is the entire argument for having one.

## Design decisions worth reading

**Assignment is hashed, salted per experiment, and verified uniform.** Without a
per-experiment salt every experiment splits on the same hash, so the *same* users
land in treatment forever and any user-level bias repeats instead of averaging
out. `test_different_experiments_assign_independently` asserts ~50% agreement
between two experiments' splits; a shared salt would give 100%. The traffic ramp
hashes on a **separate** salt (`:exposure`), or the ramp would correlate with the
variant split.

**Analysis happens at the randomisation unit.** Users are randomised, so users
are analysed. Running a test at session or event level when randomisation was
per-user understates variance by roughly the average sessions-per-user and
produces confidently wrong p-values. `user_level()` is where that is enforced.

**Welch's t-test, never Student's.** Equal variance is exactly the assumption
that fails when a treatment works — changing revenue usually changes its variance
too. Welch costs nothing when variances happen to match.

**Pooled SE for the p-value, unpooled SE for the interval.** The pooled estimate
is correct under the null being tested; the unpooled one is correct for
estimating the true difference. Using one for both produces intervals that
disagree with their own p-value near the boundary.

**Benjamini-Hochberg on secondary metrics only.** The pre-declared primary metric
is one test and correcting it throws away power for nothing. BH (false discovery
rate) rather than Bonferroni (family-wise) across the guardrail set, because
Bonferroni over 15 metrics makes nothing significant and blindness is the more
expensive error.

## The readout: guardrails can veto

`python -m expkit.readout --lift 0.10 --latency-delta 30` produces a decision, not
a wall of p-values:

```
## Decision: DO NOT SHIP

* guardrail(s) tripped: p50_latency_ms +27.6%
* primary metric won (+3.89%) but a guardrail regression vetoes it --
  quantify the trade and escalate, do not ship on the primary alone
```

A guardrail trips only on a **significant** regression **beyond a stated
tolerance** — "no metric may ever dip" is not a policy, it is a way to block
every ship. And "not significant" is never reported as "no effect": an
inconclusive readout states the MDE, because a test powered to detect +10% that
came back flat has shown the effect is probably under 10%, which is a completely
different claim.

## Run it

```bash
pip install -r requirements.txt
make test            # 20 fast unit tests
make validate-aa     # 1000 A/A experiments (~3 min)
make validate-power  # 300 experiments with a known injected effect
make validate-peek   # 400 experiments, daily peeking vs fixed horizon
make validate-cuped  # 200 experiments, variance reduction vs r^2
make validate-seq    # 400 experiments, mSPRT under daily peeking
make readout         # a worked experiment readout with a tripped guardrail
```

## Roadmap (the remaining ~60%)

| Milestone | Status |
|---|---|
| Hash-based salted assignment + exposure ramp | done |
| Ground-truth event generator | done |
| Fixed-horizon analysis (Welch, two-proportion) | done |
| Power / MDE calculator | done |
| **A/A validation: 1000 runs, FPR + p-value uniformity** | done |
| **Power validation against injected effects** | done |
| **Peeking demo with quantified FPR inflation** | done |
| Guardrail-aware readout with a ship decision | done |
| Multiple-comparison control (BH) | done |
| CUPED + measured variance reduction, validated unbiased | done |
| Sequential testing (mSPRT), validated under daily peeking | done |
| Always-valid confidence sequences | done |
| **Metrics layer as code (dbt) with tests and docs** | not started |
| **Results dashboard (web UI)** | not started |
| **Written experiment review memo** | not started |
| **Switchback / interference-aware designs** | not started |

`test_pre_period_has_no_treatment_effect` asserts the treatment does not leak
into the pre-period, which is the precondition CUPED depends on — a covariate
contaminated by the treatment biases the estimate silently.

## Honesty notes

* All data is **simulated**. That is the point — ground truth has to be an input
  for the validation to mean anything — but no number here describes real users.
* The validation numbers above are from the committed runs in `results/`, at the
  stated run counts. Re-running with different seeds will move them by roughly
  the CI widths shown.
* **The CUPED win here is small (4.3%)** because this simulated product has only
  a modest persistent per-user component (r = 0.20). That is an honest property
  of the generator, not a limitation of CUPED — the r² relationship is what
  generalises, and it is the number the tests pin.
* **mSPRT is conservative** (0.67% against a nominal 5%). That is correct
  behaviour for an always-valid test, but it means the power cost is real and
  this repo has **not** measured how much extra sample it needs to match
  fixed-horizon power. That measurement is a roadmap item, not a claim.
