# Mini Experimentation Platform

An A/B testing engine that **validates its own statistics against known truth**.
1,000 simulated A/A experiments confirm the false-positive rate is what the
platform claims; injected effects confirm the power calculator is honest; and a
peeking simulation quantifies exactly how much early stopping inflates error.

> **Status: ~100% of the spec built.** Assignment, the statistics engine, the
> validation suite, the guardrail-aware readout, **CUPED**, **sequential testing
> (mSPRT)**, a **metrics layer as code**, a **written experiment review**, an
> **SRM gate**, a **generated results page**, and **switchback designs for
> interference** are done and **measured against ground truth**. What remains is
> named in [Roadmap](#roadmap), and none of it is a statistics gap.

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

## The metrics layer: defined once, tested, documented

The spec's requirement is "no metric math hidden in notebooks", and the reason is
not tidiness. When conversion is computed one way in the experiment readout and
another way in a growth dashboard, the two disagree, someone notices six months
later, and every historical decision made with either becomes suspect.

`metrics_layer.REGISTRY` is the single source of truth. A metric is a
**declaration** — unit of analysis, aggregation, direction, guardrail tolerance,
and a validity range — not a call site:

| metric | type | direction | guardrail |
|---|---|---|---|
| `conversion_rate` | proportion | increase | no |
| `revenue_per_user` | mean | increase | no |
| `p50_latency_ms` | mean | decrease | yes (tol 2%) |
| `sessions_per_user` | mean | increase | yes (tol 5%) |

`validate_registry` is the metrics-layer equivalent of a dbt test: it runs each
metric's validity check against real data and fails when a definition and the
data disagree. A test corrupts revenue to negative values and asserts the
registry catches it.

Two definitions carry their reasoning because both are silently wrong the other
way:

* **conversion is a user-level `max`, not a row count.** Randomisation is per
  user, so the unit of analysis must be the user; a session-level count of a
  user-level test understates variance and produces confident nonsense.
* **revenue-per-user includes zero-spend users.** Excluding them silently
  changes the metric to revenue-per-*purchaser*, which moves for entirely
  different reasons.

**Why a registry and not dbt here:** dbt is the right tool once metrics live in a
warehouse. These are computed from user-level frames in-process, so a tested
registry is the honest equivalent rather than dragging in a warehouse to hold
four definitions.

## The experiment review memo

**[docs/EXPERIMENT_REVIEW.md](docs/EXPERIMENT_REVIEW.md)** is a worked review of
the readout above: a variant that wins conversion by +3.89% and trips the latency
guardrail at +27.6%.

It leads with the decision (**do not ship, fix the latency and re-run**), then
does the part that is usually skipped — **quantifies the trade rather than
hiding behind the guardrail**. Answering "is +3.89% conversion worth +30 ms?"
needs a latency-conversion elasticity *for this product*, which we do not have,
and the memo says so instead of borrowing a published constant. It proposes the
one experiment that would settle it, and ends with what the author would have
designed differently — including that the most interesting number in the test
(+11.5% revenue) is the one they are least entitled to use, because it was not
pre-registered.

## Interference: where the whole platform's core assumption fails

Everything above assumes **SUTVA** — one user's outcome depends only on that
user's own assignment. In a marketplace that is false, and the platform will
report a confident, significant, badly wrong number without any of its other
checks firing. A/A validation passes. SRM passes. The interval is tight. The
launch still under-delivers.

`src/expkit/interference.py` simulates a two-sided marketplace where couriers are
a shared finite resource, so the ground-truth global effect is computable: run the
whole world at 100% control, run it again at 100% treatment, subtract.

Over 30 worlds, true effect **+9.56pp** fulfillment:

| design | estimate | bias | RMSE |
|---|---|---|---|
| user-randomised A/B | +69.5pp | **+627%** | 0.599 |
| switchback, 4h buckets | +8.86pp | -7.3% | 0.0295 |
| switchback, 4h, **stratified by hour of day** | +9.56pp | **-0.008%** | 0.0040 |

Three things came out of building this that were not in the plan.

**The direction of the bias is a property of the matching policy, not of
interference.** Under priority matching the A/B overstates by 7.3x. Re-run with
proportional rationing — no arm jumps the queue — and the A/B reports *exactly
zero* against a real +9.6pp effect, because the treated arm's saved supply flows
back into the shared pool and lifts control by the same amount. "Interference
inflates your estimate" is the wrong lesson. `make interference-mechanism` runs
that falsification.

**Blocking beat sample size, and it reversed the recommendation.** The first sweep
said *longer* buckets were better, which makes no sense — fewer randomisation
units should be worse. The real driver was hour-of-day imbalance, not unit count.
Stratifying so each slot-of-day splits evenly between arms cut RMSE **6-14x at
every bucket length** and moved the answer from 24h buckets to 2-4h.

**Carryover was never the problem; the estimand was.** At high fill the estimate
ran 25% low and no amount of burn-in fixed it. The unweighted mean of per-bucket
rates answers *"what did the average hour look like"*; a launch delivers *"what
did the average request experience"*. Off-peak buckets are saturated in both arms
and carry no effect; peak buckets carry all of it and all the demand.
Demand-weighting holds bias inside +/-1.5% across every supply regime, where the
unweighted estimator ranges from **+11% to -33%**.

### When does this matter at all

Switchback is not free, so the useful question is when to pay for it. Sweeping how
binding the supply constraint is:

| baseline fill | true effect | A/B estimate | A/B error |
|---|---|---|---|
| 33% | +6.0pp | +72.9pp | **12.1x over** |
| 57% | +9.6pp | +69.5pp | 7.3x over |
| 79% | +9.6pp | +30.0pp | 3.1x over |
| 90% | +8.0pp | +11.4pp | 1.4x over |
| 99% | +1.2pp | +0.07pp | **misses it entirely** |
| 100% | 0 | 0 | no effect to find |

Displacement requires scarcity. Where supply is abundant, taking a courier from
control costs control nothing, and a user-randomised test is fine.

### Inference, and the price of being robust

Every interval here **over**-covers — 100% against a nominal 95%. That is a bug in
the other direction, which is why the coverage check reports interval *width*
against the estimator's actual spread rather than just the hit rate:

| interval | coverage | width vs calibrated |
|---|---|---|
| design-matched (within-slot) | 1.00 | 1.5x |
| iid t | 1.00 | 4.2x |
| day-level block bootstrap | 1.00 | **7.5x** |

The "robust" choice is the worst one. Resampling whole days discards exactly the
hour-of-day blocking the design paid for. The analysis has to match the design;
robustness is neither free nor automatically correct.

## SRM: the check that runs before the statistics

A sample ratio mismatch does not mean the effect is smaller than reported. It
means the arms are **not comparable populations**, so the difference of means is
not estimating a treatment effect at all. The usual causes sit upstream of the
statistics entirely — a redirect that fails more often for one variant, a bot
filter keying on something the treatment changed, an SDK dropping events under the
treatment's extra latency — and every one of them also moves the metric, in the
same direction as the "win".

So it is a **hard gate**, not a warning. The readout short-circuits to `INVALID`
and gives exactly one reason; it does not also announce the guardrail verdict,
because that invites someone to act on the half they liked.

```
## Decision: INVALID

* SAMPLE RATIO MISMATCH: expected 50.0% treatment, observed 48.469% (n=38776,
  chi2 p=4.33e-09 < 0.001). The arms are not comparable populations, so no metric
  below is interpretable. Fix the assignment or logging path and rerun; do not
  analyse around it.
```

The threshold is **0.001, not 0.05**, and the reason is multiplicity rather than
severity: this check runs on every experiment every day, so at 0.05 the alert
channel is noise inside a week and everyone learns to ignore it.

Validated against the **actual hash assignment** rather than binomial draws —
which matters, because assignment is deterministic in `user_id`, so for a fixed
population and a fixed experiment name the split is not random at all; only the
salt varies it. Over 1,500 salts at n=40,000: **0 fires** at the 0.001 threshold,
3.7% at 0.05, median p 0.497. Simulating binomial counts would have tested numpy
rather than the thing that can actually break.

Sensitivity, measured: a **1.9%** shortfall at n≈39k is caught; **1.5%** is not
(p=0.13). The check is a floor on how broken assignment can be before anyone
notices — not a proof that it is fine.

## The results page

`python -m expkit.dashboard` renders one self-contained HTML file — no CDN, no
external requests, opens from disk. Panels are ordered the way a reviewer should
read them, which is not the order they are computed in: **SRM first and alone**,
then the decision and its reasons in sentences, then the primary metric with its
interval, then guardrails with their tolerance drawn as a band, then secondaries,
then the always-valid p-value by day — the only panel it is safe to read early.

There is no aggregate health score and no significance traffic light anywhere on
the page. Both invite the reader to skip the confidence interval, which is the
number carrying the uncertainty.

Two things this got wrong first, both now pinned by tests:

* The interval chart plotted **absolute** effects on a shared axis. Conversion
  moves by 0.014 and latency by 8, so the axis was owned entirely by whichever
  metric had the largest unit and the conversion interval rendered as a dot.
  Percent change is what makes an axis shareable.
* MDE was rendered only when the result was *not* significant. It belongs on every
  readout: "not significant" from an underpowered test and "no effect" are
  different findings, and that number is what separates them.

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
make test            # 61 tests
make validate-aa     # 1000 A/A experiments (~3 min)
make validate-power  # 300 experiments with a known injected effect
make validate-peek   # 400 experiments, daily peeking vs fixed horizon
make validate-cuped  # 200 experiments, variance reduction vs r^2
make validate-seq    # 400 experiments, mSPRT under daily peeking
make readout         # a worked experiment readout with a tripped guardrail
make dashboard       # results/dashboard.html plus the SRM-failure variant
make interference    # switchback vs user-randomised, against ground truth
make interference-sweep       # bucket length x burn-in x stratification
make interference-mechanism   # the falsification: is the bias really displacement
make interference-regimes     # when interference matters at all
make interference-coverage    # do the intervals actually cover
```

`make test` runs 61 tests, of which 13 pin the interference findings and 12 pin
the SRM gate and the results page.

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
| Metrics layer as code, with validity tests and a catalogue | done |
| Results page, self-contained, SRM-gated | done |
| SRM as a hard gate, validated on the real assignment function | done |
| Written experiment review memo | done |
| Switchback vs user-randomised, scored against a computable global effect | done |
| Stratified switchback + the estimand fix (bias 627% -> 0.008%) | done |
| **Switchback on a real marketplace rather than a simulator** | not possible here |
| **Interference through a social graph rather than a shared resource** | not started |

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
* **The marketplace is a simulator, not a marketplace.** The bias figures are
  exact for *this* model of displacement, and the model was chosen because its
  ground truth is computable. What generalises is the method — compute the global
  effect, score designs against it — not the 7.3x.
* **Every interference interval over-covers**, by 1.5x to 7.5x. The designs are
  unbiased; the *inference* around them still leaves precision on the table, and
  this repo says so rather than reporting "coverage 1.00" as a pass.
* **mSPRT is conservative** (0.67% against a nominal 5%). That is correct
  behaviour for an always-valid test, but it means the power cost is real and
  this repo has **not** measured how much extra sample it needs to match
  fixed-horizon power. That measurement is a roadmap item, not a claim.
