# Mini Experimentation Platform

An A/B testing engine that **validates its own statistics against known truth**.
1,000 simulated A/A experiments confirm the false-positive rate is what the
platform claims; injected effects confirm the power calculator is honest; and a
peeking simulation quantifies exactly how much early stopping inflates error.

> **Status: ~40% built.** Assignment, the statistics engine, the validation suite
> and the guardrail-aware readout are done and **measured**. CUPED, sequential
> testing, and the dbt metrics layer are not — see [Roadmap](#roadmap).

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
| **CUPED variance reduction + measured sample-size saving** | not started |
| **Sequential testing (mSPRT / group-sequential)** | not started |
| **Metrics layer as code (dbt) with tests and docs** | not started |
| **Results dashboard (web UI)** | not started |
| **Written experiment review memo** | not started |
| **Switchback / interference-aware designs** | not started |

The generator already emits a pre-period (`with_pre_period=True`) and
`test_pre_period_has_no_treatment_effect` asserts the treatment does not leak
into it, so the foundation CUPED needs is in place and tested — but CUPED itself
is not implemented and **no variance-reduction number is claimed anywhere.**

## Honesty notes

* All data is **simulated**. That is the point — ground truth has to be an input
  for the validation to mean anything — but no number here describes real users.
* The validation numbers above are from the committed runs in `results/`, at the
  stated run counts. Re-running with different seeds will move them by roughly
  the CI widths shown.
* **Sequential testing is not implemented**, so the peeking demo shows the
  *problem* (17% vs 5%) without yet shipping the *fix*. The comparison "our
  sequential method vs naive peeking" is not in this repo and is not claimed.
