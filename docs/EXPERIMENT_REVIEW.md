# Experiment review: EXP-2026-041 — simplified checkout

**Decision: DO NOT SHIP as-is. Re-run with the latency regression fixed.**

**Reviewers:** Analytics, Growth PM, Checkout eng
**Date:** 2026-08-19
**Data:** simulated via `expkit.generator` with a known ground truth; every number
below is reproducible with `make readout`.

---

## Recommendation in one paragraph

The variant lifted conversion by **+3.89%** (95% CI +1.65pp to +3.72pp, p < 0.001)
— a real win, comfortably powered, and the effect is where we expected it. But
p50 latency rose **+27.6%**, tripping the latency guardrail. The conversion win
does not license shipping a 27.6% latency regression to 100% of users, and the
guardrail exists precisely so that decision is not made by whoever is most
excited. **Fix the latency regression and re-run.** The conversion mechanism
appears sound; there is no reason to abandon the feature.

## Results

| metric | role | control | treatment | rel. effect | 95% CI (abs) | p |
|---|---|---|---|---|---|---|
| conversion_rate | **primary** | 0.6909 | 0.7177 | **+3.89%** | [+0.0165, +0.0372] | < 0.001 |
| revenue_per_user | secondary | 55.40 | 61.78 | +11.52% | [+3.64, +9.12] | < 0.001 |
| p50_latency_ms | **guardrail** | 107.97 | 137.82 | **+27.64%** | [+29.58, +30.12] | < 0.001 |
| sessions_per_user | guardrail | 23.955 | 23.952 | −0.01% | [−0.114, +0.108] | 0.96 |

**Power.** n = 14,944 per arm; MDE at 80% power was **+2.2% relative**. The
observed +3.89% is comfortably above it, so this is not a fragile result scraping
past significance.

## Why this is a "do not ship" and not a judgement call

The guardrail policy is written down *before* the test: a guardrail trips on a
**significant** regression **beyond a stated tolerance** (2% for latency). This
regression is 27.6% — an order of magnitude past tolerance, with a CI nowhere
near zero. There is no reading of the data in which this is noise.

Note what the policy deliberately does *not* say: it does not trip on *any*
regression. With 30,000 users every metric moves, and a rule of "no metric may
ever dip" blocks every ship. The tolerance is what makes the guardrail a decision
rule rather than a veto.

## Quantifying the trade, because "guardrail tripped" is not the whole answer

The PM's question is fair: *is +3.89% conversion worth +30 ms?* Answering it
requires a number we do not have — the conversion elasticity of latency **on this
product**. Published figures exist but are from other products with other users,
and importing them would be borrowing someone else's constant to justify our
decision.

What we can say:

* the latency regression is **uniform**, not concentrated in a slow tail
  (the CI is tight at [+29.6, +30.1] ms), so it affects every user
* sessions-per-user did **not** move, so within this window latency has not yet
  suppressed engagement — but a 7-day window is short for a habit effect
* the revenue win (+11.5%) is larger than the conversion win, which suggests the
  variant also shifts basket composition. **That is not what this test was
  designed to measure**, and treating it as a finding would be reading a
  secondary metric as though it had been pre-registered.

**The measurement that would settle it** is a latency-only A/B: inject +30 ms with
no other change and measure conversion. That isolates the elasticity, costs one
experiment slot, and converts this argument into arithmetic. Recommended as the
follow-up if the regression turns out to be expensive to fix.

## Caveats a reader should weigh

1. **Seven days is one week.** No day-of-week effect can be separated from a
   trend, and no novelty decay is observable. A two-week re-run is cheap.
2. **The revenue result is secondary** and survives BH correction, but it was not
   the pre-declared metric. It is a hypothesis for the next test, not a result of
   this one.
3. **Latency was measured server-side.** Client-perceived latency includes
   network and render, and could be better or worse than +27.6%.
4. **Simulated data.** This memo is a worked example of the review format on data
   with a known ground truth. It is not a claim about a real product.

## Decisions and owners

| # | action | owner |
|---|---|---|
| 1 | Do not ship the variant at its current latency | Growth PM |
| 2 | Profile the checkout path; identify the +30 ms | Checkout eng |
| 3 | Re-run for **14 days** once latency is within 2% | Analytics |
| 4 | If the regression is expensive to fix, run the latency-only elasticity test | Analytics |
| 5 | Do not cite the +11.5% revenue figure externally — secondary, not pre-registered | All |

## What I would have done differently designing this test

* **Pre-register the revenue metric or drop it.** It produced the most
  interesting number in the test and the one we are least entitled to use.
* **Instrument client-side latency**, not just server-side. The guardrail is a
  proxy for user experience and we measured the half that was easy.
* **Run 14 days from the start.** The 7-day window saved nothing — the test was
  well-powered on day 7 — and it costs us the day-of-week and novelty arguments
  that will now be raised about any re-run.
