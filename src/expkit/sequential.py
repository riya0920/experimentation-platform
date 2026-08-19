"""Sequential testing: valid inference while you peek.

The fixed-horizon peeking demo in `validation.py` shows the problem -- checking a
5% test daily for a week inflates the false-positive rate to 17%. This module is
the fix.

**mSPRT (mixture Sequential Probability Ratio Test).** Put a normal prior
N(0, tau^2) on the true effect and track the likelihood ratio of "some effect"
against "no effect". The ratio is a martingale under the null, so by Ville's
inequality `P(sup_n Lambda_n >= 1/alpha) <= alpha` -- the probability of EVER
crossing the threshold, at any sample size, is bounded by alpha. That is what
"always valid" means: you may look as often as you like.

In terms of the observed effect and its standard error:

    Lambda = sqrt(se^2 / (se^2 + tau^2)) * exp( tau^2 * d^2 / (2 * se^2 * (se^2 + tau^2)) )

and the always-valid p-value is the running minimum of `1 / Lambda`.

**What you give up versus fixed-horizon.** Nothing is free: an always-valid test
needs a larger sample to reach the same power, because it is protecting against
every possible stopping time rather than one. The cost depends on tau -- roughly
20-50% more samples at the effect sizes this platform simulates. The right way
to state the trade is: fixed-horizon is more efficient *if you can actually
commit to the horizon*, and sequential is more efficient in practice if the
alternative is peeking anyway.

**Choosing tau.** It is the prior standard deviation of the effect you expect.
Too small and the test is slow to detect large effects; too large and it is slow
for small ones. Setting it to the MDE you designed for is a defensible default,
and it is a *decision that must be made before the test starts* -- tuning tau
after seeing data destroys the guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SequentialState:
    """Running state of one sequentially-monitored experiment."""
    tau: float
    alpha: float = 0.05
    always_valid_p: float = 1.0
    max_lambda: float = 0.0
    crossed: bool = False
    crossed_at_n: int | None = None
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("history", None)
        return d


def msprt_lambda(effect: float, se: float, tau: float) -> float:
    """Likelihood ratio for a normal-mixture alternative against H0: effect = 0."""
    if se <= 0 or not math.isfinite(se):
        return 0.0
    v = se * se
    t2 = tau * tau
    shrink = math.sqrt(v / (v + t2))
    expo = t2 * effect * effect / (2.0 * v * (v + t2))
    # exp can overflow long before the test does anything interesting; the
    # threshold is 1/alpha, so anything past ~700 in the exponent is "crossed".
    if expo > 700:
        return float("inf")
    return shrink * math.exp(expo)


def update(state: SequentialState, effect: float, se: float, n: int) -> SequentialState:
    """Fold one observation of (effect, se) into the sequential state."""
    lam = msprt_lambda(effect, se, state.tau)
    state.max_lambda = max(state.max_lambda, lam)
    p = 1.0 if lam <= 0 else min(1.0, 1.0 / lam)
    # The always-valid p-value is the RUNNING MINIMUM. Without this it could
    # wander back up and a stopping rule based on it would not be valid.
    state.always_valid_p = min(state.always_valid_p, p)
    if not state.crossed and state.always_valid_p <= state.alpha:
        state.crossed = True
        state.crossed_at_n = n
    state.history.append({"n": n, "effect": effect, "se": se, "lambda": lam,
                          "always_valid_p": state.always_valid_p})
    return state


def confidence_sequence(effect: float, se: float, tau: float, alpha: float = 0.05):
    """An always-valid confidence interval.

    Unlike a fixed-horizon CI, this one is simultaneously valid at every sample
    size -- the probability that it EVER excludes the true effect is at most
    alpha. It is correspondingly wider, and the width is the price of being
    allowed to look.
    """
    if se <= 0:
        return (float("-nan"), float("nan"))
    v = se * se
    t2 = tau * tau
    # Invert the mSPRT boundary: the set of nulls not rejected at level alpha.
    radius = math.sqrt(2.0 * v * (v + t2) / t2 * math.log((1.0 / alpha) * math.sqrt((v + t2) / v)))
    return (effect - radius, effect + radius)


def analyse_stream(effects, ses, ns, tau: float, alpha: float = 0.05) -> SequentialState:
    """Convenience: run a whole peeking schedule through the test."""
    state = SequentialState(tau=tau, alpha=alpha)
    for effect, se, n in zip(effects, ses, ns):
        update(state, effect, se, n)
    return state


def suggested_tau(baseline: float, mde_relative: float) -> float:
    """tau defaulted to the absolute MDE the test was designed to detect.

    Declared before the run. Tuning tau after seeing data voids the guarantee,
    so this exists to make the default a documented choice rather than a knob
    someone reaches for mid-experiment.
    """
    return abs(baseline * mde_relative)
