"""Bucketing: deterministic, salted, and independent across experiments.

Three properties that matter, each of which has burned a real experimentation
platform when it was missing:

1. **Deterministic.** The same user must get the same variant on every request,
   across processes and restarts, with no lookup table. Hashing gives this;
   `random.choice()` at exposure time does not.
2. **Salted per experiment.** Without a per-experiment salt, every experiment
   splits users on the same hash and the *same* users land in treatment every
   time. Effects then correlate across experiments and any user-level bias
   (a whale in treatment) is repeated forever instead of averaging out.
3. **Uniform.** Verified empirically in the test suite with a chi-square test,
   not assumed from "md5 is random".
"""
from __future__ import annotations

import hashlib

MAX_HASH = 2 ** 32


def bucket(unit_id: str, salt: str, n_buckets: int = 10_000) -> int:
    """Stable hash of (unit, salt) -> [0, n_buckets)."""
    digest = hashlib.md5(("%s:%s" % (salt, unit_id)).encode()).digest()
    return int.from_bytes(digest[:4], "big") % n_buckets


def assign(unit_id: str, experiment: str, variants=("control", "treatment"), weights=None,
           n_buckets: int = 10_000) -> str:
    """Assign a unit to a variant. `experiment` is the salt."""
    if weights is None:
        weights = [1.0 / len(variants)] * len(variants)
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1, got %r" % (weights,))

    b = bucket(unit_id, experiment, n_buckets) / n_buckets
    cum = 0.0
    for variant, w in zip(variants, weights):
        cum += w
        if b < cum:
            return variant
    return variants[-1]


def in_experiment(unit_id: str, experiment: str, exposure_rate: float = 1.0, n_buckets: int = 10_000) -> bool:
    """Traffic ramp, hashed on a DIFFERENT salt than the variant assignment.

    Using the same salt for "is this user in the experiment" and "which variant"
    makes the ramp correlated with the variant split: at 10% exposure you would
    get 10% of the traffic but not a random 10% of each arm. The `:exposure`
    suffix keeps the two decisions independent.
    """
    return bucket(unit_id, experiment + ":exposure", n_buckets) / n_buckets < exposure_rate
