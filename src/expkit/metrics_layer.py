"""The metrics layer: every metric defined ONCE, in code, with tests.

The spec's requirement is "no metric math hidden in notebooks", and the reason is
not tidiness. When conversion is computed one way in the experiment readout and
another way in the growth dashboard, the two disagree, someone notices six months
later, and every historical decision made with either number becomes suspect.

So a metric here is a **declaration**, not a function call site:

  * a name, an owner-facing description, and its type
  * the unit of analysis (always the randomisation unit for experiment metrics)
  * how to aggregate a user's rows into one value
  * the direction that counts as good
  * whether it is a guardrail, and the regression it tolerates
  * an optional validity check that runs against real data

`REGISTRY` is the single source of truth. The readout, the validation suite and
any dashboard all resolve metrics through it, so a definition change propagates
rather than diverging.

This is dbt's contract (define once, test, document) applied to metrics rather
than tables. dbt itself is the right tool once the metrics live in a warehouse;
here they are computed from user-level frames in-process, so a registry is the
honest equivalent rather than dragging in a warehouse to hold four definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MetricDefinition:
    name: str
    description: str
    kind: str                       # "proportion" | "mean" | "ratio"
    column: str
    unit: str = "user_id"           # the randomisation unit
    aggregation: str = "max"        # how to collapse a unit's rows: max|sum|mean
    direction: str = "increase"     # which way is good
    guardrail: bool = False
    tolerance_relative: float = 0.0
    # A cheap validity assertion run against real data by `validate_registry`.
    # A metric definition nobody checks is a metric definition that drifts.
    valid_range: tuple = (-np.inf, np.inf)

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse event rows to one value per unit."""
        if self.column not in df.columns:
            raise KeyError("metric %r needs column %r" % (self.name, self.column))
        agg = {"max": "max", "sum": "sum", "mean": "mean"}[self.aggregation]
        return df.groupby([self.unit, "variant"], as_index=False).agg(**{self.name: (self.column, agg)})

    def is_good(self, delta: float) -> bool:
        return delta > 0 if self.direction == "increase" else delta < 0


REGISTRY = {
    "conversion_rate": MetricDefinition(
        name="conversion_rate",
        description=("Fraction of exposed users who converted at least once during the "
                     "experiment window. NOT a per-session rate: randomisation is per user, "
                     "so the unit of analysis must be the user."),
        kind="proportion", column="converted", aggregation="max",
        direction="increase", valid_range=(0.0, 1.0),
    ),
    "revenue_per_user": MetricDefinition(
        name="revenue_per_user",
        description=("Total revenue per exposed user, including users who spent nothing. "
                     "Excluding zero-spend users would measure revenue per PURCHASER, which "
                     "moves for a completely different reason and is a different metric."),
        kind="mean", column="revenue", aggregation="sum",
        direction="increase", valid_range=(0.0, np.inf),
    ),
    "p50_latency_ms": MetricDefinition(
        name="p50_latency_ms",
        description=("Median request latency per user, averaged across users. A guardrail: "
                     "a feature that wins on conversion by making the product slower is "
                     "usually not a win."),
        kind="mean", column="latency_ms", aggregation="mean",
        direction="decrease", guardrail=True, tolerance_relative=0.02,
        valid_range=(0.0, np.inf),
    ),
    "sessions_per_user": MetricDefinition(
        name="sessions_per_user",
        description=("Sessions per exposed user. A guardrail against a change that lifts "
                     "conversion by driving users away, which shows up here first."),
        kind="mean", column="sessions", aggregation="sum",
        direction="increase", guardrail=True, tolerance_relative=0.05,
        valid_range=(0.0, np.inf),
    ),
}

NORTH_STAR = "conversion_rate"
GUARDRAILS = [name for name, m in REGISTRY.items() if m.guardrail]


def build_metric_frame(events: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Materialise every registered metric onto one user-level frame.

    One pass, one place. A caller that wants conversion and revenue gets both
    computed by the same definitions, which is what stops two dashboards
    disagreeing.
    """
    names = metrics or list(REGISTRY)
    frame = None
    for name in names:
        part = REGISTRY[name].aggregate(events)
        frame = part if frame is None else frame.merge(part, on=["user_id", "variant"], how="outer")
    return frame


def validate_registry(events: pd.DataFrame) -> dict:
    """Run every metric's validity check against real data.

    This is the metrics-layer equivalent of a dbt test: it fails when a
    definition and the data disagree, which is how a silently-broken metric gets
    caught before it reaches a decision.
    """
    frame = build_metric_frame(events)
    results, failures = {}, []
    for name, m in REGISTRY.items():
        values = frame[name].dropna()
        lo, hi = m.valid_range
        out_of_range = int(((values < lo) | (values > hi)).sum())
        results[name] = {
            "n": int(len(values)),
            "min": float(values.min()) if len(values) else float("nan"),
            "max": float(values.max()) if len(values) else float("nan"),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "valid_range": [lo, hi],
            "out_of_range": out_of_range,
            "ok": out_of_range == 0 and len(values) > 0,
        }
        if not results[name]["ok"]:
            failures.append(name)
    return {"metrics": results, "failures": failures, "passed": not failures}


def describe() -> str:
    """Human-readable catalogue. The 'docs' half of define-once-test-document."""
    lines = ["| metric | type | unit | direction | guardrail | description |",
             "|---|---|---|---|---|---|"]
    for name, m in REGISTRY.items():
        lines.append("| `%s` | %s | %s | %s | %s | %s |"
                     % (name, m.kind, m.unit, m.direction,
                        ("yes (tol %.0f%%)" % (100 * m.tolerance_relative)) if m.guardrail else "no",
                        m.description))
    return "\n".join(lines)
