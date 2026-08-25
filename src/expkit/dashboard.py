"""The results page, generated as one self-contained HTML file.

    python -m expkit.dashboard --lift 0.04 --latency-delta 8

A real platform serves this from a web app over a warehouse. There is no
warehouse here, so this renders the same page from the readout objects into a
single static file - no CDN, no external requests, works from disk.

## What is on it, and what is deliberately not

The panels are ordered the way a reviewer should read them, which is *not* the
order they are computed in:

  1. **The SRM check, first and alone.** If the arms are not comparable
     populations, nothing below it means anything, so it gets its own band across
     the top rather than a row in a table.
  2. **The decision**, with its reasons in plain sentences.
  3. **The primary metric**, exactly one, with its confidence interval drawn to
     scale against zero. A CI you have to read as digits is a CI nobody reads.
  4. **Guardrails**, with the tolerance band drawn in, because "did it regress"
     and "did it regress by more than we agreed to accept" are different
     questions and only the second one is a decision.
  5. **Secondary metrics**, greyed, with BH-adjusted significance. Greyed on
     purpose: they are hypothesis-generating, and the visual weight should say so.
  6. **The always-valid p-value over time**, which is the only panel on the page
     it is safe to look at early.

There is no "significance" traffic light on the whole page and no aggregate
health score. Both invite the reader to skip the CI, which is the number that
actually carries the uncertainty.
"""
from __future__ import annotations

import argparse
import html
import json
import os

import numpy as np

from .generator import Effect, ProductConfig, simulate, user_level
from .readout import DEFAULT_METRICS, analyse
from .sequential import SequentialState, suggested_tau, update
from .stats import proportion_test

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")

GOOD, BAD, NEUTRAL, MUTED = "#2f8f5b", "#c0392b", "#4c78a8", "#8a8f98"


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def _ci_svg(rows, width=560, row_h=42, pad_l=170, pad_r=44):
    """Confidence intervals on one shared axis, in RELATIVE units.

    The first version of this panel plotted absolute effects. That was wrong:
    conversion moves by 0.014 and latency by 8, so a shared absolute axis is
    entirely dominated by whichever metric happens to be measured in the largest
    unit, and the conversion interval collapses to a dot. Percent change is
    unit-free, which is what makes the axis shareable at all. Bounds are divided
    by the control mean -- a delta-method approximation that treats the control
    mean as fixed, fine at these sample sizes and noted because it is an
    approximation.

    Guardrails get their tolerance drawn as a faint band, because "did it
    regress" and "did it regress past what we agreed to accept" are different
    questions and only the second one is a decision.
    """
    if not rows:
        return "<p class='empty'>no metrics</p>"

    def rel(r, key):
        base = r["control_mean"]
        return r[key] / base if base else 0.0

    lo = min(min(rel(r, "ci_low"), 0.0) for r in rows)
    hi = max(max(rel(r, "ci_high"), 0.0) for r in rows)
    span = (hi - lo) or 1.0
    lo -= 0.10 * span
    hi += 0.10 * span
    span = hi - lo
    plot_w = width - pad_l - pad_r
    x = lambda v: pad_l + (v - lo) / span * plot_w
    height = row_h * len(rows) + 34

    parts = ["<svg viewBox='0 0 %d %d' width='100%%' role='img'>" % (width, height)]
    zx = x(0.0)
    parts.append("<line x1='%.1f' y1='14' x2='%.1f' y2='%d' stroke='#bbb' stroke-dasharray='3 3'/>"
                 % (zx, zx, height - 20))
    parts.append("<text x='%.1f' y='%d' font-size='10' fill='%s' text-anchor='middle'>0%%</text>"
                 % (zx, height - 6, MUTED))

    for i, r in enumerate(rows):
        y = 30 + i * row_h
        colour = BAD if r.get("tripped") else (GOOD if r["significant"] and r["moved_in_good_direction"]
                                               else (BAD if r["significant"] else MUTED))
        if r["is_guardrail"] and r.get("tolerance_relative"):
            tol = r["tolerance_relative"]
            parts.append("<rect x='%.1f' y='%.1f' width='%.1f' height='20' fill='#f0f0f0'/>"
                         % (x(-tol), y - 10, max(x(tol) - x(-tol), 1.0)))
        parts.append("<text x='%d' y='%.1f' font-size='12' fill='#333' text-anchor='end'>%s</text>"
                     % (pad_l - 10, y + 4, html.escape(r["metric"])))
        parts.append("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' stroke-width='3'/>"
                     % (x(rel(r, "ci_low")), y, x(rel(r, "ci_high")), y, colour))
        for v in (rel(r, "ci_low"), rel(r, "ci_high")):
            parts.append("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='%s' stroke-width='2'/>"
                         % (x(v), y - 5, x(v), y + 5, colour))
        parts.append("<circle cx='%.1f' cy='%.1f' r='4' fill='%s'/>"
                     % (x(r["relative_effect"]), y, colour))
        parts.append("<text x='%.1f' y='%.1f' font-size='10' fill='%s'>%+.2f%%</text>"
                     % (min(x(rel(r, "ci_high")) + 6, width - pad_r + 4), y + 3, MUTED,
                        100 * r["relative_effect"]))
    parts.append("</svg>")
    return "".join(parts)


def _sequence_svg(seq, alpha=0.05, width=560, height=170, pad=34):
    """Always-valid p-value over calendar time, on a log axis.

    The fixed-horizon p-value is deliberately absent. Putting both on one chart
    is how people end up reading the wrong one.
    """
    ps = [max(float(p), 1e-6) for p in seq]
    if len(ps) < 2:
        return "<p class='empty'>not enough days</p>"
    ys = [np.log10(p) for p in ps]
    lo, hi = min(ys + [np.log10(alpha)]) - 0.3, 0.05
    span = (hi - lo) or 1.0
    xs = lambda i: pad + i / (len(ps) - 1) * (width - 2 * pad)
    yv = lambda v: height - pad - (v - lo) / span * (height - 2 * pad)

    pts = " ".join("%.1f,%.1f" % (xs(i), yv(y)) for i, y in enumerate(ys))
    ay = yv(np.log10(alpha))
    parts = ["<svg viewBox='0 0 %d %d' width='100%%' role='img'>" % (width, height),
             "<line x1='%d' y1='%.1f' x2='%d' y2='%.1f' stroke='%s' stroke-dasharray='4 3'/>"
             % (pad, ay, width - pad, ay, BAD),
             "<text x='%d' y='%.1f' font-size='10' fill='%s'>alpha=%.2f</text>"
             % (width - pad - 52, ay - 4, BAD, alpha),
             "<polyline points='%s' fill='none' stroke='%s' stroke-width='2'/>" % (pts, NEUTRAL)]
    for i, y in enumerate(ys):
        parts.append("<circle cx='%.1f' cy='%.1f' r='2.5' fill='%s'/>" % (xs(i), yv(y), NEUTRAL))
    parts.append("<text x='%d' y='%d' font-size='10' fill='%s'>day 1</text>" % (pad, height - 10, MUTED))
    parts.append("<text x='%d' y='%d' font-size='10' fill='%s' text-anchor='end'>day %d</text>"
                 % (width - pad, height - 10, MUTED, len(ps)))
    parts.append("<text x='6' y='%d' font-size='10' fill='%s'>always-valid p (log)</text>" % (pad - 12, MUTED))
    parts.append("</svg>")
    return "".join(parts)


def _guardrail_rows(rows) -> str:
    out = []
    for r in rows:
        if not r["is_guardrail"]:
            continue
        tol = r.get("tolerance_relative", 0.0)
        state = ("TRIPPED" if r.get("tripped")
                 else ("within tolerance" if not r["moved_in_good_direction"] else "improved"))
        cls = "bad" if r.get("tripped") else "ok"
        out.append(
            "<tr class='%s'><td>%s</td><td>%+.2f%%</td><td>+/-%.1f%%</td>"
            "<td>%.4f</td><td>%s</td></tr>"
            % (cls, html.escape(r["metric"]), 100 * r["relative_effect"], 100 * tol,
               r["p_value"], state))
    return "".join(out) or "<tr><td colspan='5' class='empty'>none declared</td></tr>"


def _secondary_rows(rows) -> str:
    out = []
    for r in rows:
        if r["is_primary"] or r["is_guardrail"]:
            continue
        bh = r.get("significant_after_bh")
        out.append(
            "<tr><td>%s</td><td>%+.2f%%</td><td>[%+.4f, %+.4f]</td><td>%.4f</td><td>%s</td></tr>"
            % (html.escape(r["metric"]), 100 * r["relative_effect"], r["ci_low"], r["ci_high"],
               r["p_value"], "yes" if bh else "no"))
    return "".join(out) or "<tr><td colspan='5' class='empty'>none</td></tr>"


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:26px;color:#222;background:#fafafa}
h1{font-size:19px;margin:0 0 2px}h2{font-size:14px;margin:22px 0 8px;color:#444;
text-transform:uppercase;letter-spacing:.06em}
.sub{color:#8a8f98;font-size:12px;margin-bottom:18px}
.band{padding:10px 14px;border-radius:6px;font-size:13px;margin-bottom:16px}
.band.pass{background:#eef7f1;border:1px solid #cfe6d9;color:#2f6b4a}
.band.fail{background:#fdecea;border:1px solid #f5c6c2;color:#922}
.decision{font-size:26px;font-weight:600;letter-spacing:-.01em}
.d-ship{color:#2f8f5b}.d-no{color:#c0392b}.d-inc{color:#8a6d1f}.d-invalid{color:#7a2a8a}
ul.reasons{margin:8px 0 0;padding-left:18px;color:#444;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
.card{background:#fff;border:1px solid #e6e6e6;border-radius:8px;padding:14px 16px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #efefef}
th{color:#8a8f98;font-weight:500}
tr.bad td{background:#fdf2f1}
.kv{display:flex;gap:26px;flex-wrap:wrap;margin-top:6px}
.kv div{font-size:12px;color:#8a8f98}.kv b{display:block;font-size:17px;color:#222;font-weight:600}
.empty{color:#aaa;font-style:italic}
.note{font-size:12px;color:#8a8f98;margin-top:8px}
footer{margin-top:26px;font-size:11.5px;color:#aaa}
"""


def render(readout, sequence=None, ground_truth: dict = None) -> str:
    rows = readout.rows
    prow = next(r for r in rows if r["is_primary"])
    srm = readout.srm or {}
    dclass = {"SHIP": "d-ship", "DO NOT SHIP": "d-no",
              "INCONCLUSIVE": "d-inc", "INVALID": "d-invalid"}.get(readout.decision, "")

    srm_band = ("<div class='band %s'><b>SRM %s</b> - observed %.3f%% treatment of n=%s, "
                "chi-square p=%.3g (threshold %.3f). %s</div>"
                % ("pass" if srm.get("passed", True) else "fail",
                   "pass" if srm.get("passed", True) else "FAIL",
                   100 * srm.get("observed_ratio", float("nan")), "{:,}".format(srm.get("n", 0)),
                   srm.get("p_value", 1.0), srm.get("alpha", 0.001),
                   "Arms are comparable; the readout below is interpretable."
                   if srm.get("passed", True) else
                   "The arms are not comparable populations. Nothing below is interpretable."))

    gt = ""
    if ground_truth:
        gt = ("<div class='card'><h2 style='margin-top:0'>Ground truth</h2>"
              "<p class='note'>This is a simulation, so the true effect is an input rather than "
              "something to be estimated. Everything above was computed without access to it.</p>"
              "<div class='kv'><div>true conversion lift<b>%+.1f%%</b></div>"
              "<div>true latency delta<b>%+.0f ms</b></div>"
              "<div>estimated conversion lift<b>%+.2f%%</b></div></div></div>"
              % (100 * ground_truth.get("conversion_lift", 0.0),
                 ground_truth.get("latency_ms_delta", 0.0), 100 * prow["relative_effect"]))

    seq_panel = ""
    if sequence is not None:
        seq_panel = ("<div class='card'><h2 style='margin-top:0'>Always-valid p-value by day</h2>%s"
                     "<p class='note'>The only panel here that is safe to read early. A "
                     "fixed-horizon p-value checked daily inflates the false-positive rate to "
                     "roughly 25%%; this one is valid at every stopping time, and pays for it "
                     "with a later crossing.</p></div>" % _sequence_svg(sequence))

    return """<!doctype html><meta charset='utf-8'>
<title>Experiment readout - %(exp)s</title><style>%(css)s</style>
<h1>Experiment readout - %(exp)s</h1>
<div class='sub'>primary metric: <b>%(primary)s</b> &middot; n=%(n)s per arm &middot; alpha 0.05</div>
%(srm)s
<div class='card'>
  <div class='decision %(dclass)s'>%(decision)s</div>
  <ul class='reasons'>%(reasons)s</ul>
</div>
<div class='grid' style='margin-top:16px'>
  <div class='card'>
    <h2 style='margin-top:0'>Effect sizes, shared relative axis</h2>
    %(ci)s
    <p class='note'>Point estimate and 95%% interval as <b>percent change</b>, all metrics on one
    axis against zero. Green = significant and in the declared good direction, red = significant
    and against it, grey = not significant. The grey band on a guardrail row is its agreed
    tolerance.</p>
  </div>
  <div class='card'>
    <h2 style='margin-top:0'>Primary metric</h2>
    <div class='kv'>
      <div>control<b>%(cmean).4f</b></div>
      <div>treatment<b>%(tmean).4f</b></div>
      <div>relative<b>%(rel)+.2f%%</b></div>
      <div>p-value<b>%(p).4f</b></div>
    </div>
    <div class='kv' style='margin-top:12px'>
      <div>95%% CI (abs)<b>[%(cil)+.4f, %(cih)+.4f]</b></div>
      <div>MDE at this n<b>%(mde)+.1f%%</b></div>
      <div>power at observed effect<b>%(power).2f</b></div>
    </div>
    <p class='note'>MDE is on the page whether or not the result is significant. "Not
    significant" from an underpowered test and "no effect" are different findings, and only
    this number separates them.</p>
  </div>
  <div class='card'>
    <h2 style='margin-top:0'>Guardrails</h2>
    <table><tr><th>metric</th><th>rel. effect</th><th>tolerance</th><th>p</th><th>state</th></tr>%(guard)s</table>
    <p class='note'>A guardrail trips on a <i>significant</i> regression beyond an agreed
    tolerance. "No metric may ever dip" is not a policy - with enough traffic every metric
    moves, and that rule blocks every ship.</p>
  </div>
  <div class='card'>
    <h2 style='margin-top:0'>Secondary metrics</h2>
    <table><tr><th>metric</th><th>rel. effect</th><th>95%% CI</th><th>p</th><th>sig. after BH</th></tr>%(sec)s</table>
    <p class='note'>Benjamini-Hochberg across the secondaries only. The primary is one
    pre-declared test; correcting it too would throw away power for nothing.</p>
  </div>
  %(seq)s
  %(gt)s
</div>
<footer>Generated by <code>python -m expkit.dashboard</code>. Self-contained: no external
requests, no CDN, opens from disk.</footer>
""" % {
        "exp": html.escape(readout.experiment), "css": CSS,
        "primary": html.escape(readout.primary),
        "n": "{:,}".format(int(min(prow["control_n"], prow["treatment_n"]))),
        "srm": srm_band, "dclass": dclass, "decision": html.escape(readout.decision),
        "reasons": "".join("<li>%s</li>" % html.escape(r) for r in readout.reasons),
        "ci": _ci_svg(rows),
        "cmean": prow["control_mean"], "tmean": prow["treatment_mean"],
        "rel": 100 * prow["relative_effect"], "p": prow["p_value"],
        "cil": prow["ci_low"], "cih": prow["ci_high"],
        "mde": 100 * prow["mde_at_this_n"],
        "power": prow["achieved_power_at_observed_effect"],
        "guard": _guardrail_rows(rows), "sec": _secondary_rows(rows),
        "seq": seq_panel, "gt": gt,
    }


def _daily_always_valid_p(frame, days: int, alpha: float = 0.05):
    """The always-valid p-value after each day, for the time-series panel.

    Recomputed cumulatively rather than incrementally, because that is what a
    daily batch job would actually do and it makes the panel reproducible from
    the warehouse rather than from in-memory state.
    """
    post = frame[frame["variant"] != "excluded"]
    tau = suggested_tau(0.5, 0.05)      # declared up front; tuning it later voids the guarantee
    state = SequentialState(tau=tau, alpha=alpha)
    out = []
    for day in range(1, days + 1):
        cum = post[post["day"] < day]
        users = cum.groupby(["user_id", "variant"], as_index=False).agg(converted=("converted", "max"))
        c = users[users["variant"] == "control"]["converted"].to_numpy()
        t = users[users["variant"] == "treatment"]["converted"].to_numpy()
        if len(c) < 30 or len(t) < 30:
            continue
        r = proportion_test(c, t, alpha)
        se = (r.ci_high - r.ci_low) / (2 * 1.959963985)
        update(state, r.absolute_effect, se, len(c) + len(t))
        out.append(state.always_valid_p)
    return out


def build(lift: float = 0.04, latency_delta: float = 0.0, users: int = 40_000,
          days: int = 14, seed: int = 7, drop_treatment: float = 0.0) -> dict:
    """Simulate one experiment and render its page.

    `drop_treatment` deliberately loses a fraction of treatment users, which is
    what a broken redirect looks like from the warehouse. It is the demo for the
    SRM band: the page must refuse to be read, not quietly show a big win.
    """
    cfg = ProductConfig(n_users=users, days=days, seed=seed)
    effect = Effect(conversion_lift=lift, latency_ms_delta=latency_delta)
    frame = simulate(cfg, effect, "exp_dash", with_pre_period=False)
    people = user_level(frame)

    if drop_treatment > 0:
        rng = np.random.default_rng(seed + 1)
        is_t = (people["variant"] == "treatment").to_numpy()
        keep = ~(is_t & (rng.random(len(people)) < drop_treatment))
        people = people[keep]

    readout = analyse(people, experiment="exp_dash")

    seq = _daily_always_valid_p(frame, days=days)

    return {"readout": readout,
            "html": render(readout, sequence=seq,
                           ground_truth={"conversion_lift": lift, "latency_ms_delta": latency_delta})}


def main() -> int:
    ap = argparse.ArgumentParser(description="render the experiment results page")
    ap.add_argument("--lift", type=float, default=0.04)
    ap.add_argument("--latency-delta", type=float, default=0.0)
    ap.add_argument("--users", type=int, default=40_000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--drop-treatment", type=float, default=0.0,
                    help="simulate a broken redirect losing this fraction of treatment users")
    ap.add_argument("--out", default=os.path.join(RESULTS, "dashboard.html"))
    args = ap.parse_args()

    built = build(lift=args.lift, latency_delta=args.latency_delta, users=args.users,
                  days=args.days, drop_treatment=args.drop_treatment)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(built["html"])
    r = built["readout"]
    print("decision: %s" % r.decision)
    print("SRM: %s (p=%.3g)" % ("pass" if r.srm["passed"] else "FAIL", r.srm["p_value"]))
    print("written:", os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
