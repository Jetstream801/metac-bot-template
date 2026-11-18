"""Score every forecasting configuration the bot logged, lever by lever. Read-only, no model calls.

forecast_log/<post>.json keeps, for each question, the published forecast, every ensemble
member's own forecast, and one "control" forecast made the old way (first research only, usual
model, never published). So on every question we can ask what each lever changed:

  second research pass   usual_1 (both research passes)  vs  control (first pass only)
  ensemble               aggregate of the usual-model members  vs  usual_1 alone
  stronger model         strong_1  vs  the usual-model members (average score)
  strong in the ensemble aggregate of all members  vs  aggregate of the usual-model members
  everything together    published  vs  control
  aggregation method     the setting in quality.py  vs  the alternative (re-computed here, free)
  cap / floor            published  vs  the same aggregate without the cap

Two yardsticks:
  now      the community prediction once Metaculus reveals it (at close): how far each
           configuration sits from it. A proxy only: it says "different", not "better".
  later    the resolution. Scores are in peer-score points: the Metaculus peer score of forecast
           A minus that of forecast B on the same question is exactly 100 x (ln pA - ln pB)
           (half that for numeric questions), whoever else forecast. So the differences printed
           here are the tournament points a lever gained or lost per question.

Usage:
  python evaluate.py                    # from the log alone (no outcomes: the data section only)
  python evaluate.py --fetch            # read outcomes from Metaculus (needs METACULUS_TOKEN; read-only)
  python evaluate.py --fetch --summary quality.md
"""
import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import quality

MIN_RESOLVED = 20        # fewer paired resolutions than this never gives "keep" or "drop"
FAR_LOGODDS = 1.0        # a binary forecast this far (log-odds) from the community is listed
NUMERIC_KINDS = ("NumericQuestion", "DiscreteQuestion", "DateQuestion")


# ---- Reading the log ----------------------------------------------------------------------------

def load_log(directory):
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("status") == "forecast":
            records.append(record)
    return records


def kind_of(record):
    t = record.get("type") or ""
    if t == "BinaryQuestion":
        return "binary"
    if t == "MultipleChoiceQuestion":
        return "choice"
    return "numeric" if t in NUMERIC_KINDS else None


def canonical(kind, prediction, cdf=None):
    """A logged forecast as plain data: float, {option: p}, or the CDF as a list. None if unusable."""
    try:
        if kind == "binary":
            return float(prediction) if isinstance(prediction, (int, float)) else None
        if kind == "choice":
            options = prediction["predicted_options"]
            return {o["option_name"]: float(o["probability"]) for o in options}
        if kind == "numeric":
            if cdf:
                return [float(x) for x in cdf]
            declared = prediction["declared_percentiles"]
            if len(declared) == prediction.get("cdf_size"):     # an aggregate: its points are the CDF
                return [float(p["percentile"]) for p in declared]
    except (KeyError, TypeError, ValueError):
        pass
    return None


def aggregate(kind, forecasts, method=None, capped=True):
    if not forecasts:
        return None
    if kind == "binary":
        p = quality.aggregate_binary(forecasts, method)
        return quality.cap(p) if capped else p
    if kind == "choice":
        return quality.aggregate_options(forecasts, None if capped else 0.0)
    return forecasts[0] if len(forecasts) == 1 else quality.aggregate_cdfs(forecasts, method)


def configurations(record):
    """{name: forecast} for one question: what was published, each member, the control, and the
    aggregates that can be re-computed from the members."""
    kind = kind_of(record)
    if kind is None:
        return {}
    out = {"published": canonical(kind, record.get("prediction"), record.get("prediction_cdf"))}
    members = {}
    for entry in record.get("rationales") or []:
        name = entry.get("member")
        if not name or entry.get("question_id") not in (None, record.get("question_id")):
            continue
        forecast = canonical(kind, entry.get("prediction"), entry.get("cdf"))
        if forecast is None:
            continue
        if entry.get("role") == "control":
            out["control"] = forecast
        else:
            members[name] = forecast
    out.update(members)
    usual = [f for n, f in members.items() if n.startswith("usual") or n.endswith("_fallback")]
    strong = [f for n, f in members.items() if n.startswith("strong") and not n.endswith("_fallback")]
    everyone = list(members.values())
    if len(everyone) > 1:
        alt = {"binary": "median", "numeric": "median"}.get(kind)
        own = {"binary": "trimmed_logodds", "numeric": "mixture"}.get(kind)
        out["agg_all"] = aggregate(kind, everyone, own)
        if alt:
            out["agg_all_alt"] = aggregate(kind, everyone, alt)
        if kind != "numeric":
            out["agg_all_uncapped"] = aggregate(kind, everyone, own, capped=False)
    if len(usual) > 1:
        out["agg_usual"] = aggregate(kind, usual, {"binary": "trimmed_logodds", "numeric": "mixture"}.get(kind))
    elif len(usual) == 1:
        out["agg_usual"] = aggregate(kind, usual)
    out["_usual_names"] = [n for n in members if n.startswith("usual") or n.endswith("_fallback")]
    out["_strong_names"] = [n for n in members if n.startswith("strong") and not n.endswith("_fallback")]
    return {k: v for k, v in out.items() if v is not None}


# ---- Outcomes from Metaculus (read-only) ----------------------------------------------------------

def _location(value, scaling):
    """Where a value sits between the question's bounds: 0 at the lower, 1 at the upper."""
    low, high, zero = scaling.get("range_min"), scaling.get("range_max"), scaling.get("zero_point")
    if low is None or high is None or high == low:
        return None
    if zero is not None:
        try:
            return math.log((value - zero) / (low - zero)) / math.log((high - zero) / (low - zero))
        except (ValueError, ZeroDivisionError):
            return None
    return (value - low) / (high - low)


def outcome_from_api(post):
    """{'resolution', 'community'} from one /api/posts/<id>/ reply, in the same plain forms as
    canonical(). Either may be None (not resolved, annulled, community prediction still hidden).
    Numeric resolution is a location in [0, 1], or 'below' / 'above' the bounds."""
    q = (post or {}).get("question") or {}
    qtype = q.get("type")
    kind = {"binary": "binary", "multiple_choice": "choice", "numeric": "numeric", "discrete": "numeric",
            "date": "numeric"}.get(qtype)
    out = {"kind": kind, "resolution": None, "community": None}
    if kind is None:
        return out
    raw = q.get("resolution")
    scaling = q.get("scaling") or {}
    if isinstance(raw, str) and raw.strip().lower() not in ("", "annulled", "ambiguous"):
        text = raw.strip()
        if kind == "binary" and text.lower() in ("yes", "no"):
            out["resolution"] = 1 if text.lower() == "yes" else 0
        elif kind == "choice":
            out["resolution"] = text
        elif kind == "numeric":
            if text == "below_lower_bound":
                out["resolution"] = "below"
            elif text == "above_upper_bound":
                out["resolution"] = "above"
            else:
                try:
                    value = float(text)
                except ValueError:
                    try:
                        value = datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        value = None
                loc = _location(value, scaling) if value is not None else None
                if loc is not None:
                    out["resolution"] = "below" if loc < 0 else "above" if loc > 1 else loc
    latest = None
    for name in ("recency_weighted", "unweighted", "metaculus_prediction"):
        latest = ((q.get("aggregations") or {}).get(name) or {}).get("latest")
        if latest:
            break
    values = (latest or {}).get("forecast_values")
    centers = (latest or {}).get("centers")
    try:
        if kind == "binary":
            if values and len(values) == 2:
                out["community"] = float(values[1])
            elif centers:
                out["community"] = float(centers[0])
        elif kind == "choice" and values and q.get("options") and len(values) == len(q["options"]):
            out["community"] = dict(zip(q["options"], [float(v) for v in values]))
        elif kind == "numeric" and values and len(values) > 2:
            out["community"] = [float(v) for v in values]
    except (TypeError, ValueError):
        pass
    return out


def fetch_outcomes(post_ids, token=None, pause=0.3, get=None):
    """{post_id (str): outcome} for each post, straight from Metaculus. Skips what fails."""
    import requests
    get = get or requests.get
    headers = {"Authorization": f"Token {token}"} if token else {}
    outcomes = {}
    for post_id in post_ids:
        try:
            reply = get(f"https://www.metaculus.com/api/posts/{post_id}/", headers=headers, timeout=30)
            reply.raise_for_status()
            data = reply.json()
            outcomes[str(post_id)] = outcome_from_api(data)
            if len(outcomes) == 1:       # once a run: the reply's shape, so a change at Metaculus is seen at once
                q = data.get("question") or {}
                print(f"[evaluate] post {post_id}: type {q.get('type')!r}, resolution {q.get('resolution')!r}, aggregations "
                      f"{ {k: sorted((v or {}).get('latest') or {})[:8] for k, v in (q.get('aggregations') or {}).items()} }")
        except Exception as exc:
            print(f"[evaluate] could not read post {post_id}: {type(exc).__name__}: {str(exc)[:120]}")
        time.sleep(pause)
    return outcomes


# ---- Scores -------------------------------------------------------------------------------------

def probability_of_outcome(kind, forecast, resolution):
    """The probability a forecast put on what happened (numeric: on the bucket it fell in)."""
    if kind == "binary":
        return forecast if resolution == 1 else 1 - forecast
    if kind == "choice":
        return forecast.get(resolution)
    n = len(forecast) - 1
    if resolution == "below":
        return forecast[0]
    if resolution == "above":
        return 1 - forecast[-1]
    i = min(n - 1, max(0, int(float(resolution) * n)))
    return forecast[i + 1] - forecast[i]


def points(kind, forecast, resolution):
    """Log score in peer-score points (differences between two forecasts are real peer points)."""
    p = probability_of_outcome(kind, forecast, resolution)
    if p is None:
        return None
    p = max(float(p), 1e-6)
    if kind == "numeric":
        return 50 * math.log(p * (len(forecast) - 1))
    return 100 * math.log(p)


def brier(forecast, resolution):
    return (forecast - resolution) ** 2


def community_gap(kind, forecast, community):
    """Distance from the community prediction. binary: log-odds. choice: share of probability
    placed differently. numeric: distance between medians, as a share of the question's range."""
    if kind == "binary":
        return abs(quality.logit(forecast) - quality.logit(community))
    if kind == "choice":
        return 0.5 * sum(abs(forecast.get(k, 0.0) - v) for k, v in community.items())
    return abs(_quantile(forecast, 0.5) - _quantile(community, 0.5))


def _quantile(cdf, q):
    n = len(cdf) - 1
    for i, c in enumerate(cdf):
        if c >= q:
            if i == 0 or c == cdf[i - 1]:
                return i / n
            return (i - 1 + (q - cdf[i - 1]) / (c - cdf[i - 1])) / n
    return 1.0


def covers(cdf, location, low=0.1, high=0.9):
    """Is a location (0..1, or the community's median) inside the forecast's 10-90% interval?"""
    if location in ("below", "above"):
        return cdf[0] > low if location == "below" else 1 - cdf[-1] > 1 - high
    return _quantile(cdf, low) <= location <= _quantile(cdf, high)


# ---- Levers ---------------------------------------------------------------------------------------

LEVERS = (
    # name, what A is, what B is, the log stages that are its extra cost
    ("everything together", "published", "control", ("research2", "strong", "check", "forecast-extra")),
    ("second research pass", "usual_1", "control", ("research2",)),
    ("ensemble (usual model)", "agg_usual", "usual_1", ("forecast-extra",)),
    ("stronger model, alone", "@strong", "@usual", ("strong",)),
    ("stronger model in the ensemble", "agg_all", "agg_usual", ("strong",)),
    ("aggregation method (ours vs median)", "agg_all", "agg_all_alt", ()),
    ("cap and floor", "agg_all", "agg_all_uncapped", ()),
)


def _score_of(name, configs, kind, resolution):
    if name.startswith("@"):
        names = configs.get("_strong_names" if name == "@strong" else "_usual_names", [])
        scores = [points(kind, configs[n], resolution) for n in names if n in configs]
        scores = [s for s in scores if s is not None]
        return sum(scores) / len(scores) if scores else None
    return points(kind, configs[name], resolution) if name in configs else None


def lever_differences(records, outcomes):
    """{lever: [points A - points B on each resolved question that has both]}."""
    out = {name: [] for name, *_ in LEVERS}
    for record in records:
        outcome = outcomes.get(str(record.get("post_id"))) or {}
        resolution, kind = outcome.get("resolution"), kind_of(record)
        if resolution is None or kind is None:
            continue
        configs = configurations(record)
        for name, a, b, _ in LEVERS:
            sa, sb = _score_of(a, configs, kind, resolution), _score_of(b, configs, kind, resolution)
            if sa is not None and sb is not None:
                out[name].append(sa - sb)
    return out


def lever_cost(records, stages):
    """Mean extra dollars a question for a lever, from the measured cost by stage in the log."""
    costs_ = []
    for record in records:
        by_stage = record.get("cost_by_stage") or {}
        if not (record.get("quality") or {}).get("members"):
            continue
        total = 0.0
        for stage in stages:
            if stage == "forecast-extra":      # every usual-model forecast beyond the first
                f = by_stage.get("forecast") or {}
                calls = f.get("calls", 0)
                total += f.get("cost", 0.0) * (calls - 1) / calls if calls > 1 else 0.0
            else:
                total += (by_stage.get(stage) or {}).get("cost", 0.0)
        costs_.append(total)
    return sum(costs_) / len(costs_) if costs_ else 0.0


def recommend(differences, min_n=MIN_RESOLVED):
    """'keep', 'drop' or 'needs N more resolutions', from paired score differences.

    keep/drop only when there are at least min_n pairs and the mean is two standard errors from
    zero. Otherwise: how many more pairs it would take for a difference of the size seen so far
    to be two standard errors from zero (or just the shortfall to min_n)."""
    n = len(differences)
    if n < 2:
        return {"n": n, "mean": differences[0] if n else None, "se": None,
                "verdict": f"needs {min_n - n} more resolutions"}
    mean = sum(differences) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in differences) / (n - 1))
    se = sd / math.sqrt(n)
    if n >= min_n and se > 0 and abs(mean) > 2 * se:
        verdict = "keep" if mean > 0 else "drop"
    elif n >= min_n and sd == 0:
        verdict = "no difference: drop if it costs anything"
    else:
        need = min_n - n
        if abs(mean) > 1e-9 and sd > 0:
            need = max(need, math.ceil((2 * sd / abs(mean)) ** 2) - n)
        need = max(1, need)
        verdict = "needs 500+ more resolutions (no effect visible)" if need > 500 else f"needs {need} more resolutions"
    return {"n": n, "mean": mean, "se": se, "verdict": verdict}


# ---- The report -----------------------------------------------------------------------------------

def report(records, outcomes):
    lines = ["# Forecast quality: what each lever is worth", ""]
    with_members = [r for r in records if (r.get("quality") or {}).get("members")]
    resolved = [r for r in records if (outcomes.get(str(r.get("post_id"))) or {}).get("resolution") is not None]
    with_cp = [r for r in records if (outcomes.get(str(r.get("post_id"))) or {}).get("community") is not None]
    costs_ = [r["cost"] for r in records if isinstance(r.get("cost"), (int, float))]
    lines += [f"- Forecasts in the log: {len(records)} ({len(with_members)} with ensemble members logged)",
              f"- With a community prediction revealed: {len(with_cp)}; resolved: {len(resolved)}"]
    if costs_:
        lines.append(f"- Measured cost: ${sum(costs_) / len(costs_):.4f} a question (${sum(costs_):.2f} over {len(costs_)})")
    by_tier = {}
    for r in with_members:
        if isinstance(r.get("cost"), (int, float)):
            by_tier.setdefault(r["quality"].get("tier", "?"), []).append(r["cost"])
    for name, values in sorted(by_tier.items()):
        lines.append(f"  - tier {name}: ${sum(values) / len(values):.4f} a question over {len(values)}")

    lines += ["", "## Against real resolutions (peer-score points per question; positive = the lever helped)", ""]
    diffs = lever_differences(records, outcomes)
    lines += ["| lever | pairs | mean gain | +/- 2 s.e. | extra cost a question | recommendation |", "|---|---|---|---|---|---|"]
    for name, _, _, stages in LEVERS:
        r = recommend(diffs[name])
        mean = "n/a" if r["mean"] is None else f"{r['mean']:+.1f}"
        band = "n/a" if r["se"] is None else f"{2 * r['se']:.1f}"
        cost = lever_cost(records, stages)
        lines.append(f"| {name} | {r['n']} | {mean} | {band} | {'free' if not stages else f'${cost:.4f}'} | {r['verdict']} |")
    flagged, clean = [], []
    for record in resolved:
        outcome, kind = outcomes[str(record["post_id"])], kind_of(record)
        check = (record.get("quality") or {}).get("check")
        published = configurations(record).get("published")
        if check is None or published is None or kind is None:
            continue
        (flagged if check else clean).append(points(kind, published, outcome["resolution"]))
    if flagged or clean:
        mean = lambda xs: f"{sum(xs) / len(xs):+.1f}" if xs else "n/a"
        lines += ["", f"Shadow check: {len(flagged)} flagged forecast(s) scored {mean(flagged)} on average, "
                      f"{len(clean)} unflagged scored {mean(clean)} (raw log score, not paired). "
                      + ("needs more flagged resolutions before a flag may change a forecast"
                         if len(flagged) < 10 else "enough flagged forecasts to judge: compare the two means")]
    scored = [(points(kind_of(r), configurations(r).get("published"), outcomes[str(r["post_id"])]["resolution"]), r)
              for r in resolved if configurations(r).get("published") is not None]
    binary = [(configurations(r)["published"], outcomes[str(r["post_id"])]["resolution"]) for r in resolved
              if kind_of(r) == "binary" and "published" in configurations(r)]
    if binary:
        lines += ["", f"Binary questions resolved: {len(binary)}; Brier score of the published forecasts "
                      f"{sum(brier(p, o) for p, o in binary) / len(binary):.3f} (0.25 = always saying 50%)."]
    numeric = [(configurations(r)["published"], outcomes[str(r["post_id"])]["resolution"]) for r in resolved
               if kind_of(r) == "numeric" and "published" in configurations(r)]
    if numeric:
        inside = sum(1 for cdf, loc in numeric if covers(cdf, loc))
        lines.append(f"Numeric questions resolved: {len(numeric)}; the outcome fell inside the published 10-90% interval "
                     f"{inside} time(s) ({100 * inside / len(numeric):.0f}%; about 80% is right, much less means too narrow).")
    worst = sorted((s, r) for s, r in scored if s is not None)[:5]
    if worst:
        lines += ["", "Worst-scoring published forecasts (read these first):"]
        lines += [f"- {s:.0f}: [{(r.get('title') or '')[:90]}]({r.get('url')})" for s, r in worst]

    lines += ["", "## Against the community prediction (a proxy: it shows 'different', not 'better')", ""]
    gaps, coverage, far = {}, {}, []
    for record in with_cp:
        outcome, kind = outcomes[str(record["post_id"])], kind_of(record)
        if kind is None:
            continue
        for name, forecast in configurations(record).items():
            if name.startswith("_"):
                continue
            try:
                gap = community_gap(kind, forecast, outcome["community"])
            except Exception:
                continue
            gaps.setdefault((kind, name), []).append(gap)
            if kind == "numeric":
                coverage.setdefault(name, []).append(covers(forecast, _quantile(outcome["community"], 0.5)))
            if kind == "binary" and name == "published" and gap >= FAR_LOGODDS:
                far.append((gap, forecast, outcome["community"], record))
    if gaps:
        units = {"binary": "log-odds", "choice": "share of probability placed differently", "numeric": "share of the range between medians"}
        lines += ["| type | configuration | questions | mean gap from the community |", "|---|---|---|---|"]
        for (kind, name), values in sorted(gaps.items()):
            lines.append(f"| {kind} | {name} | {len(values)} | {sum(values) / len(values):.3f} {units[kind]} |")
        for name, values in sorted(coverage.items()):
            lines.append(f"- numeric coverage, {name}: the community's median is inside our 10-90% interval on {sum(values)} of {len(values)}")
    else:
        lines.append("No community predictions read yet (run with --fetch after questions close).")
    if far:
        lines += ["", f"Binary forecasts at least {FAR_LOGODDS:.1f} log-odds from the community:"]
        lines += [f"- ours {100 * p:.0f}% vs community {100 * c:.0f}% (gap {g:.2f}): [{(r.get('title') or '')[:80]}]({r.get('url')})"
                  for g, p, c, r in sorted(far, key=lambda x: -x[0])]

    noted = [(r, (r.get("quality") or {})) for r in records]
    noted = [(r, q) for r, q in noted if q.get("flags") or q.get("check")]
    if noted:
        lines += ["", "## Flags raised on our own forecasts", ""]
        for r, q in noted[-25:]:
            items = list(q.get("flags") or []) + [f"{c['kind']}: {c['why']}" for c in (q.get("check") or [])]
            lines.append(f"- [{(r.get('title') or '')[:80]}]({r.get('url')}): " + "; ".join(items))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the bot's logged forecasts, lever by lever (read-only).")
    parser.add_argument("--log-dir", default=str(Path(__file__).parent / "forecast_log"))
    parser.add_argument("--fetch", action="store_true", help="read resolutions and community predictions from Metaculus")
    parser.add_argument("--outcomes", help="JSON file of outcomes to read (and, with --fetch, to write)")
    parser.add_argument("--summary", help="write the report here as well as printing it")
    args = parser.parse_args(argv)
    records = load_log(args.log_dir)
    outcomes = {}
    if args.outcomes and Path(args.outcomes).exists() and not args.fetch:
        outcomes = json.loads(Path(args.outcomes).read_text())
    if args.fetch:
        outcomes = fetch_outcomes([r["post_id"] for r in records if r.get("post_id")], os.getenv("METACULUS_TOKEN"))
        if args.outcomes:
            Path(args.outcomes).write_text(json.dumps(outcomes, indent=1, sort_keys=True) + "\n")
    text = report(records, outcomes)
    print(text)
    if args.summary:
        Path(args.summary).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
