"""Quality levers: where the spare API credit is spent to buy accuracy, and the pure maths behind them.

Each lever is one setting below (override any of them with an environment variable QUALITY_<NAME>,
for example a repository variable passed to the workflow). docs/quality.md has the evidence for
each one, its cost a question, and why it ships on or off. Nothing in this file calls a model or
the network, so everything here is tested with plain numbers (tests/test_quality.py).
"""
import calendar
import math
import os
import re
from statistics import median


def _env(name, default):
    raw = os.getenv(f"QUALITY_{name}")
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


# ---- The levers -------------------------------------------------------------------------------
# ON: several independent forecasts, aggregated. Outside evidence: strongest of any scaffold lever.
ENSEMBLE_MEMBERS = _env("ENSEMBLE_MEMBERS", 2)          # forecasts from the usual forecasting model
# ON: one more member from a stronger model. Outside evidence: the base model matters most.
STRONG_MODEL = _env("STRONG_MODEL", "anthropic/claude-opus-5-5")   # "" turns it off
STRONG_MEMBERS = _env("STRONG_MEMBERS", 1)
# ON: a second research pass for the base rate, the primary source and the case against.
SECOND_RESEARCH = _env("SECOND_RESEARCH", True)
# ON: one extra forecast made the old way (first research only, usual model). It is logged and
# never published: it is what the bot would have forecast before these levers, so every lever
# can be scored against it on the same questions.
CONTROL_MEMBER = _env("CONTROL_MEMBER", True)
# How members are combined. Both alternatives are re-computed for free by evaluate.py.
BINARY_AGGREGATE = _env("BINARY_AGGREGATE", "trimmed_logodds")     # or "median"
NUMERIC_AGGREGATE = _env("NUMERIC_AGGREGATE", "mixture")           # or "median"
# ON (mild): never publish a binary forecast outside these limits; floor for each option.
BINARY_MIN = _env("BINARY_MIN", 0.02)
BINARY_MAX = _env("BINARY_MAX", 0.98)
OPTION_FLOOR = _env("OPTION_FLOOR", 0.005)
# ON: discrete questions are answered as a probability per outcome (see docs: 46618).
DISCRETE_OUTCOME_PROBABILITIES = _env("DISCRETE_OUTCOME_PROBABILITIES", True)
# ON, shadow only: a cheap check of dates, units, bounds and "already resolved?". It writes flags
# to the forecast log and never changes a forecast.
SHADOW_CHECK = _env("SHADOW_CHECK", True)
# OFF: different personas for the members. A prompt change with no evidence behind it.
PERSONAS = _env("PERSONAS", False)
# OFF: shrink/extremise forecasts. Contested outside; needs our own resolutions first.
EXTREMISE = _env("EXTREMISE", 1.0)                      # 1.0 = no change (exponent on the odds)
# The API credit this bot may use in a calendar month (it expires monthly).
MONTHLY_CREDIT = _env("MONTHLY_CREDIT", 45.0)
# ON: spend credit the pace has left over on extra forecasts from the stronger model, on the same
# question, inside the same 3-hour window (docs/quality.md, "Where the unused credit goes").
# Re-forecasting old questions was the other candidate and is NOT built: the tournament scores only
# the forecast standing when a question closes, 3 hours after it opens, and asks for one forecast
# a question. Off-switch: QUALITY_SPARE_BOOST=0 (or set False here).
SPARE_BOOST = _env("SPARE_BOOST", True)
BOOST_STRONG_MEMBERS = _env("BOOST_STRONG_MEMBERS", 2)   # extra strong-model forecasts a question
# The extras run only while this month's spending is under this share of the pacing line, so they
# can only ever use credit the pace has left over, never push the month over it.
BOOST_BELOW = _env("BOOST_BELOW", 0.5)
# ...and only while the run has used at most this share of what it may spend, so on a busy run the
# money goes to covering every open question first.
BOOST_RUN_SHARE = _env("BOOST_RUN_SHARE", 0.25)
EXPECTED_BOOST_MEMBER_COST = 0.035                       # dollars for one strong-model forecast

SECOND_MARK = "\n\n---\n## Second research pass (base rate, primary source, the case against)\n"
PERSONA_LINES = (
    "",
    "For this answer, start from the base rate of similar past events and adjust from there.",
    "For this answer, look hardest for the reason the obvious answer could be wrong.",
)
# Expected dollars a question by tier (docs/quality.md has the arithmetic; the log has the truth).
EXPECTED_COST = {"full": 0.14, "lean": 0.075, "base": 0.049}


def settings():
    """The levers as a dict, written into each forecast's log so results can be split by setting."""
    return {
        "ensemble_members": ENSEMBLE_MEMBERS, "strong_model": STRONG_MODEL, "strong_members": STRONG_MEMBERS,
        "second_research": SECOND_RESEARCH, "control_member": CONTROL_MEMBER,
        "binary_aggregate": BINARY_AGGREGATE, "numeric_aggregate": NUMERIC_AGGREGATE,
        "binary_min": BINARY_MIN, "binary_max": BINARY_MAX, "option_floor": OPTION_FLOOR,
        "discrete_outcome_probabilities": DISCRETE_OUTCOME_PROBABILITIES, "shadow_check": SHADOW_CHECK,
        "personas": PERSONAS, "extremise": EXTREMISE,
        "spare_boost": SPARE_BOOST, "boost_strong_members": BOOST_STRONG_MEMBERS,
    }


# ---- Pacing: how much quality this month's credit can afford ------------------------------------

def month_spent(by_day, now):
    """Dollars the ledger shows for the calendar month of `now` ({'2026-11-18': 0.39, ...})."""
    prefix = now.strftime("%Y-%m")
    return round(sum(v for k, v in (by_day or {}).items() if str(k).startswith(prefix)), 4)


def days_in_month(now):
    return calendar.monthrange(now.year, now.month)[1]


def tier(spent_this_month, now, credit=None):
    """'full', 'lean' or 'base'. No question is ever skipped for pacing: only the extras go.

    full: spending is on or under the straight line from $0 to the month's credit (two days' slack).
    lean: ahead of the line. Drop the second research pass, the control and the strong model.
    base: the month's credit is used. One forecast a question, as before these levers.
    """
    credit = MONTHLY_CREDIT if credit is None else credit
    if credit <= 0 or spent_this_month >= credit:
        return "base"
    line = pace_line(now, credit, slack_days=2)
    return "full" if spent_this_month <= line else "lean"


def pace_line(now, credit=None, slack_days=0):
    """Dollars the month may have used by today on the straight line from $0 to the credit."""
    credit = MONTHLY_CREDIT if credit is None else credit
    n = days_in_month(now)
    return credit * min(n, now.day + slack_days) / n


def boost_members(spent_this_month, now, tier_name, credit=None):
    """The extra strong-model forecasters this run may add to each question, or [] for none.

    They are added only at the full tier (so new questions already get everything else), only
    when a strong model is set, and only while the month's spending is under BOOST_BELOW of the
    pacing line: that is the measure of "the pace has left credit over". As spending catches up
    with the line the extras stop by themselves, well before the tier would drop to lean."""
    if not SPARE_BOOST or BOOST_STRONG_MEMBERS <= 0 or not STRONG_MODEL or tier_name != "full":
        return []
    if spent_this_month > BOOST_BELOW * pace_line(now, credit):
        return []
    return [{"name": f"boost_{i + 1}", "model": "strong", "stage": "boost", "role": "boost", "persona": ""}
            for i in range(int(BOOST_STRONG_MEMBERS))]


def boost_fits_run(run_used, run_allowed):
    """May this question get its extras, given what the run has used of what it may spend?
    Extras go to the first questions of a run only; a run with many open questions keeps its
    allowance for covering all of them."""
    return run_allowed > 0 and run_used <= BOOST_RUN_SHARE * run_allowed


def plan(tier_name="full"):
    """Who forecasts a question at this tier: the members, and whether the extras run."""
    usual = [{"name": f"usual_{i + 1}", "model": "usual", "stage": "forecast", "role": "member",
              "persona": PERSONA_LINES[i % len(PERSONA_LINES)] if PERSONAS else ""}
             for i in range(max(1, ENSEMBLE_MEMBERS))]
    strong = [{"name": f"strong_{i + 1}", "model": "strong", "stage": "strong", "role": "member", "persona": ""}
              for i in range(max(0, STRONG_MEMBERS))] if STRONG_MODEL else []
    if tier_name == "base":
        return {"tier": "base", "members": usual[:1], "second_research": False, "control": False}
    if tier_name == "lean":
        members = [{"name": f"usual_{i + 1}", "model": "usual", "stage": "forecast", "role": "member", "persona": ""}
                   for i in range(max(3, len(usual)))]
        return {"tier": "lean", "members": members, "second_research": False, "control": False}
    return {"tier": "full", "members": usual + strong, "second_research": bool(SECOND_RESEARCH),
            "control": bool(CONTROL_MEMBER and SECOND_RESEARCH)}


# ---- Aggregation ------------------------------------------------------------------------------

def logit(p):
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1 - p))


def expit(x):
    return 1 / (1 + math.exp(-x))


def cap(p, low=None, high=None):
    low = BINARY_MIN if low is None else low
    high = BINARY_MAX if high is None else high
    return max(low, min(high, float(p)))


def extremise(p, exponent=None):
    """Push a probability away from 50% (exponent > 1) or towards it (< 1), in odds space."""
    exponent = EXTREMISE if exponent is None else exponent
    return p if exponent == 1.0 else expit(exponent * logit(p))


def aggregate_binary(probabilities, method=None):
    """One probability from several. trimmed_logodds: drop the highest and the lowest (when there
    are three or more) and average the rest in log-odds; with three members that is the median."""
    ps = sorted(float(p) for p in probabilities)
    if not ps:
        raise ValueError("no probabilities to aggregate")
    method = method or BINARY_AGGREGATE
    if method == "median":
        return float(median(ps))
    if method == "mean_logodds":
        return expit(sum(logit(p) for p in ps) / len(ps))
    kept = ps[1:-1] if len(ps) >= 3 else ps
    return expit(sum(logit(p) for p in kept) / len(kept))


def floor_and_normalise(probabilities, floor=None):
    """Raise every probability to at least `floor`, then scale the rest so the total is 1."""
    floor = OPTION_FLOOR if floor is None else floor
    ps = [max(0.0, float(p)) for p in probabilities]
    total = sum(ps)
    if total <= 0:
        return [1 / len(ps)] * len(ps)
    ps = [p / total for p in ps]
    floor = min(floor, 1 / len(ps))
    for _ in range(len(ps)):
        low = [i for i, p in enumerate(ps) if p < floor]
        if not low:
            break
        rest = sum(p for p in ps if p > floor)
        over = sum(floor - ps[i] for i in low)
        ps = [floor if p <= floor else p - over * p / rest for p in ps]
    total = sum(ps)
    return [p / total for p in ps]


def aggregate_options(members, floor=None):
    """[{option: probability}, ...] -> {option: probability}: the mean, floored and normalised."""
    names = list(members[0])
    means = [sum(float(m.get(name, 0.0)) for m in members) / len(members) for name in names]
    return dict(zip(names, floor_and_normalise(means, floor)))


def aggregate_cdfs(cdfs, method=None):
    """Several CDFs on the same x grid -> one. mixture: the mean at each point (keeps every
    member's tails). median: the middle value at each point (what the library does)."""
    method = method or NUMERIC_AGGREGATE
    if len({len(c) for c in cdfs}) != 1:
        raise ValueError("CDFs are not on the same grid")
    columns = list(zip(*cdfs))
    out = [float(median(col)) if method == "median" else sum(col) / len(col) for col in columns]
    for i in range(1, len(out)):          # a mean of rising curves rises; guard rounding anyway
        out[i] = max(out[i], out[i - 1])
    return out


# ---- Discrete questions: a probability per outcome ----------------------------------------------

def _tidy(x):
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}"


def discrete_outcomes(lower, upper, cdf_size, open_lower, open_upper):
    """The outcomes of a discrete question as [(label, kind)], lowest first.
    Example (46618): lower 1.5, upper 6.5, cdf_size 6, open above -> 2, 3, 4, 5, 6, 'above 6'."""
    n = int(cdf_size) - 1
    step = (upper - lower) / n
    values = [lower + step * (k + 0.5) for k in range(n)]
    out = [(f"below {_tidy(values[0])}", "below")] if open_lower else []
    out += [(_tidy(v), "in") for v in values]
    if open_upper:
        out.append((f"above {_tidy(values[-1])}", "above"))
    return out


def parse_outcome_probabilities(text, labels):
    """Read 'Outcome <label>: NN%' lines (the last one for each label) -> [probability], or None
    when a label is missing or the total is not close to 100%."""
    found = {}
    for label in labels:
        pattern = r"Outcome\s+" + re.escape(label) + r"\s*:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)"
        hits = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if not hits:
            return None
        number, percent = hits[-1]
        found[label] = float(number) / (100 if percent or float(number) > 1 else 1)
    total = sum(found.values())
    if not 0.9 <= total <= 1.1:
        return None
    return [found[label] / total for label in labels]


def outcome_cdf_points(probabilities, kinds, lower, upper, floor=0.003):
    """[(value, cumulative probability)] at the edges between outcomes, for the forecast object.
    Every outcome keeps at least `floor` so the curve always rises."""
    ps = floor_and_normalise(probabilities, floor)
    below = ps[0] if kinds[0] == "below" else 0.0
    inside = [p for p, kind in zip(ps, kinds) if kind == "in"]
    step = (upper - lower) / len(inside)
    points, running = [(lower, below)], below
    for k, p in enumerate(inside):
        running += p
        points.append((lower + step * (k + 1), min(1.0, running)))
    return points


# ---- Free checks on a forecast (flags go to the log; nothing here changes a forecast) -----------

def numeric_flags(cdf, open_lower, open_upper):
    """Things worth a look in a numeric forecast, from its CDF (cumulative probability at evenly
    spaced points from the lower to the upper bound)."""
    flags = []
    if any(b < a - 1e-9 for a, b in zip(cdf, cdf[1:])):
        flags.append("cdf_not_rising")
    n = len(cdf) - 1
    def where(q):
        for i, c in enumerate(cdf):
            if c >= q:
                return i / n
        return 1.0
    width = where(0.9) - where(0.1)
    if n >= 50 and width < 0.03:
        flags.append(f"narrow: the 10-90% interval covers {100 * width:.1f}% of the question's range")
    if open_lower and cdf[0] > 0.25:
        flags.append(f"{100 * cdf[0]:.0f}% below the lower bound")
    if open_upper and 1 - cdf[-1] > 0.25:
        flags.append(f"{100 * (1 - cdf[-1]):.0f}% above the upper bound")
    return flags


CHECK_KINDS = ("ALREADY_RESOLVED", "DATE", "UNITS", "BOUNDS", "CRITERIA", "IMPOSSIBLE_OPTION")


def parse_check(text):
    """'FLAG <KIND>: why' lines from the shadow checker -> [{'kind','why'}]. 'OK' or junk -> []."""
    flags = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*[-*]?\s*FLAG\s+([A-Z_]+)\s*:\s*(.+)", line.strip())
        if m and m.group(1) in CHECK_KINDS:
            flags.append({"kind": m.group(1), "why": m.group(2).strip()[:300]})
    return flags
