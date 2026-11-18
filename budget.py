"""Hard spending limits for the bot, kept in a small ledger file in the repo.

The owner set a ceiling for the season. Every run reads the ledger, refuses to
start if the ceiling or today's allowance is used up, limits its own spending
to what is left, then records what it spent. The workflow commits the ledger
back to the repo so the next run sees it.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).parent / "spend_ledger.json"
# 2026-11-17 (owner): run at full throttle, forecast every question, and measure what that
# costs. The caps below are now guards against a bug, not pacing. The Anthropic console showed
# $3.83 really spent by this key through Nov 18 04:36 UTC (the old ledger said $9.80: it charged
# about 2.6 times the real cost), and the ledger was corrected to that figure.
# Expected at full throttle: about $0.045 a question, some 50 questions a day = $2 to $2.50 a day.
SEASON_CAP = 90.00   # dollars, total through the end of the season (bug guard)
# 2026-11-18: raised from $5 to $8 with the quality levers (quality.py; docs/quality.md). A question now
# costs about $0.14, and the busiest day seen (2026-11-16, about 40 questions) would cost about $5.60 at the
# full tier; at $5 the guard itself would have skipped questions,
# and a skipped question is the most expensive mistake there is. The month's total is held by the pacing
# in quality.tier (about $45), not by this cap.
DAILY_CAP = 8.00     # dollars per UTC day (bug guard)
PER_RUN_CAP = 1.00   # dollars per run
# No padding since 2026-11-17: the ledger now charges the measured cost of every call
# (costs.py: tokens and web searches from the API's own usage figures, at list prices).
# Before that, the 1.3 was applied to the library's cost meter, which (a) already
# included web-search fees, so the padding counted them a second time, and (b) added
# every call twice, so the ledger charged about 2.6 times the real cost. Entries dated
# before the change are in those inflated dollars.
# The one thing still unmeasured is a call that times out on our side after Anthropic
# has done billable work; costs.py charges each of those the stage's average call cost.
PADDING = 1.0

# Pacing inside the owner's limits (review of 2026-11-16). MiniBench opens about two
# questions an hour ($1,000 pool per fortnight); the main tournament opens one or two a
# day ($50,000 pool). Left alone, MiniBench used the whole daily allowance by breakfast
# and would have emptied the season budget by late November, leaving nothing for the main
# tournament in December. So: money for the main tournament is set aside first, and
# MiniBench gets an even daily share of what is left. MiniBench is kept at all because its
# questions resolve within about two weeks, which is the feedback the bot is tuned on.
SEASON_END = "2026-12-20"        # main-tournament questions stop opening a few weeks before Jan 1
MAIN_RESERVE_PER_DAY = 0.45      # dollars a day kept back for main-tournament questions


def days_left(now=None):
    now = now or datetime.now(timezone.utc)
    end = datetime.strptime(SEASON_END, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max(1, (end.date() - now.date()).days + 1)


def minibench_room(data, now=None):
    """Dollars MiniBench may still spend today (0 means skip MiniBench this run)."""
    n = days_left(now)
    pool = SEASON_CAP - data["spent"] - MAIN_RESERVE_PER_DAY * n
    per_day = max(0.0, pool) / n
    used = data.get("minibench_by_day", {}).get(today(now), 0.0)
    left_today = DAILY_CAP - data["by_day"].get(today(now), 0.0)
    room = left_today            # no pacing since 2026-11-17 (owner: full throttle); per_day is kept for the record
    return round(room, 4) if room >= 0.05 else 0.0


def load(path=LEDGER):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault("spent", 0.0)
    data.setdefault("by_day", {})
    return data


def today(now=None):
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def allowance(data, now=None):
    """Dollars this run may spend (0 means do not run)."""
    left_season = SEASON_CAP - data["spent"]
    left_today = DAILY_CAP - data["by_day"].get(today(now), 0.0)
    room = min(PER_RUN_CAP, left_season, left_today)
    return round(room, 4) if room >= 0.05 else 0.0


def record(data, estimated_cost, now=None, path=LEDGER, minibench_cost=0.0):
    cost = round(max(0.0, estimated_cost) * PADDING, 4)
    day = today(now)
    mb = round(max(0.0, min(minibench_cost, estimated_cost)) * PADDING, 4)
    if mb:
        data.setdefault("minibench_by_day", {})
        data["minibench_by_day"][day] = round(data["minibench_by_day"].get(day, 0.0) + mb, 4)
    data["spent"] = round(data["spent"] + cost, 4)
    data["by_day"][day] = round(data["by_day"].get(day, 0.0) + cost, 4)
    data["cap"] = SEASON_CAP
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return cost
