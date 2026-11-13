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
SEASON_CAP = 20.00   # dollars, total through the end of the season (owner's limit)
DAILY_CAP = 1.00     # dollars per UTC day, so a backlog cannot drain the season in a day
PER_RUN_CAP = 0.60   # dollars per run
# Web-search fees are billed per search and are not in the library's cost
# estimate, so recorded spending is padded to stay on the safe side.
PADDING = 1.3


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


def record(data, estimated_cost, now=None, path=LEDGER):
    cost = round(max(0.0, estimated_cost) * PADDING, 4)
    day = today(now)
    data["spent"] = round(data["spent"] + cost, 4)
    data["by_day"][day] = round(data["by_day"].get(day, 0.0) + cost, 4)
    data["cap"] = SEASON_CAP
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return cost
