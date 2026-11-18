"""One small JSON file per question in forecast_log/, committed with the spending ledger.

Kept for two reasons:
- Scoring and diagnosis when questions resolve: the forecast, when it was made, the
  question id and type, the community prediction at that moment (when Metaculus shows
  one), the research text, the rationale, and what each stage cost.
- Not paying for the same research twice: if a question was researched but the forecast
  step then failed, the next run reuses that research while it is still fresh.

DIR is None until main.py turns logging on (tournament runs only). Nothing here raises.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

DIR = None
RESEARCH_REUSE_HOURS = 6     # a failed question is retried within minutes; older research is redone
MIN_RESEARCH_CHARS = 400     # same threshold as the "nearly empty research" guard in main.py


def _now(now=None):
    return now or datetime.now(timezone.utc)


def _iso(value):
    try:
        return value.isoformat() if value is not None else None
    except Exception:
        return str(value)


def to_jsonable(value):
    """A forecast (float, option list, distribution, conditional) as plain JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            pass
    return str(value)


def question_key(question):
    return str(getattr(question, "id_of_post", None) or getattr(question, "id_of_question", None) or "unknown")


def question_fields(question):
    """What identifies a question, plus what Metaculus showed for free when it was fetched."""
    return {
        "post_id": getattr(question, "id_of_post", None),
        "question_id": getattr(question, "id_of_question", None),
        "type": type(question).__name__,
        "url": getattr(question, "page_url", None),
        "title": getattr(question, "question_text", None),
        "tournaments": to_jsonable(getattr(question, "tournament_slugs", None)),
        "open_time": _iso(getattr(question, "open_time", None)),
        "close_time": _iso(getattr(question, "close_time", None)),
        "scheduled_resolution_time": _iso(getattr(question, "scheduled_resolution_time", None)),
        "cp_reveal_time": _iso(getattr(question, "cp_reveal_time", None)),
        # None while the community prediction is hidden (usual on bot tournaments).
        "community_prediction_at_access": getattr(question, "community_prediction_at_access_time", None),
        "num_forecasters": getattr(question, "num_forecasters", None),
        "accessed_at": _iso(getattr(question, "date_accessed", None)),
    }


def path_for(key, directory=None):
    directory = directory if directory is not None else DIR
    return None if directory is None else Path(directory) / f"{key}.json"


def read(key, directory=None):
    path = path_for(key, directory)
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write(key, record, directory=None):
    path = path_for(key, directory)
    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
        return True
    except OSError:
        return False


def save_research(question, research, retried=False, reused=False, directory=None, now=None, searches=None, second=None):
    """Called as soon as research exists, so it survives a failure in the forecast step."""
    key = question_key(question)
    record = read(key, directory) or {}
    record.update(question_fields(question))
    record.setdefault("status", "researched")
    if not reused:
        record["research"] = research
        record["research_time"] = _iso(_now(now))
        record["research_retried"] = bool(retried)
        if searches is not None:      # web searches this research ran (the ceiling is 3; fewer is the saving)
            record["research_searches"] = int(searches)
        if second is not None:        # whether the second research pass added a section (quality.py)
            record["research_second"] = bool(second)
    record["research_reused"] = bool(reused) or record.get("research_reused", False)
    return write(key, record, directory)


def reusable_research(record, now=None, max_hours=RESEARCH_REUSE_HOURS):
    """Research text from an earlier run worth reusing, or None.

    Only when no forecast was made from it, it is recent, and it is a real write-up
    (not the nearly-empty kind that main.py retries).
    """
    if not record or record.get("status") == "forecast":
        return None
    text = record.get("research") or ""
    if len(text.strip()) < MIN_RESEARCH_CHARS or "web research returned almost nothing" in text:
        return None
    try:
        made = datetime.fromisoformat(record["research_time"])
    except (KeyError, TypeError, ValueError):
        return None
    age_hours = (_now(now) - made).total_seconds() / 3600
    return text if 0 <= age_hours <= max_hours else None


def save_forecast(question, prediction, rationales=None, errors=None, directory=None, now=None, extra=None):
    """Called once the forecast has been made and published. extra: more fields to keep (the
    published CDF, the quality settings and tier, check flags: see main.py and evaluate.py)."""
    key = question_key(question)
    record = read(key, directory) or question_fields(question)
    record.update({
        "status": "forecast",
        "forecast_time": _iso(_now(now)),
        "prediction": to_jsonable(prediction),
        "rationales": to_jsonable(rationales or []),
        "errors": to_jsonable(errors or []),
    })
    record.update(to_jsonable(extra or {}))
    return write(key, record, directory)


def add_costs(key, stages, directory=None):
    """Add one run's per-stage usage for a question to its file (a question that failed
    and was retried in a later run ends up with the total across runs)."""
    record = read(key, directory)
    if record is None or not stages:
        return False
    kept = record.setdefault("cost_by_stage", {})
    for stage, usage in stages.items():
        slot = kept.setdefault(stage, {})
        for field, value in usage.items():
            slot[field] = round(slot.get(field, 0) + value, 6)
    record["cost"] = round(sum(s.get("cost", 0.0) for s in kept.values()), 6)
    return write(key, record, directory)
