"""Measured API cost by stage (research, research retry, summarise, forecast, parse).

Every model call the bot makes goes through GeneralLlm._call_litellm. install() wraps
that one method, reads the token and web-search counts from the API response's usage
block, prices them with the table below, and files them under the stage and question
that were current when the call was made. The ledger (budget.py) charges this measured
figure, and a short per-stage summary is printed at the end of each run.

Prices live here and nowhere else.
"""
import contextvars
from datetime import datetime, timezone

# Dollars per million tokens. Source: Anthropic pricing page, checked 2026-11-17.
# "long" applies to a request whose prompt is over long_over tokens (Haiku 5.5 only).
PRICES = {
    "claude-haiku-5-5": {
        "in": 0.10, "out": 0.50, "cache_write": 0.125, "cache_read": 0.01,
        "long_over": 100_000,
        "long": {"in": 0.50, "out": 2.50, "cache_write": 0.625, "cache_read": 0.05},
    },
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_write": 2.50, "cache_read": 0.10},
}
WEB_SEARCH_PER_1000 = 10.00  # dollars per 1,000 searches; failed searches are not billed

STAGES = ("research", "research_retry", "summarise", "forecast", "parse", "other")
# Charged for a call that timed out on our side (see Recorder.add_timeout) when the stage
# has no measured call yet to take an average from.
TIMEOUT_FALLBACK = {"research": 0.04, "research_retry": 0.03, "forecast": 0.01}
TIMEOUT_FALLBACK_OTHER = 0.001

_stage = contextvars.ContextVar("cost_stage", default="other")
_question = contextvars.ContextVar("cost_question", default=None)


def price_for(model):
    """Price row for a model name such as 'anthropic/claude-haiku-5-5', or None."""
    name = str(model or "").split("/")[-1]
    if name in PRICES:
        return PRICES[name]
    for key, row in PRICES.items():  # dated variants, e.g. claude-haiku-5-5-20261001
        if name.startswith(key):
            return row
    return None


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def usage_from_response(response):
    """Token and search counts from a litellm response (object or dict). Never raises.

    litellm's prompt_tokens for Anthropic includes cache reads and cache writes, so the
    plain input count is what is left after taking those out.
    """
    usage = _get(response, "usage")
    prompt = int(_get(usage, "prompt_tokens", 0) or 0)
    cache_read = int(_get(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(_get(usage, "cache_creation_input_tokens", 0) or 0)
    searches = _get(_get(usage, "server_tool_use"), "web_search_requests", 0) or 0
    return {
        "in": max(0, prompt - cache_read - cache_write),
        "out": int(_get(usage, "completion_tokens", 0) or 0),
        "cache_read": cache_read,
        "cache_write": cache_write,
        "searches": int(searches),
    }


def call_cost(model, usage):
    """Dollars for one call, or None when the model has no price row."""
    row = price_for(model)
    if row is None:
        return None
    prompt = usage["in"] + usage["cache_read"] + usage["cache_write"]
    if "long" in row and prompt > row["long_over"]:
        row = row["long"]
    tokens = (
        usage["in"] * row["in"] + usage["out"] * row["out"]
        + usage["cache_read"] * row["cache_read"] + usage["cache_write"] * row["cache_write"]
    ) / 1_000_000
    return tokens + usage["searches"] * WEB_SEARCH_PER_1000 / 1000


def empty_stage():
    return {"calls": 0, "searches": 0, "in": 0, "out": 0, "cache_read": 0, "cache_write": 0,
            "timeouts": 0, "cost": 0.0}


def add_into(total, part):
    """Add one stage record into another, in place."""
    for key, value in part.items():
        total[key] = round(total.get(key, 0) + value, 6) if key == "cost" else total.get(key, 0) + value
    return total


class Recorder:
    """What this run spent, by question and stage."""

    def __init__(self):
        self.by_question = {}   # question key -> stage -> record
        self.unpriced = set()   # models with no price row (charged at litellm's figure)
        self.library_cost = 0.0  # litellm's own figure for the same calls, as a cross-check
        self.season_stages = {}  # season totals before this run, for averaging timeouts

    @property
    def current_usage(self):
        return round(sum(r["cost"] for q in self.by_question.values() for r in q.values()), 6)

    def _slot(self, stage=None, question=None):
        stage = stage or _stage.get()
        question = question if question is not None else _question.get()
        return self.by_question.setdefault(str(question), {}).setdefault(stage, empty_stage())

    def add_call(self, model, usage, library_cost=None, stage=None, question=None):
        cost = call_cost(model, usage)
        if cost is None:
            self.unpriced.add(str(model))
            cost = float(library_cost or 0.0)
        self.library_cost += float(library_cost or 0.0)
        slot = self._slot(stage, question)
        add_into(slot, {"calls": 1, "cost": cost, **usage})
        return cost

    def add_timeout(self, stage=None, question=None):
        """A call that timed out on our side returned no usage, but Anthropic may still
        bill the work done. Charge it the stage's average measured call cost."""
        stage = stage or _stage.get()
        seen = empty_stage()
        add_into(seen, self.run_stages().get(stage, empty_stage()))
        add_into(seen, self.season_stages.get(stage, empty_stage()))
        if seen["calls"]:
            cost = seen["cost"] / (seen["calls"] + seen["timeouts"])
        else:
            cost = TIMEOUT_FALLBACK.get(stage, TIMEOUT_FALLBACK_OTHER)
        add_into(self._slot(stage, question), {"timeouts": 1, "cost": cost})
        return cost

    def question_stages(self, question):
        return self.by_question.get(str(question), {})

    def run_stages(self):
        totals = {}
        for stages in self.by_question.values():
            for stage, record in stages.items():
                add_into(totals.setdefault(stage, empty_stage()), record)
        return totals

    def research_searches(self):
        """{question key: web searches its research ran this run (first call plus any retry)}, for
        the questions that were researched this run. This is what the adaptive search count is
        measured by: it was a fixed 3 a question until 2026-11-18."""
        out = {}
        for key, stages in self.by_question.items():
            calls = [stages[s] for s in ("research", "research_retry") if s in stages and stages[s]["calls"]]
            if calls and key != "None":
                out[key] = sum(r["searches"] for r in calls)
        return out

    def question_count(self):
        return len([q for q in self.by_question if q != "None"])


RECORDER = Recorder()


class stage:
    """with costs.stage("research", question_key): ... files calls under that stage."""

    def __init__(self, name, question=None):
        self.name, self.question = name, question

    def __enter__(self):
        self._s = _stage.set(self.name)
        self._q = _question.set(self.question) if self.question is not None else None
        return self

    def __exit__(self, *exc):
        _stage.reset(self._s)
        if self._q is not None:
            _question.reset(self._q)


def staged(name, func):
    """Wrap an async function so the calls it makes are filed under one stage."""
    async def wrapper(*args, **kwargs):
        with stage(name):
            return await func(*args, **kwargs)
    return wrapper


def _is_timeout(exc):
    return "timeout" in type(exc).__name__.lower()


def install(llm_class, recorder=None):
    """Wrap llm_class._call_litellm so every model call is recorded. Safe to call twice."""
    recorder = recorder or RECORDER
    if getattr(llm_class._call_litellm, "_cost_stages", False):
        return
    original = llm_class._call_litellm

    async def _call_litellm(self, prompt):
        try:
            response = await original(self, prompt)
        except Exception as exc:
            if _is_timeout(exc):
                recorder.add_timeout()
            raise
        try:
            hidden = getattr(response, "_hidden_params", None) or {}
            recorder.add_call(getattr(self, "model", None), usage_from_response(response),
                              library_cost=hidden.get("response_cost"))
        except Exception:  # accounting must never break a forecast
            pass
        return response

    _call_litellm._cost_stages = True
    llm_class._call_litellm = _call_litellm


def merge_season(ledger, recorder, now=None):
    """Add this run's stage totals into the ledger dict (saved by budget.record)."""
    season = ledger.setdefault("stages", {})
    for stage_name, record in recorder.run_stages().items():
        add_into(season.setdefault(stage_name, empty_stage()), record)
    hist = ledger.setdefault("research_search_hist", {})   # searches used -> researched questions
    for used in recorder.research_searches().values():
        hist[str(used)] = hist.get(str(used), 0) + 1
    ledger["stage_questions"] = ledger.get("stage_questions", 0) + recorder.question_count()
    ledger.setdefault("stage_since", (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d"))
    return ledger


def searches_line(hist):
    """'1.67 a question (1: 1, 2: 2)' for {searches used: questions}; 'n/a' when empty."""
    hist = {int(k): int(v) for k, v in (hist or {}).items() if int(v) > 0}
    n = sum(hist.values())
    if not n:
        return "n/a"
    mean = sum(k * v for k, v in hist.items()) / n
    return f"{mean:.2f} a question ({', '.join(f'{k}: {hist[k]}' for k in sorted(hist))})"


def summary(run_stages, run_questions, season_stages, season_questions, since="", run_searches=None, season_searches=None):
    """Short per-stage table: this run and season-to-date."""
    run_total = sum(r["cost"] for r in run_stages.values())
    season_total = sum(r["cost"] for r in season_stages.values())
    lines = ["[cost] stage          calls srch  tok_in tok_out   run $ share | season $ share"]
    for name in STAGES:
        r, s = run_stages.get(name), season_stages.get(name)
        if not r and not s:
            continue
        r, s = r or empty_stage(), s or empty_stage()
        lines.append(
            f"[cost] {name:<14} {r['calls']:>5} {r['searches']:>4} "
            f"{r['in'] + r['cache_read'] + r['cache_write']:>7} {r['out']:>7} "
            f"{r['cost']:>7.4f} {_share(r['cost'], run_total):>5} | "
            f"{s['cost']:>8.4f} {_share(s['cost'], season_total):>5}"
        )
    lines.append(
        f"[cost] run: {run_questions} question(s), ${run_total:.4f}, {_per(run_total, run_questions)} a question | "
        f"season{' since ' + since if since else ''}: {season_questions} question(s), ${season_total:.2f}, "
        f"{_per(season_total, season_questions)} a question"
    )
    if run_searches or season_searches:
        run_hist = {}
        for used in (run_searches or {}).values():
            run_hist[used] = run_hist.get(used, 0) + 1
        lines.append(f"[cost] web searches per researched question (ceiling 3; always 3 before 2026-11-18): run {searches_line(run_hist)} | "
                     f"since adaptive search: {searches_line(season_searches)}")
    timeouts = sum(s.get("timeouts", 0) for s in season_stages.values())
    if timeouts:
        lines.append(f"[cost] {timeouts} timed-out call(s) this season charged at the stage average (unmeasured)")
    return "\n".join(lines)


def _share(part, total):
    return f"{100 * part / total:.0f}%" if total > 0 else "-"


def _per(total, count):
    return f"${total / count:.4f}" if count else "n/a"
