"""The spare-credit boost: when it runs, what it publishes, what it logs, how it is scored.
Fake models and plain numbers only; nothing here touches the network."""
import asyncio
import logging
import math
import os
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("ANTHROPIC_API_KEY", "not-a-real-key")
os.environ.setdefault("METACULUS_TOKEN", "not-a-real-token")

from forecasting_tools import BinaryQuestion, PredictedOption, PredictedOptionList

import costs
import evaluate
import forecast_log
import main
import quality

logging.getLogger("main").setLevel(logging.CRITICAL)
NOV_19 = datetime(2026, 11, 19, tzinfo=timezone.utc)
BOTH = "First rundown. " * 40 + quality.SECOND_MARK + "Base rate: 3 of 20 years. " * 10


class BoostRuleTest(unittest.TestCase):
    """quality.boost_members: the extras only ever use credit the pace has left over."""

    def test_on_when_the_month_is_far_under_the_pace(self):
        # 19 November, $45 credit: the pace line is 45 * 19 / 30 = $28.50; half of it is $14.25.
        self.assertAlmostEqual(quality.pace_line(NOV_19, 45.0), 28.5)
        names = [m["name"] for m in quality.boost_members(3.17, NOV_19, "full", 45.0)]
        self.assertEqual(names, ["boost_1", "boost_2"])
        self.assertEqual(len(quality.boost_members(14.25, NOV_19, "full", 45.0)), 2)

    def test_off_once_spending_reaches_half_the_pace_line(self):
        self.assertEqual(quality.boost_members(14.26, NOV_19, "full", 45.0), [])
        # The extras stop long before the tier itself drops: $20 is still "full" but not boosted.
        self.assertEqual(quality.tier(20.0, NOV_19, 45.0), "full")
        self.assertEqual(quality.boost_members(20.0, NOV_19, "full", 45.0), [])

    def test_never_at_the_lean_or_base_tier(self):
        self.assertEqual(quality.boost_members(0.0, NOV_19, "lean", 45.0), [])
        self.assertEqual(quality.boost_members(0.0, NOV_19, "base", 45.0), [])

    def test_off_switch_and_no_strong_model(self):
        with mock.patch.object(quality, "SPARE_BOOST", False):
            self.assertEqual(quality.boost_members(0.0, NOV_19, "full", 45.0), [])
        with mock.patch.object(quality, "STRONG_MODEL", ""):
            self.assertEqual(quality.boost_members(0.0, NOV_19, "full", 45.0), [])
        with mock.patch.object(quality, "BOOST_STRONG_MEMBERS", 0):
            self.assertEqual(quality.boost_members(0.0, NOV_19, "full", 45.0), [])

    def test_the_off_switch_reads_the_environment(self):
        with mock.patch.dict(os.environ, {"QUALITY_SPARE_BOOST": "0"}):
            self.assertFalse(quality._env("SPARE_BOOST", True))
        with mock.patch.dict(os.environ, {"QUALITY_SPARE_BOOST": ""}):      # an unset repository variable
            self.assertTrue(quality._env("SPARE_BOOST", True))

    def test_extras_go_to_the_first_questions_of_a_run_only(self):
        self.assertTrue(quality.boost_fits_run(0.0, 1.0))
        self.assertTrue(quality.boost_fits_run(0.25, 1.0))
        self.assertFalse(quality.boost_fits_run(0.26, 1.0))
        self.assertFalse(quality.boost_fits_run(0.0, 0.0))

    def test_a_boosted_month_stays_under_the_credit(self):
        # Every question boosted (the worst case) at today's volume, 4 a day: 4 * 30 * (0.14 + 0.07).
        per_question = quality.EXPECTED_COST["full"] + 2 * quality.EXPECTED_BOOST_MEMBER_COST
        self.assertLess(4 * 30 * per_question, quality.MONTHLY_CREDIT)
        # And whatever the volume, boosting stops at half the line, so the extras alone can never
        # carry the month past the line: the last boosted question starts at or under half of it.
        self.assertLessEqual(quality.BOOST_BELOW, 0.5)


class FakeLlm:
    def __init__(self, model, answer):
        self.model, self.answer, self.prompts = model, answer, []

    async def invoke(self, prompt, *a, **kw):
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return f"Reasoning.\nProbability: {self.answer}%"


async def fake_structure_output(text, output_type, **kw):
    return SimpleNamespace(prediction_in_decimal=float(re.findall(r"Probability: ([0-9.]+)%", text)[-1]) / 100)


class BoostInTheBotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.usual, self.strong = FakeLlm("anthropic/claude-sonnet-5-5", 30), FakeLlm("anthropic/claude-opus-5-5", 60)
        self.bot = main.FallTemplateBot2026(predictions_per_research_report=3,
                                            llms={"default": "anthropic/claude-sonnet-5-5", "summarizer": "anthropic/claude-haiku-5-5",
                                                  "researcher": "anthropic/claude-haiku-5-5", "parser": "anthropic/claude-haiku-5-5"})
        self.question = BinaryQuestion(question_text="Will it happen?", id_of_post=77, id_of_question=78,
                                       page_url="https://example.test/q/77", resolution_criteria="If it happens.",
                                       fine_print="", background_info="")
        real_get = main.FallTemplateBot2026.get_llm
        usual = self.usual

        def get_llm(bot, purpose="default", guarantee_type=None):
            return usual if purpose == "default" else real_get(bot, purpose, guarantee_type)
        self.recorder = costs.Recorder()
        for p in (mock.patch.object(costs, "RECORDER", self.recorder),
                  mock.patch.object(forecast_log, "DIR", Path(self.tmp.name)),
                  mock.patch.object(main, "structure_output", fake_structure_output),
                  mock.patch.object(main.FallTemplateBot2026, "get_llm", get_llm),
                  mock.patch.object(main.FallTemplateBot2026, "plan", quality.plan("full")),
                  mock.patch.object(main.FallTemplateBot2026, "boost", quality.boost_members(0.0, NOV_19, "full", 45.0)),
                  mock.patch.object(main.FallTemplateBot2026, "strong_llm", self.strong),
                  mock.patch.object(main.FallTemplateBot2026, "rationales", {}),
                  mock.patch.object(main.FallTemplateBot2026, "_boost_values", {}),
                  mock.patch.object(main.FallTemplateBot2026, "_member_index", {})):
            p.start()
            self.addCleanup(p.stop)

    def forecast(self):
        """What the library does for one question: one call per member, then combine them."""
        async def go():
            costs._question.set("77")
            self.bot._note_pads.append(await self.bot._initialize_notepad(self.question))
            results = await asyncio.gather(*[self.bot._make_prediction(self.question, BOTH) for _ in range(3)])
            members = [r.prediction_value for r in results]
            return members, await self.bot._aggregate_predictions(members, self.question)
        return asyncio.run(go())

    def logged(self):
        return {r["member"]: r for r in main.FallTemplateBot2026.rationales.get("77", [])}

    def test_extras_forecast_with_the_strong_model_and_join_the_published_aggregate(self):
        members, published = self.forecast()
        self.assertEqual(sorted(members), [0.3, 0.3, 0.6])              # the library still sees the usual three
        # Five forecasts: 0.3, 0.3, 0.6, 0.6, 0.6. Highest and lowest dropped, the rest averaged in log-odds.
        expected = quality.expit((quality.logit(0.3) + 2 * quality.logit(0.6)) / 3)
        self.assertAlmostEqual(published, expected)
        self.assertEqual(len(self.strong.prompts), 3)                   # strong_1 and the two extras
        logged = self.logged()
        self.assertEqual(set(logged), {"usual_1", "usual_2", "strong_1", "boost_1", "boost_2", "control"})
        self.assertEqual((logged["boost_1"]["role"], logged["boost_1"]["model"], logged["boost_1"]["research"]),
                         ("boost", "anthropic/claude-opus-5-5", "both"))
        self.assertEqual(logged["control"]["role"], "control")          # the control is still made and logged
        self.assertEqual(main.FallTemplateBot2026._boost_values, {})    # used once, then cleared

    def test_off_means_exactly_the_old_ensemble(self):
        with mock.patch.object(main.FallTemplateBot2026, "boost", []):
            members, published = self.forecast()
        self.assertAlmostEqual(published, 0.3)
        self.assertEqual(len(self.strong.prompts), 1)
        self.assertEqual(set(self.logged()), {"usual_1", "usual_2", "strong_1", "control"})

    def test_a_failing_extra_is_dropped_not_replaced_and_never_loses_the_forecast(self):
        self.strong.answer = RuntimeError("overloaded")                 # the strong model is down for everyone
        members, published = self.forecast()
        self.assertAlmostEqual(published, 0.3)                          # three forecasts, as if there were no boost
        self.assertEqual(len(members), 3)
        # Usual-model calls: two members, the control, and the stand-in for strong_1 (the existing
        # rule that the ensemble keeps its size). No stand-in is paid for the extras.
        self.assertEqual(len(self.usual.prompts), 4)
        self.assertFalse(any(name.startswith("boost") for name in self.logged()))
        self.assertIn("strong_1_fallback", self.logged())

    def test_no_extras_once_the_run_has_used_a_quarter_of_what_it_may_spend(self):
        self.recorder.add_call("claude-sonnet-5-5", {"in": 200000, "out": 0, "cache_read": 0, "cache_write": 0, "searches": 0},
                               stage="forecast", question="earlier")
        self.assertGreater(self.recorder.current_usage, 0.25)
        with mock.patch.object(main.FallTemplateBot2026, "run_budget", (self.recorder, 1.0)):
            members, published = self.forecast()
        self.assertAlmostEqual(published, 0.3)
        self.assertEqual(len(self.strong.prompts), 1)

    def test_extras_run_at_the_start_of_a_run(self):
        with mock.patch.object(main.FallTemplateBot2026, "run_budget", (self.recorder, 1.0)):
            self.forecast()
        self.assertEqual(len(self.strong.prompts), 3)

    def test_extras_of_a_different_shape_are_left_out(self):
        options = lambda a: PredictedOptionList(predicted_options=[PredictedOption(option_name="A", probability=a),
                                                                  PredictedOption(option_name="B", probability=1 - a)])
        other = PredictedOptionList(predicted_options=[PredictedOption(option_name="A", probability=0.5),
                                                       PredictedOption(option_name="C", probability=0.5)])
        main.FallTemplateBot2026._boost_values["77"] = [options(0.9), other, 0.4]
        got = self.bot._with_boost([options(0.2), options(0.4)], self.question)
        self.assertEqual([p.predicted_options[0].probability for p in got], [0.2, 0.4, 0.9])
        self.assertEqual(self.bot._with_boost([0.2], self.question), [0.2])      # nothing kept for the next question

    def test_the_log_names_the_extras_that_ran(self):
        async def go():
            costs._question.set("77")
            self.bot._note_pads.append(await self.bot._initialize_notepad(self.question))
            await asyncio.gather(*[self.bot._make_prediction(self.question, BOTH) for _ in range(3)])
        asyncio.run(go())
        names = [r["member"] for r in main.FallTemplateBot2026.rationales["77"] if r.get("role") == "boost"]
        self.assertEqual(sorted(names), ["boost_1", "boost_2"])


def entry(name, prediction, role="member"):
    return {"member": name, "role": role, "prediction": prediction, "reasoning": "", "question_id": 1}


def boosted_record(post=1, with_boost=True):
    rationales = [entry("usual_1", 0.3), entry("usual_2", 0.2), entry("strong_1", 0.4), entry("control", 0.5, "control")]
    by_stage = {"research2": {"cost": 0.025}, "strong": {"cost": 0.03}, "forecast": {"cost": 0.03, "calls": 2}}
    if with_boost:
        rationales += [entry("boost_1", 0.6, "boost"), entry("boost_2", 0.7, "boost")]
        by_stage["boost"] = {"cost": 0.07, "calls": 2}
    return {"status": "forecast", "post_id": post, "question_id": 1, "type": "BinaryQuestion",
            "prediction": 0.4 if with_boost else 0.3, "title": f"Q{post}", "url": f"https://example.test/{post}",
            "cost": 0.21 if with_boost else 0.14, "cost_by_stage": by_stage,
            "quality": {"tier": "full", "members": ["usual_1", "usual_2", "strong_1"], "check": [], "flags": [],
                        "boost": ["boost_1", "boost_2"] if with_boost else []},
            "rationales": rationales}


class BoostScoringTest(unittest.TestCase):
    """evaluate.py: the boost is judged against the same question's ensemble without the extras."""
    NAME = "spare-credit boost (extra strong forecasts)"

    def test_with_and_without_the_extras_on_the_same_question(self):
        configs = evaluate.configurations(boosted_record())
        self.assertAlmostEqual(configs["agg_all"], 0.3)                 # the usual three: unchanged by the extras
        self.assertAlmostEqual(configs["agg_boost"], quality.expit((quality.logit(0.3) + quality.logit(0.4) + quality.logit(0.6)) / 3))
        self.assertEqual(configs["boost_1"], 0.6)
        self.assertEqual(configs["_strong_names"], ["strong_1"])        # the other levers do not see the extras

    def test_the_lever_gain_is_the_peer_point_difference(self):
        record = boosted_record()
        boosted = evaluate.configurations(record)["agg_boost"]
        diffs = evaluate.lever_differences([record], {"1": {"resolution": 1}})
        self.assertAlmostEqual(diffs[self.NAME][0], 100 * (math.log(boosted) - math.log(0.3)))
        diffs = evaluate.lever_differences([record], {"1": {"resolution": 0}})
        self.assertLess(diffs[self.NAME][0], 0)                         # it resolved No: the extras cost points

    def test_questions_without_extras_are_not_counted_and_the_other_levers_are_unchanged(self):
        plain, boosted = boosted_record(1, with_boost=False), boosted_record(2)
        outcomes = {"1": {"resolution": 0}, "2": {"resolution": 0}}
        diffs = evaluate.lever_differences([plain, boosted], outcomes)
        self.assertEqual(len(diffs[self.NAME]), 1)
        self.assertEqual(len(diffs["second research pass"]), 2)
        self.assertAlmostEqual(diffs["stronger model in the ensemble"][0], diffs["stronger model in the ensemble"][1])
        self.assertNotIn("agg_boost", evaluate.configurations(plain))

    def test_report_line_and_cost_over_boosted_questions_only(self):
        records = [boosted_record(1, with_boost=False), boosted_record(2), boosted_record(3)]
        text = evaluate.report(records, {"2": {"resolution": 1}, "3": {"resolution": 0}})
        self.assertIn(f"| {self.NAME} | 2 | ", text)
        self.assertIn("$0.0700", text)
        self.assertAlmostEqual(evaluate.lever_cost(records, ("boost",)), 0.07)


if __name__ == "__main__":
    unittest.main()
