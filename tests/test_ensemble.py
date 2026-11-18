"""The ensemble in main.py: who forecasts, what is published, what is logged. Fake models only."""
import asyncio
import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("ANTHROPIC_API_KEY", "not-a-real-key")
os.environ.setdefault("METACULUS_TOKEN", "not-a-real-token")

from forecasting_tools import BinaryQuestion, NumericDistribution, Percentile, PredictedOption, PredictedOptionList

import costs
import forecast_log
import main
import quality

logging.getLogger("main").setLevel(logging.CRITICAL)
FIRST = "First rundown. " * 40
BOTH = FIRST + quality.SECOND_MARK + "Base rate: 3 of 20 years. " * 10


class FakeLlm:
    """Answers with a fixed probability and remembers the prompts it was given."""
    def __init__(self, model, answer):
        self.model, self.answer, self.prompts = model, answer, []

    async def invoke(self, prompt, *a, **kw):
        self.prompts.append(prompt)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return f"Reasoning.\nProbability: {self.answer}%"


async def fake_structure_output(text, output_type, **kw):
    return SimpleNamespace(prediction_in_decimal=float(re.findall(r"Probability: ([0-9.]+)%", text)[-1]) / 100)


class EnsembleTest(unittest.TestCase):
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
        for p in (mock.patch.object(costs, "RECORDER", costs.Recorder()),
                  mock.patch.object(forecast_log, "DIR", Path(self.tmp.name)),
                  mock.patch.object(main, "structure_output", fake_structure_output),
                  mock.patch.object(main.FallTemplateBot2026, "get_llm", get_llm),
                  mock.patch.object(main.FallTemplateBot2026, "plan", quality.plan("full")),
                  mock.patch.object(main.FallTemplateBot2026, "strong_llm", self.strong),
                  mock.patch.object(main.FallTemplateBot2026, "rationales", {}),
                  mock.patch.object(main.FallTemplateBot2026, "_member_index", {})):
            p.start()
            self.addCleanup(p.stop)

    def members(self, research):
        """What the library does for one question: one _make_prediction per member, together."""
        async def go():
            costs._question.set("77")
            self.bot._note_pads.append(await self.bot._initialize_notepad(self.question))
            return await asyncio.gather(*[self.bot._make_prediction(self.question, research) for _ in range(3)],
                                        return_exceptions=True)
        return asyncio.run(go())

    def test_full_tier_two_usual_one_strong_and_a_control_that_is_only_logged(self):
        results = self.members(BOTH)
        self.assertEqual(sorted(r.prediction_value for r in results), [0.3, 0.3, 0.6])       # the control is not among them
        logged = {r["member"]: r for r in main.FallTemplateBot2026.rationales["77"]}
        self.assertEqual(set(logged), {"usual_1", "usual_2", "strong_1", "control"})
        self.assertEqual((logged["control"]["role"], logged["control"]["research"]), ("control", "first"))
        self.assertEqual((logged["usual_1"]["research"], logged["strong_1"]["model"]), ("both", "anthropic/claude-opus-5-5"))
        self.assertEqual(len(self.strong.prompts), 1)
        self.assertEqual(len(self.usual.prompts), 3)                                          # two members and the control
        without = [p for p in self.usual.prompts if "Second research pass" not in p]
        self.assertEqual(len(without), 1)                                                     # the control saw the first pass only
        self.assertIn("Second research pass", self.strong.prompts[0])

    def test_no_control_without_a_second_research_pass(self):
        self.members(FIRST)
        self.assertNotIn("control", {r["member"] for r in main.FallTemplateBot2026.rationales["77"]})

    def test_a_failing_strong_model_falls_back_to_the_usual_one(self):
        self.strong.answer = RuntimeError("model not found")
        results = self.members(BOTH)
        self.assertEqual([r.prediction_value for r in results], [0.3, 0.3, 0.3])
        self.assertIn("strong_1_fallback", {r["member"] for r in main.FallTemplateBot2026.rationales["77"]})

    def test_a_failing_control_never_loses_the_forecast(self):
        real = self.usual.invoke

        async def invoke(prompt, *a, **kw):
            if "Second research pass" not in prompt:
                raise RuntimeError("control down")
            return await real(prompt)
        with mock.patch.object(self.usual, "invoke", invoke):
            results = self.members(BOTH)
        self.assertEqual(sorted(r.prediction_value for r in results), [0.3, 0.3, 0.6])

    def test_base_plan_is_the_old_single_forecast(self):
        with mock.patch.object(main.FallTemplateBot2026, "plan", quality.plan("base")):
            async def go():
                costs._question.set("77")
                self.bot._note_pads.append(await self.bot._initialize_notepad(self.question))
                return await self.bot._make_prediction(self.question, FIRST)
            self.assertEqual(asyncio.run(go()).prediction_value, 0.3)
        self.assertEqual(len(self.usual.prompts), 1)
        self.assertEqual(self.strong.prompts, [])

    def test_aggregation_binary_choice_numeric(self):
        agg = lambda preds, q=self.question: asyncio.run(self.bot._aggregate_predictions(preds, q))
        self.assertAlmostEqual(agg([0.3, 0.3, 0.6]), 0.3)
        self.assertEqual(agg([0.999]), 0.98)                                                  # capped
        lists = [PredictedOptionList(predicted_options=[PredictedOption(option_name="A", probability=a),
                                                        PredictedOption(option_name="B", probability=1 - a)]) for a in (1.0, 0.998)]
        got = {o.option_name: o.probability for o in agg(lists).predicted_options}
        self.assertAlmostEqual(got["B"], 0.005, places=3)                                     # floored
        self.assertAlmostEqual(sum(got.values()), 1.0)
        numeric_q = SimpleNamespace(upper_bound=10.0, lower_bound=0.0, open_upper_bound=True, open_lower_bound=False,
                                    zero_point=None, cdf_size=201, page_url="x")
        def dist(values):
            return NumericDistribution.from_question(
                [Percentile(percentile=p, value=v) for p, v in zip((0.1, 0.2, 0.4, 0.6, 0.8, 0.9), values)], numeric_q)
        low, high = dist([1, 1.5, 2, 2.5, 3, 3.5]), dist([6, 6.5, 7, 7.5, 8, 8.5])
        mixed = agg([low, high], numeric_q).get_cdf()
        expected = [(a.percentile + b.percentile) / 2 for a, b in zip(low.get_cdf(), high.get_cdf())]
        self.assertLess(max(abs(m.percentile - e) for m, e in zip(mixed, expected)), 0.01)


class DiscreteQuestionTest(unittest.TestCase):
    def test_a_discrete_question_is_published_as_the_stated_probabilities(self):
        question = SimpleNamespace(question_type="discrete", lower_bound=1.5, upper_bound=6.5, cdf_size=6, zero_point=None,
                                   open_lower_bound=False, open_upper_bound=True, page_url="https://example.test/q/9")
        bot = main.FallTemplateBot2026(llms={"default": "anthropic/claude-sonnet-5-5", "summarizer": "anthropic/claude-haiku-5-5",
                                             "researcher": "anthropic/claude-haiku-5-5", "parser": "anthropic/claude-haiku-5-5"})
        prompts = []

        async def invoke(prompt):
            prompts.append(prompt)
            return "Reasoning.\nOutcome 2: 55%\nOutcome 3: 33%\nOutcome 4: 9%\nOutcome 5: 2%\nOutcome 6: 0.5%\nOutcome above 6: 0.5%"
        outcomes = bot._discrete_outcomes(question)
        with mock.patch.object(bot, "_invoke_forecaster", invoke):
            got = asyncio.run(bot._discrete_prompt_to_forecast(
                question, "Question text.\nThe last thing you write is your final answer as:\nPercentile 10: XX", outcomes))
        cdf = [p.percentile for p in got.prediction_value.get_cdf()]
        self.assertAlmostEqual(cdf[1], 0.55, places=1)            # 0.69 by the percentile route (question 46618)
        self.assertAlmostEqual(cdf[2] - cdf[1], 0.33, places=1)
        self.assertIn("Outcome above 6: XX%", prompts[0])
        self.assertNotIn("Percentile 10", prompts[0])
        self.assertIsNone(bot._discrete_outcomes(SimpleNamespace(question_type="numeric")))

        async def junk(prompt):
            return "Percentile 10: 2"
        with mock.patch.object(bot, "_invoke_forecaster", junk), self.assertRaises(ValueError):
            asyncio.run(bot._discrete_prompt_to_forecast(question, "x\nThe last thing you write is your final answer as:\n", outcomes))


if __name__ == "__main__":
    unittest.main()


class SecondResearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.recorder = costs.Recorder()
        self.first = main.GeneralLlm(model="anthropic/claude-haiku-5-5", temperature=0.2)
        self.second = FakeLlm("anthropic/claude-haiku-5-5", 0)
        self.bot = main.FallTemplateBot2026(llms={"default": "anthropic/claude-sonnet-5-5", "summarizer": "anthropic/claude-haiku-5-5",
                                                  "researcher": self.first, "parser": "anthropic/claude-haiku-5-5"})
        self.question = SimpleNamespace(id_of_post=55, id_of_question=55, page_url="https://example.test/q/55",
                                        question_text="Will it happen?", resolution_criteria="If it happens.", fine_print="")
        for p in (mock.patch.object(costs, "RECORDER", self.recorder), mock.patch.object(forecast_log, "DIR", Path(self.tmp.name)),
                  mock.patch.object(main.FallTemplateBot2026, "run_budget", None),
                  mock.patch.object(main.FallTemplateBot2026, "plan", quality.plan("full")),
                  mock.patch.object(main.FallTemplateBot2026, "second_researcher", self.second)):
            p.start()
            self.addCleanup(p.stop)

    def research(self, second_reply):
        async def first(prompt, *a, **kw):
            return FIRST

        async def second(prompt, *a, **kw):
            self.second.prompts.append(prompt)
            with_stage = costs._stage.get()
            self.stage_seen = with_stage
            if isinstance(second_reply, BaseException):
                raise second_reply
            return second_reply
        with mock.patch.object(self.first, "invoke", first), mock.patch.object(self.second, "invoke", second):
            return asyncio.run(self.bot.run_research(self.question)), forecast_log.read("55")

    def test_the_second_pass_is_added_under_its_own_heading_and_costed_separately(self):
        text, record = self.research("Base rate: 3 of the last 20 years. " * 8)
        first, second = text.split(quality.SECOND_MARK)
        self.assertEqual(first, FIRST)
        self.assertIn("Base rate: 3 of the last 20", second)
        self.assertEqual(self.stage_seen, "research2")
        self.assertIn(FIRST.strip(), self.second.prompts[0])           # it is told what the first pass found
        self.assertIn("at most 2 web searches", self.second.prompts[0])
        self.assertTrue(record["research_second"])
        self.assertEqual(record["research"], text)

    def test_a_failed_or_empty_second_pass_leaves_the_first_untouched(self):
        for reply in (RuntimeError("search down"), "too short"):
            text, record = self.research(reply)
            self.assertEqual(text, FIRST)
            self.assertFalse(record["research_second"])
            Path(self.tmp.name, "55.json").unlink()

    def test_off_at_the_base_tier(self):
        with mock.patch.object(main.FallTemplateBot2026, "plan", quality.plan("base")):
            text, record = self.research("Base rate. " * 40)
        self.assertEqual(text, FIRST)
        self.assertEqual(self.second.prompts, [])
        self.assertNotIn("research_second", record)


class ShadowCheckTest(unittest.TestCase):
    def test_flags_are_returned_for_the_log_and_failures_are_swallowed(self):
        bot = main.FallTemplateBot2026(llms={"default": "anthropic/claude-sonnet-5-5", "summarizer": "anthropic/claude-haiku-5-5",
                                             "researcher": "anthropic/claude-haiku-5-5", "parser": "anthropic/claude-haiku-5-5"})
        question = SimpleNamespace(question_text="Will it happen by 31 Dec?", resolution_criteria="If it happens.", fine_print="",
                                   page_url="https://example.test/q/5")
        checker = SimpleNamespace()

        async def invoke(prompt):
            checker.prompt = prompt
            return "FLAG DATE: the deadline is 31 December, the reasoning used 30 November"
        checker.invoke = invoke
        with mock.patch.object(main.FallTemplateBot2026, "plan", quality.plan("full")), \
                mock.patch.object(main.FallTemplateBot2026, "checker", checker), \
                mock.patch.object(costs, "RECORDER", costs.Recorder()):
            flags = asyncio.run(bot._shadow_check(question, 0.4, "5"))
            self.assertEqual(flags[0]["kind"], "DATE")
            self.assertIn("The forecast: 0.4", checker.prompt)

            async def broken(prompt):
                raise RuntimeError("down")
            checker.invoke = broken
            self.assertIsNone(asyncio.run(bot._shadow_check(question, 0.4, "5")))
        self.assertIsNone(asyncio.run(bot._shadow_check(question, 0.4, "5")))      # base plan, no checker: off
