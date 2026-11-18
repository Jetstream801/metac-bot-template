import asyncio
import unittest

import costs


def usage(prompt=0, out=0, cache_read=0, cache_write=0, searches=None):
    u = {"prompt_tokens": prompt, "completion_tokens": out,
         "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write}
    if searches is not None:
        u["server_tool_use"] = {"web_search_requests": searches}
    return {"usage": u}


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class UsageTests(unittest.TestCase):
    def test_reads_dict_and_object_responses(self):
        want = {"in": 700, "out": 50, "cache_read": 200, "cache_write": 100, "searches": 3}
        self.assertEqual(costs.usage_from_response(usage(1000, 50, 200, 100, 3)), want)
        obj = Obj(usage=Obj(prompt_tokens=1000, completion_tokens=50, cache_read_input_tokens=200,
                            cache_creation_input_tokens=100, server_tool_use=Obj(web_search_requests=3)))
        self.assertEqual(costs.usage_from_response(obj), want)

    def test_missing_usage_is_zero(self):
        self.assertEqual(costs.usage_from_response(Obj()),
                         {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "searches": 0})
        self.assertEqual(costs.usage_from_response(usage(10, 5))["searches"], 0)


class PriceTests(unittest.TestCase):
    def test_research_call_tokens_plus_search_fee(self):
        u = costs.usage_from_response(usage(50_000, 1_000, searches=3))
        # 50k in at $0.10/M + 1k out at $0.50/M + 3 searches at $0.01
        self.assertAlmostEqual(costs.call_cost("anthropic/claude-haiku-5-5", u), 0.005 + 0.0005 + 0.03)

    def test_forecast_call(self):
        u = costs.usage_from_response(usage(2_000, 450))
        self.assertAlmostEqual(costs.call_cost("anthropic/claude-sonnet-5-5", u), 0.004 + 0.0045)

    def test_cache_tokens_priced_separately(self):
        u = costs.usage_from_response(usage(10_000, 0, cache_read=8_000, cache_write=1_000))
        self.assertAlmostEqual(costs.call_cost("claude-sonnet-5-5", u),
                               (1_000 * 2.0 + 8_000 * 0.10 + 1_000 * 2.50) / 1e6)

    def test_haiku_long_prompt_tier(self):
        u = costs.usage_from_response(usage(120_000, 1_000))
        self.assertAlmostEqual(costs.call_cost("claude-haiku-5-5", u), 0.06 + 0.0025)

    def test_unknown_model_and_dated_name(self):
        u = costs.usage_from_response(usage(1000, 10))
        self.assertIsNone(costs.call_cost("openai/gpt-4o", u))
        self.assertIsNotNone(costs.call_cost("anthropic/claude-haiku-5-5-20261001", u))


class RecorderTests(unittest.TestCase):
    def test_totals_by_stage_and_question(self):
        r = costs.Recorder()
        res = costs.usage_from_response(usage(50_000, 1_000, searches=3))
        fc = costs.usage_from_response(usage(2_000, 450))
        r.add_call("claude-haiku-5-5", res, stage="research", question="1")
        r.add_call("claude-sonnet-5-5", fc, stage="forecast", question="1")
        r.add_call("claude-haiku-5-5", res, stage="research", question="2")
        stages = r.run_stages()
        self.assertEqual(stages["research"]["calls"], 2)
        self.assertEqual(stages["research"]["searches"], 6)
        self.assertAlmostEqual(r.current_usage, 2 * 0.0355 + 0.0085)
        self.assertEqual(r.question_count(), 2)
        self.assertAlmostEqual(r.question_stages("1")["forecast"]["cost"], 0.0085)

    def test_unpriced_model_charged_at_library_figure(self):
        r = costs.Recorder()
        r.add_call("openai/gpt-4o", costs.usage_from_response(usage(10, 10)), library_cost=0.02,
                   stage="parse", question="1")
        self.assertAlmostEqual(r.current_usage, 0.02)
        self.assertEqual(r.unpriced, {"openai/gpt-4o"})

    def test_timeout_charged_at_stage_average(self):
        r = costs.Recorder()
        r.add_call("claude-haiku-5-5", costs.usage_from_response(usage(50_000, 1_000, searches=3)),
                   stage="research", question="1")
        self.assertAlmostEqual(r.add_timeout(stage="research", question="2"), 0.0355)
        self.assertEqual(r.run_stages()["research"]["timeouts"], 1)

    def test_timeout_uses_season_average_then_fallback(self):
        r = costs.Recorder()
        r.season_stages = {"forecast": {**costs.empty_stage(), "calls": 4, "cost": 0.04}}
        self.assertAlmostEqual(r.add_timeout(stage="forecast", question="1"), 0.01)
        self.assertAlmostEqual(r.add_timeout(stage="research", question="1"), costs.TIMEOUT_FALLBACK["research"])


class FakeTimeout(Exception):
    pass


class FakeLlm:
    model = "anthropic/claude-sonnet-5-5"

    async def _call_litellm(self, prompt):
        if prompt == "slow":
            raise FakeTimeout()
        if prompt == "bad":
            raise ValueError("rejected")
        return Obj(usage=Obj(prompt_tokens=1000, completion_tokens=100), _hidden_params={"response_cost": 0.003})


class InstallTests(unittest.TestCase):
    def test_calls_recorded_once_under_current_stage(self):
        rec = costs.Recorder()

        class Llm(FakeLlm):
            pass

        costs.install(Llm, rec)
        costs.install(Llm, rec)  # second install must not double-wrap

        async def go():
            with costs.stage("research", "7"):
                await Llm()._call_litellm("x")
                with costs.stage("research_retry"):
                    await Llm()._call_litellm("x")
            await costs.staged("parse", Llm()._call_litellm)("x")
            with costs.stage("forecast", "7"):
                for bad in ("slow", "bad"):
                    try:
                        await Llm()._call_litellm(bad)
                    except (FakeTimeout, ValueError):
                        pass

        asyncio.run(go())
        q7 = rec.question_stages("7")
        self.assertEqual(q7["research"]["calls"], 1)
        self.assertEqual(q7["research_retry"]["calls"], 1)
        self.assertAlmostEqual(q7["research"]["cost"], 0.003)
        self.assertEqual(q7["forecast"]["timeouts"], 1)   # the timeout is charged,
        self.assertEqual(q7["forecast"]["calls"], 0)      # the rejected call is not
        self.assertEqual(rec.question_stages(None)["parse"]["calls"], 1)
        self.assertAlmostEqual(rec.library_cost, 0.009)


class SeasonTests(unittest.TestCase):
    def test_merge_and_summary(self):
        rec = costs.Recorder()
        rec.add_call("claude-haiku-5-5", costs.usage_from_response(usage(50_000, 1_000, searches=3)),
                     stage="research", question="1")
        rec.add_call("claude-sonnet-5-5", costs.usage_from_response(usage(2_000, 450)),
                     stage="forecast", question="1")
        ledger = {"stages": {"research": {**costs.empty_stage(), "calls": 1, "cost": 0.0355}},
                  "stage_questions": 1, "stage_since": "2026-11-18"}
        costs.merge_season(ledger, rec)
        self.assertEqual(ledger["stage_questions"], 2)
        self.assertAlmostEqual(ledger["stages"]["research"]["cost"], 0.071)
        self.assertEqual(ledger["stage_since"], "2026-11-18")
        text = costs.summary(rec.run_stages(), 1, ledger["stages"], 2, "2026-11-18")
        self.assertIn("research", text)
        self.assertIn("81%", text)            # research share of this run: 0.0355 / 0.044
        self.assertIn("$0.0440 a question", text)
        self.assertLessEqual(len(text.splitlines()), 8)

    def test_summary_with_nothing_spent(self):
        self.assertIn("n/a", costs.summary({}, 0, {}, 0))


if __name__ == "__main__":
    unittest.main()
