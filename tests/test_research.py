"""The research step's search count (main.py): a ceiling, a first search that is not optional,
and a record of how many searches each question used. Fakes only: no network, no API key."""
import asyncio
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("ANTHROPIC_API_KEY", "not-a-real-key")
os.environ.setdefault("METACULUS_TOKEN", "not-a-real-token")

import costs
import forecast_log
import main

logging.getLogger("main").setLevel(logging.CRITICAL)      # the guard's own warnings are not test output
LONG = "Findings. " * 60          # 600 characters: a real write-up


def question(post_id=101):
    return SimpleNamespace(id_of_post=post_id, id_of_question=post_id, page_url=f"https://example.test/q/{post_id}",
                           question_text="Will it happen?", resolution_criteria="Resolves YES if it happens.", fine_print="")


class ResearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recorder = costs.Recorder()
        patches = (mock.patch.object(costs, "RECORDER", self.recorder), mock.patch.object(forecast_log, "DIR", Path(self.tmp.name)),
                   mock.patch.object(main.FallTemplateBot2026, "run_budget", None))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.researcher = main.GeneralLlm(model="anthropic/claude-haiku-5-5", temperature=0.2, tools=main.research_tools())
        self.bot = main.FallTemplateBot2026(llms={"default": "anthropic/claude-sonnet-5-5", "summarizer": "anthropic/claude-haiku-5-5",
                                                  "researcher": self.researcher, "parser": "anthropic/claude-haiku-5-5"})
        self.prompts = []

    def run_research(self, replies, q=None):
        """replies: (text, searches the call ran) for each call the researcher gets, in order."""
        replies = list(replies)

        async def invoke(prompt, *a, **kw):
            self.prompts.append(prompt)
            text, searches = replies.pop(0)
            if isinstance(text, BaseException):
                raise text
            self.recorder.add_call("anthropic/claude-haiku-5-5", {"in": 16000 * searches + 400, "out": 900, "cache_read": 0,
                                                                  "cache_write": 0, "searches": searches})
            return text
        q = q or question()
        with mock.patch.object(self.researcher, "invoke", invoke):
            return asyncio.run(self.bot.run_research(q)), forecast_log.read(forecast_log.question_key(q))

    def test_the_ceiling_stays_at_three_and_the_prompt_says_when_to_stop(self):
        self.assertEqual(main.research_tools(), [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}])
        self.assertEqual(main.research_tools(1)[0]["max_uses"], 3)          # never below 3
        self.assertEqual(main.research_tools(5)[0]["max_uses"], 5)
        prompt = main.FallTemplateBot2026._get_research_prompt(question(), self.researcher)
        for words in ("at most 3 web searches", "Always run the first search", "Search again only to fill a specific gap",
                      "stop searching and write it", "how often things like this have happened before", "Will it happen?"):
            self.assertIn(words, prompt)

    def test_one_search_is_enough_when_the_write_up_is_full_and_the_count_is_recorded(self):
        research, record = self.run_research([(LONG, 1)])
        self.assertEqual((research, len(self.prompts)), (LONG, 1))
        self.assertEqual((record["research_searches"], record["research_retried"]), (1, False))
        self.assertEqual(self.recorder.research_searches(), {"101": 1})

    def test_short_research_is_still_retried_once(self):
        research, record = self.run_research([("Too short.", 3), (LONG, 2)])
        self.assertEqual((research, len(self.prompts)), (LONG, 2))
        self.assertIn("Use at most 2 searches", self.prompts[1])
        self.assertNotIn("without any web search", self.prompts[1])
        self.assertEqual((record["research_searches"], record["research_retried"]), (5, True))
        self.assertEqual(self.recorder.question_stages("101")["research_retry"]["searches"], 2)

    def test_short_research_twice_is_marked_for_the_forecaster(self):
        research, _ = self.run_research([("Too short.", 3), ("Still short.", 2)])
        self.assertIn("web research returned almost nothing", research)

    def test_a_write_up_made_with_no_search_at_all_is_retried_with_a_search_required(self):
        research, record = self.run_research([("From memory. " * 50, 0), (LONG, 1)])
        self.assertEqual(research, LONG)
        self.assertIn("written without any web search. Run at least one search now.", self.prompts[1])
        self.assertEqual((record["research_searches"], record["research_retried"]), (1, True))

    def test_a_full_first_write_up_is_kept_when_that_retry_fails_or_comes_back_thin(self):
        memory = "From memory. " * 50
        for second in ((RuntimeError("boom"), 0), ("Thin.", 1)):
            self.prompts.clear()
            research, _ = self.run_research([(memory, 0), second], q=question(200 + len(str(second[0]))))
            self.assertEqual((research, len(self.prompts)), (memory, 2))


class SearchCountTest(unittest.TestCase):
    def test_searches_per_question_are_summed_printed_and_kept_for_the_season(self):
        r = costs.Recorder()
        use = lambda s: {"in": 1000, "out": 100, "cache_read": 0, "cache_write": 0, "searches": s}  # noqa: E731
        r.add_call("claude-haiku-5-5", use(1), stage="research", question="q1")
        r.add_call("claude-haiku-5-5", use(3), stage="research", question="q2")
        r.add_call("claude-haiku-5-5", use(2), stage="research_retry", question="q2")
        r.add_call("claude-sonnet-5-5", use(0), stage="forecast", question="q3")       # research reused: not counted
        self.assertEqual(r.research_searches(), {"q1": 1, "q2": 5})
        ledger = {"research_search_hist": {"1": 4, "3": 1}}
        costs.merge_season(ledger, r)
        self.assertEqual(ledger["research_search_hist"], {"1": 5, "3": 1, "5": 1})
        self.assertEqual(costs.searches_line({"1": 1, "2": 2}), "1.67 a question (1: 1, 2: 2)")
        self.assertEqual(costs.searches_line({}), "n/a")
        text = costs.summary(r.run_stages(), r.question_count(), ledger["stages"], ledger["stage_questions"],
                             run_searches=r.research_searches(), season_searches=ledger["research_search_hist"])
        self.assertIn("web searches per researched question", text)
        self.assertIn("run 3.00 a question (1: 1, 5: 1)", text)
        self.assertIn("since adaptive search: 1.86 a question (1: 5, 3: 1, 5: 1)", text)
        self.assertNotIn("web searches per", costs.summary({}, 0, {}, 0))             # callers without the counts: as before


if __name__ == "__main__":
    unittest.main()
