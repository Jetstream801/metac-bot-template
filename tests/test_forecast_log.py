import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import budget
import forecast_log as fl

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
LONG = "Findings. " * 60


class Q:
    id_of_post = 46481
    id_of_question = 45900
    page_url = "https://www.metaculus.com/questions/46481"
    question_text = "Will X happen?"
    community_prediction_at_access_time = 0.31
    num_forecasters = 12
    close_time = NOW + timedelta(hours=2)


class Model:
    def model_dump(self, mode=None):
        return {"declared_percentiles": [{"percentile": 0.1, "value": 3.0}]}


class ForecastLogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_off_by_default(self):
        self.assertIsNone(fl.DIR)
        self.assertFalse(fl.save_research(Q(), LONG))
        self.assertIsNone(fl.read("46481"))

    def test_research_then_forecast_then_costs(self):
        self.assertTrue(fl.save_research(Q(), LONG, retried=True, directory=self.dir, now=NOW))
        rec = fl.read("46481", self.dir)
        self.assertEqual(rec["status"], "researched")
        self.assertEqual(rec["research"], LONG)
        self.assertTrue(rec["research_retried"])
        self.assertEqual(rec["type"], "Q")
        self.assertEqual(rec["community_prediction_at_access"], 0.31)

        fl.save_forecast(Q(), 0.27, rationales=[{"reasoning": "because"}], directory=self.dir,
                         now=NOW + timedelta(minutes=1))
        fl.add_costs("46481", {"research": {"calls": 1, "searches": 3, "cost": 0.0355}}, self.dir)
        fl.add_costs("46481", {"research": {"calls": 1, "searches": 2, "cost": 0.02},
                               "forecast": {"calls": 1, "cost": 0.0085}}, self.dir)
        rec = fl.read("46481", self.dir)
        self.assertEqual(rec["status"], "forecast")
        self.assertEqual(rec["prediction"], 0.27)
        self.assertEqual(rec["forecast_time"], "2026-11-18T12:01:00+00:00")
        self.assertEqual(rec["research"], LONG)
        self.assertEqual(rec["rationales"][0]["reasoning"], "because")
        self.assertEqual(rec["cost_by_stage"]["research"]["searches"], 5)
        self.assertAlmostEqual(rec["cost"], 0.064)

    def test_costs_not_written_for_unknown_question(self):
        self.assertFalse(fl.add_costs("999", {"research": {"cost": 0.1}}, self.dir))

    def test_prediction_shapes(self):
        self.assertEqual(fl.to_jsonable(Model()), {"declared_percentiles": [{"percentile": 0.1, "value": 3.0}]})
        self.assertEqual(fl.to_jsonable([0.2, Model()])[0], 0.2)
        self.assertIsInstance(fl.to_jsonable(object()), str)

    def test_reuse_only_fresh_real_unforecast_research(self):
        base = {"status": "researched", "research": LONG, "research_time": (NOW - timedelta(hours=1)).isoformat()}
        self.assertEqual(fl.reusable_research(base, now=NOW), LONG)
        self.assertIsNone(fl.reusable_research(None, now=NOW))
        self.assertIsNone(fl.reusable_research({**base, "status": "forecast"}, now=NOW))
        self.assertIsNone(fl.reusable_research({**base, "research": "short"}, now=NOW))
        self.assertIsNone(fl.reusable_research(
            {**base, "research": LONG + "(Note: web research returned almost nothing for this question)"}, now=NOW))
        self.assertIsNone(fl.reusable_research(
            {**base, "research_time": (NOW - timedelta(hours=7)).isoformat()}, now=NOW))
        self.assertIsNone(fl.reusable_research({**base, "research_time": "garbage"}, now=NOW))

    def test_reused_research_keeps_original_text_and_time(self):
        fl.save_research(Q(), LONG, directory=self.dir, now=NOW)
        fl.save_research(Q(), LONG, reused=True, directory=self.dir, now=NOW + timedelta(minutes=20))
        rec = fl.read("46481", self.dir)
        self.assertTrue(rec["research_reused"])
        self.assertEqual(rec["research_time"], NOW.isoformat())


class BudgetTests(unittest.TestCase):
    def test_caps_unchanged(self):
        self.assertEqual((budget.SEASON_CAP, budget.DAILY_CAP), (90.00, 8.00))

    def test_ledger_charges_measured_cost_and_keeps_stage_totals(self):
        path = tempfile.mktemp()
        data = budget.load(path)
        data["stages"] = {"research": {"cost": 0.0355}}
        charged = budget.record(data, 0.044, now=NOW, path=path, minibench_cost=0.044)
        self.assertAlmostEqual(charged, 0.044)
        saved = budget.load(path)
        self.assertAlmostEqual(saved["spent"], 0.044)
        self.assertAlmostEqual(saved["minibench_by_day"]["2026-11-18"], 0.044)
        self.assertEqual(saved["stages"]["research"]["cost"], 0.0355)


if __name__ == "__main__":
    unittest.main()
