"""The quality levers' maths (quality.py) and the lever scoring (evaluate.py). Plain numbers only."""
import math
import unittest
from datetime import datetime, timezone

import evaluate
import quality


class PacingTest(unittest.TestCase):
    def test_month_spend_counts_only_this_month(self):
        by_day = {"2026-10-31": 9.0, "2026-11-01": 1.0, "2026-11-18": 0.5}
        self.assertEqual(quality.month_spent(by_day, datetime(2026, 11, 18, tzinfo=timezone.utc)), 1.5)

    def test_tier_follows_the_straight_line_to_the_months_credit(self):
        mid = datetime(2026, 11, 15, tzinfo=timezone.utc)      # line: 45 * 17 / 30 = 25.50
        self.assertEqual(quality.tier(25.0, mid, 45.0), "full")
        self.assertEqual(quality.tier(26.0, mid, 45.0), "lean")
        self.assertEqual(quality.tier(45.0, mid, 45.0), "base")
        self.assertEqual(quality.tier(44.9, datetime(2026, 11, 30, tzinfo=timezone.utc), 45.0), "full")
        self.assertEqual(quality.days_in_month(datetime(2027, 2, 3)), 28)

    def test_plans(self):
        full, lean, base = quality.plan("full"), quality.plan("lean"), quality.plan("base")
        self.assertEqual([m["name"] for m in full["members"]], ["usual_1", "usual_2", "strong_1"])
        self.assertTrue(full["second_research"] and full["control"])
        self.assertEqual(len(lean["members"]), 3)
        self.assertFalse(lean["second_research"] or lean["control"] or any(m["model"] == "strong" for m in lean["members"]))
        self.assertEqual(len(base["members"]), 1)
        self.assertTrue(all(m["persona"] == "" for m in full["members"]))     # personas ship off

    def test_expected_monthly_spend_fits_the_credit(self):
        # 10 questions a day at the full tier: 10 * 30 * 0.14 = $42, under the $45 credit.
        self.assertLessEqual(10 * 30 * quality.EXPECTED_COST["full"], quality.MONTHLY_CREDIT)


class AggregationTest(unittest.TestCase):
    def test_binary_three_members_is_the_median(self):
        self.assertAlmostEqual(quality.aggregate_binary([0.9, 0.2, 0.3]), 0.3)

    def test_binary_trims_then_averages_in_log_odds(self):
        got = quality.aggregate_binary([0.01, 0.2, 0.5, 0.99])
        self.assertAlmostEqual(got, quality.expit((quality.logit(0.2) + quality.logit(0.5)) / 2))
        self.assertAlmostEqual(quality.aggregate_binary([0.2, 0.5], "median"), 0.35)
        self.assertAlmostEqual(quality.aggregate_binary([0.4]), 0.4)

    def test_cap_and_extremise(self):
        self.assertEqual(quality.cap(0.995, 0.02, 0.98), 0.98)
        self.assertEqual(quality.cap(0.001, 0.02, 0.98), 0.02)
        self.assertEqual(quality.extremise(0.7), 0.7)                 # ships off: exponent 1
        self.assertGreater(quality.extremise(0.7, 1.5), 0.7)

    def test_options_mean_floor_and_total(self):
        got = quality.aggregate_options([{"A": 0.7, "B": 0.3, "C": 0.0}, {"A": 0.5, "B": 0.5, "C": 0.0}], floor=0.01)
        self.assertAlmostEqual(sum(got.values()), 1.0)
        self.assertAlmostEqual(got["C"], 0.01)
        self.assertAlmostEqual(got["A"] / got["B"], 0.6 / 0.4)
        self.assertEqual(quality.floor_and_normalise([0, 0, 0], 0.01), [1 / 3] * 3)

    def test_mixture_keeps_tails_the_median_drops(self):
        narrow, wide, other = [0.0, 0.0, 1.0, 1.0], [0.0, 0.3, 0.6, 0.9], [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(quality.aggregate_cdfs([narrow, wide, other], "median"), narrow)
        mixture = quality.aggregate_cdfs([narrow, wide, other], "mixture")
        self.assertAlmostEqual(mixture[1], 0.1)
        self.assertAlmostEqual(1 - mixture[-1], 0.1 / 3)
        with self.assertRaises(ValueError):
            quality.aggregate_cdfs([[0, 1], [0, 0.5, 1]])


class DiscreteTest(unittest.TestCase):
    """Question 46618: bounds 1.5 to 6.5, six CDF points, open above."""
    outcomes = quality.discrete_outcomes(1.5, 6.5, 6, False, True)
    labels = [label for label, _ in outcomes]
    kinds = [kind for _, kind in outcomes]

    def test_outcomes(self):
        self.assertEqual(self.labels, ["2", "3", "4", "5", "6", "above 6"])
        self.assertEqual(quality.discrete_outcomes(-0.5, 2.5, 4, True, False)[0], ("below 0", "below"))

    def test_the_stated_probabilities_are_what_is_published(self):
        text = ("Draft: Outcome 2: 70%\n...\nOutcome 2: 55%\nOutcome 3: 33%\nOutcome 4: 9%\n"
                "Outcome 5: 2%\nOutcome 6: 0.5%\nOutcome above 6: 0.5%")
        ps = quality.parse_outcome_probabilities(text, self.labels)
        self.assertAlmostEqual(ps[0], 0.55)                           # the last statement wins
        points = quality.outcome_cdf_points(ps, self.kinds, 1.5, 6.5)
        self.assertEqual([v for v, _ in points], [1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        self.assertEqual(points[0][1], 0.0)
        self.assertAlmostEqual(points[1][1], 0.55, places=2)          # the percentile route published 0.69
        self.assertAlmostEqual(points[2][1], 0.88, places=2)
        self.assertLess(points[-1][1], 1.0)                           # something left above the open bound

    def test_the_real_forecast_object_accepts_it(self):
        from forecasting_tools import NumericDistribution, Percentile
        for ps in ([0.55, 0.33, 0.09, 0.02, 0.005, 0.005], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]):
            points = quality.outcome_cdf_points(ps, self.kinds, 1.5, 6.5)
            dist = NumericDistribution(declared_percentiles=[Percentile(value=v, percentile=p) for v, p in points],
                                       open_upper_bound=True, open_lower_bound=False, upper_bound=6.5,
                                       lower_bound=1.5, zero_point=None, cdf_size=6)
            cdf = [p.percentile for p in dist.get_cdf()]
            self.assertEqual(len(cdf), 6)
            self.assertTrue(all(b > a for a, b in zip(cdf, cdf[1:])))
        self.assertAlmostEqual(cdf[1], 0.005, places=2)

    def test_unreadable_answers_are_refused(self):
        self.assertIsNone(quality.parse_outcome_probabilities("Outcome 2: 55%", self.labels))
        bad = "\n".join(f"Outcome {label}: 50%" for label in self.labels)
        self.assertIsNone(quality.parse_outcome_probabilities(bad, self.labels))


class ChecksTest(unittest.TestCase):
    def test_numeric_flags(self):
        grid = [i / 200 for i in range(201)]
        self.assertEqual(quality.numeric_flags(grid, False, False), [])
        spike = [0.0] * 100 + [1.0] * 101
        self.assertTrue(any(f.startswith("narrow") for f in quality.numeric_flags(spike, False, False)))
        self.assertIn("40% below the lower bound", quality.numeric_flags([0.4 + 0.6 * x for x in grid], True, False))

    def test_parse_check(self):
        text = "FLAG DATE: the window is 15-31 December, not all of December\n- FLAG NONSENSE: x\nlooks fine otherwise"
        self.assertEqual(quality.parse_check(text), [{"kind": "DATE", "why": "the window is 15-31 December, not all of December"}])
        self.assertEqual(quality.parse_check("OK"), [])


def member(name, prediction, role="member", cdf=None):
    entry = {"member": name, "role": role, "prediction": prediction, "reasoning": "", "question_id": 1}
    if cdf:
        entry["cdf"] = cdf
    return entry


def binary_record(post, published, control, usual_1, usual_2, strong, check=()):
    return {"status": "forecast", "post_id": post, "question_id": 1, "type": "BinaryQuestion", "prediction": published,
            "title": f"Q{post}", "url": f"https://example.test/{post}", "cost": 0.14,
            "cost_by_stage": {"research2": {"cost": 0.025}, "strong": {"cost": 0.03}, "forecast": {"cost": 0.03, "calls": 2}},
            "quality": {"tier": "full", "members": ["usual_1", "usual_2", "strong_1"], "check": list(check), "flags": []},
            "rationales": [member("usual_1", usual_1), member("usual_2", usual_2), member("strong_1", strong),
                           member("control", control, role="control")]}


class EvaluateTest(unittest.TestCase):
    def test_outcome_from_api(self):
        binary = {"question": {"type": "binary", "resolution": "yes",
                               "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.7, 0.3], "centers": [0.3]}}}}}
        self.assertEqual(evaluate.outcome_from_api(binary), {"kind": "binary", "resolution": 1, "community": 0.3})
        hidden = {"question": {"type": "binary", "resolution": None, "aggregations": {"recency_weighted": {"latest": None}}}}
        self.assertEqual(evaluate.outcome_from_api(hidden), {"kind": "binary", "resolution": None, "community": None})
        numeric = {"question": {"type": "numeric", "resolution": "7.5", "scaling": {"range_min": 6.5, "range_max": 8.5, "zero_point": None},
                                "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.0, 0.5, 1.0]}}}}}
        got = evaluate.outcome_from_api(numeric)
        self.assertAlmostEqual(got["resolution"], 0.5)
        self.assertEqual(got["community"], [0.0, 0.5, 1.0])
        numeric["question"]["resolution"] = "above_upper_bound"
        self.assertEqual(evaluate.outcome_from_api(numeric)["resolution"], "above")
        numeric["question"]["resolution"] = "annulled"
        self.assertIsNone(evaluate.outcome_from_api(numeric)["resolution"])
        choice = {"question": {"type": "multiple_choice", "resolution": "B", "options": ["A", "B"],
                               "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.4, 0.6]}}}}}
        self.assertEqual(evaluate.outcome_from_api(choice)["community"], {"A": 0.4, "B": 0.6})
        self.assertEqual(evaluate.outcome_from_api({})["kind"], None)

    def test_points_are_peer_score_differences(self):
        self.assertAlmostEqual(evaluate.points("binary", 0.8, 1) - evaluate.points("binary", 0.4, 1), 100 * math.log(2))
        self.assertAlmostEqual(evaluate.points("binary", 0.8, 0), 100 * math.log(0.2))
        self.assertAlmostEqual(evaluate.points("choice", {"A": 0.5, "B": 0.5}, "B"), 100 * math.log(0.5))
        cdf = [0.0, 0.1, 0.7, 1.0]                 # three buckets
        self.assertAlmostEqual(evaluate.probability_of_outcome("numeric", cdf, 0.5), 0.6)
        self.assertAlmostEqual(evaluate.probability_of_outcome("numeric", [0.05, 0.5, 0.9], "above"), 0.1)
        self.assertAlmostEqual(evaluate.points("numeric", cdf, 0.5), 50 * math.log(0.6 * 3))

    def test_configurations_and_lever_differences(self):
        record = binary_record(1, published=0.3, control=0.5, usual_1=0.3, usual_2=0.2, strong=0.4)
        configs = evaluate.configurations(record)
        self.assertEqual((configs["control"], configs["usual_1"], configs["strong_1"]), (0.5, 0.3, 0.4))
        self.assertAlmostEqual(configs["agg_all"], 0.3)
        self.assertAlmostEqual(configs["agg_usual"], quality.expit((quality.logit(0.3) + quality.logit(0.2)) / 2))
        diffs = evaluate.lever_differences([record], {"1": {"resolution": 0, "community": 0.25}})
        self.assertAlmostEqual(diffs["second research pass"][0], 100 * (math.log(0.7) - math.log(0.5)))
        self.assertAlmostEqual(diffs["everything together"][0], 100 * (math.log(0.7) - math.log(0.5)))
        self.assertAlmostEqual(diffs["stronger model, alone"][0],
                               100 * (math.log(0.6) - (math.log(0.7) + math.log(0.8)) / 2))
        old = {"status": "forecast", "post_id": 2, "type": "BinaryQuestion", "prediction": 0.2,
               "rationales": [{"prediction": 0.2, "reasoning": ""}]}          # logged before the levers
        self.assertEqual(set(evaluate.configurations(old)) - {"_usual_names", "_strong_names"}, {"published"})
        self.assertEqual(evaluate.lever_differences([old], {"2": {"resolution": 1}})["second research pass"], [])

    def test_numeric_configurations_use_the_logged_cdfs(self):
        a, b = [0.0, 0.2, 1.0], [0.0, 0.6, 1.0]
        record = {"status": "forecast", "post_id": 3, "question_id": 1, "type": "NumericQuestion", "prediction": {},
                  "prediction_cdf": [0.0, 0.4, 1.0],
                  "rationales": [member("usual_1", {}, cdf=a), member("usual_2", {}, cdf=b), member("control", {}, "control", cdf=b)]}
        configs = evaluate.configurations(record)
        self.assertEqual(configs["agg_all"], [0.0, 0.4, 1.0])
        self.assertEqual(configs["published"], [0.0, 0.4, 1.0])
        self.assertTrue(evaluate.covers([i / 10 for i in range(11)], 0.5))
        self.assertFalse(evaluate.covers([0.0] * 5 + [1.0] * 6, 0.9))

    def test_recommend(self):
        self.assertEqual(evaluate.recommend([])["verdict"], "needs 20 more resolutions")
        self.assertEqual(evaluate.recommend([5.0] * 3 + [6.0] * 2)["verdict"], "needs 15 more resolutions")
        self.assertEqual(evaluate.recommend([5.0, 6.0] * 15)["verdict"], "keep")
        self.assertEqual(evaluate.recommend([-5.0, -6.0] * 15)["verdict"], "drop")
        noisy = evaluate.recommend([40.0, -38.0] * 15)             # mean +1, s.d. about 40
        self.assertTrue(noisy["verdict"].startswith("needs 500+"))
        self.assertTrue(evaluate.recommend([10.0, -5.0] * 10)["verdict"].startswith("needs "))

    def test_report_has_a_recommendation_line_for_every_lever(self):
        records = [binary_record(i, 0.3, 0.5, 0.3, 0.2, 0.4, check=[{"kind": "DATE", "why": "x"}] if i == 1 else ())
                   for i in range(1, 4)]
        outcomes = {"1": {"resolution": 0, "community": 0.05}, "2": {"resolution": 1, "community": 0.3},
                    "3": {"resolution": None, "community": None}}
        text = evaluate.report(records, outcomes)
        for name, *_ in evaluate.LEVERS:
            self.assertIn(f"| {name} | ", text)
        self.assertIn("| second research pass | 2 | ", text)
        self.assertIn("needs 18 more resolutions", text)
        self.assertIn("$0.0250", text)                                # the second pass's measured cost
        self.assertIn("ours 30% vs community 5%", text)               # far from the community: listed with the gap
        self.assertIn("DATE: x", text)
        self.assertIn("No community predictions", evaluate.report(records, {}))

    def test_fetch_uses_only_get_and_survives_failures(self):
        class Reply:
            def __init__(self, data): self.data = data
            def raise_for_status(self): pass
            def json(self): return self.data
        seen = []
        def get(url, headers=None, timeout=None):
            seen.append((url, headers))
            if url.endswith("/2/"):
                raise OSError("down")
            return Reply({"question": {"type": "binary", "resolution": "no"}})
        got = evaluate.fetch_outcomes([1, 2], token="t", pause=0, get=get)
        self.assertEqual(got, {"1": {"kind": "binary", "resolution": 0, "community": None}})
        self.assertEqual(seen[0], ("https://www.metaculus.com/api/posts/1/", {"Authorization": "Token t"}))


if __name__ == "__main__":
    unittest.main()
