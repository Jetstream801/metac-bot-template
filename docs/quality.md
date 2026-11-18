# Forecast quality: where the spare credit goes, and how we will know it worked

Written 2026-11-18. Settings are in `quality.py`; the scoring harness is `evaluate.py`.

## The situation in one paragraph

The bot can only forecast the questions Metaculus opens (about 8 to 12 a day across the main
tournament and MiniBench), and at $0.049 a question that uses $12 to $18 of a $45 monthly credit
that expires. Questions are the scarce thing, so the credit should buy score per question. This
change spends about $0.14 a question on the levers with the best outside evidence, keeps a
"control" forecast on every question so each lever can be scored against the old bot on the same
questions, and adds a harness that prints keep / drop / needs-N-more for each lever.

## How the tournament scores us (what the rules reward)

- **Spot peer score at close.** Every question in both tournaments is open for exactly 3 hours
  (40 of 40 main and 40 of 40 MiniBench questions in our run logs), the community prediction is
  hidden until the question closes (`cp_reveal_time` equals `close_time` on all 5 logged
  questions), and only the forecast standing at that moment counts
  ([Scores FAQ](https://www.metaculus.com/help/scores-faq/): "For a spot score, only the prediction
  at a specified time is considered"; evaluated when the community prediction is revealed).
  **So updating forecasts later earns nothing, and nothing is spent on it.** The whole budget goes
  into the one forecast made inside the 3-hour window.
- **Peer score** is 100 times the average difference in log score between us and every other
  forecaster. It is relative and roughly zero-sum, and one confident miss costs far more than a
  confident hit earns.
- **Prize share is proportional to the square of the summed peer score, and nothing if the sum is
  negative** (Scores FAQ, tournament section). Missing a question scores 0, so coverage counts,
  and being better than the average bot on every question compounds.
- Out-of-bounds numeric forecasts are rejected by the API rather than scored
  ([spring results](https://www.lesswrong.com/posts/wZBbDqzfBjYG58CxK/futureeval-spring-results-pros-beat-bots-but-the-gap-is)).

Not read: the participation page (https://www.metaculus.com/futureeval/participate/) and the
resources page (https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/) could not be
opened from the build sandbox. Prize-eligibility details must be checked there (see the January
list).

## What the organisers' analyses say matters, strongest first

| What | Evidence | Source |
|---|---|---|
| The base model | "the most important factor for accurate forecasting is the base model used"; model beats scaffolding. In spring 2026, 8 of 8 high-reasoning variants beat their standard twin (p = 0.004). | [Q2 results](https://forum.effectivealtruism.org/posts/F2stjK9wHSy3HPEC9/q2-ai-benchmark-results-pros-maintain-clear-lead), [Q1 results](https://forum.effectivealtruism.org/posts/mwcWxwdsEMexm98GF/q1-ai-benchmarking-results-human-pros-crush-bots), [spring results](https://www.lesswrong.com/posts/wZBbDqzfBjYG58CxK/futureeval-spring-results-pros-beat-bots-but-the-gap-is) |
| Aggregating several forecasts | Bots that aggregated scored 1,799 points higher (95% CI +1,017 to +2,582), the second-largest effect in the Q2 survey. Metaculus's own best bots forecast 5 times and aggregate. 86% of Fall 2025 winners aggregate. | Q2 results; [11 analyses](https://www.lesswrong.com/posts/a82q6yd8zKpYk56cF/ai-forecasting-in-2026-what-11-analyses-say) |
| Coverage | Ranking is by summed score. One bot would have been top 5 on average score and ranked 21st on 137 of 297 questions. | spring results |
| Research breadth | Number of research sources correlates with score (r = 0.42); no single search provider is reliably best; removing search made Brier about 3.6 times worse in one ablation; looking up similar past questions: 34% of winners, 0% of the rest. | 11 analyses; spring results |
| Capping extremes | Among winners, capping correlates with score (r = +0.48, p = 0.005); a top-8 bot capped "to avoid large point losses from occasional AI overconfidence". | 11 analyses; Q2 results |
| Sanity checks, robust handling of each question type | Bot makers' advice ("Add lots of sanity checks", "robustly test different question types"); multiple choice and numeric are where bots lose most to the pros. | Q2 results |
| Prompt wording | "prompt engineering has less of an impact with stronger models"; "Simplicity beats complexity". Prompts titled "Bayesian" did worse. | Q2 results; spring results; 11 analyses |
| Extremising / recalibration | Contested. Platt scaling helped on pooled bot data (Brier +0.016), but it needs our own resolved forecasts to fit. | 11 analyses |

Caveats the sources give themselves: most of this is correlational and from surveys; ensemble
gains flatten quickly (a team of 2 to 10 bots scored about the same; number of model calls
correlates only weakly with score).

## Weaknesses found in our own runs (the 5 logged forecasts and 5 run logs, out of 506 runs listed, 1 failed)

1. **Discrete questions were published wrong.** Question 46618 (named storms by 28 Nov): the
   forecaster wrote "stays at 2: about 55%, reaches 3: about 33%, 4: about 9%, 5 or more: 3%", then
   had to answer as six percentiles (2, 2, 2, 2, 3, 4). What was published: 69% on 2, 15% on 3,
   6% on 4, and 5% above 6. Six percentiles cannot express a probability per outcome. **Fixed**
   (`DISCRETE_OUTCOME_PROBABILITIES`).
2. **Research did not reach the primary source in 3 of 5.** 46482: "I could not open the live FLI
   tracker page", sources disagreed (9 or 10) exactly on an option boundary, and the forecast
   split 38% / 33% across it. 46619: "could not retrieve La Jolla's live observations or a La
   Jolla-specific prediction". 46483: the all-time low that decides the question "is unclear"
   (36.3 against the question's own 38.29 example).
3. **No base rate in 3 of 5, so the forecaster supplied one from memory.** 46618: research "did
   not find a systematic rate"; the rationale then asserts "about 40-50% of years". 46620: "did
   not find any precedent". 46483: "standard deviation is about 1-1.5 points", unsourced.
   Items 2 and 3 are what the second research pass is aimed at.
4. **Forecasts are round numbers** (20%, 40%, 33%, 94%), and an earlier test found two forecasts
   from the same model and prompt only 2.75 points apart: resampling one model adds little
   diversity. That is why the ensemble adds a different, stronger model instead of five copies.
5. **Questions were missed.** 15 Nov: research ran out of output tokens and question 46463 failed
   (already fixed: larger limit and a retry). 16 Nov 14:43 UTC: "Spending limit reached ... today
   $2.97 of $3.00. Not forecasting" with seven questions open (already fixed by the full-throttle
   change; the daily guard is raised again here so it cannot bind at the new cost).
6. **Gap to the community: not measurable at forecast time.** The community prediction was hidden
   on 5 of 5. `evaluate.py --fetch` reads it after close and lists every binary forecast more than
   1.0 log-odds away, with the gap.
7. No parse failures, no malformed or too-narrow numeric distributions, and no multiple-choice
   mass on impossible options in the sample. Five forecasts is a small sample; the free checks now
   flag these on every question.

## What was built, what ships on, and why

| Lever (setting in `quality.py`) | Ships | Extra cost a question | Why |
|---|---|---|---|
| Ensemble of 3: two forecasts from Sonnet 5.5 and one from Opus 5.5 (`ENSEMBLE_MEMBERS`, `STRONG_MODEL`, `STRONG_MEMBERS`) | ON | about $0.014 + $0.035 | The two best-evidenced levers together. If Opus fails for any reason that member is forecast by Sonnet, so the ensemble never shrinks. |
| Aggregation: binary in log-odds with the highest and lowest dropped (the median for three); multiple choice as the mean; numeric as a mixture of the members' distributions (`BINARY_AGGREGATE`, `NUMERIC_AGGREGATE`) | ON | free | Aggregation itself is well evidenced. The exact method is not, so the alternative (median) is re-scored for free from the logged members. The mixture keeps each member's tails, which is the safe side under a log score. |
| Second research pass: base rate, primary source, strongest case against; at most 2 searches (`SECOND_RESEARCH`) | ON | about $0.025 | Research breadth evidence, and weaknesses 2 and 3 above. It adds a section and never replaces the first rundown; a failure leaves the first untouched. |
| Control forecast: one old-style forecast (first research only, Sonnet), logged, never published (`CONTROL_MEMBER`) | ON | about $0.012 | Measurement only. It is the old bot on the same question, which is what makes every row of the lever table a paired comparison. Turn off once the levers are decided. |
| Cap binary at 2% to 98%; floor each option at 0.5% (`BINARY_MIN/MAX`, `OPTION_FLOOR`) | ON | free | Capping evidence above. Costs 1 point when a 99% call is right, saves 69 when it is wrong. Re-scored uncapped for free. |
| Discrete questions answered as a probability per outcome (`DISCRETE_OUTCOME_PROBABILITIES`) | ON | free | Fixes a demonstrated error (46618). Falls back to percentiles if the answer cannot be read. This is the one prompt change that ships on, and only for discrete questions. |
| Shadow sanity check: dates, units, bounds, criteria read literally, "already resolved?" (`SHADOW_CHECK`) | ON, log only | about $0.001 | It never changes a forecast. Flags are logged and the harness reports whether flagged forecasts score worse; only then may a flag trigger a re-forecast. |
| Free checks on numeric forecasts (too narrow, mass outside bounds) | ON, log only | free | Same. |
| Personas / different prompts per member (`PERSONAS`) | OFF | free | A prompt change with no evidence; the sources say prompts matter little. |
| Extremising or recalibration (`EXTREMISE`) | OFF | free | Contested outside, and it must be fitted to our own resolutions. Revisit at 50+ resolved binaries. |
| Re-forecasting open questions | not built | - | The rules make it worthless: 3-hour window, spot score at close. |
| Prompt caching for the ensemble | not built | would save about $0.01 | Could not be tested without paid calls. A candidate once the levers are settled. |

### Cost and the monthly arithmetic

| | a question |
|---|---|
| Before | $0.049 (research $0.034, one forecast $0.012, the rest $0.003) |
| After, full tier | about **$0.14**: research $0.034 + second pass $0.025 + two Sonnet forecasts $0.029 + Opus $0.035 + control $0.012 + parsing, summary, check $0.005 |
| Lean tier (three Sonnet forecasts, no second pass, Opus or control) | about $0.075 |
| Base tier (the old bot) | $0.049 |

The Opus and second-pass figures are estimates until the first runs; the log measures both by stage.

Questions a day times cost a question times 30:

- 8 a day (measured 18 Nov: $0.39 at $0.049): 8 x $0.14 x 30 = **$34**
- 10 a day: **$42**
- 12 a day (the owner's figure, 3 per 6 hours): $50 at full tier, so pacing moves some to lean and the month lands at about **$45**

**Pacing (`quality.tier`)** holds the month to the credit by rule, not by guess: each run compares
this calendar month's spending in the ledger with a straight line from $0 to $45 (two days'
slack). On or under the line: full. Over it: lean. Credit used up: base. No question is ever
skipped for pacing. The rule scales with `MONTHLY_CREDIT`; nothing else needs changing if the
credit changes.

**Bug-guard caps.** `DAILY_CAP` is raised from $5 to $8, because the arithmetic needs it: the
busiest day seen (16 Nov, about 40 questions) would cost about $5.60 at the full tier, and a guard
that skips questions is the costliest failure. `PER_RUN_CAP` ($1) and `SEASON_CAP` ($90) are
unchanged.

## How results are measured

`python evaluate.py --fetch` (also a step in the review workflow, printed on the run page):

- **Now (proxy):** distance of every configuration from the community prediction once revealed;
  the list of binary forecasts more than 1.0 log-odds away; for numeric, whether the community's
  median sits inside our 10-90% interval. This shows "different", not "better".
- **From 26-27 Nov (real):** for each resolved question, each lever's gain in peer-score points.
  This is exact, not an approximation: the peer score of forecast A minus that of forecast B on
  the same question is 100 x (ln pA - ln pB) whoever else forecast (numeric questions are halved, as Metaculus
  does for continuous scores; that scales the numbers, not which lever wins). Also
  the Brier score on binaries and the share of numeric outcomes inside our 10-90% interval.
- **A recommendation per lever:** `keep` or `drop` only with 20 or more paired resolutions and a
  mean two standard errors from zero; otherwise `needs N more resolutions`, with N computed from
  the spread seen so far. The extra cost a question is printed beside it from the measured stages.

When we will know: forecasts made before this change have no members logged, so they score only
the old bot. Questions forecast from the day this merges that resolve by 26-29 Nov (MiniBench
questions resolve within about ten days) give the first pairs; at 8 to 12 questions a day, 20
pairs should exist in early December and a first keep/drop on the larger effects by mid-December.
Small effects (a few points a question) will need 100 or more resolutions, which means January.

## Before the January season

1. **Owner:** fill in the participation form (three required questions; the same form applies for
   donated LLM credits): https://forms.gle/aQdYMq9Pisrf1v7d8 (from the README).
2. **Owner or main session:** read the rules at https://www.metaculus.com/futureeval/participate/
   for prize eligibility: bot description, whether code must be public or shared, one bot per
   person, no human in the loop. Not readable from the sandbox.
3. Update `forecasting-tools` when the season rotates (the tournament id comes from
   `client.CURRENT_AI_COMPETITION_ID`; `uv.lock` pins the version), and the hard-coded `33121` in
   `question_windows.py` and the URL in `main.py`.
4. Reset the season in `budget.py`: `SEASON_END`, and either reset `spent` in the ledger or raise
   `SEASON_CAP`. It counts from November and would stop the bot mid-season otherwise.
5. Be running on the first day: prize share goes with the square of the summed score.
6. Decide each lever from the December resolutions (the lever table), then turn the control off.
7. Re-check model names and prices in `costs.py`, and whether a stronger model has shipped.
8. Set the repository variable `REVIEW_BOT_ENABLED` to `true` so the weekly review runs unprompted.
