import argparse
import asyncio
import contextvars
import logging
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MonetaryCostManager,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

import budget
import costs
import forecast_log
import quality

# Every model call is recorded by stage (costs.py). The reformatting calls made by
# structure_output are filed under "parse".
costs.install(GeneralLlm)
structure_output = costs.staged("parse", structure_output)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Which ensemble member (quality.plan) the forecast being made right now belongs to.
_member = contextvars.ContextVar("ensemble_member", default=None)
CONTROL = {"name": "control", "model": "usual", "stage": "control", "role": "control", "persona": ""}

# Web searches one research call may run. This is a CEILING (the search tool's max_uses), not a
# target: at $0.01 each the searches were about 60% of a question's cost when all three were
# always used (measured 2026-11-18: 9 searches on 3 questions), so the prompt now tells the model
# to stop as soon as it can write the rundown. Do not set it below 3.
RESEARCH_MAX_SEARCHES = 3
# Facts checked against Anthropic's docs on 2026-11-18 (web search tool; prompt caching):
# - The search tool has no parameter that limits how much page text comes back (only max_uses,
#   allowed_/blocked_domains, user_location). Every result is read again by each later search step
#   of the same call, so tokens grow with the square of the searches: fewer searches is the trim.
# - Prompt caching is not used. The fixed part of the research prompt is about 150 tokens (the
#   minimum that can be cached is 512 for these models) and every forecast prompt is used once.
#   Turning caching on would also make the API write each search result to the cache at 1.25 times
#   the input price; that pays back only from the second search of a call on.


def research_tools(max_searches: int = RESEARCH_MAX_SEARCHES) -> list[dict]:
    """The researcher's tools: Anthropic's built-in web search, at most max_searches a call."""
    return [{"type": "web_search_20250305", "name": "web_search", "max_uses": max(3, int(max_searches))}]


def searches_so_far(key: str) -> int | None:
    """Web searches the research call for this question ran, or None when no call was measured."""
    record = costs.RECORDER.question_stages(key).get("research")
    return record["searches"] if record and record["calls"] else None


class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self._invoke_forecaster(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    ##################################### RESEARCH #####################################

    # Set in the main block: (cost manager, dollars this run may spend). The library's
    # own limit only logs when crossed, so the stop is enforced here, per question.
    run_budget = None
    # question key -> rationales written for it this run (saved to forecast_log/)
    rationales: dict[str, list] = {}
    # Quality levers (quality.py; docs/quality.md). The main block sets these from the month's
    # spending; the defaults here are the plain one-forecast bot, which is what the tests build.
    plan = quality.plan("base")
    strong_llm = None          # the stronger model, for the members whose spec says "strong"
    second_researcher = None   # the model (with web search) for the second research pass
    checker = None             # the cheap model for the shadow sanity check
    _member_index: dict[str, int] = {}

    async def run_research(self, question: MetaculusQuestion) -> str:
        key = forecast_log.question_key(question)
        costs._question.set(key)  # the forecast steps that follow run in this same task
        async with self._concurrency_limiter:
            with costs.stage("research"):
                if self.run_budget is not None:
                    manager, allowed = self.run_budget
                    if manager.current_usage >= allowed:
                        raise RuntimeError(
                            f"Run spending limit reached (${manager.current_usage:.2f} of ${allowed:.2f}); "
                            "skipping this question until a later run."
                        )
                # A question researched in an earlier run whose forecast step then failed:
                # use that research again instead of paying for the searches twice.
                earlier = forecast_log.reusable_research(forecast_log.read(key))
                if earlier:
                    logger.info(f"Reusing research from an earlier run for {question.page_url}")
                    forecast_log.save_research(question, earlier, reused=True)
                    return earlier
                research = ""
                retried = False
                researcher = self.get_llm("researcher")

                prompt = self._get_research_prompt(question, researcher)

                if isinstance(researcher, GeneralLlm):
                    research = await researcher.invoke(prompt)
                elif (
                    researcher == "asknews/news-summaries"
                    or researcher == "asknews/deep-research/low-depth"
                    or researcher == "asknews/deep-research/medium-depth"
                    or researcher == "asknews/deep-research/high-depth"
                ):
                    research = await AskNewsSearcher().call_preconfigured_version(
                        researcher, prompt
                    )
                elif researcher.startswith("smart-searcher"):
                    model_name = researcher.removeprefix("smart-searcher/")
                    searcher = SmartSearcher(
                        model=model_name,
                        temperature=0,
                        num_searches_to_run=2,
                        num_sites_per_search=10,
                        use_advanced_filters=False,
                    )
                    research = await searcher.invoke(prompt)
                elif not researcher or researcher == "None" or researcher == "no_research":
                    research = ""
                else:
                    research = await self.get_llm("researcher", "llm").invoke(prompt)
                # Guard (review of 2026-11-16): one question was forecast on 43 characters of
                # research after the searches used up the reply. Try once more, briefly.
                # Second guard (2026-11-18, with the adaptive search count): the first search is
                # not optional. A write-up made with no search at all is the model's memory, not
                # research, so that is tried once more too.
                short = len(str(research or "").strip()) < 400
                unsearched = bool(getattr(researcher, "litellm_kwargs", {}).get("tools")) and searches_so_far(key) == 0
                if isinstance(researcher, GeneralLlm) and (short or unsearched):
                    logger.warning(f"Research for {question.page_url} came back "
                                   f"{'nearly empty' if short else 'without any web search'}; retrying once")
                    retried = True
                    try:
                        with costs.stage("research_retry"):
                            again = await researcher.invoke(
                                prompt + ("\n\nYour last answer was written without any web search. Run at least one search now. "
                                          if unsearched and not short else "\n\n")
                                + "Use at most 2 searches, then write your findings in under 500 words. "
                                "You must write the findings even if the searches found little."
                            )
                        # A full first write-up is not thrown away for a thinner second one.
                        if short or len(str(again or "").strip()) >= 400:
                            research = again
                    except Exception as exc:
                        logger.warning(f"Research retry failed: {exc}")
                    if len(str(research or "").strip()) < 400:
                        research = (str(research or "").strip() + "\n\n(Note: web research returned almost nothing for this "
                                    "question; rely on base rates and say so.)").strip()
                second = await self._second_research(question, research)
                if second:
                    research = research + quality.SECOND_MARK + second
                logger.info(f"Found Research for URL {question.page_url}:\n{research}")
                forecast_log.save_research(question, research, retried=retried,
                                           searches=costs.RECORDER.research_searches().get(key),
                                           second=bool(second) if self.plan["second_research"] else None)
                return research

    async def _second_research(self, question: MetaculusQuestion, first: str) -> str:
        """The second research pass (quality.SECOND_RESEARCH): the base rate, the primary source
        and the strongest case against what the first pass implies. '' when off or when it fails:
        a question is never lost to this step."""
        if not self.plan["second_research"] or self.second_researcher is None:
            return ""
        try:
            with costs.stage("research2"):
                text = await self.second_researcher.invoke(self._get_second_research_prompt(question, first))
            text = str(text or "").strip()
            return text if len(text) >= 200 else ""
        except Exception as exc:
            logger.warning(f"Second research pass failed for {question.page_url}: {exc}")
            return ""

    @staticmethod
    def _get_second_research_prompt(question: MetaculusQuestion, first: str) -> str:
        return clean_indents(
            f"""
            You are the second research assistant to a superforecaster. Today's date is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.
            A first assistant already wrote the rundown below. Do not repeat it. Your job is what it is missing:
            1. The base rate: how often has this kind of thing happened over a comparable period before? Give counts and years.
               If the question is a number, give its recent history and how far it usually moves over the time left.
            2. The primary source the question resolves on (the page, dataset or official body named in the criteria):
               its latest value or statement and its date, if the first rundown did not have it first-hand.
            3. The strongest evidence AGAINST the outcome the first rundown points to.
            Also say so plainly if anything suggests the question has in effect already been decided.

            You may run at most 2 web searches, each aimed at one of the gaps above with a query the first assistant
            would not have used. Then write under 350 words, with dates and figures. Say plainly what you could not find.
            Do not produce a forecast yourself.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}

            The first rundown:
            {first}
            """
        )

    async def _run_individual_question(self, question: MetaculusQuestion):
        report = await super()._run_individual_question(question)
        # The forecast is published by now; keep our own copy for scoring later.
        try:
            key = forecast_log.question_key(question)
            extra = {"quality": {"tier": self.plan["tier"], "settings": quality.settings(),
                                 "members": [m["name"] for m in self.plan["members"]],
                                 "control": self.plan["control"], "flags": []}}
            if hasattr(report.prediction, "get_cdf"):
                cdf = [round(p.percentile, 6) for p in report.prediction.get_cdf()]
                extra["prediction_cdf"] = cdf
                extra["quality"]["flags"] = quality.numeric_flags(
                    cdf, getattr(question, "open_lower_bound", False), getattr(question, "open_upper_bound", False))
            extra["quality"]["check"] = await self._shadow_check(question, report.prediction, key)
            forecast_log.save_forecast(
                question, report.prediction, errors=report.errors,
                rationales=self.rationales.get(key), extra=extra,
            )
        except Exception as exc:  # record-keeping must never lose a forecast
            logger.warning(f"Could not save the forecast record: {exc}")
        return report

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        with costs.stage("summarise"):
            return await super().summarize_research(question, research)

    async def _shadow_check(self, question: MetaculusQuestion, prediction, key: str) -> list | None:
        """The shadow sanity check (quality.SHADOW_CHECK): a cheap model reads the question, the
        research and the published forecast and flags date, unit, bound and criteria problems and
        "already resolved". The flags are only logged; evaluate.py reports whether flagged
        forecasts score worse, which is the evidence needed before a flag may change a forecast.
        None when off or failed."""
        if not quality.SHADOW_CHECK or self.checker is None or self.plan["tier"] == "base":
            return None
        try:
            record = forecast_log.read(key) or {}
            reasons = [r.get("reasoning") or "" for r in self.rationales.get(key, []) if r.get("role") != "control"]
            prompt = clean_indents(
                f"""
                Check a forecast for mistakes of fact and reading, not of judgement. Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.

                Question: {question.question_text}
                Resolution criteria (read them literally): {question.resolution_criteria}
                {question.fine_print}
                Options or bounds: {getattr(question, "options", None) or (getattr(question, "lower_bound", None), getattr(question, "upper_bound", None), getattr(question, "unit_of_measure", None))}

                Research the forecast was made from:
                {str(record.get("research") or "")[:7000]}

                The forecast: {str(forecast_log.to_jsonable(prediction) if not hasattr(prediction, "declared_percentiles") else [(round(p.percentile, 3), p.value) for p in prediction.declared_percentiles][:: max(1, len(prediction.declared_percentiles) // 12)])[:1500]}

                One forecaster's reasoning:
                {(reasons[0] if reasons else "")[:3500]}

                Reply with one line for each real problem, in exactly this form, or the single word OK:
                FLAG ALREADY_RESOLVED: the research shows the outcome is already decided and the forecast does not reflect it
                FLAG DATE: a date or deadline was misread (wrong window, wrong year, time left miscounted)
                FLAG UNITS: the forecast is in the wrong units or scale
                FLAG BOUNDS: probability sits on values the criteria or the bounds make impossible
                FLAG CRITERIA: the reasoning answers a different question from the one the criteria ask
                FLAG IMPOSSIBLE_OPTION: an option that can no longer happen has substantial probability
                After the colon, say what is wrong in one sentence. Flag only what the text above shows.
                """
            )
            with costs.stage("check", key):
                reply = await self.checker.invoke(prompt)
            return quality.parse_check(reply)
        except Exception as exc:  # a failed check must never lose a forecast
            logger.warning(f"Shadow check failed for {question.page_url}: {exc}")
            return None

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        """One ensemble member's forecast. The library calls this once per member (as many times
        as quality.plan has members); each call takes the next member's spec. The first call also
        makes the control forecast (old style: first research only), which is logged, not used."""
        if _member.get() is not None:      # a sub-question of a conditional: same member
            return await self._member_prediction(question, research, _member.get())
        key = str(costs._question.get())
        index = self._member_index[key] = self._member_index.get(key, -1) + 1
        members = self.plan["members"]
        tasks = [self._member_prediction(question, research, members[index % len(members)])]
        if index == 0 and self.plan["control"] and quality.SECOND_MARK in research:
            tasks.append(self._member_prediction(question, research.split(quality.SECOND_MARK)[0], CONTROL))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if len(results) > 1 and isinstance(results[1], BaseException):
            logger.warning(f"Control forecast failed for {question.page_url}: {results[1]}")
        if isinstance(results[0], BaseException):
            raise results[0]
        return results[0]

    async def _member_prediction(self, question: MetaculusQuestion, research: str, spec: dict):
        token = _member.set(spec)
        try:
            try:
                with costs.stage(spec["stage"]):
                    prediction = await ForecastBot._make_prediction(self, question, research)
            except Exception as exc:
                if spec["model"] != "strong":
                    raise
                # The stronger model failed (outage, refusal, a parameter it rejects): this member
                # forecasts with the usual model instead, so the ensemble keeps its size.
                logger.warning(f"Strong model failed for {question.page_url} ({exc}); using the usual model for this member")
                spec = {**spec, "name": spec["name"] + "_fallback", "model": "usual", "stage": "forecast"}
                _member.set(spec)
                with costs.stage("forecast"):
                    prediction = await ForecastBot._make_prediction(self, question, research)
        finally:
            _member.reset(token)
        try:
            value = prediction.prediction_value
            entry = {
                "question_id": getattr(question, "id_of_question", None),
                "prediction": forecast_log.to_jsonable(value),
                "reasoning": prediction.reasoning,
                "member": spec["name"], "role": spec["role"],
                "model": str(getattr(self._forecaster(spec), "model", "")),
                "research": "both" if quality.SECOND_MARK in research else "first",
            }
            if hasattr(value, "get_cdf"):
                entry["cdf"] = [round(p.percentile, 6) for p in value.get_cdf()]
            self.rationales.setdefault(str(costs._question.get()), []).append(entry)
        except Exception as exc:  # record-keeping must never lose a forecast
            logger.warning(f"Could not keep the rationale: {exc}")
        return prediction

    def _forecaster(self, spec: dict | None = None) -> GeneralLlm:
        spec = spec if spec is not None else _member.get()
        if spec and spec.get("model") == "strong" and self.strong_llm is not None:
            return self.strong_llm
        return self.get_llm("default", "llm")

    async def _invoke_forecaster(self, prompt: str) -> str:
        """The forecasting call for the current member: its model, and its persona when
        quality.PERSONAS is on (off: no evidence that personas help)."""
        spec = _member.get() or {}
        if spec.get("persona"):
            prompt = prompt + "\n\n" + spec["persona"]
        return await self._forecaster(spec).invoke(prompt)

    async def _aggregate_predictions(self, predictions, question: MetaculusQuestion):
        """Combine the members (quality.py): binary in log-odds with the extremes dropped, then
        capped; multiple choice as the mean with a floor; numeric as a mixture of the members'
        distributions. Anything else, or any failure here, falls back to the library's way."""
        try:
            if predictions and all(isinstance(p, (int, float)) for p in predictions):
                return quality.cap(quality.extremise(quality.aggregate_binary(predictions)))
            if predictions and all(isinstance(p, PredictedOptionList) for p in predictions):
                members = [{o.option_name: o.probability for o in p.predicted_options} for p in predictions]
                if all(set(m) == set(members[0]) for m in members):
                    return PredictedOptionList(predicted_options=[
                        PredictedOption(option_name=name, probability=p)
                        for name, p in quality.aggregate_options(members).items()])
            if (len(predictions) > 1 and quality.NUMERIC_AGGREGATE == "mixture"
                    and all(isinstance(p, NumericDistribution) for p in predictions)):
                cdfs = [p.get_cdf() for p in predictions]
                ys = quality.aggregate_cdfs([[c.percentile for c in cdf] for cdf in cdfs], "mixture")
                return NumericDistribution.from_question(
                    [Percentile(value=c.value, percentile=y) for c, y in zip(cdfs[0], ys)], question)
        except Exception as exc:
            logger.warning(f"Custom aggregation failed for {question.page_url} ({exc}); using the library's")
        return await super()._aggregate_predictions(predictions, question)

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            return question.question_text

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            Today's date is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.
            Search the web for the most recent, relevant information, then write a concise but detailed rundown:
            the current state of things (with dates and figures), anything scheduled before the question closes,
            how often things like this have happened before, and how the question would resolve if nothing changed.
            Say plainly when you could not find something. Do not produce a forecast yourself.

            Searching: you may run at most {RESEARCH_MAX_SEARCHES} web searches, and each one costs money.
            Always run the first search. After each search, check what you have against the four points above.
            Search again only to fill a specific gap that matters for the forecast (a missing current figure,
            an unconfirmed date, no base rate), with a different query aimed at that gap. Never search again for
            something you already have. As soon as you can write the rundown, stop searching and write it.

            Question:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        outcomes = self._discrete_outcomes(question)
        if outcomes:
            try:
                return await self._discrete_prompt_to_forecast(question, prompt, outcomes)
            except Exception as exc:
                logger.warning(f"Outcome-probability answer failed for {question.page_url} ({exc}); asking for percentiles instead")
        return await self._numeric_prompt_to_forecast(question, prompt)

    @staticmethod
    def _discrete_outcomes(question: NumericQuestion) -> list | None:
        """The outcomes of a discrete question, or None when it is not one (or the lever is off)."""
        if not quality.DISCRETE_OUTCOME_PROBABILITIES or getattr(question, "question_type", "") != "discrete":
            return None
        size = getattr(question, "cdf_size", None)
        if not size or not 3 <= size <= 41:
            return None
        return quality.discrete_outcomes(question.lower_bound, question.upper_bound, size,
                                         question.open_lower_bound, question.open_upper_bound)

    async def _discrete_prompt_to_forecast(self, question: NumericQuestion, prompt: str, outcomes: list):
        """A discrete question answered as one probability per outcome (quality.py).

        Why: six percentiles cannot say "55% on 2, 33% on 3". On question 46618 the forecaster's
        reasoning said exactly that and the percentile answer (2, 2, 2, 2, 3, 4) was published as
        69% on 2 and 15% on 3. Here the stated probabilities are what gets published."""
        labels = [label for label, _ in outcomes]
        marker = "The last thing you write is your final answer as:"
        if marker not in prompt:
            raise ValueError("numeric prompt has no final-answer block to replace")
        lines = "\n".join(f"Outcome {label}: XX%" for label in labels)
        prompt = prompt.split(marker)[0] + (
            "This question has only these possible outcomes: " + ", ".join(labels) + ".\n"
            "The last thing you write is the probability of each outcome, one per line, exactly in this form, "
            "adding up to 100% (leave a little on every outcome that is still possible):\n" + lines + "\n"
        )
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        probabilities = quality.parse_outcome_probabilities(reasoning, labels)
        if probabilities is None:
            raise ValueError("could not read a probability for every outcome")
        points = quality.outcome_cdf_points(probabilities, [kind for _, kind in outcomes],
                                            question.lower_bound, question.upper_bound)
        prediction = NumericDistribution.from_question(
            [Percentile(value=value, percentile=p) for value, p in points], question)
        prediction.get_cdf()      # raises here, not at publishing, if the distribution is not valid
        logger.info(f"Forecasted URL {question.page_url} with outcome probabilities: {dict(zip(labels, probabilities))}.")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self._invoke_forecaster(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. The `llms=` block below is commented out to use
    # whichever default models forecasting-tools picks based on your env vars;
    # uncomment and edit to pin specific models.
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # set below from quality.plan: one forecast per ensemble member
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        # Pinned to Claude models through the owner's Anthropic key. The cheap model
        # does the summarising and reformatting; the stronger one does the forecasting.
        llms={
            "default": GeneralLlm(
                model="anthropic/claude-sonnet-5-5",
                temperature=0.3,
                timeout=90,
                allowed_tries=2,
            ),
            "summarizer": "anthropic/claude-haiku-5-5",
            # Research uses the cheap model with Anthropic's built-in web search.
            "researcher": GeneralLlm(
                model="anthropic/claude-haiku-5-5",
                temperature=0.2,
                timeout=240,
                allowed_tries=2,
                max_tokens=6000,  # 2500 was sometimes used up by searching before any answer was written
                tools=research_tools(),  # up to RESEARCH_MAX_SEARCHES searches; the prompt says when to stop
            ),
            "parser": "anthropic/claude-haiku-5-5",
        },
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    if run_mode == "tournament":
        import question_windows

        question_windows.report()

    # Spending limits (see budget.py): stop here if the season or daily allowance is used up.
    ledger = budget.load()
    run_allowance = budget.allowance(ledger)
    if run_allowance <= 0:
        print(
            f"Spending limit reached (season ${ledger['spent']:.2f} of ${budget.SEASON_CAP:.2f}, "
            f"today ${ledger['by_day'].get(budget.today(), 0.0):.2f} of ${budget.DAILY_CAP:.2f}). Not forecasting."
        )
        raise SystemExit(0)
    # Quality tier for this run (quality.py): how much of each question's extras this month's
    # credit can still afford. No question is skipped for pacing; only the extras are dropped.
    now_utc = datetime.now(timezone.utc)
    spent_this_month = quality.month_spent(ledger["by_day"], now_utc)
    plan = quality.plan(quality.tier(spent_this_month, now_utc))
    FallTemplateBot2026.plan = plan
    template_bot.predictions_per_research_report = len(plan["members"])
    if any(m["model"] == "strong" for m in plan["members"]):
        # No temperature: the stronger model always thinks before answering and sets its own.
        FallTemplateBot2026.strong_llm = GeneralLlm(model=quality.STRONG_MODEL, timeout=180, allowed_tries=2, max_tokens=8000)
    if plan["second_research"]:
        FallTemplateBot2026.second_researcher = GeneralLlm(
            model="anthropic/claude-haiku-5-5", temperature=0.2, timeout=180, allowed_tries=1,
            max_tokens=6000, tools=research_tools())   # the prompt allows 2 searches; the tool ceiling stays 3
    if quality.SHADOW_CHECK:
        FallTemplateBot2026.checker = GeneralLlm(model="anthropic/claude-haiku-5-5", temperature=0, timeout=60,
                                                 allowed_tries=1, max_tokens=1500)
    print(f"[quality] tier {plan['tier']} (this month ${spent_this_month:.2f} of ${quality.MONTHLY_CREDIT:.2f}): "
          f"members {', '.join(m['name'] for m in plan['members'])}; second research {'on' if plan['second_research'] else 'off'}; "
          f"control {'on' if plan['control'] else 'off'}; expected about ${quality.EXPECTED_COST[plan['tier']]:.3f} a question")
    forecast_reports = []
    minibench_usage = 0.0
    # The meter is costs.RECORDER: measured dollars from each call's usage figures. The
    # library's own manager stays on because the library expects one, but it is not used
    # for the limits: it adds every call twice (once directly, once from a late callback).
    meter = costs.RECORDER
    meter.season_stages = ledger.get("stages", {})
    cost_manager = MonetaryCostManager(run_allowance * 6)
    FallTemplateBot2026.run_budget = (meter, run_allowance)
    if run_mode == "tournament":
        forecast_log.DIR = forecast_log.Path(__file__).parent / "forecast_log"
    try:
        with cost_manager:
            client = MetaculusClient()
            if run_mode == "tournament":
                seasonal_tournament_reports = asyncio.run(
                    template_bot.forecast_on_tournament(
                        client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                    )
                )
                # MiniBench only gets its paced daily share (see budget.py); the main
                # tournament above always comes first.
                main_usage = meter.current_usage
                mb_room = budget.minibench_room(ledger)
                minibench_reports = []
                if mb_room > 0:
                    FallTemplateBot2026.run_budget = (meter, min(run_allowance, main_usage + mb_room / budget.PADDING))
                    minibench_reports = asyncio.run(
                        template_bot.forecast_on_tournament(
                            client.CURRENT_MINIBENCH_ID, return_exceptions=True
                        )
                    )
                else:
                    print("MiniBench skipped this run: its paced share for today is used (see budget.py).")
                minibench_usage = meter.current_usage - main_usage
                forecast_reports = seasonal_tournament_reports + minibench_reports
            elif run_mode == "metaculus_cup":
                # The Metaculus Cup may be uninitialized near the start of a season
                # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
                # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
                # targets here.
                template_bot.skip_previously_forecasted_questions = False
                forecast_reports = asyncio.run(
                    template_bot.forecast_on_tournament(
                        client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                    )
                )
            elif run_mode == "test_questions":
                # The bot-testing-area tournament contains all question types and is
                # the recommended target for smoke-testing your bot.
                # https://www.metaculus.com/tournament/bot-testing-area/
                template_bot.skip_previously_forecasted_questions = False
                forecast_reports = asyncio.run(
                    template_bot.forecast_on_tournament(
                        "bot-testing-area", return_exceptions=True
                    )
                )
    finally:
        costs.merge_season(ledger, meter)
        recorded = budget.record(ledger, meter.current_usage, minibench_cost=minibench_usage)
        print(f"Recorded ${recorded:.2f} for this run; season total ${ledger['spent']:.2f} of ${budget.SEASON_CAP:.2f}.")
        for key, stages in meter.by_question.items():
            forecast_log.add_costs(key, stages)
        print(costs.summary(meter.run_stages(), meter.question_count(), ledger["stages"],
                            ledger["stage_questions"], ledger.get("stage_since", ""),
                            run_searches=meter.research_searches(), season_searches=ledger.get("research_search_hist")))
        if meter.unpriced:
            print(f"[cost] no price set in costs.py for {sorted(meter.unpriced)}; charged at the library's figure")
        if meter.library_cost and abs(meter.current_usage - meter.library_cost) > 0.1 * meter.library_cost:
            print(f"[cost] check prices in costs.py: measured ${meter.current_usage:.4f}, library's figure ${meter.library_cost:.4f}")

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
