"""Offline stubbed runs for multiple-choice and numeric questions.

The tournament mixes binary, multiple-choice and numeric questions, and the
binary path is the only one the live dry run exercised. These tests push one
question of each other type through the real forecast pipeline (research,
prompt, parse, aggregate, report) with every LLM call stubbed and every
Metaculus client method set to raise, so a type-specific crash shows up here
instead of on the first tournament run.

Run from the repo root: .venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import os
from datetime import datetime, timezone
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _key in ("OPENROUTER_API_KEY", "METACULUS_TOKEN", "PERPLEXITY_API_KEY"):
    os.environ[_key] = "fake-offline-test"
for _key in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
    os.environ.pop(_key, None)

import main  # noqa: E402
import model_policy  # noqa: E402
from forecasting_tools import (  # noqa: E402
    BinaryPrediction,
    BinaryQuestion,
    BinaryReport,
    DatePercentile,
    DateQuestion,
    DateReport,
    DiscreteQuestion,
    DiscreteReport,
    GeneralLlm,
    MetaculusClient,
    MultipleChoiceQuestion,
    MultipleChoiceReport,
    NumericQuestion,
    NumericReport,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)

MC_OPTIONS = ["0 or 1", "2 or 3", "4 or more"]


def _mc_question():
    return MultipleChoiceQuestion(
        question_text="How many named Atlantic hurricanes will make US landfall in October 2026?",
        options=MC_OPTIONS,
        resolution_criteria="Per the NHC tropical cyclone reports.",
        background_info="Offline test fixture. Not a Metaculus question.",
        page_url="local://mc1",
    )


def _binary_question():
    return BinaryQuestion(
        question_text="Will a named Atlantic hurricane make US landfall in October 2026?",
        resolution_criteria="Per the NHC tropical cyclone reports.",
        background_info="Offline test fixture. Not a Metaculus question.",
        page_url="local://b1",
    )


def _numeric_question():
    return NumericQuestion(
        question_text="What will the US unemployment rate be for November 2026, in percent?",
        unit_of_measure="%",
        lower_bound=2.0,
        upper_bound=10.0,
        open_lower_bound=False,
        open_upper_bound=True,
        resolution_criteria="Per the BLS Employment Situation release.",
        background_info="Offline test fixture. Not a Metaculus question.",
        page_url="local://num1",
    )


def _date_question():
    return DateQuestion(
        question_text="When will the next US federal government shutdown begin?",
        lower_bound=datetime(2026, 10, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2028, 1, 1, tzinfo=timezone.utc),
        open_lower_bound=False,
        open_upper_bound=True,
        resolution_criteria="Per OPM lapse-in-appropriations notices.",
        background_info="Offline test fixture. Not a Metaculus question.",
        page_url="local://date1",
    )


def _discrete_question():
    return DiscreteQuestion(
        question_text="How many Starship launches will occur in Q4 2026?",
        unit_of_measure="launches",
        lower_bound=-0.5,
        upper_bound=10.5,
        open_lower_bound=False,
        open_upper_bound=True,
        cdf_size=12,
        resolution_criteria="Per SpaceX launch announcements.",
        background_info="Offline test fixture. Not a Metaculus question.",
        page_url="local://disc1",
    )


def _forbid(*args, **kwargs):
    raise AssertionError("a Metaculus network call was attempted during a dry run")


class _Stubs:
    """LLM text is a fixed stub; the parser returns whatever the test supplies
    for the requested output type."""

    def __init__(self, mc_probs, percentiles):
        self.mc_probs = mc_probs
        self.percentiles = percentiles
        self.parsed_types = []
        self.prompts = []

        async def invoke(llm_self, prompt, *args, **kwargs):
            self.prompts.append(prompt)
            return "Stub reasoning for a dry run."

        self.invoke = invoke

    async def structure_output(self, *args, **kwargs):
        # main.py calls this positionally for numeric and by keyword for
        # multiple choice, so accept both.
        output_type = kwargs["output_type"] if "output_type" in kwargs else args[1]
        self.parsed_types.append(output_type)
        if output_type is BinaryPrediction:
            return BinaryPrediction(prediction_in_decimal=0.4)
        if output_type is PredictedOptionList:
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=o, probability=p)
                    for o, p in zip(MC_OPTIONS, self.mc_probs)
                ]
            )
        if output_type == list[Percentile]:
            return [Percentile(percentile=p, value=v) for p, v in self.percentiles]
        if output_type == list[DatePercentile]:
            return [
                DatePercentile(percentile=p, value=datetime(2027, m, 15, tzinfo=timezone.utc))
                for p, m in [(0.1, 2), (0.2, 3), (0.4, 5), (0.6, 7), (0.8, 9), (0.9, 10)]
            ]
        raise AssertionError(f"unexpected parse type {output_type!r}")


def _run(questions, mc_probs=(0.5, 0.3, 0.2), percentiles=None):
    if percentiles is None:
        percentiles = [(0.1, 3.9), (0.2, 4.0), (0.4, 4.2), (0.6, 4.3), (0.8, 4.5), (0.9, 4.7)]
    stubs = _Stubs(list(mc_probs), percentiles)
    bot = main.SummerTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=False,
        llms=model_policy.pinned_llms(),
    )
    network_methods = [
        name
        for name in dir(MetaculusClient)
        if not name.startswith("_") and callable(getattr(MetaculusClient, name))
    ]
    with mock.patch.object(GeneralLlm, "invoke", stubs.invoke), mock.patch.object(
        main, "structure_output", stubs.structure_output
    ), mock.patch.object(BinaryReport, "publish_report_to_metaculus", _forbid), mock.patch.object(
        MultipleChoiceReport, "publish_report_to_metaculus", _forbid
    ), mock.patch.object(
        NumericReport, "publish_report_to_metaculus", _forbid
    ), mock.patch.object(
        DateReport, "publish_report_to_metaculus", _forbid
    ), mock.patch.object(
        DiscreteReport, "publish_report_to_metaculus", _forbid
    ), mock.patch.multiple(
        MetaculusClient, **{name: _forbid for name in network_methods}
    ):
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
    return stubs, reports


class MultipleChoiceRunTest(unittest.TestCase):
    def test_mc_question_gets_a_report_whose_probabilities_sum_to_one(self):
        stubs, reports = _run([_mc_question()])
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertNotIsInstance(report, BaseException, repr(report))
        self.assertIsInstance(report, MultipleChoiceReport)
        self.assertTrue(report.explanation.strip())
        probs = {o.option_name: o.probability for o in report.prediction.predicted_options}
        self.assertEqual(sorted(probs), sorted(MC_OPTIONS))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
        self.assertEqual(stubs.parsed_types.count(PredictedOptionList), 3)


class NumericRunTest(unittest.TestCase):
    def test_numeric_question_gets_a_monotone_cdf_inside_the_bounds(self):
        stubs, reports = _run([_numeric_question()])
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertNotIsInstance(report, BaseException, repr(report))
        self.assertIsInstance(report, NumericReport)
        self.assertTrue(report.explanation.strip())
        cdf = [p.percentile for p in report.prediction.get_cdf()]
        self.assertEqual(len(cdf), 201)
        self.assertTrue(all(b >= a for a, b in zip(cdf, cdf[1:])), "cdf is not monotone")
        # Closed lower bound: no mass below 2.0. Open upper bound: some mass may sit above 10.
        self.assertGreaterEqual(cdf[0], 0.0)
        self.assertLessEqual(cdf[-1], 1.0)
        # The stubbed median (~4.25) should land near the middle of the CDF.
        median_value = next(
            p.value for p in report.prediction.get_cdf() if p.percentile >= 0.5
        )
        self.assertTrue(4.0 <= median_value <= 4.5, median_value)
        self.assertEqual(stubs.parsed_types.count(list[Percentile]), 3)

    def test_mixed_batch_returns_one_report_per_question_in_order(self):
        _, reports = _run([_mc_question(), _numeric_question()])
        self.assertEqual(len(reports), 2)
        self.assertIsInstance(reports[0], MultipleChoiceReport, repr(reports[0]))
        self.assertIsInstance(reports[1], NumericReport, repr(reports[1]))


class DateAndDiscreteRunTest(unittest.TestCase):
    def test_date_question_gets_a_report_with_a_median_inside_the_window(self):
        stubs, reports = _run([_date_question()])
        report = reports[0]
        self.assertNotIsInstance(report, BaseException, repr(report))
        self.assertIsInstance(report, DateReport)
        cdf = report.prediction.get_cdf()
        median_ts = next(p.value for p in cdf if p.percentile >= 0.5)
        median = datetime.fromtimestamp(median_ts, tz=timezone.utc)
        # Stubbed 40th/60th percentiles are May 15 and Jul 15, 2027.
        self.assertTrue(
            datetime(2027, 5, 1, tzinfo=timezone.utc) <= median <= datetime(2027, 7, 31, tzinfo=timezone.utc),
            median,
        )
        self.assertEqual(stubs.parsed_types.count(list[DatePercentile]), 3)

    def test_discrete_question_routes_through_the_numeric_path(self):
        stubs, reports = _run(
            [_discrete_question()],
            percentiles=[(0.1, 1.0), (0.2, 2.0), (0.4, 3.0), (0.6, 4.0), (0.8, 5.0), (0.9, 6.0)],
        )
        report = reports[0]
        self.assertNotIsInstance(report, BaseException, repr(report))
        self.assertIsInstance(report, DiscreteReport)
        self.assertEqual(stubs.parsed_types.count(list[Percentile]), 3)



class QuestionTimelineTest(unittest.TestCase):
    """The forecast prompt asks for the time left, so it must carry the close
    and resolution dates. The live 9/24 run showed the model saying the close
    date was not provided."""

    def test_close_and_resolution_dates_reach_every_forecast_prompt(self):
        questions = [_binary_question(), _mc_question(), _numeric_question(), _date_question()]
        for q in questions:
            q.close_time = datetime(2026, 10, 9, tzinfo=timezone.utc)
            q.scheduled_resolution_time = datetime(2026, 12, 31, tzinfo=timezone.utc)
        stubs, reports = _run(questions)
        forecast_prompts = [p for p in stubs.prompts if "Today is" in p]
        # 4 questions x 3 predictions each
        self.assertEqual(len(forecast_prompts), 12)
        for p in forecast_prompts:
            self.assertIn("closes 2026-10-09", p)
            self.assertIn("resolve 2026-12-31", p)

    def test_missing_dates_leave_no_placeholder_text(self):
        q = _mc_question()
        q.close_time = None
        q.scheduled_resolution_time = None
        self.assertEqual(main.question_timeline(q), "")
        stubs, reports = _run([q])
        for p in stubs.prompts:
            self.assertNotIn("closes None", p)
            self.assertNotIn("Forecasting on this question closes", p)
            self.assertNotIn("scheduled to resolve", p)


if __name__ == "__main__":
    unittest.main()
