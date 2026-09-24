"""Offline checks for the local bot changes: model allow-list, key scrubbing,
publish-off default, and a full stubbed forecast run on the fixture questions.

No network and no spend: every LLM call is patched, and every Metaculus client
method that could reach the site raises if touched.

Run from the repo root: .venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Fake provider keys, set before main's load_dotenv() (which never overrides),
# so a call that slips past the stubs fails auth instead of spending.
for _key in ("OPENROUTER_API_KEY", "METACULUS_TOKEN", "PERPLEXITY_API_KEY"):
    os.environ[_key] = "fake-offline-test"
for _key in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
    os.environ.pop(_key, None)

import main  # noqa: E402  (scrubs the blocked keys at import)
import bot_helpers  # noqa: E402
import model_policy  # noqa: E402
from forecasting_tools import (  # noqa: E402
    BinaryPrediction,
    BinaryReport,
    GeneralLlm,
    MetaculusClient,
)
from local_questions import load_local_questions  # noqa: E402

FIXTURE = ROOT / "fixtures" / "dry_run_questions.json"


class AllowListTest(unittest.TestCase):
    def test_refuses_models_outside_the_allow_list(self):
        for purpose, name in [
            ("default", "anthropic/claude-x"),
            ("default", "gpt-4o"),
            ("default", "openai/gpt-4o"),
            ("researcher", "smart-searcher/openrouter/openai/gpt-5.6-luna"),
            ("default", "asknews/news-summaries"),  # AskNews is researcher-only
            ("parser", ""),
            ("parser", None),
        ]:
            with self.subTest(purpose=purpose, name=name):
                self.assertFalse(model_policy.is_allowed(purpose, name))

    def test_allows_pinned_prefixes_and_researcher_presets(self):
        for purpose, name in [
            ("default", "openrouter/openai/gpt-5.6-terra"),
            ("parser", "metaculus/gpt-4o"),
            ("researcher", "asknews/news-summaries"),
            ("researcher", "no_research"),
            ("researcher", "openrouter/perplexity/sonar"),
        ]:
            with self.subTest(purpose=purpose, name=name):
                self.assertTrue(model_policy.is_allowed(purpose, name))

    def test_assert_allowed_names_every_bad_purpose(self):
        with self.assertRaises(model_policy.DisallowedModelError) as ctx:
            model_policy.assert_allowed(
                {"default": GeneralLlm(model="anthropic/claude-x"), "parser": "gpt-4o"}
            )
        self.assertIn("default='anthropic/claude-x'", str(ctx.exception))
        self.assertIn("parser='gpt-4o'", str(ctx.exception))

    def test_bot_construction_with_a_bad_model_raises(self):
        with self.assertRaises(model_policy.DisallowedModelError):
            main.SummerTemplateBot2026(
                llms={"default": GeneralLlm(model="anthropic/claude-x")}
            )

    def test_partial_llms_are_filled_from_the_pins_not_library_defaults(self):
        bot = main.SummerTemplateBot2026(
            llms={"default": GeneralLlm(model="openrouter/openai/gpt-5.6-luna")}
        )
        names = {p: model_policy.model_name(v) for p, v in bot._llms.items()}
        self.assertEqual(names["default"], "openrouter/openai/gpt-5.6-luna")
        for purpose in ("summarizer", "parser", "researcher"):
            self.assertTrue(
                model_policy.is_allowed(purpose, names[purpose]), f"{purpose}={names[purpose]}"
            )

    def test_env_override_outside_allow_list_is_refused_at_construction(self):
        with mock.patch.dict(os.environ, {"BOT_DEFAULT_MODEL": "anthropic/claude-x"}):
            with self.assertRaises(model_policy.DisallowedModelError):
                main.SummerTemplateBot2026(llms=model_policy.pinned_llms())

    def test_researcher_falls_back_without_asknews_keys(self):
        env = {"BOT_RESEARCHER": "asknews/news-summaries"}
        self.assertEqual(
            model_policy.pinned_model_names(env)["researcher"],
            model_policy.FALLBACK_RESEARCHER,
        )
        env_keys = {"ASKNEWS_CLIENT_ID": "x", "ASKNEWS_SECRET": "y"}
        self.assertEqual(
            model_policy.pinned_model_names(env_keys)["researcher"],
            model_policy.ASKNEWS_RESEARCHER,
        )


class KeyScrubTest(unittest.TestCase):
    def test_scrub_removes_blocked_keys_only(self):
        env = {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "b", "OPENROUTER_API_KEY": "c"}
        removed = model_policy.scrub_blocked_keys(env)
        self.assertEqual(sorted(removed), ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
        self.assertEqual(env, {"OPENROUTER_API_KEY": "c"})

    def test_importing_main_scrubs_the_process_environment(self):
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-fake", OPENAI_API_KEY="sk-fake")
        code = (
            "import os, main; "
            "print(sorted(k for k in ('ANTHROPIC_API_KEY','OPENAI_API_KEY') if k in os.environ))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")

    def test_blocked_keys_are_gone_before_any_llm_library_loads(self):
        # Scrubbing after the import is too late for anything a library reads
        # from the environment while it loads. A probe on the import system
        # records which blocked keys are still set when each library first loads.
        env = dict(os.environ, ANTHROPIC_API_KEY="sk-fake", OPENAI_API_KEY="sk-fake")
        code = textwrap.dedent(
            """
            import importlib.abc, json, os, sys
            WATCH = ("forecasting_tools", "litellm", "openai", "anthropic")
            seen = {}
            class Probe(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    top = name.split(".")[0]
                    if top in WATCH and top not in seen:
                        seen[top] = sorted(
                            k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if k in os.environ
                        )
                    return None
            sys.meta_path.insert(0, Probe())
            import main
            print(json.dumps(seen))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        seen = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertIn("forecasting_tools", seen, "probe never saw forecasting_tools load")
        self.assertEqual({lib: keys for lib, keys in seen.items() if keys}, {})

    def test_blocked_keys_stay_gone_when_a_library_reloads_dotenv(self):
        # litellm calls dotenv.load_dotenv() while it loads, which would put back
        # any blocked key the user keeps in their dotenv file. The fake below
        # stands in for such a file: every load_dotenv() call, from main.py or
        # from a library, restores both keys. They must stay gone during every
        # library load and after the import finishes.
        env = {k: v for k, v in os.environ.items() if k not in model_policy.BLOCKED_ENV_KEYS}
        env.pop("LITELLM_MODE", None)
        code = textwrap.dedent(
            """
            import importlib.abc, json, os, sys
            import dotenv
            BLOCKED = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            callers = []
            def fake_load_dotenv(*args, **kwargs):
                callers.append(sys._getframe(1).f_globals.get("__name__"))
                for key in BLOCKED:
                    os.environ.setdefault(key, "sk-fake")
                return True
            dotenv.load_dotenv = fake_load_dotenv
            WATCH = ("forecasting_tools", "litellm", "openai", "anthropic")
            seen = {}
            class Probe(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    top = name.split(".")[0]
                    if top in WATCH and top not in seen:
                        seen[top] = sorted(k for k in BLOCKED if k in os.environ)
                    return None
            sys.meta_path.insert(0, Probe())
            import main
            after = sorted(k for k in BLOCKED if k in os.environ)
            print(json.dumps({"seen": seen, "after": after, "callers": callers}))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertIn("litellm", result["seen"], "probe never saw litellm load")
        self.assertEqual({lib: keys for lib, keys in result["seen"].items() if keys}, {})
        self.assertEqual(result["after"], [])
        # Only main.py loads the dotenv file; a library reload is switched off.
        self.assertEqual(result["callers"], ["main"])

    def test_blocked_keys_are_gone_even_if_a_library_sets_them_while_loading(self):
        # Covers main.py's second scrub: a library that writes the keys into the
        # environment while it loads, by any route other than load_dotenv, must
        # still leave them gone once main.py has finished importing.
        env = {k: v for k, v in os.environ.items() if k not in model_policy.BLOCKED_ENV_KEYS}
        code = textwrap.dedent(
            """
            import importlib.abc, json, os, sys
            BLOCKED = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            planted = []
            class Plant(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name == "litellm" and not planted:
                        for key in BLOCKED:
                            os.environ[key] = "sk-fake"
                        planted.append(name)
                    return None
            sys.meta_path.insert(0, Plant())
            import main
            after = sorted(k for k in BLOCKED if k in os.environ)
            print(json.dumps({"planted": planted, "after": after}))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        result = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(result["planted"], ["litellm"], "litellm never loaded")
        self.assertEqual(result["after"], [])


class EntryPointPublishTest(unittest.TestCase):
    """Runs main.py as the workflows do and records what it hands the bot.

    The bot's constructor is replaced with one that records the publish setting
    and stops the run, so nothing past construction executes: no Metaculus
    client, no forecasts, no network.
    """

    def publish_setting(self, *argv: str):
        code = textwrap.dedent(
            f"""
            import inspect, json, runpy, sys
            import forecasting_tools
            original = forecasting_tools.ForecastBot.__init__
            class Stop(Exception):
                pass
            def record(self, *args, **kwargs):
                bound = inspect.signature(original).bind(self, *args, **kwargs)
                bound.apply_defaults()
                print(json.dumps(bound.arguments["publish_reports_to_metaculus"]))
                raise Stop
            forecasting_tools.ForecastBot.__init__ = record
            sys.argv = ["main.py", *{list(argv)!r}]
            try:
                runpy.run_path("main.py", run_name="__main__")
            except Stop:
                pass
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_publishes_only_with_the_flag(self):
        for argv, expected in [
            ((), False),
            (("--mode", "test_questions"), False),
            (("--mode", "metaculus_cup"), False),
            (("--publish",), True),
            (("--mode", "test_questions", "--publish"), True),
        ]:
            with self.subTest(argv=argv):
                self.assertIs(self.publish_setting(*argv), expected)

    def test_local_questions_never_publish_even_with_the_flag(self):
        for argv in [("--local-questions", str(FIXTURE)), ("--local-questions", str(FIXTURE), "--publish")]:
            with self.subTest(argv=argv):
                self.assertIs(self.publish_setting(*argv), False)


class EnvironmentCheckTest(unittest.TestCase):
    def test_warns_when_only_blocked_keys_are_set(self):
        # OpenAI/Anthropic keys are scrubbed, so they cannot stand in for OpenRouter.
        env = {"METACULUS_TOKEN": "fake-offline-test", "OPENAI_API_KEY": "sk-fake"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("OPENROUTER_API_KEY", None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bot_helpers.check_environment(strict=False)
        self.assertIn("OPENROUTER_API_KEY", buf.getvalue())
        self.assertNotIn("fall back", buf.getvalue())

    def test_quiet_when_openrouter_key_is_set(self):
        env = {"METACULUS_TOKEN": "fake-offline-test", "OPENROUTER_API_KEY": "fake-offline-test"}
        with mock.patch.dict(os.environ, env):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                bot_helpers.check_environment(strict=False)
        self.assertEqual(buf.getvalue(), "")


class ModeAndPublishTest(unittest.TestCase):
    def test_tournament_mode_targets_the_current_ai_competition(self):
        self.assertEqual(MetaculusClient.CURRENT_AI_COMPETITION_ID, 33121)
        self.assertEqual(MetaculusClient.CURRENT_MINIBENCH_ID, "minibench")

    def test_publish_is_off_unless_asked(self):
        parser = main.build_arg_parser()
        self.assertFalse(parser.parse_args([]).publish)
        self.assertFalse(parser.parse_args(["--mode", "test_questions"]).publish)
        self.assertTrue(parser.parse_args(["--publish"]).publish)

    def test_bot_default_is_publish_off(self):
        bot = main.SummerTemplateBot2026(llms=model_policy.pinned_llms())
        self.assertFalse(bot.publish_reports_to_metaculus)


class AggregationTest(unittest.TestCase):
    def test_binary_aggregate_is_the_median(self):
        value = asyncio.run(
            BinaryReport.aggregate_predictions([0.2, 0.9, 0.4, 0.6, 0.5], question=None)
        )
        self.assertEqual(value, 0.5)


class _Stubs:
    """Patched LLM + parser. Each question gets its own pass through `values`,
    so concurrent questions cannot interleave one another's sequence."""

    def __init__(self, values, questions):
        self.values = list(values)
        self.texts = [q.question_text for q in questions]
        self.per_question = {}
        self.invoke_calls = 0
        self.parse_calls = 0

        async def invoke(llm_self, prompt, *args, **kwargs):
            self.invoke_calls += 1
            idx = next((i for i, t in enumerate(self.texts) if t in str(prompt)), -1)
            return f"Stub reasoning for a dry run. [Q{idx}] Probability: 50%"

        self.invoke = invoke  # plain function, so it binds as GeneralLlm.invoke

    async def structure_output(self, text, output_type, *args, **kwargs):
        assert output_type is BinaryPrediction, output_type
        key = str(text).split("[Q", 1)[1].split("]", 1)[0]
        n = self.per_question.get(key, 0)
        self.per_question[key] = n + 1
        self.parse_calls += 1
        return BinaryPrediction(prediction_in_decimal=self.values[n % len(self.values)])


def _forbid(*args, **kwargs):
    raise AssertionError("a Metaculus network call was attempted during a dry run")


class StubbedRunTest(unittest.TestCase):
    def _run(self, values):
        questions = load_local_questions(FIXTURE)
        stubs = _Stubs(values, questions)
        bot = main.SummerTemplateBot2026(
            research_reports_per_question=1,
            predictions_per_research_report=5,
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
        ), mock.patch.object(
            BinaryReport, "publish_report_to_metaculus", _forbid
        ), mock.patch.multiple(
            MetaculusClient, **{name: _forbid for name in network_methods}
        ):
            reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        return stubs, reports

    def test_every_fixture_question_gets_a_report_with_an_explanation(self):
        stubs, reports = self._run([0.2, 0.9, 0.4, 0.6, 0.5])
        self.assertEqual(len(reports), 3)
        for report in reports:
            self.assertNotIsInstance(report, BaseException, repr(report))
            self.assertIsInstance(report, BinaryReport)
            self.assertTrue(report.explanation.strip())
            self.assertEqual(report.prediction, 0.5)
        self.assertEqual(stubs.parse_calls, 15)

    def test_extreme_predictions_are_clamped(self):
        _, reports = self._run([0.999])
        self.assertTrue(all(r.prediction == 0.99 for r in reports), [r.prediction for r in reports])
        _, reports = self._run([0.0])
        self.assertTrue(all(r.prediction == 0.01 for r in reports), [r.prediction for r in reports])


if __name__ == "__main__":
    unittest.main()
