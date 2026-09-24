"""Every GitHub workflow that runs main.py must actually submit.

main.py publishes only when run with --publish (off by default so a local run
can never send by accident; tests/test_policy.py covers that default). That
makes the workflows the one place the flag has to be present: a tournament
workflow without it runs every 20 minutes, spends model credits, prints
forecasts, and submits nothing, which looks healthy in the Actions log while
scoring zero.

test_bot.yaml publishes too. It targets the bot-testing-area sandbox, and its
whole job (README step 4) is to prove the token can post before the live
tournament does: a dry-run smoke test proves nothing and leaves the bot's
profile empty.

Run from the repo root: .venv/bin/python -m unittest discover -s tests -v
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
BLOCKED_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def workflow_files() -> list[Path]:
    # GitHub runs both spellings, so both are checked.
    return sorted([*WORKFLOWS.glob("*.yaml"), *WORKFLOWS.glob("*.yml")])

MUST_PUBLISH = ("run_bot_on_tournament.yaml", "run_bot_on_metaculus_cup.yaml", "test_bot.yaml")


def main_py_invocations(workflow: str) -> list[str]:
    text = (WORKFLOWS / workflow).read_text()
    return [line.strip() for line in text.splitlines() if re.search(r"\bmain\.py\b", line)]


class WorkflowPublishTest(unittest.TestCase):
    def test_workflows_pass_publish(self):
        for workflow in MUST_PUBLISH:
            with self.subTest(workflow=workflow):
                runs = main_py_invocations(workflow)
                self.assertTrue(runs, f"{workflow} no longer runs main.py")
                for line in runs:
                    self.assertIn("--publish", line.split(), f"{workflow}: {line}")

    def test_no_workflow_passes_openai_or_anthropic_keys(self):
        # The bot strips these at startup anyway; not passing them at all means
        # a key added to the repo secrets by mistake never reaches the process.
        # The name is checked anywhere in the file and in any case, not just as
        # secrets.KEY, so secrets['KEY'] and secrets.openai_api_key are caught too
        # (GitHub secret names are case-insensitive).
        for path in workflow_files():
            text = path.read_text().lower()
            for key in BLOCKED_KEYS:
                with self.subTest(workflow=path.name, key=key):
                    self.assertNotIn(key.lower(), text)

    def test_env_template_does_not_ask_for_openai_or_anthropic_keys(self):
        # A template line (even commented) invites a key the bot then throws away.
        lines = (ROOT / ".env.template").read_text().splitlines()
        for key in BLOCKED_KEYS:
            with self.subTest(key=key):
                offers = [line for line in lines if f"{key}=" in line.replace(" ", "")]
                self.assertEqual(offers, [])

    def test_every_workflow_that_runs_main_is_listed(self):
        # A new workflow added later must be classified here, not silently skipped.
        runners = sorted(p.name for p in workflow_files() if main_py_invocations(p.name))
        self.assertEqual(runners, sorted(MUST_PUBLISH))


if __name__ == "__main__":
    unittest.main()
