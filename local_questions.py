"""Load binary questions from a local JSON file, for dry runs that never touch Metaculus.

The file is a JSON list of objects. Only question_text is required; the other
keys (resolution_criteria, fine_print, background_info, page_url) are passed
through when present. page_url defaults to local://q<n> so reports stay
distinguishable in the summary.
"""

import json
from pathlib import Path

from forecasting_tools import BinaryQuestion

_OPTIONAL_KEYS = ("resolution_criteria", "fine_print", "background_info", "page_url")


def load_local_questions(path: str | Path) -> list[BinaryQuestion]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty JSON list of questions")
    questions = []
    for n, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not row.get("question_text"):
            raise ValueError(f"{path}: question {n} has no question_text")
        fields = {k: row[k] for k in _OPTIONAL_KEYS if row.get(k)}
        fields.setdefault("page_url", f"local://q{n}")
        questions.append(BinaryQuestion(question_text=row["question_text"], **fields))
    return questions
