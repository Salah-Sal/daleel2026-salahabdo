"""Loaders for the official Daleel2026 dataset.

The organizers publish training data, the dev-phase input, and the scoring
scripts at https://github.com/Argmining/Daleel2026, cloned (gitignored) at
resources/repos/Daleel2026. Record schemas (verified against the files):

- train_task_1.jsonl: {paragraph_id, text, type, labels: [str, ...]}
  (labels may be empty — 60/612 training paragraphs carry no ADU label)
- train_task_2.jsonl: {paragraph_id, text, type,
  labels: [{label, start_offset, end_offset}, ...]}
- dev_in.jsonl: {paragraph_id, text, type} — no labels.

Paths assume the package is installed editable inside this repo (uv sync's
default), so everything resolves relative to this file.
"""

from pathlib import Path

from .io import read_jsonl
from .metrics import Span

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_CLONE = REPO_ROOT / "resources" / "repos" / "Daleel2026"
TRAIN_TASK1 = OFFICIAL_CLONE / "data" / "train" / "train_task_1.jsonl"
TRAIN_TASK2 = OFFICIAL_CLONE / "data" / "train" / "train_task_2.jsonl"
DEV_INPUT = OFFICIAL_CLONE / "data" / "dev" / "dev_in.jsonl"
OFFICIAL_EVAL_DIR = OFFICIAL_CLONE / "evaluation"


def load_records(path: str | Path, genre: str | None = None) -> list[dict]:
    """Read a Daleel jsonl file, optionally keeping one genre only.

    `genre` is matched against the record's "type" field ("editorial" or
    "debate") — the same filtering the official scorers apply for the
    per-genre leaderboard columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — clone https://github.com/Argmining/Daleel2026 "
            f"into {OFFICIAL_CLONE.parent}/ (see README.md, Setup)"
        )
    records = read_jsonl(path)
    if genre is not None:
        records = [r for r in records if r.get("type") == genre]
    return records


def task1_labels(records: list[dict]) -> dict[object, set[str]]:
    """paragraph_id -> label set, the shape task1_macro_f1 consumes."""
    return {r["paragraph_id"]: set(r["labels"]) for r in records}


def task2_spans(records: list[dict]) -> dict[object, list[Span]]:
    """paragraph_id -> [Span, ...], the shape span_partial_f1 consumes."""
    return {
        r["paragraph_id"]: [
            Span(s["start_offset"], s["end_offset"], s["label"]) for s in r["labels"]
        ]
        for r in records
    }
