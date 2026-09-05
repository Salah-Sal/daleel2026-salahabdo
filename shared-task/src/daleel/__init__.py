"""Working code for the Daleel 2026 Arabic Argumentative Discourse Mining shared task."""

from .constants import (
    CODABENCH_USERNAME,
    GENRES,
    LABEL_NAMES,
    LABEL_NAMES_AR,
    LABELS,
    OFFICIAL_REPO_URL,
    TEAM_NAME,
    TRAINING_SETTINGS,
)
from .data import (
    DEV_INPUT,
    TRAIN_TASK1,
    TRAIN_TASK2,
    load_records,
    task1_labels,
    task2_spans,
)
from .io import read_jsonl, write_jsonl
from .metrics import Span, span_partial_f1, task1_macro_f1
from .submission import (
    package_submission,
    validate_records_against_source,
    validate_source_records,
    validate_task1_records,
    validate_task2_records,
)

__all__ = [
    "CODABENCH_USERNAME",
    "DEV_INPUT",
    "GENRES",
    "LABEL_NAMES",
    "LABEL_NAMES_AR",
    "LABELS",
    "OFFICIAL_REPO_URL",
    "Span",
    "TEAM_NAME",
    "TRAINING_SETTINGS",
    "TRAIN_TASK1",
    "TRAIN_TASK2",
    "load_records",
    "package_submission",
    "read_jsonl",
    "span_partial_f1",
    "task1_labels",
    "task1_macro_f1",
    "task2_spans",
    "validate_records_against_source",
    "validate_source_records",
    "validate_task1_records",
    "validate_task2_records",
    "write_jsonl",
]
