"""Parity checks: daleel.metrics must reproduce the official scoring scripts.

The official scripts live in the (gitignored) clone of
https://github.com/Argmining/Daleel2026 — these tests import them directly
and compare results on synthetic data and on the real training set. They
skip when the clone is absent.
"""

import importlib.util
import random

import pytest

from daleel.data import OFFICIAL_EVAL_DIR, TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.data import task1_labels as to_label_sets
from daleel.data import task2_spans as to_spans
from daleel.metrics import LABELS, Span, span_partial_f1, task1_macro_f1

pytestmark = pytest.mark.skipif(
    not OFFICIAL_EVAL_DIR.exists(), reason="official Daleel2026 clone not present"
)


def _import(name):
    spec = importlib.util.spec_from_file_location(name, OFFICIAL_EVAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_label_sets(rng, n, id_offset=0):
    return {
        i + id_offset: {label for label in LABELS if rng.random() < 0.3} for i in range(n)
    }


def _random_spans(rng, n, id_offset=0):
    result = {}
    for i in range(n):
        spans = []
        for _ in range(rng.randrange(0, 6)):
            start = rng.randrange(0, 190)
            spans.append(Span(start, start + rng.randrange(1, 60), rng.choice(LABELS)))
        result[i + id_offset] = spans
    return result


def test_task1_parity_synthetic():
    official = _import("task1_scoring")
    rng = random.Random(0)
    gold = _random_label_sets(rng, 60)
    pred = _random_label_sets(rng, 60)
    theirs = official.evaluate(gold, pred, per_label=False)
    ours = task1_macro_f1(gold, pred)["macro_f1"]
    assert ours == pytest.approx(theirs)


def test_task2_parity_synthetic():
    official = _import("task2_scoring")
    rng = random.Random(1)
    gold = _random_spans(rng, 60)
    pred = _random_spans(rng, 60)
    as_tuples = lambda d: {  # noqa: E731 — official input shape
        i: [(s.label, [s.start, s.end]) for s in spans] for i, spans in d.items()
    }
    _, _, theirs, *_ = official.score_per_span(as_tuples(gold), as_tuples(pred))
    ours = span_partial_f1(gold, pred)["f1"]
    assert ours == pytest.approx(theirs)


def _degrade_task1(gold, rng):
    """Deterministically drop/add labels to fake an imperfect system."""
    pred = {}
    for pid, labels in gold.items():
        kept = {label for label in labels if rng.random() > 0.25}
        if rng.random() < 0.15:
            kept.add(rng.choice(LABELS))
        pred[pid] = kept
    return pred


def test_task1_parity_on_real_training_data():
    official = _import("task1_scoring")
    gold = to_label_sets(load_records(TRAIN_TASK1))
    pred = _degrade_task1(gold, random.Random(2))
    theirs = official.evaluate(gold, pred, per_label=False)
    ours = task1_macro_f1(gold, pred)["macro_f1"]
    assert ours == pytest.approx(theirs)


def test_task2_parity_on_real_training_data():
    official = _import("task2_scoring")
    gold = to_spans(load_records(TRAIN_TASK2))
    rng = random.Random(3)
    pred = {}
    for pid, spans in gold.items():
        kept = []
        for s in spans:
            if rng.random() > 0.2:
                shift = rng.randrange(-10, 11)
                kept.append(Span(max(0, s.start + shift), s.end + max(shift, 0) + 1, s.label))
        pred[pid] = kept
    as_tuples = lambda d: {  # noqa: E731
        i: [(s.label, [s.start, s.end]) for s in spans] for i, spans in d.items()
    }
    _, _, theirs, *_ = official.score_per_span(as_tuples(gold), as_tuples(pred))
    ours = span_partial_f1(gold, pred)["f1"]
    assert ours == pytest.approx(theirs)


def test_genre_filter_matches_official_type_counts():
    records = load_records(TRAIN_TASK1)
    editorials = load_records(TRAIN_TASK1, genre="editorial")
    debates = load_records(TRAIN_TASK1, genre="debate")
    assert len(editorials) + len(debates) == len(records) == 612
