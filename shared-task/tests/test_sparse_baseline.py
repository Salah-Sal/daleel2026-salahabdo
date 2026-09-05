"""Contracts for the strict TF-IDF/LinearSVC baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from daleel.constants import LABELS
from daleel.metrics import Span
from daleel.sparse_baseline import (
    SEGMENT_LABELS,
    SparseSegment,
    Task1SparseConfig,
    Task2SparseConfig,
    build_sparse_segments,
    cross_fitted_task1_predictions,
    fit_task1_sparse,
    fit_task2_sparse,
    render_sparse_segment,
    segment_prediction_map,
    task1_prediction_map,
    tune_binary_threshold,
    tune_task1_thresholds,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.train_sparse_baseline as runner


def test_sparse_import_closure_has_no_model_runtime():
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.train_sparse_baseline; "
            "raise SystemExit(int(any(x in sys.modules for x in "
            "('dspy','torch','transformers'))))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_sparse_config_validation_and_cli_guards():
    with pytest.raises(ValueError, match="n-gram range"):
        Task1SparseConfig(char_ngram_min=4, char_ngram_max=2)
    with pytest.raises(ValueError, match="non-negative"):
        Task2SparseConfig(context_chars=-1)
    with pytest.raises(SystemExit):
        runner.parse_args(["--folds", "1"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--pool", "all", "--confirm"])
    parsed = runner.parse_args(["--task", "1", "--seed", "9"])
    assert runner._task1_config(parsed).random_state == 9


def test_task1_sparse_fit_scores_all_labels_and_handles_constant_fold_targets():
    records = {
        1: {"paragraph_id": 1, "text": "دليل مشترك أول", "type": "editorial"},
        2: {"paragraph_id": 2, "text": "دليل مشترك ثان", "type": "debate"},
        3: {"paragraph_id": 3, "text": "خبر مشترك ثالث", "type": "editorial"},
        4: {"paragraph_id": 4, "text": "خبر مشترك رابع", "type": "debate"},
    }
    gold = {
        1: {"AS"},
        2: {"AS", "TE"},
        3: {"TE"},
        4: set(),
    }
    model = fit_task1_sparse(
        records,
        gold,
        [1, 2, 3, 4],
        Task1SparseConfig(min_df=1, random_state=17),
    )
    scores = model.decision_function(["دليل جديد", "خبر جديد"])
    assert scores.shape == (2, len(LABELS))
    assert np.isfinite(scores).all()
    # CO is absent from the fold, so its deterministic fallback is negative.
    assert np.all(scores[:, LABELS.index("CO")] == -1.0)


def test_sparse_segments_are_source_bound_and_context_is_feature_only():
    text = "دليل، ولذلك نتيجة."
    records = [{"paragraph_id": 1, "text": text, "type": "editorial"}]
    examples = build_sparse_segments(
        records,
        {1: [Span(0, len(text), "AS")]},
        {1},
        granularity="connective",
    )
    assert examples
    assert all(text[item.start : item.end] == item.text for item in examples)
    assert render_sparse_segment(examples[0]) == examples[0].text
    contextual = render_sparse_segment(examples[0], 8)
    assert "<المقطع>" in contextual and "</المقطع>" in contextual


def test_task2_sparse_fit_emits_one_margin_per_registered_class():
    examples = []
    for index, label in enumerate(SEGMENT_LABELS):
        text = f"نص مشترك فئة {index}"
        examples.append(
            SparseSegment(index, text, "debate", 0, len(text), text, label)
        )
    config = Task2SparseConfig(min_df=1, random_state=3)
    model = fit_task2_sparse(examples, config)
    rendered = [render_sparse_segment(example) for example in examples]
    assert model.decision_function(rendered).shape == (
        len(examples),
        len(SEGMENT_LABELS),
    )
    predictions = model.predict(rendered)
    assert predictions.shape == (len(examples),)
    mapped = segment_prediction_map(examples, predictions, range(len(examples)))
    assert set(mapped) == set(range(len(examples)))


def test_sparse_thresholds_are_conservative_and_cross_fitted():
    threshold, score = tune_binary_threshold([0, 1, 1], [-1.0, -0.2, 0.4])
    assert threshold == pytest.approx(-0.2)
    assert score == 1.0
    gold = np.asarray(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
        ],
        dtype=bool,
    )
    scores = np.where(gold, 0.5, -0.5)
    thresholds, per_label = tune_task1_thresholds(gold, scores)
    assert thresholds.shape == (len(LABELS),)
    assert set(per_label) == set(LABELS)
    predicted, by_fold = cross_fitted_task1_predictions(
        gold, scores, [0, 0, 1, 1]
    )
    assert predicted.shape == gold.shape
    assert set(by_fold) == {0, 1}
    mapping = task1_prediction_map([1, 2, 3, 4], predicted)
    assert set(mapping) == {1, 2, 3, 4}

