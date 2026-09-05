"""Run strict TF-IDF/LinearSVC Daleel baselines with OOF evaluation.

This is the reproducible form of the disposable CPU probe from the
non-LLM baseline survey.  It uses no pretrained model, PyTorch,
Transformers, DSPy, API, prompt, or external labeled data.

Example from shared-task/:

  .venv/bin/python scripts/train_sparse_baseline.py --task both --confirm
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from daleel.artifacts import (
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    output_record,
    write_completion_marker,
)
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.folds import (
    DEFAULT_FOLD_SEED,
    Fold,
    fold_manifest,
    fold_manifest_hash,
    make_stratified_folds,
)
from daleel.io import write_jsonl
from daleel.metrics import Span, span_partial_f1, task1_macro_f1
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
    tune_task1_thresholds,
)
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids
from daleel.submission import validate_records_against_source


SCRIPT_VERSION = 1
ARCHITECTURES = {
    1: "char-tfidf-six-linearsvc-v1",
    2: "char-word-tfidf-connective-linearsvc-v1",
}
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def _installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _json_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=("1", "2", "both"), default="both")
    parser.add_argument("--pool", choices=("legacy-train", "all"), default="legacy-train")
    parser.add_argument("--setting", choices=("both", "editorial", "debate"), default="both")
    parser.add_argument("--folds", type=_positive_int, default=5)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="fit on legacy-train and read the repeatedly-used fixed 182 confirmation side",
    )

    parser.add_argument("--task1-char-min", type=_positive_int, default=2)
    parser.add_argument("--task1-char-max", type=_positive_int, default=5)
    parser.add_argument("--task1-min-df", type=_positive_int, default=2)
    parser.add_argument("--task1-c", type=_positive_float, default=1.0)
    parser.add_argument("--task1-balanced", action="store_true")

    parser.add_argument("--task2-char-min", type=_positive_int, default=2)
    parser.add_argument("--task2-char-max", type=_positive_int, default=5)
    parser.add_argument("--task2-word-min", type=_positive_int, default=1)
    parser.add_argument("--task2-word-max", type=_positive_int, default=2)
    parser.add_argument("--task2-min-df", type=_positive_int, default=2)
    parser.add_argument("--task2-c", type=_positive_float, default=0.5)
    parser.add_argument("--task2-balanced", action="store_true")
    parser.add_argument("--context-chars", type=int, default=0)
    parser.add_argument(
        "--granularity",
        choices=("sentence", "clause", "connective"),
        default="connective",
    )
    parser.add_argument("--max-iter", type=_positive_int, default=10_000)
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.confirm and args.pool != "legacy-train":
        parser.error("--confirm is defined only for --pool legacy-train")
    if args.context_chars < 0:
        parser.error("--context-chars must be non-negative")
    if args.task1_char_max < args.task1_char_min:
        parser.error("Task 1 character n-gram maximum must be >= minimum")
    if args.task2_char_max < args.task2_char_min:
        parser.error("Task 2 character n-gram maximum must be >= minimum")
    if args.task2_word_max < args.task2_word_min:
        parser.error("Task 2 word n-gram maximum must be >= minimum")
    return args


def _selected_tasks(raw: str) -> tuple[int, ...]:
    return (1, 2) if raw == "both" else (int(raw),)


def _select_ids(
    task1_source: list[dict[str, Any]],
    task2_source: list[dict[str, Any]],
    *,
    pool: str,
    setting: str,
) -> tuple[list[int], list[int]]:
    by_id = {row["paragraph_id"]: row for row in task1_source}
    legacy_train, legacy_confirmation = train_val_ids(task1_source, task2_source)
    selected = legacy_train if pool == "legacy-train" else sorted(by_id)
    selected = [
        pid for pid in selected if setting == "both" or by_id[pid]["type"] == setting
    ]
    confirmation = [
        pid
        for pid in legacy_confirmation
        if setting == "both" or by_id[pid]["type"] == setting
    ]
    return selected, confirmation


def _task1_config(args: argparse.Namespace) -> Task1SparseConfig:
    return Task1SparseConfig(
        char_ngram_min=args.task1_char_min,
        char_ngram_max=args.task1_char_max,
        min_df=args.task1_min_df,
        c=args.task1_c,
        class_weight="balanced" if args.task1_balanced else None,
        max_iter=args.max_iter,
        random_state=args.seed,
    )


def _task2_config(args: argparse.Namespace) -> Task2SparseConfig:
    return Task2SparseConfig(
        char_ngram_min=args.task2_char_min,
        char_ngram_max=args.task2_char_max,
        word_ngram_min=args.task2_word_min,
        word_ngram_max=args.task2_word_max,
        min_df=args.task2_min_df,
        c=args.task2_c,
        class_weight="balanced" if args.task2_balanced else None,
        max_iter=args.max_iter,
        random_state=args.seed,
        context_chars=args.context_chars,
        granularity=args.granularity,
    )


def _task1_rows(
    source_by_id: dict[int, dict[str, Any]],
    predictions: dict[int, set[str]],
    paragraph_ids: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            **source_by_id[paragraph_id],
            "labels": [
                label for label in LABELS if label in predictions.get(paragraph_id, ())
            ],
        }
        for paragraph_id in paragraph_ids
    ]


def _task2_rows(
    source_by_id: dict[int, dict[str, Any]],
    predictions: dict[int, list[Span]],
    paragraph_ids: Sequence[int],
) -> list[dict[str, Any]]:
    return [
        {
            **source_by_id[paragraph_id],
            "labels": [
                {
                    "label": span.label,
                    "start_offset": span.start,
                    "end_offset": span.end,
                }
                for span in predictions.get(paragraph_id, ())
            ],
        }
        for paragraph_id in paragraph_ids
    ]


def _genre_task1(
    source_by_id: dict[int, dict[str, Any]],
    gold: dict[int, set[str]],
    predictions: dict[int, set[str]],
    paragraph_ids: Sequence[int],
) -> dict[str, Any]:
    result = {}
    for genre in ("editorial", "debate"):
        ids = [pid for pid in paragraph_ids if source_by_id[pid]["type"] == genre]
        if ids:
            result[genre] = task1_macro_f1(
                {pid: gold[pid] for pid in ids},
                {pid: predictions[pid] for pid in ids},
            )
    return result


def _genre_task2(
    source_by_id: dict[int, dict[str, Any]],
    gold: dict[int, list[Span]],
    predictions: dict[int, list[Span]],
    paragraph_ids: Sequence[int],
) -> dict[str, Any]:
    result = {}
    for genre in ("editorial", "debate"):
        ids = [pid for pid in paragraph_ids if source_by_id[pid]["type"] == genre]
        if ids:
            result[genre] = span_partial_f1(
                {pid: gold[pid] for pid in ids},
                {pid: predictions[pid] for pid in ids},
            )
    return result


def run_task1(
    folds: Sequence[Fold],
    selected_ids: Sequence[int],
    confirmation_ids: Sequence[int],
    source_by_id: dict[int, dict[str, Any]],
    gold: dict[int, set[str]],
    config: Task1SparseConfig,
    *,
    confirm: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    score_by_id: dict[int, np.ndarray] = {}
    fold_rows = []
    started = time.monotonic()
    for fold in folds:
        fold_started = time.monotonic()
        model = fit_task1_sparse(source_by_id, gold, fold.train_ids, config)
        scores = model.decision_function(
            [source_by_id[pid]["text"] for pid in fold.val_ids]
        )
        for paragraph_id, row in zip(fold.val_ids, scores):
            if paragraph_id in score_by_id:
                raise ValueError(f"duplicate Task 1 OOF paragraph {paragraph_id}")
            score_by_id[paragraph_id] = row
        fold_rows.append(
            {
                "fold": fold.index,
                "train_paragraph_count": len(fold.train_ids),
                "validation_paragraph_count": len(fold.val_ids),
                "feature_count": len(model.vectorizer.vocabulary_),
                "seconds": time.monotonic() - fold_started,
            }
        )
    paragraph_ids = list(selected_ids)
    scores = np.stack([score_by_id[pid] for pid in paragraph_ids])
    gold_matrix = np.asarray(
        [[label in gold[pid] for label in LABELS] for pid in paragraph_ids],
        dtype=bool,
    )
    fold_by_id = {pid: fold.index for fold in folds for pid in fold.val_ids}
    fold_indices = [fold_by_id[pid] for pid in paragraph_ids]
    thresholds, threshold_fit_f1 = tune_task1_thresholds(gold_matrix, scores)
    cross_decisions, thresholds_by_fold = cross_fitted_task1_predictions(
        gold_matrix, scores, fold_indices
    )
    cross_predictions = task1_prediction_map(paragraph_ids, cross_decisions)
    apparent_predictions = task1_prediction_map(paragraph_ids, scores >= thresholds)
    zero_predictions = task1_prediction_map(paragraph_ids, scores >= 0.0)
    cross_metric = task1_macro_f1(
        {pid: gold[pid] for pid in paragraph_ids}, cross_predictions
    )
    metrics: dict[str, Any] = {
        "architecture": ARCHITECTURES[1],
        "config": asdict(config),
        "primary": {
            "name": "official_task1_macro_f1_cross_fitted_oof_thresholds",
            "score": cross_metric["macro_f1"],
        },
        "oof": {
            "cross_fitted_thresholds": cross_metric,
            "per_genre": _genre_task1(
                source_by_id, gold, cross_predictions, paragraph_ids
            ),
            "same_oof_threshold_fit_report_compatible": task1_macro_f1(
                {pid: gold[pid] for pid in paragraph_ids}, apparent_predictions
            ),
            "default_zero_margin": task1_macro_f1(
                {pid: gold[pid] for pid in paragraph_ids}, zero_predictions
            ),
            "final_thresholds_fit_on_all_oof_margins": {
                label: float(threshold)
                for label, threshold in zip(LABELS, thresholds)
            },
            "threshold_fit_f1_by_label": threshold_fit_f1,
            "cross_fitted_thresholds_by_heldout_fold": thresholds_by_fold,
        },
        "folds": fold_rows,
        "seconds": time.monotonic() - started,
    }
    files = {
        "oof_predictions": _task1_rows(
            source_by_id, cross_predictions, paragraph_ids
        ),
        "oof_scores": [
            {
                "paragraph_id": pid,
                "fold": fold_by_id[pid],
                "margins": {
                    label: float(scores[row_index, label_index])
                    for label_index, label in enumerate(LABELS)
                },
            }
            for row_index, pid in enumerate(paragraph_ids)
        ],
    }
    if confirm:
        final_started = time.monotonic()
        final_model = fit_task1_sparse(source_by_id, gold, paragraph_ids, config)
        confirmation_scores = final_model.decision_function(
            [source_by_id[pid]["text"] for pid in confirmation_ids]
        )
        confirmation_predictions = task1_prediction_map(
            confirmation_ids, confirmation_scores >= thresholds
        )
        confirmation_metric = task1_macro_f1(
            {pid: gold[pid] for pid in confirmation_ids}, confirmation_predictions
        )
        metrics["confirmation"] = {
            "warning": "repeatedly inspected fixed 182-side; not an unbiased test",
            "official": confirmation_metric,
            "per_genre": _genre_task1(
                source_by_id,
                gold,
                confirmation_predictions,
                confirmation_ids,
            ),
            "feature_count": len(final_model.vectorizer.vocabulary_),
            "seconds": time.monotonic() - final_started,
        }
        files["confirmation_predictions"] = _task1_rows(
            source_by_id, confirmation_predictions, confirmation_ids
        )
        files["confirmation_scores"] = [
            {
                "paragraph_id": pid,
                "margins": {
                    label: float(confirmation_scores[row_index, label_index])
                    for label_index, label in enumerate(LABELS)
                },
            }
            for row_index, pid in enumerate(confirmation_ids)
        ]
    return metrics, files


def run_task2(
    folds: Sequence[Fold],
    selected_ids: Sequence[int],
    confirmation_ids: Sequence[int],
    source_by_id: dict[int, dict[str, Any]],
    task2_source_by_id: dict[int, dict[str, Any]],
    gold_task1: dict[int, set[str]],
    gold_task2: dict[int, list[Span]],
    config: Task2SparseConfig,
    *,
    confirm: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    all_examples: list[SparseSegment] = []
    all_predictions: list[int] = []
    all_scores: list[np.ndarray] = []
    all_fold_indices: list[int] = []
    fold_rows = []
    started = time.monotonic()
    for fold in folds:
        fold_started = time.monotonic()
        train_examples = build_sparse_segments(
            [task2_source_by_id[pid] for pid in fold.train_ids],
            gold_task2,
            fold.train_ids,
            granularity=config.granularity,
        )
        validation_examples = build_sparse_segments(
            [task2_source_by_id[pid] for pid in fold.val_ids],
            gold_task2,
            fold.val_ids,
            granularity=config.granularity,
        )
        model = fit_task2_sparse(train_examples, config)
        rendered = [
            render_sparse_segment(example, config.context_chars)
            for example in validation_examples
        ]
        predictions = model.predict(rendered)
        scores = model.decision_function(rendered)
        feature_count = sum(
            len(transformer.vocabulary_)
            for _, transformer in model.vectorizer.transformer_list
        )
        all_examples.extend(validation_examples)
        all_predictions.extend(int(value) for value in predictions)
        all_scores.append(scores)
        all_fold_indices.extend([fold.index] * len(validation_examples))
        fold_rows.append(
            {
                "fold": fold.index,
                "train_paragraph_count": len(fold.train_ids),
                "validation_paragraph_count": len(fold.val_ids),
                "train_segment_count": len(train_examples),
                "validation_segment_count": len(validation_examples),
                "feature_count": feature_count,
                "seconds": time.monotonic() - fold_started,
            }
        )
    paragraph_ids = list(selected_ids)
    scores = np.concatenate(all_scores, axis=0)
    predictions = segment_prediction_map(
        all_examples, all_predictions, paragraph_ids
    )
    oracle = segment_prediction_map(
        all_examples, [example.label_index for example in all_examples], paragraph_ids
    )
    official = span_partial_f1(
        {pid: gold_task2[pid] for pid in paragraph_ids}, predictions
    )
    task1_from_spans = {
        pid: {span.label for span in predictions[pid]} for pid in paragraph_ids
    }
    metrics: dict[str, Any] = {
        "architecture": ARCHITECTURES[2],
        "config": asdict(config),
        "primary": {
            "name": "official_task2_partial_f1_oof",
            "score": official["f1"],
        },
        "oof": {
            "official": official,
            "per_genre": _genre_task2(
                source_by_id, gold_task2, predictions, paragraph_ids
            ),
            "deterministic_segmentation_oracle": span_partial_f1(
                {pid: gold_task2[pid] for pid in paragraph_ids}, oracle
            ),
            "task1_derived_from_spans": task1_macro_f1(
                {pid: gold_task1[pid] for pid in paragraph_ids}, task1_from_spans
            ),
            "segment_accuracy": float(
                np.mean(
                    np.asarray([example.label_index for example in all_examples])
                    == np.asarray(all_predictions)
                )
            ),
            "gold_segment_class_counts": dict(
                sorted(Counter(example.label for example in all_examples).items())
            ),
            "predicted_segment_class_counts": {
                label: int(sum(value == index for value in all_predictions))
                for index, label in enumerate(SEGMENT_LABELS)
            },
        },
        "folds": fold_rows,
        "seconds": time.monotonic() - started,
    }
    files = {
        "oof_predictions": _task2_rows(source_by_id, predictions, paragraph_ids),
        "oof_scores": [
            {
                "paragraph_id": example.paragraph_id,
                "fold": fold,
                "start_offset": example.start,
                "end_offset": example.end,
                "gold_oracle_label": example.label,
                "predicted_label": SEGMENT_LABELS[predicted],
                "margins": {
                    label: float(scores[row_index, label_index])
                    for label_index, label in enumerate(SEGMENT_LABELS)
                },
            }
            for row_index, (example, fold, predicted) in enumerate(
                zip(all_examples, all_fold_indices, all_predictions)
            )
        ],
    }
    if confirm:
        final_started = time.monotonic()
        train_examples = build_sparse_segments(
            [task2_source_by_id[pid] for pid in paragraph_ids],
            gold_task2,
            paragraph_ids,
            granularity=config.granularity,
        )
        confirmation_examples = build_sparse_segments(
            [task2_source_by_id[pid] for pid in confirmation_ids],
            gold_task2,
            confirmation_ids,
            granularity=config.granularity,
        )
        model = fit_task2_sparse(train_examples, config)
        rendered = [
            render_sparse_segment(example, config.context_chars)
            for example in confirmation_examples
        ]
        confirmation_predictions_raw = model.predict(rendered)
        confirmation_scores = model.decision_function(rendered)
        confirmation_predictions = segment_prediction_map(
            confirmation_examples,
            confirmation_predictions_raw,
            confirmation_ids,
        )
        confirmation_metric = span_partial_f1(
            {pid: gold_task2[pid] for pid in confirmation_ids},
            confirmation_predictions,
        )
        confirmation_oracle = segment_prediction_map(
            confirmation_examples,
            [example.label_index for example in confirmation_examples],
            confirmation_ids,
        )
        confirmation_task1_from_spans = {
            pid: {span.label for span in confirmation_predictions[pid]}
            for pid in confirmation_ids
        }
        metrics["confirmation"] = {
            "warning": "repeatedly inspected fixed 182-side; not an unbiased test",
            "official": confirmation_metric,
            "per_genre": _genre_task2(
                source_by_id,
                gold_task2,
                confirmation_predictions,
                confirmation_ids,
            ),
            "deterministic_segmentation_oracle": span_partial_f1(
                {pid: gold_task2[pid] for pid in confirmation_ids},
                confirmation_oracle,
            ),
            "task1_derived_from_spans": task1_macro_f1(
                {pid: gold_task1[pid] for pid in confirmation_ids},
                confirmation_task1_from_spans,
            ),
            "train_segment_count": len(train_examples),
            "confirmation_segment_count": len(confirmation_examples),
            "seconds": time.monotonic() - final_started,
        }
        files["confirmation_predictions"] = _task2_rows(
            source_by_id, confirmation_predictions, confirmation_ids
        )
        files["confirmation_scores"] = [
            {
                "paragraph_id": example.paragraph_id,
                "start_offset": example.start,
                "end_offset": example.end,
                "gold_oracle_label": example.label,
                "predicted_label": SEGMENT_LABELS[int(predicted)],
                "margins": {
                    label: float(confirmation_scores[row_index, label_index])
                    for label_index, label in enumerate(SEGMENT_LABELS)
                },
            }
            for row_index, (example, predicted) in enumerate(
                zip(confirmation_examples, confirmation_predictions_raw)
            )
        ]
    return metrics, files


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    tasks = _selected_tasks(args.task)
    task1_source = load_records(TRAIN_TASK1)
    task2_source = load_records(TRAIN_TASK2)
    source_by_id = {row["paragraph_id"]: row for row in task1_source}
    task2_source_by_id = {row["paragraph_id"]: row for row in task2_source}
    if set(source_by_id) != set(task2_source_by_id):
        raise SystemExit("official Task 1/Task 2 paragraph IDs differ")
    gold_task1 = clean_task1_gold(task1_source, task2_source)
    gold_task2 = clean_task2_gold(task2_source)
    selected_ids, confirmation_ids = _select_ids(
        task1_source,
        task2_source,
        pool=args.pool,
        setting=args.setting,
    )
    if len(selected_ids) < args.folds:
        raise SystemExit("selected pool is smaller than the requested fold count")
    folds = make_stratified_folds(
        selected_ids,
        {pid: gold_task1[pid] for pid in selected_ids},
        {pid: source_by_id[pid]["type"] for pid in selected_ids},
        n_splits=args.folds,
        seed=args.fold_seed,
    )
    folds_data = fold_manifest(folds, seed=args.fold_seed)
    task_configs = {}
    if 1 in tasks:
        task_configs["task1"] = asdict(_task1_config(args))
    if 2 in tasks:
        task_configs["task2"] = asdict(_task2_config(args))
    contract = {
        "script_version": SCRIPT_VERSION,
        "tasks": list(tasks),
        "architectures": {str(task): ARCHITECTURES[task] for task in tasks},
        "pool": args.pool,
        "setting": args.setting,
        "fold_count": args.folds,
        "fold_seed": args.fold_seed,
        "confirm": args.confirm,
        "task_configs": task_configs,
        "selected_ids_sha256": _json_sha256(selected_ids),
        "confirmation_ids_sha256": (
            _json_sha256(confirmation_ids) if args.confirm else None
        ),
        "fold_manifest_sha256": fold_manifest_hash(folds_data),
        "source_sha256": {
            "task1": file_sha256(TRAIN_TASK1),
            "task2": file_sha256(TRAIN_TASK2),
        },
    }
    slug = f"sparse-t{args.task}-{args.setting}-{args.pool}"
    run_dir = create_experiment_dir(EXPERIMENTS_DIR, slug, contract)
    config_path = atomic_write_json(
        run_dir / "config.json",
        contract
        | {
            "script": "train_sparse_baseline.py",
            "arguments": vars(args),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scikit_learn": _installed_version("scikit-learn"),
            },
        },
    )
    folds_path = atomic_write_json(run_dir / "folds.json", folds_data)
    provenance = {
        "schema_version": 1,
        "status": "running",
        "tasks": list(tasks),
        "track": "closed",
        "setting": args.setting,
        "architectures": {str(task): ARCHITECTURES[task] for task in tasks},
        "dspy_used": False,
        "generative_inference_used": False,
        "pretrained_language_model_used": False,
        "external_labeled_data_used": False,
        "model": {
            "family": "scikit-learn LinearSVC",
            "version": _installed_version("scikit-learn"),
        },
        "data_sources": [
            {
                "id": "daleel2026:train-task-1",
                "path": str(TRAIN_TASK1),
                "sha256": file_sha256(TRAIN_TASK1),
            },
            {
                "id": "daleel2026:train-task-2",
                "path": str(TRAIN_TASK2),
                "sha256": file_sha256(TRAIN_TASK2),
            },
        ],
        "training_ids": selected_ids,
        "training_ids_sha256": _json_sha256(selected_ids),
        "confirmation_ids": confirmation_ids if args.confirm else [],
        "fold_manifest_sha256": fold_manifest_hash(folds_data),
        "cleaning_policy": "daleel.splits clean_task1_gold/clean_task2_gold",
    }
    pending_path = atomic_write_json(
        run_dir / "provenance.pending.json", provenance
    )

    all_metrics: dict[str, Any] = {}
    all_files: dict[int, dict[str, list[dict[str, Any]]]] = {}
    if 1 in tasks:
        print("running Task 1 sparse OOF", flush=True)
        all_metrics["task1"], all_files[1] = run_task1(
            folds,
            selected_ids,
            confirmation_ids,
            source_by_id,
            gold_task1,
            _task1_config(args),
            confirm=args.confirm,
        )
    if 2 in tasks:
        print("running Task 2 sparse OOF", flush=True)
        all_metrics["task2"], all_files[2] = run_task2(
            folds,
            selected_ids,
            confirmation_ids,
            source_by_id,
            task2_source_by_id,
            gold_task1,
            gold_task2,
            _task2_config(args),
            confirm=args.confirm,
        )
    all_metrics["run"] = {
        "tasks": list(tasks),
        "selection_paragraph_count": len(selected_ids),
        "confirmation_paragraph_count": len(confirmation_ids) if args.confirm else 0,
        "confirmation_is_unbiased": False if args.confirm else None,
    }
    metrics_path = atomic_write_json(run_dir / "metrics.json", all_metrics)

    output_paths: list[Path] = []
    output_manifest = []
    for task, file_groups in all_files.items():
        source = [source_by_id[pid] for pid in selected_ids]
        oof_rows = file_groups["oof_predictions"]
        problems = validate_records_against_source(oof_rows, source, f"task_{task}")
        if problems:
            raise ValueError("invalid sparse OOF output:\n- " + "\n- ".join(problems[:30]))
        prediction_path = run_dir / "predictions" / f"oof_task_{task}.jsonl"
        score_path = run_dir / "predictions" / f"oof_task_{task}_scores.jsonl"
        write_jsonl(prediction_path, oof_rows)
        write_jsonl(score_path, file_groups["oof_scores"])
        output_paths.extend((prediction_path, score_path))
        output_manifest.extend(
            [
                output_record(
                    run_dir,
                    prediction_path,
                    kind=f"sparse-task-{task}-oof-predictions",
                ),
                output_record(
                    run_dir, score_path, kind=f"sparse-task-{task}-oof-margins"
                ),
            ]
        )
        if args.confirm:
            confirmation_source = [source_by_id[pid] for pid in confirmation_ids]
            confirmation_rows = file_groups["confirmation_predictions"]
            problems = validate_records_against_source(
                confirmation_rows, confirmation_source, f"task_{task}"
            )
            if problems:
                raise ValueError(
                    "invalid sparse confirmation output:\n- "
                    + "\n- ".join(problems[:30])
                )
            confirmation_path = (
                run_dir / "predictions" / f"confirmation_task_{task}.jsonl"
            )
            confirmation_scores_path = (
                run_dir
                / "predictions"
                / f"confirmation_task_{task}_scores.jsonl"
            )
            write_jsonl(confirmation_path, confirmation_rows)
            write_jsonl(
                confirmation_scores_path, file_groups["confirmation_scores"]
            )
            output_paths.extend((confirmation_path, confirmation_scores_path))
            output_manifest.extend(
                [
                    output_record(
                        run_dir,
                        confirmation_path,
                        kind=f"sparse-task-{task}-confirmation-predictions",
                    ),
                    output_record(
                        run_dir,
                        confirmation_scores_path,
                        kind=f"sparse-task-{task}-confirmation-margins",
                    ),
                ]
            )

    output_manifest.extend(
        [
            output_record(run_dir, metrics_path, kind="sparse-baseline-metrics"),
            output_record(run_dir, folds_path, kind="paragraph-fold-manifest"),
        ]
    )
    provenance |= {
        "status": "complete",
        "outputs": output_manifest,
        "primary_metrics": {
            key: value["primary"]
            for key, value in all_metrics.items()
            if key.startswith("task")
        },
    }
    provenance_path = atomic_write_json(run_dir / "provenance.json", provenance)
    pending_path.unlink()
    completion = write_completion_marker(
        run_dir,
        [
            config_path,
            folds_path,
            metrics_path,
            provenance_path,
            *output_paths,
        ],
        metadata={
            "tasks": list(tasks),
            "primary_scores": {
                key: value["primary"]["score"]
                for key, value in all_metrics.items()
                if key.startswith("task")
            },
        },
    )
    summary = {
        key: value["primary"]
        for key, value in all_metrics.items()
        if key.startswith("task")
    }
    if args.confirm:
        for key in list(summary):
            summary[key]["confirmation_score"] = all_metrics[key]["confirmation"][
                "official"
            ]["macro_f1" if key == "task1" else "f1"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"completed: {completion}")
    return run_dir


if __name__ == "__main__":
    main()
