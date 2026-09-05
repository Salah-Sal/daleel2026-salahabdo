"""Strict no-language-model TF-IDF/LinearSVC baselines for Daleel.

This module depends only on NumPy and scikit-learn.  It intentionally does
not import PyTorch, Transformers, DSPy, or any pretrained representation.
All Task 2 offsets are copied from deterministic source segments.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .constants import LABELS
from .metrics import Span
from .segment import oracle_label, segment_paragraph


NONE_LABEL = "NONE"
SEGMENT_LABELS = LABELS + (NONE_LABEL,)


@dataclass(frozen=True, slots=True)
class Task1SparseConfig:
    char_ngram_min: int = 2
    char_ngram_max: int = 5
    min_df: int = 2
    sublinear_tf: bool = True
    c: float = 1.0
    class_weight: str | None = None
    max_iter: int = 10_000
    random_state: int = 20260710

    def __post_init__(self) -> None:
        _validate_ngram(self.char_ngram_min, self.char_ngram_max, "character")
        if self.min_df < 1:
            raise ValueError("min_df must be at least 1")
        if self.c <= 0:
            raise ValueError("LinearSVC C must be positive")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")


@dataclass(frozen=True, slots=True)
class Task2SparseConfig:
    char_ngram_min: int = 2
    char_ngram_max: int = 5
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    min_df: int = 2
    sublinear_tf: bool = True
    c: float = 0.5
    class_weight: str | None = None
    max_iter: int = 10_000
    random_state: int = 20260710
    context_chars: int = 0
    granularity: str = "connective"

    def __post_init__(self) -> None:
        _validate_ngram(self.char_ngram_min, self.char_ngram_max, "character")
        _validate_ngram(self.word_ngram_min, self.word_ngram_max, "word")
        if self.min_df < 1:
            raise ValueError("min_df must be at least 1")
        if self.c <= 0:
            raise ValueError("LinearSVC C must be positive")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.context_chars < 0:
            raise ValueError("context_chars must be non-negative")
        if self.granularity not in {"sentence", "clause", "connective"}:
            raise ValueError(f"unknown segment granularity: {self.granularity!r}")


def _validate_ngram(minimum: int, maximum: int, name: str) -> None:
    if minimum < 1 or maximum < minimum:
        raise ValueError(
            f"invalid {name} n-gram range ({minimum}, {maximum}); "
            "expected 1 <= minimum <= maximum"
        )


@dataclass(frozen=True, slots=True)
class SparseSegment:
    paragraph_id: int
    paragraph_text: str
    genre: str
    start: int
    end: int
    text: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in SEGMENT_LABELS:
            raise ValueError(f"unknown segment label: {self.label!r}")
        if not 0 <= self.start < self.end <= len(self.paragraph_text):
            raise ValueError(f"invalid segment [{self.start}, {self.end})")
        if self.paragraph_text[self.start : self.end] != self.text:
            raise ValueError("segment text does not match source")

    @property
    def label_index(self) -> int:
        return SEGMENT_LABELS.index(self.label)


class ConstantBinaryMargin:
    """Deterministic fallback for a CV train fold containing one class."""

    def __init__(self, positive: bool) -> None:
        self.positive = bool(positive)

    def decision_function(self, features: Any) -> np.ndarray:
        return np.full(features.shape[0], 1.0 if self.positive else -1.0)


@dataclass(slots=True)
class Task1SparseModel:
    vectorizer: TfidfVectorizer
    classifiers: tuple[BaseEstimator | ConstantBinaryMargin, ...]

    def decision_function(self, texts: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(texts)
        return np.column_stack(
            [classifier.decision_function(features) for classifier in self.classifiers]
        ).astype(float, copy=False)


@dataclass(slots=True)
class Task2SparseModel:
    vectorizer: FeatureUnion
    classifier: LinearSVC

    def decision_function(self, inputs: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(inputs)
        raw = np.asarray(self.classifier.decision_function(features), dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(SEGMENT_LABELS):
            raise ValueError(
                "Task 2 classifier did not emit one margin per segment class; "
                f"got shape {raw.shape}"
            )
        return raw

    def predict(self, inputs: Sequence[str]) -> np.ndarray:
        features = self.vectorizer.transform(inputs)
        return np.asarray(self.classifier.predict(features), dtype=int)


def build_sparse_segments(
    records: Sequence[Mapping[str, Any]],
    gold: Mapping[int, Sequence[Span]],
    ids: Collection[int],
    *,
    granularity: str = "connective",
) -> list[SparseSegment]:
    wanted = set(ids)
    seen: set[int] = set()
    examples: list[SparseSegment] = []
    for record in records:
        paragraph_id = record["paragraph_id"]
        if paragraph_id not in wanted:
            continue
        if paragraph_id in seen:
            raise ValueError(f"duplicate paragraph ID {paragraph_id}")
        seen.add(paragraph_id)
        text = record["text"]
        spans = list(gold[paragraph_id])
        for candidate in segment_paragraph(text, granularity):
            examples.append(
                SparseSegment(
                    paragraph_id=paragraph_id,
                    paragraph_text=text,
                    genre=record["type"],
                    start=candidate.start,
                    end=candidate.end,
                    text=candidate.text,
                    label=oracle_label(candidate, spans) or NONE_LABEL,
                )
            )
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"Task 2 records are missing requested IDs: {missing}")
    return examples


def render_sparse_segment(example: SparseSegment, context_chars: int = 0) -> str:
    """Return target-only text or a marked, source-preserving context feature."""

    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")
    if context_chars == 0:
        return example.text
    left = max(0, example.start - context_chars)
    right = min(len(example.paragraph_text), example.end + context_chars)
    return (
        example.paragraph_text[left : example.start]
        + " <المقطع> "
        + example.text
        + " </المقطع> "
        + example.paragraph_text[example.end : right]
    )


def fit_task1_sparse(
    records_by_id: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, Collection[str]],
    paragraph_ids: Sequence[int],
    config: Task1SparseConfig,
) -> Task1SparseModel:
    texts = [records_by_id[paragraph_id]["text"] for paragraph_id in paragraph_ids]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(config.char_ngram_min, config.char_ngram_max),
        min_df=config.min_df,
        sublinear_tf=config.sublinear_tf,
        dtype=np.float64,
    )
    features = vectorizer.fit_transform(texts)
    classifiers: list[BaseEstimator | ConstantBinaryMargin] = []
    for label in LABELS:
        targets = np.asarray(
            [label in gold[paragraph_id] for paragraph_id in paragraph_ids],
            dtype=int,
        )
        unique = np.unique(targets)
        if len(unique) == 1:
            classifiers.append(ConstantBinaryMargin(bool(unique[0])))
            continue
        classifier = LinearSVC(
            C=config.c,
            class_weight=config.class_weight,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        classifier.fit(features, targets)
        classifiers.append(classifier)
    return Task1SparseModel(vectorizer, tuple(classifiers))


def task2_vectorizer(config: Task2SparseConfig) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "character",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(config.char_ngram_min, config.char_ngram_max),
                    min_df=config.min_df,
                    sublinear_tf=config.sublinear_tf,
                    dtype=np.float64,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(config.word_ngram_min, config.word_ngram_max),
                    min_df=config.min_df,
                    sublinear_tf=config.sublinear_tf,
                    dtype=np.float64,
                ),
            ),
        ]
    )


def fit_task2_sparse(
    examples: Sequence[SparseSegment],
    config: Task2SparseConfig,
) -> Task2SparseModel:
    if not examples:
        raise ValueError("cannot train Task 2 on zero segments")
    inputs = [
        render_sparse_segment(example, config.context_chars) for example in examples
    ]
    targets = np.asarray([example.label_index for example in examples], dtype=int)
    missing = [
        label
        for index, label in enumerate(SEGMENT_LABELS)
        if not bool(np.any(targets == index))
    ]
    if missing:
        raise ValueError(f"Task 2 training segments are missing classes: {missing}")
    vectorizer = task2_vectorizer(config)
    features = vectorizer.fit_transform(inputs)
    classifier = LinearSVC(
        C=config.c,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    classifier.fit(features, targets)
    return Task2SparseModel(vectorizer, classifier)


def binary_f1(gold: np.ndarray, predicted: np.ndarray) -> float:
    gold = np.asarray(gold, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if gold.shape != predicted.shape:
        raise ValueError(f"binary shape mismatch: {gold.shape} != {predicted.shape}")
    true_positive = int(np.sum(gold & predicted))
    false_positive = int(np.sum(~gold & predicted))
    false_negative = int(np.sum(gold & ~predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def tune_binary_threshold(
    gold: Sequence[int | bool | float], scores: Sequence[float]
) -> tuple[float, float]:
    gold_array = np.asarray(gold, dtype=bool)
    score_array = np.asarray(scores, dtype=float)
    if gold_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("gold and scores must be one-dimensional")
    if len(gold_array) != len(score_array) or not len(gold_array):
        raise ValueError("gold and scores must have equal non-zero length")
    if not np.isfinite(score_array).all():
        raise ValueError("threshold scores must all be finite")
    candidates = np.unique(
        np.concatenate(
            [
                score_array,
                [
                    np.nextafter(score_array.min(), -math.inf),
                    np.nextafter(score_array.max(), math.inf),
                ],
            ]
        )
    )
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        score = binary_f1(gold_array, score_array >= threshold)
        if score > best_score or (
            math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-15)
            and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def tune_task1_thresholds(
    gold: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    gold = np.asarray(gold)
    scores = np.asarray(scores, dtype=float)
    expected = (len(gold), len(LABELS))
    if gold.shape != expected or scores.shape != expected:
        raise ValueError(
            f"expected Task 1 gold/scores shape {expected}, got "
            f"{gold.shape}/{scores.shape}"
        )
    thresholds = np.zeros(len(LABELS), dtype=float)
    per_label = {}
    for index, label in enumerate(LABELS):
        threshold, score = tune_binary_threshold(gold[:, index], scores[:, index])
        thresholds[index] = threshold
        per_label[label] = score
    return thresholds, per_label


def cross_fitted_task1_predictions(
    gold: np.ndarray,
    scores: np.ndarray,
    fold_indices: Sequence[int],
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    gold = np.asarray(gold)
    scores = np.asarray(scores, dtype=float)
    folds = np.asarray(fold_indices, dtype=int)
    expected = (len(gold), len(LABELS))
    if gold.shape != expected or scores.shape != expected:
        raise ValueError(
            f"expected Task 1 gold/scores shape {expected}, got "
            f"{gold.shape}/{scores.shape}"
        )
    if len(folds) != len(gold):
        raise ValueError("fold_indices length must equal number of examples")
    unique = sorted(set(folds.tolist()))
    if len(unique) < 2:
        raise ValueError("cross-fitted thresholds require at least two folds")
    predictions = np.zeros_like(gold, dtype=bool)
    thresholds_by_fold: dict[int, dict[str, float]] = {}
    for fold in unique:
        calibration = folds != fold
        held_out = folds == fold
        thresholds, _ = tune_task1_thresholds(gold[calibration], scores[calibration])
        predictions[held_out] = scores[held_out] >= thresholds
        thresholds_by_fold[fold] = {
            label: float(threshold)
            for label, threshold in zip(LABELS, thresholds)
        }
    return predictions, thresholds_by_fold


def task1_prediction_map(
    paragraph_ids: Sequence[int], decisions: np.ndarray
) -> dict[int, set[str]]:
    decisions = np.asarray(decisions, dtype=bool)
    expected = (len(paragraph_ids), len(LABELS))
    if decisions.shape != expected:
        raise ValueError(f"expected Task 1 decision shape {expected}")
    return {
        paragraph_id: {
            label
            for index, label in enumerate(LABELS)
            if decisions[row_index, index]
        }
        for row_index, paragraph_id in enumerate(paragraph_ids)
    }


def segment_prediction_map(
    examples: Sequence[SparseSegment],
    predicted_indices: Sequence[int],
    paragraph_ids: Collection[int],
) -> dict[int, list[Span]]:
    if len(examples) != len(predicted_indices):
        raise ValueError("segment example/prediction length mismatch")
    output = {paragraph_id: [] for paragraph_id in paragraph_ids}
    seen: set[tuple[int, int, int, str]] = set()
    for example, raw_index in zip(examples, predicted_indices):
        index = int(raw_index)
        if not 0 <= index < len(SEGMENT_LABELS):
            raise ValueError(f"invalid segment prediction index: {index}")
        label = SEGMENT_LABELS[index]
        if label == NONE_LABEL:
            continue
        key = (example.paragraph_id, example.start, example.end, label)
        if key in seen:
            continue
        seen.add(key)
        output[example.paragraph_id].append(Span(example.start, example.end, label))
    label_order = {label: index for index, label in enumerate(LABELS)}
    for spans in output.values():
        spans.sort(key=lambda span: (span.start, span.end, label_order[span.label]))
    return output


__all__ = [
    "NONE_LABEL",
    "SEGMENT_LABELS",
    "SparseSegment",
    "Task1SparseConfig",
    "Task1SparseModel",
    "Task2SparseConfig",
    "Task2SparseModel",
    "binary_f1",
    "build_sparse_segments",
    "cross_fitted_task1_predictions",
    "fit_task1_sparse",
    "fit_task2_sparse",
    "render_sparse_segment",
    "segment_prediction_map",
    "task1_prediction_map",
    "task2_vectorizer",
    "tune_binary_threshold",
    "tune_task1_thresholds",
]
