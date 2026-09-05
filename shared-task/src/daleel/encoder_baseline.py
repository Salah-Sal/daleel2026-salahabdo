"""Non-generative Arabic encoder baselines for Daleel Tasks 1 and 2.

This module deliberately has no DSPy dependency. It provides paragraph-
grouped Task 1 examples, window-aware batching, source-preserving Task 2
segment examples, compact encoder heads, deterministic threshold calibration,
and conversion back to the official Daleel metric shapes.

Training orchestration lives in scripts/train_encoder_baseline.py so the
data and decoding contracts can be tested without launching a model run.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .constants import LABELS
from .metrics import Span
from .segment import oracle_label, segment_paragraph


NONE_LABEL = "NONE"
SEGMENT_LABELS = LABELS + (NONE_LABEL,)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
SEGMENT_LABEL_TO_INDEX = {
    label: index for index, label in enumerate(SEGMENT_LABELS)
}

GENRE_TEXT = {
    "editorial": "مقال افتتاحي",
    "debate": "مناظرة",
}


@dataclass(frozen=True, slots=True)
class ParagraphExample:
    """One Task 1 paragraph and its six binary targets."""

    paragraph_id: int
    text: str
    genre: str
    labels: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.labels) != len(LABELS):
            raise ValueError(
                f"paragraph {self.paragraph_id}: expected {len(LABELS)} targets, "
                f"got {len(self.labels)}"
            )
        if self.genre not in GENRE_TEXT:
            raise ValueError(f"unknown paragraph genre: {self.genre!r}")


@dataclass(frozen=True, slots=True)
class SegmentExample:
    """One exact Task 2 candidate segment and its single training role."""

    paragraph_id: int
    paragraph_text: str
    genre: str
    start: int
    end: int
    text: str
    label: str
    position_bucket: int

    def __post_init__(self) -> None:
        if self.genre not in GENRE_TEXT:
            raise ValueError(f"unknown paragraph genre: {self.genre!r}")
        if self.label not in SEGMENT_LABEL_TO_INDEX:
            raise ValueError(f"unknown segment label: {self.label!r}")
        if not 0 <= self.start < self.end <= len(self.paragraph_text):
            raise ValueError(
                f"paragraph {self.paragraph_id}: invalid segment "
                f"[{self.start}, {self.end})"
            )
        if self.paragraph_text[self.start : self.end] != self.text:
            raise ValueError(
                f"paragraph {self.paragraph_id}: segment text does not match source"
            )
        if not 0 <= self.position_bucket <= 9:
            raise ValueError("position_bucket must be in [0, 9]")

    @property
    def label_index(self) -> int:
        return SEGMENT_LABEL_TO_INDEX[self.label]


def build_paragraph_examples(
    records: Sequence[dict[str, Any]],
    gold: Mapping[int, Collection[str]],
    ids: Collection[int],
) -> list[ParagraphExample]:
    """Build Task 1 examples in stable source-record order."""

    wanted = set(ids)
    examples: list[ParagraphExample] = []
    seen: set[int] = set()
    for record in records:
        paragraph_id = record["paragraph_id"]
        if paragraph_id not in wanted:
            continue
        if paragraph_id in seen:
            raise ValueError(f"duplicate paragraph ID {paragraph_id}")
        seen.add(paragraph_id)
        labels = set(gold[paragraph_id])
        examples.append(
            ParagraphExample(
                paragraph_id=paragraph_id,
                text=record["text"],
                genre=record["type"],
                labels=tuple(float(label in labels) for label in LABELS),
            )
        )
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"paragraph records are missing requested IDs: {missing}")
    return examples


def build_segment_examples(
    records: Sequence[dict[str, Any]],
    gold: Mapping[int, Sequence[Span]],
    ids: Collection[int],
    *,
    granularity: str = "connective",
) -> list[SegmentExample]:
    """Build deterministic source-preserving Task 2 segment examples."""

    wanted = set(ids)
    examples: list[SegmentExample] = []
    seen: set[int] = set()
    for record in records:
        paragraph_id = record["paragraph_id"]
        if paragraph_id not in wanted:
            continue
        if paragraph_id in seen:
            raise ValueError(f"duplicate paragraph ID {paragraph_id}")
        seen.add(paragraph_id)
        text = record["text"]
        spans = gold[paragraph_id]
        for segment in segment_paragraph(text, granularity):
            label = oracle_label(segment, list(spans)) or NONE_LABEL
            position_bucket = min(
                9,
                math.floor(10 * segment.start / max(1, len(text))),
            )
            examples.append(
                SegmentExample(
                    paragraph_id=paragraph_id,
                    paragraph_text=text,
                    genre=record["type"],
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    label=label,
                    position_bucket=position_bucket,
                )
            )
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"Task 2 records are missing requested IDs: {missing}")
    return examples


def render_segment_input(
    example: SegmentExample,
    *,
    context_chars: int = 0,
) -> str:
    """Render a classifier input while keeping the target first.

    Target-first ordering ensures that truncation removes auxiliary context
    before it removes the segment being classified. The rendered string is a
    model feature only; output offsets always come from SegmentExample.
    """

    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")
    lines = [
        example.text,
        f"نوع النص: {GENRE_TEXT[example.genre]}",
        f"الموضع: {example.position_bucket + 1} من 10",
    ]
    if context_chars:
        left = max(0, example.start - context_chars)
        right = min(len(example.paragraph_text), example.end + context_chars)
        context = (
            example.paragraph_text[left : example.start]
            + " <المقطع> "
            + example.text
            + " </المقطع> "
            + example.paragraph_text[example.end : right]
        )
        lines.append(f"السياق: {context}")
    return "\n".join(lines)


class ParagraphDataset(torch.utils.data.Dataset):
    """Thin immutable wrapper over Task 1 examples."""

    def __init__(self, examples: Sequence[ParagraphExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ParagraphExample:
        return self.examples[index]


class SegmentDataset(torch.utils.data.Dataset):
    """Thin immutable wrapper over Task 2 examples."""

    def __init__(self, examples: Sequence[SegmentExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SegmentExample:
        return self.examples[index]


class ParagraphCollator:
    """Tokenize paragraphs into overflow windows and flatten one batch."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        stride: int,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        if not 0 <= stride < max_length - 2:
            raise ValueError("stride must lie in [0, max_length - 2)")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[ParagraphExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty paragraph batch")
        windows: list[dict[str, list[int]]] = []
        window_to_paragraph: list[int] = []
        for paragraph_index, example in enumerate(examples):
            encoded = self.tokenizer(
                example.text,
                truncation=True,
                max_length=self.max_length,
                stride=self.stride,
                return_overflowing_tokens=True,
                return_attention_mask=True,
                add_special_tokens=True,
            )
            n_windows = len(encoded["input_ids"])
            if n_windows == 0:
                raise ValueError(
                    f"paragraph {example.paragraph_id} produced zero tokenizer windows"
                )
            for window_index in range(n_windows):
                window = {
                    key: encoded[key][window_index]
                    for key in ("input_ids", "attention_mask", "token_type_ids")
                    if key in encoded
                }
                windows.append(window)
                window_to_paragraph.append(paragraph_index)
        inputs = self.tokenizer.pad(
            windows,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "inputs": dict(inputs),
            "window_to_paragraph": torch.tensor(
                window_to_paragraph, dtype=torch.long
            ),
            "labels": torch.tensor(
                [example.labels for example in examples], dtype=torch.float32
            ),
            "paragraph_ids": [example.paragraph_id for example in examples],
            "window_counts": [
                window_to_paragraph.count(index) for index in range(len(examples))
            ],
        }


class SegmentCollator:
    """Render and tokenize exact candidate segments."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        context_chars: int = 0,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        if context_chars < 0:
            raise ValueError("context_chars must be non-negative")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_chars = context_chars
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: Sequence[SegmentExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty segment batch")
        rendered = [
            render_segment_input(example, context_chars=self.context_chars)
            for example in examples
        ]
        inputs = self.tokenizer(
            rendered,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "inputs": dict(inputs),
            "labels": torch.tensor(
                [example.label_index for example in examples], dtype=torch.long
            ),
            "examples": list(examples),
        }


class ParagraphEncoderClassifier(nn.Module):
    """Arabic encoder plus a six-logit, overflow-window-aware Task 1 head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        dropout: float,
        window_pool: Literal["max", "mean"] = "max",
    ) -> None:
        super().__init__()
        if window_pool not in {"max", "mean"}:
            raise ValueError(f"unsupported window pool: {window_pool!r}")
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABELS))
        self.window_pool = window_pool

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        window_to_paragraph: torch.Tensor,
        *,
        n_paragraphs: int,
    ) -> torch.Tensor:
        output = self.encoder(**inputs)
        pooled = self.dropout(output.last_hidden_state[:, 0])
        window_logits = self.classifier(pooled)
        paragraph_logits: list[torch.Tensor] = []
        for paragraph_index in range(n_paragraphs):
            mask = window_to_paragraph == paragraph_index
            if not bool(mask.any()):
                raise ValueError(
                    f"paragraph index {paragraph_index} has no encoded window"
                )
            values = window_logits[mask]
            if self.window_pool == "max":
                paragraph_logits.append(values.max(dim=0).values)
            else:
                paragraph_logits.append(values.mean(dim=0))
        return torch.stack(paragraph_logits)


class SegmentEncoderClassifier(nn.Module):
    """Arabic encoder plus a seven-way Task 2 segment-role head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(SEGMENT_LABELS))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**inputs)
        pooled = self.dropout(output.last_hidden_state[:, 0])
        return self.classifier(pooled)


def binary_f1(gold: np.ndarray, predicted: np.ndarray) -> float:
    """Binary F1 with the official zero-division convention."""

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
    gold: Sequence[int | bool | float],
    scores: Sequence[float],
) -> tuple[float, float]:
    """Return the exact score-cut threshold maximizing binary F1.

    Every unique observed score is a candidate because predictions change
    only at those points. The all-negative decision is also considered. Exact
    F1 ties choose the higher, more conservative threshold.
    """

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
                np.array(
                    [
                        np.nextafter(score_array.max(), math.inf),
                        np.nextafter(score_array.min(), -math.inf),
                    ]
                ),
            ]
        )
    )
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        score = binary_f1(gold_array, score_array >= threshold)
        if score > best_f1 or (
            math.isclose(score, best_f1, rel_tol=0.0, abs_tol=1e-15)
            and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def tune_task1_thresholds(
    gold_matrix: np.ndarray,
    score_matrix: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Tune one independent threshold per official Task 1 label."""

    gold_matrix = np.asarray(gold_matrix)
    score_matrix = np.asarray(score_matrix, dtype=float)
    expected = (len(gold_matrix), len(LABELS))
    if gold_matrix.shape != expected or score_matrix.shape != expected:
        raise ValueError(
            f"expected gold/scores shape {expected}; got "
            f"{gold_matrix.shape}/{score_matrix.shape}"
        )
    thresholds = np.zeros(len(LABELS), dtype=float)
    f1_by_label: dict[str, float] = {}
    for index, label in enumerate(LABELS):
        threshold, score = tune_binary_threshold(
            gold_matrix[:, index],
            score_matrix[:, index],
        )
        thresholds[index] = threshold
        f1_by_label[label] = score
    return thresholds, f1_by_label


def cross_fitted_task1_predictions(
    gold_matrix: np.ndarray,
    score_matrix: np.ndarray,
    fold_indices: Sequence[int],
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    """Apply thresholds fitted on other folds to each held-out fold."""

    gold_matrix = np.asarray(gold_matrix)
    score_matrix = np.asarray(score_matrix, dtype=float)
    folds = np.asarray(fold_indices, dtype=int)
    if len(folds) != len(gold_matrix):
        raise ValueError("fold_indices length must equal number of examples")
    unique_folds = sorted(set(folds.tolist()))
    if len(unique_folds) < 2:
        raise ValueError("cross-fitted thresholds require at least two folds")
    predicted = np.zeros_like(gold_matrix, dtype=bool)
    thresholds_by_fold: dict[int, dict[str, float]] = {}
    for fold in unique_folds:
        calibration = folds != fold
        held_out = folds == fold
        thresholds, _ = tune_task1_thresholds(
            gold_matrix[calibration],
            score_matrix[calibration],
        )
        predicted[held_out] = score_matrix[held_out] >= thresholds
        thresholds_by_fold[fold] = {
            label: float(threshold)
            for label, threshold in zip(LABELS, thresholds)
        }
    return predicted, thresholds_by_fold


def task1_prediction_map(
    paragraph_ids: Sequence[int],
    decisions: np.ndarray,
) -> dict[int, set[str]]:
    """Convert a Boolean decision matrix into the official mapping shape."""

    decisions = np.asarray(decisions, dtype=bool)
    expected = (len(paragraph_ids), len(LABELS))
    if decisions.shape != expected:
        raise ValueError(f"expected Task 1 decisions shape {expected}")
    return {
        paragraph_id: {
            label
            for index, label in enumerate(LABELS)
            if decisions[row_index, index]
        }
        for row_index, paragraph_id in enumerate(paragraph_ids)
    }


def segment_prediction_map(
    examples: Sequence[SegmentExample],
    predicted_indices: Sequence[int],
    paragraph_ids: Collection[int],
) -> dict[int, list[Span]]:
    """Convert segment class indices into exact scorer-ready spans."""

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
        output[example.paragraph_id].append(
            Span(example.start, example.end, label)
        )
    for spans in output.values():
        spans.sort(key=lambda span: (span.start, span.end, LABEL_TO_INDEX[span.label]))
    return output


def task1_records(
    source_by_id: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[int, Collection[str]],
    paragraph_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Create source-bound Task 1 JSONL records."""

    records = []
    for paragraph_id in paragraph_ids:
        source = source_by_id[paragraph_id]
        records.append(
            {
                "paragraph_id": paragraph_id,
                "text": source["text"],
                "type": source["type"],
                "labels": [
                    label
                    for label in LABELS
                    if label in predictions.get(paragraph_id, ())
                ],
            }
        )
    return records


def task2_records(
    source_by_id: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[int, Sequence[Span]],
    paragraph_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Create source-bound Task 2 JSONL records."""

    records = []
    for paragraph_id in paragraph_ids:
        source = source_by_id[paragraph_id]
        labels = [
            {
                "label": span.label,
                "start_offset": span.start,
                "end_offset": span.end,
            }
            for span in predictions.get(paragraph_id, ())
        ]
        records.append(
            {
                "paragraph_id": paragraph_id,
                "text": source["text"],
                "type": source["type"],
                "labels": labels,
            }
        )
    return records


def score_records(
    paragraph_ids: Sequence[int],
    scores: np.ndarray,
) -> list[dict[str, Any]]:
    """Serialize Task 1 probability scores without losing label order."""

    scores = np.asarray(scores, dtype=float)
    expected = (len(paragraph_ids), len(LABELS))
    if scores.shape != expected:
        raise ValueError(f"expected Task 1 scores shape {expected}")
    return [
        {
            "paragraph_id": paragraph_id,
            "scores": {
                label: float(scores[row_index, label_index])
                for label_index, label in enumerate(LABELS)
            },
        }
        for row_index, paragraph_id in enumerate(paragraph_ids)
    ]


__all__ = [
    "GENRE_TEXT",
    "LABEL_TO_INDEX",
    "NONE_LABEL",
    "ParagraphCollator",
    "ParagraphDataset",
    "ParagraphEncoderClassifier",
    "ParagraphExample",
    "SEGMENT_LABELS",
    "SEGMENT_LABEL_TO_INDEX",
    "SegmentCollator",
    "SegmentDataset",
    "SegmentEncoderClassifier",
    "SegmentExample",
    "binary_f1",
    "build_paragraph_examples",
    "build_segment_examples",
    "cross_fitted_task1_predictions",
    "render_segment_input",
    "score_records",
    "segment_prediction_map",
    "task1_prediction_map",
    "task1_records",
    "task2_records",
    "tune_binary_threshold",
    "tune_task1_thresholds",
]
