"""Cross-fit a dev-supervised Task 2 span relabel/drop stacker and deploy it.

The stacker learns only on organizer-released references.  Its candidates and
boundaries are frozen by an existing Task 2 system; it may retain, relabel, or
drop each candidate.  Features are the candidate's current role, CAMeLBERT
overlap-weighted role posteriors, Task 1 label presence, genre, position,
length, and neighboring roles.  Paragraph-grouped five-fold predictions give
the development-set measurement used to choose a small regularization grid.

This is an evaluation-week transductive-selection tool, not an OOF estimate
over the original training pool.  Reports name the released dev references
and preserve the exact target prediction provenance supplied on the command
line.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from daleel.constants import LABELS
from daleel.folds import make_stratified_folds
from daleel.metrics import Span, span_partial_f1

from t2_eval_week_filter import (
    compact_scores,
    load_span_records,
    load_t1_labels,
    score_slices,
    write_predictions,
)
from t2_structural_gate import dedup_spans, dominant_label, load_segments, span_encoder_dist

CLASSES = (*LABELS, "NONE")
CLASS_INDEX = {label: index for index, label in enumerate(CLASSES)}
TAU_GRID = tuple(round(0.50 + 0.05 * step, 2) for step in range(10))
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
WEIGHT_GRID = (None, "balanced")


def onehot(label: str | None) -> list[float]:
    values = [0.0] * len(CLASSES)
    values[CLASS_INDEX.get(label, CLASS_INDEX["NONE"])] = 1.0
    return values


def build_features(
    spans: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    task1: Mapping[int, set[str]],
    segments: Mapping[int, Sequence[tuple]],
) -> dict[int, np.ndarray]:
    output = {}
    for pid, paragraph_spans in spans.items():
        text_length = max(len(records[pid]["text"]), 1)
        rows = []
        for index, span in enumerate(paragraph_spans):
            prev_label = paragraph_spans[index - 1].label if index else None
            next_label = (
                paragraph_spans[index + 1].label
                if index + 1 < len(paragraph_spans)
                else None
            )
            encoder = span_encoder_dist(segments.get(pid, ()), span)
            rows.append(
                [encoder[label] for label in LABELS]
                + onehot(span.label)
                + [float(label in task1[pid]) for label in LABELS]
                + [
                    float(records[pid]["type"] == "editorial"),
                    math.log(max(len(span), 1)),
                    span.start / text_length,
                    float(index == 0),
                    float(index + 1 == len(paragraph_spans)),
                    math.log1p(len(paragraph_spans)),
                ]
                + onehot(prev_label)
                + onehot(next_label)
            )
        output[pid] = np.asarray(rows, dtype=float)
    return output


def dominant_target(gold: Sequence[Span], span: Span) -> str:
    return dominant_label(gold, span) or "NONE"


def flatten_training(
    pids: Sequence[int],
    spans: Mapping[int, Sequence[Span]],
    features: Mapping[int, np.ndarray],
    gold: Mapping[int, Sequence[Span]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [features[pid] for pid in pids if len(features[pid])]
    X = np.concatenate(rows, axis=0)
    y = np.asarray(
        [
            dominant_target(gold.get(pid, ()), span)
            for pid in pids
            for span in spans[pid]
        ]
    )
    return X, y


def fit_model(X: np.ndarray, y: np.ndarray, c_value: float, class_weight: str | None):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            C=c_value,
            class_weight=class_weight,
            max_iter=2000,
        ),
    )
    model.fit(X, y)
    return model


def probabilities_by_pid(model, pids: Sequence[int], features: Mapping[int, np.ndarray]):
    classes = tuple(model[-1].classes_)
    output = {}
    for pid in pids:
        if not len(features[pid]):
            output[pid] = np.empty((0, len(classes)), dtype=float)
        else:
            output[pid] = model.predict_proba(features[pid])
    return output, classes


def decode(
    pids: Sequence[int],
    spans: Mapping[int, Sequence[Span]],
    probabilities: Mapping[int, np.ndarray],
    classes: Sequence[str],
    tau: float,
) -> dict[int, list[Span]]:
    output = {}
    for pid in pids:
        decoded = []
        for span, row in zip(spans[pid], probabilities[pid]):
            index = int(np.argmax(row))
            top = classes[index]
            label = top if top != span.label and float(row[index]) >= tau else span.label
            if label != "NONE":
                decoded.append(Span(span.start, span.end, label))
        output[pid] = dedup_spans(decoded)
    return output


def select_tau(
    pids: Sequence[int],
    spans: Mapping[int, Sequence[Span]],
    probabilities: Mapping[int, np.ndarray],
    classes: Sequence[str],
    gold: Mapping[int, Sequence[Span]],
) -> tuple[float, float]:
    sub_gold = {pid: gold[pid] for pid in pids}
    scored = []
    for tau in TAU_GRID:
        pred = decode(pids, spans, probabilities, classes, tau)
        score = span_partial_f1(sub_gold, pred)["f1"]
        scored.append((score, tau))
    score, tau = max(scored, key=lambda item: (item[0], item[1]))
    return tau, score


def cross_fit_configuration(
    folds,
    spans,
    features,
    gold,
    c_value,
    class_weight,
):
    predictions = {}
    fold_report = {}
    class_counts = Counter()
    for fold in folds:
        train_ids = list(fold.train_ids)
        val_ids = list(fold.val_ids)
        X_train, y_train = flatten_training(train_ids, spans, features, gold)
        class_counts.update(y_train.tolist())
        model = fit_model(X_train, y_train, c_value, class_weight)
        train_prob, classes = probabilities_by_pid(model, train_ids, features)
        tau, fit_score = select_tau(
            train_ids, spans, train_prob, classes, gold
        )
        val_prob, val_classes = probabilities_by_pid(model, val_ids, features)
        if tuple(val_classes) != tuple(classes):
            raise AssertionError("classifier classes changed between train and validation")
        fold_pred = decode(val_ids, spans, val_prob, classes, tau)
        predictions.update(fold_pred)
        fold_report[str(fold.index)] = {
            "tau": tau,
            "fit_f1_insample": round(fit_score, 6),
            "validation_f1": round(
                span_partial_f1(
                    {pid: gold[pid] for pid in val_ids}, fold_pred
                )["f1"],
                6,
            ),
            "n_train_paragraphs": len(train_ids),
            "n_validation_paragraphs": len(val_ids),
        }
    score = span_partial_f1(gold, predictions)["f1"]
    return predictions, score, fold_report, dict(sorted(class_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-base", type=Path, required=True)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--train-t1", type=Path, required=True)
    parser.add_argument("--train-encoder-scores", type=Path, required=True)
    parser.add_argument("--target-base", type=Path)
    parser.add_argument("--target-t1", type=Path)
    parser.add_argument("--target-encoder-scores", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--crossfit-output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    train_spans, train_records = load_span_records(args.train_base)
    gold, _ = load_span_records(args.train_gold)
    task1 = load_t1_labels(args.train_t1)
    segments = load_segments(args.train_encoder_scores)
    pids = sorted(train_spans)
    missing = sorted(set(pids) - set(gold))
    if missing:
        raise ValueError(f"{len(missing)} training paragraphs lack gold")
    features = build_features(train_spans, train_records, task1, segments)
    labels = {pid: {span.label for span in gold[pid]} for pid in pids}
    genres = {pid: train_records[pid]["type"] for pid in pids}
    folds = make_stratified_folds(pids, labels, genres, n_splits=5, seed=20260730)

    base_scores = score_slices(gold, train_spans, train_records)
    grid = []
    predictions_by_config = {}
    for class_weight in WEIGHT_GRID:
        for c_value in C_GRID:
            pred, score, fold_report, counts = cross_fit_configuration(
                folds,
                train_spans,
                features,
                gold,
                c_value,
                class_weight,
            )
            key = (class_weight or "none", c_value)
            predictions_by_config[key] = pred
            grid.append(
                {
                    "class_weight": class_weight or "none",
                    "C": c_value,
                    "crossfit_f1": round(score, 6),
                    "folds": fold_report,
                    "training_target_counts_accumulated_across_folds": counts,
                }
            )
    grid.sort(
        key=lambda row: (
            row["crossfit_f1"],
            row["class_weight"] == "none",
            -row["C"],
        ),
        reverse=True,
    )
    best = grid[0]
    best_key = (best["class_weight"], best["C"])
    crossfit = predictions_by_config[best_key]
    crossfit_scores = score_slices(gold, crossfit, train_records)

    report = {
        "method": "dev-supervised paragraph-grouped five-fold logistic stacker",
        "train": {
            "base": str(args.train_base),
            "gold": str(args.train_gold),
            "t1": str(args.train_t1),
            "encoder_scores": str(args.train_encoder_scores),
            "n_paragraphs": len(pids),
            "n_candidate_spans": sum(len(spans) for spans in train_spans.values()),
        },
        "base_scores": compact_scores(base_scores),
        "grid": grid,
        "selected": {
            "class_weight": best["class_weight"],
            "C": best["C"],
            "scores": compact_scores(crossfit_scores),
        },
    }
    if args.crossfit_output:
        write_predictions(args.crossfit_output, crossfit, train_records)
        report["crossfit_output"] = str(args.crossfit_output)

    target_args = (
        args.target_base,
        args.target_t1,
        args.target_encoder_scores,
        args.output,
    )
    if any(target_args) and not all(target_args):
        raise ValueError("target deployment requires all three target inputs and --output")
    if all(target_args):
        X_train, y_train = flatten_training(pids, train_spans, features, gold)
        model = fit_model(
            X_train,
            y_train,
            best["C"],
            None if best["class_weight"] == "none" else best["class_weight"],
        )
        train_prob, classes = probabilities_by_pid(model, pids, features)
        tau, fit_score = select_tau(pids, train_spans, train_prob, classes, gold)

        target_spans, target_records = load_span_records(args.target_base)
        target_task1 = load_t1_labels(args.target_t1)
        target_segments = load_segments(args.target_encoder_scores)
        target_features = build_features(
            target_spans, target_records, target_task1, target_segments
        )
        target_ids = sorted(target_spans)
        target_prob, target_classes = probabilities_by_pid(
            model, target_ids, target_features
        )
        target_pred = decode(
            target_ids, target_spans, target_prob, target_classes, tau
        )
        write_predictions(args.output, target_pred, target_records)
        report["deployment"] = {
            "target_base": str(args.target_base),
            "target_t1": str(args.target_t1),
            "target_encoder_scores": str(args.target_encoder_scores),
            "output": str(args.output),
            "tau_selected_insample": tau,
            "fit_f1_insample": round(fit_score, 6),
            "n_input_spans": sum(len(spans) for spans in target_spans.values()),
            "n_output_spans": sum(len(spans) for spans in target_pred.values()),
            "changed_span_count": sum(
                len(set(target_spans[pid]) ^ set(target_pred[pid]))
                for pid in target_ids
            ),
        }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
