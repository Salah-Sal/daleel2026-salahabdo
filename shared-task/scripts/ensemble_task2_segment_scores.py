"""Average compatible Task 2 segment-encoder score runs.

Inputs may be five-fold OOF runs (``oof_task_2_segment_scores.jsonl``) or
full-train target deploys (``task_2_segment_scores.jsonl``).  Every segment
key and all non-probability metadata must match across seeds.  The command
then averages all seven softmax probabilities and emits an experiment with
the same score-file interface consumed by the S1 structural decoder.

Examples (from ``shared-task/``)::

    uv run scripts/ensemble_task2_segment_scores.py \
      --runs experiments/seed-a experiments/seed-b --name quarter-oof-2seed

    uv run scripts/ensemble_task2_segment_scores.py \
      --runs experiments/deploy-a experiments/deploy-b --name quarter-test-2seed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.data import TRAIN_TASK2, load_records
from daleel.encoder_baseline import SEGMENT_LABELS
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold

SCORE_NAMES = ("oof_task_2_segment_scores.jsonl", "task_2_segment_scores.jsonl")


def find_scores(run: Path) -> tuple[Path, str]:
    for name in SCORE_NAMES:
        for candidate in (run / "predictions" / name, run / name):
            if candidate.is_file():
                return candidate, name
    raise SystemExit(f"no Task 2 segment score file under {run}")


def segment_key(row: Mapping) -> tuple[int, int, int]:
    return (
        int(row["paragraph_id"]),
        int(row["start_offset"]),
        int(row["end_offset"]),
    )


def load_run(run: Path) -> tuple[list[dict], Path, str]:
    path, name = find_scores(run)
    rows = read_jsonl(path)
    keys = [segment_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{run}: duplicate segment keys")
    for row in rows:
        missing = set(SEGMENT_LABELS) - set(row["scores"])
        if missing:
            raise SystemExit(f"{run}: segment {segment_key(row)} lacks {sorted(missing)}")
    return rows, path, name


def validate_contract(base: Sequence[dict], candidate: Sequence[dict], run: Path) -> None:
    if len(candidate) != len(base):
        raise SystemExit(f"{run}: segment count differs ({len(candidate)} != {len(base)})")
    base_by_key = {segment_key(row): row for row in base}
    candidate_by_key = {segment_key(row): row for row in candidate}
    if set(candidate_by_key) != set(base_by_key):
        raise SystemExit(f"{run}: segment boundary set differs")
    for key, left in base_by_key.items():
        right = candidate_by_key[key]
        for field in ("fold", "gold_oracle_label"):
            if left.get(field) != right.get(field):
                raise SystemExit(
                    f"{run}: segment {key} differs on {field}: "
                    f"{left.get(field)!r} != {right.get(field)!r}"
                )


def prediction_map(rows: Sequence[dict]) -> dict[int, list[Span]]:
    output: defaultdict[int, list[Span]] = defaultdict(list)
    seen = set()
    for row in rows:
        label = row["predicted_label"]
        pid, start, end = segment_key(row)
        output.setdefault(pid, [])
        if label == "NONE":
            continue
        key = (pid, start, end, label)
        if key not in seen:
            seen.add(key)
            output[pid].append(Span(start, end, label))
    for spans in output.values():
        spans.sort(key=lambda span: (span.start, span.end, span.label))
    return dict(output)


def oof_metrics(rows: Sequence[dict]) -> dict:
    train = load_records(TRAIN_TASK2)
    gold_all = clean_task2_gold(train)
    pids = sorted({int(row["paragraph_id"]) for row in rows})
    gold = {pid: gold_all[pid] for pid in pids}
    genre = {int(row["paragraph_id"]): row["type"] for row in train}
    pred = prediction_map(rows)
    result = {"overall": span_partial_f1(gold, pred)}
    for name in ("editorial", "debate"):
        ids = [pid for pid in pids if genre[pid] == name]
        result[name] = span_partial_f1(
            {pid: gold[pid] for pid in ids}, {pid: pred.get(pid, ()) for pid in ids}
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help=(
            "optional non-negative run weights in the same order as --runs; "
            "weights are normalized to sum to one (default: uniform mean)"
        ),
    )
    args = parser.parse_args()
    if len(args.runs) < 2:
        raise SystemExit("an ensemble needs at least two runs")
    if args.weights is not None and len(args.weights) != len(args.runs):
        raise SystemExit("--weights must provide exactly one value per --runs entry")
    raw_weights = args.weights or [1.0] * len(args.runs)
    if any(weight < 0.0 for weight in raw_weights):
        raise SystemExit("--weights values must be non-negative")
    weight_sum = float(sum(raw_weights))
    if weight_sum <= 0.0:
        raise SystemExit("--weights must contain at least one positive value")
    weights = [float(weight) / weight_sum for weight in raw_weights]

    loaded = [load_run(run) for run in args.runs]
    names = {name for _, _, name in loaded}
    if len(names) != 1:
        raise SystemExit(f"cannot mix OOF and target deploy score files: {sorted(names)}")
    output_name = names.pop()
    base_rows = loaded[0][0]
    for run, (rows, _, _) in zip(args.runs[1:], loaded[1:]):
        validate_contract(base_rows, rows, run)

    row_maps = [{segment_key(row): row for row in rows} for rows, _, _ in loaded]
    ensemble_rows = []
    for base in base_rows:
        key = segment_key(base)
        mean = {
            label: float(
                np.average(
                    [rows[key]["scores"][label] for rows in row_maps],
                    weights=weights,
                )
            )
            for label in SEGMENT_LABELS
        }
        predicted = max(SEGMENT_LABELS, key=lambda label: mean[label])
        ensemble_rows.append(base | {"predicted_label": predicted, "scores": mean})

    resolved_config = {
        "script": "ensemble_task2_segment_scores.py",
        "argv": sys.argv[1:],
        "method": "weighted arithmetic mean of seven-way segment softmax probabilities",
        "weights": weights,
        "kind": "oof" if output_name.startswith("oof_") else "target-deploy",
        "inputs": [str(run) for run in args.runs],
        "input_score_sha256": [file_sha256(path) for _, path, _ in loaded],
        "segment_contract": "identical (paragraph_id,start_offset,end_offset,fold,gold_oracle_label)",
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t2-encoder-ensemble-{args.name}-s{len(args.runs)}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    output_path = exp_dir / "predictions" / output_name
    write_jsonl(output_path, ensemble_rows)

    metrics = {
        "n_seeds": len(args.runs),
        "n_segments": len(ensemble_rows),
        "n_paragraphs": len({row["paragraph_id"] for row in ensemble_rows}),
        "predicted_segment_class_counts": dict(
            sorted(Counter(row["predicted_label"] for row in ensemble_rows).items())
        ),
    }
    if output_name.startswith("oof_"):
        metrics["official"] = oof_metrics(ensemble_rows)
        metrics["primary"] = {
            "name": "official_task2_partial_f1_oof",
            "score": metrics["official"]["overall"]["f1"],
        }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nensemble run dir: {exp_dir}")


if __name__ == "__main__":
    main()
