"""Mean-sigmoid seed ensemble for Task 1 encoder OOF runs (HEADROOM_AUDIT.md P5).

Takes N runs of the same encoder recipe differing only in --seed, verifies
they share the frozen fold contract, averages the per-label sigmoids, and
re-fits cross-fitted thresholds on the ensembled scores with the exact
trainer machinery (daleel.encoder_baseline.cross_fitted_task1_predictions).
Emits a synthetic experiments/ run dir exposing the router's encoder
interface:

  predictions/oof_task_1_scores.jsonl   {paragraph_id, scores, fold}
  predictions/oof_task_1.jsonl          {paragraph_id, labels}   (text-free)

so route_task1_v2.py can consume the ensemble via --encoder unchanged.

Each input run may be a local experiments dir (files under predictions/) or
a Kaggle campaign export dir (files at the dir root). Per-seed OOF macros
are recomputed with the same threshold machinery and must reproduce the
trainer-reported primaries digit-for-digit; the script reports them so any
drift is visible before the ensemble is trusted.

Example (from shared-task/):
  uv run scripts/ensemble_encoder_seeds.py \
      --runs /path/to/t1-quarter-seed20260710 ... \
      --name quarter-5seed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.encoder_baseline import (
    cross_fitted_task1_predictions,
    task1_prediction_map,
)
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import task1_macro_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids


def scores_path(run: Path) -> Path:
    for candidate in (
        run / "predictions" / "oof_task_1_scores.jsonl",
        run / "oof_task_1_scores.jsonl",
    ):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no oof_task_1_scores.jsonl under {run}")


def load_run(run: Path) -> tuple[dict[int, dict[str, float]], dict[int, int]]:
    scores: dict[int, dict[str, float]] = {}
    fold_of: dict[int, int] = {}
    for row in read_jsonl(scores_path(run)):
        pid = row["paragraph_id"]
        if pid in scores:
            raise SystemExit(f"{run}: duplicate paragraph {pid}")
        scores[pid] = {label: float(row["scores"][label]) for label in LABELS}
        fold_of[pid] = int(row["fold"])
    return scores, fold_of


def oof_macro(
    gold_matrix: np.ndarray,
    score_matrix: np.ndarray,
    folds: list[int],
    pids: list[int],
    gold: dict[int, set[str]],
) -> dict:
    decisions, thresholds_by_fold = cross_fitted_task1_predictions(
        gold_matrix, score_matrix, folds
    )
    predicted = task1_prediction_map(pids, decisions)
    official = task1_macro_f1(gold, predicted)
    return {
        "macro_f1": round(official["macro_f1"], 4),
        "per_label_f1": {
            label: round(v["f1"], 4) for label, v in official["per_label"].items()
        },
        "thresholds_by_fold": {
            str(fold): mapping for fold, mapping in thresholds_by_fold.items()
        },
        "predicted": predicted,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True,
                    help="two or more same-recipe seed runs to ensemble")
    ap.add_argument("--name", required=True,
                    help="short slug describing the ensemble, e.g. quarter-5seed")
    args = ap.parse_args()
    if len(args.runs) < 2:
        raise SystemExit("an ensemble needs at least two seed runs")

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {pid: labels for pid, labels in clean_task1_gold(t1, t2).items()
            if pid in set(train_ids)}

    runs = [load_run(run) for run in args.runs]
    pids = sorted(runs[0][0])
    if set(pids) != set(gold):
        raise SystemExit("seed run does not cover the 430 train-side paragraphs")
    fold_of = runs[0][1]
    for run_path, (scores, folds) in zip(args.runs, runs):
        if sorted(scores) != pids:
            raise SystemExit(f"{run_path}: paragraph set differs across seeds")
        if folds != fold_of:
            raise SystemExit(f"{run_path}: fold assignment differs — fold contract broken")

    gold_matrix = np.array(
        [[label in gold[pid] for label in LABELS] for pid in pids], dtype=bool
    )
    folds = [fold_of[pid] for pid in pids]
    per_seed_matrices = [
        np.array([[scores[pid][label] for label in LABELS] for pid in pids])
        for scores, _ in runs
    ]
    mean_matrix = np.mean(per_seed_matrices, axis=0)

    per_seed = [
        {
            "run": str(run_path),
            "scores_sha256": file_sha256(scores_path(run_path)),
            **{k: v for k, v in
               oof_macro(gold_matrix, matrix, folds, pids, gold).items()
               if k in ("macro_f1", "per_label_f1")},
        }
        for run_path, matrix in zip(args.runs, per_seed_matrices)
    ]
    ensemble = oof_macro(gold_matrix, mean_matrix, folds, pids, gold)
    predicted = ensemble.pop("predicted")

    resolved_config = {
        "script": "ensemble_encoder_seeds.py",
        "argv": sys.argv[1:],
        "milestone": "HEADROOM_AUDIT.md P5",
        "method": "mean sigmoid across seeds, cross-fitted thresholds re-fit "
                  "on the ensembled scores with trainer machinery",
        "inputs": [entry["run"] for entry in per_seed],
        "input_scores_sha256": [entry["scores_sha256"] for entry in per_seed],
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t1-encoder-ensemble-{args.name}-n{len(pids)}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)

    metrics = {
        "n_paragraphs": len(pids),
        "n_seeds": len(args.runs),
        "per_seed_oof": per_seed,
        "ensemble_oof": ensemble,
        "seed_macro_range": [
            min(e["macro_f1"] for e in per_seed),
            max(e["macro_f1"] for e in per_seed),
        ],
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    write_jsonl(
        exp_dir / "predictions" / "oof_task_1_scores.jsonl",
        [
            {
                "paragraph_id": pid,
                "scores": {
                    label: float(mean_matrix[row, col])
                    for col, label in enumerate(LABELS)
                },
                "fold": fold_of[pid],
            }
            for row, pid in enumerate(pids)
        ],
    )
    write_jsonl(
        exp_dir / "predictions" / "oof_task_1.jsonl",
        [
            {"paragraph_id": pid, "labels": sorted(predicted[pid])}
            for pid in pids
        ],
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nensemble run dir: {exp_dir}")


if __name__ == "__main__":
    main()
