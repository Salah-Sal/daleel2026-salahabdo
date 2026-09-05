"""No-network tests for the P5 mean-sigmoid seed-ensemble builder."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.ensemble_encoder_seeds as ensembler
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.splits import clean_task1_gold, train_val_ids


def _train_pool():
    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = clean_task1_gold(t1, t2)
    return {pid: gold[pid] for pid in train_ids}


def _write_run(directory, gold, fold_of, high, low, nested):
    rows = [
        {
            "paragraph_id": pid,
            "scores": {
                label: (high if label in gold[pid] else low) for label in LABELS
            },
            "fold": fold_of[pid],
        }
        for pid in sorted(gold)
    ]
    target = directory / "predictions" if nested else directory
    target.mkdir(parents=True)
    write_jsonl(target / "oof_task_1_scores.jsonl", rows)
    return directory


def _folds(gold):
    return {pid: index % 5 for index, pid in enumerate(sorted(gold))}


def test_ensemble_means_scores_and_refits_thresholds(tmp_path, monkeypatch):
    gold = _train_pool()
    fold_of = _folds(gold)
    run_a = _write_run(tmp_path / "seed-a", gold, fold_of, 0.9, 0.1, nested=True)
    run_b = _write_run(tmp_path / "seed-b", gold, fold_of, 0.7, 0.3, nested=False)

    monkeypatch.setattr(ensembler, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(
        sys, "argv",
        ["ensemble_encoder_seeds.py", "--runs", str(run_a), str(run_b),
         "--name", "unit"],
    )
    ensembler.main()

    (exp_dir,) = (tmp_path / "experiments").iterdir()
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_seeds"] == 2
    assert metrics["n_paragraphs"] == len(gold)
    # Perfectly separated scores must yield a perfect cross-fitted macro,
    # for each seed alone and for the mean.
    assert [entry["macro_f1"] for entry in metrics["per_seed_oof"]] == [1.0, 1.0]
    assert metrics["ensemble_oof"]["macro_f1"] == 1.0

    scores = {
        row["paragraph_id"]: row
        for row in read_jsonl(exp_dir / "predictions" / "oof_task_1_scores.jsonl")
    }
    assert set(scores) == set(gold)
    probe = min(gold)
    assert scores[probe]["fold"] == fold_of[probe]
    for label in LABELS:
        expected = 0.8 if label in gold[probe] else 0.2
        assert scores[probe]["scores"][label] == pytest.approx(expected)

    decisions = {
        row["paragraph_id"]: set(row["labels"])
        for row in read_jsonl(exp_dir / "predictions" / "oof_task_1.jsonl")
    }
    assert decisions[probe] == gold[probe]
    assert all("text" not in row for row in read_jsonl(
        exp_dir / "predictions" / "oof_task_1.jsonl"))


def test_fold_contract_mismatch_is_fatal(tmp_path, monkeypatch):
    gold = _train_pool()
    fold_of = _folds(gold)
    shifted = {pid: (fold + 1) % 5 for pid, fold in fold_of.items()}
    run_a = _write_run(tmp_path / "seed-a", gold, fold_of, 0.9, 0.1, nested=True)
    run_b = _write_run(tmp_path / "seed-b", gold, shifted, 0.9, 0.1, nested=True)

    monkeypatch.setattr(ensembler, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(
        sys, "argv",
        ["ensemble_encoder_seeds.py", "--runs", str(run_a), str(run_b),
         "--name", "unit"],
    )
    with pytest.raises(SystemExit, match="fold contract"):
        ensembler.main()


def test_single_run_is_rejected(tmp_path, monkeypatch):
    gold = _train_pool()
    run_a = _write_run(
        tmp_path / "seed-a", gold, _folds(gold), 0.9, 0.1, nested=True
    )
    monkeypatch.setattr(
        sys, "argv",
        ["ensemble_encoder_seeds.py", "--runs", str(run_a), "--name", "unit"],
    )
    with pytest.raises(SystemExit, match="at least two"):
        ensembler.main()
