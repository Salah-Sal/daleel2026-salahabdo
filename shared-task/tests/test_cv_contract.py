import json

import pytest

from daleel.artifacts import write_completion_marker
from daleel.cv import cv_contract_sha256, load_cv_runs
from daleel.folds import Fold, fold_manifest, fold_manifest_hash


def _config(manifest, fold, **overrides):
    config = {
        "architecture": "proposal-atom-role-v3",
        "model": "gemma-4-31b-paid",
        "track": "closed",
        "setting": "both",
        "optimizer": "gepa",
        "objective": "task2",
        "pool": "legacy-train",
        "n_folds": 2,
        "n_inner_folds": 2,
        "inner_fold": 0,
        "fold_seed": 7,
        "fold_manifest_sha256": fold_manifest_hash(manifest),
        "granularity": "connective",
        "containment_threshold": 0.8,
        "context_chars": 500,
        "demo_context_chars": 350,
        "role_cot": False,
        "demo_selector": True,
        "max_demos": 7,
        "same_genre_bonus": 0.05,
        "balance_per_class": 80,
        "optimizer_val_max": 300,
        "optimizer_val_distribution": "natural-without-replacement",
        "optimizer_train_distribution": "class-balanced-with-cycling",
        "auto": "light",
        "max_metric_calls": None,
        "max_tokens": 2500,
        "max_rerank_candidates": 0,
        "proposal_sha256": "a" * 64,
        "accept_delta": 0.02,
        "outer_fold": fold,
        "final": False,
    }
    config.update(overrides)
    return config


def _runs(tmp_path):
    folds = (
        Fold(0, (3, 4), (1, 2)),
        Fold(1, (1, 2), (3, 4)),
    )
    manifest = fold_manifest(folds, seed=7)
    runs = []
    for fold in range(2):
        run = tmp_path / f"run-{fold}"
        run.mkdir()
        (run / "predictions").mkdir()
        (run / "config.json").write_text(json.dumps(_config(manifest, fold)))
        (run / "folds.json").write_text(json.dumps({"outer": manifest, "inner": manifest}))
        (run / "provenance.json").write_text("{}")
        (run / "predictions" / "outer_task_1.jsonl").write_text("")
        (run / "predictions" / "outer_task_2.jsonl").write_text("")
        write_completion_marker(
            run,
            [
                run / "config.json",
                run / "folds.json",
                run / "provenance.json",
                run / "predictions" / "outer_task_1.jsonl",
                run / "predictions" / "outer_task_2.jsonl",
            ],
        )
        runs.append(run)
    return runs


def test_load_cv_runs_requires_one_exact_complete_contract(tmp_path):
    runs = _runs(tmp_path)
    paths, configs, contract, fold_ids = load_cv_runs(runs)
    assert paths == runs
    assert len(configs) == 2
    assert fold_ids == {0: {1, 2}, 1: {3, 4}}
    assert len(cv_contract_sha256(contract)) == 64


def test_load_cv_runs_rejects_incompatibility_and_missing_folds(tmp_path):
    runs = _runs(tmp_path)
    config = json.loads((runs[1] / "config.json").read_text())
    config["proposal_sha256"] = "b" * 64
    (runs[1] / "config.json").write_text(json.dumps(config))
    write_completion_marker(
        runs[1],
        [
            runs[1] / "config.json",
            runs[1] / "folds.json",
            runs[1] / "provenance.json",
            runs[1] / "predictions" / "outer_task_1.jsonl",
            runs[1] / "predictions" / "outer_task_2.jsonl",
        ],
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_cv_runs(runs)

    with pytest.raises(ValueError, match="outer fold set mismatch"):
        load_cv_runs(runs[:1])
    _, _, _, partial = load_cv_runs(runs[:1], allow_partial=True)
    assert partial == {0: {1, 2}}


def test_load_cv_runs_rejects_incomplete_directory(tmp_path):
    runs = _runs(tmp_path)
    (runs[0] / "COMPLETED.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        load_cv_runs(runs)
