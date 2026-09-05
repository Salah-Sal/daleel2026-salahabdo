"""No-network OOF aggregation and Task 1 fusion adoption integration."""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.aggregate_role_cv as aggregate
import scripts.compile_span_roles as compiler
import scripts.select_task1_fusion as fusion

from daleel.artifacts import file_sha256, write_completion_marker
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.folds import fold_manifest, fold_manifest_hash, make_stratified_folds
from daleel.io import write_jsonl
from daleel.models import SPECS
from daleel.provenance import build_provenance_manifest
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids


def _contract_config(manifest, fold, proposal_sha):
    return {
        "script": "compile_span_roles.py",
        "architecture": "proposal-atom-role-v3",
        "model": "gemma-4-31b-paid",
        "track": "closed",
        "setting": "editorial",
        "optimizer": "none",
        "objective": "task2",
        "pool": "legacy-train",
        "n_folds": 2,
        "n_inner_folds": 2,
        "inner_fold": 0,
        "fold_seed": 17,
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
        "proposal_sha256": proposal_sha,
        "accept_delta": 0.02,
        "outer_fold": fold,
        "final": False,
    }


def _stage0_artifact(root, filename, rows, *, task, architecture, kind):
    run = root / filename
    prediction_dir = run / "predictions"
    prediction_dir.mkdir(parents=True)
    path = prediction_dir / "preds.jsonl"
    write_jsonl(path, rows)
    manifest = build_provenance_manifest(
        tasks=task,
        track="closed",
        setting="both",
        task_model=SPECS["gemma-4-31b-paid"],
        optimizer=None,
        training_ids=[],
        data_sources=[f"daleel2026:train-task-{task}"],
        architecture_version=architecture,
    )
    manifest["outputs"] = [
        {
            "kind": kind,
            "relative_path": "predictions/preds.jsonl",
            "sha256": file_sha256(path),
        }
    ]
    manifest_path = run / "provenance.json"
    manifest_path.write_text(json.dumps(manifest))
    write_completion_marker(run, [path, manifest_path])
    return path


def _cv_fixture(tmp_path):
    if not TRAIN_TASK1.exists() or not TRAIN_TASK2.exists():
        pytest.skip("official Daleel training clone is not available")
    t1 = load_records(TRAIN_TASK1)
    t2 = load_records(TRAIN_TASK2)
    by_id = {record["paragraph_id"]: record for record in t1}
    train_ids = train_val_ids(t1, t2)[0]
    ids = [pid for pid in train_ids if by_id[pid]["type"] == "editorial"]
    gold1 = clean_task1_gold(t1, t2)
    gold2 = clean_task2_gold(t2)
    folds = make_stratified_folds(
        ids,
        {pid: gold1[pid] for pid in ids},
        {pid: by_id[pid]["type"] for pid in ids},
        n_splits=2,
        seed=17,
    )
    manifest = fold_manifest(folds, seed=17)
    proposal_sha = "a" * 64
    runs = []
    for fold in folds:
        run = tmp_path / f"outer-{fold.index}"
        prediction_dir = run / "predictions"
        prediction_dir.mkdir(parents=True)
        source = [by_id[pid] for pid in fold.val_ids]
        rows1 = [
            {**record, "labels": sorted(gold1[record["paragraph_id"]])}
            for record in source
        ]
        rows2 = [
            {
                **record,
                "labels": [
                    {
                        "label": span.label,
                        "start_offset": span.start,
                        "end_offset": span.end,
                    }
                    for span in gold2[record["paragraph_id"]]
                ],
            }
            for record in source
        ]
        path1 = prediction_dir / "outer_task_1.jsonl"
        path2 = prediction_dir / "outer_task_2.jsonl"
        write_jsonl(path1, rows1)
        write_jsonl(path2, rows2)
        config = _contract_config(manifest, fold.index, proposal_sha)
        (run / "config.json").write_text(json.dumps(config))
        (run / "folds.json").write_text(json.dumps({"outer": manifest, "inner": manifest}))
        provenance = build_provenance_manifest(
            tasks=(1, 2),
            track="closed",
            setting="editorial",
            task_model=SPECS["gemma-4-31b-paid"],
            optimizer=None,
            training_ids=fold.train_ids,
            data_sources=("daleel2026:train-task-1", "daleel2026:train-task-2"),
            architecture_version="proposal-atom-role-v3",
        )
        provenance["outputs"] = [
            {
                "kind": "v3-outer-task1",
                "relative_path": "predictions/outer_task_1.jsonl",
                "sha256": file_sha256(path1),
            },
            {
                "kind": "v3-outer-task2",
                "relative_path": "predictions/outer_task_2.jsonl",
                "sha256": file_sha256(path2),
            },
        ]
        provenance_path = run / "provenance.json"
        provenance_path.write_text(json.dumps(provenance))
        write_completion_marker(
            run,
            [run / "config.json", run / "folds.json", provenance_path, path1, path2],
        )
        runs.append(run)

    source = [by_id[pid] for pid in ids]
    empty1 = [{**record, "labels": []} for record in source]
    empty2 = [{**record, "labels": []} for record in source]
    baseline1 = _stage0_artifact(
        tmp_path,
        "baseline1",
        empty1,
        task=1,
        architecture="stage0-task1-v1",
        kind="task-1-predictions",
    )
    baseline2 = _stage0_artifact(
        tmp_path,
        "baseline2",
        empty2,
        task=2,
        architecture="stage0-quote-v1",
        kind="quote-proposals",
    )
    return runs, baseline1, baseline2


def test_oof_and_fusion_commands_emit_adopted_hash_bound_artifacts(
    monkeypatch, tmp_path, capsys
):
    runs, baseline1, baseline2 = _cv_fixture(tmp_path)
    aggregate_root = tmp_path / "aggregate"
    monkeypatch.setattr(aggregate, "EXPERIMENTS_DIR", aggregate_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_role_cv.py",
            *(str(run) for run in runs),
            "--baseline-task1",
            str(baseline1),
            "--baseline-task2",
            str(baseline2),
        ],
    )
    aggregate.main()
    capsys.readouterr()
    aggregate_run = next(aggregate_root.iterdir())
    report = json.loads((aggregate_run / "adoption_report.json").read_text())
    assert report["complete"] is True
    assert report["primary"]["adopt"] is True
    assert report["task2"]["delta"] >= 0.02
    assert (aggregate_run / "COMPLETED.json").is_file()

    contract = report["contract"]
    final_args = argparse.Namespace(
        track=contract["track"],
        setting=contract["setting"],
        model=contract["model"],
        optimizer=contract["optimizer"],
        objective=contract["objective"],
        pool=contract["pool"],
        fold_seed=contract["fold_seed"],
        n_folds=contract["n_folds"],
        n_inner_folds=contract["n_inner_folds"],
        inner_fold=contract["inner_fold"],
        granularity=contract["granularity"],
        containment_threshold=contract["containment_threshold"],
        context_chars=contract["context_chars"],
        demo_context_chars=contract["demo_context_chars"],
        role_cot=contract["role_cot"],
        no_demo_selector=not contract["demo_selector"],
        max_demos=contract["max_demos"],
        balance_per_class=contract["balance_per_class"],
        optimizer_val_max=contract["optimizer_val_max"],
        auto=contract["auto"],
        max_metric_calls=contract["max_metric_calls"],
        max_tokens=contract["max_tokens"],
        max_rerank_candidates=contract["max_rerank_candidates"],
        proposals=tmp_path / "synthetic-proposals.jsonl",
    )
    monkeypatch.setattr(compiler, "file_sha256", lambda path: contract["proposal_sha256"])
    loaded_report, _ = compiler._load_adoption_report(
        aggregate_run / "adoption_report.json", final_args
    )
    assert loaded_report["primary"]["adopt"] is True

    fusion_root = tmp_path / "fusion"
    monkeypatch.setattr(fusion, "EXPERIMENTS_DIR", fusion_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_task1_fusion.py",
            *(str(run) for run in runs),
            "--direct",
            str(baseline1),
        ],
    )
    fusion.main()
    capsys.readouterr()
    fusion_run = next(fusion_root.iterdir())
    fusion_report = json.loads((fusion_run / "report.json").read_text())
    assert fusion_report["adoption"]["adopt"] is True
    provenance = json.loads((fusion_run / "provenance.json").read_text())
    assert provenance["adoption"]["adopt"] is True
    assert any(output["kind"] == "task1-fusion-policy" for output in provenance["outputs"])
