"""No-network end-to-end test of the four-way OOF comparison gate."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.compare_structural_cv as comparison
from daleel.artifacts import atomic_write_json, write_completion_marker
from daleel.constants import V3_ADOPTION_DELTA
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.folds import fold_manifest, fold_manifest_hash, make_stratified_folds
from daleel.io import write_jsonl
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids


def _base_contract(fold_hash):
    return {
        "model": "gemma-4-31b-paid",
        "track": "closed",
        "setting": "both",
        "optimizer": "gepa",
        "objective": "task2",
        "pool": "legacy-train",
        "n_folds": 2,
        "n_inner_folds": 2,
        "inner_fold": 0,
        "fold_seed": 20260710,
        "fold_manifest_sha256": fold_hash,
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
        "accept_delta": V3_ADOPTION_DELTA,
    }


def _make_runs(tmp_path):
    t1 = load_records(TRAIN_TASK1)
    t2 = load_records(TRAIN_TASK2)
    by_id = {record["paragraph_id"]: record for record in t1}
    ids = train_val_ids(t1, t2)[0]
    labels_all = clean_task1_gold(t1, t2)
    spans = clean_task2_gold(t2)
    labels = {pid: labels_all[pid] for pid in ids}
    genres = {pid: by_id[pid]["type"] for pid in ids}
    folds = make_stratified_folds(ids, labels, genres, n_splits=2, seed=20260710)
    manifest = fold_manifest(folds, seed=20260710)
    contract = _base_contract(fold_manifest_hash(manifest))
    groups = {mode: [] for mode in comparison.MODES}
    for mode in comparison.MODES:
        for fold in folds:
            run = tmp_path / f"{mode}-{fold.index}"
            prediction_dir = run / "predictions"
            prediction_dir.mkdir(parents=True)
            config = {
                **contract,
                "architecture": (
                    "proposal-atom-role-v3"
                    if mode == "baseline"
                    else "proposal-atom-role-v4-structure"
                ),
                "structure_mode": mode,
                "forest_sha256": None if mode == "baseline" else "b" * 64,
                "forest_contract_sha256": None if mode == "baseline" else "c" * 64,
                "shuffle_seed": 20260711,
                "final": False,
                "outer_fold": fold.index,
            }
            records = [by_id[pid] for pid in fold.val_ids]
            rows1 = [
                {**record, "labels": sorted(labels[record["paragraph_id"]])}
                for record in records
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
                        for span in spans[record["paragraph_id"]]
                    ],
                }
                for record in records
            ]
            config_path = atomic_write_json(run / "config.json", config)
            folds_path = atomic_write_json(run / "folds.json", {"outer": manifest})
            metrics_path = atomic_write_json(
                run / "metrics.json",
                {
                    "outer_holdout": {
                        "decision": {
                            "n": len(rows2),
                            "exact_accuracy": 1.0,
                            "mean_set_f1": 1.0,
                            "parse_failures": 0,
                            "confusions": {"AS->AS": len(rows2)},
                        }
                    }
                },
            )
            provenance_path = atomic_write_json(run / "provenance.json", {"test": True})
            task1_path = prediction_dir / "outer_task_1.jsonl"
            task2_path = prediction_dir / "outer_task_2.jsonl"
            write_jsonl(task1_path, rows1)
            write_jsonl(task2_path, rows2)
            write_completion_marker(
                run,
                [config_path, folds_path, metrics_path, provenance_path,
                 task1_path, task2_path],
            )
            groups[mode].append(run)
    return groups


def test_comparison_requires_real_relations_to_beat_both_controls(
    monkeypatch, tmp_path, capsys
):
    groups = _make_runs(tmp_path / "runs")
    output_root = tmp_path / "reports"
    monkeypatch.setattr(comparison, "EXPERIMENTS_DIR", output_root)
    argv = ["compare_structural_cv.py"]
    for mode in comparison.MODES:
        argv.append(f"--{mode}-runs")
        argv.extend(str(path) for path in groups[mode])
    monkeypatch.setattr(sys, "argv", argv)
    comparison.main()
    capsys.readouterr()
    report_path = next(output_root.iterdir()) / "comparison_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"]["forest"]["task2_f1"] >= 1.0
    assert report["task2_deltas"]["forest_minus_flat"] == 0.0
    assert report["adoption"]["forest_beats_flat"] is False
    assert report["adoption"]["adopt"] is False
