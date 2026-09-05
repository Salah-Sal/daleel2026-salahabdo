"""Aggregate strict v3 outer folds and execute the fixed adoption gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import (  # noqa: E402
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    load_artifact_lineage,
    output_record,
    write_completion_marker,
)
from daleel.constants import V3_ADOPTION_DELTA  # noqa: E402
from daleel.cv import cv_contract_sha256, load_cv_runs  # noqa: E402
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records  # noqa: E402
from daleel.io import read_jsonl  # noqa: E402
from daleel.metrics import Span, span_partial_f1, task1_macro_f1  # noqa: E402
from daleel.models import SPECS  # noqa: E402
from daleel.provenance import build_provenance_manifest  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids  # noqa: E402
from daleel.submission import validate_records_against_source  # noqa: E402


AGGREGATE_ARCHITECTURE = "proposal-atom-role-v3-cv"


def _objective(task1: float, task2: float, name: str) -> float:
    if name == "task1":
        return task1
    if name == "task2":
        return task2
    return (task1 + task2) / 2


def _task2(rows: list[dict]) -> dict[int, list[Span]]:
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--baseline-task1", type=Path, required=True)
    parser.add_argument("--baseline-task2", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="optional convenience copy")
    args = parser.parse_args()

    try:
        runs, configs, contract, fold_ids = load_cv_runs(
            args.runs, allow_partial=args.allow_partial
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    setting, pool = contract["setting"], contract["pool"]
    track, model = contract["track"], contract["model"]

    t1 = load_records(TRAIN_TASK1)
    t2 = load_records(TRAIN_TASK2)
    by_id = {record["paragraph_id"]: record for record in t1}
    pool_ids = train_val_ids(t1, t2)[0] if pool == "legacy-train" else sorted(by_id)
    expected_records = [
        by_id[pid]
        for pid in pool_ids
        if setting == "both" or by_id[pid]["type"] == setting
    ]
    expected_ids = {record["paragraph_id"] for record in expected_records}

    task1_rows: list[dict] = []
    task2_rows: list[dict] = []
    upstream_runs: list[dict] = []
    for run, config in zip(runs, configs):
        fold = config["outer_fold"]
        rows1 = read_jsonl(run / "predictions" / "outer_task_1.jsonl")
        rows2 = read_jsonl(run / "predictions" / "outer_task_2.jsonl")
        ids1 = {row.get("paragraph_id") for row in rows1}
        ids2 = {row.get("paragraph_id") for row in rows2}
        if ids1 != fold_ids[fold] or ids2 != fold_ids[fold]:
            raise SystemExit(
                f"outer fold {fold} prediction IDs do not equal folds.json val_ids: "
                f"task1_delta={sorted(ids1 ^ fold_ids[fold])}, "
                f"task2_delta={sorted(ids2 ^ fold_ids[fold])}"
            )
        source = [by_id[pid] for pid in sorted(fold_ids[fold])]
        problems = (
            validate_records_against_source(rows1, source, "task_1")
            + validate_records_against_source(rows2, source, "task_2")
        )
        if problems:
            raise SystemExit(f"invalid outer fold {fold}:\n- " + "\n- ".join(problems[:30]))
        task1_rows.extend(rows1)
        task2_rows.extend(rows2)
        upstream_runs.append(
            {
                "run": str(run),
                "config_sha256": file_sha256(run / "config.json"),
                "provenance_sha256": file_sha256(run / "provenance.json"),
                "task1_sha256": file_sha256(run / "predictions" / "outer_task_1.jsonl"),
                "task2_sha256": file_sha256(run / "predictions" / "outer_task_2.jsonl"),
                "outer_fold": fold,
            }
        )

    got_ids = {row["paragraph_id"] for row in task1_rows}
    complete = got_ids == expected_ids
    if not args.allow_partial and not complete:
        raise SystemExit(
            f"OOF coverage mismatch: missing={sorted(expected_ids - got_ids)}, "
            f"extra={sorted(got_ids - expected_ids)}"
        )
    source = [by_id[pid] for pid in sorted(got_ids)]
    task1_rows.sort(key=lambda row: row["paragraph_id"])
    task2_rows.sort(key=lambda row: row["paragraph_id"])
    problems = (
        validate_records_against_source(task1_rows, source, "task_1")
        + validate_records_against_source(task2_rows, source, "task_2")
    )
    if problems:
        raise SystemExit("invalid OOF files:\n- " + "\n- ".join(problems[:30]))

    baseline1_lineage = load_artifact_lineage(
        args.baseline_task1,
        expected_task=1,
        requested_track=track,
        expected_architecture="stage0-task1-v1",
        expected_model=model,
        expected_output_kind="task-1-predictions",
        kind="task1-champion-baseline",
    )
    baseline2_lineage = load_artifact_lineage(
        args.baseline_task2,
        expected_task=2,
        requested_track=track,
        expected_architecture="stage0-quote-v1",
        expected_model=model,
        expected_output_kind="quote-proposals",
        kind="task2-champion-baseline",
    )
    baseline1_rows = [
        row for row in read_jsonl(args.baseline_task1) if row.get("paragraph_id") in got_ids
    ]
    baseline2_rows = [
        row for row in read_jsonl(args.baseline_task2) if row.get("paragraph_id") in got_ids
    ]
    problems = (
        validate_records_against_source(baseline1_rows, source, "task_1")
        + validate_records_against_source(baseline2_rows, source, "task_2")
    )
    if problems:
        raise SystemExit("invalid same-pool baselines:\n- " + "\n- ".join(problems[:30]))

    clean_t1 = clean_task1_gold(t1, t2)
    clean_t2 = clean_task2_gold(t2)
    gold_t1 = {pid: clean_t1[pid] for pid in got_ids}
    gold_t2 = {pid: clean_t2[pid] for pid in got_ids}
    candidate1 = task1_macro_f1(
        gold_t1, {row["paragraph_id"]: set(row["labels"]) for row in task1_rows}
    )
    candidate2 = span_partial_f1(gold_t2, _task2(task2_rows))
    baseline1 = task1_macro_f1(
        gold_t1, {row["paragraph_id"]: set(row["labels"]) for row in baseline1_rows}
    )
    baseline2 = span_partial_f1(gold_t2, _task2(baseline2_rows))

    task1_delta = candidate1["macro_f1"] - baseline1["macro_f1"]
    task2_delta = candidate2["f1"] - baseline2["f1"]
    candidate_primary = _objective(
        candidate1["macro_f1"], candidate2["f1"], contract["objective"]
    )
    baseline_primary = _objective(
        baseline1["macro_f1"], baseline2["f1"], contract["objective"]
    )
    primary_delta = candidate_primary - baseline_primary
    report = {
        "report_schema_version": 1,
        "architecture": AGGREGATE_ARCHITECTURE,
        "contract": contract,
        "contract_sha256": cv_contract_sha256(contract),
        "outer_folds": sorted(fold_ids),
        "n_paragraphs": len(got_ids),
        "complete": complete,
        "task1": {
            "candidate_macro_f1": candidate1["macro_f1"],
            "baseline_macro_f1": baseline1["macro_f1"],
            "delta": task1_delta,
            "required_delta": V3_ADOPTION_DELTA,
            "adopt": bool(complete and task1_delta >= V3_ADOPTION_DELTA),
            "candidate_micro_f1": candidate1["micro_f1"],
            "candidate_per_label_f1": {
                label: values["f1"] for label, values in candidate1["per_label"].items()
            },
        },
        "task2": {
            "candidate_f1": candidate2["f1"],
            "baseline_f1": baseline2["f1"],
            "delta": task2_delta,
            "required_delta": V3_ADOPTION_DELTA,
            "adopt": bool(complete and task2_delta >= V3_ADOPTION_DELTA),
            "candidate_precision": candidate2["precision"],
            "candidate_recall": candidate2["recall"],
            "candidate_per_label_f1": {
                label: values["f1"] for label, values in candidate2["per_label"].items()
            },
        },
        "primary": {
            "objective": contract["objective"],
            "candidate_score": candidate_primary,
            "baseline_score": baseline_primary,
            "delta": primary_delta,
            "required_delta": V3_ADOPTION_DELTA,
            "adopt": bool(complete and primary_delta >= V3_ADOPTION_DELTA),
        },
        "baselines": {
            "task1": {
                key: value
                for key, value in baseline1_lineage.items()
                if key != "manifest_data"
            },
            "task2": {
                key: value
                for key, value in baseline2_lineage.items()
                if key != "manifest_data"
            },
        },
        "upstream_runs": upstream_runs,
    }

    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"v3-cv-adoption-{model}-{setting}-{contract['objective']}",
        {"contract": contract, "runs": upstream_runs},
    )
    config_path = atomic_write_json(
        exp_dir / "config.json",
        {
            "script": "aggregate_role_cv.py",
            "argv": sys.argv[1:],
            "contract": contract,
            "contract_sha256": report["contract_sha256"],
        },
    )
    report_path = atomic_write_json(exp_dir / "adoption_report.json", report)
    provenance = build_provenance_manifest(
        tasks=(1, 2),
        track=track,
        setting=setting,
        task_model=SPECS[model],
        optimizer=contract["optimizer"],
        training_ids=[],
        data_sources=("daleel2026:train-task-1", "daleel2026:train-task-2"),
        architecture_version=AGGREGATE_ARCHITECTURE,
    )
    provenance["evaluation_ids"] = sorted(got_ids)
    provenance["upstream_artifacts"] = list(report["baselines"].values())
    provenance["upstream_runs"] = upstream_runs
    provenance["adoption"] = report["primary"]
    provenance["outputs"] = [
        output_record(exp_dir, report_path, kind="v3-adoption-report")
    ]
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    write_completion_marker(
        exp_dir,
        [config_path, report_path, provenance_path],
        metadata={"adopt": report["primary"]["adopt"]},
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nadoption report: {report_path}")


if __name__ == "__main__":
    main()
