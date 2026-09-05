"""Cross-fit and adoption-gate Task 1 direct/span reconciliation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import (  # noqa: E402
    atomic_write_json,
    create_experiment_dir,
    load_artifact_lineage,
    output_record,
    write_completion_marker,
)
from daleel.constants import LABELS, V3_ADOPTION_DELTA  # noqa: E402
from daleel.cv import cv_contract_sha256, load_cv_runs  # noqa: E402
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records  # noqa: E402
from daleel.io import read_jsonl  # noqa: E402
from daleel.metrics import task1_macro_f1  # noqa: E402
from daleel.models import SPECS  # noqa: E402
from daleel.provenance import build_provenance_manifest  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import clean_task1_gold, train_val_ids  # noqa: E402
from daleel.submission import validate_records_against_source  # noqa: E402


ARCHITECTURE_VERSION = "task1-fusion-v1"
# Ties preserve the champion first, then the span-only view; union is last
# because it is the most aggressive precision-risk operation.
OPERATIONS = ("direct", "spans", "intersection", "union")


def apply_operation(label: str, spans: set[str], direct: set[str], operation: str) -> bool:
    if operation == "spans":
        return label in spans
    if operation == "direct":
        return label in direct
    if operation == "union":
        return label in spans or label in direct
    if operation == "intersection":
        return label in spans and label in direct
    raise ValueError(f"unknown operation {operation!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    args = parser.parse_args()

    try:
        runs, configs, contract, expected_fold_ids = load_cv_runs(args.runs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(runs) < 2:
        raise SystemExit("Task 1 fusion requires at least two outer folds")
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

    direct_lineage = load_artifact_lineage(
        args.direct,
        expected_task=1,
        requested_track=track,
        expected_architecture="stage0-task1-v1",
        expected_model=model,
        expected_output_kind="task-1-predictions",
        kind="direct-task1-champion",
    )
    direct_rows = [
        row for row in read_jsonl(args.direct) if row.get("paragraph_id") in expected_ids
    ]
    problems = validate_records_against_source(direct_rows, expected_records, "task_1")
    if problems:
        raise SystemExit("invalid direct predictions:\n- " + "\n- ".join(problems[:30]))
    direct = {row["paragraph_id"]: set(row["labels"]) for row in direct_rows}

    span: dict[int, set[str]] = {}
    fold_ids: dict[int, set[int]] = {}
    upstream = [
        {key: value for key, value in direct_lineage.items() if key != "manifest_data"}
    ]
    for run, config in zip(runs, configs):
        fold = config["outer_fold"]
        prediction_path = run / "predictions" / "outer_task_1.jsonl"
        lineage = load_artifact_lineage(
            prediction_path,
            expected_task=1,
            requested_track=track,
            expected_architecture="proposal-atom-role-v3",
            expected_setting=setting,
            expected_model=model,
            expected_output_kind="v3-outer-task1",
            kind=f"v3-outer-task1-fold-{fold}",
        )
        upstream.append({key: value for key, value in lineage.items() if key != "manifest_data"})
        rows = read_jsonl(prediction_path)
        ids = {row["paragraph_id"] for row in rows}
        if ids != expected_fold_ids[fold]:
            raise SystemExit(f"fold {fold} prediction IDs do not match folds.json")
        source = [by_id[pid] for pid in sorted(ids)]
        problems = validate_records_against_source(rows, source, "task_1")
        if problems:
            raise SystemExit(f"invalid fold {fold}:\n- " + "\n- ".join(problems[:30]))
        fold_ids[fold] = ids
        for row in rows:
            pid = row["paragraph_id"]
            if pid in span:
                raise SystemExit(f"paragraph {pid} occurs in multiple outer folds")
            span[pid] = set(row["labels"])
    if set(span) != expected_ids:
        raise SystemExit(
            f"span OOF coverage mismatch: missing={sorted(expected_ids - set(span))}, "
            f"extra={sorted(set(span) - expected_ids)}"
        )

    gold = clean_task1_gold(t1, t2)
    crossfit: dict[int, set[str]] = {pid: set() for pid in expected_ids}
    fold_policies: dict[int, dict[str, str]] = {}
    fold_selection_scores: dict[int, dict[str, dict[str, float]]] = {}
    for fold, held_ids in sorted(fold_ids.items()):
        selection_ids = expected_ids - held_ids
        if not selection_ids:
            raise SystemExit("fusion fold has no disjoint selection examples")
        policy: dict[str, str] = {}
        selection_scores: dict[str, dict[str, float]] = {}
        for label in LABELS:
            scores: dict[str, float] = {}
            for operation in OPERATIONS:
                candidate = {
                    pid: (
                        {label}
                        if apply_operation(label, span[pid], direct[pid], operation)
                        else set()
                    )
                    for pid in selection_ids
                }
                scores[operation] = task1_macro_f1(
                    {pid: ({label} if label in gold[pid] else set()) for pid in selection_ids},
                    candidate,
                    labels=(label,),
                )["macro_f1"]
            best = max(OPERATIONS, key=lambda operation: scores[operation])
            policy[label] = best
            selection_scores[label] = scores
            for pid in held_ids:
                if apply_operation(label, span[pid], direct[pid], best):
                    crossfit[pid].add(label)
        fold_policies[fold] = policy
        fold_selection_scores[fold] = selection_scores

    gold_expected = {pid: gold[pid] for pid in expected_ids}
    crossfit_score = task1_macro_f1(gold_expected, crossfit)
    span_score = task1_macro_f1(gold_expected, span)
    direct_score = task1_macro_f1(gold_expected, direct)
    delta = crossfit_score["macro_f1"] - direct_score["macro_f1"]

    final_policy: dict[str, str] = {}
    stability: dict[str, dict[str, int]] = {}
    for label in LABELS:
        counts = Counter(policy[label] for policy in fold_policies.values())
        stability[label] = dict(counts)
        final_policy[label] = max(
            OPERATIONS,
            key=lambda operation: (counts[operation], -OPERATIONS.index(operation)),
        )
    adoption = {
        "candidate_score": crossfit_score["macro_f1"],
        "baseline_score": direct_score["macro_f1"],
        "delta": delta,
        "required_delta": V3_ADOPTION_DELTA,
        "adopt": delta >= V3_ADOPTION_DELTA,
    }
    report = {
        "report_schema_version": 1,
        "architecture": ARCHITECTURE_VERSION,
        "contract": contract,
        "contract_sha256": cv_contract_sha256(contract),
        "setting": setting,
        "pool": pool,
        "n_paragraphs": len(expected_ids),
        "folds": sorted(fold_ids),
        "span_macro_f1": span_score["macro_f1"],
        "direct_macro_f1": direct_score["macro_f1"],
        "crossfit_macro_f1": crossfit_score["macro_f1"],
        "crossfit_micro_f1": crossfit_score["micro_f1"],
        "crossfit_per_label_f1": {
            label: values["f1"] for label, values in crossfit_score["per_label"].items()
        },
        "adoption": adoption,
        "fold_policies": fold_policies,
        "fold_selection_scores": fold_selection_scores,
        "policy_stability": stability,
        "final_majority_policy": final_policy,
    }

    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"task1-fusion-{model}-{setting}",
        {"contract": contract, "direct_sha256": direct_lineage["sha256"]},
    )
    config_path = atomic_write_json(
        exp_dir / "config.json",
        {"script": "select_task1_fusion.py", "argv": sys.argv[1:], "contract": contract},
    )
    policy_path = atomic_write_json(exp_dir / "policy.json", final_policy)
    report_path = atomic_write_json(exp_dir / "report.json", report)
    provenance = build_provenance_manifest(
        tasks=1,
        track=track,
        setting=setting,
        task_model=SPECS[model],
        optimizer="crossfit-fusion",
        training_ids=sorted(expected_ids),
        data_sources=("daleel2026:train-task-1", "daleel2026:train-task-2"),
        architecture_version=ARCHITECTURE_VERSION,
    )
    provenance["upstream_artifacts"] = upstream
    provenance["adoption"] = adoption
    provenance["outputs"] = [
        output_record(exp_dir, policy_path, kind="task1-fusion-policy"),
        output_record(exp_dir, report_path, kind="task1-fusion-report"),
    ]
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    write_completion_marker(
        exp_dir,
        [config_path, policy_path, report_path, provenance_path],
        metadata={"adopt": adoption["adopt"]},
    )
    if args.policy_output:
        atomic_write_json(args.policy_output, final_policy)
    if args.report_output:
        atomic_write_json(args.report_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\npolicy: {policy_path}")


if __name__ == "__main__":
    main()
