"""Compare complete baseline/flat/forest/shuffled OOF campaigns.

The primary contrast is forest minus the exact v3 baseline on official Task 2
F1.  Flat isolates explicit local ordering; shuffled preserves graph shape and
relation vocabulary while breaking attachment to atom text.  Adoption also
requires the real forest to beat both controls, not merely the baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import (  # noqa: E402
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    output_record,
    validate_completion_marker,
    write_completion_marker,
)
from daleel.constants import V3_ADOPTION_DELTA  # noqa: E402
from daleel.cv import CV_CONTRACT_FIELDS  # noqa: E402
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records  # noqa: E402
from daleel.discourse_forest import STRUCTURAL_ROLE_ARCHITECTURE  # noqa: E402
from daleel.folds import fold_manifest_hash  # noqa: E402
from daleel.io import read_jsonl  # noqa: E402
from daleel.metrics import Span, span_partial_f1, task1_macro_f1  # noqa: E402
from daleel.models import SPECS  # noqa: E402
from daleel.provenance import build_provenance_manifest  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids  # noqa: E402
from daleel.submission import validate_records_against_source  # noqa: E402


COMPARISON_ARCHITECTURE = "discourse-forest-structural-ablation-cv-v1"
MODES = ("baseline", "flat", "forest", "shuffled")
BASE_ARCHITECTURE = "proposal-atom-role-v3"
COMMON_FIELDS = tuple(field for field in CV_CONTRACT_FIELDS if field != "architecture")


def _task2(rows: list[dict]) -> dict[int, list[Span]]:
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in rows
    }


def _load_group(
    paths: list[Path],
    mode: str,
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    configs = []
    fold_ids: dict[int, set[int]] = {}
    task1_rows: list[dict] = []
    task2_rows: list[dict] = []
    run_records = []
    manifest_reference = None
    common_reference = None
    for run in paths:
        validate_completion_marker(
            run,
            required_paths=(
                run / "config.json",
                run / "folds.json",
                run / "provenance.json",
                run / "predictions" / "outer_task_1.jsonl",
                run / "predictions" / "outer_task_2.jsonl",
            ),
        )
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        folds = json.loads((run / "folds.json").read_text(encoding="utf-8"))
        got_mode = config.get("structure_mode", "baseline")
        if got_mode != mode:
            raise ValueError(f"{run} is mode {got_mode!r}, expected {mode!r}")
        expected_architecture = (
            BASE_ARCHITECTURE if mode == "baseline" else STRUCTURAL_ROLE_ARCHITECTURE
        )
        if config.get("architecture") != expected_architecture:
            raise ValueError(
                f"{run} architecture {config.get('architecture')!r}, "
                f"expected {expected_architecture!r}"
            )
        if config.get("final"):
            raise ValueError(f"final compile has no OOF holdout: {run}")
        common = {field: config.get(field) for field in COMMON_FIELDS}
        missing = [field for field, value in common.items() if value is None and field not in {
            "max_metric_calls"
        }]
        if missing:
            raise ValueError(f"{run} is missing controlled fields {missing}")
        if common_reference is None:
            common_reference = common
        elif common != common_reference:
            differing = {
                key: {"first": common_reference.get(key), "run": common.get(key)}
                for key in COMMON_FIELDS
                if common_reference.get(key) != common.get(key)
            }
            raise ValueError(f"incompatible {mode} run {run}: {differing}")
        manifest = folds.get("outer")
        if not isinstance(manifest, dict):
            raise ValueError(f"run has no outer fold manifest: {run}")
        if fold_manifest_hash(manifest) != config.get("fold_manifest_sha256"):
            raise ValueError(f"fold manifest hash mismatch: {run}")
        if manifest_reference is None:
            manifest_reference = manifest
        elif manifest != manifest_reference:
            raise ValueError(f"{mode} runs use different fold manifests: {run}")
        fold = config.get("outer_fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold in fold_ids:
            raise ValueError(f"invalid/duplicate fold {fold!r} in {run}")
        entries = [entry for entry in manifest["folds"] if entry.get("index") == fold]
        if len(entries) != 1:
            raise ValueError(f"fold {fold} missing/duplicated in {run}")
        fold_ids[fold] = set(entries[0]["val_ids"])
        rows1 = read_jsonl(run / "predictions" / "outer_task_1.jsonl")
        rows2 = read_jsonl(run / "predictions" / "outer_task_2.jsonl")
        if {row["paragraph_id"] for row in rows1} != fold_ids[fold] or {
            row["paragraph_id"] for row in rows2
        } != fold_ids[fold]:
            raise ValueError(f"{run} prediction IDs do not match fold {fold}")
        task1_rows.extend(rows1)
        task2_rows.extend(rows2)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        run_records.append(
            {
                "run": str(run),
                "fold": fold,
                "config_sha256": file_sha256(run / "config.json"),
                "provenance_sha256": file_sha256(run / "provenance.json"),
                "task1_sha256": file_sha256(run / "predictions" / "outer_task_1.jsonl"),
                "task2_sha256": file_sha256(run / "predictions" / "outer_task_2.jsonl"),
                "outer_decision": (metrics.get("outer_holdout") or {}).get("decision", {}),
            }
        )
        configs.append(config)

    expected_folds = set(range(common_reference["n_folds"]))
    complete = set(fold_ids) == expected_folds
    if not allow_partial and not complete:
        raise ValueError(
            f"{mode} fold set mismatch: expected={sorted(expected_folds)}, "
            f"actual={sorted(fold_ids)}"
        )
    return {
        "mode": mode,
        "configs": configs,
        "common": common_reference,
        "manifest": manifest_reference,
        "fold_ids": fold_ids,
        "task1_rows": task1_rows,
        "task2_rows": task2_rows,
        "runs": run_records,
        "complete": complete,
        "forest_sha256": configs[0].get("forest_sha256"),
        "forest_contract_sha256": configs[0].get("forest_contract_sha256"),
        "shuffle_seed": configs[0].get("shuffle_seed"),
    }


def _decision_totals(group: dict[str, Any]) -> dict[str, Any]:
    totals = Counter()
    confusions = Counter()
    for run in group["runs"]:
        decision = run["outer_decision"]
        n = int(decision.get("n", 0))
        totals["n"] += n
        totals["exact"] += float(decision.get("exact_accuracy", 0.0)) * n
        totals["set_f1"] += float(decision.get("mean_set_f1", 0.0)) * n
        totals["parse_failures"] += int(decision.get("parse_failures", 0))
        for key, count in (decision.get("confusions") or {}).items():
            confusions[key] += count
    n = totals["n"]
    return {
        "n": n,
        "exact_accuracy": totals["exact"] / n if n else 0.0,
        "mean_set_f1": totals["set_f1"] / n if n else 0.0,
        "parse_failures": totals["parse_failures"],
        "confusions": dict(sorted(confusions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for mode in MODES:
        parser.add_argument(f"--{mode}-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    groups = {}
    try:
        for mode in MODES:
            groups[mode] = _load_group(
                getattr(args, f"{mode}_runs"),
                mode,
                allow_partial=args.allow_partial,
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    baseline = groups["baseline"]
    for mode, group in groups.items():
        if group["common"] != baseline["common"]:
            differing = {
                key: {"baseline": baseline["common"].get(key), "mode": group["common"].get(key)}
                for key in COMMON_FIELDS
                if baseline["common"].get(key) != group["common"].get(key)
            }
            raise SystemExit(f"{mode} controlled contract differs from baseline: {differing}")
        if group["manifest"] != baseline["manifest"] or group["fold_ids"] != baseline["fold_ids"]:
            raise SystemExit(f"{mode} does not use the baseline's exact folds")
    structural_hashes = {groups[mode]["forest_sha256"] for mode in MODES[1:]}
    contract_hashes = {groups[mode]["forest_contract_sha256"] for mode in MODES[1:]}
    shuffle_seeds = {groups[mode]["shuffle_seed"] for mode in MODES[1:]}
    if len(structural_hashes) != 1 or None in structural_hashes:
        raise SystemExit("flat/forest/shuffled do not share one frozen forest artifact")
    if len(contract_hashes) != 1 or None in contract_hashes:
        raise SystemExit("flat/forest/shuffled do not share one forest contract")
    if len(shuffle_seeds) != 1:
        raise SystemExit("flat/forest/shuffled do not share one shuffle seed")

    t1 = load_records(TRAIN_TASK1)
    t2 = load_records(TRAIN_TASK2)
    by_id = {record["paragraph_id"]: record for record in t1}
    common = baseline["common"]
    pool_ids = train_val_ids(t1, t2)[0] if common["pool"] == "legacy-train" else sorted(by_id)
    expected_records = [
        by_id[pid] for pid in pool_ids
        if common["setting"] == "both" or by_id[pid]["type"] == common["setting"]
    ]
    expected_ids = {record["paragraph_id"] for record in expected_records}
    clean_t1 = clean_task1_gold(t1, t2)
    clean_t2 = clean_task2_gold(t2)

    results = {}
    all_complete = True
    for mode, group in groups.items():
        ids = {row["paragraph_id"] for row in group["task1_rows"]}
        all_complete &= group["complete"] and ids == expected_ids
        source = [by_id[pid] for pid in sorted(ids)]
        rows1 = sorted(group["task1_rows"], key=lambda row: row["paragraph_id"])
        rows2 = sorted(group["task2_rows"], key=lambda row: row["paragraph_id"])
        problems = (
            validate_records_against_source(rows1, source, "task_1")
            + validate_records_against_source(rows2, source, "task_2")
        )
        if problems:
            raise SystemExit(f"invalid {mode} OOF predictions:\n- " + "\n- ".join(problems[:30]))
        gold1 = {pid: clean_t1[pid] for pid in ids}
        gold2 = {pid: clean_t2[pid] for pid in ids}
        score1 = task1_macro_f1(
            gold1, {row["paragraph_id"]: set(row["labels"]) for row in rows1}
        )
        pred2 = _task2(rows2)
        score2 = span_partial_f1(gold2, pred2)
        genre_scores = {}
        for genre in ("editorial", "debate"):
            genre_ids = {pid for pid in ids if by_id[pid]["type"] == genre}
            genre_scores[genre] = span_partial_f1(
                {pid: clean_t2[pid] for pid in genre_ids},
                {pid: pred2[pid] for pid in genre_ids},
            )["f1"]
        length_scores = {}
        for name, predicate in (
            ("short_le_300", lambda n: n <= 300),
            ("medium_301_769", lambda n: 300 < n <= 769),
            ("long_gt_769", lambda n: n > 769),
        ):
            bin_ids = {pid for pid in ids if predicate(len(by_id[pid]["text"]))}
            length_scores[name] = {
                "n": len(bin_ids),
                "task2_f1": span_partial_f1(
                    {pid: clean_t2[pid] for pid in bin_ids},
                    {pid: pred2[pid] for pid in bin_ids},
                )["f1"],
            }
        results[mode] = {
            "n_paragraphs": len(ids),
            "task1_macro_f1": score1["macro_f1"],
            "task1_per_label_f1": {
                label: value["f1"] for label, value in score1["per_label"].items()
            },
            "task2_f1": score2["f1"],
            "task2_precision": score2["precision"],
            "task2_recall": score2["recall"],
            "task2_per_label_f1": {
                label: value["f1"] for label, value in score2["per_label"].items()
            },
            "task2_by_genre": genre_scores,
            "task2_by_length": length_scores,
            "decision": _decision_totals(group),
        }

    deltas = {
        "flat_minus_baseline": results["flat"]["task2_f1"] - results["baseline"]["task2_f1"],
        "forest_minus_baseline": results["forest"]["task2_f1"] - results["baseline"]["task2_f1"],
        "forest_minus_flat": results["forest"]["task2_f1"] - results["flat"]["task2_f1"],
        "forest_minus_shuffled": results["forest"]["task2_f1"] - results["shuffled"]["task2_f1"],
        "shuffled_minus_flat": results["shuffled"]["task2_f1"] - results["flat"]["task2_f1"],
    }
    adoption = {
        "complete": all_complete,
        "required_delta": V3_ADOPTION_DELTA,
        "forest_beats_baseline_gate": deltas["forest_minus_baseline"] >= V3_ADOPTION_DELTA,
        "forest_beats_flat": deltas["forest_minus_flat"] > 0,
        "forest_beats_shuffled": deltas["forest_minus_shuffled"] > 0,
    }
    adoption["adopt"] = bool(
        adoption["complete"]
        and adoption["forest_beats_baseline_gate"]
        and adoption["forest_beats_flat"]
        and adoption["forest_beats_shuffled"]
    )
    report = {
        "report_schema_version": 1,
        "architecture": COMPARISON_ARCHITECTURE,
        "controlled_contract": common,
        "forest_sha256": next(iter(structural_hashes)),
        "forest_contract_sha256": next(iter(contract_hashes)),
        "shuffle_seed": next(iter(shuffle_seeds)),
        "results": results,
        "task2_deltas": deltas,
        "adoption": adoption,
        "upstream_runs": {mode: group["runs"] for mode, group in groups.items()},
    }

    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"structural-cv-comparison-{common['model']}-{common['setting']}",
        {"contract": common, "forest": report["forest_sha256"]},
    )
    config_path = atomic_write_json(
        exp_dir / "config.json",
        {"script": "compare_structural_cv.py", "argv": sys.argv[1:]},
    )
    report_path = atomic_write_json(exp_dir / "comparison_report.json", report)
    provenance = build_provenance_manifest(
        tasks=(1, 2),
        track=common["track"],
        setting=common["setting"],
        task_model=SPECS[common["model"]],
        optimizer=common["optimizer"],
        training_ids=[],
        data_sources=("daleel2026:train-task-1", "daleel2026:train-task-2"),
        architecture_version=COMPARISON_ARCHITECTURE,
    )
    provenance["upstream_runs"] = report["upstream_runs"]
    provenance["adoption"] = adoption
    provenance["outputs"] = [
        output_record(exp_dir, report_path, kind="structural-cv-comparison")
    ]
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    write_completion_marker(
        exp_dir,
        [config_path, report_path, provenance_path],
        metadata={"adopt": adoption["adopt"]},
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\ncomparison report: {report_path}")


if __name__ == "__main__":
    main()
