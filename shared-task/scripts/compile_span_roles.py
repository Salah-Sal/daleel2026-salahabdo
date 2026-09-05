"""Compile and officially rerank the v3 one-atom Daleel role program.

Compilation is paragraph-grouped: proposal-shaped atom decisions from one fold
train the optimizer, and a disjoint natural-distribution fold is reassembled
and scored with the exact official corpus metrics.  GEPA's internally retained
candidates are all externally reranked; the proxy-selected winner is not
silently assumed to be the end-to-end winner.

Closed-track runs are rejected before any API call if the task, reflection, or
prompt model is proprietary, unregistered, or over 70B.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy  # noqa: E402

from daleel.artifacts import (  # noqa: E402
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    load_artifact_lineage,
    output_record,
    write_completion_marker,
)
from daleel.constants import V3_ADOPTION_DELTA  # noqa: E402
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records  # noqa: E402
from daleel.discourse_forest import (  # noqa: E402
    DEFAULT_SHUFFLE_SEED,
    FOREST_ARCHITECTURE,
    STRUCTURAL_ROLE_ARCHITECTURE,
    STRUCTURE_MODES,
    load_forest_artifact,
)
from daleel.dspy_span_roles import (  # noqa: E402
    BalancedRoleDemoSelector,
    SpanRoleDecision,
    SpanRoleMetric,
    assemble_example_predictions,
    balance_role_examples,
    role_diagnostics,
    span_role_examples,
    structurize_role_examples,
)
from daleel.folds import fold_manifest, fold_manifest_hash, make_stratified_folds  # noqa: E402
from daleel.io import read_jsonl, write_jsonl  # noqa: E402
from daleel.metrics import Span, span_partial_f1, task1_macro_f1  # noqa: E402
from daleel.models import SPECS, make_lm  # noqa: E402
from daleel.provenance import build_provenance_manifest  # noqa: E402
from daleel.role_artifacts import build_role_bundle  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import (  # noqa: E402
    clean_task1_gold,
    clean_task2_gold,
    train_val_ids,
)
from daleel.submission import validate_records_against_source  # noqa: E402

ARCHITECTURE_VERSION = "proposal-atom-role-v3"


def _load_proposals(path: Path, records: list[dict]) -> dict[int, list[Span]]:
    wanted = {record["paragraph_id"] for record in records}
    rows = [row for row in read_jsonl(path) if row.get("paragraph_id") in wanted]
    problems = validate_records_against_source(rows, records, "task_2")
    if problems:
        raise SystemExit("invalid/missing training proposals:\n- " + "\n- ".join(problems[:30]))
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in rows
    }


def _evaluate_candidate(
    program: SpanRoleDecision,
    examples: list[dspy.Example],
    *,
    gold_t1: dict[int, set[str]],
    gold_t2: dict[int, list[Span]],
    paragraph_ids: list[int],
    threads: int,
) -> tuple[dict, dict[int, list[Span]], dict[int, set[str]]]:
    predictions, failed, exceptions = program.batch(
        examples,
        num_threads=threads,
        # DSPy cancels when the threshold is reached.  +1 lets us observe and
        # disqualify a candidate even if every example fails.
        max_errors=max(10, len(examples) + 1),
        return_failed_examples=True,
        provide_traceback=False,
        timeout=0,
        disable_progress_bar=False,
    )
    span_pred, label_pred, assembly = assemble_example_predictions(
        examples,
        predictions,
        fallback_to_draft=False,
    )
    assisted_spans, assisted_labels, assisted_assembly = assemble_example_predictions(
        examples,
        predictions,
        fallback_to_draft=True,
    )
    # Paragraphs with no proposal atoms are valid empty predictions.
    for pid in paragraph_ids:
        span_pred.setdefault(pid, [])
        label_pred.setdefault(pid, set())
        assisted_spans.setdefault(pid, [])
        assisted_labels.setdefault(pid, set())
    t1 = task1_macro_f1(gold_t1, label_pred)
    t2 = span_partial_f1(gold_t2, span_pred)
    assisted_t1 = task1_macro_f1(gold_t1, assisted_labels)
    assisted_t2 = span_partial_f1(gold_t2, assisted_spans)
    diagnostics = role_diagnostics(examples, predictions)
    metrics = {
        "decision": diagnostics,
        "task1_macro_f1": t1["macro_f1"],
        "task1_micro_f1": t1["micro_f1"],
        "task1_per_label_f1": {
            label: values["f1"] for label, values in t1["per_label"].items()
        },
        "task2_f1": t2["f1"],
        "task2_precision": t2["precision"],
        "task2_recall": t2["recall"],
        "task2_per_label_f1": {
            label: values["f1"] for label, values in t2["per_label"].items()
        },
        "assembly": assembly,
        "fallback_assisted": {
            "task1_macro_f1": assisted_t1["macro_f1"],
            "task2_f1": assisted_t2["f1"],
            "assembly": assisted_assembly,
        },
        "failed_examples": len(failed),
        "failure_sample": [f"{type(exc).__name__}: {exc}" for exc in exceptions[:5]],
    }
    return metrics, span_pred, label_pred


def _candidate_ineligibility(metrics: dict) -> list[str]:
    """Reasons a classifier cannot win through fallback-assisted behavior."""

    reasons: list[str] = []
    if metrics.get("failed_examples"):
        reasons.append(f"{metrics['failed_examples']} program failures")
    decision = metrics.get("decision") or {}
    if decision.get("parse_failures"):
        reasons.append(f"{decision['parse_failures']} parse failures")
    if decision.get("unknown_token_count"):
        reasons.append(f"{decision['unknown_token_count']} unknown role tokens")
    if (metrics.get("fallback_assisted") or {}).get("assembly", {}).get("fallback_atoms"):
        reasons.append("fallback-assisted atoms")
    return reasons


def _objective(metrics: dict, name: str) -> float:
    if name == "task1":
        return metrics["task1_macro_f1"]
    if name == "task2":
        return metrics["task2_f1"]
    return (metrics["task1_macro_f1"] + metrics["task2_f1"]) / 2


def _load_adoption_report(path: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    lineage = load_artifact_lineage(
        path,
        expected_task=2,
        requested_track=args.track,
        expected_architecture="proposal-atom-role-v3-cv",
        expected_setting=args.setting,
        expected_model=args.model,
        expected_output_kind="v3-adoption-report",
        require_adopted=True,
        kind="v3-adoption-report",
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid --adoption-report {path}: {exc}") from exc
    primary = report.get("primary") or {}
    if not primary.get("adopt", False):
        raise SystemExit(f"v3 adoption report did not accept its primary objective: {path}")
    if primary.get("required_delta") != V3_ADOPTION_DELTA:
        raise SystemExit(
            f"adoption report threshold {primary.get('required_delta')!r} does not match "
            f"fixed project threshold {V3_ADOPTION_DELTA}"
        )
    contract = report.get("contract") or {}
    expected = {
        "architecture": ARCHITECTURE_VERSION,
        "model": args.model,
        "track": args.track,
        "setting": args.setting,
        "optimizer": args.optimizer,
        "objective": args.objective,
        "pool": args.pool,
        "fold_seed": args.fold_seed,
        "n_folds": args.n_folds,
        "n_inner_folds": args.n_inner_folds,
        "inner_fold": args.inner_fold,
        "granularity": args.granularity,
        "containment_threshold": args.containment_threshold,
        "context_chars": args.context_chars,
        "demo_context_chars": args.demo_context_chars,
        "role_cot": args.role_cot,
        "demo_selector": not args.no_demo_selector,
        "max_demos": args.max_demos,
        "same_genre_bonus": 0.05,
        "balance_per_class": args.balance_per_class,
        "optimizer_val_max": args.optimizer_val_max,
        "optimizer_val_distribution": "natural-without-replacement",
        "optimizer_train_distribution": "class-balanced-with-cycling",
        "auto": args.auto,
        "max_metric_calls": args.max_metric_calls,
        "max_tokens": args.max_tokens,
        "max_rerank_candidates": args.max_rerank_candidates,
        "proposal_sha256": file_sha256(args.proposals),
        "accept_delta": V3_ADOPTION_DELTA,
    }
    mismatches = {
        key: {"report": contract.get(key), "requested": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"final compile does not match adopted CV contract: {mismatches}")
    return report, lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--track", choices=("closed", "open"), default="closed")
    parser.add_argument("--setting", choices=("editorial", "debate", "both"), default="both")
    parser.add_argument("--proposals", type=Path, required=True,
                        help="frozen quote-program Task 2 predictions over the CV pool")
    parser.add_argument("--pool", choices=("legacy-train", "all"), default="legacy-train",
                        help="legacy-train uses the 430-paragraph training side and preserves "
                             "the old 182-paragraph frozen confirmation set")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--inner-fold", type=int, default=0,
                        help="candidate-selection fold carved only from the outer train side")
    parser.add_argument("--n-inner-folds", type=int, default=4)
    parser.add_argument("--final", action="store_true",
                        help="compile with the full pool as outer-train; no OOF holdout evaluation")
    parser.add_argument("--fold-seed", type=int, default=20260710)
    parser.add_argument("--optimizer", choices=("none", "gepa", "mipro"), default="gepa")
    parser.add_argument("--objective", choices=("task1", "task2", "joint"), default="task2")
    parser.add_argument(
        "--adoption-report",
        type=Path,
        default=None,
        help="accepted five-fold report required by --final",
    )
    parser.add_argument(
        "--allow-unapproved-final",
        action="store_true",
        help="engineering-only override; recorded in provenance and never a submission artifact",
    )
    parser.add_argument("--reflection-model", default=None, choices=sorted(SPECS),
                        help="GEPA reflector; defaults to --model")
    parser.add_argument("--prompt-model", default=None, choices=sorted(SPECS),
                        help="MIPRO instruction proposer; defaults to --model")
    parser.add_argument("--auto", choices=("light", "medium", "heavy"), default="light")
    parser.add_argument("--max-metric-calls", type=int, default=None,
                        help="GEPA explicit smoke budget; replaces --auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--context-chars", type=int, default=500)
    parser.add_argument("--demo-context-chars", type=int, default=350)
    parser.add_argument("--max-demos", type=int, default=7,
                        help="balanced dynamic official-data demos (GEPA/none)")
    parser.add_argument("--no-demo-selector", action="store_true")
    parser.add_argument("--balance-per-class", type=int, default=80,
                        help="optimizer-only examples per CO/AS/TE/ST/AN/OT/NONE class")
    parser.add_argument("--optimizer-val-max", type=int, default=300,
                        help="natural, nonduplicated inner-val atom cap; 0 uses all")
    parser.add_argument("--granularity", choices=("sentence", "clause", "connective"),
                        default="connective")
    parser.add_argument("--containment-threshold", type=float, default=0.80)
    parser.add_argument("--role-cot", action="store_true",
                        help="off by default so labeled demos have no missing rationale")
    parser.add_argument(
        "--structure-mode",
        choices=STRUCTURE_MODES,
        default="baseline",
        help="controlled input ablation; baseline preserves the adopted v3 signature",
    )
    parser.add_argument(
        "--forests",
        type=Path,
        default=None,
        help="frozen discourse-forest artifact required by flat/forest/shuffled",
    )
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--max-rerank-candidates", type=int, default=0,
                        help="0 reranks every retained GEPA candidate")
    parser.add_argument("--skip-eval", action="store_true",
                        help="compile/save only; useful for a minimal API smoke run")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate artifacts/folds/contracts and print counts; make no LM")
    args = parser.parse_args()

    numeric_requirements = {
        "--n-folds": (args.n_folds, 2),
        "--n-inner-folds": (args.n_inner_folds, 2),
        "--threads": (args.threads, 1),
        "--max-tokens": (args.max_tokens, 1),
        "--balance-per-class": (args.balance_per_class, 1),
    }
    for name, (value, minimum) in numeric_requirements.items():
        if value < minimum:
            raise SystemExit(f"{name} must be >= {minimum}, got {value}")
    for name, value in {
        "--context-chars": args.context_chars,
        "--demo-context-chars": args.demo_context_chars,
        "--max-demos": args.max_demos,
        "--optimizer-val-max": args.optimizer_val_max,
        "--max-rerank-candidates": args.max_rerank_candidates,
    }.items():
        if value < 0:
            raise SystemExit(f"{name} must be non-negative, got {value}")
    if not 0.0 < args.containment_threshold <= 1.0:
        raise SystemExit("--containment-threshold must lie in (0, 1]")
    if args.max_metric_calls is not None and args.max_metric_calls <= 0:
        raise SystemExit("--max-metric-calls must be positive")
    if not 0 <= args.fold < args.n_folds:
        raise SystemExit(f"--fold must be in [0, {args.n_folds - 1}]")
    if not 0 <= args.inner_fold < args.n_inner_folds:
        raise SystemExit(f"--inner-fold must be in [0, {args.n_inner_folds - 1}]")
    if args.optimizer == "mipro" and not args.no_demo_selector:
        raise SystemExit(
            "MIPRO searches static demos, while call-level dynamic demos override them; "
            "pass --no-demo-selector for a meaningful MIPRO run"
        )
    if args.structure_mode == "baseline" and args.forests is not None:
        raise SystemExit("--forests is unused when --structure-mode=baseline")
    if args.structure_mode != "baseline" and args.forests is None:
        raise SystemExit("flat/forest/shuffled structure modes require --forests")
    if args.structure_mode != "baseline" and args.final:
        raise SystemExit(
            "structural ablations are OOF experiments only; final compilation requires "
            "a completed controlled comparison and a deployment artifact design"
        )
    if args.pool == "all" and not args.final:
        raise SystemExit(
            "--pool all folds the frozen 182-paragraph confirmation set into OOF "
            "selection, spending the one untouched confirmation; it is only legal "
            "for a --final compile that has already passed the adoption gate"
        )
    if args.final and not args.allow_unapproved_final and args.adoption_report is None:
        raise SystemExit("--final requires --adoption-report unless --allow-unapproved-final is set")
    if not args.final and args.adoption_report is not None:
        raise SystemExit("--adoption-report is only valid with --final")
    adoption_report = None
    adoption_lineage = None
    if args.adoption_report is not None:
        adoption_report, adoption_lineage = _load_adoption_report(args.adoption_report, args)

    t1_records = load_records(TRAIN_TASK1)
    t2_records = load_records(TRAIN_TASK2)
    clean_t1 = clean_task1_gold(t1_records, t2_records)
    clean_t2 = clean_task2_gold(t2_records)
    by_id = {record["paragraph_id"]: record for record in t1_records}
    if args.pool == "legacy-train":
        pool_ids = train_val_ids(t1_records, t2_records)[0]
    else:
        pool_ids = sorted(by_id)
    pool_records = [
        by_id[pid]
        for pid in pool_ids
        if args.setting == "both" or by_id[pid]["type"] == args.setting
    ]
    allowed_ids = [record["paragraph_id"] for record in pool_records]
    if len(allowed_ids) < args.n_folds:
        raise SystemExit("training setting/pool is smaller than --n-folds")

    labels_for_folds = {pid: clean_t1[pid] for pid in allowed_ids}
    genres_for_folds = {pid: by_id[pid]["type"] for pid in allowed_ids}
    folds = make_stratified_folds(
        allowed_ids,
        labels_for_folds,
        genres_for_folds,
        n_splits=args.n_folds,
        seed=args.fold_seed,
    )
    fold_info = fold_manifest(folds, seed=args.fold_seed)
    fold_hash = fold_manifest_hash(fold_info)
    if adoption_report is not None and adoption_report["contract"].get(
        "fold_manifest_sha256"
    ) != fold_hash:
        raise SystemExit("final compile fold manifest does not match the adopted CV report")

    if args.final:
        outer_train_ids = tuple(allowed_ids)
        outer_val_ids: tuple[int, ...] = ()
    else:
        outer_fold = folds[args.fold]
        outer_train_ids = outer_fold.train_ids
        outer_val_ids = outer_fold.val_ids

    # Nested selection is deliberate.  GEPA/MIPRO and official candidate
    # reranking may inspect only this inner validation set; the outer fold is
    # evaluated once after one candidate has already been selected.
    inner_labels = {pid: clean_t1[pid] for pid in outer_train_ids}
    inner_genres = {pid: by_id[pid]["type"] for pid in outer_train_ids}
    inner_folds = make_stratified_folds(
        outer_train_ids,
        inner_labels,
        inner_genres,
        n_splits=args.n_inner_folds,
        seed=args.fold_seed + 1000 + args.fold,
    )
    selected_inner = inner_folds[args.inner_fold]
    inner_fold_info = fold_manifest(
        inner_folds, seed=args.fold_seed + 1000 + args.fold
    )
    inner_fold_hash = fold_manifest_hash(inner_fold_info)

    proposal_lineage = load_artifact_lineage(
        args.proposals,
        expected_task=2,
        requested_track=args.track,
        expected_architecture="stage0-quote-v1",
        expected_output_kind="quote-proposals",
        kind="frozen-quote-proposals",
    )
    proposals = _load_proposals(args.proposals, pool_records)
    forest_lineage = None
    forest_contract = None
    forests = None
    if args.structure_mode != "baseline":
        forest_lineage = load_artifact_lineage(
            args.forests,
            expected_task=2,
            requested_track=args.track,
            expected_architecture=FOREST_ARCHITECTURE,
            expected_setting=args.setting,
            expected_model=args.model,
            expected_output_kind="discourse-forests",
            kind="frozen-discourse-forests",
        )
        try:
            forest_payload = json.loads(args.forests.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid --forests artifact {args.forests}: {exc}") from exc
        try:
            forests, forest_contract = load_forest_artifact(
                forest_payload,
                source_records=pool_records,
                proposals=proposals,
                expected_proposal_sha256=proposal_lineage["sha256"],
                expected_granularity=args.granularity,
            )
        except Exception as exc:
            raise SystemExit(f"forest artifact validation failed: {exc}") from exc
        for key, expected in {
            "model": args.model,
            "track": args.track,
            "setting": args.setting,
            "pool": args.pool,
        }.items():
            if forest_contract.get(key) != expected:
                raise SystemExit(
                    f"forest contract {key}={forest_contract.get(key)!r}, "
                    f"requested {expected!r}"
                )
    all_examples = span_role_examples(
        pool_records,
        clean_t2,
        proposals=proposals,
        granularity=args.granularity,
        containment_threshold=args.containment_threshold,
    )
    if args.structure_mode != "baseline":
        all_examples = structurize_role_examples(
            all_examples,
            forests,
            mode=args.structure_mode,
            shuffle_seed=args.shuffle_seed,
        )
    inner_train_id_set = set(selected_inner.train_ids)
    inner_val_id_set = set(selected_inner.val_ids)
    outer_train_id_set = set(outer_train_ids)
    outer_val_id_set = set(outer_val_ids)
    natural_inner_train = [
        ex for ex in all_examples if ex.paragraph_id in inner_train_id_set
    ]
    natural_inner_val = [
        ex for ex in all_examples if ex.paragraph_id in inner_val_id_set
    ]
    natural_outer_train = [
        ex for ex in all_examples if ex.paragraph_id in outer_train_id_set
    ]
    natural_outer_val = [
        ex for ex in all_examples if ex.paragraph_id in outer_val_id_set
    ]
    optimizer_train = balance_role_examples(
        natural_inner_train,
        per_class=args.balance_per_class,
        seed=args.fold_seed + args.fold,
    )
    optimizer_val = list(natural_inner_val)
    random.Random(args.fold_seed + 100 + args.fold).shuffle(optimizer_val)
    if args.optimizer_val_max > 0:
        optimizer_val = optimizer_val[: args.optimizer_val_max]

    selector = None
    if not args.no_demo_selector:
        selector = BalancedRoleDemoSelector(
            natural_inner_train,
            max_demos=args.max_demos,
            context_chars=args.demo_context_chars,
            include_structure=args.structure_mode != "baseline",
        )
    program = SpanRoleDecision(
        cot=args.role_cot,
        demo_selector=selector,
        context_chars=args.context_chars,
        structured=args.structure_mode != "baseline",
    )
    metric = SpanRoleMetric()

    task_spec = SPECS[args.model]
    reflection_spec = SPECS[args.reflection_model or args.model] if args.optimizer == "gepa" else None
    prompt_spec = SPECS[args.prompt_model or args.model] if args.optimizer == "mipro" else None
    provenance = build_provenance_manifest(
        tasks=(1, 2),
        track=args.track,
        setting=args.setting,
        task_model=task_spec,
        optimizer=None if args.optimizer == "none" else args.optimizer,
        training_ids=outer_train_ids,
        data_sources=("daleel2026:train-task-1", "daleel2026:train-task-2"),
        architecture_version=(
            ARCHITECTURE_VERSION
            if args.structure_mode == "baseline"
            else STRUCTURAL_ROLE_ARCHITECTURE
        ),
        reflection_model=reflection_spec,
        prompt_model=prompt_spec,
    )
    proposal_lineage_for_manifest = {
        key: value for key, value in proposal_lineage.items() if key != "manifest_data"
    }
    provenance["upstream_artifacts"] = [proposal_lineage_for_manifest]
    if forest_lineage is not None:
        provenance["upstream_artifacts"].append(
            {key: value for key, value in forest_lineage.items() if key != "manifest_data"}
        )
    provenance["data_artifacts"] = [
        {
            "source_id": "daleel2026:train-task-1",
            "path": str(TRAIN_TASK1),
            "sha256": file_sha256(TRAIN_TASK1),
        },
        {
            "source_id": "daleel2026:train-task-2",
            "path": str(TRAIN_TASK2),
            "sha256": file_sha256(TRAIN_TASK2),
        },
    ]
    if args.adoption_report is not None:
        provenance["adoption_report"] = {
            key: value
            for key, value in adoption_lineage.items()
            if key != "manifest_data"
        }
    provenance["unapproved_final_override"] = bool(
        args.final and args.allow_unapproved_final
    )

    architecture_version = (
        ARCHITECTURE_VERSION
        if args.structure_mode == "baseline"
        else STRUCTURAL_ROLE_ARCHITECTURE
    )
    pre_registration = {
        "architecture": architecture_version,
        "objective": args.objective,
        "accept_delta": V3_ADOPTION_DELTA,
        "candidate_selection": "official-inner-fold-rerank",
        "outer_evaluation": "untouched-once" if not args.final else None,
        "outer_fold": None if args.final else args.fold,
        "inner_fold": args.inner_fold,
        "outer_fold_manifest_sha256": fold_hash,
        "inner_fold_manifest_sha256": inner_fold_hash,
        "optimizer": args.optimizer,
        "optimizer_budget": {
            "auto": None if args.max_metric_calls is not None else args.auto,
            "max_metric_calls": args.max_metric_calls,
        },
        "optimizer_train_distribution": "class-balanced-with-cycling",
        "optimizer_val_distribution": "natural-without-replacement",
        "optimizer_val_max": args.optimizer_val_max,
        "model": args.model,
        "track": args.track,
        "setting": args.setting,
        "pool": args.pool,
        "n_folds": args.n_folds,
        "n_inner_folds": args.n_inner_folds,
        "fold_seed": args.fold_seed,
        "role_cot": args.role_cot,
        "demo_selector": not args.no_demo_selector,
        "max_demos": args.max_demos,
        "context_chars": args.context_chars,
        "demo_context_chars": args.demo_context_chars,
        "same_genre_bonus": 0.05,
        "granularity": args.granularity,
        "containment_threshold": args.containment_threshold,
        "proposal_sha256": proposal_lineage["sha256"],
        "structure_mode": args.structure_mode,
        "forest_sha256": (
            forest_lineage["sha256"] if forest_lineage is not None else None
        ),
        "forest_contract_sha256": (
            forest_payload["contract_sha256"]
            if args.structure_mode != "baseline"
            else None
        ),
        "shuffle_seed": args.shuffle_seed,
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "balance_per_class": args.balance_per_class,
        "max_rerank_candidates": args.max_rerank_candidates,
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight_ok": True,
                    "pre_registration": pre_registration,
                    "counts": {
                        "pool_paragraphs": len(pool_records),
                        "outer_train_paragraphs": len(outer_train_ids),
                        "outer_val_paragraphs": len(outer_val_ids),
                        "natural_inner_train_decisions": len(natural_inner_train),
                        "natural_inner_val_decisions": len(natural_inner_val),
                        "optimizer_train_decisions": len(optimizer_train),
                        "optimizer_val_decisions": len(optimizer_val),
                        "natural_outer_train_decisions": len(natural_outer_train),
                        "natural_outer_val_decisions": len(natural_outer_val),
                    },
                    "proposal_lineage": proposal_lineage_for_manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    split_tag = "final" if args.final else f"outer{args.fold}"
    experiment_slug = (
        f"v3-role-{args.optimizer}-{task_spec.key}-"
        f"{args.setting}-{split_tag}-inner{args.inner_fold}"
        if args.structure_mode == "baseline"
        else f"v4-structure-{args.structure_mode}-{args.optimizer}-{task_spec.key}-"
        f"{args.setting}-{split_tag}-inner{args.inner_fold}"
    )
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        experiment_slug,
        pre_registration,
    )
    compiled_dir = exp_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=False)
    pre_registration_path = atomic_write_json(
        exp_dir / "pre_registration.json", pre_registration
    )
    folds_path = atomic_write_json(
        exp_dir / "folds.json", {"outer": fold_info, "inner": inner_fold_info}
    )
    pending_provenance_path = atomic_write_json(
        exp_dir / "provenance.pending.json", provenance
    )

    # All validation and preregistration above intentionally happens before LM
    # construction or any provider call.
    task_lm = make_lm(task_spec, max_tokens=args.max_tokens)
    # ChatAdapter, not JSONAdapter: gemma via OpenRouter fails structured-output
    # parsing on ~2.8% of atom decisions (fold-0 smoke: 25/903, empty {} and
    # tag-repetition loops), which zero-tolerance eligibility turns into "no
    # eligible candidate". ChatAdapter emits the same list[str] shape with 1.0
    # measured format compliance on this model.
    dspy.configure(lm=task_lm, adapter=dspy.ChatAdapter())

    started = time.time()
    if args.optimizer == "none":
        compiled = program
    elif args.optimizer == "gepa":
        budget = (
            {"max_metric_calls": args.max_metric_calls}
            if args.max_metric_calls is not None
            else {"auto": args.auto}
        )
        optimizer = dspy.GEPA(
            metric=metric,
            reflection_lm=make_lm(
                reflection_spec,
                temperature=1.0,
                max_tokens=max(args.max_tokens, 8000),
            ),
            num_threads=args.threads,
            add_format_failure_as_feedback=True,
            track_stats=True,
            log_dir=str(exp_dir / "gepa_logs"),
            seed=args.fold_seed + args.fold,
            **budget,
        )
        compiled = optimizer.compile(
            program,
            trainset=optimizer_train,
            valset=optimizer_val,
        )
    else:
        optimizer = dspy.MIPROv2(
            metric=metric,
            prompt_model=make_lm(prompt_spec, max_tokens=max(args.max_tokens, 6000)),
            task_model=task_lm,
            auto=args.auto,
            max_bootstrapped_demos=0,
            max_labeled_demos=args.max_demos,
            num_threads=args.threads,
            seed=args.fold_seed + args.fold,
            track_stats=True,
            log_dir=str(exp_dir / "mipro_logs"),
        )
        compiled = optimizer.compile(
            program,
            trainset=optimizer_train,
            valset=optimizer_val,
        )
    compile_seconds = time.time() - started

    candidates = [compiled]
    proxy_scores = [None]
    if args.optimizer == "gepa" and hasattr(compiled, "detailed_results"):
        candidates = list(compiled.detailed_results.candidates)
        proxy_scores = list(compiled.detailed_results.val_aggregate_scores)
    if args.max_rerank_candidates > 0:
        order = sorted(
            range(len(candidates)),
            key=lambda i: float("-inf") if proxy_scores[i] is None else proxy_scores[i],
            reverse=True,
        )[: args.max_rerank_candidates]
        candidates = [candidates[i] for i in order]
        proxy_scores = [proxy_scores[i] for i in order]

    inner_val_ids = list(selected_inner.val_ids)
    gold_t1_inner = {pid: clean_t1[pid] for pid in inner_val_ids}
    gold_t2_inner = {pid: clean_t2[pid] for pid in inner_val_ids}
    reranked: list[dict] = []
    if not args.skip_eval:
        for index, (candidate, proxy_score) in enumerate(zip(candidates, proxy_scores)):
            candidate_path = compiled_dir / f"candidate-{index:03d}.json"
            candidate.save(candidate_path, save_program=False)
            try:
                metrics, _, _ = _evaluate_candidate(
                    candidate,
                    natural_inner_val,
                    gold_t1=gold_t1_inner,
                    gold_t2=gold_t2_inner,
                    paragraph_ids=inner_val_ids,
                    threads=args.threads,
                )
                ineligibility = _candidate_ineligibility(metrics)
            except Exception as exc:  # isolate one invalid retained candidate
                metrics = {
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                    "failed_examples": len(natural_inner_val),
                }
                ineligibility = ["candidate evaluation raised an exception"]
            objective = None if ineligibility else _objective(metrics, args.objective)
            reranked.append(
                {
                    "candidate_index": index,
                    "dspy_proxy_score": (
                        None if proxy_score is None else float(proxy_score)
                    ),
                    "eligible": not ineligibility,
                    "ineligibility_reasons": ineligibility,
                    "external_objective": objective,
                    "state": candidate_path.relative_to(exp_dir).as_posix(),
                    "metrics": metrics,
                }
            )
        eligible = [row for row in reranked if row["eligible"]]
        if not eligible:
            failure_metrics_path = atomic_write_json(
                exp_dir / "metrics.failed.json",
                {
                    "compile_seconds": round(compile_seconds, 2),
                    "objective": args.objective,
                    "candidates": reranked,
                    "failure": "no retained candidate passed native-output eligibility",
                },
            )
            raise SystemExit(
                "no retained candidate passed native-output eligibility; "
                f"details: {failure_metrics_path}"
            )
        best = max(eligible, key=lambda row: row["external_objective"])
        selected_candidate = candidates[best["candidate_index"]]
        selected_index = best["candidate_index"]
    else:
        selected_candidate = compiled
        selected_index = None

    # Dynamic demonstrations are data, not compiled predictor parameters.
    # Once prompt selection is over, rebuild their pool from the entire outer
    # training side before the untouched outer-fold evaluation/deployment.
    if not args.no_demo_selector:
        selected_candidate.demo_selector = BalancedRoleDemoSelector(
            natural_outer_train,
            max_demos=args.max_demos,
            context_chars=args.demo_context_chars,
            include_structure=args.structure_mode != "baseline",
        )

    outer_metrics = None
    outer_span_predictions: dict[int, list[Span]] | None = None
    outer_label_predictions: dict[int, set[str]] | None = None
    if not args.final and not args.skip_eval:
        outer_ids = list(outer_val_ids)
        outer_metrics, outer_span_predictions, outer_label_predictions = _evaluate_candidate(
            selected_candidate,
            natural_outer_val,
            gold_t1={pid: clean_t1[pid] for pid in outer_ids},
            gold_t2={pid: clean_t2[pid] for pid in outer_ids},
            paragraph_ids=outer_ids,
            threads=args.threads,
        )
        outer_ineligibility = _candidate_ineligibility(outer_metrics)
        if outer_ineligibility:
            atomic_write_json(
                exp_dir / "outer_holdout.failed.json",
                {"metrics": outer_metrics, "ineligibility_reasons": outer_ineligibility},
            )
            raise SystemExit(
                "selected candidate failed on untouched outer holdout: "
                + "; ".join(outer_ineligibility)
            )

    selected_path = compiled_dir / "role.json"
    temporary_state = compiled_dir / ".role.tmp.json"
    selected_candidate.save(temporary_state, save_program=False)
    os.replace(temporary_state, selected_path)
    bundle_settings = {
        "role_cot": args.role_cot,
        "demo_selector": not args.no_demo_selector,
        "max_demos": args.max_demos,
        "context_chars": args.context_chars,
        "demo_context_chars": args.demo_context_chars,
        "same_genre_bonus": 0.05,
        "granularity": args.granularity,
        "containment_threshold": args.containment_threshold,
    }
    bundle_path = None
    if args.structure_mode == "baseline":
        bundle_path, _ = build_role_bundle(
            run_dir=exp_dir,
            role_state=selected_path,
            memory_examples=(None if args.no_demo_selector else natural_outer_train),
            settings=bundle_settings,
            proposal_sha256=proposal_lineage["sha256"],
            fold_manifest_sha256=fold_hash,
            model=args.model,
            track=args.track,
            setting=args.setting,
        )
    instructions = {
        name: predictor.signature.instructions
        for name, predictor in selected_candidate.named_predictors()
    }
    instructions_path = atomic_write_json(compiled_dir / "instructions.json", instructions)

    task1_prediction_path: Path | None = None
    task2_prediction_path: Path | None = None
    if outer_span_predictions is not None and outer_label_predictions is not None:
        prediction_dir = exp_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        outer_records = [by_id[pid] for pid in outer_val_ids]
        task1_rows = [
            {
                "paragraph_id": record["paragraph_id"],
                "text": record["text"],
                "type": record["type"],
                "labels": sorted(outer_label_predictions.get(record["paragraph_id"], set())),
            }
            for record in outer_records
        ]
        task2_rows = [
            {
                "paragraph_id": record["paragraph_id"],
                "text": record["text"],
                "type": record["type"],
                "labels": [
                    {
                        "label": span.label,
                        "start_offset": span.start,
                        "end_offset": span.end,
                    }
                    for span in outer_span_predictions.get(record["paragraph_id"], ())
                ],
            }
            for record in outer_records
        ]
        task1_problems = validate_records_against_source(task1_rows, outer_records, "task_1")
        task2_problems = validate_records_against_source(task2_rows, outer_records, "task_2")
        if task1_problems or task2_problems:
            raise SystemExit(
                "outer-fold source preflight failed:\n- "
                + "\n- ".join((task1_problems + task2_problems)[:30])
            )
        task1_prediction_path = prediction_dir / "outer_task_1.jsonl"
        task2_prediction_path = prediction_dir / "outer_task_2.jsonl"
        write_jsonl(task1_prediction_path, task1_rows)
        write_jsonl(task2_prediction_path, task2_rows)

    config = {
        "script": "compile_span_roles.py",
        "argv": sys.argv[1:],
        "architecture": architecture_version,
        "model": args.model,
        "track": args.track,
        "setting": args.setting,
        "optimizer": args.optimizer,
        "objective": args.objective,
        "accept_delta": V3_ADOPTION_DELTA,
        "adoption_report": str(args.adoption_report) if args.adoption_report else None,
        "unapproved_final_override": bool(args.final and args.allow_unapproved_final),
        "proposal_file": str(args.proposals),
        "proposal_sha256": proposal_lineage["sha256"],
        "structure_mode": args.structure_mode,
        "forest_file": str(args.forests) if args.forests else None,
        "forest_sha256": (
            forest_lineage["sha256"] if forest_lineage is not None else None
        ),
        "forest_contract_sha256": (
            forest_payload["contract_sha256"]
            if args.structure_mode != "baseline"
            else None
        ),
        "shuffle_seed": args.shuffle_seed,
        "pool": args.pool,
        "final": args.final,
        "outer_fold": None if args.final else args.fold,
        "n_folds": args.n_folds,
        "inner_fold": args.inner_fold,
        "n_inner_folds": args.n_inner_folds,
        "fold_seed": args.fold_seed,
        "fold_manifest_sha256": fold_hash,
        "inner_fold_manifest_sha256": inner_fold_hash,
        "granularity": args.granularity,
        "containment_threshold": args.containment_threshold,
        "context_chars": args.context_chars,
        "demo_context_chars": args.demo_context_chars,
        "role_cot": args.role_cot,
        "demo_selector": not args.no_demo_selector,
        "max_demos": args.max_demos,
        "same_genre_bonus": 0.05,
        "balance_per_class": args.balance_per_class,
        "optimizer_val_max": args.optimizer_val_max,
        "auto": args.auto,
        "max_metric_calls": args.max_metric_calls,
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "max_rerank_candidates": args.max_rerank_candidates,
        "optimizer_train_decisions": len(optimizer_train),
        "optimizer_val_decisions": len(optimizer_val),
        "optimizer_val_distribution": "natural-without-replacement",
        "optimizer_train_distribution": "class-balanced-with-cycling",
        "natural_inner_train_decisions": len(natural_inner_train),
        "natural_inner_val_decisions": len(natural_inner_val),
        "natural_outer_train_decisions": len(natural_outer_train),
        "natural_outer_val_decisions": len(natural_outer_val),
        "selected_candidate_index": selected_index,
        "selected_state": selected_path.relative_to(exp_dir).as_posix(),
        "role_bundle": (
            bundle_path.relative_to(exp_dir).as_posix()
            if bundle_path is not None
            else None
        ),
    }
    metrics = {
        "compile_seconds": round(compile_seconds, 2),
        "n_candidates": len(candidates),
        "objective": args.objective,
        "selected_candidate_index": selected_index,
        "candidates": reranked,
        "outer_holdout": outer_metrics,
    }
    config_path = atomic_write_json(exp_dir / "config.json", config)
    metrics_path = atomic_write_json(exp_dir / "metrics.json", metrics)

    outputs = [
        output_record(
            exp_dir,
            selected_path,
            kind=(
                "v3-role-state"
                if args.structure_mode == "baseline"
                else "structural-role-state"
            ),
        ),
        output_record(exp_dir, instructions_path, kind="v3-role-instructions"),
    ]
    if bundle_path is not None:
        outputs.append(output_record(exp_dir, bundle_path, kind="v3-role-bundle"))
    memory_path = compiled_dir / "role_memory.jsonl"
    if memory_path.exists():
        outputs.append(output_record(exp_dir, memory_path, kind="v3-role-memory"))
    if task1_prediction_path is not None and task2_prediction_path is not None:
        outputs.extend(
            [
                output_record(exp_dir, task1_prediction_path, kind="v3-outer-task1"),
                output_record(exp_dir, task2_prediction_path, kind="v3-outer-task2"),
            ]
        )
    provenance["outputs"] = outputs
    provenance_path = atomic_write_json(exp_dir / "provenance.json", provenance)
    pending_provenance_path.unlink(missing_ok=True)
    completion_paths = [
        pre_registration_path,
        folds_path,
        config_path,
        metrics_path,
        provenance_path,
        selected_path,
        instructions_path,
    ]
    if bundle_path is not None:
        completion_paths.append(bundle_path)
    if memory_path.exists():
        completion_paths.append(memory_path)
    if task1_prediction_path is not None and task2_prediction_path is not None:
        completion_paths.extend([task1_prediction_path, task2_prediction_path])
    write_completion_marker(
        exp_dir,
        completion_paths,
        metadata={
            "architecture": architecture_version,
            "structure_mode": args.structure_mode,
            "outer_fold": None if args.final else args.fold,
            "final": args.final,
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nselected state: {selected_path}")


if __name__ == "__main__":
    main()
