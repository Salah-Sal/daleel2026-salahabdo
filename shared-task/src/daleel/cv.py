"""Strict compatibility and fold-membership checks for v3 CV runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .artifacts import canonical_json_sha256, validate_completion_marker
from .folds import fold_manifest_hash


CV_CONTRACT_FIELDS = (
    "architecture",
    "model",
    "track",
    "setting",
    "optimizer",
    "objective",
    "pool",
    "n_folds",
    "n_inner_folds",
    "inner_fold",
    "fold_seed",
    "fold_manifest_sha256",
    "granularity",
    "containment_threshold",
    "context_chars",
    "demo_context_chars",
    "role_cot",
    "demo_selector",
    "max_demos",
    "same_genre_bonus",
    "balance_per_class",
    "optimizer_val_max",
    "optimizer_val_distribution",
    "optimizer_train_distribution",
    "auto",
    "max_metric_calls",
    "max_tokens",
    "max_rerank_candidates",
    "proposal_sha256",
    "accept_delta",
)


def cv_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields that must identify one algorithmic experiment."""

    missing = [field for field in CV_CONTRACT_FIELDS if field not in config]
    if missing:
        raise ValueError(f"v3 run config is missing CV contract fields: {missing}")
    return {field: config[field] for field in CV_CONTRACT_FIELDS}


def load_cv_runs(
    runs: Sequence[str | Path],
    *,
    allow_partial: bool = False,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any], dict[int, set[int]]]:
    """Validate completed runs, one common contract, and exact fold membership."""

    if not runs:
        raise ValueError("at least one v3 CV run is required")
    paths = [Path(run) for run in runs]
    configs: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    fold_ids: dict[int, set[int]] = {}
    outer_manifest: dict[str, Any] | None = None

    for run in paths:
        try:
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
        except Exception as exc:
            raise ValueError(f"v3 run is incomplete or corrupt: {run}: {exc}") from exc
        try:
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            folds = json.loads((run / "folds.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid v3 run metadata in {run}: {exc}") from exc
        if config.get("final"):
            raise ValueError(f"final compile has no untouched outer predictions: {run}")
        contract = cv_contract(config)
        manifest = folds.get("outer")
        if not isinstance(manifest, dict):
            raise ValueError(f"run has no outer fold manifest: {run}")
        actual_hash = fold_manifest_hash(manifest)
        if actual_hash != config.get("fold_manifest_sha256"):
            raise ValueError(f"outer fold manifest hash mismatch: {run}")
        if outer_manifest is None:
            outer_manifest = manifest
        elif manifest != outer_manifest:
            raise ValueError(f"runs do not share an identical outer fold manifest: {run}")

        fold = config.get("outer_fold")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"invalid outer_fold {fold!r}: {run}")
        if fold in fold_ids:
            raise ValueError(f"duplicate outer fold {fold}")
        entries = [entry for entry in manifest.get("folds", []) if entry.get("index") == fold]
        if len(entries) != 1:
            raise ValueError(f"outer fold {fold} is missing/duplicated in {run}/folds.json")
        fold_ids[fold] = set(entries[0].get("val_ids", []))
        configs.append(config)
        contracts.append(contract)

    first_contract = contracts[0]
    for run, contract in zip(paths[1:], contracts[1:]):
        if contract != first_contract:
            differing = {
                key: {"first": first_contract.get(key), "run": contract.get(key)}
                for key in CV_CONTRACT_FIELDS
                if first_contract.get(key) != contract.get(key)
            }
            raise ValueError(f"incompatible v3 CV run {run}: {differing}")

    n_folds = first_contract["n_folds"]
    expected_folds = set(range(n_folds))
    actual_folds = set(fold_ids)
    if not allow_partial and actual_folds != expected_folds:
        raise ValueError(
            f"outer fold set mismatch: expected={sorted(expected_folds)}, "
            f"actual={sorted(actual_folds)}"
        )
    if allow_partial and not actual_folds <= expected_folds:
        raise ValueError(f"unexpected outer fold indices: {sorted(actual_folds - expected_folds)}")
    return paths, configs, first_contract, fold_ids


def cv_contract_sha256(contract: dict[str, Any]) -> str:
    return canonical_json_sha256(contract)


__all__ = [
    "CV_CONTRACT_FIELDS",
    "cv_contract",
    "cv_contract_sha256",
    "load_cv_runs",
]
