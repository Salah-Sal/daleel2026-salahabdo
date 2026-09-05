"""Complete, hash-bound artifacts for the v3 atom-role classifier.

DSPy state-only serialization intentionally stores predictor parameters, not
ordinary Python module attributes.  The v3 program also depends on a dynamic
demonstration selector, its exact legal example pool, and structural settings.
This module snapshots those dependencies and verifies every byte before a
saved role program can be reconstructed.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from . import runtime as _runtime  # noqa: F401 -- before importing dspy

import dspy

from .artifacts import atomic_write_json, file_sha256
from .io import read_jsonl, write_jsonl
from .provenance import ComplianceError, training_ids_sha256


ROLE_BUNDLE_SCHEMA_VERSION = 1
ROLE_BUNDLE_FILENAME = "role_bundle.json"
ROLE_MEMORY_FILENAME = "role_memory.jsonl"

_REQUIRED_EXAMPLE_FIELDS = (
    "paragraph_id",
    "text",
    "genre",
    "start",
    "end",
    "adu_roles",
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:  # pragma: no cover - environment failure
        raise ComplianceError(f"required package version is unavailable: {name}") from exc


def _example_record(example: dspy.Example) -> dict[str, Any]:
    record = {field: getattr(example, field) for field in _REQUIRED_EXAMPLE_FIELDS}
    record["adu_roles"] = list(record["adu_roles"])
    return record


def snapshot_role_examples(
    path: str | Path,
    examples: Sequence[dspy.Example],
) -> dict[str, Any]:
    """Write exact selector examples and return their bundle metadata."""

    path = Path(path)
    rows = [_example_record(example) for example in examples]
    write_jsonl(path, rows)
    paragraph_ids = sorted({row["paragraph_id"] for row in rows})
    return {
        "relative_path": path.name if path.parent.name != "compiled" else f"compiled/{path.name}",
        "sha256": file_sha256(path),
        "example_count": len(rows),
        "paragraph_ids": paragraph_ids,
        "paragraph_ids_sha256": training_ids_sha256(paragraph_ids),
    }


def load_role_examples(path: str | Path) -> list[dspy.Example]:
    """Load and strictly validate a selector-memory snapshot."""

    path = Path(path)
    rows = read_jsonl(path)
    examples: list[dspy.Example] = []
    for index, row in enumerate(rows, start=1):
        missing = [field for field in _REQUIRED_EXAMPLE_FIELDS if field not in row]
        if missing:
            raise ComplianceError(
                f"role memory {path}, row {index} is missing fields {missing}"
            )
        pid, text = row["paragraph_id"], row["text"]
        start, end = row["start"], row["end"]
        genre, roles = row["genre"], row["adu_roles"]
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise ComplianceError(f"role memory row {index} has invalid paragraph_id {pid!r}")
        if not isinstance(text, str) or not text:
            raise ComplianceError(f"role memory row {index} has invalid text")
        if genre not in {"editorial", "debate"}:
            raise ComplianceError(f"role memory row {index} has invalid genre {genre!r}")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            raise ComplianceError(
                f"role memory row {index} has invalid target [{start!r}, {end!r})"
            )
        if not isinstance(roles, list) or any(not isinstance(role, str) for role in roles):
            raise ComplianceError(f"role memory row {index} has invalid adu_roles")
        examples.append(
            dspy.Example(
                paragraph_id=pid,
                text=text,
                genre=genre,
                start=start,
                end=end,
                adu_roles=roles,
            ).with_inputs("paragraph_id", "text", "genre", "start", "end")
        )
    return examples


def build_role_bundle(
    *,
    run_dir: str | Path,
    role_state: str | Path,
    memory_examples: Sequence[dspy.Example] | None,
    settings: dict[str, Any],
    proposal_sha256: str,
    fold_manifest_sha256: str,
    model: str,
    track: str,
    setting: str,
) -> tuple[Path, dict[str, Any]]:
    """Snapshot all non-predictor state and write ``role_bundle.json``."""

    run_dir = Path(run_dir).resolve()
    role_state = Path(role_state).resolve()
    try:
        state_relative = role_state.relative_to(run_dir).as_posix()
    except ValueError as exc:
        raise ComplianceError("role state must live inside its experiment directory") from exc
    memory_metadata = None
    if memory_examples is not None:
        memory_path = role_state.parent / ROLE_MEMORY_FILENAME
        memory_metadata = snapshot_role_examples(memory_path, memory_examples)
        memory_metadata["relative_path"] = memory_path.relative_to(run_dir).as_posix()

    required_settings = {
        "role_cot",
        "demo_selector",
        "max_demos",
        "context_chars",
        "demo_context_chars",
        "same_genre_bonus",
        "granularity",
        "containment_threshold",
    }
    missing = sorted(required_settings - set(settings))
    if missing:
        raise ComplianceError(f"role bundle settings are missing {missing}")

    bundle = {
        "role_bundle_schema_version": ROLE_BUNDLE_SCHEMA_VERSION,
        "architecture_version": "proposal-atom-role-v3",
        "model": model,
        "track": track,
        "setting": setting,
        "state": {
            "relative_path": state_relative,
            "sha256": file_sha256(role_state),
        },
        "memory": memory_metadata,
        "settings": settings,
        "proposal_sha256": proposal_sha256,
        "fold_manifest_sha256": fold_manifest_sha256,
        "dependencies": {
            "dspy": _package_version("dspy"),
            "scikit-learn": _package_version("scikit-learn"),
        },
    }
    bundle_path = role_state.parent / ROLE_BUNDLE_FILENAME
    atomic_write_json(bundle_path, bundle)
    return bundle_path, bundle


def _contained_path(run_dir: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ComplianceError(f"role bundle {label} relative_path is invalid")
    candidate = (run_dir / relative_path).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError as exc:
        raise ComplianceError(f"role bundle {label} escapes its experiment directory") from exc
    return candidate


def load_role_bundle(
    role_state: str | Path,
) -> tuple[dict[str, Any], list[dspy.Example] | None]:
    """Verify a bundle and return metadata plus its exact selector examples."""

    role_state = Path(role_state).resolve()
    run_dir = role_state.parent.parent.resolve()
    bundle_path = role_state.parent / ROLE_BUNDLE_FILENAME
    if not bundle_path.is_file():
        raise ComplianceError(f"compiled role state is missing {bundle_path.name}: {bundle_path}")
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComplianceError(f"invalid role bundle {bundle_path}: {exc}") from exc
    if bundle.get("role_bundle_schema_version") != ROLE_BUNDLE_SCHEMA_VERSION:
        raise ComplianceError(
            f"unsupported role bundle schema {bundle.get('role_bundle_schema_version')!r}"
        )
    if bundle.get("architecture_version") != "proposal-atom-role-v3":
        raise ComplianceError(
            f"role bundle architecture is {bundle.get('architecture_version')!r}"
        )

    state = bundle.get("state") or {}
    recorded_state = _contained_path(run_dir, state.get("relative_path"), label="state")
    if recorded_state != role_state:
        raise ComplianceError(
            f"requested role state {role_state} is not bundle state {recorded_state}"
        )
    if not role_state.is_file() or file_sha256(role_state) != state.get("sha256"):
        raise ComplianceError("role state SHA-256 does not match role bundle")

    dependencies = bundle.get("dependencies") or {}
    for package in ("dspy", "scikit-learn"):
        if dependencies.get(package) != _package_version(package):
            raise ComplianceError(
                f"role bundle requires {package} {dependencies.get(package)!r}, "
                f"installed {_package_version(package)!r}"
            )

    settings = bundle.get("settings")
    if not isinstance(settings, dict):
        raise ComplianceError("role bundle settings must be an object")
    required_settings = {
        "role_cot",
        "demo_selector",
        "max_demos",
        "context_chars",
        "demo_context_chars",
        "same_genre_bonus",
        "granularity",
        "containment_threshold",
    }
    missing_settings = sorted(required_settings - set(settings))
    if missing_settings:
        raise ComplianceError(f"role bundle settings are missing {missing_settings}")

    memory_info = bundle.get("memory")
    examples = None
    if memory_info is not None:
        memory_path = _contained_path(
            run_dir, memory_info.get("relative_path"), label="memory"
        )
        if not memory_path.is_file() or file_sha256(memory_path) != memory_info.get("sha256"):
            raise ComplianceError("role memory SHA-256 does not match role bundle")
        examples = load_role_examples(memory_path)
        if len(examples) != memory_info.get("example_count"):
            raise ComplianceError("role memory example count does not match role bundle")
        paragraph_ids = sorted({example.paragraph_id for example in examples})
        if paragraph_ids != memory_info.get("paragraph_ids"):
            raise ComplianceError("role memory paragraph IDs do not match role bundle")
        if training_ids_sha256(paragraph_ids) != memory_info.get("paragraph_ids_sha256"):
            raise ComplianceError("role memory paragraph-ID hash does not match role bundle")
    if bool(settings.get("demo_selector")) != (memory_info is not None):
        raise ComplianceError(
            "role bundle demo_selector setting and memory presence are inconsistent"
        )
    return bundle, examples


__all__ = [
    "ROLE_BUNDLE_FILENAME",
    "ROLE_BUNDLE_SCHEMA_VERSION",
    "ROLE_MEMORY_FILENAME",
    "build_role_bundle",
    "load_role_bundle",
    "load_role_examples",
    "snapshot_role_examples",
]
