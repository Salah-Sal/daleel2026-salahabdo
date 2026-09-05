"""Hash and validate model-generated artifact lineage across Daleel stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable

from .provenance import ComplianceError, ORGANIZER_DATA_SOURCE_ALLOWLIST


def file_sha256(path: str | Path) -> str:
    """Stream a file into a stable SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-native value using a stable, strict serialization."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Atomically replace a UTF-8 text file and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically write indented strict JSON with a trailing newline."""

    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    return atomic_write_text(path, rendered + "\n")


def create_experiment_dir(
    base_dir: str | Path,
    slug: str,
    run_contract: Any,
) -> Path:
    """Create a collision-resistant experiment directory exactly once.

    UTC microseconds distinguish concurrent launches while the short canonical
    contract hash makes otherwise similar directory names auditable.
    """

    safe_slug = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in slug
    ).strip("-")
    if not safe_slug:
        raise ValueError("experiment slug must contain at least one safe character")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    digest = canonical_json_sha256(run_contract)[:10]
    path = Path(base_dir) / f"{stamp}-{safe_slug}-{digest}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def output_record(run_dir: str | Path, path: str | Path, *, kind: str) -> dict[str, str]:
    """Return a path/SHA-bound provenance output record."""

    run_dir = Path(run_dir).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError(f"output {path} is outside experiment directory {run_dir}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "kind": kind,
        "relative_path": relative.as_posix(),
        "sha256": file_sha256(path),
    }


def write_completion_marker(
    run_dir: str | Path,
    paths: Iterable[str | Path],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the final marker only after all required run files exist.

    The marker binds every required file except itself.  Its presence is the
    unambiguous distinction between a completed experiment and an interrupted
    directory containing partial but individually valid artifacts.
    """

    run_dir = Path(run_dir).resolve()
    files = []
    for item in paths:
        path = Path(item).resolve()
        try:
            relative = path.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError(f"completion file {path} is outside {run_dir}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"relative_path": relative.as_posix(), "sha256": file_sha256(path)})
    marker = {
        "completion_schema_version": 1,
        "complete": True,
        "files": sorted(files, key=lambda row: row["relative_path"]),
        "metadata": metadata or {},
    }
    return atomic_write_json(run_dir / "COMPLETED.json", marker)


def validate_completion_marker(
    run_dir: str | Path,
    *,
    required_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Verify a completion marker and every file hash it claims."""

    run_dir = Path(run_dir).resolve()
    marker_path = run_dir / "COMPLETED.json"
    if not marker_path.is_file():
        raise ComplianceError(f"experiment is incomplete (missing {marker_path})")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComplianceError(f"invalid completion marker {marker_path}: {exc}") from exc
    if marker.get("completion_schema_version") != 1 or marker.get("complete") is not True:
        raise ComplianceError(f"invalid completion marker schema/status: {marker_path}")
    rows = marker.get("files")
    if not isinstance(rows, list):
        raise ComplianceError(f"completion marker files must be a list: {marker_path}")
    bound: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ComplianceError(f"invalid completion file entry in {marker_path}")
        relative, expected_hash = row.get("relative_path"), row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ComplianceError(f"invalid completion file entry in {marker_path}")
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError as exc:
            raise ComplianceError(f"completion entry escapes run directory: {relative}") from exc
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ComplianceError(f"completion hash mismatch for {relative} in {marker_path}")
        if relative in bound:
            raise ComplianceError(f"duplicate completion entry {relative!r}")
        bound[relative] = expected_hash
    for item in required_paths:
        path = Path(item).resolve()
        try:
            relative = path.relative_to(run_dir).as_posix()
        except ValueError as exc:
            raise ComplianceError(f"required completion path is outside run: {path}") from exc
        if bound.get(relative) != file_sha256(path):
            raise ComplianceError(f"required file is not completion-bound: {relative}")
    return marker


def artifact_run_dir(path: str | Path) -> Path:
    """Resolve an experiment root from a file in predictions/ or compiled/."""

    path = Path(path)
    return path.parent.parent if path.parent.name in {"predictions", "compiled"} else path.parent


def load_artifact_lineage(
    path: str | Path,
    *,
    expected_task: int,
    requested_track: str,
    expected_architecture: str | None = None,
    expected_setting: str | None = None,
    expected_model: str | None = None,
    expected_output_kind: str | None = None,
    require_adopted: bool = False,
    kind: str,
) -> dict[str, object]:
    """Validate a model-generated upstream file and return hash-bound lineage.

    Closed runs require a sibling experiment ``provenance.json`` and re-check
    its model against the live registry rather than trusting self-asserted JSON
    flags.  Open runs may consume a legacy unmanifested artifact, but it is
    explicitly recorded as unverified rather than disappearing from lineage.
    """

    # The generative-model registry imports DSPy.  Keep that dependency local
    # to the lineage validator so generic hashing/atomic-write helpers remain
    # usable by the non-DSPy encoder pipeline.
    from .models import SPECS

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ComplianceError(f"{kind} artifact does not exist: {artifact_path}")
    run_dir = artifact_run_dir(artifact_path)
    manifest_path = run_dir / "provenance.json"
    base = {
        "kind": kind,
        "path": str(artifact_path),
        "sha256": file_sha256(artifact_path),
    }
    if not manifest_path.exists():
        if requested_track == "closed":
            raise ComplianceError(
                f"closed track refuses {kind} without provenance sidecar: {manifest_path}"
            )
        return base | {"verified": False, "manifest": None}

    if requested_track == "closed":
        validate_completion_marker(
            run_dir,
            required_paths=(artifact_path, manifest_path),
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComplianceError(f"invalid provenance sidecar {manifest_path}: {exc}") from exc
    if expected_task not in manifest.get("tasks", []):
        raise ComplianceError(
            f"{kind} provenance tasks {manifest.get('tasks')!r} do not include {expected_task}"
        )
    try:
        # as_posix() to match how output_record/write_completion_marker
        # serialize relative paths; str() would use backslashes on Windows.
        relative_path = artifact_path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ComplianceError(f"{kind} is not contained by its claimed run directory") from exc
    matching_outputs = [
        output
        for output in manifest.get("outputs", [])
        if output.get("relative_path") == relative_path
        and output.get("sha256") == base["sha256"]
        and (
            expected_output_kind is None
            or output.get("kind") == expected_output_kind
        )
    ]
    if not matching_outputs and requested_track == "closed":
        raise ComplianceError(
            f"closed {kind} is not hash-bound in {manifest_path}; expected output "
            f"entry for {relative_path!r} with SHA-256 {base['sha256']}"
        )
    if expected_architecture is not None and manifest.get("architecture_version") != expected_architecture:
        raise ComplianceError(
            f"{kind} architecture {manifest.get('architecture_version')!r} != "
            f"{expected_architecture!r}"
        )
    if expected_setting is not None and manifest.get("setting") != expected_setting:
        raise ComplianceError(
            f"{kind} setting {manifest.get('setting')!r} != {expected_setting!r}"
        )
    if expected_model is not None and (manifest.get("task_model") or {}).get("key") != expected_model:
        raise ComplianceError(
            f"{kind} task model {(manifest.get('task_model') or {}).get('key')!r} "
            f"!= {expected_model!r}"
        )
    if require_adopted and not (manifest.get("adoption") or {}).get("adopt", False):
        raise ComplianceError(f"{kind} was not accepted by its recorded adoption gate")
    if requested_track == "closed":
        if manifest.get("track") != "closed":
            raise ComplianceError(
                f"closed run cannot consume {kind} produced for track {manifest.get('track')!r}"
            )
        model_info = manifest.get("task_model") or {}
        key = model_info.get("key")
        spec = SPECS.get(key)
        if (
            spec is None
            or not spec.closed_track
            or spec.license_status != "verified-open"
            or not 0 < spec.params_b <= 70
        ):
            raise ComplianceError(
                f"closed run rejects {kind} model lineage {key!r}; expected a registered "
                "open-weight checkpoint with 0 < params_b <= 70"
            )
        if model_info.get("endpoint") != spec.litellm_id or float(
            model_info.get("params_b", -1)
        ) != float(spec.params_b) or model_info.get("license_status") != spec.license_status:
            raise ComplianceError(f"{kind} manifest model metadata does not match SPECS[{key!r}]")
        forbidden_sources = sorted(
            set(manifest.get("data_sources", [])) - ORGANIZER_DATA_SOURCE_ALLOWLIST
        )
        if forbidden_sources:
            raise ComplianceError(
                f"closed {kind} lineage contains non-organizer sources: {forbidden_sources}"
            )
        for role, auxiliary in (manifest.get("auxiliary_models") or {}).items():
            if auxiliary is None:
                continue
            auxiliary_spec = SPECS.get(auxiliary.get("key"))
            if (
                auxiliary_spec is None
                or not auxiliary_spec.closed_track
                or auxiliary_spec.license_status != "verified-open"
                or not 0 < auxiliary_spec.params_b <= 70
            ):
                raise ComplianceError(
                    f"closed {kind} lineage has ineligible {role} model "
                    f"{auxiliary.get('key')!r}"
                )
            expected_auxiliary = {
                "key": auxiliary_spec.key,
                "endpoint": auxiliary_spec.litellm_id,
                "params_b": float(auxiliary_spec.params_b),
                "closed_track": bool(auxiliary_spec.closed_track),
                "license_status": auxiliary_spec.license_status,
            }
            if auxiliary != expected_auxiliary:
                raise ComplianceError(
                    f"closed {kind} {role} metadata does not match "
                    f"SPECS[{auxiliary_spec.key!r}]"
                )

    return base | {
        "verified": True,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_data": manifest,
        "output_bound": bool(matching_outputs),
        "producer": {
            "architecture_version": manifest.get("architecture_version"),
            "track": manifest.get("track"),
            "setting": manifest.get("setting"),
            "task_model": manifest.get("task_model"),
            "optimizer": manifest.get("optimizer"),
            "data_sources": manifest.get("data_sources"),
        },
    }


__all__ = [
    "artifact_run_dir",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_json_sha256",
    "create_experiment_dir",
    "file_sha256",
    "load_artifact_lineage",
    "output_record",
    "validate_completion_marker",
    "write_completion_marker",
]
