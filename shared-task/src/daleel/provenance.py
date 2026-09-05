"""Compliance checks and reproducibility manifests for Daleel runs.

The closed track permits only organizer-provided data and open-weight language
models with at most 70 billion *total* parameters.  Those restrictions apply
to every model that can influence the resulting program, including models used
only while compiling prompts or demonstrations.  This module therefore checks
the task LM and the DSPy teacher, reflection, and prompt-proposal LMs together.

``build_provenance_manifest`` returns only JSON-native values.  It deliberately
does not add a timestamp: identical run inputs should yield identical manifest
content.  Training IDs retain their caller-supplied order because that order can
affect a DSPy compile; the SHA-256 digest covers the canonical UTF-8 JSON form
of that exact ordered sequence.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .models import ModelSpec


class ComplianceError(ValueError):
    """A run configuration cannot be represented as track-compliant."""


VALID_TRACKS = frozenset({"closed", "open"})
VALID_TRAINING_SETTINGS = frozenset({"editorial", "debate", "both"})
VALID_TASKS = frozenset({1, 2})

# Stable source identifiers, rather than machine-specific absolute paths.  The
# fourth entry reserves the obvious identifier for the final organizer input
# when it is released; it does not claim that the file is currently present.
ORGANIZER_DATA_SOURCE_ALLOWLIST = frozenset(
    {
        "daleel2026:train-task-1",
        "daleel2026:train-task-2",
        "daleel2026:dev-input",
        "daleel2026:test-input",
    }
)

_AUXILIARY_MODEL_ROLES = ("teacher", "reflection", "prompt")


def _code_state() -> dict[str, object]:
    """Bind manifests to the executable Python tree and Git revision."""

    shared_task_root = Path(__file__).resolve().parents[2]
    repo_root = shared_task_root.parent
    digest = hashlib.sha256()
    paths = sorted(
        list((shared_task_root / "src" / "daleel").glob("*.py"))
        + list((shared_task_root / "scripts").glob("*.py"))
    )
    for path in paths:
        digest.update(path.relative_to(shared_task_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    commit = None
    dirty = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "shared-task"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - exported archive
        pass
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python_tree_sha256": digest.hexdigest(),
    }


def _validate_choice(value: str, *, name: str, choices: frozenset[str]) -> str:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ComplianceError(f"invalid {name} {value!r}; expected one of: {allowed}")
    return value


def _normalise_tasks(tasks: int | Iterable[int]) -> list[int]:
    if isinstance(tasks, bool):
        raise ComplianceError("tasks must contain task numbers 1 and/or 2, not bool")
    if isinstance(tasks, int):
        values = [tasks]
    else:
        if isinstance(tasks, (str, bytes)):
            raise ComplianceError("tasks must be an integer or iterable of integers")
        try:
            values = list(tasks)
        except TypeError as exc:
            raise ComplianceError(
                "tasks must be an integer or iterable of integers"
            ) from exc

    if not values:
        raise ComplianceError("at least one task is required")
    invalid = [
        task
        for task in values
        if isinstance(task, bool)
        or not isinstance(task, int)
        or task not in VALID_TASKS
    ]
    if invalid:
        raise ComplianceError(f"invalid Daleel task number(s): {invalid!r}; use 1 and/or 2")
    if len(values) != len(set(values)):
        raise ComplianceError(f"duplicate task number(s) are not allowed: {values!r}")
    return sorted(values)


def _normalise_training_ids(training_ids: Iterable[int | str]) -> list[int | str]:
    if isinstance(training_ids, (str, bytes)):
        raise ComplianceError("training_ids must be an iterable of paragraph IDs")
    if isinstance(training_ids, (set, frozenset)):
        raise ComplianceError(
            "training_ids must be ordered (for example, a list or tuple), not a set"
        )
    try:
        values = list(training_ids)
    except TypeError as exc:
        raise ComplianceError("training_ids must be an iterable of paragraph IDs") from exc

    for paragraph_id in values:
        if isinstance(paragraph_id, bool) or not isinstance(paragraph_id, (int, str)):
            raise ComplianceError(
                "each training ID must be a JSON string or integer; "
                f"got {paragraph_id!r}"
            )
        if isinstance(paragraph_id, str) and not paragraph_id:
            raise ComplianceError("training IDs may not be empty strings")
    if len(values) != len(set(values)):
        raise ComplianceError("training_ids contains duplicate paragraph IDs")
    return values


def training_ids_sha256(training_ids: Iterable[int | str]) -> str:
    """Return SHA-256 for the exact ordered ID sequence using canonical JSON."""

    values = _normalise_training_ids(training_ids)
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_data_sources(data_sources: Iterable[str], *, track: str) -> list[str]:
    if isinstance(data_sources, (str, bytes)):
        raise ComplianceError("data_sources must be an iterable of source identifiers")
    if isinstance(data_sources, (set, frozenset)):
        raise ComplianceError(
            "data_sources must be ordered (for example, a list or tuple), not a set"
        )
    try:
        values = list(data_sources)
    except TypeError as exc:
        raise ComplianceError(
            "data_sources must be an iterable of source identifiers"
        ) from exc

    if any(not isinstance(source, str) or not source.strip() for source in values):
        raise ComplianceError("every data source must be a non-empty string identifier")
    if len(values) != len(set(values)):
        raise ComplianceError("data_sources contains duplicate identifiers")

    if track == "closed":
        forbidden = sorted(set(values) - ORGANIZER_DATA_SOURCE_ALLOWLIST)
        if forbidden:
            allowed = ", ".join(sorted(ORGANIZER_DATA_SOURCE_ALLOWLIST))
            raise ComplianceError(
                "closed track forbids non-organizer data source(s) "
                f"{forbidden!r}; allowed sources: {allowed}"
            )
    return values


def _validate_model(spec: ModelSpec, *, role: str, track: str) -> ModelSpec:
    # Importing daleel.models configures DSPy.  Provenance's generic helpers
    # (notably training_ids_sha256 and ComplianceError) are also used by the
    # non-DSPy encoder track, so load the generative registry only here.
    from .models import SPECS, ModelSpec

    if not isinstance(spec, ModelSpec):
        requirement = "registered ModelSpec" if track == "closed" else "ModelSpec"
        raise ComplianceError(
            f"{role} model must be a {requirement}, got "
            f"{type(spec).__name__}"
        )

    if track == "closed":
        registered = SPECS.get(spec.key)
        if registered != spec:
            raise ComplianceError(
                f"{role} model {spec.key!r} is not an exact registered SPECS entry"
            )

        failures: list[str] = []
        if not spec.closed_track:
            failures.append("is not marked closed_track")
        if spec.license_status != "verified-open":
            failures.append(
                f"does not have verified-open license assurance ({spec.license_status!r})"
            )
        if spec.params_b <= 0:
            failures.append("does not declare a positive total parameter count")
        if spec.params_b > 70:
            failures.append(f"has {spec.params_b:g}B parameters, over the 70B cap")
        if failures:
            detail = "; ".join(failures)
            raise ComplianceError(
                f"closed track rejects {role} model {spec.key!r}: {detail}. "
                "Use a registered open-weight model with 0 < params_b <= 70, "
                "or declare the run as open track."
            )
    return spec


def _serialise_model(spec: ModelSpec) -> dict[str, object]:
    return {
        "key": spec.key,
        "endpoint": spec.litellm_id,
        "params_b": float(spec.params_b),
        "closed_track": bool(spec.closed_track),
        "license_status": spec.license_status,
    }


def _installed_dspy_version() -> str:
    try:
        installed = version("dspy")
    except PackageNotFoundError as exc:
        raise ComplianceError(
            "cannot create provenance manifest because the dspy distribution "
            "version is unavailable"
        ) from exc
    if not installed:
        raise ComplianceError("installed dspy version is empty")
    return installed


def build_provenance_manifest(
    *,
    tasks: int | Iterable[int],
    track: str,
    setting: str,
    task_model: ModelSpec,
    optimizer: str | None,
    training_ids: Iterable[int | str],
    data_sources: Iterable[str],
    architecture_version: str,
    teacher_model: ModelSpec | None = None,
    reflection_model: ModelSpec | None = None,
    prompt_model: ModelSpec | None = None,
) -> dict[str, object]:
    """Validate a Daleel run and build its deterministic JSON manifest.

    In a closed run, every supplied model role must name an exact entry in
    :data:`daleel.models.SPECS`, be marked ``closed_track``, and declare a total
    size in ``(0, 70]`` billion parameters.  Open runs still require structured
    ``ModelSpec`` objects so endpoints and sizes cannot disappear from the
    provenance record, but those specs may be unregistered, proprietary, or
    over-cap entries.
    """

    track = _validate_choice(track, name="track", choices=VALID_TRACKS)
    setting = _validate_choice(
        setting,
        name="training setting",
        choices=VALID_TRAINING_SETTINGS,
    )
    task_values = _normalise_tasks(tasks)
    id_values = _normalise_training_ids(training_ids)
    source_values = _normalise_data_sources(data_sources, track=track)

    if id_values and not source_values:
        raise ComplianceError("training IDs were supplied but data_sources is empty")
    if not isinstance(architecture_version, str) or not architecture_version.strip():
        raise ComplianceError("architecture_version must be a non-empty string")
    if optimizer is not None and (
        not isinstance(optimizer, str) or not optimizer.strip()
    ):
        raise ComplianceError("optimizer must be None or a non-empty string")

    task_spec = _validate_model(task_model, role="task", track=track)
    auxiliary = {
        "teacher": teacher_model,
        "reflection": reflection_model,
        "prompt": prompt_model,
    }
    serialised_auxiliary: dict[str, dict[str, object] | None] = {}
    for role in _AUXILIARY_MODEL_ROLES:
        spec = auxiliary[role]
        serialised_auxiliary[role] = (
            _serialise_model(_validate_model(spec, role=role, track=track))
            if spec is not None
            else None
        )

    manifest: dict[str, object] = {
        "manifest_schema_version": 1,
        "architecture_version": architecture_version,
        "dspy_version": _installed_dspy_version(),
        "tasks": task_values,
        "track": track,
        "setting": setting,
        "optimizer": optimizer,
        "task_model": _serialise_model(task_spec),
        "auxiliary_models": serialised_auxiliary,
        "training_ids": id_values,
        "training_ids_sha256": training_ids_sha256(id_values),
        "data_sources": source_values,
        "code": _code_state(),
    }

    # A defensive assertion at the API boundary: future fields must remain
    # JSON-native.  ``allow_nan=False`` also rejects non-standard float values.
    try:
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - future-proofing
        raise ComplianceError(f"provenance manifest is not JSON-serializable: {exc}") from exc
    return manifest


# Short alias for callers that already sit inside a provenance-named module.
build_manifest = build_provenance_manifest


__all__ = [
    "ComplianceError",
    "ORGANIZER_DATA_SOURCE_ALLOWLIST",
    "VALID_TRACKS",
    "VALID_TRAINING_SETTINGS",
    "build_manifest",
    "build_provenance_manifest",
    "training_ids_sha256",
]
