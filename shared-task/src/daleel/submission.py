"""Validation and packaging of Codabench submissions.

Rules encoded here (official Codabench Task 1 page + organizer answers to
participants, 2026-06-17):
- archive named `<team_name>_<training_setting>.zip`, setting one of
  editorial | debate | both;
- the archive contains the task file (task_1.jsonl / task_2.jsonl) at its
  root;
- each line echoes the original paragraph_id and text together with the
  predicted labels ("Each line should include the original paragraph_id and
  text, together with the predicted label or labels" — organizers);
- `type` is also required here because the official scorers filter
  predictions by it for the per-genre leaderboard columns — a file without
  it scores 0 rows on the Editorials/Debates columns;
- only the LAST submission per setting before the phase deadline counts.
"""

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

from .constants import LABELS, TASK1_FILENAME, TASK2_FILENAME, TRAINING_SETTINGS
from .io import read_jsonl

TASK_FILENAMES = {"task_1": TASK1_FILENAME, "task_2": TASK2_FILENAME}


def _is_plain_int(value: object) -> bool:
    """True for JSON integer IDs/offsets, excluding Python booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_common(record: dict, where: str, seen_ids: set) -> list[str]:
    errors = []
    if "paragraph_id" not in record:
        errors.append(f"{where}: missing 'paragraph_id'")
    else:
        pid = record["paragraph_id"]
        if not _is_plain_int(pid):
            errors.append(f"{where}: 'paragraph_id' must be an integer, not {pid!r}")
        else:
            if pid in seen_ids:
                errors.append(f"{where}: duplicate paragraph_id {pid!r}")
            seen_ids.add(pid)
    if not record.get("text"):
        errors.append(f"{where}: missing 'text' (organizers require the original text)")
    if record.get("type") not in ("editorial", "debate"):
        errors.append(
            f"{where}: 'type' must be 'editorial' or 'debate' "
            "(the scorer filters on it for per-genre columns)"
        )
    return errors


def validate_task1_records(records: list[dict]) -> list[str]:
    """Return a list of human-readable format errors (empty = valid).

    An empty labels list is valid — 60/612 official training paragraphs
    carry no ADU label.
    """
    errors = []
    seen_ids = set()
    if not records:
        errors.append("no records found")
    for n, record in enumerate(records, start=1):
        where = f"record {n}"
        errors += _validate_common(record, where, seen_ids)
        labels = record.get("labels")
        if not isinstance(labels, list):
            errors.append(f"{where}: 'labels' missing or not a list")
            continue
        bad = [x for x in labels if x not in LABELS]
        if bad:
            errors.append(f"{where}: unknown labels {bad!r} (allowed: {', '.join(LABELS)})")
        if len(set(labels)) != len(labels):
            errors.append(f"{where}: duplicate labels {labels!r}")
    return errors


def validate_task2_records(records: list[dict]) -> list[str]:
    """Return format errors for Task 2 span predictions (empty = valid).

    Schema per official train_task_2.jsonl: labels is a list of
    {label, start_offset, end_offset} objects; offsets are character
    positions into `text` with end exclusive. Overlapping spans are not an
    error (7 official training paragraphs have them).
    """
    errors = []
    seen_ids = set()
    if not records:
        errors.append("no records found")
    for n, record in enumerate(records, start=1):
        where = f"record {n}"
        errors += _validate_common(record, where, seen_ids)
        labels = record.get("labels")
        if not isinstance(labels, list):
            errors.append(f"{where}: 'labels' missing or not a list")
            continue
        seen_spans = set()
        for k, span in enumerate(labels):
            here = f"{where}, span {k + 1}"
            if not isinstance(span, dict) or not {"label", "start_offset", "end_offset"} <= set(
                span
            ):
                errors.append(f"{here}: needs keys label/start_offset/end_offset")
                continue
            if span["label"] not in LABELS:
                errors.append(f"{here}: unknown label {span['label']!r}")
            start, end = span["start_offset"], span["end_offset"]
            # Overlap is legal, including identical boundaries carrying
            # different labels.  Only an exact repeated labelled span is a
            # duplicate prediction.  Restrict this check to hashable values;
            # malformed values are reported by the schema checks below.
            if (
                isinstance(span["label"], str)
                and _is_plain_int(start)
                and _is_plain_int(end)
            ):
                span_key = (span["label"], start, end)
                if span_key in seen_spans:
                    errors.append(
                        f"{here}: duplicate same-label span "
                        f"{span['label']!r} [{start}, {end})"
                    )
                seen_spans.add(span_key)
            if not (_is_plain_int(start) and _is_plain_int(end) and 0 <= start < end):
                errors.append(f"{here}: bad offsets [{start!r}, {end!r})")
            elif isinstance(record.get("text"), str) and end > len(record["text"]):
                errors.append(f"{here}: end_offset {end} beyond text length {len(record['text'])}")
    return errors


def validate_records_against_source(
    records: list[dict],
    source_records: list[dict],
    task: str,
) -> list[str]:
    """Return submission errors after binding predictions to their input.

    ``validate_task1_records`` and ``validate_task2_records`` establish that
    predictions have the right per-record schema.  This source-bound
    preflight additionally proves that the submission is for *exactly* the
    supplied input file: record counts and unique paragraph ID sets match,
    neither side contains duplicate IDs, and each prediction echoes the
    source ``text`` and ``type`` byte-for-byte/value-for-value.

    Args:
        records: Task 1 or Task 2 prediction records.
        source_records: Authoritative input records (normally ``dev_in`` or
            the released test input), without prediction labels.
        task: ``"task_1"`` or ``"task_2"``.

    Returns:
        Human-readable errors, or an empty list when the submission passes.

    Raises:
        ValueError: If ``task`` is not a supported task name.  This denotes
            a caller error rather than invalid submission contents.
    """
    validators = {
        "task_1": validate_task1_records,
        "task_2": validate_task2_records,
    }
    if task not in validators:
        raise ValueError(f"task must be one of {tuple(validators)}, got {task!r}")

    # Preserve all existing task-specific schema checks as the first layer.
    errors = validators[task](records)

    if len(records) != len(source_records):
        errors.append(
            "record count mismatch: "
            f"predictions contain {len(records)} records but source contains {len(source_records)}"
        )

    source_by_id = {}
    for n, source in enumerate(source_records, start=1):
        where = f"source record {n}"
        if "paragraph_id" not in source:
            errors.append(f"{where}: missing 'paragraph_id'")
            continue
        pid = source["paragraph_id"]
        if not _is_plain_int(pid):
            errors.append(f"{where}: 'paragraph_id' must be an integer, not {pid!r}")
            continue
        if pid in source_by_id:
            errors.append(f"{where}: duplicate paragraph_id {pid!r} in source")
        else:
            source_by_id[pid] = source
        if "text" not in source:
            errors.append(f"{where}: missing 'text'")
        if "type" not in source:
            errors.append(f"{where}: missing 'type'")

    prediction_ids = {
        record["paragraph_id"]
        for record in records
        if _is_plain_int(record.get("paragraph_id"))
    }
    missing_ids = [pid for pid in source_by_id if pid not in prediction_ids]
    extra_ids = [pid for pid in prediction_ids if pid not in source_by_id]
    if missing_ids:
        errors.append(f"missing paragraph_id values from source: {missing_ids!r}")
    if extra_ids:
        errors.append(f"unexpected paragraph_id values not in source: {extra_ids!r}")

    # Check every matching prediction rather than only the first one.  That
    # keeps echo mismatches visible even when a duplicate prediction ID has
    # already been diagnosed by the schema validator.
    for n, record in enumerate(records, start=1):
        pid = record.get("paragraph_id")
        if pid not in source_by_id:
            continue
        source = source_by_id[pid]
        if record.get("text") != source.get("text"):
            errors.append(
                f"record {n}: text does not exactly match source for paragraph_id {pid!r}"
            )
        if record.get("type") != source.get("type"):
            errors.append(
                f"record {n}: type {record.get('type')!r} does not exactly match "
                f"source {source.get('type')!r} for paragraph_id {pid!r}"
            )

    return errors


def validate_source_records(source_records: list[dict]) -> list[str]:
    """Validate an organizer input before it can drive inference/packaging."""

    errors: list[str] = []
    seen_ids: set[int] = set()
    if not source_records:
        return ["source contains no records"]
    for n, source in enumerate(source_records, start=1):
        where = f"source record {n}"
        if not isinstance(source, dict):
            errors.append(f"{where}: expected an object")
            continue
        errors.extend(_validate_common(source, where, seen_ids))
        if not isinstance(source.get("text"), str):
            errors.append(f"{where}: 'text' must be a string")
    return errors


def package_submission(
    jsonl_path: str | Path,
    team_name: str,
    training_setting: str,
    out_dir: str | Path,
    task: str = "task_1",
    *,
    source_path: str | Path,
) -> Path:
    """Validate and zip a predictions file into a submission archive.

    Returns the path of the written zip. Raises ValueError on any rule
    violation rather than producing a rejectable archive.
    """
    jsonl_path = Path(jsonl_path)
    if training_setting not in TRAINING_SETTINGS:
        raise ValueError(
            f"training_setting must be one of {TRAINING_SETTINGS}, got {training_setting!r}"
        )
    if task not in TASK_FILENAMES:
        raise ValueError(f"task must be one of {tuple(TASK_FILENAMES)}, got {task!r}")
    if not team_name or any(c in team_name for c in "/\\ "):
        raise ValueError(f"team_name must be non-empty with no spaces or slashes: {team_name!r}")
    if "_" in team_name:
        raise ValueError(
            "team_name should not contain '_' — the archive name "
            "<team_name>_<training_setting>.zip would become ambiguous"
        )

    source_path = Path(source_path)
    if not source_path.is_file():
        raise ValueError(f"source input does not exist: {source_path}")
    records = read_jsonl(jsonl_path)
    source_records = read_jsonl(source_path)
    source_errors = validate_source_records(source_records)
    errors = source_errors + validate_records_against_source(records, source_records, task)
    if errors:
        listing = "\n  - ".join(errors[:20])
        more = f"\n  … and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"invalid {jsonl_path}:\n  - {listing}{more}")

    # Task-specific directories preserve the organizer-mandated archive
    # basename without allowing Task 1 and Task 2 packages to overwrite one
    # another locally.
    out_dir = Path(out_dir) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{team_name}_{training_setting}.zip"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out_dir,
            prefix=f".{zip_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(jsonl_path, arcname=TASK_FILENAMES[task])
        os.replace(temporary, zip_path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

    receipt = {
        "receipt_schema_version": 1,
        "team": team_name,
        "training_setting": training_setting,
        "task": task,
        "record_count": len(records),
        "predictions": {"path": str(jsonl_path), "sha256": _sha256(jsonl_path)},
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "archive": {"path": str(zip_path), "sha256": _sha256(zip_path)},
    }
    receipt_path = zip_path.with_suffix(".receipt.json")
    receipt_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=receipt_path.parent,
            prefix=f".{receipt_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            receipt_temporary = Path(handle.name)
            handle.write(
                json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(receipt_temporary, receipt_path)
    except Exception:
        if receipt_temporary is not None:
            receipt_temporary.unlink(missing_ok=True)
        raise
    return zip_path
