import json

import pytest

from daleel.artifacts import (
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    load_artifact_lineage,
    output_record,
    validate_completion_marker,
    write_completion_marker,
)
from daleel.models import SPECS
from daleel.provenance import ComplianceError, build_provenance_manifest


def _artifact(tmp_path, model="gemma-4-31b-paid", track="closed"):
    run = tmp_path / "run"
    predictions = run / "predictions"
    predictions.mkdir(parents=True)
    path = predictions / "preds.jsonl"
    path.write_text('{"paragraph_id":1}\n', encoding="utf-8")
    manifest = build_provenance_manifest(
        tasks=2,
        track=track,
        setting="both",
        task_model=SPECS[model],
        optimizer=None,
        training_ids=[],
        data_sources=["daleel2026:train-task-2"],
        architecture_version="stage0-quote-v1",
    )
    manifest["outputs"] = [
        {
            "kind": "quote-proposals",
            "relative_path": "predictions/preds.jsonl",
            "sha256": file_sha256(path),
        }
    ]
    manifest_path = run / "provenance.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    write_completion_marker(run, [path, manifest_path])
    return path


def test_hash_and_closed_lineage(tmp_path):
    path = _artifact(tmp_path)
    lineage = load_artifact_lineage(
        path,
        expected_task=2,
        requested_track="closed",
        expected_architecture="stage0-quote-v1",
        kind="quote-proposals",
    )
    assert lineage["verified"] is True
    assert lineage["sha256"] == file_sha256(path)
    assert len(lineage["manifest_sha256"]) == 64

    with pytest.raises(ComplianceError, match="not hash-bound"):
        load_artifact_lineage(
            path,
            expected_task=2,
            requested_track="closed",
            expected_output_kind="wrong-kind",
            kind="quote-proposals",
        )


def test_closed_refuses_missing_or_wrong_lineage(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ComplianceError, match="without provenance"):
        load_artifact_lineage(
            path, expected_task=2, requested_track="closed", kind="proposals"
        )
    open_lineage = load_artifact_lineage(
        path, expected_task=2, requested_track="open", kind="proposals"
    )
    assert open_lineage["verified"] is False

    manifested = _artifact(tmp_path / "other")
    with pytest.raises(ComplianceError, match="architecture"):
        load_artifact_lineage(
            manifested,
            expected_task=2,
            requested_track="closed",
            expected_architecture="wrong-v0",
            kind="proposals",
        )


def test_closed_refuses_open_proprietary_producer(tmp_path):
    path = _artifact(tmp_path, model="deepseek-v4-pro", track="open")
    with pytest.raises(ComplianceError, match="track"):
        load_artifact_lineage(
            path, expected_task=2, requested_track="closed", kind="proposals"
        )


def test_atomic_experiment_directory_outputs_and_completion_marker(tmp_path):
    first = create_experiment_dir(tmp_path, "v3 smoke", {"fold": 0})
    second = create_experiment_dir(tmp_path, "v3 smoke", {"fold": 0})
    assert first != second
    artifact = atomic_write_json(first / "metrics.json", {"score": 0.7})
    record = output_record(first, artifact, kind="metrics")
    assert record["relative_path"] == "metrics.json"
    assert record["sha256"] == file_sha256(artifact)
    marker = write_completion_marker(first, [artifact], metadata={"fold": 0})
    payload = json.loads(marker.read_text())
    assert payload["complete"] is True
    assert payload["files"][0]["sha256"] == file_sha256(artifact)
    validate_completion_marker(first, required_paths=[artifact])
    artifact.write_text("tampered")
    with pytest.raises(ComplianceError, match="completion hash mismatch"):
        validate_completion_marker(first)
