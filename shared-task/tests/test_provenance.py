"""Focused tests for track compliance and deterministic run provenance."""

import hashlib
import json
from importlib.metadata import version

import pytest

from daleel.models import ModelSpec, SPECS
from daleel.provenance import (
    ComplianceError,
    ORGANIZER_DATA_SOURCE_ALLOWLIST,
    build_provenance_manifest,
    training_ids_sha256,
)


OFFICIAL_T1 = "daleel2026:train-task-1"
OFFICIAL_T2 = "daleel2026:train-task-2"


def _manifest(**overrides):
    kwargs = {
        "tasks": (2, 1),
        "track": "closed",
        "setting": "both",
        "task_model": SPECS["gemma-4-31b-paid"],
        "optimizer": "gepa",
        "training_ids": [964, 7, "arabic-id-أ"],
        "data_sources": [OFFICIAL_T1, OFFICIAL_T2],
        "architecture_version": "decomposed-v1",
        "teacher_model": SPECS["gemma-3-27b"],
        "reflection_model": SPECS["gemma-4-31b-paid"],
        "prompt_model": SPECS["gemma-3-12b"],
    }
    kwargs.update(overrides)
    return build_provenance_manifest(**kwargs)


def test_closed_manifest_is_complete_deterministic_and_json_serializable():
    manifest = _manifest()
    assert manifest["tasks"] == [1, 2]
    assert manifest["track"] == "closed"
    assert manifest["setting"] == "both"
    assert manifest["optimizer"] == "gepa"
    assert manifest["architecture_version"] == "decomposed-v1"
    assert manifest["dspy_version"] == version("dspy")

    assert manifest["task_model"] == {
        "key": "gemma-4-31b-paid",
        "endpoint": "openrouter/google/gemma-4-31b-it",
        "params_b": 31.0,
        "closed_track": True,
        "license_status": "verified-open",
    }
    assert set(manifest["auxiliary_models"]) == {"teacher", "reflection", "prompt"}
    assert manifest["auxiliary_models"]["teacher"]["params_b"] == 27.0
    assert manifest["training_ids"] == [964, 7, "arabic-id-أ"]
    assert manifest["data_sources"] == [OFFICIAL_T1, OFFICIAL_T2]
    assert len(manifest["code"]["python_tree_sha256"]) == 64
    assert manifest["code"]["git_commit"]

    # Canonical digest covers the exact ordered list and is independently
    # reproducible without importing this module.
    payload = json.dumps(
        manifest["training_ids"],
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert manifest["training_ids_sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(manifest["training_ids_sha256"]) == 64
    assert _manifest() == manifest
    json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True)


def test_training_id_hash_preserves_optimizer_relevant_order():
    assert training_ids_sha256([1, 2, 3]) == training_ids_sha256((1, 2, 3))
    assert training_ids_sha256([1, 2, 3]) != training_ids_sha256([3, 2, 1])


@pytest.mark.parametrize("track", ["CLOSED", "closed-track", "", None])
def test_invalid_track_fails_loudly(track):
    with pytest.raises(ComplianceError, match="invalid track"):
        _manifest(track=track)


@pytest.mark.parametrize("setting", ["cross-domain", "all", "", None])
def test_invalid_training_setting_fails_loudly(setting):
    with pytest.raises(ComplianceError, match="invalid training setting"):
        _manifest(setting=setting)


@pytest.mark.parametrize("tasks", [(), (3,), (1, 1), ([1],), "1", True])
def test_invalid_tasks_fail_loudly(tasks):
    with pytest.raises(ComplianceError):
        _manifest(tasks=tasks)


@pytest.mark.parametrize(
    ("model_kwarg", "model_key", "message"),
    [
        ("teacher_model", "deepseek-v4-pro", "closed track rejects teacher model"),
        (
            "reflection_model",
            "gemini-3-flash-preview",
            "closed track rejects reflection model",
        ),
        ("prompt_model", "qwen3-next-80b", "closed track rejects prompt model"),
        ("task_model", "gpt-oss-120b", "closed track rejects task model"),
    ],
)
def test_closed_track_rejects_deepseek_proprietary_and_over_cap_models(
    model_kwarg, model_key, message
):
    with pytest.raises(ComplianceError, match=message):
        _manifest(**{model_kwarg: SPECS[model_key]})


def test_closed_track_requires_registered_modelspec_entries():
    with pytest.raises(ComplianceError, match="registered ModelSpec"):
        _manifest(task_model="openrouter/google/gemma-4-31b-it")

    forged = ModelSpec(
        key="unregistered-open-model",
        litellm_id="provider/unregistered-open-model",
        params_b=31,
        closed_track=True,
    )
    with pytest.raises(ComplianceError, match="not an exact registered SPECS entry"):
        _manifest(task_model=forged)


def test_closed_track_rejects_registry_entries_with_unverified_licenses():
    with pytest.raises(ComplianceError, match="verified-open license assurance"):
        _manifest(task_model=SPECS["qwen3.5-9b"])


def test_closed_track_uses_an_explicit_organizer_source_allowlist():
    assert {OFFICIAL_T1, OFFICIAL_T2} <= ORGANIZER_DATA_SOURCE_ALLOWLIST
    with pytest.raises(ComplianceError, match="forbids non-organizer data source"):
        _manifest(data_sources=[OFFICIAL_T1, "webis-editorials-16"])


def test_open_track_records_but_allows_external_models_and_data():
    manifest = _manifest(
        track="open",
        task_model=SPECS["gpt-oss-120b"],
        teacher_model=SPECS["deepseek-v4-pro"],
        reflection_model=SPECS["gemini-3-flash-preview"],
        prompt_model=None,
        data_sources=[OFFICIAL_T1, "webis-editorials-16"],
    )
    assert manifest["task_model"]["params_b"] == 120.0
    assert manifest["auxiliary_models"]["teacher"]["key"] == "deepseek-v4-pro"
    assert manifest["auxiliary_models"]["reflection"]["closed_track"] is False
    assert manifest["auxiliary_models"]["reflection"]["license_status"] == "proprietary"
    assert manifest["auxiliary_models"]["prompt"] is None
    assert "webis-editorials-16" in manifest["data_sources"]

    # The open track is not artificially limited to the current ladder.  It
    # still requires a ModelSpec so endpoint and parameter provenance exists.
    future_model = ModelSpec(
        key="future-proprietary-model",
        litellm_id="provider/future-proprietary-model-v1",
        params_b=0,
        closed_track=False,
    )
    future_manifest = _manifest(track="open", task_model=future_model)
    assert future_manifest["task_model"]["endpoint"] == future_model.litellm_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_ids", {1, 2}, "must be ordered"),
        ("training_ids", [1, 1], "duplicate"),
        ("training_ids", [1.5], "JSON string or integer"),
        ("data_sources", {OFFICIAL_T1}, "must be ordered"),
        ("data_sources", [OFFICIAL_T1, OFFICIAL_T1], "duplicate"),
        ("architecture_version", "", "non-empty string"),
        ("optimizer", "", "None or a non-empty string"),
    ],
)
def test_manifest_rejects_ambiguous_or_incomplete_provenance(field, value, message):
    with pytest.raises(ComplianceError, match=message):
        _manifest(**{field: value})


def test_training_ids_require_a_declared_source():
    with pytest.raises(ComplianceError, match="data_sources is empty"):
        _manifest(data_sources=[])
