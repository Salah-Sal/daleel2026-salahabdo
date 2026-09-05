"""CLI guards and a no-network completed-run integration path."""

import json
import sys
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.compile_span_roles as compiler
import scripts.run_decomposed as runner
import scripts.run_zero_shot as stage0
from daleel.artifacts import file_sha256, write_completion_marker
from daleel.io import write_jsonl
from daleel.models import SPECS
from daleel.provenance import build_provenance_manifest


def _forbid_lm(*args, **kwargs):  # pragma: no cover - only called on regression
    pytest.fail("preflight regression: make_lm was called")


def test_default_direct_fusion_fails_before_lm(monkeypatch):
    monkeypatch.setattr(runner, "make_lm", _forbid_lm)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_decomposed.py", "--model", "gemma-4-31b-paid", "--live-proposals"],
    )
    with pytest.raises(SystemExit, match="--direct-task1 is required"):
        runner.main()


def test_arbitrary_input_requires_source_id_before_lm(monkeypatch, tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"paragraph_id":1,"text":"نص","type":"debate"}\n')
    monkeypatch.setattr(runner, "make_lm", _forbid_lm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_decomposed.py",
            "--model",
            "gemma-4-31b-paid",
            "--input",
            str(input_path),
            "--live-proposals",
            "--fusion",
            "spans",
        ],
    )
    with pytest.raises(SystemExit, match="--input requires --source-id"):
        runner.main()


def test_uncompiled_balanced_mode_requires_proposal_shaped_memory_before_lm(monkeypatch):
    monkeypatch.setattr(runner, "make_lm", _forbid_lm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_decomposed.py",
            "--model",
            "gemma-4-31b-paid",
            "--on",
            "val:1",
            "--live-proposals",
            "--fusion",
            "spans",
            "--demo-selector",
            "balanced",
        ],
    )
    with pytest.raises(SystemExit, match="requires --training-proposals"):
        runner.main()


def test_compile_numeric_and_final_gate_preflight_before_lm(monkeypatch):
    monkeypatch.setattr(compiler, "make_lm", _forbid_lm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model",
            "gemma-4-31b-paid",
            "--proposals",
            "missing.jsonl",
            "--threads",
            "0",
        ],
    )
    with pytest.raises(SystemExit, match="--threads must be"):
        compiler.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model",
            "gemma-4-31b-paid",
            "--proposals",
            "missing.jsonl",
            "--final",
        ],
    )
    with pytest.raises(SystemExit, match="--final requires --adoption-report"):
        compiler.main()


def test_compile_pool_all_refused_outside_final_before_lm(monkeypatch):
    monkeypatch.setattr(compiler, "make_lm", _forbid_lm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model",
            "gemma-4-31b-paid",
            "--proposals",
            "missing.jsonl",
            "--pool",
            "all",
        ],
    )
    with pytest.raises(SystemExit, match="frozen 182-paragraph confirmation set"):
        compiler.main()


def test_closed_stage0_rejects_legacy_compiled_state_before_lm(monkeypatch):
    monkeypatch.setattr(stage0, "make_lm", _forbid_lm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_zero_shot.py",
            "--task",
            "1",
            "--model",
            "gemma-4-31b-paid",
            "--track",
            "closed",
            "--on",
            "val:1",
            "--compiled",
            "legacy.json",
        ],
    )
    with pytest.raises(SystemExit, match="closed stage-0 runs refuse --compiled"):
        stage0.main()


def test_decomposed_runner_completes_no_network_artifact(monkeypatch, tmp_path, capsys):
    records, _, _, _ = runner._select_known_records("val:1")
    record = records[0]
    producer = tmp_path / "producer"
    predictions = producer / "predictions"
    predictions.mkdir(parents=True)
    proposal_path = predictions / "preds.jsonl"
    write_jsonl(
        proposal_path,
        [
            {
                **record,
                "labels": [
                    {
                        "label": "AS",
                        "start_offset": 0,
                        "end_offset": len(record["text"]),
                    }
                ],
            }
        ],
    )
    manifest = build_provenance_manifest(
        tasks=2,
        track="closed",
        setting="both",
        task_model=SPECS["gemma-4-31b-paid"],
        optimizer=None,
        training_ids=[],
        data_sources=["daleel2026:train-task-2"],
        architecture_version="stage0-quote-v1",
    )
    manifest["outputs"] = [
        {
            "kind": "quote-proposals",
            "relative_path": "predictions/preds.jsonl",
            "sha256": file_sha256(proposal_path),
        }
    ]
    manifest_path = producer / "provenance.json"
    manifest_path.write_text(json.dumps(manifest))
    write_completion_marker(producer, [proposal_path, manifest_path])

    dummy = DummyLM([{"adu_roles": ["AS"]}] * 100)
    monkeypatch.setattr(runner, "make_lm", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(
        runner.SpanRoleDecision,
        "batch",
        lambda self, examples, **kwargs: (
            [dspy.Prediction(adu_roles=["AS"]) for _ in examples],
            [],
            [],
        ),
    )
    experiment_root = tmp_path / "experiments"
    monkeypatch.setattr(runner, "EXPERIMENTS_DIR", experiment_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_decomposed.py",
            "--model",
            "gemma-4-31b-paid",
            "--on",
            "val:1",
            "--proposals",
            str(proposal_path),
            "--demo-selector",
            "none",
            "--fusion",
            "spans",
            "--threads",
            "1",
        ],
    )
    runner.main()
    capsys.readouterr()
    runs = list(experiment_root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert (run / "COMPLETED.json").is_file()
    assert (run / "provenance.json").is_file()
    assert not (run / "provenance.pending.json").exists()
    assert (run / "predictions" / "task_1.jsonl").is_file()
    assert (run / "predictions" / "task_2.jsonl").is_file()


def test_compiler_completes_hash_bound_bundle_without_network(monkeypatch, tmp_path, capsys):
    records = compiler.load_records(compiler.TRAIN_TASK1)
    producer = tmp_path / "producer"
    predictions = producer / "predictions"
    predictions.mkdir(parents=True)
    proposal_path = predictions / "all.jsonl"
    write_jsonl(
        proposal_path,
        [
            {
                **record,
                "labels": [
                    {
                        "label": "AS",
                        "start_offset": 0,
                        "end_offset": len(record["text"]),
                    }
                ],
            }
            for record in records
        ],
    )
    manifest = build_provenance_manifest(
        tasks=2,
        track="closed",
        setting="both",
        task_model=SPECS["gemma-4-31b-paid"],
        optimizer=None,
        training_ids=[],
        data_sources=["daleel2026:train-task-2"],
        architecture_version="stage0-quote-v1",
    )
    manifest["outputs"] = [
        {
            "kind": "quote-proposals",
            "relative_path": "predictions/all.jsonl",
            "sha256": file_sha256(proposal_path),
        }
    ]
    manifest_path = producer / "provenance.json"
    manifest_path.write_text(json.dumps(manifest))
    write_completion_marker(producer, [proposal_path, manifest_path])

    dummy = DummyLM([{"adu_roles": ["AS"]}])
    monkeypatch.setattr(compiler, "make_lm", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(
        compiler.SpanRoleDecision,
        "batch",
        lambda self, examples, **kwargs: (
            [dspy.Prediction(adu_roles=list(example.adu_roles)) for example in examples],
            [],
            [],
        ),
    )
    experiment_root = tmp_path / "experiments"
    monkeypatch.setattr(compiler, "EXPERIMENTS_DIR", experiment_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model",
            "gemma-4-31b-paid",
            "--setting",
            "editorial",
            "--proposals",
            str(proposal_path),
            "--optimizer",
            "none",
            "--objective",
            "task2",
            "--fold",
            "0",
            "--n-folds",
            "2",
            "--inner-fold",
            "0",
            "--n-inner-folds",
            "2",
            "--max-demos",
            "3",
            "--threads",
            "1",
        ],
    )
    compiler.main()
    capsys.readouterr()
    runs = list(experiment_root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert (run / "COMPLETED.json").is_file()
    assert (run / "compiled" / "role.json").is_file()
    assert (run / "compiled" / "role_bundle.json").is_file()
    assert (run / "compiled" / "role_memory.jsonl").is_file()
    provenance = json.loads((run / "provenance.json").read_text())
    kinds = {output["kind"] for output in provenance["outputs"]}
    assert {"v3-role-state", "v3-role-bundle", "v3-role-memory"} <= kinds
