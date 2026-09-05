"""No-network integration tests for forest freezing and structural OOF runs."""

import json
import sys
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.compile_span_roles as compiler
import scripts.freeze_discourse_forests as freezer
from daleel.artifacts import file_sha256, write_completion_marker
from daleel.io import write_jsonl
from daleel.models import SPECS
from daleel.provenance import build_provenance_manifest


def _proposal_artifact(tmp_path):
    records = compiler.load_records(compiler.TRAIN_TASK1)
    producer = tmp_path / "proposal-producer"
    prediction_dir = producer / "predictions"
    prediction_dir.mkdir(parents=True)
    path = prediction_dir / "all.jsonl"
    write_jsonl(
        path,
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
            "sha256": file_sha256(path),
        }
    ]
    provenance = producer / "provenance.json"
    provenance.write_text(json.dumps(manifest), encoding="utf-8")
    write_completion_marker(producer, [path, provenance])
    return path


def _freeze(monkeypatch, tmp_path, proposal_path):
    dummy = DummyLM([{"edges": []}])
    monkeypatch.setattr(freezer, "make_lm", lambda *args, **kwargs: dummy)
    monkeypatch.setattr(
        freezer.DiscourseForestParser,
        "batch",
        lambda self, examples, **kwargs: (
            [dspy.Prediction(edges=[]) for _ in examples], [], []
        ),
    )
    root = tmp_path / "forest-experiments"
    monkeypatch.setattr(freezer, "EXPERIMENTS_DIR", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_discourse_forests.py",
            "--model", "gemma-4-31b-paid",
            "--track", "closed",
            "--setting", "both",
            "--pool", "legacy-train",
            "--proposals", str(proposal_path),
            "--threads", "1",
        ],
    )
    freezer.main()
    runs = list(root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert (run / "COMPLETED.json").is_file()
    artifact = run / "predictions" / "forests.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 430
    assert payload["telemetry"]["parser_errors"] == 0
    return artifact


def test_freezer_and_structural_compiler_complete_without_network(
    monkeypatch, tmp_path, capsys
):
    proposal_path = _proposal_artifact(tmp_path)
    forest_path = _freeze(monkeypatch, tmp_path, proposal_path)
    capsys.readouterr()

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
    root = tmp_path / "role-experiments"
    monkeypatch.setattr(compiler, "EXPERIMENTS_DIR", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model", "gemma-4-31b-paid",
            "--setting", "both",
            "--proposals", str(proposal_path),
            "--structure-mode", "forest",
            "--forests", str(forest_path),
            "--optimizer", "none",
            "--objective", "task2",
            "--fold", "0",
            "--n-folds", "2",
            "--inner-fold", "0",
            "--n-inner-folds", "2",
            "--max-demos", "3",
            "--threads", "1",
        ],
    )
    compiler.main()
    capsys.readouterr()
    runs = list(root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    assert (run / "COMPLETED.json").is_file()
    assert (run / "compiled" / "role.json").is_file()
    assert not (run / "compiled" / "role_bundle.json").exists()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["architecture"] == "proposal-atom-role-v4-structure"
    assert config["structure_mode"] == "forest"
    assert config["forest_sha256"] == file_sha256(forest_path)


def test_structural_modes_require_exact_forest_arguments_before_lm(monkeypatch):
    monkeypatch.setattr(
        compiler,
        "make_lm",
        lambda *args, **kwargs: pytest.fail("make_lm called before argument guard"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model", "gemma-4-31b-paid",
            "--proposals", "missing.jsonl",
            "--structure-mode", "forest",
        ],
    )
    with pytest.raises(SystemExit, match="require --forests"):
        compiler.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_span_roles.py",
            "--model", "gemma-4-31b-paid",
            "--proposals", "missing.jsonl",
            "--forests", "unused.json",
        ],
    )
    with pytest.raises(SystemExit, match="unused"):
        compiler.main()
