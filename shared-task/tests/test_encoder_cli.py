"""Fast CLI/protocol tests for scripts/train_encoder_baseline.py."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.train_encoder_baseline as runner
from daleel.constants import LABELS
from daleel.encoder_baseline import ParagraphExample, SegmentExample
from daleel.folds import Fold


def test_runner_import_is_dspy_free_in_fresh_interpreter(tmp_path):
    del tmp_path  # pytest fixture keeps the test signature consistent.
    # The current process may have imported DSPy through unrelated tests, so
    # use a fresh interpreter to assert this runner's actual import closure.
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.train_encoder_baseline; "
            "raise SystemExit(int('dspy' in sys.modules))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_closed_track_accepts_pinned_camelbert_and_rejects_unverified_arabert():
    camel_args = runner.parse_args(["--task", "1", "--model", "camelbert-msa"])
    camel = runner.resolve_model(camel_args)
    assert camel.closed_track_verified
    assert len(camel.revision) == 40

    arabert_args = runner.parse_args(["--task", "1", "--model", "arabert-v02"])
    with pytest.raises(SystemExit, match="no runner-verified closed-track license"):
        runner.resolve_model(arabert_args)
    open_args = runner.parse_args(
        ["--task", "1", "--model", "arabert-v02", "--track", "open"]
    )
    assert runner.resolve_model(open_args).repository.endswith("arabertv02")


def test_custom_model_requires_revision_and_cannot_self_certify_closed():
    missing = runner.parse_args(["--task", "1", "--model", "org/custom"])
    with pytest.raises(SystemExit, match="require --revision"):
        runner.resolve_model(missing)
    custom = runner.parse_args(
        [
            "--task",
            "1",
            "--model",
            "org/custom",
            "--revision",
            "a" * 40,
            "--license-id",
            "apache-2.0",
            "--license-url",
            "https://example.test/license",
        ]
    )
    with pytest.raises(SystemExit, match="no runner-verified closed-track license"):
        runner.resolve_model(custom)


def test_parse_guards_task_specific_and_numeric_arguments():
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "1", "--folds", "1"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "1", "--context-chars", "10"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "2", "--stride", "0"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "1", "--learning-rate", "0"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "1", "--limit-strategy", "longest-text"])
    parsed = runner.parse_args(["--task", "1"])
    assert parsed.attention_implementation == "eager"
    assert parsed.limit_strategy == "random"


def test_longest_text_smoke_selection_is_deterministic():
    task1 = [
        {"paragraph_id": 1, "text": "قصير", "type": "editorial", "labels": []},
        {"paragraph_id": 2, "text": "أطول بقليل", "type": "debate", "labels": []},
        {
            "paragraph_id": 3,
            "text": "هذا هو النص الأطول في المجموعة",
            "type": "editorial",
            "labels": [],
        },
    ]
    task2 = [dict(row) for row in task1]
    selected = runner.select_pool_ids(
        task1,
        task2,
        pool="all",
        setting="both",
        limit=2,
        limit_strategy="longest-text",
        seed=1,
    )
    assert selected == [2, 3]


class FakeScaler:
    def __init__(self, execute: bool) -> None:
        self.execute = execute
        self.updated = False

    def step(self, optimizer):
        if self.execute:
            optimizer.step()

    def update(self):
        self.updated = True


def _adamw_test_objects():
    parameter = runner.nn.Parameter(runner.torch.tensor([1.0]))
    optimizer = runner.torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
    scheduler = runner.torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    return parameter, optimizer, scheduler


def test_successful_scaler_attempt_updates_adamw_scheduler_and_accounting():
    parameter, optimizer, scheduler = _adamw_test_objects()
    scheduler_epoch = scheduler.last_epoch
    before = parameter.detach().clone()
    parameter.grad = runner.torch.ones_like(parameter)
    accounting = runner.OptimizerStepAccounting()

    scaler = FakeScaler(execute=True)
    assert runner._attempt_scaled_optimizer_step(
        scaler, optimizer, scheduler, accounting
    )

    assert scaler.updated
    assert not runner.torch.equal(parameter.detach(), before)
    assert scheduler.last_epoch == scheduler_epoch + 1
    assert accounting.optimizer_step_attempts == 1
    assert accounting.optimizer_steps == 1
    assert accounting.skipped_optimizer_steps == 0
    assert accounting.scheduler_steps == 1
    assert parameter.grad is None


def test_skipped_scaler_attempt_does_not_update_adamw_or_scheduler():
    parameter, optimizer, scheduler = _adamw_test_objects()
    scheduler_epoch = scheduler.last_epoch
    before = parameter.detach().clone()
    parameter.grad = runner.torch.ones_like(parameter)
    accounting = runner.OptimizerStepAccounting()

    scaler = FakeScaler(execute=False)
    assert not runner._attempt_scaled_optimizer_step(
        scaler, optimizer, scheduler, accounting
    )

    assert scaler.updated
    assert runner.torch.equal(parameter.detach(), before)
    assert scheduler.last_epoch == scheduler_epoch
    assert accounting.optimizer_step_attempts == 1
    assert accounting.optimizer_steps == 0
    assert accounting.skipped_optimizer_steps == 1
    assert accounting.scheduler_steps == 0
    assert parameter.grad is None


def test_skip_then_success_reaches_max_steps_only_after_real_update():
    parameter, optimizer, scheduler = _adamw_test_objects()
    accounting = runner.OptimizerStepAccounting()
    attempts_consumed = 0

    for execute in (False, True):
        parameter.grad = runner.torch.ones_like(parameter)
        attempts_consumed += 1
        runner._attempt_scaled_optimizer_step(
            FakeScaler(execute=execute), optimizer, scheduler, accounting
        )
        if accounting.optimizer_steps >= 1:
            break

    assert attempts_consumed == 2
    assert accounting.optimizer_step_attempts == 2
    assert accounting.optimizer_steps == 1
    assert accounting.skipped_optimizer_steps == 1
    assert accounting.scheduler_steps == 1


def test_bounded_smoke_requires_requested_successful_updates():
    runner._require_requested_optimizer_steps(
        fold_index=0,
        requested=1,
        completed=1,
        attempts=2,
        skipped=1,
    )
    with pytest.raises(RuntimeError, match="requested 1 successful optimizer steps"):
        runner._require_requested_optimizer_steps(
            fold_index=0,
            requested=1,
            completed=0,
            attempts=1,
            skipped=1,
        )


def test_bounded_main_failure_does_not_write_completion(monkeypatch, tmp_path):
    records = [
        {"paragraph_id": 1, "text": "النص الأول", "type": "editorial"},
        {"paragraph_id": 2, "text": "النص الثاني", "type": "debate"},
    ]
    task1_path = tmp_path / "train_task_1.jsonl"
    task2_path = tmp_path / "train_task_2.jsonl"
    task1_path.write_text("{}\n", encoding="utf-8")
    task2_path.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "bounded-run"

    class FakeConfig:
        max_position_embeddings = 512
        _commit_hash = runner.ENCODER_SPECS["camelbert-msa-quarter"].revision

    class FakeTokenizer:
        is_fast = True

    monkeypatch.setattr(runner, "TRAIN_TASK1", task1_path)
    monkeypatch.setattr(runner, "TRAIN_TASK2", task2_path)
    monkeypatch.setattr(runner, "load_records", lambda _: records)
    monkeypatch.setattr(
        runner,
        "clean_task1_gold",
        lambda *_: {1: set(), 2: set()},
    )
    monkeypatch.setattr(
        runner,
        "clean_task2_gold",
        lambda *_: {1: [], 2: []},
    )
    monkeypatch.setattr(
        runner,
        "make_stratified_folds",
        lambda *_args, **_kwargs: [
            Fold(index=0, train_ids=(2,), val_ids=(1,)),
            Fold(index=1, train_ids=(1,), val_ids=(2,)),
        ],
    )
    monkeypatch.setattr(
        runner.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeConfig(),
    )
    monkeypatch.setattr(
        runner.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    def create_run_dir(*_args, **_kwargs):
        run_dir.mkdir()
        return run_dir

    def exhaust_without_update(*_args, **_kwargs):
        runner._require_requested_optimizer_steps(
            fold_index=0,
            requested=1,
            completed=0,
            attempts=1,
            skipped=1,
        )

    monkeypatch.setattr(runner, "create_experiment_dir", create_run_dir)
    monkeypatch.setattr(runner, "train_fold", exhaust_without_update)

    with pytest.raises(RuntimeError, match="requested 1 successful optimizer steps"):
        runner.main(
            [
                "--task",
                "1",
                "--pool",
                "all",
                "--folds",
                "2",
                "--limit-paragraphs",
                "2",
                "--epochs",
                "1",
                "--max-steps",
                "1",
                "--device",
                "cpu",
                "--precision",
                "fp32",
            ]
        )

    assert (run_dir / "config.json").is_file()
    assert (run_dir / "provenance.pending.json").is_file()
    assert not (run_dir / "COMPLETED.json").exists()


def test_eager_attention_and_strict_determinism_are_wired(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEncoder(runner.nn.Module):
        pass

    def fake_from_pretrained(*args, **kwargs):
        captured["model_args"] = args
        captured["model_kwargs"] = kwargs
        return FakeEncoder()

    deterministic_calls = []
    monkeypatch.setattr(runner.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        runner.torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only: deterministic_calls.append(
            (enabled, warn_only)
        ),
    )
    args = runner.parse_args(["--task", "1", "--model", "camelbert-msa-quarter"])
    spec = runner.resolve_model(args)
    runner._make_model(1, spec, SimpleNamespace(hidden_size=3), args)
    assert captured["model_kwargs"]["attn_implementation"] == "eager"

    runner.seed_everything(17, strict_determinism=True)
    assert deterministic_calls[-1] == (True, False)
    assert runner.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_class_weights_are_finite_bounded_and_task_shaped():
    paragraph_examples = [
        ParagraphExample(
            index,
            "نص",
            "editorial",
            tuple(float((index + label_index) % 2) for label_index in range(len(LABELS))),
        )
        for index in range(4)
    ]
    weights1 = runner._class_weights(1, paragraph_examples, "balanced", 10.0)
    assert tuple(weights1.shape) == (len(LABELS),)
    assert np.isfinite(weights1.numpy()).all()

    text = "abcdefg"
    segment_examples = [
        SegmentExample(
            paragraph_id=index,
            paragraph_text=text,
            genre="debate",
            start=0,
            end=1,
            text="a",
            label=label,
            position_bucket=0,
        )
        for index, label in enumerate(runner.SEGMENT_LABELS)
    ]
    weights2 = runner._class_weights(2, segment_examples, "sqrt", 10.0)
    assert tuple(weights2.shape) == (len(runner.SEGMENT_LABELS),)
    assert np.isfinite(weights2.numpy()).all()
    assert runner._class_weights(1, paragraph_examples, "none", 10.0) is None


def test_device_and_precision_resolution_fail_loudly_when_unavailable():
    assert runner.resolve_device("cpu").type == "cpu"
    assert runner.resolve_precision("auto", runner.torch.device("cpu"))[0] == "fp32"
    with pytest.raises(SystemExit, match="fp16"):
        runner.resolve_precision("fp16", runner.torch.device("cpu"))
