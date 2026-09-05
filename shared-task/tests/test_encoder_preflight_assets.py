"""Local contract tests for the private Kaggle encoder preflight assets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SHARED_TASK = Path(__file__).resolve().parents[1]
PREFLIGHT = SHARED_TASK / "kaggle" / "encoder-preflight"
sys.path.insert(0, str(SHARED_TASK))

import scripts.train_encoder_baseline as runner


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assets = load_module("daleel_encoder_preflight_assets", PREFLIGHT / "prepare_assets.py")
kernel = load_module(
    "daleel_encoder_preflight_kernel",
    PREFLIGHT / "kernel" / "encoder_preflight.py",
)
campaign = load_module(
    "daleel_encoder_campaign_kernel",
    SHARED_TASK / "kaggle" / "encoder-campaign" / "kernel" / "encoder_campaign.py",
)


def test_dataset_builder_uses_exact_handoff_path_and_whitelist(tmp_path):
    dataset = assets.prepare_dataset(tmp_path, "test-owner")

    assert dataset == tmp_path / assets.DATASET_SLUG
    assert assets.DEFAULT_OUT == Path("/private/tmp/daleel-encoder-preflight-upload")
    files = {
        path.relative_to(dataset).as_posix()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    manifest = json.loads(
        (dataset / "bundle-manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest) == files - {"bundle-manifest.json"}
    assert all(
        assets.sha256(dataset / relative) == expected_hash
        for relative, expected_hash in manifest.items()
    )
    assert {
        "dataset-metadata.json",
        "shared-task/scripts/train_encoder_baseline.py",
        "resources/repos/Daleel2026/data/train/train_task_1.jsonl",
        "resources/repos/Daleel2026/data/train/train_task_2.jsonl",
    } <= files
    assert not any(
        forbidden in relative
        for relative in files
        for forbidden in ("experiments/", "predictions/", "kaggle.json", "__pycache__/")
    )


def test_kernel_metadata_is_durable_private_and_t4(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "encoder_preflight.py").write_text("pass\n", encoding="utf-8")

    assert assets.prepare_kernel(kernel_dir, "test-owner") == kernel_dir
    metadata = json.loads(
        (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["id"] == "test-owner/daleel-camelbert-t4-preflight"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "test-owner/daleel-encoder-preflight-bundle-v1"
    ]

    durable = json.loads(
        (PREFLIGHT / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert durable["id"] == "salah1992/daleel-camelbert-t4-preflight"
    assert durable["code_file"] == "encoder_preflight.py"


@pytest.mark.parametrize(
    "mount_parts",
    [
        (kernel.DATASET_SLUG,),
        ("datasets", "test-owner", kernel.DATASET_SLUG),
    ],
)
def test_input_discovery_supports_legacy_and_owner_slug_mounts(
    monkeypatch, tmp_path, mount_parts
):
    bundle = tmp_path.joinpath(*mount_parts)
    (bundle / "shared-task" / "scripts").mkdir(parents=True)
    train = bundle / "resources" / "repos" / "Daleel2026" / "data" / "train"
    train.mkdir(parents=True)
    (bundle / "shared-task" / "scripts" / "train_encoder_baseline.py").write_text(
        "pass\n", encoding="utf-8"
    )
    (train / "train_task_1.jsonl").write_text("{}\n", encoding="utf-8")
    (train / "train_task_2.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(kernel, "RAW_INPUT", tmp_path)

    assert kernel.locate_expanded_bundle() == bundle


def test_forbidden_log_scanner_rejects_the_prior_remote_warnings():
    old_remote_warnings = "\n".join(
        [
            "Memory Efficient attention defaults to a non-deterministic algorithm.",
            "Detected call of `lr_scheduler.step()` before `optimizer.step()`.",
        ]
    )
    assert set(kernel.forbidden_log_signatures(old_remote_warnings)) == {
        "nondeterministic_attention",
        "scheduler_before_optimizer",
    }
    assert kernel.forbidden_log_signatures(
        "Warning: unauthenticated requests to the HF Hub"
    ) == []


@pytest.mark.parametrize(
    ("message", "signature"),
    [
        ("Tesla P100 compatibility warning", "p100_compatibility"),
        ("torch.OutOfMemoryError: CUDA out of memory", "cuda_out_of_memory"),
    ],
)
def test_forbidden_log_scanner_rejects_hardware_failures(message, signature):
    assert signature in kernel.forbidden_log_signatures(message)


def test_kernel_contract_is_long_text_strict_eager_effective_batch_eight():
    arguments = kernel.EXPECTED_ARGUMENTS
    assert arguments["limit_paragraphs"] == 20
    assert arguments["limit_strategy"] == "longest-text"
    assert arguments["batch_size"] * arguments["gradient_accumulation"] == 8
    assert arguments["max_length"] == 512
    assert arguments["stride"] == 128
    assert arguments["attention_implementation"] == "eager"
    assert arguments["freeze_encoder"] is False
    assert arguments["max_steps"] == 1
    assert "predictions" not in " ".join(kernel.SAFE_ARTIFACTS)


def test_kernel_acceptance_harness_passes_complete_synthetic_contract(
    monkeypatch, tmp_path
):
    raw_input = tmp_path / "input"
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    bundle = raw_input / kernel.DATASET_SLUG
    (bundle / "shared-task" / "scripts").mkdir(parents=True)
    train = bundle / "resources" / "repos" / "Daleel2026" / "data" / "train"
    train.mkdir(parents=True)
    (bundle / "shared-task" / "scripts" / "train_encoder_baseline.py").write_text(
        "pass\n", encoding="utf-8"
    )
    (train / "train_task_1.jsonl").write_text("{}\n", encoding="utf-8")
    (train / "train_task_2.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(kernel, "RAW_INPUT", raw_input)
    monkeypatch.setattr(kernel, "WORKSPACE", workspace)
    monkeypatch.setattr(kernel, "OUTPUT", output)
    monkeypatch.setattr(
        kernel,
        "gpu_inventory",
        lambda: [
            "0, Tesla T4, 15360 MiB, 7.5",
            "1, Tesla T4, 15360 MiB, 7.5",
        ],
    )

    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def fake_run_trainer(command, cwd, environment):
        del command, environment
        run_dir = (
            cwd
            / "experiments"
            / "20260711-encoder-t1-camelbert-msa-quarter-both-legacy-train-"
            "smoke-synthetic"
        )
        run_dir.mkdir(parents=True)
        arguments = dict(kernel.EXPECTED_ARGUMENTS)
        config = {
            "script_version": kernel.EXPECTED_SCRIPT_VERSION,
            "model": {
                "repository": kernel.EXPECTED_MODEL_REPOSITORY,
                "revision": kernel.EXPECTED_MODEL_REVISION,
            },
            "resolved_commit": kernel.EXPECTED_MODEL_REVISION,
            "tokenizer": {
                "repository": kernel.EXPECTED_MODEL_REPOSITORY,
                "revision": kernel.EXPECTED_MODEL_REVISION,
                "is_fast": True,
            },
            "arguments": arguments,
            "runtime": {
                "device": "cuda",
                "cuda_available": True,
                "cuda_device_count": 2,
                "cuda_device_index": 0,
                "cuda_device_name": "Tesla T4",
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
                "cublas_workspace_config": ":4096:8",
            },
            "selected_paragraph_count": 20,
            "comparable_run": False,
        }
        history = []
        for fold, allocated, reserved in ((0, 100, 200), (1, 120, 240)):
            history.append(
                {
                    "fold": fold,
                    "optimizer_step_attempts": 1,
                    "optimizer_steps": 1,
                    "skipped_optimizer_steps": 0,
                    "scheduler_steps": 1,
                    "training_seconds": 1.0,
                    "validation_loss": 0.5,
                    "max_cuda_memory_allocated_bytes": allocated,
                    "max_cuda_memory_reserved_bytes": reserved,
                    "epochs": [
                        {
                            "mean_training_loss": 0.5,
                            "optimizer_step_attempts_total": 1,
                            "optimizer_steps_total": 1,
                            "skipped_optimizer_steps_total": 0,
                            "scheduler_steps_total": 1,
                        }
                    ],
                }
            )
        metrics = {
            "fold_count": 2,
            "comparable_run": False,
            "primary": {"name": "synthetic", "score": 0.5},
            "training": {
                "optimizer_step_attempts": 2,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "scheduler_steps": 2,
                "max_cuda_memory_allocated_bytes": 120,
                "max_cuda_memory_reserved_bytes": 240,
            },
        }
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "folds.json", {"folds": 2})
        write_json(run_dir / "training_history.json", history)
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "provenance.json", {"status": "complete"})
        completion_files = []
        for name in kernel.SAFE_ARTIFACTS:
            if name == "COMPLETED.json":
                continue
            completion_files.append(
                {"relative_path": name, "sha256": kernel.sha256(run_dir / name)}
            )
        write_json(
            run_dir / "COMPLETED.json",
            {
                "completion_schema_version": 1,
                "complete": True,
                "files": completion_files,
            },
        )
        return []

    monkeypatch.setattr(kernel, "run_trainer", fake_run_trainer)
    kernel.main()

    exported = {path.name for path in output.iterdir()}
    assert exported == set(kernel.SAFE_ARTIFACTS) | {"preflight_checks.json"}
    checks = json.loads(
        (output / "preflight_checks.json").read_text(encoding="utf-8")
    )
    assert checks["optimizer_steps_by_fold"] == [1, 1]
    assert checks["scheduler_steps_by_fold"] == [1, 1]
    assert checks["trainer_log_forbidden_signatures"] == []
    assert checks["completion_validated"] is True


def test_campaign_scope_and_preregistered_arguments_are_exact():
    assert set(campaign.TASKS) == {"task1", "task2"}
    task1 = campaign.expected_arguments("task1")
    task2 = campaign.expected_arguments("task2")
    assert task1["task"] == 1
    assert task1["batch_size"] == 4
    assert task1["gradient_accumulation"] == 2
    assert task2["task"] == 2
    assert task2["batch_size"] == 16
    assert task2["gradient_accumulation"] == 1
    for arguments in (task1, task2):
        assert arguments["folds"] == 5
        assert arguments["epochs"] == 4
        assert arguments["limit_paragraphs"] is None
        assert arguments["max_steps"] is None
        assert arguments["model"] == "camelbert-msa-quarter"
        assert arguments["pool"] == "legacy-train"
        assert arguments["class_weighting"] == "none"
        assert arguments["max_length"] == 512
        assert arguments["precision"] == "fp16"
        assert arguments["attention_implementation"] == "eager"
        assert arguments["freeze_encoder"] is False


@pytest.mark.parametrize("task_key", ["task1", "task2"])
def test_campaign_commands_parse_to_the_expected_remote_contract(task_key):
    parsed = vars(runner.parse_args(campaign.TASKS[task_key]["command"])).copy()
    # ``--precision auto`` deterministically resolves to FP16 on the required
    # T4 CUDA runtime before the runner writes config.json.
    assert parsed["precision"] == "auto"
    parsed["precision"] = "fp16"
    assert parsed == campaign.expected_arguments(task_key)


def test_campaign_manifest_accepts_kaggle_consumed_dataset_metadata(tmp_path):
    bundle = assets.prepare_dataset(tmp_path, "test-owner")
    (bundle / "dataset-metadata.json").unlink()
    campaign.validate_bundle_manifest(bundle)


def test_campaign_expected_optimizer_attempts_match_task1_fold_shape():
    row = {"train_item_count": 344}
    assert campaign.expected_optimizer_attempts(
        row, campaign.expected_arguments("task1")
    ) == 172


def test_campaign_log_scanner_rejects_prior_remote_defects():
    message = "\n".join(
        [
            "Memory Efficient attention defaults to a non-deterministic algorithm.",
            "Detected call of `lr_scheduler.step()` before `optimizer.step()`.",
        ]
    )
    assert set(campaign.forbidden_log_signatures(message)) == {
        "nondeterministic_attention",
        "scheduler_before_optimizer",
    }


@pytest.mark.parametrize("task_key", ["task1", "task2"])
def test_campaign_validates_complete_synthetic_fivefold_run(tmp_path, task_key):
    run_dir = tmp_path / task_key
    run_dir.mkdir()
    arguments = campaign.expected_arguments(task_key)
    train_items = 344 if task_key == "task1" else 800
    training_items = 430 if task_key == "task1" else 1000
    attempts_per_epoch = (
        campaign.expected_optimizer_attempts(
            {"train_item_count": train_items}, arguments
        )
        // 4
    )
    attempts_per_fold = attempts_per_epoch * 4
    history = []
    skipped_by_fold = []
    for fold in range(5):
        fold_skips = 0
        if task_key == "task1" and fold == 3:
            fold_skips = 2
        elif task_key == "task1" and fold == 4:
            fold_skips = 1
        skipped_by_fold.append(fold_skips)
        epochs = []
        for epoch in range(1, 5):
            total = attempts_per_epoch * epoch
            cumulative_skips = fold_skips if epoch == 4 else 0
            epochs.append(
                {
                    "epoch": epoch,
                    "mean_training_loss": 0.8 - epoch * 0.05,
                    "optimizer_step_attempts_total": total,
                    "optimizer_steps_total": total - cumulative_skips,
                    "skipped_optimizer_steps_total": cumulative_skips,
                    "scheduler_steps_total": total - cumulative_skips,
                }
            )
        history.append(
            {
                "fold": fold,
                "train_item_count": train_items,
                "optimizer_step_attempts": attempts_per_fold,
                "optimizer_steps": attempts_per_fold - fold_skips,
                "skipped_optimizer_steps": fold_skips,
                "scheduler_steps": attempts_per_fold - fold_skips,
                "training_seconds": 10.0,
                "validation_loss": 0.5,
                "max_cuda_memory_allocated_bytes": 100 + fold,
                "max_cuda_memory_reserved_bytes": 200 + fold,
                "epochs": epochs,
            }
        )
    config = {
        "script_version": campaign.EXPECTED_SCRIPT_VERSION,
        "model": {
            "repository": campaign.EXPECTED_MODEL_REPOSITORY,
            "revision": campaign.EXPECTED_MODEL_REVISION,
        },
        "resolved_commit": campaign.EXPECTED_MODEL_REVISION,
        "tokenizer": {
            "repository": campaign.EXPECTED_MODEL_REPOSITORY,
            "revision": campaign.EXPECTED_MODEL_REVISION,
            "is_fast": True,
        },
        "arguments": arguments,
        "runtime": {
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_count": 2,
            "cuda_device_index": 0,
            "cuda_device_name": "Tesla T4",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cublas_workspace_config": ":4096:8",
        },
        "selected_paragraph_count": 430,
        "training_item_count": training_items,
        "comparable_run": True,
    }
    total_attempts = attempts_per_fold * 5
    total_skips = sum(skipped_by_fold)
    metrics = {
        "paragraph_count": 430,
        "fold_count": 5,
        "comparable_run": True,
        "primary": {
            "name": campaign.TASKS[task_key]["primary_metric"],
            "score": 0.5,
        },
        "training": {
            "optimizer_step_attempts": total_attempts,
            "optimizer_steps": total_attempts - total_skips,
            "skipped_optimizer_steps": total_skips,
            "scheduler_steps": total_attempts - total_skips,
            "max_cuda_memory_allocated_bytes": 104,
            "max_cuda_memory_reserved_bytes": 204,
        },
    }
    provenance = {
        "status": "complete",
        "comparable_run": True,
        "dspy_used": False,
        "generative_inference_used": False,
    }

    def write_json(path, value):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    write_json(run_dir / "config.json", config)
    write_json(run_dir / "folds.json", {"fold_count": 5})
    write_json(run_dir / "training_history.json", history)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "provenance.json", provenance)
    completion_files = []
    for name in campaign.SAFE_RUN_ARTIFACTS:
        if name == "COMPLETED.json":
            continue
        completion_files.append(
            {"relative_path": name, "sha256": campaign.sha256(run_dir / name)}
        )
    write_json(
        run_dir / "COMPLETED.json",
        {
            "completion_schema_version": 1,
            "complete": True,
            "files": completion_files,
        },
    )
    inventory = [
        "0, Tesla T4, 15360 MiB, 7.5",
        "1, Tesla T4, 15360 MiB, 7.5",
    ]
    checks = campaign.validate_run(task_key, run_dir, inventory, [])
    assert checks["completed"] is True
    assert checks["comparable_run"] is True
    expected_steps = [attempts_per_fold - value for value in skipped_by_fold]
    assert checks["optimizer_steps_by_fold"] == expected_steps
    assert checks["scheduler_steps_by_fold"] == expected_steps
    assert checks["skipped_optimizer_steps_by_fold"] == skipped_by_fold
    assert checks["skipped_optimizer_steps_total"] == total_skips


def test_campaign_kernel_metadata_is_private_t4_and_existing_bundle():
    metadata = json.loads(
        (
            SHARED_TASK
            / "kaggle"
            / "encoder-campaign"
            / "kernel"
            / "kernel-metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["id"] == "salah1992/daleel-camelbert-fivefold-campaign"
    assert assets.slug(metadata["title"]) == metadata["id"].split("/", 1)[1]
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "salah1992/daleel-encoder-preflight-bundle-v1"
    ]
