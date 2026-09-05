"""Run the two preregistered CAMeLBERT-MSA-quarter five-fold primaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


RAW_INPUT = Path("/kaggle/input")
WORKSPACE = Path("/tmp/daleel-encoder-campaign-workspace")
OUTPUT = Path("/kaggle/working/encoder_campaign_artifacts")
DATASET_SLUG = "daleel-encoder-preflight-bundle-v1"
EXPECTED_SCRIPT_VERSION = 3
EXPECTED_MODEL_REPOSITORY = "CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter"
EXPECTED_MODEL_REVISION = "3e48534705c153737cbec1c5748bb02359b7b239"
EXPECTED_PARAGRAPH_COUNT = 430
EXPECTED_FOLD_COUNT = 5
MAX_RESERVED_BYTES = int(13.5 * 1024**3)
SAFE_RUN_ARTIFACTS = (
    "config.json",
    "folds.json",
    "training_history.json",
    "metrics.json",
    "provenance.json",
    "COMPLETED.json",
)
COMMON_EXPECTED_ARGUMENTS = {
    "model": "camelbert-msa-quarter",
    "revision": None,
    "license_id": None,
    "license_url": None,
    "track": "closed",
    "offline": False,
    "pool": "legacy-train",
    "setting": "both",
    "folds": 5,
    "fold_seed": 20260710,
    "seed": 20260710,
    "limit_paragraphs": None,
    "limit_strategy": "random",
    "epochs": 4,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "max_steps": None,
    "dropout": 0.1,
    "freeze_encoder": False,
    "gradient_checkpointing": False,
    "class_weighting": "none",
    "max_class_weight": 20.0,
    "max_length": 512,
    "stride": 128,
    "window_pool": "max",
    "context_chars": 0,
    "granularity": "connective",
    "pad_to_multiple_of": 8,
    "num_workers": 0,
    "cpu_threads": None,
    "device": "auto",
    "precision": "fp16",
    "attention_implementation": "eager",
}
TASKS = {
    "task1": {
        "task": 1,
        "batch_size": 4,
        "gradient_accumulation": 2,
        "primary_metric": "official_task1_macro_f1_cross_fitted_oof_thresholds",
        "command": [
            "--task",
            "1",
            "--model",
            "camelbert-msa-quarter",
            "--track",
            "closed",
            "--pool",
            "legacy-train",
            "--setting",
            "both",
            "--folds",
            "5",
            "--epochs",
            "4",
            "--batch-size",
            "4",
            "--gradient-accumulation",
            "2",
            "--learning-rate",
            "2e-5",
            "--weight-decay",
            "0.01",
            "--max-length",
            "512",
            "--stride",
            "128",
            "--window-pool",
            "max",
            "--class-weighting",
            "none",
            "--precision",
            "auto",
            "--attention-implementation",
            "eager",
        ],
    },
    "task2": {
        "task": 2,
        "batch_size": 16,
        "gradient_accumulation": 1,
        "primary_metric": "official_task2_partial_f1_oof",
        "command": [
            "--task",
            "2",
            "--model",
            "camelbert-msa-quarter",
            "--track",
            "closed",
            "--pool",
            "legacy-train",
            "--setting",
            "both",
            "--folds",
            "5",
            "--epochs",
            "4",
            "--batch-size",
            "16",
            "--gradient-accumulation",
            "1",
            "--learning-rate",
            "2e-5",
            "--weight-decay",
            "0.01",
            "--max-length",
            "512",
            "--context-chars",
            "0",
            "--granularity",
            "connective",
            "--class-weighting",
            "none",
            "--precision",
            "auto",
            "--attention-implementation",
            "eager",
        ],
    },
}
FORBIDDEN_LOG_PATTERNS = {
    "p100_compatibility": re.compile(r"\b(?:Tesla\s+)?P100\b", re.IGNORECASE),
    "cuda_out_of_memory": re.compile(
        r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
        re.IGNORECASE,
    ),
    "nondeterministic_attention": re.compile(
        r"(?:non[- ]deterministic.{0,200}attention|"
        r"attention.{0,200}non[- ]deterministic)",
        re.IGNORECASE | re.DOTALL,
    ),
    "scheduler_before_optimizer": re.compile(
        r"lr_scheduler\.step\(\).{0,240}before.{0,100}optimizer\.step\(\)",
        re.IGNORECASE | re.DOTALL,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def gpu_inventory() -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def describe_tree(root: Path, max_entries: int = 200) -> None:
    print(f"KAGGLE_RAW_INPUT_ROOT={root}", flush=True)
    if not root.exists():
        raise SystemExit(f"Kaggle input root does not exist: {root}")
    entries = sorted(root.rglob("*"))
    if not entries:
        raise SystemExit(f"Kaggle input root is empty: {root}")
    for index, path in enumerate(entries):
        if index >= max_entries:
            print(f"KAGGLE_RAW_INPUT_MORE={len(entries) - max_entries}", flush=True)
            break
        kind = "D" if path.is_dir() else "F"
        print(f"KAGGLE_RAW_INPUT_{kind}={path.relative_to(root)}", flush=True)


def locate_expanded_bundle() -> Path:
    candidates = [
        RAW_INPUT / DATASET_SLUG,
        RAW_INPUT / "datasets" / "salah1992" / DATASET_SLUG,
        RAW_INPUT,
    ]
    candidates.extend(path for path in RAW_INPUT.rglob("*") if path.is_dir())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "bundle-manifest.json").is_file()
            and (candidate / "shared-task/scripts/train_encoder_baseline.py").is_file()
            and (
                candidate
                / "resources/repos/Daleel2026/data/train/train_task_1.jsonl"
            ).is_file()
            and (
                candidate
                / "resources/repos/Daleel2026/data/train/train_task_2.jsonl"
            ).is_file()
        ):
            return candidate
    raise SystemExit("attached private encoder bundle could not be found")


def validate_bundle_manifest(bundle: Path) -> None:
    manifest_path = bundle / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise SystemExit(f"invalid or empty bundle manifest: {manifest_path}")
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    # Kaggle consumes dataset-metadata.json during version creation and does
    # not expose it in the mounted dataset.  Every executable/data payload
    # file must still be present and hash-bound, and no extra payload is
    # accepted.
    optional_consumed_metadata = {"dataset-metadata.json"}
    required_files = set(manifest) - optional_consumed_metadata
    allowed_files = set(manifest) | {"bundle-manifest.json"}
    if not required_files <= actual_files or not actual_files <= allowed_files:
        raise SystemExit(
            "bundle whitelist mismatch: "
            f"missing={sorted(required_files - actual_files)}, "
            f"extra={sorted(actual_files - allowed_files)}"
        )
    for relative, expected_hash in manifest.items():
        if relative not in actual_files:
            continue
        path = (bundle / relative).resolve()
        try:
            path.relative_to(bundle.resolve())
        except ValueError as exc:
            raise SystemExit(f"bundle manifest path escapes root: {relative}") from exc
        if not path.is_file() or sha256(path) != expected_hash:
            raise SystemExit(f"bundle manifest hash mismatch: {relative}")


def materialize_bundle(source: Path) -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copytree(source / "shared-task", WORKSPACE / "shared-task")
    shutil.copytree(source / "resources", WORKSPACE / "resources")
    print(f"COPIED_EXPANDED_KAGGLE_INPUT={source}", flush=True)


def forbidden_log_signatures(log: str) -> list[str]:
    return [
        name for name, pattern in FORBIDDEN_LOG_PATTERNS.items() if pattern.search(log)
    ]


def run_trainer(command: list[str], cwd: Path, environment: dict[str, str]) -> list[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
        raise RuntimeError("trainer stdout pipe was not created")
    lines: list[str] = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return_code = process.wait()
    violations = forbidden_log_signatures("".join(lines))
    if violations:
        raise SystemExit(f"forbidden trainer log signatures detected: {violations}")
    if return_code != 0:
        raise SystemExit(f"encoder trainer exited with status {return_code}")
    return violations


def validate_completion(run_dir: Path) -> None:
    marker_path = run_dir / "COMPLETED.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("completion_schema_version") != 1 or marker.get("complete") is not True:
        raise SystemExit(f"invalid completion marker: {marker_path}")
    rows = marker.get("files")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"completion marker has no file manifest: {marker_path}")
    for row in rows:
        relative = row.get("relative_path") if isinstance(row, dict) else None
        expected_hash = row.get("sha256") if isinstance(row, dict) else None
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise SystemExit(f"invalid completion entry: {row!r}")
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise SystemExit(f"completion entry escapes run directory: {relative}") from exc
        if not path.is_file() or sha256(path) != expected_hash:
            raise SystemExit(f"completion hash mismatch: {relative}")


def expected_arguments(task_key: str) -> dict[str, object]:
    task = TASKS[task_key]
    return COMMON_EXPECTED_ARGUMENTS | {
        "task": task["task"],
        "batch_size": task["batch_size"],
        "gradient_accumulation": task["gradient_accumulation"],
    }


def expected_optimizer_attempts(row: dict[str, object], arguments: dict[str, object]) -> int:
    train_items = int(row["train_item_count"])
    batches = math.ceil(train_items / int(arguments["batch_size"]))
    attempts_per_epoch = math.ceil(
        batches / int(arguments["gradient_accumulation"])
    )
    return attempts_per_epoch * int(arguments["epochs"])


def validate_run(
    task_key: str,
    run_dir: Path,
    inventory: list[str],
    log_violations: list[str],
) -> dict[str, object]:
    for name in SAFE_RUN_ARTIFACTS:
        if not (run_dir / name).is_file():
            raise SystemExit(f"{task_key}: required safe artifact missing: {name}")
    validate_completion(run_dir)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (run_dir / "provenance.json").read_text(encoding="utf-8")
    )
    arguments = config["arguments"]
    runtime = config["runtime"]
    training_metrics = metrics["training"]
    expected = expected_arguments(task_key)
    mismatches = {
        key: {"expected": value, "actual": arguments.get(key)}
        for key, value in expected.items()
        if arguments.get(key) != value
    }
    optimizer_attempts = [row["optimizer_step_attempts"] for row in history]
    optimizer_steps = [row["optimizer_steps"] for row in history]
    skipped_steps = [row["skipped_optimizer_steps"] for row in history]
    scheduler_steps = [row["scheduler_steps"] for row in history]
    allocated_by_fold = [row["max_cuda_memory_allocated_bytes"] for row in history]
    reserved_by_fold = [row["max_cuda_memory_reserved_bytes"] for row in history]
    training_losses = [
        epoch["mean_training_loss"] for row in history for epoch in row["epochs"]
    ]
    validation_losses = [row["validation_loss"] for row in history]
    checks: dict[str, object] = {
        "task_key": task_key,
        "run_directory": run_dir.name,
        "gpu_inventory": inventory,
        "script_version": config["script_version"],
        "argument_mismatches": mismatches,
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "resolved_commit": config["resolved_commit"],
        "tokenizer_repository": config["tokenizer"]["repository"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "tokenizer_is_fast": config["tokenizer"]["is_fast"],
        "device": runtime["device"],
        "cuda_available": runtime["cuda_available"],
        "cuda_device_count": runtime["cuda_device_count"],
        "cuda_device_index": runtime["cuda_device_index"],
        "cuda_device_name": runtime["cuda_device_name"],
        "precision": arguments["precision"],
        "attention_implementation": arguments["attention_implementation"],
        "deterministic_algorithms": runtime["deterministic_algorithms"],
        "deterministic_warn_only": runtime["deterministic_warn_only"],
        "cublas_workspace_config": runtime["cublas_workspace_config"],
        "selected_paragraph_count": config["selected_paragraph_count"],
        "training_item_count": config["training_item_count"],
        "fold_count": metrics["fold_count"],
        "epoch_count_by_fold": [len(row["epochs"]) for row in history],
        "optimizer_step_attempts_by_fold": optimizer_attempts,
        "optimizer_steps_by_fold": optimizer_steps,
        "skipped_optimizer_steps_by_fold": skipped_steps,
        "skipped_optimizer_steps_total": sum(skipped_steps),
        "skipped_optimizer_step_fraction": (
            sum(skipped_steps) / sum(optimizer_attempts)
        ),
        "scheduler_steps_by_fold": scheduler_steps,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "max_cuda_memory_allocated_bytes": training_metrics[
            "max_cuda_memory_allocated_bytes"
        ],
        "max_cuda_memory_reserved_bytes": training_metrics[
            "max_cuda_memory_reserved_bytes"
        ],
        "max_reserved_limit_bytes": MAX_RESERVED_BYTES,
        "trainer_log_forbidden_signatures": log_violations,
        "primary": metrics["primary"],
        "provenance_status": provenance["status"],
        "dspy_used": provenance["dspy_used"],
        "generative_inference_used": provenance["generative_inference_used"],
        "completion_validated": True,
        "comparable_run": metrics["comparable_run"],
        "completed": True,
    }

    if config["script_version"] != EXPECTED_SCRIPT_VERSION or mismatches:
        raise SystemExit(f"{task_key}: stale runner or argument drift: {checks}")
    if any(
        actual != expected_value
        for actual, expected_value in (
            (config["model"]["repository"], EXPECTED_MODEL_REPOSITORY),
            (config["model"]["revision"], EXPECTED_MODEL_REVISION),
            (config["resolved_commit"], EXPECTED_MODEL_REVISION),
            (config["tokenizer"]["repository"], EXPECTED_MODEL_REPOSITORY),
            (config["tokenizer"]["revision"], EXPECTED_MODEL_REVISION),
        )
    ) or not config["tokenizer"]["is_fast"]:
        raise SystemExit(f"{task_key}: pinned model/tokenizer contract failed: {checks}")
    if (
        runtime["device"] != "cuda"
        or not runtime["cuda_available"]
        or runtime["cuda_device_count"] != 2
        or runtime["cuda_device_index"] != 0
        or "T4" not in (runtime["cuda_device_name"] or "")
    ):
        raise SystemExit(f"{task_key}: runner did not use cuda:0 on T4 x2: {checks}")
    if (
        arguments["precision"] != "fp16"
        or arguments["attention_implementation"] != "eager"
        or not runtime["deterministic_algorithms"]
        or runtime["deterministic_warn_only"] is not False
        or runtime["cublas_workspace_config"] != ":4096:8"
    ):
        raise SystemExit(f"{task_key}: strict deterministic eager FP16 failed: {checks}")
    if (
        config["selected_paragraph_count"] != EXPECTED_PARAGRAPH_COUNT
        or metrics["paragraph_count"] != EXPECTED_PARAGRAPH_COUNT
        or metrics["fold_count"] != EXPECTED_FOLD_COUNT
        or len(history) != EXPECTED_FOLD_COUNT
        or any(len(row["epochs"]) != 4 for row in history)
    ):
        raise SystemExit(f"{task_key}: pool/fold/epoch contract failed: {checks}")
    for row in history:
        expected_attempts = expected_optimizer_attempts(row, arguments)
        if (
            row["optimizer_step_attempts"] != expected_attempts
            or row["optimizer_steps"] + row["skipped_optimizer_steps"]
            != expected_attempts
            or row["scheduler_steps"] != row["optimizer_steps"]
        ):
            raise SystemExit(
                f"{task_key}: fixed-epoch attempt contract failed: {checks}"
            )
        previous_attempts = 0
        previous_steps = 0
        previous_skips = 0
        for epoch in row["epochs"]:
            epoch_attempts = epoch["optimizer_step_attempts_total"]
            epoch_steps = epoch["optimizer_steps_total"]
            epoch_skips = epoch["skipped_optimizer_steps_total"]
            if (
                epoch_attempts != epoch_steps + epoch_skips
                or epoch["scheduler_steps_total"] != epoch_steps
                or epoch_attempts < previous_attempts
                or epoch_steps < previous_steps
                or epoch_skips < previous_skips
            ):
                raise SystemExit(
                    f"{task_key}: cumulative epoch accounting differs: {checks}"
                )
            previous_attempts = epoch_attempts
            previous_steps = epoch_steps
            previous_skips = epoch_skips
        if (
            previous_attempts != row["optimizer_step_attempts"]
            or previous_steps != row["optimizer_steps"]
            or previous_skips != row["skipped_optimizer_steps"]
        ):
            raise SystemExit(f"{task_key}: epoch/fold accounting differs: {checks}")
    if (
        training_metrics["optimizer_step_attempts"] != sum(optimizer_attempts)
        or training_metrics["optimizer_steps"] != sum(optimizer_steps)
        or training_metrics["skipped_optimizer_steps"] != sum(skipped_steps)
        or training_metrics["scheduler_steps"] != sum(scheduler_steps)
        or training_metrics["optimizer_step_attempts"]
        != training_metrics["optimizer_steps"]
        + training_metrics["skipped_optimizer_steps"]
        or training_metrics["scheduler_steps"]
        != training_metrics["optimizer_steps"]
    ):
        raise SystemExit(f"{task_key}: fold/metric accounting differs: {checks}")
    if not all(math.isfinite(value) for value in training_losses + validation_losses):
        raise SystemExit(f"{task_key}: non-finite training or validation loss: {checks}")
    if (
        metrics["primary"]["name"] != TASKS[task_key]["primary_metric"]
        or not math.isfinite(metrics["primary"]["score"])
    ):
        raise SystemExit(f"{task_key}: invalid primary metric: {checks}")
    if (
        training_metrics["max_cuda_memory_allocated_bytes"] != max(allocated_by_fold)
        or training_metrics["max_cuda_memory_reserved_bytes"] != max(reserved_by_fold)
        or any(
            allocated > reserved
            for allocated, reserved in zip(allocated_by_fold, reserved_by_fold)
        )
        or training_metrics["max_cuda_memory_reserved_bytes"] > MAX_RESERVED_BYTES
    ):
        raise SystemExit(f"{task_key}: CUDA memory contract failed: {checks}")
    if (
        config["comparable_run"] is not True
        or metrics["comparable_run"] is not True
        or provenance["comparable_run"] is not True
        or provenance["status"] != "complete"
        or provenance["dspy_used"] is not False
        or provenance["generative_inference_used"] is not False
    ):
        raise SystemExit(f"{task_key}: scientific provenance contract failed: {checks}")
    return checks


def copy_safe_run(run_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in SAFE_RUN_ARTIFACTS:
        shutil.copy2(run_dir / name, destination / name)


def write_campaign_completion(output: Path, checks: dict[str, object]) -> Path:
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "CAMPAIGN_COMPLETED.json":
            files.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "sha256": sha256(path),
                }
            )
    marker = {
        "campaign_completion_schema_version": 1,
        "complete": True,
        "tasks": ["task1", "task2"],
        "primary_metrics": {
            task: value["primary"] for task, value in checks.items()
        },
        "files": files,
    }
    path = output / "CAMPAIGN_COMPLETED.json"
    write_json(path, marker)
    return path


def main() -> None:
    describe_tree(RAW_INPUT)
    bundle = locate_expanded_bundle()
    validate_bundle_manifest(bundle)
    materialize_bundle(bundle)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    inventory = gpu_inventory()
    if len(inventory) != 2 or not all("Tesla T4" in row for row in inventory):
        raise SystemExit(f"expected T4 x2, found {inventory}")
    print("KAGGLE_T4_INVENTORY=" + json.dumps(inventory), flush=True)

    shared_task = WORKSPACE / "shared-task"
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HOME": "/tmp/huggingface-cache",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    campaign_checks: dict[str, object] = {}
    for task_key, task in TASKS.items():
        command = [
            sys.executable,
            "scripts/train_encoder_baseline.py",
            *task["command"],
        ]
        print(
            f"DALEEL_CAMPAIGN_COMMAND_{task_key.upper()}=" + json.dumps(command),
            flush=True,
        )
        violations = run_trainer(command, shared_task, environment)
        run_dirs = sorted(
            (shared_task / "experiments").glob(
                f"*-encoder-t{task['task']}-camelbert-msa-quarter-both-"
                "legacy-train-*"
            ),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if len(run_dirs) != 1:
            raise SystemExit(
                f"{task_key}: expected exactly one experiment, found {run_dirs}"
            )
        run_dir = run_dirs[0]
        checks = validate_run(task_key, run_dir, inventory, violations)
        campaign_checks[task_key] = checks
        copy_safe_run(run_dir, OUTPUT / task_key)
        print(
            f"DALEEL_CAMPAIGN_RESULT_{task_key.upper()}="
            + json.dumps(checks, sort_keys=True),
            flush=True,
        )

    write_json(OUTPUT / "campaign_checks.json", campaign_checks)
    completion_path = write_campaign_completion(OUTPUT, campaign_checks)
    expected_top_level = {
        "task1",
        "task2",
        "campaign_checks.json",
        "CAMPAIGN_COMPLETED.json",
    }
    if {path.name for path in OUTPUT.iterdir()} != expected_top_level:
        raise SystemExit("unsafe or incomplete campaign artifact export")
    print(f"DALEEL_ENCODER_CAMPAIGN_COMPLETED={completion_path}", flush=True)


if __name__ == "__main__":
    main()
