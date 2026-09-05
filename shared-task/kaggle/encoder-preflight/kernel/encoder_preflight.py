"""Run the real Daleel CAMeLBERT trainer on a strict long-text T4 smoke."""

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
WORKSPACE = Path("/tmp/daleel-encoder-preflight-workspace")
OUTPUT = Path("/kaggle/working/preflight_artifacts")
DATASET_SLUG = "daleel-encoder-preflight-bundle-v1"
EXPECTED_SCRIPT_VERSION = 3
EXPECTED_MODEL_REPOSITORY = "CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter"
EXPECTED_MODEL_REVISION = "3e48534705c153737cbec1c5748bb02359b7b239"
MAX_RESERVED_BYTES = int(13.5 * 1024**3)
SAFE_ARTIFACTS = (
    "config.json",
    "folds.json",
    "training_history.json",
    "metrics.json",
    "provenance.json",
    "COMPLETED.json",
)
EXPECTED_ARGUMENTS = {
    "task": 1,
    "model": "camelbert-msa-quarter",
    "revision": None,
    "license_id": None,
    "license_url": None,
    "track": "closed",
    "offline": False,
    "pool": "legacy-train",
    "setting": "both",
    "folds": 2,
    "fold_seed": 20260710,
    "seed": 20260710,
    "limit_paragraphs": 20,
    "limit_strategy": "longest-text",
    "epochs": 1,
    "batch_size": 4,
    "gradient_accumulation": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "max_steps": 1,
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
    "device": "cuda",
    "precision": "fp16",
    "attention_implementation": "eager",
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
    # Kaggle currently exposes both the legacy slug mount and an owner/slug
    # dataset mount.  The recursive fallback also tolerates a version layer.
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
            (candidate / "shared-task/scripts/train_encoder_baseline.py").is_file()
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


def materialize_bundle(source: Path) -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copytree(source / "shared-task", WORKSPACE / "shared-task")
    shutil.copytree(source / "resources", WORKSPACE / "resources")
    print(f"COPIED_EXPANDED_KAGGLE_INPUT={source}", flush=True)


def forbidden_log_signatures(log: str) -> list[str]:
    """Return stable acceptance labels for forbidden trainer log content."""

    return [
        name for name, pattern in FORBIDDEN_LOG_PATTERNS.items() if pattern.search(log)
    ]


def run_trainer(command: list[str], cwd: Path, environment: dict[str, str]) -> list[str]:
    """Stream trainer output and reject the four forbidden remote signatures."""

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
    log = "".join(lines)
    violations = forbidden_log_signatures(log)
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
    if not isinstance(rows, list):
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


def main() -> None:
    describe_tree(RAW_INPUT)
    bundle = locate_expanded_bundle()
    materialize_bundle(bundle)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    inventory = gpu_inventory()
    if len(inventory) != 2 or not all("Tesla T4" in row for row in inventory):
        raise SystemExit(f"expected T4 x2, found {inventory}")
    print("KAGGLE_T4_INVENTORY=" + json.dumps(inventory), flush=True)

    shared_task = WORKSPACE / "shared-task"
    command = [
        sys.executable,
        "scripts/train_encoder_baseline.py",
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
        "2",
        "--fold-seed",
        "20260710",
        "--seed",
        "20260710",
        "--limit-paragraphs",
        "20",
        "--limit-strategy",
        "longest-text",
        "--epochs",
        "1",
        "--max-steps",
        "1",
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
        "--device",
        "cuda",
        "--precision",
        "fp16",
        "--attention-implementation",
        "eager",
        "--pad-to-multiple-of",
        "8",
    ]
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
    print("DALEEL_PREFLIGHT_COMMAND=" + json.dumps(command), flush=True)
    log_violations = run_trainer(command, shared_task, environment)

    run_dirs = sorted(
        (shared_task / "experiments").glob(
            "*-encoder-t1-camelbert-msa-quarter-*-smoke-*"
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(run_dirs) != 1:
        raise SystemExit(f"expected exactly one encoder experiment, found {run_dirs}")
    run_dir = run_dirs[0]
    for name in SAFE_ARTIFACTS:
        if not (run_dir / name).is_file():
            raise SystemExit(f"required safe artifact missing: {run_dir / name}")
    validate_completion(run_dir)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    runtime = config["runtime"]
    arguments = config["arguments"]
    training_metrics = metrics["training"]
    argument_mismatches = {
        key: {"expected": expected, "actual": arguments.get(key)}
        for key, expected in EXPECTED_ARGUMENTS.items()
        if arguments.get(key) != expected
    }
    checks = {
        "gpu_inventory": inventory,
        "gpu_count": len(inventory),
        "run_directory": run_dir.name,
        "script_version": config["script_version"],
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"]["revision"],
        "resolved_commit": config["resolved_commit"],
        "tokenizer_repository": config["tokenizer"]["repository"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "tokenizer_is_fast": config["tokenizer"]["is_fast"],
        "argument_mismatches": argument_mismatches,
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
        "freeze_encoder": arguments["freeze_encoder"],
        "selected_paragraph_count": config["selected_paragraph_count"],
        "limit_strategy": arguments["limit_strategy"],
        "effective_batch_size": (
            arguments["batch_size"] * arguments["gradient_accumulation"]
        ),
        "fold_count": metrics["fold_count"],
        "optimizer_step_attempts_by_fold": [
            row["optimizer_step_attempts"] for row in history
        ],
        "optimizer_steps_by_fold": [row["optimizer_steps"] for row in history],
        "skipped_optimizer_steps_by_fold": [
            row["skipped_optimizer_steps"] for row in history
        ],
        "scheduler_steps_by_fold": [row["scheduler_steps"] for row in history],
        "training_seconds_by_fold": [row["training_seconds"] for row in history],
        "training_loss_by_fold": [
            row["epochs"][-1]["mean_training_loss"] for row in history
        ],
        "validation_loss_by_fold": [row["validation_loss"] for row in history],
        "max_cuda_memory_allocated_bytes": training_metrics[
            "max_cuda_memory_allocated_bytes"
        ],
        "max_cuda_memory_reserved_bytes": training_metrics[
            "max_cuda_memory_reserved_bytes"
        ],
        "max_reserved_limit_bytes": MAX_RESERVED_BYTES,
        "trainer_log_forbidden_signatures": log_violations,
        "primary": metrics["primary"],
        "completion_validated": True,
        "completed": True,
    }

    if checks["script_version"] != EXPECTED_SCRIPT_VERSION:
        raise SystemExit(f"stale runner reached preflight: {checks}")
    if argument_mismatches:
        raise SystemExit(f"preflight argument contract drifted: {checks}")
    if any(
        checks[key] != expected
        for key, expected in (
            ("model_repository", EXPECTED_MODEL_REPOSITORY),
            ("model_revision", EXPECTED_MODEL_REVISION),
            ("resolved_commit", EXPECTED_MODEL_REVISION),
            ("tokenizer_repository", EXPECTED_MODEL_REPOSITORY),
            ("tokenizer_revision", EXPECTED_MODEL_REVISION),
        )
    ) or not checks["tokenizer_is_fast"]:
        raise SystemExit(f"pinned model/tokenizer contract failed: {checks}")
    if (
        checks["device"] != "cuda"
        or not checks["cuda_available"]
        or checks["cuda_device_count"] != 2
        or checks["cuda_device_index"] != 0
        or "T4" not in (checks["cuda_device_name"] or "")
    ):
        raise SystemExit(f"runner did not use cuda:0 on T4 x2: {checks}")
    if (
        checks["precision"] != "fp16"
        or checks["attention_implementation"] != "eager"
        or not checks["deterministic_algorithms"]
        or checks["deterministic_warn_only"] is not False
        or checks["cublas_workspace_config"] != ":4096:8"
    ):
        raise SystemExit(f"runner did not use strict deterministic eager FP16: {checks}")
    if checks["freeze_encoder"]:
        raise SystemExit(f"encoder was unexpectedly frozen: {checks}")
    if (
        checks["selected_paragraph_count"] != 20
        or checks["limit_strategy"] != "longest-text"
        or checks["effective_batch_size"] != 8
        or checks["fold_count"] != 2
        or len(history) != 2
    ):
        raise SystemExit(f"long-text/fold/effective-batch contract failed: {checks}")
    if checks["optimizer_steps_by_fold"] != [1, 1]:
        raise SystemExit(f"successful optimizer-step contract failed: {checks}")
    if checks["scheduler_steps_by_fold"] != checks["optimizer_steps_by_fold"]:
        raise SystemExit(f"optimizer/scheduler step parity failed: {checks}")
    if any(
        value < 0 or value > 1
        for value in checks["skipped_optimizer_steps_by_fold"]
    ):
        raise SystemExit(f"too many FP16 overflow skips: {checks}")
    for attempts, completed, skipped in zip(
        checks["optimizer_step_attempts_by_fold"],
        checks["optimizer_steps_by_fold"],
        checks["skipped_optimizer_steps_by_fold"],
    ):
        if attempts != completed + skipped:
            raise SystemExit(f"optimizer-step accounting identity failed: {checks}")
    for row in history:
        final_epoch = row["epochs"][-1]
        if (
            final_epoch["optimizer_step_attempts_total"]
            != row["optimizer_step_attempts"]
            or final_epoch["optimizer_steps_total"] != row["optimizer_steps"]
            or final_epoch["skipped_optimizer_steps_total"]
            != row["skipped_optimizer_steps"]
            or final_epoch["scheduler_steps_total"] != row["scheduler_steps"]
        ):
            raise SystemExit(f"fold/epoch accounting disagrees: {checks}")
    if (
        training_metrics["optimizer_step_attempts"]
        != sum(checks["optimizer_step_attempts_by_fold"])
        or training_metrics["optimizer_steps"]
        != sum(checks["optimizer_steps_by_fold"])
        or training_metrics["skipped_optimizer_steps"]
        != sum(checks["skipped_optimizer_steps_by_fold"])
        or training_metrics["scheduler_steps"]
        != sum(checks["scheduler_steps_by_fold"])
    ):
        raise SystemExit(f"fold/metric accounting disagrees: {checks}")
    if not all(
        math.isfinite(value)
        for value in checks["training_loss_by_fold"]
        + checks["validation_loss_by_fold"]
    ):
        raise SystemExit(f"non-finite training or validation loss: {checks}")
    if not math.isfinite(checks["primary"]["score"]):
        raise SystemExit(f"non-finite primary score: {checks}")
    per_fold_allocated = [row["max_cuda_memory_allocated_bytes"] for row in history]
    per_fold_reserved = [row["max_cuda_memory_reserved_bytes"] for row in history]
    if (
        checks["max_cuda_memory_allocated_bytes"] != max(per_fold_allocated)
        or checks["max_cuda_memory_reserved_bytes"] != max(per_fold_reserved)
        or any(
            allocated > reserved
            for allocated, reserved in zip(per_fold_allocated, per_fold_reserved)
        )
    ):
        raise SystemExit(f"CUDA memory accounting failed: {checks}")
    if checks["max_cuda_memory_reserved_bytes"] > MAX_RESERVED_BYTES:
        raise SystemExit(f"CUDA memory headroom contract failed: {checks}")
    if config["comparable_run"] or metrics["comparable_run"]:
        raise SystemExit(f"smoke was incorrectly marked comparable: {checks}")

    for name in SAFE_ARTIFACTS:
        shutil.copy2(run_dir / name, OUTPUT / name)
    (OUTPUT / "preflight_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    exported = {path.name for path in OUTPUT.iterdir() if path.is_file()}
    expected_exported = set(SAFE_ARTIFACTS) | {"preflight_checks.json"}
    if exported != expected_exported or any(path.is_dir() for path in OUTPUT.iterdir()):
        raise SystemExit(f"unsafe or incomplete artifact export: {sorted(exported)}")
    print(
        "DALEEL_CAMELBERT_PREFLIGHT_RESULT="
        + json.dumps(checks, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
