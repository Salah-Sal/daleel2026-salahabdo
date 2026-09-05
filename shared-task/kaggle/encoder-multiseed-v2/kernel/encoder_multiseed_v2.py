"""Multi-seed / multi-model encoder campaign on T4 x2 (HEADROOM_AUDIT.md P5).

Twenty five-fold runs, two GPU workers (one trainer process pinned per T4
via CUDA_VISIBLE_DEVICES), consuming a shared queue:

- Task 1: {quarter, mix, da} x seeds {20260710..20260714}  (15 runs)
- Task 2 target-segment: {quarter, mix, da} x seed 20260710  (3 runs)
- Task 1 class-weighting=sqrt probe: {mix, da} x seed 20260710  (2 runs)

Every run keeps the committed campaign recipe (430-paragraph legacy-train
pool, 5 folds, 4 epochs, lr 2e-5, wd 0.01, fp16, eager, strict
determinism); only model / seed / task / class-weighting vary, and the
FOLD SEED IS NEVER TOUCHED, so every run shares the local fold contract.

Export policy (deliberate, documented deviation from campaign v1): Task 1
runs additionally export `oof_task_1_scores.jsonl` (paragraph_id + six
sigmoid floats + fold index) and a stripped `oof_decisions.jsonl`
(paragraph_id + label codes). Neither contains organizer text or span
offsets; paragraph ids are public in the task repo. These files are what
seed-ensembling consumes locally. Task 2 runs export safe artifacts only.

Failure containment: a crashed job is recorded and the queue continues; the
campaign marker lists per-job status and is `complete` only if every job
validated. A wall-clock guard stops launching new jobs after 9.5 h so the
12 h session always has time to export.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

RAW_INPUT = Path("/kaggle/input")
WORKSPACE = Path("/tmp/daleel-encoder-multiseed-workspace")
OUTPUT = Path("/kaggle/working/encoder_multiseed_v2_artifacts")
DATASET_SLUG = "daleel-encoder-preflight-bundle-v1"
EXPECTED_SCRIPT_VERSION = 3
EXPECTED_PARAGRAPH_COUNT = 430
EXPECTED_FOLD_COUNT = 5
MAX_RESERVED_BYTES = int(13.5 * 1024**3)
LAUNCH_DEADLINE_SECONDS = 9.5 * 3600
SAFE_RUN_ARTIFACTS = (
    "config.json",
    "folds.json",
    "training_history.json",
    "metrics.json",
    "provenance.json",
    "COMPLETED.json",
)
MODELS = {
    "camelbert-msa-quarter": (
        "CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter",
        "3e48534705c153737cbec1c5748bb02359b7b239",
    ),
    "camelbert-mix": (
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        "9be352797bdf28a9ae21e2ae582aaaca7abdb22d",
    ),
    "camelbert-da": (
        "CAMeL-Lab/bert-base-arabic-camelbert-da",
        "231698eab9ebf0ae7b518a64277b81b2fe829f2d",
    ),
}
SEEDS = (20260710, 20260711, 20260712, 20260713, 20260714)


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for model in MODELS:
        for seed in SEEDS:
            jobs.append(
                {
                    "name": f"t1-{model}-seed{seed}",
                    "task": 1,
                    "model": model,
                    "seed": seed,
                    "class_weighting": "none",
                    "batch_size": 4,
                    "gradient_accumulation": 2,
                    "primary_metric": "official_task1_macro_f1_cross_fitted_oof_thresholds",
                }
            )
    for model in MODELS:
        jobs.append(
            {
                "name": f"t2-{model}-seed20260710",
                "task": 2,
                "model": model,
                "seed": 20260710,
                "class_weighting": "none",
                "batch_size": 16,
                "gradient_accumulation": 1,
                "primary_metric": "official_task2_partial_f1_oof",
            }
        )
    for model in ("camelbert-mix", "camelbert-da"):
        jobs.append(
            {
                "name": f"t1-{model}-sqrt-seed20260710",
                "task": 1,
                "model": model,
                "seed": 20260710,
                "class_weighting": "sqrt",
                "batch_size": 4,
                "gradient_accumulation": 2,
                "primary_metric": "official_task1_macro_f1_cross_fitted_oof_thresholds",
            }
        )
    return jobs


def job_command(job: dict) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_encoder_baseline.py",
        "--task", str(job["task"]),
        "--model", job["model"],
        "--track", "closed",
        "--pool", "legacy-train",
        "--setting", "both",
        "--folds", "5",
        "--epochs", "4",
        "--batch-size", str(job["batch_size"]),
        "--gradient-accumulation", str(job["gradient_accumulation"]),
        "--learning-rate", "2e-5",
        "--weight-decay", "0.01",
        "--max-length", "512",
        "--class-weighting", job["class_weighting"],
        "--precision", "auto",
        "--attention-implementation", "eager",
        "--seed", str(job["seed"]),
    ]
    if job["task"] == 1:
        command += ["--stride", "128", "--window-pool", "max"]
    else:
        command += ["--context-chars", "0", "--granularity", "connective"]
    return command


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


def locate_expanded_bundle() -> Path:
    candidates = [RAW_INPUT / DATASET_SLUG, RAW_INPUT]
    candidates.extend(path for path in RAW_INPUT.rglob("*") if path.is_dir())
    for candidate in candidates:
        if (
            (candidate / "bundle-manifest.json").is_file()
            and (candidate / "shared-task/scripts/train_encoder_baseline.py").is_file()
        ):
            return candidate
    raise SystemExit("attached private encoder bundle could not be found")


def validate_bundle_manifest(bundle: Path) -> None:
    manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest.items():
        if relative == "dataset-metadata.json":
            continue
        path = bundle / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise SystemExit(f"bundle manifest hash mismatch: {relative}")


def materialize_bundle(source: Path) -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    shutil.copytree(source / "shared-task", WORKSPACE / "shared-task")
    shutil.copytree(source / "resources", WORKSPACE / "resources")


def gpu_inventory() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def validate_run(job: dict, run_dir: Path) -> dict:
    for name in SAFE_RUN_ARTIFACTS:
        if not (run_dir / name).is_file():
            raise RuntimeError(f"{job['name']}: missing safe artifact {name}")
    marker = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    if marker.get("complete") is not True:
        raise RuntimeError(f"{job['name']}: incomplete completion marker")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    arguments = config["arguments"]
    runtime = config["runtime"]
    training = metrics["training"]
    repo, revision = MODELS[job["model"]]
    problems = []
    if config["script_version"] != EXPECTED_SCRIPT_VERSION:
        problems.append("script_version")
    if (config["model"]["repository"], config["resolved_commit"]) != (repo, revision):
        problems.append("model_pin")
    if arguments["seed"] != job["seed"] or arguments["fold_seed"] != 20260710:
        problems.append("seed_contract")
    if arguments["class_weighting"] != job["class_weighting"]:
        problems.append("class_weighting")
    if arguments["precision"] != "fp16" or arguments["attention_implementation"] != "eager":
        problems.append("precision_attention")
    if not runtime["deterministic_algorithms"] or runtime["deterministic_warn_only"]:
        problems.append("determinism")
    if runtime["device"] != "cuda" or "T4" not in (runtime["cuda_device_name"] or ""):
        problems.append("device")
    if (
        config["selected_paragraph_count"] != EXPECTED_PARAGRAPH_COUNT
        or metrics["fold_count"] != EXPECTED_FOLD_COUNT
    ):
        problems.append("pool_folds")
    if training["max_cuda_memory_reserved_bytes"] > MAX_RESERVED_BYTES:
        problems.append("memory_bar")
    if (
        training["optimizer_step_attempts"]
        != training["optimizer_steps"] + training["skipped_optimizer_steps"]
        or training["scheduler_steps"] != training["optimizer_steps"]
    ):
        problems.append("step_accounting")
    losses = [e["mean_training_loss"] for row in history for e in row["epochs"]]
    losses += [row["validation_loss"] for row in history]
    if not all(math.isfinite(v) for v in losses):
        problems.append("nonfinite_loss")
    if (
        metrics["comparable_run"] is not True
        or provenance["status"] != "complete"
        or provenance["dspy_used"] is not False
        or provenance["generative_inference_used"] is not False
    ):
        problems.append("provenance")
    if problems:
        raise RuntimeError(f"{job['name']}: contract violations: {problems}")
    return {
        "job": job["name"],
        "run_directory": run_dir.name,
        "primary": metrics["primary"],
        "max_cuda_memory_reserved_bytes": training["max_cuda_memory_reserved_bytes"],
        "cuda_device_name": runtime["cuda_device_name"],
        "wall_by_fold_seconds": [row.get("training_seconds") for row in history],
    }


def export_run(job: dict, run_dir: Path) -> None:
    destination = OUTPUT / job["name"]
    destination.mkdir(parents=True, exist_ok=False)
    for name in SAFE_RUN_ARTIFACTS:
        shutil.copy2(run_dir / name, destination / name)
    if job["task"] == 1:
        scores = run_dir / "predictions" / "oof_task_1_scores.jsonl"
        shutil.copy2(scores, destination / "oof_task_1_scores.jsonl")
        stripped = []
        for line in (run_dir / "predictions" / "oof_task_1.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            row = json.loads(line)
            stripped.append(
                {"paragraph_id": row["paragraph_id"], "labels": row["labels"]}
            )
        (destination / "oof_decisions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in stripped) + "\n",
            encoding="utf-8",
        )


def claim_run_dir(job: dict, shared_task: Path, before: set[str]) -> Path:
    # Both workers share experiments/, so a bare before/after diff races with
    # the sibling GPU finishing a job in the same window.  Claim by the job's
    # unique signature instead: task (from the trainer's dir name) plus
    # (model repository, seed, class weighting) read from each candidate's
    # config.json.  No two jobs in one campaign share this signature.
    repo, _ = MODELS[job["model"]]
    after = {p.name for p in (shared_task / "experiments").glob("*")}
    matches = []
    for name in sorted(after - before):
        run_dir = shared_task / "experiments" / name
        if f"-encoder-t{job['task']}-" not in name:
            continue
        if not (run_dir / "COMPLETED.json").is_file():
            continue
        if not (run_dir / "config.json").is_file():
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        arguments = config["arguments"]
        if (
            config["model"]["repository"] == repo
            and arguments["seed"] == job["seed"]
            and arguments["class_weighting"] == job["class_weighting"]
        ):
            matches.append(run_dir)
    if len(matches) != 1:
        raise RuntimeError(
            f"run dir signature match failed for {job['name']}: "
            f"{[m.name for m in matches]} among new dirs {sorted(after - before)}"
        )
    return matches[0]


def worker(
    gpu: int,
    queue: list[dict],
    lock: threading.Lock,
    results: dict,
    shared_task: Path,
    environment: dict,
    started: float,
) -> None:
    while True:
        with lock:
            if not queue:
                return
            if time.time() - started > LAUNCH_DEADLINE_SECONDS:
                for job in queue:
                    results[job["name"]] = {"status": "skipped_deadline"}
                queue.clear()
                return
            job = queue.pop(0)
        print(f"WORKER{gpu} START {job['name']}", flush=True)
        env = dict(environment)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        before = {p.name for p in (shared_task / "experiments").glob("*")}
        try:
            completed = subprocess.run(
                job_command(job),
                cwd=shared_task,
                env=env,
                capture_output=True,
                text=True,
                timeout=3.0 * 3600,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"exit {completed.returncode}: "
                    f"stdout={completed.stdout[-800:]!r} "
                    f"stderr={completed.stderr[-1200:]!r}"
                )
            run_dir = claim_run_dir(job, shared_task, before)
            checks = validate_run(job, run_dir)
            export_run(job, run_dir)
            results[job["name"]] = {"status": "completed", **checks}
            print(
                f"WORKER{gpu} DONE {job['name']} "
                f"primary={checks['primary']['score']:.4f}",
                flush=True,
            )
        except Exception as error:  # containment: record, continue queue
            results[job["name"]] = {"status": "failed", "error": str(error)[:2000]}
            print(f"WORKER{gpu} FAILED {job['name']}: {error}", flush=True)


def main() -> None:
    started = time.time()
    bundle = locate_expanded_bundle()
    validate_bundle_manifest(bundle)
    materialize_bundle(bundle)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    inventory = gpu_inventory()
    print("GPU_INVENTORY=" + json.dumps(inventory), flush=True)
    if len(inventory) != 2 or not all("T4" in row for row in inventory):
        raise SystemExit(f"expected T4 x2, found {inventory}")

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

    jobs = build_jobs()
    queue = list(jobs)
    results: dict[str, dict] = {}
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, queue, lock, results, shared_task, environment, started),
            daemon=True,
        )
        for gpu in (0, 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    completed_jobs = [n for n, r in results.items() if r.get("status") == "completed"]
    files = [
        {"relative_path": p.relative_to(OUTPUT).as_posix(), "sha256": sha256(p)}
        for p in sorted(OUTPUT.rglob("*"))
        if p.is_file() and p.name != "CAMPAIGN_COMPLETED.json"
    ]
    marker = {
        "campaign_completion_schema_version": 2,
        "complete": len(completed_jobs) == len(jobs),
        "jobs_total": len(jobs),
        "jobs_completed": len(completed_jobs),
        "results": results,
        "export_policy": (
            "task1 runs export oof_task_1_scores.jsonl (paragraph_id + sigmoid "
            "floats + fold) and oof_decisions.jsonl (paragraph_id + label codes); "
            "no organizer text or offsets leave the ephemeral workspace"
        ),
        "wall_seconds": round(time.time() - started, 1),
        "files": files,
    }
    write_json(OUTPUT / "CAMPAIGN_COMPLETED.json", marker)
    print("CAMPAIGN_SUMMARY=" + json.dumps(
        {n: r.get("primary", r.get("status")) for n, r in results.items()},
        ensure_ascii=False,
    ), flush=True)


if __name__ == "__main__":
    main()
