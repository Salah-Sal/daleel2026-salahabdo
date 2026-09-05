#!/usr/bin/env python3
"""Build private Kaggle dataset staging and durable encoder-kernel metadata.

The command only prepares local files. It never reads Kaggle credentials or
uploads anything. The dataset whitelist deliberately excludes predictions,
experiments, secrets, caches, and unrelated repository content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_TASK = REPO_ROOT / "shared-task"
OFFICIAL_TRAIN = REPO_ROOT / "resources" / "repos" / "Daleel2026" / "data" / "train"
KERNEL_SOURCE = Path(__file__).resolve().parent / "kernel" / "encoder_preflight.py"
DEFAULT_OUT = Path("/private/tmp/daleel-encoder-preflight-upload")
DATASET_SLUG = "daleel-encoder-preflight-bundle-v1"
KERNEL_SLUG = "daleel-camelbert-t4-preflight"
PACKAGE_FILES = (
    "__init__.py",
    "artifacts.py",
    "constants.py",
    "data.py",
    "encoder_baseline.py",
    "folds.py",
    "io.py",
    "metrics.py",
    "provenance.py",
    "segment.py",
    "splits.py",
    "submission.py",
)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not cleaned:
        raise ValueError("Kaggle owner cannot be empty")
    return cleaned


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_dataset(out: Path, owner: str) -> Path:
    # Keep this exact layout in sync with the documented manual
    # ``kaggle datasets version -p`` handoff.
    dataset = out / DATASET_SLUG
    reset_dir(dataset)
    copy_required(
        SHARED_TASK / "scripts" / "train_encoder_baseline.py",
        dataset / "shared-task" / "scripts" / "train_encoder_baseline.py",
    )
    for name in PACKAGE_FILES:
        copy_required(
            SHARED_TASK / "src" / "daleel" / name,
            dataset / "shared-task" / "src" / "daleel" / name,
        )
    for name in ("train_task_1.jsonl", "train_task_2.jsonl"):
        copy_required(
            OFFICIAL_TRAIN / name,
            dataset / "resources" / "repos" / "Daleel2026" / "data" / "train" / name,
        )

    write_json(
        dataset / "dataset-metadata.json",
        {
            "title": "Daleel Encoder Preflight Bundle v1",
            "subtitle": "Private source and official train files for the Daleel encoder GPU preflight",
            "description": (
                "Private minimal bundle containing the Daleel encoder runner, "
                "required local modules, and organizer-provided training JSONL files. "
                "Not for redistribution."
            ),
            "id": f"{owner}/{DATASET_SLUG}",
            "licenses": [{"name": "unknown"}],
            "keywords": ["arabic"],
        },
    )
    manifest = {
        path.relative_to(dataset).as_posix(): sha256(path)
        for path in sorted(dataset.rglob("*"))
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    write_json(dataset / "bundle-manifest.json", manifest)
    return dataset


def prepare_kernel(kernel: Path, owner: str) -> Path:
    if not (kernel / KERNEL_SOURCE.name).is_file():
        raise FileNotFoundError(kernel / KERNEL_SOURCE.name)
    write_json(
        kernel / "kernel-metadata.json",
        {
            "id": f"{owner}/{KERNEL_SLUG}",
            "title": "Daleel CAMeLBERT T4 Preflight",
            "code_file": KERNEL_SOURCE.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,
            "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    return kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="salah1992")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=KERNEL_SOURCE.parent,
        help="durable kernel directory that receives kernel-metadata.json",
    )
    args = parser.parse_args()
    owner = slug(args.owner)
    dataset = prepare_dataset(args.out, owner)
    kernel = prepare_kernel(args.kernel_dir, owner)
    print(f"dataset={dataset}")
    print(f"kernel={kernel}")
    print(
        "dataset-version-command="
        f".venv_kaggle/bin/kaggle datasets version -p {dataset} "
        "--dir-mode zip "
        '-m "Fix FP16 step accounting and add strict deterministic T4 preflight"'
    )
    print(f"kernel-push-command=.venv_kaggle/bin/kaggle kernels push -p {kernel}")


if __name__ == "__main__":
    main()
