"""Train and evaluate DSPy-free Arabic encoder baselines with strict OOF CV.

The runner implements the first two practical non-generative systems from
the non-LLM baseline survey:

* Task 1: six-logit paragraph classification with overflow windows and
  cross-fitted per-label thresholds.
* Task 2: seven-way classification of exact deterministic connective
  segments (the six Daleel roles plus NONE).

Every headline score is computed once over concatenated paragraph-grouped
out-of-fold predictions.  The script uses no DSPy module, prompt, API, or
external labeled data.  Large prediction files remain under the existing
gitignored experiments/*/predictions/ convention.

Full Task 1 example (run from shared-task/):

  .venv/bin/python scripts/train_encoder_baseline.py --task 1 \
      --model camelbert-msa-quarter --epochs 4 --folds 5

Bounded CPU plumbing smoke:

  .venv/bin/python scripts/train_encoder_baseline.py --task 1 \
      --model camelbert-msa --folds 2 --limit-paragraphs 8 \
      --epochs 1 --max-steps 1 --freeze-encoder --max-length 64 --offline
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from daleel.artifacts import (
    atomic_write_json,
    create_experiment_dir,
    file_sha256,
    output_record,
    write_completion_marker,
)
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.encoder_baseline import (
    NONE_LABEL,
    SEGMENT_LABELS,
    ParagraphCollator,
    ParagraphDataset,
    ParagraphEncoderClassifier,
    ParagraphExample,
    SegmentCollator,
    SegmentDataset,
    SegmentEncoderClassifier,
    SegmentExample,
    build_paragraph_examples,
    build_segment_examples,
    cross_fitted_task1_predictions,
    score_records,
    segment_prediction_map,
    task1_prediction_map,
    task1_records,
    task2_records,
    tune_task1_thresholds,
)
from daleel.folds import (
    DEFAULT_FOLD_SEED,
    Fold,
    fold_manifest,
    fold_manifest_hash,
    make_stratified_folds,
)
from daleel.io import write_jsonl
from daleel.metrics import Span, span_partial_f1, task1_macro_f1
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids
from daleel.submission import validate_records_against_source


SCRIPT_VERSION = 3
ARCHITECTURES = {
    1: "arabic-encoder-multilabel-window-v1",
    2: "arabic-encoder-connective-segment-v1",
}
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Pinned public checkpoint metadata used by the reproducible defaults."""

    key: str
    repository: str
    revision: str
    license_id: str | None
    license_url: str | None
    closed_track_verified: bool


ENCODER_SPECS = {
    spec.key: spec
    for spec in (
        EncoderSpec(
            key="camelbert-msa-quarter",
            repository="CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter",
            revision="3e48534705c153737cbec1c5748bb02359b7b239",
            license_id="apache-2.0",
            license_url=(
                "https://huggingface.co/"
                "CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter"
            ),
            closed_track_verified=True,
        ),
        EncoderSpec(
            key="camelbert-msa",
            repository="CAMeL-Lab/bert-base-arabic-camelbert-msa",
            revision="277069fd3645fedb22b746caf38d111aadee0241",
            license_id="apache-2.0",
            license_url=(
                "https://huggingface.co/"
                "CAMeL-Lab/bert-base-arabic-camelbert-msa"
            ),
            closed_track_verified=True,
        ),
        EncoderSpec(
            key="camelbert-mix",
            repository="CAMeL-Lab/bert-base-arabic-camelbert-mix",
            revision="9be352797bdf28a9ae21e2ae582aaaca7abdb22d",
            license_id="apache-2.0",
            license_url=(
                "https://huggingface.co/"
                "CAMeL-Lab/bert-base-arabic-camelbert-mix"
            ),
            closed_track_verified=True,
        ),
        # Dialectal-Arabic pretraining (54GB DA) — theory match for the
        # debate genre where the routed AN leg has its recall hole.
        EncoderSpec(
            key="camelbert-da",
            repository="CAMeL-Lab/bert-base-arabic-camelbert-da",
            revision="231698eab9ebf0ae7b518a64277b81b2fe829f2d",
            license_id="apache-2.0",
            license_url=(
                "https://huggingface.co/"
                "CAMeL-Lab/bert-base-arabic-camelbert-da"
            ),
            closed_track_verified=True,
        ),
        # The checkpoint is useful for research, but its current model card
        # has no explicit license tag.  Refuse to self-certify it as Closed.
        # Same verdict for UBC-NLP MARBERTv2/ARBERT (checked 2026-07-11):
        # no license tag on the model card despite public weights.
        EncoderSpec(
            key="arabert-v02",
            repository="aubmindlab/bert-base-arabertv02",
            revision="016fb9d6768f522a59c6e0d2d5d5d43a4e1bff60",
            license_id=None,
            license_url=None,
            closed_track_verified=False,
        ),
        # Local TAPT checkpoint: camelbert-msa-quarter continued-pretrained
        # (MLM) on organizer-provided input text (tapt_pretrain.py, run
        # 20260715-202537-...-transductive-829, registration cd77061).
        # revision = `git hash-object model.safetensors` (content hash).
        # closed_track_verified attests LICENSE ONLY: the weights derive from
        # the apache-2.0 base checkpoint + organizer text (no external data,
        # no labels) — see the run's provenance.json. It does NOT authorize
        # deployment: transductive use of provided inputs is an open organizer
        # question, so any run off this spec is OOF-measurement-only until a
        # written organizer ruling exists (enforced by the gate, not here).
        EncoderSpec(
            key="camelbert-msa-quarter-tapt",
            repository=str(MODELS_DIR / "camelbert-msa-quarter-tapt-transductive-v1"),
            revision="1577e68b0fa6226ea37b3b3c3cfc4f4b2db2bfb4",
            license_id="apache-2.0",
            license_url=(
                "https://huggingface.co/"
                "CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter"
            ),
            closed_track_verified=True,
        ),
    )
}


@dataclass(slots=True)
class FoldResult:
    fold: int
    train_count: int
    validation_count: int
    train_item_count: int
    validation_item_count: int
    optimizer_step_attempts: int
    optimizer_steps: int
    skipped_optimizer_steps: int
    scheduler_steps: int
    training_seconds: float
    validation_loss: float
    max_cuda_memory_allocated_bytes: int | None
    max_cuda_memory_reserved_bytes: int | None
    class_weights: list[float] | None
    epoch_history: list[dict[str, Any]]
    paragraph_ids: list[int]
    scores: np.ndarray
    predicted_indices: np.ndarray | None = None
    segment_examples: list[SegmentExample] | None = None


@dataclass(slots=True)
class OptimizerStepAccounting:
    """Counts attempted, successful, skipped, and scheduled training updates."""

    optimizer_step_attempts: int = 0
    optimizer_steps: int = 0
    skipped_optimizer_steps: int = 0
    scheduler_steps: int = 0


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _json_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _probability(raw: str) -> float:
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must lie in [0, 1]")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--model",
        default="camelbert-msa-quarter",
        help=(
            "registered alias (camelbert-msa-quarter, camelbert-msa, "
            "camelbert-mix, arabert-v02) "
            "or a Hugging Face repository"
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="immutable 40-hex model commit; aliases supply a pinned default",
    )
    parser.add_argument("--license-id", default=None, help="custom-model SPDX/license ID")
    parser.add_argument("--license-url", default=None, help="custom-model license evidence URL")
    parser.add_argument("--track", choices=("closed", "open"), default="closed")
    parser.add_argument("--offline", action="store_true", help="load only cached model files")

    parser.add_argument("--pool", choices=("legacy-train", "all"), default="legacy-train")
    parser.add_argument("--setting", choices=("both", "editorial", "debate"), default="both")
    parser.add_argument("--folds", type=_positive_int, default=5)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--limit-paragraphs",
        type=_positive_int,
        default=None,
        help="deterministic smoke-only paragraph subset; never a comparable score",
    )
    parser.add_argument(
        "--limit-strategy",
        choices=("random", "longest-text"),
        default="random",
        help="smoke subset selection; longest-text stress-tests overflow memory",
    )

    parser.add_argument("--epochs", type=_positive_int, default=4)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--gradient-accumulation", type=_positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=_probability, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=None,
        help="successful optimizer-update cap per fold (smoke/debug only)",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--class-weighting",
        choices=("none", "sqrt", "balanced"),
        default="none",
        help="Task 1 positive weights or Task 2 class weights; none is primary",
    )
    parser.add_argument("--max-class-weight", type=float, default=20.0)

    parser.add_argument("--max-length", type=_positive_int, default=512)
    parser.add_argument("--stride", type=_non_negative_int, default=128)
    parser.add_argument("--window-pool", choices=("max", "mean"), default="max")
    parser.add_argument("--context-chars", type=_non_negative_int, default=0)
    parser.add_argument(
        "--granularity",
        choices=("sentence", "clause", "connective"),
        default="connective",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=_non_negative_int,
        default=8,
        help="0 disables multiple padding",
    )
    parser.add_argument("--num-workers", type=_non_negative_int, default=0)
    parser.add_argument("--cpu-threads", type=_positive_int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="eager",
        help=(
            "eager enforces strict deterministic algorithms; sdpa permits "
            "faster seeded-but-not-bitwise-deterministic attention"
        ),
    )

    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2 for OOF evaluation")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0, 1)")
    if args.max_class_weight < 1.0:
        parser.error("--max-class-weight must be at least 1")
    if args.gradient_checkpointing and args.freeze_encoder:
        parser.error("--gradient-checkpointing is meaningless with --freeze-encoder")
    if args.limit_strategy != "random" and args.limit_paragraphs is None:
        parser.error("--limit-strategy applies only with --limit-paragraphs")
    if args.task == 2 and args.stride != 128:
        parser.error("--stride applies only to Task 1; leave it at its default for Task 2")
    if args.task == 1 and args.context_chars:
        parser.error("--context-chars applies only to Task 2")
    args.batch_size = args.batch_size or (4 if args.task == 1 else 16)
    return args


def resolve_model(args: argparse.Namespace) -> EncoderSpec:
    registered = ENCODER_SPECS.get(args.model)
    if registered is not None:
        revision = args.revision or registered.revision
        spec = EncoderSpec(
            key=registered.key,
            repository=registered.repository,
            revision=revision,
            license_id=args.license_id or registered.license_id,
            license_url=args.license_url or registered.license_url,
            closed_track_verified=(
                registered.closed_track_verified
                and (args.license_id is None or args.license_id == registered.license_id)
                and (args.license_url is None or args.license_url == registered.license_url)
            ),
        )
    else:
        if not args.revision:
            raise SystemExit("custom --model repositories require --revision")
        spec = EncoderSpec(
            key=args.model.replace("/", "--"),
            repository=args.model,
            revision=args.revision,
            license_id=args.license_id,
            license_url=args.license_url,
            closed_track_verified=False,
        )

    if args.track == "closed":
        if not COMMIT_RE.fullmatch(spec.revision):
            raise SystemExit("closed-track model revisions must be immutable 40-hex commits")
        if not spec.closed_track_verified:
            raise SystemExit(
                f"model {args.model!r} has no runner-verified closed-track license; "
                "use --track open or add a reviewed registry entry"
            )
    return spec


def seed_everything(seed: int, *, strict_determinism: bool) -> None:
    # CUDA matrix multiplications require one of PyTorch's documented cuBLAS
    # workspace settings when deterministic algorithms are enforced.  Set it
    # before any CUDA RNG or model operation can initialize a cuBLAS handle.
    if strict_determinism:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        True,
        warn_only=not strict_determinism,
    )
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested, but MPS is unavailable")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    resolved = requested
    if resolved == "auto":
        resolved = "fp16" if device.type == "cuda" else "fp32"
    if resolved == "fp16" and device.type != "cuda":
        raise SystemExit("fp16 training is supported only on CUDA by this runner")
    if resolved == "bf16" and device.type not in {"cuda", "cpu"}:
        raise SystemExit("bf16 training is supported only on CUDA/CPU by this runner")
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[resolved]
    return resolved, dtype


def select_pool_ids(
    task1_records_source: Sequence[dict[str, Any]],
    task2_records_source: Sequence[dict[str, Any]],
    *,
    pool: str,
    setting: str,
    limit: int | None,
    limit_strategy: str,
    seed: int,
) -> list[int]:
    by_id = {row["paragraph_id"]: row for row in task1_records_source}
    if len(by_id) != len(task1_records_source):
        raise ValueError("Task 1 source contains duplicate paragraph IDs")
    if pool == "legacy-train":
        ids = train_val_ids(list(task1_records_source), list(task2_records_source))[0]
    elif pool == "all":
        ids = sorted(by_id)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unknown pool {pool!r}")
    ids = [pid for pid in ids if setting == "both" or by_id[pid]["type"] == setting]
    if limit is not None and limit < len(ids):
        if limit_strategy == "random":
            ids = sorted(random.Random(seed).sample(ids, limit))
        elif limit_strategy == "longest-text":
            ids = sorted(
                sorted(ids, key=lambda pid: (-len(by_id[pid]["text"]), pid))[:limit]
            )
        else:  # pragma: no cover - argparse prevents this
            raise ValueError(f"unknown limit strategy {limit_strategy!r}")
    return ids


def _move_inputs(
    inputs: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: tensor.to(device, non_blocking=device.type == "cuda") for key, tensor in inputs.items()}


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def _class_weights(
    task: int,
    examples: Sequence[ParagraphExample] | Sequence[SegmentExample],
    mode: str,
    maximum: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if task == 1:
        matrix = np.asarray([example.labels for example in examples], dtype=float)
        positives = matrix.sum(axis=0)
        negatives = len(matrix) - positives
        if np.any(positives == 0):
            missing = [LABELS[index] for index in np.flatnonzero(positives == 0)]
            raise ValueError(f"cannot weight Task 1 labels with no positives: {missing}")
        weights = negatives / positives
    else:
        counts = np.bincount(
            [example.label_index for example in examples],
            minlength=len(SEGMENT_LABELS),
        ).astype(float)
        if np.any(counts == 0):
            missing = [SEGMENT_LABELS[index] for index in np.flatnonzero(counts == 0)]
            raise ValueError(f"cannot weight Task 2 classes with no examples: {missing}")
        weights = len(examples) / (len(SEGMENT_LABELS) * counts)
        weights /= weights.mean()
    if mode == "sqrt":
        weights = np.sqrt(weights)
    weights = np.clip(weights, 1.0 / maximum, maximum)
    return torch.tensor(weights, dtype=torch.float32)


def _make_model(
    task: int,
    spec: EncoderSpec,
    config: Any,
    args: argparse.Namespace,
) -> nn.Module:
    encoder = AutoModel.from_pretrained(
        spec.repository,
        revision=spec.revision,
        local_files_only=args.offline,
        attn_implementation=args.attention_implementation,
    )
    if args.freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    elif args.gradient_checkpointing:
        if not hasattr(encoder, "gradient_checkpointing_enable"):
            raise ValueError("selected encoder does not support gradient checkpointing")
        encoder.gradient_checkpointing_enable()
    if task == 1:
        return ParagraphEncoderClassifier(
            encoder,
            hidden_size=config.hidden_size,
            dropout=args.dropout,
            window_pool=args.window_pool,
        )
    return SegmentEncoderClassifier(
        encoder,
        hidden_size=config.hidden_size,
        dropout=args.dropout,
    )


def _make_collator(task: int, tokenizer: Any, args: argparse.Namespace):
    multiple = args.pad_to_multiple_of or None
    if task == 1:
        return ParagraphCollator(
            tokenizer,
            max_length=args.max_length,
            stride=args.stride,
            pad_to_multiple_of=multiple,
        )
    return SegmentCollator(
        tokenizer,
        max_length=args.max_length,
        context_chars=args.context_chars,
        pad_to_multiple_of=multiple,
    )


def _make_loader(
    task: int,
    examples: Sequence[ParagraphExample] | Sequence[SegmentExample],
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = ParagraphDataset(examples) if task == 1 else SegmentDataset(examples)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=_make_collator(task, tokenizer, args),
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers),
    )


def _criterion(
    task: int,
    weights: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    if task == 1:
        return nn.BCEWithLogitsLoss(
            pos_weight=None if weights is None else weights.to(device)
        )
    return nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))


def _scaled_optimizer_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> bool:
    """Run a scaled step and advance the scheduler only after a real update.

    GradScaler silently skips ``optimizer.step()`` when it finds non-finite
    gradients.  An optimizer post-step hook is the public, unambiguous signal
    that AdamW actually updated parameters; scale changes are not used as a
    proxy.
    """

    executed = False

    def mark_executed(*_: Any, **__: Any) -> None:
        nonlocal executed
        executed = True

    hook = optimizer.register_step_post_hook(mark_executed)
    try:
        scaler.step(optimizer)
    finally:
        hook.remove()
    scaler.update()
    if executed:
        scheduler.step()
    return executed


def _attempt_scaled_optimizer_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accounting: OptimizerStepAccounting,
) -> bool:
    """Attempt one AMP update, account for its outcome, and clear gradients.

    Clearing lives in a ``finally`` block so an attempted update can never
    leak its gradients into a later batch, including when the scaler or
    scheduler raises and the fold is about to abort.
    """

    accounting.optimizer_step_attempts += 1
    try:
        executed = _scaled_optimizer_step(scaler, optimizer, scheduler)
        if executed:
            accounting.optimizer_steps += 1
            accounting.scheduler_steps += 1
        else:
            accounting.skipped_optimizer_steps += 1
        return executed
    finally:
        optimizer.zero_grad(set_to_none=True)


def _require_requested_optimizer_steps(
    *,
    fold_index: int,
    requested: int | None,
    completed: int,
    attempts: int,
    skipped: int,
) -> None:
    if requested is not None and completed < requested:
        raise RuntimeError(
            f"fold {fold_index}: requested {requested} successful optimizer "
            f"steps, completed {completed} after {attempts} attempts "
            f"({skipped} skipped)"
        )


def _forward(task: int, model: nn.Module, batch: dict[str, Any], device: torch.device):
    inputs = _move_inputs(batch["inputs"], device)
    if task == 1:
        return model(
            inputs,
            batch["window_to_paragraph"].to(device),
            n_paragraphs=len(batch["paragraph_ids"]),
        )
    return model(inputs)


def _evaluate(
    task: int,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[float, list[int], np.ndarray, np.ndarray | None, list[SegmentExample] | None]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    paragraph_ids: list[int] = []
    scores: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    segment_examples: list[SegmentExample] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            with _autocast(device, autocast_dtype):
                logits = _forward(task, model, batch, device)
                loss = criterion(logits, labels)
            count = len(labels)
            total_loss += float(loss.detach().cpu()) * count
            total_examples += count
            if task == 1:
                paragraph_ids.extend(batch["paragraph_ids"])
                scores.append(torch.sigmoid(logits).float().cpu().numpy())
            else:
                probabilities = torch.softmax(logits, dim=-1).float().cpu().numpy()
                scores.append(probabilities)
                indices.append(np.argmax(probabilities, axis=1))
                segment_examples.extend(batch["examples"])
    if total_examples == 0:
        raise ValueError("validation loader produced zero examples")
    score_matrix = np.concatenate(scores, axis=0)
    predicted = np.concatenate(indices, axis=0) if indices else None
    return (
        total_loss / total_examples,
        paragraph_ids,
        score_matrix,
        predicted,
        segment_examples if task == 2 else None,
    )


def train_fold(
    task: int,
    fold: Fold,
    train_examples: Sequence[ParagraphExample] | Sequence[SegmentExample],
    validation_examples: Sequence[ParagraphExample] | Sequence[SegmentExample],
    tokenizer: Any,
    config: Any,
    spec: EncoderSpec,
    args: argparse.Namespace,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> FoldResult:
    fold_seed = args.seed + fold.index
    seed_everything(
        fold_seed,
        strict_determinism=args.attention_implementation == "eager",
    )
    model = _make_model(task, spec, config, args).to(device)
    train_loader = _make_loader(
        task, train_examples, tokenizer, args, shuffle=True, seed=fold_seed
    )
    validation_loader = _make_loader(
        task,
        validation_examples,
        tokenizer,
        args,
        shuffle=False,
        seed=fold_seed,
    )
    weights = _class_weights(
        task, train_examples, args.class_weighting, args.max_class_weight
    )
    criterion = _criterion(task, weights, device)
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    planned_updates = updates_per_epoch * args.epochs
    if args.max_steps is not None:
        planned_updates = min(planned_updates, args.max_steps)
    warmup_steps = min(planned_updates, round(planned_updates * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_updates,
    )
    use_scaler = autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    accounting = OptimizerStepAccounting()
    epoch_history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch_index in range(args.epochs):
        model.train()
        loss_sum = 0.0
        example_count = 0
        accumulated = 0
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            with _autocast(device, autocast_dtype):
                logits = _forward(task, model, batch, device)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            count = len(labels)
            loss_sum += float(raw_loss.detach().cpu()) * count
            example_count += count
            accumulated += 1
            is_boundary = accumulated == args.gradient_accumulation
            is_last_batch = batch_index + 1 == len(train_loader)
            if not is_boundary and not is_last_batch:
                continue

            scaler.unscale_(optimizer)
            if accumulated != args.gradient_accumulation:
                correction = args.gradient_accumulation / accumulated
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            _attempt_scaled_optimizer_step(
                scaler,
                optimizer,
                scheduler,
                accounting,
            )
            accumulated = 0
            if (
                args.max_steps is not None
                and accounting.optimizer_steps >= args.max_steps
            ):
                break

        epoch_row = {
            "epoch": epoch_index + 1,
            "mean_training_loss": loss_sum / max(1, example_count),
            "examples_seen": example_count,
            "optimizer_step_attempts_total": accounting.optimizer_step_attempts,
            "optimizer_steps_total": accounting.optimizer_steps,
            "skipped_optimizer_steps_total": accounting.skipped_optimizer_steps,
            "scheduler_steps_total": accounting.scheduler_steps,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "seconds": time.monotonic() - epoch_started,
        }
        epoch_history.append(epoch_row)
        print(
            f"fold {fold.index + 1}/{args.folds} epoch {epoch_index + 1}/"
            f"{args.epochs}: loss={epoch_row['mean_training_loss']:.6f}, "
            f"steps={accounting.optimizer_steps}, "
            f"attempts={accounting.optimizer_step_attempts}, "
            f"skipped={accounting.skipped_optimizer_steps}",
            flush=True,
        )
        if (
            args.max_steps is not None
            and accounting.optimizer_steps >= args.max_steps
        ):
            break

    _require_requested_optimizer_steps(
        fold_index=fold.index,
        requested=args.max_steps,
        completed=accounting.optimizer_steps,
        attempts=accounting.optimizer_step_attempts,
        skipped=accounting.skipped_optimizer_steps,
    )

    validation_loss, paragraph_ids, scores, predicted, segment_examples = _evaluate(
        task,
        model,
        validation_loader,
        criterion,
        device,
        autocast_dtype,
    )
    elapsed = time.monotonic() - started
    max_cuda_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    max_cuda_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    model.to("cpu")
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return FoldResult(
        fold=fold.index,
        train_count=len(fold.train_ids),
        validation_count=len(fold.val_ids),
        train_item_count=len(train_examples),
        validation_item_count=len(validation_examples),
        optimizer_step_attempts=accounting.optimizer_step_attempts,
        optimizer_steps=accounting.optimizer_steps,
        skipped_optimizer_steps=accounting.skipped_optimizer_steps,
        scheduler_steps=accounting.scheduler_steps,
        training_seconds=elapsed,
        validation_loss=validation_loss,
        max_cuda_memory_allocated_bytes=max_cuda_allocated,
        max_cuda_memory_reserved_bytes=max_cuda_reserved,
        class_weights=None if weights is None else [float(value) for value in weights],
        epoch_history=epoch_history,
        paragraph_ids=paragraph_ids,
        scores=scores,
        predicted_indices=predicted,
        segment_examples=segment_examples,
    )


def _subset_mapping(mapping: Mapping[int, Any], ids: Sequence[int]) -> dict[int, Any]:
    return {paragraph_id: mapping[paragraph_id] for paragraph_id in ids}


def _task1_genre_metrics(
    source_by_id: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, set[str]],
    predictions: Mapping[int, set[str]],
    paragraph_ids: Sequence[int],
) -> dict[str, Any]:
    result = {}
    for genre in ("editorial", "debate"):
        ids = [pid for pid in paragraph_ids if source_by_id[pid]["type"] == genre]
        if ids:
            result[genre] = task1_macro_f1(
                _subset_mapping(gold, ids), _subset_mapping(predictions, ids)
            )
    return result


def _task2_genre_metrics(
    source_by_id: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, Sequence[Span]],
    predictions: Mapping[int, Sequence[Span]],
    paragraph_ids: Sequence[int],
) -> dict[str, Any]:
    result = {}
    for genre in ("editorial", "debate"):
        ids = [pid for pid in paragraph_ids if source_by_id[pid]["type"] == genre]
        if ids:
            result[genre] = span_partial_f1(
                _subset_mapping(gold, ids), _subset_mapping(predictions, ids)
            )
    return result


def aggregate_task1(
    fold_results: Sequence[FoldResult],
    examples: Sequence[ParagraphExample],
    folds: Sequence[Fold],
    source_by_id: Mapping[int, Mapping[str, Any]],
    gold: Mapping[int, set[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    score_by_id: dict[int, np.ndarray] = {}
    for result in fold_results:
        if len(result.paragraph_ids) != len(result.scores):
            raise ValueError(f"fold {result.fold}: Task 1 ID/score length mismatch")
        for paragraph_id, scores in zip(result.paragraph_ids, result.scores):
            if paragraph_id in score_by_id:
                raise ValueError(f"duplicate OOF Task 1 paragraph {paragraph_id}")
            score_by_id[paragraph_id] = scores

    paragraph_ids = [example.paragraph_id for example in examples]
    if set(score_by_id) != set(paragraph_ids):
        raise ValueError("Task 1 OOF scores do not cover the selected pool exactly")
    scores = np.stack([score_by_id[pid] for pid in paragraph_ids])
    gold_matrix = np.asarray([example.labels for example in examples], dtype=bool)
    fold_by_id = {
        paragraph_id: fold.index for fold in folds for paragraph_id in fold.val_ids
    }
    fold_indices = [fold_by_id[pid] for pid in paragraph_ids]

    cross_fitted, thresholds_by_fold = cross_fitted_task1_predictions(
        gold_matrix, scores, fold_indices
    )
    full_thresholds, full_threshold_f1 = tune_task1_thresholds(gold_matrix, scores)
    apparent = scores >= full_thresholds
    default = scores >= 0.5
    cross_map = task1_prediction_map(paragraph_ids, cross_fitted)
    apparent_map = task1_prediction_map(paragraph_ids, apparent)
    default_map = task1_prediction_map(paragraph_ids, default)
    official = task1_macro_f1(_subset_mapping(gold, paragraph_ids), cross_map)
    metrics = {
        "primary": {
            "name": "official_task1_macro_f1_cross_fitted_oof_thresholds",
            "score": official["macro_f1"],
        },
        "official": official,
        "per_genre": _task1_genre_metrics(
            source_by_id, gold, cross_map, paragraph_ids
        ),
        "diagnostics": {
            "default_0_5": task1_macro_f1(
                _subset_mapping(gold, paragraph_ids), default_map
            ),
            "same_oof_threshold_fit_optimistic": task1_macro_f1(
                _subset_mapping(gold, paragraph_ids), apparent_map
            ),
            "final_thresholds_fit_on_all_oof_scores": {
                label: float(threshold)
                for label, threshold in zip(LABELS, full_thresholds)
            },
            "threshold_fit_f1_by_label": full_threshold_f1,
            "cross_fitted_thresholds_by_heldout_fold": thresholds_by_fold,
        },
    }
    records = task1_records(source_by_id, cross_map, paragraph_ids)
    score_rows = score_records(paragraph_ids, scores)
    for row, fold_index in zip(score_rows, fold_indices):
        row["fold"] = int(fold_index)
    return metrics, records, score_rows


def aggregate_task2(
    fold_results: Sequence[FoldResult],
    paragraph_ids: Sequence[int],
    source_by_id: Mapping[int, Mapping[str, Any]],
    gold_task1: Mapping[int, set[str]],
    gold_task2: Mapping[int, Sequence[Span]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[SegmentExample] = []
    predicted_indices: list[int] = []
    score_matrices: list[np.ndarray] = []
    fold_indices: list[int] = []
    for result in fold_results:
        if result.segment_examples is None or result.predicted_indices is None:
            raise ValueError(f"fold {result.fold}: missing Task 2 segment outputs")
        if len(result.segment_examples) != len(result.predicted_indices):
            raise ValueError(f"fold {result.fold}: Task 2 output length mismatch")
        examples.extend(result.segment_examples)
        predicted_indices.extend(int(value) for value in result.predicted_indices)
        score_matrices.append(result.scores)
        fold_indices.extend([result.fold] * len(result.segment_examples))
    if {example.paragraph_id for example in examples} != set(paragraph_ids):
        raise ValueError("Task 2 OOF segments do not cover the selected pool exactly")
    scores = np.concatenate(score_matrices, axis=0)
    predictions = segment_prediction_map(examples, predicted_indices, paragraph_ids)
    oracle_indices = [example.label_index for example in examples]
    oracle = segment_prediction_map(examples, oracle_indices, paragraph_ids)
    official = span_partial_f1(
        _subset_mapping(gold_task2, paragraph_ids), predictions
    )
    oracle_metric = span_partial_f1(
        _subset_mapping(gold_task2, paragraph_ids), oracle
    )

    gold_indices = np.asarray(oracle_indices, dtype=int)
    predicted_array = np.asarray(predicted_indices, dtype=int)
    confusion = np.zeros(
        (len(SEGMENT_LABELS), len(SEGMENT_LABELS)), dtype=int
    )
    for gold_index, predicted_index in zip(gold_indices, predicted_array):
        confusion[gold_index, predicted_index] += 1
    task1_from_spans = {
        pid: {span.label for span in predictions[pid]} for pid in paragraph_ids
    }
    metrics = {
        "primary": {
            "name": "official_task2_partial_f1_oof",
            "score": official["f1"],
        },
        "official": official,
        "per_genre": _task2_genre_metrics(
            source_by_id, gold_task2, predictions, paragraph_ids
        ),
        "diagnostics": {
            "segment_accuracy": float(np.mean(gold_indices == predicted_array)),
            "segment_count": len(examples),
            "gold_segment_class_counts": dict(
                sorted(Counter(example.label for example in examples).items())
            ),
            "predicted_segment_class_counts": {
                label: int(np.sum(predicted_array == index))
                for index, label in enumerate(SEGMENT_LABELS)
            },
            "confusion_matrix": {
                "row_labels_gold": list(SEGMENT_LABELS),
                "column_labels_predicted": list(SEGMENT_LABELS),
                "values": confusion.tolist(),
            },
            "deterministic_segmentation_oracle": oracle_metric,
            "task1_derived_from_predicted_spans": task1_macro_f1(
                _subset_mapping(gold_task1, paragraph_ids), task1_from_spans
            ),
        },
    }
    records = task2_records(source_by_id, predictions, paragraph_ids)
    score_rows = []
    for example, fold_index, predicted_index, row_scores in zip(
        examples, fold_indices, predicted_indices, scores
    ):
        score_rows.append(
            {
                "paragraph_id": example.paragraph_id,
                "fold": fold_index,
                "start_offset": example.start,
                "end_offset": example.end,
                "gold_oracle_label": example.label,
                "predicted_label": SEGMENT_LABELS[predicted_index],
                "scores": {
                    label: float(row_scores[index])
                    for index, label in enumerate(SEGMENT_LABELS)
                },
            }
        )
    return metrics, records, score_rows


def _fold_history_rows(results: Sequence[FoldResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "fold": result.fold,
                "train_paragraph_count": result.train_count,
                "validation_paragraph_count": result.validation_count,
                "train_item_count": result.train_item_count,
                "validation_item_count": result.validation_item_count,
                "optimizer_step_attempts": result.optimizer_step_attempts,
                "optimizer_steps": result.optimizer_steps,
                "skipped_optimizer_steps": result.skipped_optimizer_steps,
                "scheduler_steps": result.scheduler_steps,
                "training_seconds": result.training_seconds,
                "validation_loss": result.validation_loss,
                "max_cuda_memory_allocated_bytes": (
                    result.max_cuda_memory_allocated_bytes
                ),
                "max_cuda_memory_reserved_bytes": (
                    result.max_cuda_memory_reserved_bytes
                ),
                "class_weights": result.class_weights,
                "epochs": result.epoch_history,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    spec = resolve_model(args)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    resolved_precision, autocast_dtype = resolve_precision(args.precision, device)
    seed_everything(
        args.seed,
        strict_determinism=args.attention_implementation == "eager",
    )

    task1_source = load_records(TRAIN_TASK1)
    task2_source = load_records(TRAIN_TASK2)
    source_by_id = {row["paragraph_id"]: row for row in task1_source}
    if set(source_by_id) != {row["paragraph_id"] for row in task2_source}:
        raise SystemExit("official Task 1/Task 2 paragraph IDs differ")
    gold_task1 = clean_task1_gold(task1_source, task2_source)
    gold_task2 = clean_task2_gold(task2_source)
    paragraph_ids = select_pool_ids(
        task1_source,
        task2_source,
        pool=args.pool,
        setting=args.setting,
        limit=args.limit_paragraphs,
        limit_strategy=args.limit_strategy,
        seed=args.seed,
    )
    if len(paragraph_ids) < args.folds:
        raise SystemExit(
            f"selected pool has {len(paragraph_ids)} paragraphs, fewer than "
            f"--folds {args.folds}"
        )
    genres = {pid: source_by_id[pid]["type"] for pid in paragraph_ids}
    labels = {pid: gold_task1[pid] for pid in paragraph_ids}
    folds = make_stratified_folds(
        paragraph_ids,
        labels,
        genres,
        n_splits=args.folds,
        seed=args.fold_seed,
    )
    folds_data = fold_manifest(folds, seed=args.fold_seed)

    config = AutoConfig.from_pretrained(
        spec.repository,
        revision=spec.revision,
        local_files_only=args.offline,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repository,
        revision=spec.revision,
        local_files_only=args.offline,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise SystemExit("encoder baseline requires a fast tokenizer")
    model_limit = int(getattr(config, "max_position_embeddings", args.max_length))
    if args.max_length > model_limit:
        raise SystemExit(
            f"--max-length {args.max_length} exceeds model position limit {model_limit}"
        )
    resolved_commit = getattr(config, "_commit_hash", None) or spec.revision
    if args.track == "closed" and resolved_commit != spec.revision:
        raise SystemExit(
            f"resolved model commit {resolved_commit!r} differs from pinned "
            f"revision {spec.revision!r}"
        )

    ordered_source = [source_by_id[pid] for pid in paragraph_ids]
    all_examples: Sequence[ParagraphExample] | Sequence[SegmentExample]
    if args.task == 1:
        all_examples = build_paragraph_examples(
            ordered_source, gold_task1, paragraph_ids
        )
    else:
        task2_by_id = {row["paragraph_id"]: row for row in task2_source}
        ordered_task2 = [task2_by_id[pid] for pid in paragraph_ids]
        all_examples = build_segment_examples(
            ordered_task2,
            gold_task2,
            paragraph_ids,
            granularity=args.granularity,
        )

    resolved_args = vars(args).copy()
    resolved_args["precision"] = resolved_precision
    cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cuda_device_index = torch.cuda.current_device() if device.type == "cuda" else None
    cuda_device_name = (
        torch.cuda.get_device_name(cuda_device_index)
        if cuda_device_index is not None
        else None
    )
    deterministic_warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: None,
    )()
    run_contract = {
        "script_version": SCRIPT_VERSION,
        "task": args.task,
        "architecture": ARCHITECTURES[args.task],
        "model": asdict(spec),
        "resolved_commit": resolved_commit,
        "tokenizer": {
            "repository": spec.repository,
            "revision": spec.revision,
            "is_fast": bool(tokenizer.is_fast),
        },
        "arguments": resolved_args,
        "training_ids_sha256": _json_sha256(paragraph_ids),
        "fold_manifest_sha256": fold_manifest_hash(folds_data),
        "source_sha256": {
            "task1": file_sha256(TRAIN_TASK1),
            "task2": file_sha256(TRAIN_TASK2),
        },
    }
    smoke = args.limit_paragraphs is not None or args.max_steps is not None
    slug = (
        f"encoder-t{args.task}-{spec.key}-{args.setting}-{args.pool}"
        + ("-smoke" if smoke else "")
    )
    run_dir = create_experiment_dir(EXPERIMENTS_DIR, slug, run_contract)
    config_path = atomic_write_json(
        run_dir / "config.json",
        run_contract
        | {
            "script": "train_encoder_baseline.py",
            "comparable_run": not smoke,
            "selected_paragraph_count": len(paragraph_ids),
            "training_item_count": len(all_examples),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": _installed_version("transformers"),
                "numpy": np.__version__,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": cuda_device_count,
                "cuda_device_index": cuda_device_index,
                "cuda_device_name": cuda_device_name,
                "mps_available": torch.backends.mps.is_available(),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "deterministic_warn_only": deterministic_warn_only,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            },
        },
    )
    folds_path = atomic_write_json(run_dir / "folds.json", folds_data)
    pending_provenance = {
        "schema_version": 1,
        "status": "running",
        "tasks": [args.task],
        "track": args.track,
        "setting": args.setting,
        "architecture": ARCHITECTURES[args.task],
        "dspy_used": False,
        "generative_inference_used": False,
        "external_labeled_data_used": False,
        "model": asdict(spec) | {"resolved_commit": resolved_commit},
        "data_sources": [
            {
                "id": "daleel2026:train-task-1",
                "path": str(TRAIN_TASK1),
                "sha256": file_sha256(TRAIN_TASK1),
            },
            {
                "id": "daleel2026:train-task-2",
                "path": str(TRAIN_TASK2),
                "sha256": file_sha256(TRAIN_TASK2),
            },
        ],
        "training_ids": paragraph_ids,
        "training_ids_sha256": _json_sha256(paragraph_ids),
        "fold_manifest_sha256": fold_manifest_hash(folds_data),
        "cleaning_policy": "daleel.splits clean_task1_gold/clean_task2_gold",
        "dependencies": {
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
        },
    }
    pending_path = atomic_write_json(
        run_dir / "provenance.pending.json", pending_provenance
    )

    by_paragraph: dict[int, list[Any]] = {pid: [] for pid in paragraph_ids}
    for example in all_examples:
        by_paragraph[example.paragraph_id].append(example)
    fold_results: list[FoldResult] = []
    print(
        f"run={run_dir.name} task={args.task} paragraphs={len(paragraph_ids)} "
        f"items={len(all_examples)} device={device} precision={resolved_precision}",
        flush=True,
    )
    for fold in folds:
        train_examples = [
            example for pid in fold.train_ids for example in by_paragraph[pid]
        ]
        validation_examples = [
            example for pid in fold.val_ids for example in by_paragraph[pid]
        ]
        print(
            f"starting fold {fold.index + 1}/{len(folds)}: "
            f"train={len(fold.train_ids)} paragraphs/{len(train_examples)} items, "
            f"validation={len(fold.val_ids)} paragraphs/"
            f"{len(validation_examples)} items",
            flush=True,
        )
        fold_results.append(
            train_fold(
                args.task,
                fold,
                train_examples,
                validation_examples,
                tokenizer,
                config,
                spec,
                args,
                device,
                autocast_dtype,
            )
        )

    if args.task == 1:
        metrics, prediction_rows, score_rows = aggregate_task1(
            fold_results,
            all_examples,
            folds,
            source_by_id,
            gold_task1,
        )
        prediction_name = "oof_task_1.jsonl"
        prediction_kind = "encoder-task-1-oof-predictions"
        score_name = "oof_task_1_scores.jsonl"
    else:
        metrics, prediction_rows, score_rows = aggregate_task2(
            fold_results,
            paragraph_ids,
            source_by_id,
            gold_task1,
            gold_task2,
        )
        prediction_name = "oof_task_2.jsonl"
        prediction_kind = "encoder-task-2-oof-predictions"
        score_name = "oof_task_2_segment_scores.jsonl"

    problems = validate_records_against_source(
        prediction_rows,
        ordered_source,
        f"task_{args.task}",
    )
    if problems:
        raise ValueError("invalid OOF output:\n- " + "\n- ".join(problems[:30]))
    predictions_path = run_dir / "predictions" / prediction_name
    scores_path = run_dir / "predictions" / score_name
    write_jsonl(predictions_path, prediction_rows)
    write_jsonl(scores_path, score_rows)
    history_path = atomic_write_json(
        run_dir / "training_history.json", _fold_history_rows(fold_results)
    )
    metrics |= {
        "task": args.task,
        "architecture": ARCHITECTURES[args.task],
        "comparable_run": not smoke,
        "paragraph_count": len(paragraph_ids),
        "training_item_count": len(all_examples),
        "fold_count": len(folds),
        "mean_validation_loss": float(
            np.mean([result.validation_loss for result in fold_results])
        ),
        "total_training_seconds": float(
            sum(result.training_seconds for result in fold_results)
        ),
        "training": {
            "optimizer_step_attempts": int(
                sum(result.optimizer_step_attempts for result in fold_results)
            ),
            "optimizer_steps": int(
                sum(result.optimizer_steps for result in fold_results)
            ),
            "skipped_optimizer_steps": int(
                sum(result.skipped_optimizer_steps for result in fold_results)
            ),
            "scheduler_steps": int(
                sum(result.scheduler_steps for result in fold_results)
            ),
            "max_cuda_memory_allocated_bytes": max(
                (
                    result.max_cuda_memory_allocated_bytes or 0
                    for result in fold_results
                ),
                default=0,
            ),
            "max_cuda_memory_reserved_bytes": max(
                (
                    result.max_cuda_memory_reserved_bytes or 0
                    for result in fold_results
                ),
                default=0,
            ),
        },
    }
    metrics_path = atomic_write_json(run_dir / "metrics.json", metrics)
    provenance = pending_provenance | {
        "status": "complete",
        "outputs": [
            output_record(run_dir, predictions_path, kind=prediction_kind),
            output_record(run_dir, scores_path, kind="encoder-oof-scores"),
            output_record(run_dir, metrics_path, kind="encoder-oof-metrics"),
            output_record(run_dir, history_path, kind="encoder-training-history"),
            output_record(run_dir, folds_path, kind="paragraph-fold-manifest"),
        ],
        "primary_metric": metrics["primary"],
        "comparable_run": not smoke,
    }
    provenance_path = atomic_write_json(run_dir / "provenance.json", provenance)
    pending_path.unlink()
    completion_path = write_completion_marker(
        run_dir,
        [
            config_path,
            folds_path,
            history_path,
            metrics_path,
            predictions_path,
            scores_path,
            provenance_path,
        ],
        metadata={
            "task": args.task,
            "primary_score": metrics["primary"]["score"],
            "comparable_run": not smoke,
        },
    )
    print(json.dumps(metrics["primary"], ensure_ascii=False, indent=2))
    print(f"completed: {completion_path}")
    return run_dir


if __name__ == "__main__":
    main()
