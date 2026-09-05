"""Deploy the Task 2 segment encoder and score an external input.

S1 deploy path (CREATIVE_HEADROOM_RESEARCH.md): the structural re-decode
needs encoder segment distributions on the target input. Mirrors
deploy_encoder_task1.py: reuses the verified runner's train_fold as one
synthetic "fold" whose validation set IS the target input (dummy empty
gold — validation loss is meaningless and recorded as such; only the
segment softmax scores matter). Hyperparameters are pinned to the local
OOF run's contract and asserted against its config so drift fails loudly.

Example (from shared-task/):
  uv run scripts/deploy_encoder_task2.py \
      --oof-run experiments/<local-t2-encoder-oof-run> \
      --input ../resources/repos/Daleel2026/data/dev/dev_in.jsonl \
      --source-id daleel2026:dev-input

Evaluation-week fit using every official training paragraph and the released
development references::

  uv run scripts/deploy_encoder_task2.py \
      --oof-run experiments/<local-t2-encoder-oof-run> \
      --fit-pool all \
      --extra-train-task2 ../resources/repos/Daleel2026/data/dev/dev_task_2_ref.jsonl \
      --input ../resources/repos/Daleel2026/data/test/test_in.jsonl \
      --source-id daleel2026:test-input
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoConfig, AutoTokenizer

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.encoder_baseline import SEGMENT_LABELS, build_segment_examples
from daleel.folds import Fold
from daleel.io import read_jsonl, write_jsonl
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold, train_val_ids
from daleel.submission import validate_source_records

from train_encoder_baseline import (  # noqa: E402 — sibling script import
    parse_args as runner_parse_args,
    resolve_device,
    resolve_model,
    resolve_precision,
    train_fold,
)

CAMPAIGN_ARGV = [
    "--task", "2",
    "--model", "{model}",  # substituted from the OOF run's own config
    "--track", "closed",
    "--pool", "{pool}",  # substituted from the OOF run's own config
    "--setting", "both",
    "--folds", "5",
    "--fold-seed", "20260710",
    "--seed", "20260710",
    "--epochs", "4",
    "--batch-size", "16",
    "--learning-rate", "2e-5",
    "--class-weighting", "none",
    "--stride", "128",
    "--granularity", "connective",
    "--device", "cpu",
    "--precision", "fp32",
]
CONTRACT_KEYS = (
    "task", "model", "epochs", "batch_size", "gradient_accumulation",
    "learning_rate", "weight_decay", "max_length", "stride", "granularity",
    "class_weighting", "seed", "fold_seed", "pool",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof-run", type=Path, required=True,
                    help="local five-fold T2 encoder OOF experiment dir (contract)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--seed-from-oof-run", action="store_true",
                    help="seed-ensemble deploys: train with the OOF run's own "
                    "seed instead of asserting it equals the campaign default "
                    "(every other contract field remains asserted)")
    ap.add_argument(
        "--fit-pool",
        choices=("contract", "all"),
        default="contract",
        help=(
            "training IDs for the target deploy: reproduce the OOF run's pool "
            "or use every paragraph in the official training file"
        ),
    )
    ap.add_argument(
        "--extra-train-task2",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional organizer-provided labeled Task 2 JSONL to append to "
            "the fit pool (repeatable; intended for released dev references)"
        ),
    )
    args = ap.parse_args()

    oof_config = json.loads((args.oof_run / "config.json").read_text())

    substitutions = {
        "{model}": oof_config["arguments"]["model"],
        "{pool}": oof_config["arguments"]["pool"],
    }
    argv = [substitutions.get(token, token) for token in CAMPAIGN_ARGV]
    run_args = runner_parse_args(argv)
    contract_keys = CONTRACT_KEYS
    if args.seed_from_oof_run:
        contract_keys = tuple(key for key in CONTRACT_KEYS if key != "seed")
        run_args.seed = oof_config["arguments"]["seed"]
    drift = {
        key: (vars(run_args)[key], oof_config["arguments"][key])
        for key in contract_keys
        if vars(run_args)[key] != oof_config["arguments"][key]
    }
    if drift:
        raise SystemExit(f"deploy args drift from the OOF run contract: {drift}")

    spec = resolve_model(run_args)
    device = resolve_device(run_args.device)
    resolved_precision, autocast_dtype = resolve_precision(run_args.precision, device)

    t1, official_t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    if args.fit_pool == "all" or run_args.pool == "all":
        train_ids = sorted(row["paragraph_id"] for row in official_t2)
    else:
        train_ids, _ = train_val_ids(t1, official_t2)

    extra_sources = []
    t2 = list(official_t2)
    seen_train_ids = {row["paragraph_id"] for row in t2}
    for path in args.extra_train_task2:
        rows = load_records(path)
        problems = validate_source_records(rows)
        if problems:
            raise SystemExit(
                f"invalid extra training source {path}:\n- "
                + "\n- ".join(problems[:20])
            )
        unlabeled = [row["paragraph_id"] for row in rows if "labels" not in row]
        if unlabeled:
            raise SystemExit(
                f"extra training source {path} has unlabeled paragraphs: {unlabeled[:5]}"
            )
        duplicate_ids = sorted(
            row["paragraph_id"] for row in rows if row["paragraph_id"] in seen_train_ids
        )
        if duplicate_ids:
            raise SystemExit(
                f"extra training source {path} overlaps existing IDs: {duplicate_ids[:5]}"
            )
        ids = [row["paragraph_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"extra training source {path} contains duplicate IDs")
        seen_train_ids.update(ids)
        train_ids.extend(ids)
        t2.extend(rows)
        extra_sources.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "record_count": len(rows),
            }
        )
    train_ids = sorted(train_ids)
    gold_task2 = clean_task2_gold(t2)
    target_records = read_jsonl(args.input)
    problems = validate_source_records(target_records)
    if problems:
        raise SystemExit("invalid target input:\n- " + "\n- ".join(problems[:20]))
    target_ids = [r["paragraph_id"] for r in target_records]

    t2_by_id = {row["paragraph_id"]: row for row in t2}
    train_examples = build_segment_examples(
        [t2_by_id[pid] for pid in train_ids], gold_task2, train_ids,
        granularity=run_args.granularity,
    )
    dummy_gold = {pid: [] for pid in target_ids}
    target_examples = build_segment_examples(
        target_records, dummy_gold, target_ids,
        granularity=run_args.granularity,
    )

    config = AutoConfig.from_pretrained(spec.repository, revision=spec.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repository, revision=spec.revision, use_fast=True
    )

    fold = Fold(index=0, train_ids=tuple(train_ids), val_ids=tuple(target_ids))
    result = train_fold(
        2, fold, train_examples, target_examples, tokenizer,
        config, spec, run_args, device, autocast_dtype,
    )
    if result.segment_examples is None or result.scores is None:
        raise SystemExit("train_fold returned no Task 2 segment outputs")
    if len(result.segment_examples) != len(result.scores):
        raise SystemExit("Task 2 segment/score length mismatch")

    score_rows = []
    for example, predicted_index, row_scores in zip(
        result.segment_examples, result.predicted_indices, result.scores
    ):
        score_rows.append(
            {
                "paragraph_id": example.paragraph_id,
                "start_offset": example.start,
                "end_offset": example.end,
                "predicted_label": SEGMENT_LABELS[int(predicted_index)],
                "scores": {
                    label: float(row_scores[index])
                    for index, label in enumerate(SEGMENT_LABELS)
                },
            }
        )

    resolved_config = {
        "script": "deploy_encoder_task2.py",
        "argv": sys.argv[1:],
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md S1 deploy",
        "campaign_argv": argv,
        "precision": resolved_precision,
        "model": asdict(spec),
        "fit_pool": args.fit_pool,
        "train_ids_count": len(train_ids),
        "training_sources": [
            {
                "path": str(TRAIN_TASK2),
                "sha256": file_sha256(TRAIN_TASK2),
                "record_count": len(official_t2),
                "selected_record_count": (
                    len(official_t2)
                    if args.fit_pool == "all" or run_args.pool == "all"
                    else len(train_ids) - sum(source["record_count"] for source in extra_sources)
                ),
            },
            *extra_sources,
        ],
        "oof_run": str(args.oof_run),
        "oof_run_sha256": file_sha256(args.oof_run / "metrics.json"),
        "input": {
            "path": str(args.input),
            "sha256": file_sha256(args.input),
            "source_id": args.source_id,
            "record_count": len(target_records),
        },
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t2-encoder-deploy-{spec.key}-{args.input.stem}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    write_jsonl(
        exp_dir / "predictions" / "task_2_segment_scores.jsonl", score_rows
    )
    from collections import Counter
    predicted_counts = Counter(row["predicted_label"] for row in score_rows)
    metrics = {
        "n_target_paragraphs": len(target_ids),
        "n_target_segments": len(score_rows),
        "training_seconds": round(result.training_seconds, 1),
        "optimizer_steps": result.optimizer_steps,
        "skipped_optimizer_steps": result.skipped_optimizer_steps,
        "validation_loss_note": "dummy empty gold — loss is meaningless by design",
        "predicted_segment_class_counts": dict(sorted(predicted_counts.items())),
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"\nsegment scores: {exp_dir / 'predictions' / 'task_2_segment_scores.jsonl'}")


if __name__ == "__main__":
    main()
