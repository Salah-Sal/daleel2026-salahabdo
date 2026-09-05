"""Deploy the Task 1 encoder: train on all 430 train-side paragraphs, predict an external input.

TIER1_ROUTING_MILESTONE.md deployment path: the AN/OT router legs and the
CO-budget ranking need encoder decisions on the target input. This script
reuses the verified runner's exact machinery (train_fold) as one synthetic
"fold" whose validation set IS the target input (dummy zero labels — the
validation loss is meaningless and recorded as such; only the sigmoid
scores matter). Label decisions use `final_thresholds_fit_on_all_oof_
scores` from the local five-fold OOF run, exactly as pre-registered.

Hyperparameters are pinned to the campaign contract and asserted against
the OOF run's config so drift fails loudly.

Example (from shared-task/):
  uv run scripts/deploy_encoder_task1.py \
      --oof-run experiments/<local-encoder-oof-run> \
      --input ../resources/repos/Daleel2026/data/dev/dev_in.jsonl \
      --source-id daleel2026:dev-input
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.encoder_baseline import (
    build_paragraph_examples,
    score_records,
    task1_prediction_map,
    task1_records,
)
from daleel.folds import Fold
from daleel.io import read_jsonl, write_jsonl
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids
from daleel.submission import validate_source_records

from train_encoder_baseline import (  # noqa: E402 — sibling script import
    parse_args as runner_parse_args,
    resolve_device,
    resolve_model,
    resolve_precision,
    train_fold,
)

CAMPAIGN_ARGV = [
    "--task", "1",
    "--model", "{model}",  # substituted from the OOF run's own config
    "--track", "closed",
    "--pool", "legacy-train",
    "--setting", "both",
    "--folds", "5",
    "--epochs", "4",
    "--batch-size", "4",
    "--gradient-accumulation", "2",
    "--learning-rate", "2e-5",
    "--weight-decay", "0.01",
    "--max-length", "512",
    "--stride", "128",
    "--window-pool", "max",
    "--class-weighting", "none",
    "--precision", "auto",
    "--attention-implementation", "eager",
    "--device", "cpu",
]
CONTRACT_KEYS = (
    "task", "model", "epochs", "batch_size", "gradient_accumulation",
    "learning_rate", "weight_decay", "max_length", "stride", "window_pool",
    "class_weighting", "attention_implementation", "seed", "fold_seed",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof-run", type=Path, required=True,
                    help="local five-fold OOF experiment dir (thresholds + contract)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--seed-from-oof-run", action="store_true",
                    help="seed-ensemble deploys: train with the OOF run's own "
                    "seed instead of asserting it equals the campaign default "
                    "(every other contract field still asserted)")
    args = ap.parse_args()

    oof_config = json.loads((args.oof_run / "config.json").read_text())
    oof_metrics = json.loads((args.oof_run / "metrics.json").read_text())
    thresholds = oof_metrics["diagnostics"]["final_thresholds_fit_on_all_oof_scores"]

    argv = [
        oof_config["arguments"]["model"] if token == "{model}" else token
        for token in CAMPAIGN_ARGV
    ]
    run_args = runner_parse_args(argv)
    contract_keys = CONTRACT_KEYS
    if args.seed_from_oof_run:
        contract_keys = tuple(k for k in CONTRACT_KEYS if k != "seed")
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

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = clean_task1_gold(t1, t2)
    target_records = read_jsonl(args.input)
    problems = validate_source_records(target_records)
    if problems:
        raise SystemExit("invalid target input:\n- " + "\n- ".join(problems[:20]))
    target_ids = [r["paragraph_id"] for r in target_records]

    train_examples = build_paragraph_examples(t1, gold, train_ids)
    dummy_gold = {pid: set() for pid in target_ids}
    target_examples = build_paragraph_examples(target_records, dummy_gold, target_ids)

    config = AutoConfig.from_pretrained(spec.repository, revision=spec.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repository, revision=spec.revision, use_fast=True
    )

    fold = Fold(index=0, train_ids=tuple(train_ids), val_ids=tuple(target_ids))
    result = train_fold(
        1, fold, train_examples, target_examples, tokenizer,
        config, spec, run_args, device, autocast_dtype,
    )

    threshold_vector = np.asarray([thresholds[label] for label in LABELS])
    decisions = result.scores >= threshold_vector
    prediction_map = task1_prediction_map(result.paragraph_ids, decisions)
    source_by_id = {r["paragraph_id"]: r for r in target_records}

    resolved_config = {
        "script": "deploy_encoder_task1.py",
        "argv": sys.argv[1:],
        "milestone": "TIER1_ROUTING_MILESTONE.md",
        "campaign_argv": argv,
        "precision": resolved_precision,
        "model": asdict(spec),
        "train_ids_count": len(train_ids),
        "thresholds": thresholds,
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
        f"t1-encoder-deploy-{spec.key}-{args.input.stem}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    write_jsonl(
        exp_dir / "predictions" / "task_1_encoder.jsonl",
        task1_records(source_by_id, prediction_map, target_ids),
    )
    write_jsonl(
        exp_dir / "predictions" / "scores.jsonl",
        score_records(result.paragraph_ids, result.scores),
    )
    label_counts = {
        label: int(sum(label in prediction_map[pid] for pid in target_ids))
        for label in LABELS
    }
    metrics = {
        "n_target_paragraphs": len(target_ids),
        "training_seconds": round(result.training_seconds, 1),
        "optimizer_steps": result.optimizer_steps,
        "skipped_optimizer_steps": result.skipped_optimizer_steps,
        "validation_loss_note": "dummy zero labels — loss is meaningless by design",
        "predicted_label_counts": label_counts,
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"\nencoder deploy predictions: {exp_dir / 'predictions' / 'task_1_encoder.jsonl'}")
    print(f"encoder deploy scores:      {exp_dir / 'predictions' / 'scores.jsonl'}")


if __name__ == "__main__":
    main()
