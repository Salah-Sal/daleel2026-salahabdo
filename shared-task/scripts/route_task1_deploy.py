"""Compose the adopted tier-1 routed Task 1 system on an external input.

TIER1_ROUTING_MILESTONE.md deployment: the same frozen router table that
passed the OOF adoption gate (routed 0.6721 vs champion 0.6496 + 0.02),
applied to a target input.

- AS/TE/ST fire from the 5-rollout vote fraction with deployment
  thresholds fit per label on ALL 430 train-side OOF votes (the symmetric
  analogue of the encoder's final_thresholds_fit_on_all_oof_scores);
- AN/OT come from the deployed encoder's decisions;
- CO fires on the top round(0.059 * n_target) paragraphs by deployed
  encoder CO sigmoid score.

Emits a submission-ready task_1.jsonl bound to the target source.

Example (from shared-task/):
  uv run scripts/route_task1_deploy.py \
      --train-rollouts experiments/<train-r0> ... experiments/<train-r4> \
      --dev-rollouts experiments/<dev-r0> ... experiments/<dev-r4> \
      --encoder-deploy experiments/<t1-encoder-deploy-...> \
      --input ../resources/repos/Daleel2026/data/dev/dev_in.jsonl \
      --source-id daleel2026:dev-input
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids
from daleel.submission import validate_records_against_source, validate_source_records

from route_task1 import (  # noqa: E402 — sibling script import
    CO_PRIOR,
    ENCODER_SOURCED,
    VOTE_SOURCED,
    fit_vote_threshold,
    load_preds,
)


def load_target_preds(exp_dir: Path) -> dict[int, set[str]]:
    rows = read_jsonl(exp_dir / "predictions" / "task_1.jsonl")
    return {r["paragraph_id"]: set(r["labels"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--dev-rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--encoder-deploy", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source-id", required=True)
    args = ap.parse_args()
    if len(args.train_rollouts) != len(args.dev_rollouts):
        raise SystemExit("train and dev rollout counts must match")
    n_rollouts = len(args.dev_rollouts)

    # ---- deployment vote thresholds: fit on ALL 430 train-side OOF votes
    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {pid: labels for pid, labels in clean_task1_gold(t1, t2).items()
            if pid in set(train_ids)}
    train_votes_preds = [load_preds(d) for d in args.train_rollouts]
    train_pids = sorted(gold)
    train_votes = {
        pid: {label: sum(label in p[pid] for p in train_votes_preds) for label in LABELS}
        for pid in train_pids
    }
    thetas = {
        label: fit_vote_threshold(train_pids, train_votes, gold, label, n_rollouts)
        for label in VOTE_SOURCED
    }

    # ---- target-side signals
    target_records = read_jsonl(args.input)
    problems = validate_source_records(target_records)
    if problems:
        raise SystemExit("invalid target input:\n- " + "\n- ".join(problems[:20]))
    target_ids = [r["paragraph_id"] for r in target_records]
    dev_rollout_preds = [load_target_preds(d) for d in args.dev_rollouts]
    enc_preds = {
        r["paragraph_id"]: set(r["labels"])
        for r in read_jsonl(args.encoder_deploy / "predictions" / "task_1_encoder.jsonl")
    }
    enc_scores = {
        r["paragraph_id"]: r["scores"]
        for r in read_jsonl(args.encoder_deploy / "predictions" / "scores.jsonl")
    }
    for name, mapping in (
        *((f"dev-rollout-{i}", p) for i, p in enumerate(dev_rollout_preds)),
        ("encoder-deploy", enc_preds),
        ("encoder-scores", enc_scores),
    ):
        if not set(target_ids) <= set(mapping):
            raise SystemExit(f"{name} does not cover the target input")

    votes = {
        pid: {label: sum(label in p[pid] for p in dev_rollout_preds) for label in LABELS}
        for pid in target_ids
    }
    k_co = round(CO_PRIOR * len(target_ids))
    co_fired = set(sorted(target_ids, key=lambda pid: -enc_scores[pid]["CO"])[:k_co])

    routed = {}
    for pid in target_ids:
        fired = {label for label in VOTE_SOURCED if votes[pid][label] >= thetas[label]}
        fired |= {label for label in ENCODER_SOURCED if label in enc_preds[pid]}
        if pid in co_fired:
            fired.add("CO")
        routed[pid] = fired

    source_by_id = {r["paragraph_id"]: r for r in target_records}
    out_records = [
        {
            "paragraph_id": pid,
            "text": source_by_id[pid]["text"],
            "type": source_by_id[pid]["type"],
            "labels": sorted(routed[pid]),
        }
        for pid in target_ids
    ]
    problems = validate_records_against_source(out_records, target_records, "task_1")
    if problems:
        raise SystemExit("routed output preflight failed:\n- " + "\n- ".join(problems[:20]))

    resolved_config = {
        "script": "route_task1_deploy.py",
        "argv": sys.argv[1:],
        "milestone": "TIER1_ROUTING_MILESTONE.md",
        "router": {
            "vote_sourced": list(VOTE_SOURCED),
            "encoder_sourced": list(ENCODER_SOURCED),
            "co_prior": CO_PRIOR,
            "n_rollouts": n_rollouts,
            "deployment_thetas": thetas,
            "co_k": k_co,
        },
        "inputs": {
            "train_rollouts": [str(d) for d in args.train_rollouts],
            "dev_rollouts": [str(d) for d in args.dev_rollouts],
            "encoder_deploy": str(args.encoder_deploy),
            "target": {
                "path": str(args.input),
                "sha256": file_sha256(args.input),
                "source_id": args.source_id,
            },
        },
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"t1-routed-deploy-{args.input.stem}", resolved_config
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    write_jsonl(exp_dir / "predictions" / "task_1.jsonl", out_records)
    label_counts = {
        label: int(sum(label in routed[pid] for pid in target_ids)) for label in LABELS
    }
    metrics = {
        "n_target_paragraphs": len(target_ids),
        "deployment_thetas": thetas,
        "co_k": k_co,
        "predicted_label_counts": label_counts,
        "n_empty_predictions": sum(not routed[pid] for pid in target_ids),
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    predictions_path = exp_dir / "predictions" / "task_1.jsonl"
    print(f"\nrouted predictions: {predictions_path}")
    print(
        "package with: uv run scripts/package_submission.py "
        f"{predictions_path} --source {args.input} --task task_1 --setting <setting>"
    )


if __name__ == "__main__":
    main()
