"""Compose the adopted tier-1 v2 routed Task 1 system on an external input.

HEADROOM_AUDIT.md P0 deployment: the same frozen v2 rule table that passed
the OOF adoption gate (v2 0.7102 vs deployed v1 0.6721 + 0.02, 5/5 fold
wins), applied to a target input.

- AS/TE/ST fire from the 5-rollout vote fraction with deployment thresholds
  = the MEDIAN of the five per-fold cross-fitted thetas (calibration now
  matches what the OOF gate estimated; v1 fit on all 430 non-cross-fitted);
- with --st-judge, vote-fired ST is dropped where the zero-shot judge says
  absent (run judge_stage_test.py --drop-labels ST on the pass-1 output);
- AN fires iff deployed-encoder fires OR (the v3 span-derived set fires AND
  vote count >= 1);
- OT fires iff >= 2 of {encoder, span-derived, votes >= 2};
- CO fires on the top round(2 * 0.0588 * n_target) paragraphs by the
  rank-mean of (deployed encoder CO sigmoid, CO vote fraction, span-derived
  CO presence).

Emits a submission-ready task_1.jsonl bound to the target source.

Example (from shared-task/):
  uv run scripts/route_task1_v2_deploy.py \
      --train-rollouts experiments/<train-r0> ... experiments/<train-r4> \
      --encoder-oof experiments/<quarter-oof-run> \
      --dev-rollouts experiments/<dev-r0> ... experiments/<dev-r4> \
      --encoder-deploy experiments/<t1-encoder-deploy-...> \
      --dev-span experiments/<v3-decomposed-dev-run> \
      --input ../resources/repos/Daleel2026/data/dev/dev_in.jsonl \
      --source-id daleel2026:dev-input
"""

import argparse
import json
import statistics
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

from route_task1 import CO_PRIOR, fit_vote_threshold, load_preds  # noqa: E402
from route_task1_v2 import (  # noqa: E402 — sibling script import
    CO_BUDGET_MULT,
    VOTE_SOURCED,
    co_rank_fusion,
    load_st_judge,
)


def load_target_preds(exp_dir: Path) -> dict[int, set[str]]:
    rows = read_jsonl(exp_dir / "predictions" / "task_1.jsonl")
    return {r["paragraph_id"]: set(r["labels"]) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--encoder-oof", type=Path, required=True,
                    help="the quarter OOF run (fold partition for the thetas)")
    ap.add_argument("--dev-rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--encoder-deploy", type=Path, required=True)
    ap.add_argument("--dev-span", type=Path, required=True,
                    help="v3 decomposed run on the target (predictions/task_2.jsonl)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--st-judge", type=Path, default=None,
                    help="optional jsonl of {paragraph_id, present} ST decisions")
    ap.add_argument("--ot-source", choices=("consensus", "encoder"), default="consensus",
                    help="encoder = partial revert to the externally validated v1 OT "
                    "leg (dev 0.7344 vs consensus 0.6338, measured 2026-07-14)")
    args = ap.parse_args()
    if len(args.train_rollouts) != len(args.dev_rollouts):
        raise SystemExit("train and dev rollout counts must match")
    n_rollouts = len(args.dev_rollouts)

    # ---- deployment thetas: median of the five per-fold cross-fitted values
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
    fold_of = {
        r["paragraph_id"]: r["fold"]
        for r in read_jsonl(args.encoder_oof / "predictions" / "oof_task_1_scores.jsonl")
    }
    folds = sorted(set(fold_of.values()))
    thetas = {}
    thetas_by_fold = {}
    for f in folds:
        fit_pids = [pid for pid in train_pids if fold_of[pid] != f]
        thetas_by_fold[f] = {
            label: fit_vote_threshold(fit_pids, train_votes, gold, label, n_rollouts)
            for label in VOTE_SOURCED
        }
    for label in VOTE_SOURCED:
        thetas[label] = int(
            statistics.median(thetas_by_fold[f][label] for f in folds)
        )

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
    span = {
        r["paragraph_id"]: {s["label"] for s in r["labels"]}
        for r in read_jsonl(args.dev_span / "predictions" / "task_2.jsonl")
    }
    st_judge = load_st_judge(args.st_judge) if args.st_judge else None
    for name, mapping in (
        *((f"dev-rollout-{i}", p) for i, p in enumerate(dev_rollout_preds)),
        ("encoder-deploy", enc_preds),
        ("encoder-scores", enc_scores),
        ("dev-span", span),
    ):
        if not set(target_ids) <= set(mapping):
            raise SystemExit(f"{name} does not cover the target input")

    votes = {
        pid: {label: sum(label in p[pid] for p in dev_rollout_preds) for label in LABELS}
        for pid in target_ids
    }
    k_co = round(CO_BUDGET_MULT * CO_PRIOR * len(target_ids))
    ranked = co_rank_fusion(
        list(target_ids), {p: enc_scores[p]["CO"] for p in target_ids}, votes, span
    )
    co_fired = set(ranked[:k_co])

    routed = {}
    for pid in target_ids:
        fired = {label for label in ("AS", "TE") if votes[pid][label] >= thetas[label]}
        if votes[pid]["ST"] >= thetas["ST"] and (
            st_judge is None or st_judge.get(pid, True)
        ):
            fired.add("ST")
        if "AN" in enc_preds[pid] or ("AN" in span[pid] and votes[pid]["AN"] >= 1):
            fired.add("AN")
        if args.ot_source == "encoder":
            if "OT" in enc_preds[pid]:
                fired.add("OT")
        else:
            ot_sources = (
                ("OT" in enc_preds[pid]) + ("OT" in span[pid]) + (votes[pid]["OT"] >= 2)
            )
            if ot_sources >= 2:
                fired.add("OT")
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
        "script": "route_task1_v2_deploy.py",
        "argv": sys.argv[1:],
        "milestone": "HEADROOM_AUDIT.md P0",
        "router_v2": {
            "vote_sourced": ["AS", "TE"],
            "st": "votes ± judge filter",
            "an": "enc | (span & votes>=1)",
            "ot": ("encoder decisions as-is (v2.1 partial revert)"
                   if args.ot_source == "encoder"
                   else "2-of-3(enc, span, votes>=2)"),
            "co": {
                "budget_mult": CO_BUDGET_MULT,
                "prior": CO_PRIOR,
                "ranker": "rank-mean(enc sigmoid, vote fraction, span presence)",
                "k": k_co,
            },
            "n_rollouts": n_rollouts,
            "deployment_thetas": thetas,
            "thetas_by_fold": {str(f): thetas_by_fold[f] for f in folds},
            "st_judge": str(args.st_judge) if args.st_judge else None,
        },
        "inputs": {
            "train_rollouts": [str(d) for d in args.train_rollouts],
            "encoder_oof": str(args.encoder_oof),
            "dev_rollouts": [str(d) for d in args.dev_rollouts],
            "encoder_deploy": str(args.encoder_deploy),
            "dev_span": str(args.dev_span),
            "dev_span_sha256": file_sha256(
                args.dev_span / "predictions" / "task_2.jsonl"
            ),
            "target": {
                "path": str(args.input),
                "sha256": file_sha256(args.input),
                "source_id": args.source_id,
            },
        },
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"t1-routed-v2-deploy-{args.input.stem}", resolved_config
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
        "st_judge_applied": st_judge is not None,
        "st_dropped_by_judge": (
            sum(
                1 for pid in target_ids
                if st_judge is not None
                and votes[pid]["ST"] >= thetas["ST"]
                and not st_judge.get(pid, True)
            )
            if st_judge is not None else 0
        ),
        "predicted_label_counts": label_counts,
        "n_empty_predictions": sum(not routed[pid] for pid in target_ids),
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    predictions_path = exp_dir / "predictions" / "task_1.jsonl"
    print(f"\nrouted v2 predictions: {predictions_path}")
    print(
        "package with: uv run scripts/package_submission.py "
        f"{predictions_path} --source {args.input} --task task_1 --setting <setting>"
    )


if __name__ == "__main__":
    main()
