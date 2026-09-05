"""Tier-1 Task 1 router: per-label routing + self-consistency + CO budget.

Implements TIER1_ROUTING_MILESTONE.md exactly (pre-registered 2026-07-11):

- AS/TE/ST fire from the champion's k-rollout vote fraction, threshold
  cross-fitted per label over the encoder's five folds (ties break toward
  the more conservative, higher threshold);
- AN/OT are the encoder's cross-fitted OOF decisions, as-is;
- CO fires on a prior budget: within each fold, top round(0.059 * n_fold)
  paragraphs by encoder CO sigmoid score (per-fold application keeps each
  ranking inside one model's calibration; the global-ranking variant and
  the LLM-vote ranker are reported as diagnostics only).

Composed cross-fitted OOF macro-F1 is compared against the T=0 champion
run on the same 430 paragraphs; the pre-registered adoption bar is +0.02.

Inputs are experiment directories produced by run_zero_shot.py (`preds.jsonl`)
and train_encoder_baseline.py (`oof_task_1.jsonl`, `oof_task_1_scores.jsonl`).

Example (from shared-task/):
  uv run scripts/route_task1.py \
      --llm-t0 experiments/<t0-run> \
      --rollouts experiments/<r0> experiments/<r1> ... \
      --encoder experiments/<encoder-run>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import task1_macro_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

VOTE_SOURCED = ("AS", "TE", "ST")
ENCODER_SOURCED = ("AN", "OT")
CO_PRIOR = 36 / 612  # full-train gold CO rate; a corpus constant, not fitted

ADOPTION_MARGIN = 0.02


def load_preds(exp_dir: Path) -> dict[int, set[str]]:
    rows = read_jsonl(exp_dir / "predictions" / "preds.jsonl")
    return {r["paragraph_id"]: set(r["labels"]) for r in rows}


def binary_f1(pred: set[int], gold: set[int]) -> float:
    tp = len(pred & gold)
    if not pred or not gold:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r) if p + r else 0.0


def fit_vote_threshold(
    pids: list[int],
    votes: dict[int, dict[str, int]],
    gold: dict[int, set[str]],
    label: str,
    n_rollouts: int,
) -> int:
    gold_pos = {pid for pid in pids if label in gold[pid]}
    best_theta, best_f1 = n_rollouts, -1.0
    for theta in range(1, n_rollouts + 1):
        fired = {pid for pid in pids if votes[pid][label] >= theta}
        f1 = binary_f1(fired, gold_pos)
        if f1 > best_f1 or (f1 == best_f1 and theta > best_theta):
            best_theta, best_f1 = theta, f1
    return best_theta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llm-t0", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    args = ap.parse_args()

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {pid: labels for pid, labels in clean_task1_gold(t1, t2).items()
            if pid in set(train_ids)}
    source_by_id = {r["paragraph_id"]: r for r in t1 if r["paragraph_id"] in gold}

    t0_preds = load_preds(args.llm_t0)
    rollout_preds = [load_preds(d) for d in args.rollouts]
    n_rollouts = len(rollout_preds)
    enc_oof = {
        r["paragraph_id"]: set(r["labels"])
        for r in read_jsonl(args.encoder / "predictions" / "oof_task_1.jsonl")
    }
    enc_scores, fold_of = {}, {}
    for r in read_jsonl(args.encoder / "predictions" / "oof_task_1_scores.jsonl"):
        enc_scores[r["paragraph_id"]] = r["scores"]
        fold_of[r["paragraph_id"]] = r["fold"]

    pids = sorted(gold)
    for name, mapping in (
        ("llm-t0", t0_preds),
        *((f"rollout-{i}", p) for i, p in enumerate(rollout_preds)),
        ("encoder-oof", enc_oof),
        ("encoder-scores", enc_scores),
    ):
        if not set(pids) <= set(mapping):
            raise SystemExit(f"{name} does not cover the 430 train-side paragraphs")

    votes: dict[int, dict[str, int]] = {
        pid: {label: sum(label in p[pid] for p in rollout_preds) for label in LABELS}
        for pid in pids
    }

    folds = sorted(set(fold_of.values()))
    fold_pids = {f: [pid for pid in pids if fold_of[pid] == f] for f in folds}

    # ---- cross-fitted vote thresholds for the LLM-sourced labels
    thetas_by_fold: dict[int, dict[str, int]] = {}
    for f in folds:
        fit_pids = [pid for pid in pids if fold_of[pid] != f]
        thetas_by_fold[f] = {
            label: fit_vote_threshold(fit_pids, votes, gold, label, n_rollouts)
            for label in VOTE_SOURCED
        }

    # ---- CO budget: per-fold top-k by encoder CO score (primary)
    co_fired: set[int] = set()
    for f in folds:
        members = fold_pids[f]
        k = round(CO_PRIOR * len(members))
        ranked = sorted(members, key=lambda pid: -enc_scores[pid]["CO"])
        co_fired.update(ranked[:k])
    # diagnostics: global ranking, and the LLM-vote ranker
    k_global = round(CO_PRIOR * len(pids))
    co_global = set(sorted(pids, key=lambda pid: -enc_scores[pid]["CO"])[:k_global])
    co_votes = set(
        sorted(pids, key=lambda pid: (-votes[pid]["CO"], -enc_scores[pid]["CO"]))[
            :k_global
        ]
    )

    def compose(co_set: set[int]) -> dict[int, set[str]]:
        routed = {}
        for pid in pids:
            fired = {
                label
                for label in VOTE_SOURCED
                if votes[pid][label] >= thetas_by_fold[fold_of[pid]][label]
            }
            fired |= {label for label in ENCODER_SOURCED if label in enc_oof[pid]}
            if pid in co_set:
                fired.add("CO")
            routed[pid] = fired
        return routed

    routed = compose(co_fired)

    def score(pred: dict[int, set[str]]) -> dict:
        official = task1_macro_f1(gold, pred)
        return {
            "macro_f1": round(official["macro_f1"], 4),
            "per_label_f1": {
                label: round(v["f1"], 4) for label, v in official["per_label"].items()
            },
        }

    t0_score = score(t0_preds)
    rollout_scores = [score(p)["macro_f1"] for p in rollout_preds]
    routed_score = score(routed)
    gate = round(t0_score["macro_f1"] + ADOPTION_MARGIN, 4)
    adopted = routed_score["macro_f1"] >= gate

    resolved_config = {
        "script": "route_task1.py",
        "argv": sys.argv[1:],
        "milestone": "TIER1_ROUTING_MILESTONE.md",
        "router": {
            "vote_sourced": list(VOTE_SOURCED),
            "encoder_sourced": list(ENCODER_SOURCED),
            "co_prior": CO_PRIOR,
            "n_rollouts": n_rollouts,
        },
        "inputs": {
            "llm_t0": str(args.llm_t0),
            "rollouts": [str(d) for d in args.rollouts],
            "encoder": str(args.encoder),
            "encoder_oof_sha256": file_sha256(
                args.encoder / "predictions" / "oof_task_1.jsonl"
            ),
        },
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"t1-routed-k{n_rollouts}-n{len(pids)}", resolved_config
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)

    metrics = {
        "n_paragraphs": len(pids),
        "champion_t0": t0_score,
        "rollout_macros": [round(v, 4) for v in rollout_scores],
        "routed_composed_oof": routed_score,
        "adoption": {
            "rule": f"routed >= champion_t0 + {ADOPTION_MARGIN}",
            "gate": gate,
            "adopted": adopted,
        },
        "vote_thresholds_by_fold": {
            str(f): thetas_by_fold[f] for f in folds
        },
        "co_budget": {
            "prior": CO_PRIOR,
            "fired_per_fold": {
                str(f): sum(pid in co_fired for pid in fold_pids[f]) for f in folds
            },
            "gold_co_count": sum("CO" in gold[pid] for pid in pids),
            "co_f1_primary_perfold_encoder": round(
                binary_f1(co_fired, {p for p in pids if "CO" in gold[p]}), 4
            ),
            "co_f1_diag_global_encoder": round(
                binary_f1(co_global, {p for p in pids if "CO" in gold[p]}), 4
            ),
            "co_f1_diag_llm_votes": round(
                binary_f1(co_votes, {p for p in pids if "CO" in gold[p]}), 4
            ),
        },
        "diagnostics": {
            "encoder_oof_alone": score(enc_oof),
            "vote_majority_alone": score(
                {
                    pid: {
                        label
                        for label in LABELS
                        if votes[pid][label] >= (n_rollouts // 2 + 1)
                    }
                    for pid in pids
                }
            ),
        },
    }
    atomic_write_json(exp_dir / "metrics.json", metrics)
    write_jsonl(
        exp_dir / "predictions" / "preds.jsonl",
        [
            {
                "paragraph_id": pid,
                "text": source_by_id[pid]["text"],
                "type": source_by_id[pid]["type"],
                "labels": sorted(routed[pid]),
            }
            for pid in pids
        ],
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nmetrics: {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
