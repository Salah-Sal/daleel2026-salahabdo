"""Tier-1 Task 1 router v2: multi-source consensus legs + CO rank fusion.

Implements HEADROOM_AUDIT.md P0 (pre-registered 2026-07-14, rules frozen
from the §A artifact-only measurements before this script existed):

- AS/TE fire from the champion's k-rollout vote fraction, threshold
  cross-fitted per label over the encoder's five folds (unchanged v1 legs);
- ST fires from votes like v1; with --st-judge, vote-fired ST is dropped
  where the twice-validated BFRS ST judge says absent (v2 report +0.032);
- AN fires iff encoder-OOF fires OR (the v3 span-derived label set fires
  AND vote count >= 1)                                  [OOF 0.691 -> 0.724];
- OT fires iff >= 2 of {encoder, span-derived, votes >= 2}
                                                        [OOF 0.774 -> 0.797];
- CO fires on a per-fold budget k = round(2 * 0.0588 * n_fold) ranked by
  the fold-local rank-mean of (encoder CO sigmoid, CO vote fraction,
  span-derived CO presence)                             [OOF 0.275 -> 0.395].

All non-vote sources are frozen artifacts; nothing here re-runs a model.
Adoption (pre-registered): composed v2 macro >= composed v1 macro + 0.02 on
the same 430 paragraphs, with the v1 composition recomputed in-run from the
identical inputs, AND v2 >= v1 on at least 3 of 5 fold-slice macros
(plateau rule; guards single-fold argmax wins). The champion-t0 gate is
reported for continuity with TIER1_ROUTING_MILESTONE.md.

Example (from shared-task/):
  uv run scripts/route_task1_v2.py \
      --llm-t0 experiments/<t0-run> \
      --rollouts experiments/<r0> ... experiments/<r4> \
      --encoder experiments/<quarter-oof-run> \
      --span-runs experiments/<v3-outer0> ... experiments/<v3-outer4>
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
from daleel.metrics import task1_macro_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

from route_task1 import (  # noqa: E402 — sibling script import
    CO_PRIOR,
    binary_f1,
    fit_vote_threshold,
    load_preds,
)

VOTE_SOURCED = ("AS", "TE", "ST")  # v1 legs kept
CO_BUDGET_MULT = 2.0  # frozen from HEADROOM_AUDIT.md §A3
ADOPTION_MARGIN = 0.02
PLATEAU_MIN_FOLD_WINS = 3


def load_span_sets(span_dirs: list[Path]) -> dict[int, set[str]]:
    """Union of the v3 outer-fold span-derived Task 1 label sets."""
    span: dict[int, set[str]] = {}
    for d in span_dirs:
        for r in read_jsonl(d / "predictions" / "outer_task_1.jsonl"):
            pid = r["paragraph_id"]
            if pid in span:
                raise SystemExit(f"paragraph {pid} appears in two span runs")
            span[pid] = set(r["labels"])
    return span


def load_st_judge(path: Path) -> dict[int, bool]:
    return {r["paragraph_id"]: bool(r["present"]) for r in read_jsonl(path)}


def rank_of(members: list[int], keyfn) -> dict[int, int]:
    return {pid: i for i, pid in enumerate(sorted(members, key=keyfn))}


def co_rank_fusion(
    members: list[int],
    enc_co: dict[int, float],
    votes: dict[int, dict[str, int]],
    span: dict[int, set[str]],
) -> list[int]:
    """Fold-local rank-mean of the three frozen CO ranking features."""
    ranks = [
        rank_of(members, lambda pid: -enc_co[pid]),
        rank_of(members, lambda pid: (-votes[pid]["CO"], -enc_co[pid])),
        rank_of(members, lambda pid: (0 if "CO" in span[pid] else 1, -enc_co[pid])),
    ]
    return sorted(members, key=lambda pid: sum(r[pid] for r in ranks) / len(ranks))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llm-t0", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, nargs="+", required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--span-runs", type=Path, nargs="+", required=True)
    ap.add_argument("--st-judge", type=Path, default=None,
                    help="optional jsonl of {paragraph_id, present} ST-judge decisions")
    ap.add_argument("--ot-source", choices=("consensus", "encoder"),
                    default="consensus",
                    help="OT leg: v2 2-of-3 consensus (default) or the "
                         "deployed v2.1 encoder-only revert")
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
    span = load_span_sets(args.span_runs)
    st_judge = load_st_judge(args.st_judge) if args.st_judge else None

    pids = sorted(gold)
    for name, mapping in (
        ("llm-t0", t0_preds),
        *((f"rollout-{i}", p) for i, p in enumerate(rollout_preds)),
        ("encoder-oof", enc_oof),
        ("encoder-scores", enc_scores),
        ("span-derived", span),
    ):
        if not set(pids) <= set(mapping):
            raise SystemExit(f"{name} does not cover the 430 train-side paragraphs")

    votes: dict[int, dict[str, int]] = {
        pid: {label: sum(label in p[pid] for p in rollout_preds) for label in LABELS}
        for pid in pids
    }

    folds = sorted(set(fold_of.values()))
    fold_pids = {f: [pid for pid in pids if fold_of[pid] == f] for f in folds}

    # ---- cross-fitted vote thresholds (shared by v1 and v2 compositions)
    thetas_by_fold: dict[int, dict[str, int]] = {}
    for f in folds:
        fit_pids = [pid for pid in pids if fold_of[pid] != f]
        thetas_by_fold[f] = {
            label: fit_vote_threshold(fit_pids, votes, gold, label, n_rollouts)
            for label in VOTE_SOURCED
        }

    # ---- CO firing sets
    co_v1: set[int] = set()   # v1 rule: encoder sigmoid, k×1
    co_v2: set[int] = set()   # v2 rule: rank fusion, k×2
    for f in folds:
        members = fold_pids[f]
        ranked_v1 = sorted(members, key=lambda pid: -enc_scores[pid]["CO"])
        co_v1.update(ranked_v1[: round(CO_PRIOR * len(members))])
        ranked_v2 = co_rank_fusion(
            members, {p: enc_scores[p]["CO"] for p in members}, votes, span
        )
        co_v2.update(ranked_v2[: round(CO_BUDGET_MULT * CO_PRIOR * len(members))])

    def st_fires(pid: int) -> bool:
        if votes[pid]["ST"] < thetas_by_fold[fold_of[pid]]["ST"]:
            return False
        if st_judge is not None and pid in st_judge:
            return st_judge[pid]
        return True

    def compose_v1(pid: int) -> set[str]:
        fired = {
            label for label in VOTE_SOURCED
            if votes[pid][label] >= thetas_by_fold[fold_of[pid]][label]
        }
        fired |= {label for label in ("AN", "OT") if label in enc_oof[pid]}
        if pid in co_v1:
            fired.add("CO")
        return fired

    def compose_v2(pid: int) -> set[str]:
        theta = thetas_by_fold[fold_of[pid]]
        fired = {label for label in ("AS", "TE") if votes[pid][label] >= theta[label]}
        if st_fires(pid):
            fired.add("ST")
        if "AN" in enc_oof[pid] or ("AN" in span[pid] and votes[pid]["AN"] >= 1):
            fired.add("AN")
        if args.ot_source == "encoder":
            if "OT" in enc_oof[pid]:
                fired.add("OT")
        else:
            ot_sources = (
                ("OT" in enc_oof[pid]) + ("OT" in span[pid]) + (votes[pid]["OT"] >= 2)
            )
            if ot_sources >= 2:
                fired.add("OT")
        if pid in co_v2:
            fired.add("CO")
        return fired

    routed_v1 = {pid: compose_v1(pid) for pid in pids}
    routed_v2 = {pid: compose_v2(pid) for pid in pids}

    def score(pred: dict[int, set[str]], subset: list[int] | None = None) -> dict:
        keep = pids if subset is None else subset
        official = task1_macro_f1(
            {pid: gold[pid] for pid in keep}, {pid: pred[pid] for pid in keep}
        )
        return {
            "macro_f1": round(official["macro_f1"], 4),
            "per_label_f1": {
                label: round(v["f1"], 4) for label, v in official["per_label"].items()
            },
        }

    t0_score = score(t0_preds)
    v1_score = score(routed_v1)
    v2_score = score(routed_v2)
    fold_macros = {
        str(f): {
            "v1": score(routed_v1, fold_pids[f])["macro_f1"],
            "v2": score(routed_v2, fold_pids[f])["macro_f1"],
        }
        for f in folds
    }
    fold_wins = sum(m["v2"] >= m["v1"] for m in fold_macros.values())

    gate = round(v1_score["macro_f1"] + ADOPTION_MARGIN, 4)
    adopted = v2_score["macro_f1"] >= gate and fold_wins >= PLATEAU_MIN_FOLD_WINS

    gold_co = {pid for pid in pids if "CO" in gold[pid]}
    resolved_config = {
        "script": "route_task1_v2.py",
        "argv": sys.argv[1:],
        "milestone": "HEADROOM_AUDIT.md P0",
        "router_v2": {
            "vote_sourced": ["AS", "TE"],
            "st": "votes ± judge filter",
            "an": "enc | (span & votes>=1)",
            "ot": (
                "encoder-only (v2.1 revert)" if args.ot_source == "encoder"
                else "2-of-3(enc, span, votes>=2)"
            ),
            "co": {
                "budget_mult": CO_BUDGET_MULT,
                "prior": CO_PRIOR,
                "ranker": "fold-local rank-mean(enc sigmoid, vote fraction, span presence)",
            },
            "n_rollouts": n_rollouts,
            "st_judge": str(args.st_judge) if args.st_judge else None,
        },
        "inputs": {
            "llm_t0": str(args.llm_t0),
            "rollouts": [str(d) for d in args.rollouts],
            "encoder": str(args.encoder),
            "span_runs": [str(d) for d in args.span_runs],
            "encoder_oof_sha256": file_sha256(
                args.encoder / "predictions" / "oof_task_1.jsonl"
            ),
            "span_sha256": [
                file_sha256(d / "predictions" / "outer_task_1.jsonl")
                for d in args.span_runs
            ],
        },
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"t1-routed-v2-k{n_rollouts}-n{len(pids)}", resolved_config
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)

    metrics = {
        "n_paragraphs": len(pids),
        "champion_t0": t0_score,
        "routed_v1_composed_oof": v1_score,
        "routed_v2_composed_oof": v2_score,
        "adoption": {
            "rule": (
                f"v2 >= v1 + {ADOPTION_MARGIN} AND v2 >= v1 on "
                f">={PLATEAU_MIN_FOLD_WINS}/5 fold slices"
            ),
            "gate": gate,
            "fold_wins": fold_wins,
            "adopted": adopted,
            "champion_gate_for_reference": round(
                t0_score["macro_f1"] + ADOPTION_MARGIN, 4
            ),
        },
        "fold_slice_macros": fold_macros,
        "vote_thresholds_by_fold": {str(f): thetas_by_fold[f] for f in folds},
        "co_budget_v2": {
            "mult": CO_BUDGET_MULT,
            "fired": len(co_v2),
            "hits": len(co_v2 & gold_co),
            "gold_co_count": len(gold_co),
            "co_f1_v2_rank_fusion": round(binary_f1(co_v2, gold_co), 4),
            "co_f1_v1_encoder_k1": round(binary_f1(co_v1, gold_co), 4),
        },
        "leg_f1": {
            "an_v1_encoder_only": round(
                binary_f1({p for p in pids if "AN" in enc_oof[p]},
                          {p for p in pids if "AN" in gold[p]}), 4),
            "an_v2_consensus": round(
                binary_f1({p for p in pids if "AN" in routed_v2[p]},
                          {p for p in pids if "AN" in gold[p]}), 4),
            "ot_v1_encoder_only": round(
                binary_f1({p for p in pids if "OT" in enc_oof[p]},
                          {p for p in pids if "OT" in gold[p]}), 4),
            "ot_v2_consensus": round(
                binary_f1({p for p in pids if "OT" in routed_v2[p]},
                          {p for p in pids if "OT" in gold[p]}), 4),
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
                "labels": sorted(routed_v2[pid]),
            }
            for pid in pids
        ],
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nmetrics: {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
