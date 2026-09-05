"""T1 TAPT gate (registered, CREATIVE_HEADROOM_RESEARCH.md commit cd77061).

Recomputes the v2.2 composition from the byte-identical inputs of the
0.7298 baseline run (CONTROL — must reproduce exactly), then scores the
single frozen variant: the TAPT-pretrained encoder's oof/scores
substituted into the AN/OT legs + CO rank features (exact ensemble_only
swap pattern of t1_bundle_v3_gate.py). Everything else untouched.
GATE: composed macro >= 0.7298 + 0.02 = 0.7498 AND >= 3/5 fold wins.
OOF-measurement-only: nothing deploys regardless of verdict, pending a
written organizer ruling on transductive input use.

Example (from shared-task/):
  uv run scripts/t1_tapt_gate.py --tapt-encoder-run <experiment dir name>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.artifacts import atomic_write_json, create_experiment_dir
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl
from daleel.metrics import task1_macro_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

from route_task1 import (  # noqa: E402 — sibling script import
    CO_PRIOR,
    fit_vote_threshold,
    load_preds,
)
from route_task1_v2 import (  # noqa: E402 — sibling script import
    CO_BUDGET_MULT,
    load_span_sets,
    load_st_judge,
    rank_of,
)

BASELINE_RUN = EXPERIMENTS_DIR / "20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa"
BASELINE_MACRO = 0.7298
ADOPTION_MARGIN = 0.02
PLATEAU_MIN_FOLD_WINS = 3
VOTE_5 = ("AS", "TE", "ST")


def load_encoder(run: Path):
    oof = {r["paragraph_id"]: set(r["labels"])
           for r in read_jsonl(run / "predictions" / "oof_task_1.jsonl")}
    scores, fold_of = {}, {}
    for r in read_jsonl(run / "predictions" / "oof_task_1_scores.jsonl"):
        scores[r["paragraph_id"]] = r["scores"]
        fold_of[r["paragraph_id"]] = r["fold"]
    return oof, scores, fold_of


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapt-encoder-run", required=True,
                        help="experiment dir name of the TAPT-encoder retrain")
    args = parser.parse_args()
    tapt_run = EXPERIMENTS_DIR / args.tapt_encoder_run

    base_cfg = json.loads((BASELINE_RUN / "config.json").read_text())
    inputs = base_cfg["inputs"]
    st_judge_path = Path(base_cfg["router_v2"]["st_judge"])
    if base_cfg["router_v2"]["ot"] != "encoder-only (v2.1 revert)":
        raise SystemExit("baseline run is not the --ot-source encoder composition")

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {pid: labels for pid, labels in clean_task1_gold(t1, t2).items()
            if pid in set(train_ids)}

    rollout_preds = [load_preds(EXPERIMENTS_DIR / Path(d).name)
                     for d in inputs["rollouts"]]
    n5 = len(rollout_preds)
    span = load_span_sets([EXPERIMENTS_DIR / Path(d).name
                           for d in inputs["span_runs"]])
    st_judge = load_st_judge(st_judge_path)

    enc_base_oof, enc_base_scores, fold_of = load_encoder(
        EXPERIMENTS_DIR / Path(inputs["encoder"]).name)
    enc_tapt_oof, enc_tapt_scores, fold_of_tapt = load_encoder(tapt_run)

    pids = sorted(gold)
    if any(fold_of[p] != fold_of_tapt[p] for p in pids):
        raise SystemExit("TAPT encoder fold assignment differs from baseline")

    votes = {pid: {label: sum(label in p[pid] for p in rollout_preds)
                   for label in LABELS} for pid in pids}

    folds = sorted(set(fold_of.values()))
    fold_pids = {f: [pid for pid in pids if fold_of[pid] == f] for f in folds}

    thetas5 = {}
    for f in folds:
        fit = [pid for pid in pids if fold_of[pid] != f]
        thetas5[f] = {label: fit_vote_threshold(fit, votes, gold, label, n5)
                      for label in VOTE_5}

    def co_fired(enc_scores):
        fired = set()
        for f in folds:
            members = fold_pids[f]
            ranks = [
                rank_of(members, lambda pid: -enc_scores[pid]["CO"]),
                rank_of(members, lambda pid: (-votes[pid]["CO"],
                                              -enc_scores[pid]["CO"])),
                rank_of(members, lambda pid: (0 if "CO" in span[pid] else 1,
                                              -enc_scores[pid]["CO"])),
            ]
            ranked = sorted(members,
                            key=lambda pid: sum(r[pid] for r in ranks) / len(ranks))
            fired.update(ranked[: round(CO_BUDGET_MULT * CO_PRIOR * len(members))])
        return fired

    def compose(enc_oof, enc_scores):
        co = co_fired(enc_scores)
        routed = {}
        for pid in pids:
            f = fold_of[pid]
            fired = set()
            if votes[pid]["AS"] >= thetas5[f]["AS"]:
                fired.add("AS")
            if votes[pid]["TE"] >= thetas5[f]["TE"]:
                fired.add("TE")
            if votes[pid]["ST"] >= thetas5[f]["ST"] and st_judge.get(pid, True):
                fired.add("ST")
            if "AN" in enc_oof[pid] or ("AN" in span[pid] and votes[pid]["AN"] >= 1):
                fired.add("AN")
            if "OT" in enc_oof[pid]:
                fired.add("OT")
            if pid in co:
                fired.add("CO")
            routed[pid] = fired
        return routed

    def score(pred, subset=None):
        keep = pids if subset is None else subset
        official = task1_macro_f1({p: gold[p] for p in keep},
                                  {p: pred[p] for p in keep})
        return {"macro_f1": round(official["macro_f1"], 4),
                "per_label_f1": {label: round(v["f1"], 4)
                                 for label, v in official["per_label"].items()}}

    compositions = {
        "v22_control": compose(enc_base_oof, enc_base_scores),
        "tapt_swap": compose(enc_tapt_oof, enc_tapt_scores),
    }
    scores = {name: score(pred) for name, pred in compositions.items()}

    control = scores["v22_control"]["macro_f1"]
    if abs(control - BASELINE_MACRO) > 1e-9:
        raise SystemExit(
            f"CONTROL FAILED: recomputed v2.2 = {control} != {BASELINE_MACRO}; aborting")

    wins, fold_slices = 0, {}
    for f in folds:
        b = score(compositions["v22_control"], fold_pids[f])["macro_f1"]
        c = score(compositions["tapt_swap"], fold_pids[f])["macro_f1"]
        wins += c >= b
        fold_slices[str(f)] = {"v22": b, "tapt": c}
    macro = scores["tapt_swap"]["macro_f1"]

    # encoder-only diagnostic: how much did TAPT move the raw encoder?
    enc_only = {
        "baseline": round(task1_macro_f1(
            gold, {p: enc_base_oof[p] for p in pids})["macro_f1"], 4),
        "tapt": round(task1_macro_f1(
            gold, {p: enc_tapt_oof[p] for p in pids})["macro_f1"], 4),
    }

    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md TAPT gate (commit cd77061)",
        "control_v22_macro": control,
        "gate": round(control + ADOPTION_MARGIN, 4),
        "tapt_swap": {
            "macro": macro,
            "delta": round(macro - control, 4),
            "fold_wins": wins,
            "fold_slices": fold_slices,
            "passed": bool(macro >= control + ADOPTION_MARGIN
                           and wins >= PLATEAU_MIN_FOLD_WINS),
        },
        "encoder_only_oof_macro": enc_only,
        "per_label": {k: v["per_label_f1"] for k, v in scores.items()},
        "deployment": "BLOCKED regardless of verdict pending organizer ruling "
                      "on transductive input use",
    }

    resolved_config = {
        "script": "t1_tapt_gate.py", "argv": sys.argv[1:],
        "baseline_run": str(BASELINE_RUN), "tapt_encoder_run": str(tapt_run),
        "registration": "CREATIVE_HEADROOM_RESEARCH.md TAPT (commit cd77061)",
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"t1-tapt-gate-n{len(pids)}", resolved_config)
    atomic_write_json(exp_dir / "config.json", resolved_config)
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
