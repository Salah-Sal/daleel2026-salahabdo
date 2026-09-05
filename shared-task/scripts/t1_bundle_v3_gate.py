"""T1 bundle-v3 gate (registered, CREATIVE_HEADROOM_RESEARCH.md commit 56d6f30).

Recomputes the v2.2 composition from the byte-identical inputs of the
0.7298 gate run (CONTROL — must reproduce exactly), then scores:
  PRIMARY   full bundle: 5-seed quarter ensemble encoder + TE 6-vote
            theta with llama + CO 4-feature rank (llama CO presence);
  SECONDARY ensemble-swap only (evaluated only if primary fails);
  ablation diagnostics (TE-only, CO-only) — reported, adopt nothing.
Gate for either test: composed macro >= v2.2 + 0.02 AND >=3/5 fold wins.
Zero API cost; all inputs are frozen cached artifacts.

Example (from shared-task/):
  uv run scripts/t1_bundle_v3_gate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.artifacts import atomic_write_json, create_experiment_dir
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
from route_task1_v2 import (  # noqa: E402 — sibling script import
    CO_BUDGET_MULT,
    load_span_sets,
    load_st_judge,
    rank_of,
)

BASELINE_RUN = EXPERIMENTS_DIR / "20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa"
BASELINE_MACRO = 0.7298
ENSEMBLE_RUN = EXPERIMENTS_DIR / (
    "20260714-233725-647050Z-t1-encoder-ensemble-quarter-5seed-cpu-n430-a7b50f0a10")
LLAMA_RUN = EXPERIMENTS_DIR / (
    "20260714-200859-913892Z-t1-zeroshot-llama-3-3-70b-paid-train-3b704b9a00")
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
    llama = load_preds(LLAMA_RUN)

    enc_base_oof, enc_base_scores, fold_of = load_encoder(
        EXPERIMENTS_DIR / Path(inputs["encoder"]).name)
    enc_ens_oof, enc_ens_scores, fold_of_ens = load_encoder(ENSEMBLE_RUN)

    pids = sorted(gold)
    if any(fold_of[p] != fold_of_ens[p] for p in pids):
        raise SystemExit("ensemble fold assignment differs from baseline encoder")
    if not set(pids) <= set(llama):
        raise SystemExit("llama run does not cover the 430 paragraphs")

    votes = {pid: {label: sum(label in p[pid] for p in rollout_preds)
                   for label in LABELS} for pid in pids}
    votes6_te = {pid: votes[pid]["TE"] + ("TE" in llama[pid]) for pid in pids}

    folds = sorted(set(fold_of.values()))
    fold_pids = {f: [pid for pid in pids if fold_of[pid] == f] for f in folds}

    thetas5, theta6_te = {}, {}
    for f in folds:
        fit = [pid for pid in pids if fold_of[pid] != f]
        thetas5[f] = {label: fit_vote_threshold(fit, votes, gold, label, n5)
                      for label in VOTE_5}
        theta6_te[f] = fit_vote_threshold(
            fit, {p: {"TE": votes6_te[p]} for p in pids}, gold, "TE", n5 + 1)

    def co_fired(enc_scores, use_llama_feature):
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
            if use_llama_feature:
                ranks.append(rank_of(members, lambda pid: (
                    0 if "CO" in llama[pid] else 1, -enc_scores[pid]["CO"])))
            ranked = sorted(members,
                            key=lambda pid: sum(r[pid] for r in ranks) / len(ranks))
            fired.update(ranked[: round(CO_BUDGET_MULT * CO_PRIOR * len(members))])
        return fired

    def compose(enc_oof, enc_scores, te_llama, co_llama):
        co = co_fired(enc_scores, co_llama)
        routed = {}
        for pid in pids:
            f = fold_of[pid]
            fired = set()
            if votes[pid]["AS"] >= thetas5[f]["AS"]:
                fired.add("AS")
            te_ok = (votes6_te[pid] >= theta6_te[f] if te_llama
                     else votes[pid]["TE"] >= thetas5[f]["TE"])
            if te_ok:
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
        "v22_control": compose(enc_base_oof, enc_base_scores, False, False),
        "bundle_full": compose(enc_ens_oof, enc_ens_scores, True, True),
        "ensemble_only": compose(enc_ens_oof, enc_ens_scores, False, False),
        "te_only_ablation": compose(enc_base_oof, enc_base_scores, True, False),
        "co_only_ablation": compose(enc_base_oof, enc_base_scores, False, True),
    }
    scores = {name: score(pred) for name, pred in compositions.items()}

    control = scores["v22_control"]["macro_f1"]
    if abs(control - BASELINE_MACRO) > 1e-9:
        raise SystemExit(
            f"CONTROL FAILED: recomputed v2.2 = {control} != {BASELINE_MACRO}; aborting")

    def gate_eval(name):
        wins = 0
        fold_slices = {}
        for f in folds:
            b = score(compositions["v22_control"], fold_pids[f])["macro_f1"]
            c = score(compositions[name], fold_pids[f])["macro_f1"]
            wins += c >= b
            fold_slices[str(f)] = {"v22": b, name: c}
        macro = scores[name]["macro_f1"]
        return {
            "macro": macro,
            "delta": round(macro - control, 4),
            "fold_wins": wins,
            "fold_slices": fold_slices,
            "passed": bool(macro >= control + ADOPTION_MARGIN
                           and wins >= PLATEAU_MIN_FOLD_WINS),
        }

    primary = gate_eval("bundle_full")
    secondary = None if primary["passed"] else gate_eval("ensemble_only")
    adopted = ("bundle_full" if primary["passed"]
               else "ensemble_only" if secondary and secondary["passed"]
               else None)

    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md bundle-v3 (commit 56d6f30)",
        "control_v22_macro": control,
        "gate": round(control + ADOPTION_MARGIN, 4),
        "primary_bundle_full": primary,
        "secondary_ensemble_only": secondary,
        "adopted": adopted,
        "composed_macros": {k: v["macro_f1"] for k, v in scores.items()},
        "per_label": {k: v["per_label_f1"] for k, v in scores.items()},
        "theta_te_6vote_by_fold": {str(f): theta6_te[f] for f in folds},
        "theta5_by_fold": {str(f): thetas5[f] for f in folds},
    }

    resolved_config = {
        "script": "t1_bundle_v3_gate.py", "argv": sys.argv[1:],
        "baseline_run": str(BASELINE_RUN), "ensemble_run": str(ENSEMBLE_RUN),
        "llama_run": str(LLAMA_RUN),
        "registration": "CREATIVE_HEADROOM_RESEARCH.md bundle-v3 (commit 56d6f30)",
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"t1-bundle-v3-n{len(pids)}", resolved_config)
    atomic_write_json(exp_dir / "config.json", resolved_config)
    atomic_write_json(exp_dir / "metrics.json", metrics)
    source_by_id = {r["paragraph_id"]: r for r in t1 if r["paragraph_id"] in gold}
    winner = compositions[adopted] if adopted else compositions["bundle_full"]
    write_jsonl(exp_dir / "predictions" / "preds.jsonl", [
        {"paragraph_id": pid, "text": source_by_id[pid]["text"],
         "type": source_by_id[pid]["type"], "labels": sorted(winner[pid])}
        for pid in pids
    ])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
