"""CO contrastive many-shot specialist as a 4th rank feature (registered
gate in CREATIVE_HEADROOM_RESEARCH.md, frozen 2026-07-15 before any call).

One binary present/absent decision per paragraph over all 430, 3 rollouts
at T=0.7, score = present-fraction. The prompt carries the official CO
definition, the four NOT-CO convention rules induced from the forensics
read, and fold-excluded contrastive exemplars: ALL out-of-fold gold-CO
paragraphs as CO-PRESENT plus the first 12 (sorted pid) out-of-fold
unanimous-FP paragraphs (6/6 gemma CO votes, not gold CO) as CO-ABSENT.

Frozen composition change (the only change): CO leg v3 = same fold-local
top-k*2 budget, rank-mean over FOUR features = the three frozen route
features + specialist fraction. Control: the 3-feature rank must
reproduce the v2.2 baseline CO fired set exactly. Gate: composed macro
(exact single-leg arithmetic vs run 4f30d408fa) >= 0.7498 AND >=3/5
fold-slice wins.

Example (from shared-task/):
  uv run scripts/co_specialist_rank.py --model gemma-4-31b-paid
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dspy

from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import task1_macro_f1
from daleel.models import SPECS, make_lm
from daleel.policy import LABEL_POLICY, SPURIOUS_HINTS
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

from route_task1_v2 import CO_BUDGET_MULT, CO_PRIOR, load_span_sets, rank_of

BASELINE_RUN = EXPERIMENTS_DIR / "20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa"
ROLLOUTS = 3

NOT_CO_RULES = """The annotators apply CO narrowly. Four patterns are NOT CO:
1. "Everyone-knows" rhetoric (كلنا نعلم / من المعلوم / نحن نعلم) wrapping a
   claim the debate contests — that is the speaker's stance (AS).
2. Settled historical events or concrete instances, however undisputed
   (a patent date, an invasion, a nation's history) — those are AN.
3. Legal texts, treaties, scripture, or official documents cited AS
   EVIDENCE for a side — those are TE.
4. Restating the debate motion, procedural openings, or scene-setting
   ledes — those are OT (or nothing).
CO IS: neutral exposition of the thing under debate itself — definitions
("X is / X means"), the legal or procedural framework being discussed,
an entity's stated goals, established policy decisions — and generic
commonsense premises both sides accept (even informal truisms embedded
in debate flow)."""


class VerifyCommonGround(dspy.Signature):
    __doc__ = f"""Decide whether a CO (common ground) unit is GENUINELY
present in the paragraph, matching the gold annotation convention as
evidenced by the labeled examples — first paragraphs where CO IS present,
then deceptively similar ones where annotators say CO is NOT present.

- {LABEL_POLICY["CO"]}
- Caution: {SPURIOUS_HINTS["CO"]}

{NOT_CO_RULES}"""

    examples: str = dspy.InputField(
        desc="gold-labeled example paragraphs, CO present and CO absent")
    text: str = dspy.InputField(desc="the Arabic paragraph to judge")
    genre: str = dspy.InputField()
    present: bool = dspy.OutputField(
        desc="True only if CO is genuinely present per the annotation convention")


def exemplar_block(pids: list[int], text_of: dict[int, str], verdict: str) -> str:
    return "\n\n".join(
        f"[{verdict} — example {i + 1}]\n{text_of[p]}"
        for i, p in enumerate(sorted(pids))
    )


def co_rank_v3(members, enc_co, votes, span, frac):
    ranks = [
        rank_of(members, lambda pid: -enc_co[pid]),
        rank_of(members, lambda pid: (-votes[pid]["CO"], -enc_co[pid])),
        rank_of(members, lambda pid: (0 if "CO" in span[pid] else 1, -enc_co[pid])),
        rank_of(members, lambda pid: (-frac[pid], -enc_co[pid])),
    ]
    return sorted(members, key=lambda pid: sum(r[pid] for r in ranks) / len(ranks))


def co_rank_v2(members, enc_co, votes, span):
    ranks = [
        rank_of(members, lambda pid: -enc_co[pid]),
        rank_of(members, lambda pid: (-votes[pid]["CO"], -enc_co[pid])),
        rank_of(members, lambda pid: (0 if "CO" in span[pid] else 1, -enc_co[pid])),
    ]
    return sorted(members, key=lambda pid: sum(r[pid] for r in ranks) / len(ranks))


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N paragraphs (smoke test); "
                    "gate arithmetic is skipped")
    args = ap.parse_args()

    cfg = json.load(open(BASELINE_RUN / "config.json"))
    argv = cfg["argv"]

    def arg_of(flag):
        return Path(argv[argv.index(flag) + 1])

    def args_of(flag):
        out, i = [], argv.index(flag) + 1
        while i < len(argv) and not argv[i].startswith("--"):
            out.append(Path(argv[i]))
            i += 1
        return out

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {p: l for p, l in clean_task1_gold(t1, t2).items()
            if p in set(train_ids)}
    text_of = {int(r["paragraph_id"]): r["text"] for r in t1}
    genre_of = {int(r["paragraph_id"]): r["type"] for r in t1}

    enc_scores, fold_of = {}, {}
    for r in read_jsonl(arg_of("--encoder") / "predictions" / "oof_task_1_scores.jsonl"):
        enc_scores[r["paragraph_id"]] = r["scores"]
        fold_of[r["paragraph_id"]] = r["fold"]
    rollout_preds = [
        {int(r["paragraph_id"]): set(r["labels"]) for r in
         read_jsonl(next((d / "predictions").glob("*.jsonl")))}
        for d in args_of("--rollouts")
    ]
    t0_preds = {int(r["paragraph_id"]): set(r["labels"]) for r in
                read_jsonl(next((arg_of("--llm-t0") / "predictions").glob("*.jsonl")))}
    span = load_span_sets(args_of("--span-runs"))

    pids = sorted(gold)
    votes = {p: {"CO": sum("CO" in rp[p] for rp in rollout_preds)} for p in pids}
    six_run_votes = {p: votes[p]["CO"] + ("CO" in t0_preds[p]) for p in pids}
    gold_co = sorted(p for p in pids if "CO" in gold[p])
    unanimous_fp = sorted(p for p in pids
                          if "CO" not in gold[p] and six_run_votes[p] == 6)

    folds = sorted(set(fold_of.values()))
    examples_by_fold = {
        f: (
            exemplar_block([p for p in gold_co if fold_of[p] != f],
                           text_of, "CO PRESENT")
            + "\n\n"
            + exemplar_block([p for p in unanimous_fp if fold_of[p] != f][:12],
                             text_of, "CO ABSENT")
        )
        for f in folds
    }

    targets = pids[: args.limit] if args.limit else pids
    spec = SPECS[args.model]
    verify = dspy.ChainOfThought(VerifyCommonGround)
    verdicts: dict[int, list[bool]] = {p: [] for p in targets}
    errors: list[str] = []

    for rollout in range(ROLLOUTS):
        lm = make_lm(spec, temperature=0.7, max_tokens=args.max_tokens,
                     rollout_id=rollout)

        def verify_one(p: int):
            last = None
            for attempt in range(4):
                try:
                    with dspy.context(lm=lm):
                        result = verify(
                            examples=examples_by_fold[fold_of[p]],
                            text=text_of[p], genre=genre_of[p])
                    return bool(result.present)
                except Exception as e:  # neutral containment: rollout excluded
                    last = f"r{rollout} p{p}: {type(e).__name__}: {e}"
                    time.sleep(min(75, 25 * (attempt + 1)))
            errors.append(last)
            return None

        with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
            for p, present in zip(targets, pool.map(verify_one, targets)):
                if present is not None:
                    verdicts[p].append(present)

    frac = {p: (sum(v) / len(v) if v else 0.5) for p, v in verdicts.items()}

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t1-co-specialist-{spec.key}-n{len(targets)}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "specialist_verdicts.jsonl", [
        {"paragraph_id": p, "rollout_present": verdicts[p],
         "fraction": frac[p]} for p in targets
    ])

    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md registered CO specialist",
        "model": spec.key,
        "rollouts": ROLLOUTS,
        "temperature": 0.7,
        "n_scored": len(targets),
        "n_gold_co": len(gold_co),
        "n_unanimous_fp_pool": len(unanimous_fp),
        "n_program_errors": len(errors),
        "errors_sample": errors[:5],
        "specialist_alone": {
            "gold_co_fraction_mean": round(
                sum(frac[p] for p in gold_co if p in frac)
                / max(1, sum(p in frac for p in gold_co)), 4),
            "unanimous_fp_fraction_mean": round(
                sum(frac[p] for p in unanimous_fp if p in frac)
                / max(1, sum(p in frac for p in unanimous_fp)), 4),
        },
        "baseline_run": str(BASELINE_RUN),
    }

    if not args.limit:
        base_metrics = json.load(open(BASELINE_RUN / "metrics.json"))
        base_macro = base_metrics["routed_v2_composed_oof"]["macro_f1"]
        base_preds = {int(r["paragraph_id"]): set(r["labels"]) for r in
                      read_jsonl(BASELINE_RUN / "predictions" / "preds.jsonl")}
        base_co_fired = {p for p, ls in base_preds.items() if "CO" in ls}

        fold_pids = {f: [p for p in pids if fold_of[p] == f] for f in folds}
        control, co_v3 = set(), set()
        for f in folds:
            members = fold_pids[f]
            enc_co = {p: enc_scores[p]["CO"] for p in members}
            k = round(CO_BUDGET_MULT * CO_PRIOR * len(members))
            control.update(co_rank_v2(members, enc_co, votes, span)[:k])
            co_v3.update(co_rank_v3(members, enc_co, votes, span, frac)[:k])
        if control != base_co_fired:
            raise SystemExit(
                f"CONTROL FAILED: 3-feature rank does not reproduce baseline "
                f"CO set (only in control: {sorted(control - base_co_fired)}, "
                f"only in baseline: {sorted(base_co_fired - control)})")

        def co_f1(fired, subset):
            tp = sum(p in fired and "CO" in gold[p] for p in subset)
            fp = sum(p in fired and "CO" not in gold[p] for p in subset)
            fn = sum(p not in fired and "CO" in gold[p] for p in subset)
            return f1(tp, fp, fn)

        co_base, co_new = co_f1(base_co_fired, pids), co_f1(co_v3, pids)
        composed_new = base_macro + (co_new - co_base) / 6

        new_preds = {p: (base_preds[p] - {"CO"}) | ({"CO"} if p in co_v3 else set())
                     for p in pids}
        wins = 0
        fold_slices = {}
        for f in folds:
            sub = set(fold_pids[f])
            g = {p: gold[p] for p in sub}
            m_old = task1_macro_f1(g, {p: base_preds[p] for p in sub})["macro_f1"]
            m_new = task1_macro_f1(g, {p: new_preds[p] for p in sub})["macro_f1"]
            wins += m_new >= m_old
            fold_slices[str(f)] = {"base": round(m_old, 4), "v3": round(m_new, 4)}

        gate = round(base_macro + 0.02, 4)
        metrics["gate"] = {
            "co_f1_base": round(co_base, 4),
            "co_f1_v3": round(co_new, 4),
            "composed_base": base_macro,
            "composed_v3": round(composed_new, 4),
            "rule": f"composed >= {gate} AND >=3/5 fold wins",
            "fold_slices": fold_slices,
            "fold_wins": wins,
            "adopted": bool(composed_new >= gate and wins >= 3),
        }
        write_jsonl(pred_dir / "co_v3_fired.jsonl",
                    [{"paragraph_id": p, "co": p in co_v3} for p in pids])

    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "co_specialist_rank.py", "argv": sys.argv[1:],
         "model": spec.__dict__, "baseline_argv": argv,
         "dspy_version": dspy.__version__},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
