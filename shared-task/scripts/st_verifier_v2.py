"""ST contrastive verifier v2 (registered gate in CREATIVE_HEADROOM_RESEARCH.md).

Precision filter over the ST-fired paragraphs of the pinned v2.1 baseline
composition. For each fired paragraph the verifier sees ALL out-of-fold
gold-ST paragraphs and ALL out-of-fold fired-false-positive paragraphs as
labeled exemplars (contrastive many-shot; supervision never crosses the
paragraph's own fold), plus the official ST definition/caution, and votes
present/absent over 3 rollouts at T=0.7. Frozen decision rule: drop ST
iff the majority says absent AND at least one cross-model zero-shot run
(qwen3-32b, llama-3.3-70b) also lacks ST — the co-signal nominates, the
verifier confirms. Program errors keep ST (containment, never drop).

Output: predictions/st_judge_v2_train430.jsonl in the route_task1_v2
--st-judge schema (old judge verdicts with the new drops overriding), so
the gate rerun is the baseline argv with only --st-judge swapped.

Example (from shared-task/):
  uv run scripts/st_verifier_v2.py --model gemma-4-31b-paid
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dspy

from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.models import SPECS, make_lm
from daleel.policy import LABEL_POLICY, SPURIOUS_HINTS
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

BASELINE_RUN = EXPERIMENTS_DIR / "20260714-230133-918438Z-t1-routed-v2-k5-n430-986aefa1c6"
FOLDS_FROM = (
    EXPERIMENTS_DIR
    / "20260711-090448-516508Z-encoder-t1-camelbert-msa-quarter-both-legacy-train-0e77e5faf4"
    / "predictions" / "oof_task_1_scores.jsonl"
)
OLD_ST_JUDGE = (
    EXPERIMENTS_DIR
    / "20260714-t1-judge-stage-gemma-4-31b-paid-val"
    / "predictions" / "st_judge_train430.jsonl"
)
QWEN_RUN = EXPERIMENTS_DIR / "20260714-201837-601645Z-t1-zeroshot-qwen3-32b-train-5bc5af9b3b"
LLAMA_RUN = EXPERIMENTS_DIR / "20260714-200859-913892Z-t1-zeroshot-llama-3-3-70b-paid-train-3b704b9a00"
ROLLOUTS = 3


class VerifyStatistics(dspy.Signature):
    __doc__ = f"""A previous system flagged the paragraph as containing ST.
Decide whether ST is GENUINELY present, using the labeled examples: they
are gold-annotated paragraphs from the same corpus — first ones where ST
IS present, then deceptively similar ones where the same system fired but
the annotators say ST is NOT present. Match the annotators' convention as
evidenced by the examples, not your own intuition.

- {LABEL_POLICY["ST"]}
- Caution: {SPURIOUS_HINTS["ST"]}"""

    examples: str = dspy.InputField(
        desc="gold-labeled example paragraphs, ST present and ST absent")
    text: str = dspy.InputField(desc="the Arabic paragraph to judge")
    genre: str = dspy.InputField()
    present: bool = dspy.OutputField(
        desc="True only if ST is genuinely present per the annotation convention")


def label_sets(run_dir: Path) -> dict[int, set[str]]:
    pred = next(run_dir.glob("predictions/*.jsonl"))
    return {int(r["paragraph_id"]): set(r.get("labels") or [])
            for r in read_jsonl(pred)}


def exemplar_block(pids: list[int], text_of: dict[int, str], verdict: str) -> str:
    return "\n\n".join(
        f"[{verdict} — example {i + 1}]\n{text_of[p]}"
        for i, p in enumerate(sorted(pids))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N fired paragraphs only (smoke test)")
    args = ap.parse_args()

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {p: l for p, l in clean_task1_gold(t1, t2).items()
            if p in set(train_ids)}
    text_of = {int(r["paragraph_id"]): r["text"] for r in t1}
    genre_of = {int(r["paragraph_id"]): r["type"] for r in t1}
    fold_of = {r["paragraph_id"]: r["fold"] for r in read_jsonl(FOLDS_FROM)}

    baseline = {int(r["paragraph_id"]): set(r["labels"]) for r in read_jsonl(
        BASELINE_RUN / "predictions" / "preds.jsonl")}
    fired = sorted(p for p, ls in baseline.items() if "ST" in ls)
    if args.limit:
        fired = fired[: args.limit]
    qwen, llama = label_sets(QWEN_RUN), label_sets(LLAMA_RUN)
    eligible = {p for p in fired
                if "ST" not in qwen.get(p, set()) or "ST" not in llama.get(p, set())}

    gold_st = sorted(p for p in gold if "ST" in gold[p])
    fired_fp = sorted(p for p in baseline
                      if "ST" in baseline[p] and "ST" not in gold[p])
    examples_by_fold = {
        f: (
            exemplar_block([p for p in gold_st if fold_of[p] != f],
                           text_of, "ST PRESENT")
            + "\n\n"
            + exemplar_block([p for p in fired_fp if fold_of[p] != f],
                             text_of, "ST ABSENT")
        )
        for f in sorted(set(fold_of.values()))
    }

    spec = SPECS[args.model]
    verify = dspy.ChainOfThought(VerifyStatistics)
    verdicts: dict[int, list[bool]] = {p: [] for p in fired}
    errors: list[str] = []

    for rollout in range(ROLLOUTS):
        lm = make_lm(spec, temperature=0.7, max_tokens=args.max_tokens,
                     rollout_id=rollout)

        def verify_one(p: int) -> bool:
            last = None
            for attempt in range(4):
                try:
                    with dspy.context(lm=lm):
                        result = verify(
                            examples=examples_by_fold[fold_of[p]],
                            text=text_of[p], genre=genre_of[p])
                    return bool(result.present)
                except Exception as e:  # containment: an error keeps ST
                    last = f"r{rollout} p{p}: {type(e).__name__}: {e}"
                    time.sleep(min(75, 25 * (attempt + 1)))
            errors.append(last)
            return True

        with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
            for p, present in zip(fired, pool.map(verify_one, fired)):
                verdicts[p].append(present)

    majority_absent = {p for p, v in verdicts.items()
                       if sum(not x for x in v) * 2 > len(v)}
    drops = sorted(majority_absent & eligible)

    old_rows = list(read_jsonl(OLD_ST_JUDGE))
    covered = {int(r["paragraph_id"]) for r in old_rows}
    out_rows = [
        {**r, "present": r["present"] and int(r["paragraph_id"]) not in drops}
        for r in old_rows
    ] + [
        {"paragraph_id": p, "present": False}
        for p in drops if p not in covered
    ]

    drop_tp = sum("ST" in gold[p] for p in drops)
    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md registered ST verifier v2",
        "model": spec.key,
        "rollouts": ROLLOUTS,
        "temperature": 0.7,
        "n_fired": len(fired),
        "n_eligible_by_cosignal": len(eligible),
        "n_majority_absent": len(majority_absent),
        "n_majority_absent_blocked_by_cosignal": len(majority_absent - eligible),
        "n_dropped": len(drops),
        "dropped_gold_split": {"gold_ST": drop_tp,
                               "not_gold_ST": len(drops) - drop_tp},
        "rollout_disagreement": sum(1 for v in verdicts.values()
                                    if len(set(v)) > 1),
        "n_program_errors": len(errors),
        "errors_sample": errors[:5],
        "baseline_run": str(BASELINE_RUN),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t1-st-verifier-v2-{spec.key}-train430"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "st_judge_v2_train430.jsonl", out_rows)
    write_jsonl(pred_dir / "verifier_verdicts.jsonl", [
        {"paragraph_id": p, "rollout_present": verdicts[p],
         "eligible": p in eligible, "dropped": p in drops}
        for p in fired
    ])
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "st_verifier_v2.py", "argv": sys.argv[1:],
         "model": spec.__dict__, "dspy_version": dspy.__version__},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nst-judge file: {pred_dir / 'st_judge_v2_train430.jsonl'}")


if __name__ == "__main__":
    main()
