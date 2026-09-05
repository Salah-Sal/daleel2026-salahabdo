"""Deploy twin of the ST contrastive verifier v2 (adopted 2026-07-15).

Same frozen design as scripts/st_verifier_v2.py, applied to the eval-side
input: targets are the ST-fired paragraphs of the v2.1 deploy run, the
exemplar block is ALL 19 train gold-ST paragraphs + ALL 19 train-OOF
fired-FP paragraphs (no fold exclusion on deploy — registration note in
CREATIVE_HEADROOM_RESEARCH.md), and the co-signal is one qwen3-32b + one
llama-3.3-70b zero-shot run on the same input. Drop rule unchanged:
majority-absent over 3 rollouts at T=0.7 AND at least one co-signal run
lacks ST. Errors keep ST.

Output: predictions/st_judge_v2_<tag>.jsonl in route_task1_v2_deploy
--st-judge schema (old deploy judge verdicts with new drops overriding).

Example (from shared-task/):
  uv run scripts/st_verifier_v2_deploy.py --model gemma-4-31b-paid \
      --qwen-run experiments/<qwen dev run> --llama-run experiments/<llama dev run>
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
from daleel.models import SPECS, make_lm
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, train_val_ids

from st_verifier_v2 import (
    BASELINE_RUN,
    ROLLOUTS,
    VerifyStatistics,
    exemplar_block,
    label_sets,
)

V21_DEPLOY_RUN = (
    EXPERIMENTS_DIR / "20260714-202259-830666Z-t1-routed-v2-deploy-dev_in-fba4d257e3"
)
OLD_DEV_ST_JUDGE = (
    EXPERIMENTS_DIR
    / "20260714-194454-761621Z-t1-routed-v2-deploy-dev_in-c75de50ea4"
    / "predictions" / "st_judge_dev217.jsonl"
)
DEV_INPUT = Path("../resources/repos/Daleel2026/data/dev/dev_in.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--qwen-run", type=Path, required=True)
    ap.add_argument("--llama-run", type=Path, required=True)
    ap.add_argument("--deploy-run", type=Path, default=V21_DEPLOY_RUN,
                    help="routed deploy run whose ST firings to verify")
    ap.add_argument("--input", type=Path, default=DEV_INPUT)
    ap.add_argument("--old-st-judge", type=Path, default=OLD_DEV_ST_JUDGE)
    ap.add_argument("--tag", default="dev217")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Frozen exemplar block: all train gold STs + all train-OOF fired FPs.
    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold = {p: l for p, l in clean_task1_gold(t1, t2).items()
            if p in set(train_ids)}
    train_text = {int(r["paragraph_id"]): r["text"] for r in t1}
    baseline = {int(r["paragraph_id"]): set(r["labels"]) for r in read_jsonl(
        BASELINE_RUN / "predictions" / "preds.jsonl")}
    gold_st = sorted(p for p in gold if "ST" in gold[p])
    fired_fp = sorted(p for p in baseline
                      if "ST" in baseline[p] and "ST" not in gold[p])
    examples = (
        exemplar_block(gold_st, train_text, "ST PRESENT")
        + "\n\n"
        + exemplar_block(fired_fp, train_text, "ST ABSENT")
    )

    dev_rows = list(read_jsonl(args.input))
    text_of = {int(r["paragraph_id"]): r["text"] for r in dev_rows}
    genre_of = {int(r["paragraph_id"]): r["type"] for r in dev_rows}

    deploy_pred = next((args.deploy_run / "predictions").glob("*.jsonl"))
    fired = sorted(int(r["paragraph_id"]) for r in read_jsonl(deploy_pred)
                   if "ST" in r["labels"])
    if args.limit:
        fired = fired[: args.limit]
    qwen, llama = label_sets(args.qwen_run), label_sets(args.llama_run)
    eligible = {p for p in fired
                if "ST" not in qwen.get(p, set()) or "ST" not in llama.get(p, set())}

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
                        result = verify(examples=examples, text=text_of[p],
                                        genre=genre_of[p])
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

    old_rows = list(read_jsonl(args.old_st_judge))
    covered = {int(r["paragraph_id"]) for r in old_rows}
    out_rows = [
        {**r, "present": r["present"] and int(r["paragraph_id"]) not in drops}
        for r in old_rows
    ] + [
        {"paragraph_id": p, "present": False}
        for p in drops if p not in covered
    ]

    metrics = {
        "milestone": "ST verifier v2 deploy twin (adopted OOF gate 0.7298)",
        "model": spec.key,
        "rollouts": ROLLOUTS,
        "temperature": 0.7,
        "deploy_run": str(args.deploy_run),
        "qwen_run": str(args.qwen_run),
        "llama_run": str(args.llama_run),
        "n_fired": len(fired),
        "n_eligible_by_cosignal": len(eligible),
        "n_majority_absent": len(majority_absent),
        "n_dropped": len(drops),
        "dropped_paragraphs": drops,
        "rollout_disagreement": sum(1 for v in verdicts.values()
                                    if len(set(v)) > 1),
        "n_program_errors": len(errors),
        "errors_sample": errors[:5],
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t1-st-verifier-v2-deploy-{spec.key}-{args.tag}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / f"st_judge_v2_{args.tag}.jsonl", out_rows)
    write_jsonl(pred_dir / "verifier_verdicts.jsonl", [
        {"paragraph_id": p, "rollout_present": verdicts[p],
         "eligible": p in eligible, "dropped": p in drops}
        for p in fired
    ])
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "st_verifier_v2_deploy.py", "argv": sys.argv[1:],
         "model": spec.__dict__, "dspy_version": dspy.__version__},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nst-judge file: {pred_dir / f'st_judge_v2_{args.tag}.jsonl'}")


if __name__ == "__main__":
    main()
