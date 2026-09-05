"""Deploy twin of the v2.1 ST seed judge (judge_stage_test.py, ST drop-only leg).

The dev seed (st_judge_dev217.jsonl, 2026-07-14) was produced by running
judge_stage_test.py --drop-labels ST --arbiter-labels '' over the pass-1
routed deploy output and deriving {paragraph_id, present} rows for its
ST-fired paragraphs (invocation preserved in
experiments/20260714-t1-judge-stage-gemma-4-31b-paid-val/config.json —
that dir name predates timestamped naming and was reused, so the config
there IS the dev judge run, not a val one). judge_stage_test.py scores
against train gold, which does not exist for eval inputs, so this twin
applies the byte-identical JudgeStage invocation (ChainOfThought, T=0,
max-tokens 6000, same 4-attempt containment ladder, errors keep ST) to a
routed prediction file and emits the seed file directly, no gold
dependency. JudgeStage with drop_labels=("ST",) makes zero LLM calls for
paragraphs without a proposed ST, so judging only ST-fired rows is call-
and output-equivalent to the dev run over all rows.

Control (free via the DSPy disk cache — identical calls replay):
  uv run scripts/st_judge_seed_deploy.py --model gemma-4-31b-paid \
      --preds experiments/20260714-194454-761621Z-t1-routed-v2-deploy-dev_in-c75de50ea4/predictions/task_1.jsonl \
      --tag dev217-control
  -> predictions/st_judge_dev217-control.jsonl must equal the frozen
     st_judge_dev217.jsonl byte-for-byte.

Eval:
  uv run scripts/st_judge_seed_deploy.py --model gemma-4-31b-paid \
      --preds experiments/<t1-routed-v2-deploy ... test_in pass 1>/predictions/task_1.jsonl \
      --tag test213
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dspy

from daleel.dspy_programs import JudgeStage
from daleel.io import read_jsonl, write_jsonl
from daleel.models import SPECS, make_lm
from daleel.runtime import EXPERIMENTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--preds", type=Path, required=True,
                    help="routed task 1 prediction file whose ST firings to judge")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000)
    args = ap.parse_args()

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens))
    stage = JudgeStage(cot=True, drop_labels=("ST",), arbiter_labels=())

    records = list(read_jsonl(args.preds))
    fired = [r for r in records if "ST" in r["labels"]]
    errors: list[str] = []

    def judge_one(r: dict) -> bool:
        last = None
        for attempt in range(4):
            try:
                pred = stage(text=r["text"], genre=r["type"],
                             proposed=set(r["labels"]))
                return "ST" in pred.adu_labels
            except Exception as e:  # containment: an error keeps ST
                last = f"{r['paragraph_id']}: {type(e).__name__}: {e}"
                time.sleep(min(75, 25 * (attempt + 1)))
        errors.append(last)
        return True

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        verdicts = list(pool.map(judge_one, fired))
    wall = time.time() - t0

    rows = sorted(
        ({"paragraph_id": r["paragraph_id"], "present": present}
         for r, present in zip(fired, verdicts)),
        key=lambda row: row["paragraph_id"],
    )

    metrics = {
        "milestone": "v2.1 ST seed judge deploy twin (dev process scripted for eval)",
        "model": spec.key,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "source_preds": str(args.preds),
        "n_source_paragraphs": len(records),
        "n_st_fired": len(fired),
        "n_judged_absent": sum(1 for row in rows if not row["present"]),
        "n_program_errors": len(errors),
        "errors_sample": errors[:5],
        "wall_seconds": round(wall, 1),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t1-st-seed-judge-deploy-{spec.key}-{args.tag}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / f"st_judge_{args.tag}.jsonl", rows)
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "st_judge_seed_deploy.py", "argv": sys.argv[1:],
         "model": spec.__dict__, "temperature": 0.0,
         "max_tokens": args.max_tokens, "dspy_version": dspy.__version__},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nseed judge file: {pred_dir / f'st_judge_{args.tag}.jsonl'}")


if __name__ == "__main__":
    main()
