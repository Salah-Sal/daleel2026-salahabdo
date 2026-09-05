"""Isolated test of the Task 1 judge stage on frozen proposals (v2 design).

Loads a completed Task 1 prediction file and runs only the JudgeStage over
the proposed label sets: drop-only verification on the fired precision-sink
labels (CO/ST/OT), authoritative both-ways verdict on AN. Scored paired
against the same D9-cleaned gold as the source run, so the delta is
attributable to verification alone. Motivation and ceiling: the 2026-07-10
error anatomy — the oracle FP-judge ceiling is 0.8607 vs the 0.7024
champion baseline.

Example (from shared-task/):
  uv run scripts/judge_stage_test.py --model gemma-4-31b-paid

Artifacts land in experiments/<date>-t1-judge-stage-<model>-val/ with the
usual split: config.json + metrics.json committed, predictions/ gitignored.
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
from daleel.dspy_programs import JudgeStage
from daleel.io import write_jsonl
from daleel.metrics import task1_macro_f1
from daleel.models import SPECS, make_lm
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold
from daleel.submission import validate_task1_records

CHAMPION_PREDS = (
    EXPERIMENTS_DIR
    / "20260709-t1-zeroshot-gemma-4-31b-paid-val"
    / "predictions"
    / "preds.jsonl"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--preds", type=Path, default=CHAMPION_PREDS,
                    help="task 1 prediction file whose label sets to verify")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--no-cot", action="store_true")
    ap.add_argument("--compiled", type=Path, default=None, metavar="STATE_JSON",
                    help="load a compiled VerifyOne state into the stage "
                    "(compile_judge.py output); tags the run with the stem")
    ap.add_argument("--drop-labels", default="CO,ST,OT",
                    help="comma-separated labels verified drop-only")
    ap.add_argument("--arbiter-labels", default="AN",
                    help="comma-separated labels judged both ways ('' = none)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N paragraphs only (smoke test)")
    args = ap.parse_args()

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens))
    stage = JudgeStage(
        cot=not args.no_cot,
        drop_labels=tuple(l for l in args.drop_labels.split(",") if l),
        arbiter_labels=tuple(l for l in args.arbiter_labels.split(",") if l),
    )
    if args.compiled:
        stage.load(args.compiled)

    records = [json.loads(line) for line in args.preds.open()]
    if args.limit:
        records = records[: args.limit]
    before = {r["paragraph_id"]: set(r["labels"]) for r in records}

    def judge_one(r: dict):
        last = None
        for attempt in range(4):
            try:
                return stage(text=r["text"], genre=r["type"],
                             proposed=before[r["paragraph_id"]]), None
            except Exception as e:  # containment: never drop a row
                last = f"{r['paragraph_id']}: {type(e).__name__}: {e}"
                time.sleep(min(75, 25 * (attempt + 1)))
        return None, last

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(judge_one, records))
    wall = time.time() - t0

    after, out_records, errors = {}, [], []
    drop_counts: dict[str, int] = {}
    add_counts: dict[str, int] = {}
    for r, (pred, err) in zip(records, results):
        pid = r["paragraph_id"]
        if err:
            errors.append(err)
        labels = set(pred.adu_labels) if pred else before[pid]  # error -> unchanged
        for label in (pred.dropped if pred else []):
            drop_counts[label] = drop_counts.get(label, 0) + 1
        for label in (pred.added if pred else []):
            add_counts[label] = add_counts.get(label, 0) + 1
        after[pid] = labels
        out_records.append(
            {"paragraph_id": pid, "text": r["text"], "type": r["type"],
             "labels": sorted(labels)}
        )

    t1_all, t2_all = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    gold = {k: v for k, v in clean_task1_gold(t1_all, t2_all).items() if k in before}
    official_before = task1_macro_f1(gold, before)
    official_after = task1_macro_f1(gold, after)

    metrics = {
        "task": 1,
        "stage": "judge",
        "model": spec.key,
        "litellm_id": spec.litellm_id,
        "compiled": str(args.compiled) if args.compiled else None,
        "drop_labels": sorted(stage.drop_labels),
        "arbiter_labels": sorted(stage.arbiter_labels),
        "source_preds": str(args.preds),
        "n_paragraphs": len(records),
        "cot": not args.no_cot,
        "n_program_errors": len(errors),
        "n_dropped_by_label": drop_counts,
        "n_added_by_label": add_counts,
        "wall_seconds": round(wall, 1),
        "errors_sample": errors[:5],
        "s1_official_macro_f1_before": round(official_before["macro_f1"], 4),
        "s1_official_macro_f1": round(official_after["macro_f1"], 4),
        "delta": round(official_after["macro_f1"] - official_before["macro_f1"], 4),
        "micro_f1": round(official_after["micro_f1"], 4),
        "s2_rare_recall": {
            label: round(official_after["per_label"][label]["recall"], 4)
            for label in ("ST", "CO")
        },
        "per_label_f1_before": {
            label: round(v["f1"], 4)
            for label, v in official_before["per_label"].items()
        },
        "per_label_f1": {
            label: round(v["f1"], 4)
            for label, v in official_after["per_label"].items()
        },
    }

    problems = validate_task1_records(out_records)
    if problems:
        metrics["validator_errors"] = problems[:10]

    source = "optval" if "optval" in str(args.preds) else "val"
    suffix = (f"-{args.compiled.stem}" if args.compiled else "") + (
        f"-limit{args.limit}" if args.limit else ""
    )
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}-t1-judge-stage-{spec.key}-{source}{suffix}"
    exp_dir = EXPERIMENTS_DIR / run_name
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "preds.jsonl", out_records)
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "script": "judge_stage_test.py",
                "argv": sys.argv[1:],
                "model": spec.__dict__,
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "dspy_version": dspy.__version__,
            },
            indent=2,
        )
    )
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\npredictions: {pred_dir / 'preds.jsonl'}")


if __name__ == "__main__":
    main()
