"""Isolated test of the Task 2 relabel stage on frozen extractions (v2 design).

Loads a completed quote-program prediction file, keeps every span boundary
exactly as extracted, and runs only the RelabelStage over the spans. Scored
paired against the same D9-cleaned gold as the source run, so the delta is
attributable to relabeling alone. Motivation and ceiling: the 2026-07-10
error anatomy — extraction covers 100% of gold span mass, the oracle
relabel-only ceiling is 0.8792 vs the 0.6805 champion baseline.

Example (from shared-task/):
  uv run scripts/relabel_stage_test.py --model gemma-4-31b-paid \
      --preds experiments/20260709-t2-quote-gemma-4-31b-paid-val/predictions/preds.jsonl

Artifacts land in experiments/<date>-t2-relabel-stage-<model>-val/ with the
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

from daleel.data import TRAIN_TASK2, load_records
from daleel.dspy_programs import RelabelStage
from daleel.io import write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.models import SPECS, make_lm
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold
from daleel.submission import validate_task2_records

CHAMPION_PREDS = (
    EXPERIMENTS_DIR
    / "20260709-t2-quote-gemma-4-31b-paid-val"
    / "predictions"
    / "preds.jsonl"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), required=True)
    ap.add_argument("--preds", type=Path, default=CHAMPION_PREDS,
                    help="quote-program prediction file whose spans to relabel")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--no-cot", action="store_true")
    ap.add_argument("--blind", action="store_true",
                    help="withhold the extractor's draft labels (anchoring ablation)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N paragraphs only (smoke test)")
    args = ap.parse_args()

    spec = SPECS[args.model]
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens))
    stage = RelabelStage(cot=not args.no_cot, blind=args.blind)

    records = [json.loads(line) for line in args.preds.open()]
    if args.limit:
        records = records[: args.limit]
    before = {
        r["paragraph_id"]: [
            Span(s["start_offset"], s["end_offset"], s["label"]) for s in r["labels"]
        ]
        for r in records
    }

    def relabel_one(r: dict):
        last = None
        for attempt in range(4):
            try:
                return stage(text=r["text"], genre=r["type"],
                             spans=before[r["paragraph_id"]]), None
            except Exception as e:  # containment: never drop a row
                last = f"{r['paragraph_id']}: {type(e).__name__}: {e}"
                time.sleep(min(75, 25 * (attempt + 1)))
        return None, last

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(relabel_one, records))
    wall = time.time() - t0

    after, out_records, errors = {}, [], []
    n_changed = n_mismatch = 0
    for r, (pred, err) in zip(records, results):
        pid = r["paragraph_id"]
        if err:
            errors.append(err)
        spans = list(pred.spans) if pred else before[pid]  # error -> unchanged
        n_changed += pred.n_changed if pred else 0
        n_mismatch += bool(pred.length_mismatch) if pred else 0
        after[pid] = spans
        out_records.append(
            {
                "paragraph_id": pid,
                "text": r["text"],
                "type": r["type"],
                "labels": [
                    {"label": s.label, "start_offset": s.start, "end_offset": s.end}
                    for s in spans
                ],
            }
        )

    gold = {
        k: v
        for k, v in clean_task2_gold(load_records(TRAIN_TASK2)).items()
        if k in before
    }
    official_before = span_partial_f1(gold, before)
    official_after = span_partial_f1(gold, after)

    metrics = {
        "task": 2,
        "stage": "relabel-blind" if args.blind else "relabel",
        "model": spec.key,
        "litellm_id": spec.litellm_id,
        "source_preds": str(args.preds),
        "n_paragraphs": len(records),
        "cot": not args.no_cot,
        "n_program_errors": len(errors),
        "n_length_mismatch": n_mismatch,
        "n_spans": sum(len(v) for v in before.values()),
        "n_spans_relabeled": n_changed,
        "wall_seconds": round(wall, 1),
        "errors_sample": errors[:5],
        "s1_official_span_f1_before": round(official_before["f1"], 4),
        "s1_official_span_f1": round(official_after["f1"], 4),
        "delta": round(official_after["f1"] - official_before["f1"], 4),
        "precision": round(official_after["precision"], 4),
        "recall": round(official_after["recall"], 4),
        "per_label_f1_before": {
            label: round(v["f1"], 4) for label, v in official_before["per_label"].items()
        },
        "per_label_f1": {
            label: round(v["f1"], 4) for label, v in official_after["per_label"].items()
        },
    }

    problems = validate_task2_records(out_records)
    if problems:
        metrics["validator_errors"] = problems[:10]

    suffix = ("-blind" if args.blind else "") + (f"-limit{args.limit}" if args.limit else "")
    run_name = f"{time.strftime('%Y%m%d')}-t2-relabel-stage-{spec.key}-val{suffix}"
    exp_dir = EXPERIMENTS_DIR / run_name
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "preds.jsonl", out_records)
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "script": "relabel_stage_test.py",
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
