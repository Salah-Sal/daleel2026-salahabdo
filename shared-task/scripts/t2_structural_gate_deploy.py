"""S1 structural bundle — deploy twin (CREATIVE_HEADROOM_RESEARCH.md, adopted gate).

Applies the adopted S1 re-decode to the v3 dev spans. Same frozen rules
as the gate run (t2_structural_gate.py, registration commit 46368f9),
fitting on ALL 430 train paragraphs instead of 4/5 folds:
  calibration priors + transitions + lambda fit on all 430 (lambda by
  the registered grid, argmax pooled train F1), encoder distributions
  for dev spans from the deploy encoder run (trained on all 430),
  Rule-A consensus against the zero-shot quote program's dev run.

Example (from shared-task/):
  uv run scripts/t2_structural_gate_deploy.py \
      --encoder-deploy-run experiments/<t2-encoder-deploy-...-dev_in-...>

An encoder ensemble must supply both sides of the calibration contract::

  uv run scripts/t2_structural_gate_deploy.py \
      --encoder-oof-run experiments/<t2-encoder-ensemble-oof> \
      --encoder-deploy-run experiments/<t2-encoder-ensemble-target>

Eval-phase example (target-side runs are dir NAMES under experiments/):
  uv run scripts/t2_structural_gate_deploy.py \
      --encoder-deploy-run experiments/<t2-encoder-deploy-...-test_in-...> \
      --v3-run <v3-decomposed-...-test_in dir name> \
      --quote-run <t2-quote-...-test dir name> \
      --tag test_in
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.data import TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold

from t2_structural_gate import (  # noqa: E402 — sibling script import
    ENCODER_RUN,
    LABELS,
    LAMBDA_GRID,
    QUOTE_RUN,
    decode_paragraph,
    dedup_spans,
    encoder_expected_prior,
    calibrate,
    fit_sequence_model,
    load_role_runs,
    load_segments,
    mass_prior,
    rule_a,
    span_encoder_dist,
)

V3_DEV_RUN = "20260711-033302-645188Z-v3-decomposed-gemma-4-31b-paid-both-dev_in-78718d42e5"
QUOTE_DEV_RUN = "20260711-032218-524548Z-t2-quote-gemma-4-31b-paid-dev-e955af86f9"
EPS = 1e-12


def load_task2_records(path):
    spans, text_of, genre_of = {}, {}, {}
    for r in read_jsonl(path):
        p = int(r["paragraph_id"])
        spans[p] = [Span(s["start_offset"], s["end_offset"], s["label"])
                    for s in r["labels"]]
        text_of[p] = r["text"]
        genre_of[p] = r["type"]
    return spans, text_of, genre_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--encoder-oof-run",
        type=Path,
        default=EXPERIMENTS_DIR / ENCODER_RUN,
        help=(
            "OOF segment-score run used to fit calibration weights and lambda "
            "(default: the adopted single-seed S1 run)"
        ),
    )
    ap.add_argument("--encoder-deploy-run", required=True,
                    help="t2-encoder-deploy experiment dir (target segment scores)")
    ap.add_argument("--v3-run", default=V3_DEV_RUN,
                    help="v3 decomposed run on the target: dir NAME under "
                    "experiments/ (default: the dev run)")
    ap.add_argument("--quote-run", default=QUOTE_DEV_RUN,
                    help="zero-shot quote run on the target for Rule-A consensus: "
                    "dir NAME under experiments/ (default: the dev run)")
    ap.add_argument("--tag", default="dev_in",
                    help="output dir suffix, e.g. test_in for the eval input")
    args = ap.parse_args()

    # ---- fit side: ALL 430 train paragraphs, same artifacts as the gate
    pred, _, genre_of, _ = load_role_runs()
    gold = {int(p): s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
            if int(p) in pred}
    train_segments = load_segments(
        args.encoder_oof_run / "predictions" / "oof_task_2_segment_scores.jsonl"
    )

    all_pids = sorted(pred)
    weights_gold = mass_prior([gold.get(p, []) for p in all_pids])
    weights_enc = encoder_expected_prior(train_segments, all_pids)
    weights = {c: weights_gold[c] / max(weights_enc[c], EPS) for c in LABELS}
    seq_model = fit_sequence_model(all_pids, gold, genre_of)

    raw_enc = {p: [span_encoder_dist(train_segments[p], s) for s in pred[p]]
               for p in all_pids}
    cal = {p: [calibrate(d, weights) for d in raw_enc[p]] for p in all_pids}
    best_lam, best_f1 = None, -1.0
    for lam in LAMBDA_GRID:
        out = {p: decode_paragraph(pred[p], cal[p], lam, seq_model, genre_of[p])
               for p in all_pids}
        f1 = span_partial_f1(gold, out)["f1"]
        if f1 > best_f1:
            best_lam, best_f1 = lam, f1

    # ---- target side (dev by default; --v3-run/--quote-run for eval)
    dev_pred, dev_text, dev_genre = load_task2_records(
        EXPERIMENTS_DIR / args.v3_run / "predictions" / "task_2.jsonl")
    quote_dev, _, _ = load_task2_records(
        EXPERIMENTS_DIR / args.quote_run / "predictions" / "task_2.jsonl")
    dev_segments = load_segments(
        Path(args.encoder_deploy_run) / "predictions" / "task_2_segment_scores.jsonl")

    missing = sorted(p for p in dev_pred if p not in dev_segments and dev_pred[p])
    if missing:
        raise ValueError(f"{len(missing)} dev paragraphs lack encoder segments, "
                         f"e.g. {missing[:5]}")

    relabeled, pre_dedup = {}, {}
    decode_changes = rule_a_changes = 0
    for p in sorted(dev_pred):
        enc = [calibrate(span_encoder_dist(dev_segments.get(p, []), s), weights)
               for s in dev_pred[p]]
        dec = decode_paragraph(dev_pred[p], enc, best_lam, seq_model,
                               dev_genre[p])
        decode_changes += sum(a.label != b.label for a, b in zip(dev_pred[p], dec))
        final, changed = rule_a(dec, quote_dev.get(p, []))
        pre_dedup[p] = final
        relabeled[p] = dedup_spans(final)
        rule_a_changes += changed

    def mass_fractions(span_map):
        return {c: round(v, 4) for c, v in
                mass_prior(span_map.values()).items()}

    n_changed = sum(1 for p in dev_pred
                    for a, b in zip(dev_pred[p], pre_dedup[p])
                    if a.label != b.label)
    n_dedup_dropped = sum(len(pre_dedup[p]) - len(relabeled[p]) for p in dev_pred)
    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md S1 deploy twin",
        "v3_dev_run": args.v3_run,
        "quote_dev_run": args.quote_run,
        "encoder_oof_run": str(args.encoder_oof_run),
        "encoder_deploy_run": str(args.encoder_deploy_run),
        "fit": {
            "lambda": best_lam,
            "lambda_fit_f1_all430_insample": round(best_f1, 4),
            "calibration_weights": {c: round(weights[c], 4) for c in LABELS},
        },
        "n_dev_paragraphs": len(dev_pred),
        "n_dev_spans": sum(len(s) for s in dev_pred.values()),
        "n_label_changes": n_changed,
        "n_decode_changes": decode_changes,
        "n_rule_a_changes": rule_a_changes,
        "n_dedup_dropped": n_dedup_dropped,
        "dev_mass_fraction_recorded": mass_fractions(dev_pred),
        "dev_mass_fraction_bundle": mass_fractions(relabeled),
        "train_gold_mass_fraction": {c: round(weights_gold[c], 4) for c in LABELS},
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t2-s1-deploy-{args.tag}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "task_2.jsonl", [
        {"paragraph_id": p, "text": dev_text[p], "type": dev_genre[p],
         "labels": [{"label": s.label, "start_offset": s.start,
                     "end_offset": s.end} for s in relabeled[p]]}
        for p in sorted(relabeled)
    ])
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "t2_structural_gate_deploy.py", "argv": sys.argv[1:],
         "gate_run": "20260715-001221-t2-s1-structural-camelbert-quarter-n430",
         "registration": "CREATIVE_HEADROOM_RESEARCH.md S1 (commit 46368f9)",
         "encoder_oof_run": str(args.encoder_oof_run), "quote_oof_run": QUOTE_RUN,
         "v3_dev_run": args.v3_run, "quote_dev_run": args.quote_run,
         "encoder_deploy_run": str(args.encoder_deploy_run)},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
