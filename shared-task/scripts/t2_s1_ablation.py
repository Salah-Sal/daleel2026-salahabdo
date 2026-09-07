"""Leave-one-out ablation of the T2 S1 structural re-decode (report-only).

Camera-ready analysis requested by reviewer 29cU ("a clearer ablation of the
individual parts of the structural re-decoding"). Ablations report, they never
adopt: no gate is defined here and nothing in this script can change a frozen
system. Zero API cost; reads the same cached artifacts as
``t2_structural_gate.py`` and re-fits every per-fold quantity inside each
variant, so each row is the honest score of *that* system, not the full
system's fit reused.

Variants (each removes exactly one part; the rest are fitted as in S1):

  full          the deployed S1 bundle (reproduces the recorded 0.7206)
  no-calibration   prior reweighting off (all class weights 1.0)
  no-blend      lambda pinned to 0: emissions are the calibrated encoder
                distribution alone, so the v3 label carries no weight
  no-genre      one pooled transition matrix for both genres instead of
                genre-conditioned ones
  no-viterbi    no sequence model: each span takes the argmax of its own
                emission
  no-rule-a     the CO cross-system consensus step is skipped

Example (from shared-task/):
  uv run scripts/t2_s1_ablation.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t2_structural_gate import (  # noqa: E402
    ALPHA,
    EPS,
    LABELS,
    LAMBDA_GRID,
    QUOTE_RUN,
    ROLE_RUNS,
    ENCODER_RUN,
    calibrate,
    dedup_spans,
    encoder_expected_prior,
    fit_sequence_model,
    load_role_runs,
    load_segments,
    mass_prior,
    rule_a,
    span_encoder_dist,
    viterbi,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.data import TRAIN_TASK2, load_records  # noqa: E402
from daleel.io import read_jsonl  # noqa: E402
from daleel.metrics import Span, span_partial_f1  # noqa: E402
from daleel.runtime import EXPERIMENTS_DIR  # noqa: E402
from daleel.splits import clean_task2_gold  # noqa: E402

VARIANTS = ("full", "no-calibration", "no-blend", "no-genre", "no-viterbi",
            "no-rule-a")


def emissions_for(spans, enc_dists, lam):
    out = []
    for s, enc in zip(spans, enc_dists):
        onehot = {c: 1.0 if c == s.label else 0.0 for c in LABELS}
        out.append({c: (1 - lam) * enc[c] + lam * onehot[c] for c in LABELS})
    return out


def decode_variant(spans, enc_dists, lam, seq_model, genre, variant):
    """Re-label one paragraph's span sequence under a leave-one-out variant."""
    if not spans:
        return []
    em = emissions_for(spans, enc_dists, lam)
    if variant == "no-viterbi":
        labels = [max(LABELS, key=lambda c: e[c]) for e in em]
    else:
        key = "__global__" if variant == "no-genre" else genre
        start_lp, trans_lp = seq_model.get(key, seq_model["__global__"])
        labels = viterbi(em, start_lp, trans_lp)
    return [Span(s.start, s.end, lab) for s, lab in zip(spans, labels)]


def run_variant(variant, pred, gold, segments, raw_enc, quote, genre_of, fold_of):
    """Full 5-fold cross-fitted run of one variant; returns its metrics."""
    lam_grid = [0.0] if variant == "no-blend" else LAMBDA_GRID
    relabeled, fitted = {}, {}
    rule_a_changes = 0

    for f in range(5):
        fit_pids = [p for p in pred if fold_of[p] != f]
        eval_pids = [p for p in pred if fold_of[p] == f]

        if variant == "no-calibration":
            weights = {c: 1.0 for c in LABELS}
        else:
            weights_gold = mass_prior([gold.get(p, []) for p in fit_pids])
            weights_enc = encoder_expected_prior(segments, fit_pids)
            weights = {c: weights_gold[c] / max(weights_enc[c], EPS) for c in LABELS}

        seq_model = fit_sequence_model(fit_pids, gold, genre_of)
        cal = {p: [calibrate(d, weights) for d in raw_enc[p]] for p in pred}

        best_lam, best_f1 = None, -1.0
        for lam in lam_grid:
            fit_out = {p: decode_variant(pred[p], cal[p], lam, seq_model,
                                         genre_of[p], variant) for p in fit_pids}
            f1 = span_partial_f1({p: gold[p] for p in fit_pids}, fit_out)["f1"]
            if f1 > best_f1:
                best_lam, best_f1 = lam, f1

        for p in eval_pids:
            dec = decode_variant(pred[p], cal[p], best_lam, seq_model,
                                 genre_of[p], variant)
            if variant == "no-rule-a":
                final, changed = dec, 0
            else:
                final, changed = rule_a(dec, quote.get(p, []))
            relabeled[p] = dedup_spans(final)
            rule_a_changes += changed
        fitted[str(f)] = {"lambda": best_lam, "lambda_fit_f1": round(best_f1, 4)}

    g_all = {p: gold[p] for p in pred}
    pooled = span_partial_f1(g_all, relabeled)["f1"]
    per_fold = {}
    for f in range(5):
        sub = [p for p in pred if fold_of[p] == f]
        per_fold[str(f)] = round(
            span_partial_f1({p: gold[p] for p in sub},
                            {p: relabeled[p] for p in sub})["f1"], 4)
    n_changed = sum(1 for p in pred for a, b in zip(pred[p], relabeled[p])
                    if a.label != b.label)
    return {
        "pooled_f1": round(pooled, 4),
        "per_fold_f1": per_fold,
        "n_label_changes": n_changed,
        "n_rule_a_changes": rule_a_changes,
        "lambda_per_fold": {f: v["lambda"] for f, v in fitted.items()},
        "fitted": fitted,
    }


def main() -> None:
    pred, text_of, genre_of, fold_of = load_role_runs()
    gold = {int(p): s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
            if int(p) in pred}
    segments = load_segments(
        EXPERIMENTS_DIR / ENCODER_RUN / "predictions" / "oof_task_2_segment_scores.jsonl")
    quote = {int(r["paragraph_id"]): [Span(s["start_offset"], s["end_offset"], s["label"])
                                      for s in r["labels"]]
             for r in read_jsonl(EXPERIMENTS_DIR / QUOTE_RUN / "predictions" / "preds.jsonl")}
    raw_enc = {p: [span_encoder_dist(segments[p], s) for s in pred[p]] for p in pred}

    g_all = {p: gold[p] for p in pred}
    recorded_f1 = round(span_partial_f1(g_all, pred)["f1"], 4)

    results = {}
    for variant in VARIANTS:
        t0 = time.time()
        results[variant] = run_variant(variant, pred, gold, segments, raw_enc,
                                       quote, genre_of, fold_of)
        results[variant]["seconds"] = round(time.time() - t0, 1)
        print(f"{variant:16s} {results[variant]['pooled_f1']:.4f} "
              f"({results[variant]['seconds']}s)", flush=True)

    full_f1 = results["full"]["pooled_f1"]
    for variant, res in results.items():
        res["delta_vs_full"] = round(res["pooled_f1"] - full_f1, 4)
        res["delta_vs_v3"] = round(res["pooled_f1"] - recorded_f1, 4)

    metrics = {
        "analysis": "S1 leave-one-out ablation (report-only; no gate, no adoption)",
        "encoder_run": ENCODER_RUN,
        "quote_run": QUOTE_RUN,
        "recorded_v3_f1": recorded_f1,
        "full_s1_f1": full_f1,
        "n_spans": sum(len(s) for s in pred.values()),
        "n_paragraphs": len(pred),
        "variants": results,
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t2-s1-ablation-loo-n{len(pred)}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "t2_s1_ablation.py", "argv": sys.argv[1:],
         "role_runs": ROLE_RUNS, "encoder_run": ENCODER_RUN,
         "quote_run": QUOTE_RUN, "labels": LABELS,
         "lambda_grid": LAMBDA_GRID, "alpha": ALPHA, "variants": VARIANTS,
         "purpose": "camera-ready reviewer 29cU point 5; report-only"},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
