"""T2 "S2" confidence-selective span combiner (registered gate, CREATIVE_HEADROOM_RESEARCH.md).

One multinomial logistic regression per fold (lbfgs, L2 C=1.0,
class_weight='balanced') over frozen span features (calibrated encoder
posterior, v3/S1/quote/T1-v2.2 labels, genre, length, position,
neighbor S1 labels). Override rule: replace the S1 label with the
combiner argmax ONLY when argmax != S1 label AND max posterior >= tau
(tau per fold from grid {0.50..0.95} on the 4 fitting folds), then
exact-duplicate dedup. Gate: pooled OOF >= recorded + 0.02 (= 0.7406)
AND >= 3/5 fold wins. Zero API cost.

Example (from shared-task/):
  uv run scripts/t2_s2_combiner_gate.py
"""

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression

from daleel.data import TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold

from t2_structural_gate import (  # noqa: E402 — sibling script import
    ENCODER_RUN,
    EPS,
    LABELS,
    QUOTE_RUN,
    calibrate,
    dedup_spans,
    dominant_label,
    encoder_expected_prior,
    load_role_runs,
    load_segments,
    mass_prior,
    span_encoder_dist,
)

S1_RUN = "20260715-001221-t2-s1-structural-camelbert-quarter-n430"
T1_RUN = "20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa"
TAU_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]
LABEL_INDEX = {c: i for i, c in enumerate(LABELS)}
GATE_MARGIN = 0.02


def onehot(label, with_none=False):
    vec = [0.0] * (len(LABELS) + (1 if with_none else 0))
    if label in LABEL_INDEX:
        vec[LABEL_INDEX[label]] = 1.0
    elif with_none:
        vec[-1] = 1.0
    return vec


def build_features(p, i, spans_v3, spans_s1, enc_cal, quote, t1_labels,
                   genre, text_len):
    s = spans_s1[i]
    prev = spans_s1[i - 1].label if i > 0 else None
    nxt = spans_s1[i + 1].label if i + 1 < len(spans_s1) else None
    qdom = dominant_label(quote, s)
    return (
        [enc_cal[i][c] for c in LABELS]
        + onehot(spans_v3[i].label)
        + onehot(s.label)
        + onehot(qdom, with_none=True)
        + [1.0 if c in t1_labels else 0.0 for c in LABELS]
        + [1.0 if genre == "editorial" else 0.0]
        + [math.log(max(s.end - s.start, 1)),
           s.start / max(text_len, 1),
           1.0 if i == 0 else 0.0,
           1.0 if i + 1 == len(spans_s1) else 0.0]
        + onehot(prev, with_none=True)
        + onehot(nxt, with_none=True)
    )


# columns standardized with fitting-fold mean/std (log length, rel start)
STANDARDIZED = (6 + 6 + 6 + 7 + 6 + 1, 6 + 6 + 6 + 7 + 6 + 1 + 1)


def main() -> None:
    v3_pred, text_of, genre_of, fold_of = load_role_runs()
    gold = {int(p): s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
            if int(p) in v3_pred}
    segments = load_segments(
        EXPERIMENTS_DIR / ENCODER_RUN / "predictions" / "oof_task_2_segment_scores.jsonl")
    quote = {int(r["paragraph_id"]): [Span(s["start_offset"], s["end_offset"], s["label"])
                                      for s in r["labels"]]
             for r in read_jsonl(EXPERIMENTS_DIR / QUOTE_RUN / "predictions" / "preds.jsonl")}
    t1 = {int(r["paragraph_id"]): set(r["labels"])
          for r in read_jsonl(EXPERIMENTS_DIR / T1_RUN / "predictions" / "preds.jsonl")}
    s1_pred = {}
    for r in read_jsonl(EXPERIMENTS_DIR / S1_RUN / "predictions" / "relabeled_task_2.jsonl"):
        p = int(r["paragraph_id"])
        s1_pred[p] = [Span(s["start_offset"], s["end_offset"], s["label"])
                      for s in r["labels"]]
        if [(a.start, a.end) for a in s1_pred[p]] != \
           [(a.start, a.end) for a in v3_pred[p]]:
            raise ValueError(f"S1/v3 span misalignment in paragraph {p}")

    raw_enc = {p: [span_encoder_dist(segments[p], s) for s in v3_pred[p]]
               for p in v3_pred}

    relabeled, pre_dedup = {}, {}
    fitted, flip_rows = {}, []
    for f in range(5):
        fit_pids = [p for p in v3_pred if fold_of[p] != f]
        eval_pids = [p for p in v3_pred if fold_of[p] == f]

        weights_gold = mass_prior([gold.get(p, []) for p in fit_pids])
        weights_enc = encoder_expected_prior(segments, fit_pids)
        cal_w = {c: weights_gold[c] / max(weights_enc[c], EPS) for c in LABELS}
        enc_cal = {p: [calibrate(d, cal_w) for d in raw_enc[p]] for p in v3_pred}

        def features_of(p):
            return [build_features(p, i, v3_pred[p], s1_pred[p], enc_cal[p],
                                   quote.get(p, []), t1.get(p, set()),
                                   genre_of[p], len(text_of[p]))
                    for i in range(len(s1_pred[p]))]

        X_fit, y_fit = [], []
        for p in fit_pids:
            feats = features_of(p)
            for i, s in enumerate(s1_pred[p]):
                dom = dominant_label(gold.get(p, []), s)
                if dom is not None:
                    X_fit.append(feats[i])
                    y_fit.append(dom)
        X_fit = np.asarray(X_fit)
        mu = X_fit[:, STANDARDIZED[0]:STANDARDIZED[1] + 1].mean(axis=0)
        sd = X_fit[:, STANDARDIZED[0]:STANDARDIZED[1] + 1].std(axis=0) + EPS

        def standardize(X):
            X = np.asarray(X, dtype=float)
            X[:, STANDARDIZED[0]:STANDARDIZED[1] + 1] = (
                X[:, STANDARDIZED[0]:STANDARDIZED[1] + 1] - mu) / sd
            return X

        clf = LogisticRegression(
            solver="lbfgs", C=1.0, class_weight="balanced", max_iter=1000)
        clf.fit(standardize(X_fit), y_fit)

        def apply_tau(pids, tau):
            out = {}
            for p in pids:
                if not s1_pred[p]:
                    out[p] = []
                    continue
                proba = clf.predict_proba(standardize(features_of(p)))
                spans = []
                for i, s in enumerate(s1_pred[p]):
                    j = int(np.argmax(proba[i]))
                    top = clf.classes_[j]
                    if top != s.label and proba[i][j] >= tau:
                        spans.append(Span(s.start, s.end, top))
                    else:
                        spans.append(s)
                out[p] = dedup_spans(spans)
            return out

        best_tau, best_f1 = None, -1.0
        g_fit = {p: gold[p] for p in fit_pids}
        for tau in TAU_GRID:
            f1 = span_partial_f1(g_fit, apply_tau(fit_pids, tau))["f1"]
            if f1 > best_f1:
                best_tau, best_f1 = tau, f1

        for p in eval_pids:
            if not s1_pred[p]:
                pre_dedup[p], relabeled[p] = [], []
                continue
            proba = clf.predict_proba(standardize(features_of(p)))
            spans = []
            for i, s in enumerate(s1_pred[p]):
                j = int(np.argmax(proba[i]))
                top = clf.classes_[j]
                if top != s.label and proba[i][j] >= best_tau:
                    spans.append(Span(s.start, s.end, top))
                    flip_rows.append({
                        "paragraph_id": p, "span_index": i,
                        "start_offset": s.start, "end_offset": s.end,
                        "from": s.label, "to": top,
                        "posterior": round(float(proba[i][j]), 4),
                        "dominant_gold": dominant_label(gold.get(p, []), s),
                    })
                else:
                    spans.append(s)
            pre_dedup[p] = spans
            relabeled[p] = dedup_spans(spans)
        fitted[str(f)] = {
            "tau": best_tau,
            "tau_fit_f1": round(best_f1, 4),
            "n_train_spans": len(y_fit),
            "train_class_counts": dict(sorted(Counter(y_fit).items())),
        }

    g_all = {p: gold[p] for p in v3_pred}
    rec_f1 = span_partial_f1(g_all, s1_pred)["f1"]
    new_f1 = span_partial_f1(g_all, relabeled)["f1"]

    per_fold, wins = {}, 0
    for f in range(5):
        sub = [p for p in v3_pred if fold_of[p] == f]
        g = {p: gold[p] for p in sub}
        r0 = span_partial_f1(g, {p: s1_pred[p] for p in sub})["f1"]
        r1 = span_partial_f1(g, {p: relabeled[p] for p in sub})["f1"]
        wins += r1 >= r0
        per_fold[str(f)] = {"recorded": round(r0, 4), "combined": round(r1, 4)}

    flips = flip_rows
    flip_matrix = Counter((r["from"], r["to"]) for r in flips)
    with_gold = [r for r in flips if r["dominant_gold"] is not None]
    flip_correct = sum(r["dominant_gold"] == r["to"] for r in with_gold)
    gate = round(rec_f1 + GATE_MARGIN, 4)

    def ratios(pred_map):
        pm = mass_prior(pred_map.values())
        gm = mass_prior(g_all.values())
        return {c: round(pm[c] / max(gm[c], EPS), 2) for c in LABELS}

    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md registered T2 S2 combiner",
        "s1_run": S1_RUN, "t1_run": T1_RUN,
        "recorded_f1": round(rec_f1, 4),
        "combined_f1": round(new_f1, 4),
        "delta": round(new_f1 - rec_f1, 4),
        "n_flips": len(flips),
        "flip_matrix": {f"{a}->{b}": n for (a, b), n in
                        sorted(flip_matrix.items(), key=lambda kv: -kv[1])},
        "flip_precision_vs_dominant_gold": (
            round(flip_correct / len(with_gold), 4) if with_gold else None),
        "n_flips_with_gold_overlap": len(with_gold),
        "n_dedup_dropped": sum(len(pre_dedup[p]) - len(relabeled[p])
                               for p in v3_pred),
        "mass_ratio_vs_gold_s1": ratios(s1_pred),
        "mass_ratio_vs_gold_s2": ratios(relabeled),
        "fitted": fitted,
        "gate": {
            "rule": f"pooled >= {gate} AND >=3/5 fold wins",
            "per_fold": per_fold,
            "fold_wins": wins,
            "adopted": bool(new_f1 >= rec_f1 + GATE_MARGIN and wins >= 3),
        },
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t2-s2-combiner-n{len(v3_pred)}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "relabeled_task_2.jsonl", [
        {"paragraph_id": p, "text": text_of[p], "type": genre_of[p],
         "labels": [{"label": s.label, "start_offset": s.start,
                     "end_offset": s.end} for s in relabeled[p]]}
        for p in sorted(relabeled)
    ])
    write_jsonl(pred_dir / "flips.jsonl", flips)
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "t2_s2_combiner_gate.py", "argv": sys.argv[1:],
         "s1_run": S1_RUN, "t1_run": T1_RUN, "encoder_run": ENCODER_RUN,
         "quote_run": QUOTE_RUN, "tau_grid": TAU_GRID,
         "model": "LogisticRegression(lbfgs, C=1.0, class_weight=balanced, max_iter=1000)",
         "standardized_columns": list(STANDARDIZED),
         "registration": "CREATIVE_HEADROOM_RESEARCH.md S2 (commit 53382af)"},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
