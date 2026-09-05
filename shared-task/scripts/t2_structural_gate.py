"""T2 structural re-decode bundle "S1" (registered gate, CREATIVE_HEADROOM_RESEARCH.md).

Pure relabeling of the recorded v3 spans (extraction untouched):
  1. encoder span distributions (overlap-weighted segment softmax, NONE
     dropped and renormalized; zero-overlap -> uniform)
  2. per-class prior calibration (gold char-mass prior / encoder expected
     char-mass prior, tau=1, fitting folds only)
  3. emission blend p = (1-lambda)*p_enc_cal + lambda*onehot(v3 label),
     lambda per fold from grid {0.0..0.9} maximizing pooled span F1 on
     the 4 fitting folds
  4. Viterbi re-decode with genre-conditioned transitions + start priors
     (gold label sequences, fitting folds, Laplace alpha=1)
  5. Rule-A CO consensus: decoded-CO spans take the quote champion's
     dominant label when it exists and differs from CO

Gate: pooled OOF span partial-F1 >= recorded + 0.02 AND >=3/5 fold wins.
Zero API cost; no LM calls.

Example (from shared-task/):
  uv run scripts/t2_structural_gate.py
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.data import TRAIN_TASK2, load_records
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task2_gold

ROLE_RUNS = [
    "20260710-193247-709106Z-v3-role-gepa-gemma-4-31b-paid-both-outer0-inner0-1447e7ced5",
    "20260710-210609-072093Z-v3-role-gepa-gemma-4-31b-paid-both-outer1-inner0-d9995eddba",
    "20260710-211213-843183Z-v3-role-gepa-gemma-4-31b-paid-both-outer2-inner0-1ce74f2e9a",
    "20260710-215536-635075Z-v3-role-gepa-gemma-4-31b-paid-both-outer3-inner0-30cdb1492f",
    "20260710-221215-667710Z-v3-role-gepa-gemma-4-31b-paid-both-outer4-inner0-49f2e2dad5",
]
ENCODER_RUN = "20260715-033553-432753Z-encoder-t2-camelbert-msa-quarter-both-legacy-train-78b1f7b0d3"
QUOTE_RUN = "20260710-190630-477294Z-t2-quote-gemma-4-31b-paid-alltrain-5aa0011e3b"
LABELS = ("CO", "AS", "TE", "ST", "AN", "OT")
LAMBDA_GRID = [round(0.1 * i, 1) for i in range(10)]
ALPHA = 1.0  # Laplace smoothing for transitions/starts
EPS = 1e-12
GATE_MARGIN = 0.02


def load_role_runs():
    pred, text_of, genre_of, fold_of = {}, {}, {}, {}
    for f, run in enumerate(ROLE_RUNS):
        for r in read_jsonl(EXPERIMENTS_DIR / run / "predictions" / "outer_task_2.jsonl"):
            p = int(r["paragraph_id"])
            pred[p] = [Span(s["start_offset"], s["end_offset"], s["label"])
                       for s in r["labels"]]
            text_of[p] = r["text"]
            genre_of[p] = r["type"]
            fold_of[p] = f
    return pred, text_of, genre_of, fold_of


def load_segments(path):
    segments = defaultdict(list)
    for r in read_jsonl(path):
        probs = {c: r["scores"].get(c, 0.0) for c in LABELS}
        z = sum(probs.values())
        probs = {c: v / z for c, v in probs.items()} if z > EPS else {
            c: 1.0 / len(LABELS) for c in LABELS}
        segments[int(r["paragraph_id"])].append(
            (r["start_offset"], r["end_offset"], probs))
    for rows in segments.values():
        rows.sort()
    return segments


def span_encoder_dist(segs, span):
    acc = {c: 0.0 for c in LABELS}
    total = 0.0
    for start, end, probs in segs:
        ov = max(0, min(end, span.end) - max(start, span.start))
        if ov > 0:
            total += ov
            for c in LABELS:
                acc[c] += ov * probs[c]
    if total <= 0:
        return {c: 1.0 / len(LABELS) for c in LABELS}
    return {c: acc[c] / total for c in LABELS}


def mass_prior(span_lists):
    mass = {c: 0.0 for c in LABELS}
    for spans in span_lists:
        for s in spans:
            mass[s.label] += s.end - s.start
    z = sum(mass.values())
    return {c: mass[c] / z for c in mass} if z > 0 else {
        c: 1.0 / len(LABELS) for c in LABELS}


def encoder_expected_prior(segments, pids):
    mass = {c: 0.0 for c in LABELS}
    for p in pids:
        for start, end, probs in segments.get(p, []):
            length = end - start
            for c in LABELS:
                mass[c] += length * probs[c]
    z = sum(mass.values())
    return {c: mass[c] / z for c in mass} if z > 0 else {
        c: 1.0 / len(LABELS) for c in LABELS}


def calibrate(dist, weights):
    adj = {c: dist[c] * weights[c] for c in LABELS}
    z = sum(adj.values())
    return {c: v / z for c, v in adj.items()} if z > EPS else dist


def fit_sequence_model(pids, gold, genre_of):
    """Genre-conditioned start/transition log-probs with Laplace alpha."""
    start_counts = defaultdict(lambda: defaultdict(float))
    trans_counts = defaultdict(lambda: defaultdict(float))
    genres = set()
    for p in pids:
        seq = [s.label for s in sorted(gold.get(p, []), key=lambda s: s.start)]
        if not seq:
            continue
        for g in (genre_of[p], "__global__"):
            genres.add(g)
            start_counts[g][seq[0]] += 1
            for a, b in zip(seq, seq[1:]):
                trans_counts[g][(a, b)] += 1
    model = {}
    for g in genres:
        start_z = sum(start_counts[g].values()) + ALPHA * len(LABELS)
        start_lp = {c: math.log((start_counts[g][c] + ALPHA) / start_z)
                    for c in LABELS}
        trans_lp = {}
        for a in LABELS:
            row_z = sum(trans_counts[g][(a, b)] for b in LABELS) + ALPHA * len(LABELS)
            trans_lp[a] = {b: math.log((trans_counts[g][(a, b)] + ALPHA) / row_z)
                           for b in LABELS}
        model[g] = (start_lp, trans_lp)
    return model


def viterbi(emissions, start_lp, trans_lp):
    cur = {c: start_lp[c] + math.log(max(emissions[0][c], EPS)) for c in LABELS}
    back = []
    for t in range(1, len(emissions)):
        nxt, bp = {}, {}
        for c in LABELS:
            best_b = max(LABELS, key=lambda b: cur[b] + trans_lp[b][c])
            nxt[c] = (cur[best_b] + trans_lp[best_b][c]
                      + math.log(max(emissions[t][c], EPS)))
            bp[c] = best_b
        back.append(bp)
        cur = nxt
    last = max(cur, key=cur.get)
    seq = [last]
    for bp in reversed(back):
        seq.append(bp[seq[-1]])
    return list(reversed(seq))


def decode_paragraph(spans, enc_dists, lam, seq_model, genre):
    if not spans:
        return []
    emissions = []
    for s, enc in zip(spans, enc_dists):
        onehot = {c: 1.0 if c == s.label else 0.0 for c in LABELS}
        emissions.append({c: (1 - lam) * enc[c] + lam * onehot[c] for c in LABELS})
    start_lp, trans_lp = seq_model.get(genre, seq_model["__global__"])
    labels = viterbi(emissions, start_lp, trans_lp)
    return [Span(s.start, s.end, lab) for s, lab in zip(spans, labels)]


def dominant_label(spans, span):
    mass = defaultdict(int)
    for q in spans:
        ov = max(0, min(q.end, span.end) - max(q.start, span.start))
        if ov > 0:
            mass[q.label] += ov
    return max(mass, key=mass.get) if mass else None


def dedup_spans(spans):
    """Drop exact (start, end, label) duplicates, keeping first occurrence.

    v3 emits same-offset spans with different labels; re-decode can
    collapse such a pair to one label, which the submission validator
    refuses as a duplicate. No-op on the S1 gate run (0 exact dupes).
    """
    seen, out = set(), []
    for s in spans:
        key = (s.start, s.end, s.label)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def rule_a(spans, quote_spans):
    out, changed = [], 0
    for s in spans:
        if s.label == "CO":
            dom = dominant_label(quote_spans, s)
            if dom is not None and dom != "CO":
                out.append(Span(s.start, s.end, dom))
                changed += 1
                continue
        out.append(s)
    return out, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder-run", default=ENCODER_RUN)
    args = ap.parse_args()

    pred, text_of, genre_of, fold_of = load_role_runs()
    gold = {int(p): s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
            if int(p) in pred}
    segments = load_segments(
        EXPERIMENTS_DIR / args.encoder_run / "predictions" / "oof_task_2_segment_scores.jsonl")
    quote = {int(r["paragraph_id"]): [Span(s["start_offset"], s["end_offset"], s["label"])
                                      for s in r["labels"]]
             for r in read_jsonl(EXPERIMENTS_DIR / QUOTE_RUN / "predictions" / "preds.jsonl")}

    missing_segments = sorted(p for p in pred if p not in segments)
    if missing_segments:
        raise ValueError(f"{len(missing_segments)} paragraphs lack encoder segments, "
                         f"e.g. {missing_segments[:5]}")

    raw_enc = {p: [span_encoder_dist(segments[p], s) for s in pred[p]] for p in pred}

    fitted = {}
    relabeled, decoded_only, pre_dedup = {}, {}, {}
    rule_a_changes = 0
    for f in range(5):
        fit_pids = [p for p in pred if fold_of[p] != f]
        eval_pids = [p for p in pred if fold_of[p] == f]

        weights_gold = mass_prior([gold.get(p, []) for p in fit_pids])
        weights_enc = encoder_expected_prior(segments, fit_pids)
        weights = {c: weights_gold[c] / max(weights_enc[c], EPS) for c in LABELS}
        seq_model = fit_sequence_model(fit_pids, gold, genre_of)
        cal = {p: [calibrate(d, weights) for d in raw_enc[p]] for p in pred}

        best_lam, best_f1 = None, -1.0
        for lam in LAMBDA_GRID:
            fit_out = {p: decode_paragraph(pred[p], cal[p], lam, seq_model,
                                           genre_of[p]) for p in fit_pids}
            f1 = span_partial_f1({p: gold[p] for p in fit_pids}, fit_out)["f1"]
            if f1 > best_f1:
                best_lam, best_f1 = lam, f1

        for p in eval_pids:
            dec = decode_paragraph(pred[p], cal[p], best_lam, seq_model, genre_of[p])
            decoded_only[p] = dec
            final, changed = rule_a(dec, quote.get(p, []))
            pre_dedup[p] = final
            relabeled[p] = dedup_spans(final)
            rule_a_changes += changed
        fitted[str(f)] = {
            "lambda": best_lam,
            "lambda_fit_f1": round(best_f1, 4),
            "calibration_weights": {c: round(weights[c], 4) for c in LABELS},
        }

    g_all = {p: gold[p] for p in pred}
    rec_f1 = span_partial_f1(g_all, pred)["f1"]
    dec_f1 = span_partial_f1(g_all, decoded_only)["f1"]
    new_f1 = span_partial_f1(g_all, relabeled)["f1"]

    per_fold, wins = {}, 0
    for f in range(5):
        sub = [p for p in pred if fold_of[p] == f]
        g = {p: gold[p] for p in sub}
        r0 = span_partial_f1(g, {p: pred[p] for p in sub})["f1"]
        r1 = span_partial_f1(g, {p: relabeled[p] for p in sub})["f1"]
        wins += r1 >= r0
        per_fold[str(f)] = {"recorded": round(r0, 4), "bundle": round(r1, 4)}

    def mass_ratios(pred_map):
        pm = mass_prior(pred_map.values())
        gm = mass_prior(g_all.values())
        return {c: round(pm[c] / max(gm[c], EPS), 2) for c in LABELS}

    n_changed = sum(1 for p in pred for a, b in zip(pred[p], pre_dedup[p])
                    if a.label != b.label)
    n_dedup_dropped = sum(len(pre_dedup[p]) - len(relabeled[p]) for p in pred)
    gate = round(rec_f1 + GATE_MARGIN, 4)
    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md registered T2 S1 structural bundle",
        "encoder_run": args.encoder_run,
        "quote_run": QUOTE_RUN,
        "recorded_f1": round(rec_f1, 4),
        "decoded_f1_pre_rule_a": round(dec_f1, 4),
        "bundle_f1": round(new_f1, 4),
        "delta": round(new_f1 - rec_f1, 4),
        "n_spans": sum(len(s) for s in pred.values()),
        "n_label_changes": n_changed,
        "n_rule_a_changes": rule_a_changes,
        "n_dedup_dropped": n_dedup_dropped,
        "mass_ratio_vs_gold_recorded": mass_ratios(pred),
        "mass_ratio_vs_gold_bundle": mass_ratios(relabeled),
        "fitted": fitted,
        "gate": {
            "rule": f"pooled >= {gate} AND >=3/5 fold wins",
            "per_fold": per_fold,
            "fold_wins": wins,
            "adopted": bool(new_f1 >= rec_f1 + GATE_MARGIN and wins >= 3),
        },
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t2-s1-structural-camelbert-quarter-n{len(pred)}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "relabeled_task_2.jsonl", [
        {"paragraph_id": p, "text": text_of[p], "type": genre_of[p],
         "labels": [{"label": s.label, "start_offset": s.start,
                     "end_offset": s.end} for s in relabeled[p]]}
        for p in sorted(relabeled)
    ])
    write_jsonl(pred_dir / "label_changes.jsonl", [
        {"paragraph_id": p, "span_index": i, "start_offset": a.start,
         "end_offset": a.end, "recorded": a.label, "decoded": d.label,
         "final": b.label}
        for p in sorted(pred)
        for i, (a, d, b) in enumerate(zip(pred[p], decoded_only[p], pre_dedup[p]))
        if a.label != b.label or a.label != d.label
    ])
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "t2_structural_gate.py", "argv": sys.argv[1:],
         "role_runs": ROLE_RUNS, "encoder_run": args.encoder_run,
         "quote_run": QUOTE_RUN, "labels": LABELS,
         "lambda_grid": LAMBDA_GRID, "alpha": ALPHA,
         "registration": "CREATIVE_HEADROOM_RESEARCH.md S1 (commit 46368f9)"},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
