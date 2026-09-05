"""T2 "G1" calibration-temperature gate (registered, CREATIVE_HEADROOM_RESEARCH.md 0f6b709).

The adopted S1 bundle with Menon-style tunable calibration strength:
adj_c = enc_c * w_c^tau (renormalized), (lambda, tau) selected jointly
per fold on the 4 fitting folds; everything else byte-identical to S1.
CONTROL: (lambda=0.9, tau=1.0) must reproduce pooled 0.7206 exactly.
GATE: selected pooled OOF >= 0.7406 AND >= 3/5 fold wins vs S1 per fold.

Example (from shared-task/):
  uv run scripts/t2_g1_tau_gate.py
"""

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
    EPS,
    LABELS,
    LAMBDA_GRID,
    QUOTE_RUN,
    decode_paragraph,
    dedup_spans,
    encoder_expected_prior,
    fit_sequence_model,
    load_role_runs,
    load_segments,
    mass_prior,
    rule_a,
    span_encoder_dist,
)

S1_RUN = "20260715-001221-t2-s1-structural-camelbert-quarter-n430"
S1_POOLED = 0.7206
TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
GATE_MARGIN = 0.02


def calibrate_tau(dist, weights, tau):
    adj = {c: dist[c] * (weights[c] ** tau) for c in LABELS}
    z = sum(adj.values())
    return {c: v / z for c, v in adj.items()} if z > EPS else dist


def selection_key(lam, tau):
    """Deterministic tie-break: prefer the S1 point, then lower tau, lower lambda."""
    return (0 if (lam == 0.9 and tau == 1.0) else 1, tau, lam)


def main() -> None:
    pred, text_of, genre_of, fold_of = load_role_runs()
    gold = {int(p): s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
            if int(p) in pred}
    segments = load_segments(
        EXPERIMENTS_DIR / ENCODER_RUN / "predictions" / "oof_task_2_segment_scores.jsonl")
    quote = {int(r["paragraph_id"]): [Span(s["start_offset"], s["end_offset"], s["label"])
                                      for s in r["labels"]]
             for r in read_jsonl(EXPERIMENTS_DIR / QUOTE_RUN / "predictions" / "preds.jsonl")}
    s1 = {int(r["paragraph_id"]): [Span(s["start_offset"], s["end_offset"], s["label"])
                                   for s in r["labels"]]
          for r in read_jsonl(EXPERIMENTS_DIR / S1_RUN / "predictions" / "relabeled_task_2.jsonl")}

    raw_enc = {p: [span_encoder_dist(segments[p], s) for s in pred[p]] for p in pred}
    g_all = {p: gold[p] for p in pred}

    s1_pooled_check = span_partial_f1(g_all, s1)["f1"]
    if round(s1_pooled_check, 4) != S1_POOLED:
        raise SystemExit(f"S1 baseline file scores {s1_pooled_check:.4f} != {S1_POOLED}")

    def run_variant(fold_sel):
        """fold_sel: f -> (lambda, tau). Returns relabeled map."""
        out = {}
        for f in range(5):
            fit_pids = [p for p in pred if fold_of[p] != f]
            eval_pids = [p for p in pred if fold_of[p] == f]
            lam, tau = fold_sel[f]
            weights_gold = mass_prior([gold.get(p, []) for p in fit_pids])
            weights_enc = encoder_expected_prior(segments, fit_pids)
            w = {c: weights_gold[c] / max(weights_enc[c], EPS) for c in LABELS}
            seq_model = fit_sequence_model(fit_pids, gold, genre_of)
            for p in eval_pids:
                cal = [calibrate_tau(d, w, tau) for d in raw_enc[p]]
                dec = decode_paragraph(pred[p], cal, lam, seq_model, genre_of[p])
                final, _ = rule_a(dec, quote.get(p, []))
                out[p] = dedup_spans(final)
        return out

    # ---- CONTROL: fixed S1 operating point must reproduce 0.7206
    control = run_variant({f: (0.9, 1.0) for f in range(5)})
    control_f1 = round(span_partial_f1(g_all, control)["f1"], 4)
    if control_f1 != S1_POOLED:
        raise SystemExit(f"CONTROL FAILED: (0.9, 1.0) pipeline = {control_f1} != {S1_POOLED}")

    # ---- joint per-fold (lambda, tau) selection on fitting folds
    fold_sel, sel_diag = {}, {}
    for f in range(5):
        fit_pids = [p for p in pred if fold_of[p] != f]
        weights_gold = mass_prior([gold.get(p, []) for p in fit_pids])
        weights_enc = encoder_expected_prior(segments, fit_pids)
        w = {c: weights_gold[c] / max(weights_enc[c], EPS) for c in LABELS}
        seq_model = fit_sequence_model(fit_pids, gold, genre_of)
        g_fit = {p: gold[p] for p in fit_pids}

        best, best_f1 = None, -1.0
        for tau in TAU_GRID:
            cal = {p: [calibrate_tau(d, w, tau) for d in raw_enc[p]] for p in fit_pids}
            for lam in LAMBDA_GRID:
                fit_out = {p: decode_paragraph(pred[p], cal[p], lam, seq_model,
                                               genre_of[p]) for p in fit_pids}
                f1 = span_partial_f1(g_fit, fit_out)["f1"]
                if f1 > best_f1 or (f1 == best_f1 and
                                    selection_key(lam, tau) < selection_key(*best)):
                    best, best_f1 = (lam, tau), f1
        fold_sel[f] = best
        sel_diag[str(f)] = {"lambda": best[0], "tau": best[1],
                            "fit_f1": round(best_f1, 4)}

    relabeled = run_variant(fold_sel)
    new_f1 = span_partial_f1(g_all, relabeled)["f1"]

    per_fold, wins = {}, 0
    for f in range(5):
        sub = [p for p in pred if fold_of[p] == f]
        g = {p: gold[p] for p in sub}
        r0 = span_partial_f1(g, {p: s1[p] for p in sub})["f1"]
        r1 = span_partial_f1(g, {p: relabeled[p] for p in sub})["f1"]
        wins += r1 >= r0
        per_fold[str(f)] = {"s1": round(r0, 4), "g1": round(r1, 4)}

    # ---- diagnostics for the paper: pooled OOF per tau at each fold's chosen lambda
    tau_curve = {}
    for tau in TAU_GRID:
        variant = run_variant({f: (fold_sel[f][0], tau) for f in range(5)})
        pm = mass_prior(variant.values())
        gm = mass_prior(g_all.values())
        tau_curve[str(tau)] = {
            "pooled_f1": round(span_partial_f1(g_all, variant)["f1"], 4),
            "an_mass_ratio": round(pm["AN"] / max(gm["AN"], EPS), 2),
            "co_mass_ratio": round(pm["CO"] / max(gm["CO"], EPS), 2),
        }

    gate = round(S1_POOLED + GATE_MARGIN, 4)
    metrics = {
        "milestone": "CREATIVE_HEADROOM_RESEARCH.md registered T2 G1 tau gate (0f6b709)",
        "s1_run": S1_RUN,
        "control_s1_point": control_f1,
        "g1_f1": round(new_f1, 4),
        "delta_vs_s1": round(new_f1 - S1_POOLED, 4),
        "selection": sel_diag,
        "tau_curve_at_selected_lambda": tau_curve,
        "gate": {
            "rule": f"pooled >= {gate} AND >=3/5 fold wins",
            "per_fold": per_fold,
            "fold_wins": wins,
            "adopted": bool(new_f1 >= S1_POOLED + GATE_MARGIN and wins >= 3),
        },
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"{stamp}-t2-g1-tau-n{len(pred)}"
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pred_dir / "relabeled_task_2.jsonl", [
        {"paragraph_id": p, "text": text_of[p], "type": genre_of[p],
         "labels": [{"label": s.label, "start_offset": s.start,
                     "end_offset": s.end} for s in relabeled[p]]}
        for p in sorted(relabeled)
    ])
    (exp_dir / "config.json").write_text(json.dumps(
        {"script": "t2_g1_tau_gate.py", "argv": sys.argv[1:],
         "s1_run": S1_RUN, "encoder_run": ENCODER_RUN, "quote_run": QUOTE_RUN,
         "tau_grid": TAU_GRID, "lambda_grid": LAMBDA_GRID,
         "registration": "CREATIVE_HEADROOM_RESEARCH.md G1 (commit 0f6b709)"},
        indent=2, default=str))
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
