"""Paper analysis bank: A1 threshold audit, A2 vote-scaling curves, A3 span
confusion + coarse-family rebound (literature-review analyses, zero-risk).

Descriptive analyses over cached OOF artifacts of the pinned systems (T1
v2.2 inputs, T2 S1 relabeled spans). No gate, no adoption decision, no
system change; outputs are numeric tables only (no dataset text).

  A1  Lipton et al. 2014: compare deployed per-label operating points to
      the pooled-OOF F1-optimal threshold and the plug-in F*/2 rule on the
      encoder sigmoids; flat-interval width = dev->eval transfer-risk proxy.
      Vote legs audited on the integer vote-count grid.
  A2  Chen et al. 2024 vote-scaling: per-label (alpha, p1, p2) from the 5
      gemma rollouts, expected-F1 curves in K under (a) per-label optimal
      integer threshold (our system's rule) and (b) fixed majority rule,
      plus the K->inf limit and the "inverted-item" mass no vote count can
      fix (gold positives the sampler fires on <50%, negatives >50%).
  A3  T2 char-mass confusion matrices (v3 recorded vs S1 relabeled),
      within-family share of misassigned mass (evidence TE/ST/AN vs
      subjective CO/AS vs OT), coarse-family partial-F1 rebound, and the
      juxtaposition cells against Al-Khatib et al. 2016 Table 3 (human CPM).

Example (from shared-task/):
  uv run scripts/paper_analysis_bank.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.artifacts import atomic_write_json, create_experiment_dir
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import read_jsonl
from daleel.metrics import Span, span_partial_f1
from daleel.runtime import EXPERIMENTS_DIR
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids

from route_task1 import CO_PRIOR, load_preds  # noqa: E402 — sibling script import
from route_task1_v2 import CO_BUDGET_MULT  # noqa: E402 — sibling script import

BASELINE_RUN = EXPERIMENTS_DIR / "20260715-003004-912987Z-t1-routed-v2-k5-n430-4f30d408fa"
S1_RUN = EXPERIMENTS_DIR / "20260715-001221-t2-s1-structural-camelbert-quarter-n430"
DEPLOYED_VOTE_THETA = {"AS": 2, "TE": 4, "ST": 4}  # deploy medians (v2 deploy rule)
N_ROLLOUTS = 5

FAMILY = {"TE": "EVID", "ST": "EVID", "AN": "EVID", "CO": "SUBJ", "AS": "SUBJ", "OT": "OTH"}
FAMILIES = ("EVID", "SUBJ", "OTH")
# Al-Khatib et al. 2016 Table 3 (Confusion Probability Matrix), verified
# digit-for-digit in the 2026-07-15 reading pass — juxtaposition anchors.
HUMAN_CPM_CELLS = {"CO->AS": 0.562, "AN->AS": 0.277, "CO->CO": 0.129}


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def best_threshold_curve(scores: list[float], gold: list[bool]) -> dict:
    """Sweep all midpoint thresholds on a continuous score; return the
    F1-optimal point, the Lipton plug-in F*/2 point, and the 95%-of-max
    threshold interval (curve flatness)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n_pos = sum(gold)
    # thresholds between consecutive distinct scores, descending
    tp = fp = 0
    points = []  # (threshold, f1, fires)
    sorted_scores = [scores[i] for i in order]
    for rank, i in enumerate(order):
        tp += gold[i]
        fp += not gold[i]
        nxt = sorted_scores[rank + 1] if rank + 1 < len(order) else sorted_scores[rank] - 1.0
        if nxt == sorted_scores[rank]:
            continue  # tie: threshold cannot fall between equal scores
        thr = (sorted_scores[rank] + nxt) / 2
        points.append((thr, f1_from_counts(tp, fp, n_pos - tp), tp + fp))
    f_star, thr_star, fires_star = 0.0, 1.0, 0
    for thr, f1, fires in points:
        if f1 > f_star:
            f_star, thr_star, fires_star = f1, thr, fires

    def eval_at(threshold: float) -> dict:
        tp = sum(1 for s, g in zip(scores, gold) if s >= threshold and g)
        fp = sum(1 for s, g in zip(scores, gold) if s >= threshold and not g)
        return {"f1": round(f1_from_counts(tp, fp, n_pos - tp), 4), "fires": tp + fp}

    lipton = f_star / 2
    flat = [thr for thr, f1, _ in points if f1 >= 0.95 * f_star]
    return {
        "f_star": round(f_star, 4),
        "theta_star": round(thr_star, 4),
        "fires_at_star": fires_star,
        "lipton_theta": round(lipton, 4),
        "at_lipton": eval_at(lipton),
        "flat95_interval": [round(min(flat), 4), round(max(flat), 4)] if flat else None,
        "n_pos": n_pos,
    }


def a1_threshold_audit(pids, gold, enc_scores, enc_metrics, votes, fold_of) -> dict:
    out = {"per_label_encoder": {}, "per_label_votes": {}}
    deployed = enc_metrics["diagnostics"]["final_thresholds_fit_on_all_oof_scores"]
    for label in LABELS:
        scores = [enc_scores[p][label] for p in pids]
        gvec = [label in gold[p] for p in pids]
        curve = best_threshold_curve(scores, gvec)
        thr_dep = deployed[label]
        tp = sum(1 for s, g in zip(scores, gvec) if s >= thr_dep and g)
        fp = sum(1 for s, g in zip(scores, gvec) if s >= thr_dep and not g)
        curve["deployed_theta"] = round(thr_dep, 4)
        curve["at_deployed"] = {
            "f1": round(f1_from_counts(tp, fp, curve["n_pos"] - tp), 4),
            "fires": tp + fp,
        }
        out["per_label_encoder"][label] = curve
    for label, theta_dep in DEPLOYED_VOTE_THETA.items():
        gvec = {p: (label in gold[p]) for p in pids}
        by_theta = {}
        for theta in range(1, N_ROLLOUTS + 1):
            tp = sum(1 for p in pids if votes[p][label] >= theta and gvec[p])
            fp = sum(1 for p in pids if votes[p][label] >= theta and not gvec[p])
            fn = sum(gvec.values()) - tp
            by_theta[theta] = round(f1_from_counts(tp, fp, fn), 4)
        best_theta = max(by_theta, key=by_theta.get)
        out["per_label_votes"][label] = {
            "f1_by_theta": by_theta,
            "theta_star": best_theta,
            "deployed_theta": theta_dep,
            "deployed_is_optimal": best_theta == theta_dep,
        }
    # CO budget leg: implied per-fold rank cutoff vs the sigmoid's own optimum
    folds = sorted(set(fold_of.values()))
    budget_fires = sum(
        round(CO_BUDGET_MULT * CO_PRIOR * sum(1 for p in pids if fold_of[p] == f))
        for f in folds
    )
    out["co_budget"] = {
        "pooled_budget_fires": budget_fires,
        "encoder_sigmoid_fires_at_star": out["per_label_encoder"]["CO"]["fires_at_star"],
        "note": "budget rule fires a fixed count per fold; the rank fusion adds vote/span features on top of the sigmoid audited above",
    }
    return out


def binom_tail(k: int, q: float, t: int) -> float:
    """P(Binomial(k, q) >= t)."""
    if t <= 0:
        return 1.0
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    return sum(math.comb(k, c) * q**c * (1 - q) ** (k - c) for c in range(t, k + 1))


def a2_vote_scaling(pids, gold, votes) -> dict:
    ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    out = {}
    for label in LABELS:
        q = {p: votes[p][label] / N_ROLLOUTS for p in pids}
        pos = [p for p in pids if label in gold[p]]
        neg = [p for p in pids if label not in gold[p]]
        p1 = sum(q[p] for p in pos) / len(pos) if pos else 0.0
        p2 = sum(q[p] for p in neg) / len(neg) if neg else 0.0
        inverted_pos = sum(1 for p in pos if q[p] < 0.5)
        inverted_neg = sum(1 for p in neg if q[p] > 0.5)

        def curve(k: int, theta: int) -> float:
            etp = sum(binom_tail(k, q[p], theta) for p in pos)
            efp = sum(binom_tail(k, q[p], theta) for p in neg)
            return f1_from_counts(etp, efp, len(pos) - etp)

        optimal, majority = {}, {}
        for k in ks:
            optimal[k] = round(max(curve(k, t) for t in range(1, k + 1)), 4)
            majority[k] = round(curve(k, k // 2 + 1), 4)
        # K -> inf: each item fires iff q > tau; best fraction cutoff on the grid
        taus = sorted({q[p] for p in pids} | {0.0})
        inf_best = 0.0
        for tau in taus:
            tp = sum(1 for p in pos if q[p] > tau)
            fp = sum(1 for p in neg if q[p] > tau)
            inf_best = max(inf_best, f1_from_counts(tp, fp, len(pos) - tp))
        peak = max(optimal.values())
        k_star = min(k for k in ks if optimal[k] >= 0.995 * peak)
        out[label] = {
            "alpha_prevalence": round(len(pos) / len(pids), 4),
            "p1_fire_rate_on_gold": round(p1, 4),
            "p2_fire_rate_on_negatives": round(p2, 4),
            "inverted_gold_frac": round(inverted_pos / len(pos), 4) if pos else None,
            "inverted_neg_count": inverted_neg,
            "f1_by_k_optimal_theta": optimal,
            "f1_by_k_majority_rule": majority,
            "f1_at_k_inf": round(inf_best, 4),
            "k_star_995pct_of_peak": k_star,
        }
    return out


def load_span_file(path: Path) -> dict[int, list[Span]]:
    out: dict[int, list[Span]] = {}
    for r in read_jsonl(path):
        out[int(r["paragraph_id"])] = [
            Span(s["start_offset"], s["end_offset"], s["label"]) for s in r["labels"]
        ]
    return out


def mass_confusion(gold: dict, pred: dict) -> dict:
    """Char-mass matrix: rows = gold label, columns = predicted label or
    UNCOVERED (gold chars no predicted span of any label overlaps)."""
    matrix = {g: dict.fromkeys(list(LABELS) + ["UNCOVERED"], 0) for g in LABELS}
    for pid, gspans in gold.items():
        pspans = pred.get(pid, [])
        for gs in gspans:
            covered_by = dict.fromkeys(LABELS, 0)
            cover = [0] * len(gs)
            for ps in pspans:
                lo, hi = max(gs.start, ps.start), min(gs.end, ps.end)
                if hi <= lo:
                    continue
                covered_by[ps.label] += hi - lo
                for c in range(lo - gs.start, hi - gs.start):
                    cover[c] = 1
            for lab, mass in covered_by.items():
                matrix[gs.label][lab] += mass
            matrix[gs.label]["UNCOVERED"] += len(gs) - sum(cover)
    return matrix


def a3_confusion_families(gold, recorded, relabeled) -> dict:
    def normalized(matrix):
        out = {}
        for g, row in matrix.items():
            total = sum(row.values())
            out[g] = {k: round(v / total, 4) if total else 0.0 for k, v in row.items()}
        return out

    def within_family_share(matrix):
        wrong = within = 0
        for g, row in matrix.items():
            for p, mass in row.items():
                if p in ("UNCOVERED", g):
                    continue
                wrong += mass
                within += mass * (FAMILY[g] == FAMILY[p])
        return round(within / wrong, 4) if wrong else None

    def family_f1(pred):
        fam = lambda spans: [Span(s.start, s.end, FAMILY[s.label]) for s in spans]  # noqa: E731
        g = {p: fam(s) for p, s in gold.items()}
        q = {p: fam(s) for p, s in pred.items()}
        return round(span_partial_f1(g, q, labels=FAMILIES)["f1"], 4)

    m_rec, m_rel = mass_confusion(gold, recorded), mass_confusion(gold, relabeled)
    n_rel = normalized(m_rel)
    juxtaposition = {
        "model_CO->AS": n_rel["CO"]["AS"],
        "model_AN->AS": n_rel["AN"]["AS"],
        "model_CO->CO": n_rel["CO"]["CO"],
        "human_cpm": HUMAN_CPM_CELLS,
    }
    return {
        "mass_confusion_recorded_v3": normalized(m_rec),
        "mass_confusion_s1_relabeled": n_rel,
        "within_family_share_of_wrong_mass": {
            "recorded_v3": within_family_share(m_rec),
            "s1_relabeled": within_family_share(m_rel),
        },
        "six_label_pooled_f1": {"recorded_v3": 0.6934, "s1_relabeled": 0.7206},
        "family_level_pooled_f1": {
            "recorded_v3": family_f1(recorded),
            "s1_relabeled": family_f1(relabeled),
        },
        "human_juxtaposition": juxtaposition,
        "families": FAMILY,
    }


def main() -> None:
    base_cfg = json.loads((BASELINE_RUN / "config.json").read_text())
    inputs = base_cfg["inputs"]

    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    gold1 = {pid: labels for pid, labels in clean_task1_gold(t1, t2).items()
             if pid in set(train_ids)}
    pids = sorted(gold1)

    rollout_preds = [load_preds(EXPERIMENTS_DIR / Path(d).name)
                     for d in inputs["rollouts"]]
    votes = {p: {label: sum(label in r[p] for r in rollout_preds)
                 for label in LABELS} for p in pids}

    enc_run = EXPERIMENTS_DIR / Path(inputs["encoder"]).name
    enc_metrics = json.loads((enc_run / "metrics.json").read_text())
    enc_scores, fold_of = {}, {}
    for r in read_jsonl(enc_run / "predictions" / "oof_task_1_scores.jsonl"):
        enc_scores[r["paragraph_id"]] = r["scores"]
        fold_of[r["paragraph_id"]] = r["fold"]

    gold2 = {int(p): s for p, s in clean_task2_gold(load_records(TRAIN_TASK2)).items()
             if int(p) in set(train_ids)}
    relabeled = load_span_file(S1_RUN / "predictions" / "relabeled_task_2.jsonl")
    s1_metrics = json.loads((S1_RUN / "metrics.json").read_text())
    recorded: dict[int, list[Span]] = {}
    for d in json.loads((S1_RUN / "config.json").read_text())["role_runs"]:
        recorded.update(load_span_file(
            EXPERIMENTS_DIR / Path(d).name / "predictions" / "outer_task_2.jsonl"))
    gold2 = {p: s for p, s in gold2.items() if p in relabeled}

    metrics = {
        "milestone": "literature-review zero-risk paper analyses",
        "a1_lipton_threshold_audit": a1_threshold_audit(
            pids, gold1, enc_scores, enc_metrics, votes, fold_of),
        "a2_vote_scaling": a2_vote_scaling(pids, gold1, votes),
        "a3_confusion_families": a3_confusion_families(gold2, recorded, relabeled),
        "inputs_note": "T1 = v2.2 gate-run inputs (rollouts/encoder); "
                       "T2 = S1 gate-run recorded+relabeled spans",
        "s1_recorded_f1_check": s1_metrics["recorded_f1"],
    }

    resolved_config = {
        "script": "paper_analysis_bank.py", "argv": sys.argv[1:],
        "baseline_run": str(BASELINE_RUN), "s1_run": str(S1_RUN),
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR, f"paper-analysis-bank-n{len(pids)}", resolved_config)
    atomic_write_json(exp_dir / "config.json", resolved_config)
    atomic_write_json(exp_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nrun dir: {exp_dir}")


if __name__ == "__main__":
    main()
