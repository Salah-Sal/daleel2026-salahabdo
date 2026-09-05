"""Paired paragraph bootstrap for Task 2 prediction comparisons.

This is a robustness diagnostic, not a correction for model-selection bias.
Every bootstrap replicate samples paragraph IDs with replacement and scores
the baseline and candidate on the identical synthetic paragraph collection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.metrics import Span, span_partial_f1

from t2_eval_week_filter import load_span_records, score_slices


def bootstrap_delta(
    pids: Sequence[int],
    gold: Mapping[int, Sequence[Span]],
    baseline: Mapping[int, Sequence[Span]],
    candidate: Mapping[int, Sequence[Span]],
    *,
    replicates: int,
    seed: int,
) -> dict:
    def components(pred: Mapping[int, Sequence[Span]]) -> np.ndarray:
        rows = []
        for pid in pids:
            one_gold = {pid: gold[pid]}
            one_pred = {pid: pred.get(pid, ())}
            score = span_partial_f1(one_gold, one_pred)
            pred_count = len(one_pred[pid])
            gold_count = len(one_gold[pid])
            rows.append(
                (
                    score["precision"] * pred_count,
                    pred_count,
                    score["recall"] * gold_count,
                    gold_count,
                )
            )
        return np.asarray(rows, dtype=float)

    def f1(rows: np.ndarray) -> np.ndarray:
        precision = np.divide(
            rows[:, 0],
            rows[:, 1],
            out=np.zeros(len(rows), dtype=float),
            where=rows[:, 1] != 0,
        )
        recall = np.divide(
            rows[:, 2],
            rows[:, 3],
            out=np.zeros(len(rows), dtype=float),
            where=rows[:, 3] != 0,
        )
        return np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(len(rows), dtype=float),
            where=(precision + recall) != 0,
        )

    rng = np.random.default_rng(seed)
    # A multinomial row is the paragraph multiplicity vector for one
    # with-replacement sample. Multiplying it by the additive official-score
    # components is exactly equivalent to materializing synthetic duplicate
    # paragraph IDs, but much faster.
    multiplicities = rng.multinomial(
        len(pids), np.full(len(pids), 1.0 / len(pids)), size=replicates
    )
    deltas = (
        f1(multiplicities @ components(candidate))
        - f1(multiplicities @ components(baseline))
        )
    lower, median, upper = np.quantile(deltas, [0.025, 0.5, 0.975])
    return {
        "n_paragraphs": len(pids),
        "replicates": replicates,
        "seed": seed,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(median),
        "percentile_95_interval": [float(lower), float(upper)],
        "probability_delta_gt_zero": float(np.mean(deltas > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")

    gold, records = load_span_records(args.gold)
    baseline, _ = load_span_records(args.baseline)
    pids = sorted(gold)
    if set(baseline) != set(pids):
        raise SystemExit("baseline paragraph IDs do not exactly match gold")

    report = {
        "method": "paired paragraph bootstrap with replacement",
        "caveat": "does not correct selection bias when candidates were chosen on this gold set",
        "gold": str(args.gold),
        "baseline": str(args.baseline),
        "baseline_scores": score_slices(gold, baseline, records),
        "candidates": [],
    }
    for candidate_path in args.candidate:
        candidate, _ = load_span_records(candidate_path)
        if set(candidate) != set(pids):
            raise SystemExit(
                f"candidate paragraph IDs do not exactly match gold: {candidate_path}"
            )
        candidate_report = {
            "path": str(candidate_path),
            "scores": score_slices(gold, candidate, records),
            "bootstrap": {},
        }
        for genre in ("overall", "editorial", "debate"):
            genre_pids = (
                pids
                if genre == "overall"
                else [pid for pid in pids if records[pid]["type"] == genre]
            )
            candidate_report["bootstrap"][genre] = bootstrap_delta(
                genre_pids,
                gold,
                baseline,
                candidate,
                replicates=args.replicates,
                seed=args.seed,
            )
        report["candidates"].append(candidate_report)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
