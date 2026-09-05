"""Add conservative Task 2 spans under a Task 1 presence constraint.

The eval-week precision filters only remove spans.  This companion recovers
recall without inventing boundaries: for a Task 1 label that is absent from a
paragraph's Task 2 output, it selects the highest-posterior deterministic
encoder segment for that label.  At most one span is added per label and
paragraph.  Exact duplicates are forbidden; same-boundary different-label
spans remain possible because the official data contains dual-role spans.

``--mode missing`` implements the primary rescue described above.
``--mode additional`` instead considers one non-overlapping extra occurrence
when the base already contains the label; this diagnoses within-paragraph
recall without creating overlapping same-label predictions.

With ``--gold --audit-grid``, every label/genre/threshold cell is evaluated
independently against the unchanged base.  ``--coordinate-search`` then adds
only strictly improving cells, one at a time, and reports the selected rules.
The search is a development-set diagnostic; it must not be described as OOF.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.constants import LABELS
from daleel.metrics import Span, span_partial_f1

from t2_eval_week_filter import (
    GENRES,
    changed_span_count,
    compact_scores,
    load_span_records,
    load_t1_labels,
    parse_posterior_rules,
    posterior_threshold,
    score_slices,
    write_predictions,
)

THRESHOLD_GRID = tuple(round(step * 0.05, 2) for step in range(20))


def load_segment_candidates(
    path: Path,
) -> dict[int, list[tuple[int, int, dict[str, float]]]]:
    candidates: defaultdict[int, list[tuple[int, int, dict[str, float]]]] = defaultdict(list)
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            pid = int(row["paragraph_id"])
            start = int(row["start_offset"])
            end = int(row["end_offset"])
            key = (pid, start, end)
            if key in seen:
                raise ValueError(f"duplicate encoder segment {key} in {path}")
            seen.add(key)
            scores = {label: float(value) for label, value in row["scores"].items()}
            missing = set(LABELS) - set(scores)
            if missing:
                raise ValueError(f"encoder segment {key} lacks scores for {sorted(missing)}")
            candidates[pid].append((start, end, scores))
    return dict(candidates)


def rescue_missing_roles(
    base: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    task1: Mapping[int, set[str]],
    candidates: Mapping[int, Sequence[tuple[int, int, Mapping[str, float]]]],
    rules: Mapping[tuple[str, str | None], float],
    mode: str = "missing",
) -> tuple[dict[int, list[Span]], dict]:
    missing_t1 = sorted(set(base) - set(task1))
    if missing_t1:
        raise ValueError(f"{len(missing_t1)} base paragraphs lack Task 1 labels")

    output: dict[int, list[Span]] = {}
    additions: Counter[tuple[str, str]] = Counter()
    for pid, spans in base.items():
        genre = str(records[pid]["type"])
        rescued = list(spans)
        exact = set(spans)
        present = {span.label for span in spans}
        for label in LABELS:
            threshold = posterior_threshold(rules, label, genre)
            if threshold is None or label not in task1[pid]:
                continue
            if mode == "missing" and label in present:
                continue
            if mode == "additional" and label not in present:
                continue
            ranked = sorted(
                candidates.get(pid, ()),
                key=lambda row: (float(row[2][label]), -row[0], -row[1]),
                reverse=True,
            )
            if mode == "additional":
                same_label = [span for span in rescued if span.label == label]
                ranked = [
                    row
                    for row in ranked
                    if all(
                        min(row[1], span.end) <= max(row[0], span.start)
                        for span in same_label
                    )
                ]
            if not ranked or float(ranked[0][2][label]) < threshold:
                continue
            start, end, _ = ranked[0]
            span = Span(start, end, label)
            if span in exact:
                raise AssertionError("rescue selected an exact duplicate")
            rescued.append(span)
            exact.add(span)
            present.add(label)
            additions[(genre, label)] += 1
        rescued.sort(key=lambda span: (span.start, span.end, span.label))
        output[pid] = rescued

    return output, {
        "n_input_spans": sum(len(spans) for spans in base.values()),
        "n_output_spans": sum(len(spans) for spans in output.values()),
        "n_added": sum(additions.values()),
        "addition_counts": {
            f"{genre}/{label}": count
            for (genre, label), count in sorted(additions.items())
        },
    }


def evaluate_rules(
    base: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    task1: Mapping[int, set[str]],
    candidates: Mapping[int, Sequence[tuple[int, int, Mapping[str, float]]]],
    gold: Mapping[int, Sequence[Span]],
    rules: Mapping[tuple[str, str | None], float],
    mode: str,
) -> tuple[dict[int, list[Span]], dict, dict]:
    pred, stats = rescue_missing_roles(base, records, task1, candidates, rules, mode)
    return pred, stats, score_slices(gold, pred, records)


def audit_grid(
    base: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    task1: Mapping[int, set[str]],
    candidates: Mapping[int, Sequence[tuple[int, int, Mapping[str, float]]]],
    gold: Mapping[int, Sequence[Span]],
    mode: str,
) -> list[dict]:
    base_score = span_partial_f1(gold, base)["f1"]
    rows = []
    for label in LABELS:
        for genre in GENRES:
            for threshold in THRESHOLD_GRID:
                rules = {(label, genre): threshold}
                pred, stats, scores = evaluate_rules(
                    base, records, task1, candidates, gold, rules, mode
                )
                rows.append(
                    {
                        "label": label,
                        "genre": genre,
                        "threshold": threshold,
                        "overall_f1": round(scores["overall"]["f1"], 6),
                        "delta": round(scores["overall"]["f1"] - base_score, 6),
                        "editorial_f1": round(scores["editorial"]["f1"], 6),
                        "debate_f1": round(scores["debate"]["f1"], 6),
                        "changed_span_count": changed_span_count(base, pred),
                        "n_added": stats["n_added"],
                    }
                )
    rows.sort(key=lambda row: (row["delta"], -row["n_added"]), reverse=True)
    return rows


def coordinate_search(
    base: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    task1: Mapping[int, set[str]],
    candidates: Mapping[int, Sequence[tuple[int, int, Mapping[str, float]]]],
    gold: Mapping[int, Sequence[Span]],
    mode: str,
) -> tuple[dict[tuple[str, str | None], float], list[dict]]:
    rules: dict[tuple[str, str | None], float] = {}
    current = span_partial_f1(gold, base)["f1"]
    trace = []
    remaining = {(label, genre) for label in LABELS for genre in GENRES}
    while remaining:
        best = None
        for cell in sorted(remaining):
            for threshold in THRESHOLD_GRID:
                candidate_rules = rules | {cell: threshold}
                pred, stats, scores = evaluate_rules(
                    base, records, task1, candidates, gold, candidate_rules, mode
                )
                score = scores["overall"]["f1"]
                item = (score, -stats["n_added"], cell, threshold, scores, stats, pred)
                if best is None or item[:2] > best[:2]:
                    best = item
        if best is None or best[0] <= current + 1e-12:
            break
        score, _, cell, threshold, scores, stats, pred = best
        rules[cell] = threshold
        remaining.remove(cell)
        trace.append(
            {
                "added_rule": f"{cell[0]}:{cell[1]}",
                "threshold": threshold,
                "overall_f1": round(score, 6),
                "delta_from_previous": round(score - current, 6),
                "editorial_f1": round(scores["editorial"]["f1"], 6),
                "debate_f1": round(scores["debate"]["f1"], 6),
                "n_added": stats["n_added"],
                "changed_span_count": changed_span_count(base, pred),
            }
        )
        current = score
    return rules, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--encoder-scores", type=Path, required=True)
    parser.add_argument("--mode", choices=("missing", "additional"), default="missing")
    parser.add_argument(
        "--rescue-rule",
        action="append",
        default=[],
        metavar="LABEL[:GENRE]=TAU",
    )
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--audit-grid", action="store_true")
    parser.add_argument("--coordinate-search", action="store_true")
    parser.add_argument("--use-coordinate-rules", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    base, records = load_span_records(args.base)
    task1 = load_t1_labels(args.t1)
    candidates = load_segment_candidates(args.encoder_scores)
    rules = parse_posterior_rules(args.rescue_rule)
    gold = None
    if args.gold:
        gold_all, _ = load_span_records(args.gold)
        missing = sorted(set(records) - set(gold_all))
        if missing:
            raise ValueError(f"{len(missing)} prediction paragraphs lack gold")
        gold = {pid: gold_all[pid] for pid in records}
    if (args.audit_grid or args.coordinate_search or args.use_coordinate_rules) and gold is None:
        raise ValueError("audit/search options require --gold")

    report: dict = {
        "config": {
            "base": str(args.base),
            "t1": str(args.t1),
            "encoder_scores": str(args.encoder_scores),
            "mode": args.mode,
            "rules": {
                f"{label}:{genre or '*'}": threshold
                for (label, genre), threshold in sorted(rules.items())
            },
        }
    }
    if args.audit_grid:
        report["audit_grid"] = audit_grid(
            base, records, task1, candidates, gold, args.mode
        )
    if args.coordinate_search or args.use_coordinate_rules:
        selected, trace = coordinate_search(
            base, records, task1, candidates, gold, args.mode
        )
        report["coordinate_search"] = {
            "selected_rules": {
                f"{label}:{genre}": threshold
                for (label, genre), threshold in sorted(selected.items())
            },
            "trace": trace,
        }
        if args.use_coordinate_rules:
            rules = selected

    pred, stats = rescue_missing_roles(
        base, records, task1, candidates, rules, args.mode
    )
    report["rescue"] = stats
    if gold is not None:
        report["scores"] = compact_scores(score_slices(gold, pred, records))
    if args.output:
        write_predictions(args.output, pred, records)
        report["output"] = str(args.output)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
