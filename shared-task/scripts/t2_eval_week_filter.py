"""Audit and deploy eval-week Task 2 span-presence filters.

The frozen S1 system is a high-recall span labeler.  During the evaluation
week, precision improved by removing an S1 span when either:

* its label was absent from the independently composed Task 1 prediction; or
* the Task 2 segment encoder assigned too little posterior mass to that label.

This command makes those rules reproducible, scores genre/label marginals when
gold is available, and writes a source-faithful Task 2 JSONL for packaging.
It never adds, duplicates, merges, or changes span boundaries.

Examples (from ``shared-task/``)::

    uv run scripts/t2_eval_week_filter.py \
      --s1 experiments/.../predictions/task_2.jsonl \
      --t1 /path/to/task_1.jsonl \
      --encoder-scores experiments/.../predictions/task_2_segment_scores.jsonl \
      --gold ../resources/repos/Daleel2026/data/dev/dev_task_2_ref.jsonl \
      --absence-labels CO,AN,OT,TE,ST \
      --posterior-rule CO=0.3 --posterior-rule OT=0.3 \
      --audit-grid

    uv run scripts/t2_eval_week_filter.py ... --output /tmp/task_2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.constants import LABELS
from daleel.io import read_jsonl, write_jsonl
from daleel.metrics import Span, span_partial_f1

GENRES = ("editorial", "debate")


def load_span_records(path: Path) -> tuple[dict[int, list[Span]], dict[int, dict]]:
    spans: dict[int, list[Span]] = {}
    records: dict[int, dict] = {}
    for row in read_jsonl(path):
        pid = int(row["paragraph_id"])
        if pid in records:
            raise ValueError(f"duplicate paragraph_id {pid} in {path}")
        rows = row.get("labels", row.get("spans"))
        if rows is None:
            raise ValueError(f"paragraph {pid} has neither labels nor spans in {path}")
        spans[pid] = [
            Span(int(s["start_offset"]), int(s["end_offset"]), str(s["label"]))
            for s in rows
        ]
        records[pid] = row
    return spans, records


def load_t1_labels(path: Path) -> dict[int, set[str]]:
    """Load either a JSON object ``pid -> labels`` or Task 1 JSONL."""
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object in {path}")
        return {int(pid): set(labels) for pid, labels in payload.items()}
    labels: dict[int, set[str]] = {}
    for row in read_jsonl(path):
        pid = int(row["paragraph_id"])
        labels[pid] = set(row.get("labels", ()))
    return labels


def load_exact_segment_scores(path: Path) -> dict[tuple[int, int, int], dict[str, float]]:
    """Load raw encoder scores keyed by an exact segment boundary.

    The eval-week A rule deliberately used an exact ``(pid, start, end)``
    join.  S1 spans without a boundary-identical deterministic encoder segment
    are retained.  This is narrower than S1's earlier overlap-weighted
    structural re-decode and is required to reproduce the scored probes.
    """
    scores: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in read_jsonl(path):
        key = (
            int(row["paragraph_id"]),
            int(row["start_offset"]),
            int(row["end_offset"]),
        )
        if key in scores:
            raise ValueError(f"duplicate encoder segment {key} in {path}")
        scores[key] = {label: float(value) for label, value in row["scores"].items()}
    return scores


def parse_absence_labels(raw: str) -> set[str]:
    labels = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = labels - set(LABELS)
    if unknown:
        raise ValueError(f"unknown absence labels: {sorted(unknown)}")
    return labels


def parse_posterior_rules(raw_rules: Iterable[str]) -> dict[tuple[str, str | None], float]:
    """Parse LABEL=TAU or LABEL:GENRE=TAU rules.

    A genre-specific rule takes precedence over a label-global rule.
    """
    rules: dict[tuple[str, str | None], float] = {}
    for raw in raw_rules:
        try:
            left, value = raw.split("=", 1)
            threshold = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid posterior rule {raw!r}; use LABEL[:GENRE]=TAU") from exc
        if ":" in left:
            label, genre = left.split(":", 1)
        else:
            label, genre = left, None
        if label not in LABELS:
            raise ValueError(f"unknown label in posterior rule {raw!r}")
        if genre is not None and genre not in GENRES:
            raise ValueError(f"unknown genre in posterior rule {raw!r}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"posterior threshold must be in [0,1]: {raw!r}")
        rules[(label, genre)] = threshold
    return rules


def posterior_threshold(
    rules: Mapping[tuple[str, str | None], float], label: str, genre: str
) -> float | None:
    return rules.get((label, genre), rules.get((label, None)))


def apply_filters(
    s1: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    t1: Mapping[int, set[str]],
    encoder_scores: Mapping[tuple[int, int, int], Mapping[str, float]],
    absence_labels: set[str],
    posterior_rules: Mapping[tuple[str, str | None], float],
) -> tuple[dict[int, list[Span]], dict]:
    output: dict[int, list[Span]] = {}
    drop_counts: Counter[tuple[str, str, str]] = Counter()
    missing_t1 = sorted(set(s1) - set(t1))
    if missing_t1:
        raise ValueError(f"{len(missing_t1)} S1 paragraphs lack Task 1 labels: {missing_t1[:5]}")

    for pid, spans in s1.items():
        genre = records[pid]["type"]
        kept: list[Span] = []
        for span in spans:
            absence_drop = span.label in absence_labels and span.label not in t1[pid]
            threshold = posterior_threshold(posterior_rules, span.label, genre)
            scores = encoder_scores.get((pid, span.start, span.end))
            posterior_drop = (
                threshold is not None
                and scores is not None
                and scores.get(span.label, 1.0) < threshold
            )
            if absence_drop or posterior_drop:
                reason = (
                    "absence+posterior"
                    if absence_drop and posterior_drop
                    else "absence"
                    if absence_drop
                    else "posterior"
                )
                drop_counts[(genre, span.label, reason)] += 1
            else:
                kept.append(span)
        output[pid] = kept

    return output, {
        "n_input_spans": sum(len(v) for v in s1.values()),
        "n_output_spans": sum(len(v) for v in output.values()),
        "n_dropped": sum(drop_counts.values()),
        "drop_counts": {
            f"{genre}/{label}/{reason}": count
            for (genre, label, reason), count in sorted(drop_counts.items())
        },
    }


def score_slices(
    gold: Mapping[int, Sequence[Span]],
    pred: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
) -> dict:
    report = {"overall": span_partial_f1(gold, pred)}
    for genre in GENRES:
        pids = [pid for pid in gold if records[pid]["type"] == genre]
        report[genre] = span_partial_f1(
            {pid: gold[pid] for pid in pids}, {pid: pred.get(pid, ()) for pid in pids}
        )
    return report


def compact_scores(report: Mapping[str, dict]) -> dict:
    return {
        split: {
            "precision": round(scores["precision"], 6),
            "recall": round(scores["recall"], 6),
            "f1": round(scores["f1"], 6),
            "per_label_f1": {
                label: round(values["f1"], 6)
                for label, values in scores["per_label"].items()
            },
        }
        for split, scores in report.items()
    }


def changed_span_count(
    left: Mapping[int, Sequence[Span]], right: Mapping[int, Sequence[Span]]
) -> int:
    return sum(
        len(set(left.get(pid, ())) ^ set(right.get(pid, ())))
        for pid in set(left) | set(right)
    )


def audit_grid(
    s1: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
    t1: Mapping[int, set[str]],
    encoder_scores: Mapping[tuple[int, int, int], Mapping[str, float]],
    gold: Mapping[int, Sequence[Span]],
    absence_labels: set[str],
    posterior_rules: Mapping[tuple[str, str | None], float],
    current: Mapping[int, Sequence[Span]],
) -> dict:
    current_score = span_partial_f1(gold, current)["f1"]
    marginals = []

    # Toggle cross-task absence filtering separately for every label.
    for label in LABELS:
        candidate_absence = set(absence_labels)
        if label in candidate_absence:
            candidate_absence.remove(label)
            action = "disable"
        else:
            candidate_absence.add(label)
            action = "enable"
        candidate, stats = apply_filters(
            s1,
            records,
            t1,
            encoder_scores,
            candidate_absence,
            posterior_rules,
        )
        scores = score_slices(gold, candidate, records)
        marginals.append(
            {
                "kind": "absence",
                "label": label,
                "action": action,
                "overall_f1": round(scores["overall"]["f1"], 6),
                "delta": round(scores["overall"]["f1"] - current_score, 6),
                "editorial_f1": round(scores["editorial"]["f1"], 6),
                "debate_f1": round(scores["debate"]["f1"], 6),
                "changed_span_count": changed_span_count(current, candidate),
                "n_output_spans": stats["n_output_spans"],
            }
        )

    # Replace the current threshold for one label/genre cell at a time.
    threshold_grid = [round(step * 0.05, 2) for step in range(17)]
    for label in LABELS:
        for genre in GENRES:
            for threshold in threshold_grid:
                candidate_rules = dict(posterior_rules)
                candidate_rules[(label, genre)] = threshold
                candidate, stats = apply_filters(
                    s1,
                    records,
                    t1,
                    encoder_scores,
                    absence_labels,
                    candidate_rules,
                )
                scores = score_slices(gold, candidate, records)
                marginals.append(
                    {
                        "kind": "posterior",
                        "label": label,
                        "genre": genre,
                        "threshold": threshold,
                        "overall_f1": round(scores["overall"]["f1"], 6),
                        "delta": round(scores["overall"]["f1"] - current_score, 6),
                        "editorial_f1": round(scores["editorial"]["f1"], 6),
                        "debate_f1": round(scores["debate"]["f1"], 6),
                        "changed_span_count": changed_span_count(current, candidate),
                        "n_output_spans": stats["n_output_spans"],
                    }
                )

    marginals.sort(key=lambda row: (row["delta"], -row["changed_span_count"]), reverse=True)
    return {"current_f1": round(current_score, 6), "marginals": marginals}


def write_predictions(
    path: Path,
    predictions: Mapping[int, Sequence[Span]],
    records: Mapping[int, dict],
) -> None:
    rows = []
    for pid in records:
        source = records[pid]
        rows.append(
            {
                "paragraph_id": pid,
                "text": source["text"],
                "type": source["type"],
                "labels": [
                    {
                        "label": span.label,
                        "start_offset": span.start,
                        "end_offset": span.end,
                    }
                    for span in predictions.get(pid, ())
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--encoder-scores", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--absence-labels", default="CO,AN,OT,TE,ST")
    parser.add_argument(
        "--posterior-rule",
        action="append",
        default=[],
        metavar="LABEL[:GENRE]=TAU",
    )
    parser.add_argument("--audit-grid", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    s1, records = load_span_records(args.s1)
    t1 = load_t1_labels(args.t1)
    encoder_scores = load_exact_segment_scores(args.encoder_scores)
    absence_labels = parse_absence_labels(args.absence_labels)
    posterior_rules = parse_posterior_rules(args.posterior_rule)
    pred, filter_stats = apply_filters(
        s1, records, t1, encoder_scores, absence_labels, posterior_rules
    )

    report: dict = {
        "config": {
            "s1": str(args.s1),
            "t1": str(args.t1),
            "encoder_scores": str(args.encoder_scores),
            "absence_labels": sorted(absence_labels),
            "posterior_rules": {
                f"{label}:{genre or '*'}": threshold
                for (label, genre), threshold in sorted(
                    posterior_rules.items(), key=lambda item: (item[0][0], item[0][1] or "")
                )
            },
        },
        "filters": filter_stats,
    }

    if args.gold:
        gold_all, _ = load_span_records(args.gold)
        missing_gold = sorted(set(records) - set(gold_all))
        if missing_gold:
            raise ValueError(f"{len(missing_gold)} prediction paragraphs lack gold: {missing_gold[:5]}")
        # OOF campaigns may intentionally cover a strict subset of a larger
        # organizer gold file. Ignore gold-only rows so genre slicing and the
        # metric operate on exactly the prediction contract.
        gold = {pid: gold_all[pid] for pid in records}
        report["scores"] = compact_scores(score_slices(gold, pred, records))
        if args.audit_grid:
            report["audit_grid"] = audit_grid(
                s1,
                records,
                t1,
                encoder_scores,
                gold,
                absence_labels,
                posterior_rules,
                pred,
            )
    elif args.audit_grid:
        raise ValueError("--audit-grid requires --gold")

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
