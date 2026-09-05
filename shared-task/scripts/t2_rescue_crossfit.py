"""Cross-fit the two-stage Task 2 encoder rescue on released dev references.

The deployment recipe in :mod:`t2_presence_rescue` first restores a role that
is completely missing from a paragraph, then considers one non-overlapping
additional occurrence of an already-present role.  A coordinate search on the
whole released development set is useful for building a candidate, but its
score is optimistic because the same paragraphs select and evaluate rules.

This audit repeats both searches inside paragraph-grouped folds and evaluates
the resulting two-stage recipe only on each held-out fold.  It writes a single
OOF prediction and records every fold's selected rules, base score, intermediate
score, and final score.  It never reads evaluation labels or changes candidate
boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from daleel.folds import make_stratified_folds
from daleel.metrics import Span

from t2_eval_week_filter import (
    compact_scores,
    load_span_records,
    load_t1_labels,
    score_slices,
    write_predictions,
)
from t2_presence_rescue import (
    coordinate_search,
    load_segment_candidates,
    rescue_missing_roles,
)

T = TypeVar("T")


def subset(mapping: Mapping[int, T], pids: Sequence[int]) -> dict[int, T]:
    return {pid: mapping[pid] for pid in pids}


def serialize_rules(rules: Mapping[tuple[str, str | None], float]) -> dict[str, float]:
    return {
        f"{label}:{genre or '*'}": threshold
        for (label, genre), threshold in sorted(rules.items())
    }


def deserialize_rules(rules: Mapping[str, float]) -> dict[tuple[str, str | None], float]:
    output = {}
    for cell, threshold in rules.items():
        label, genre = cell.split(":", 1)
        output[(label, None if genre == "*" else genre)] = float(threshold)
    return output


def rules_from_trace(trace: Sequence[Mapping], limit: int) -> dict[tuple[str, str], float]:
    rules = {}
    for row in trace[:limit]:
        label, genre = str(row["added_rule"]).split(":", 1)
        rules[(label, genre)] = float(row["threshold"])
    return rules


def changed_span_count(
    left: Mapping[int, Sequence[Span]], right: Mapping[int, Sequence[Span]]
) -> int:
    return sum(
        len(set(left.get(pid, ())) ^ set(right.get(pid, ())))
        for pid in set(left) | set(right)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--encoder-scores", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260730)
    parser.add_argument(
        "--replay-report",
        type=Path,
        help=(
            "reuse fold-local selections from a prior report and only replay "
            "held-out rule prefixes (avoids repeating the coordinate searches)"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    base, records = load_span_records(args.base)
    task1 = load_t1_labels(args.t1)
    candidates = load_segment_candidates(args.encoder_scores)
    gold, _ = load_span_records(args.gold)
    pids = sorted(base)

    for name, mapping in (("records", records), ("Task 1", task1), ("gold", gold)):
        missing = sorted(set(pids) - set(mapping))
        if missing:
            raise ValueError(f"{len(missing)} base paragraphs lack {name}: {missing[:5]}")

    labels = {pid: {span.label for span in gold[pid]} for pid in pids}
    genres = {pid: records[pid]["type"] for pid in pids}
    folds = make_stratified_folds(
        pids, labels, genres, n_splits=args.folds, seed=args.fold_seed
    )

    missing_oof: dict[int, list[Span]] = {}
    final_oof: dict[int, list[Span]] = {}
    fold_reports = []
    missing_rule_counts: Counter[str] = Counter()
    additional_rule_counts: Counter[str] = Counter()
    prefix_limit = 12
    missing_prefix_oof: dict[int, dict[int, list[Span]]] = {
        index: {} for index in range(prefix_limit + 1)
    }
    additional_prefix_oof: dict[int, dict[int, list[Span]]] = {
        index: {} for index in range(prefix_limit + 1)
    }
    replay_folds = None
    if args.replay_report:
        replay_payload = json.loads(args.replay_report.read_text())
        replay_folds = {int(row["fold"]): row for row in replay_payload["fold_reports"]}

    for fold in folds:
        train_ids = list(fold.train_ids)
        val_ids = list(fold.val_ids)
        train_base = subset(base, train_ids)
        train_records = subset(records, train_ids)
        train_task1 = subset(task1, train_ids)
        train_candidates = subset(candidates, [pid for pid in train_ids if pid in candidates])
        train_gold = subset(gold, train_ids)

        if replay_folds is None:
            missing_rules, missing_trace = coordinate_search(
                train_base,
                train_records,
                train_task1,
                train_candidates,
                train_gold,
                mode="missing",
            )
        else:
            replay = replay_folds[fold.index]
            missing_rules = deserialize_rules(replay["missing_rules"])
            missing_trace = replay["missing_search_trace"]
        train_missing, _ = rescue_missing_roles(
            train_base,
            train_records,
            train_task1,
            train_candidates,
            missing_rules,
            mode="missing",
        )
        if replay_folds is None:
            additional_rules, additional_trace = coordinate_search(
                train_missing,
                train_records,
                train_task1,
                train_candidates,
                train_gold,
                mode="additional",
            )
        else:
            additional_rules = deserialize_rules(replay["additional_rules"])
            additional_trace = replay["additional_search_trace"]

        val_base = subset(base, val_ids)
        val_records = subset(records, val_ids)
        val_task1 = subset(task1, val_ids)
        val_candidates = subset(candidates, [pid for pid in val_ids if pid in candidates])
        val_gold = subset(gold, val_ids)
        val_missing, missing_stats = rescue_missing_roles(
            val_base,
            val_records,
            val_task1,
            val_candidates,
            missing_rules,
            mode="missing",
        )
        val_final, additional_stats = rescue_missing_roles(
            val_missing,
            val_records,
            val_task1,
            val_candidates,
            additional_rules,
            mode="additional",
        )
        missing_oof.update(val_missing)
        final_oof.update(val_final)
        for limit in range(prefix_limit + 1):
            missing_prefix, _ = rescue_missing_roles(
                val_base,
                val_records,
                val_task1,
                val_candidates,
                rules_from_trace(missing_trace, limit),
                mode="missing",
            )
            missing_prefix_oof[limit].update(missing_prefix)
            additional_prefix, _ = rescue_missing_roles(
                val_missing,
                val_records,
                val_task1,
                val_candidates,
                rules_from_trace(additional_trace, limit),
                mode="additional",
            )
            additional_prefix_oof[limit].update(additional_prefix)

        serialized_missing = serialize_rules(missing_rules)
        serialized_additional = serialize_rules(additional_rules)
        missing_rule_counts.update(
            f"{cell}={threshold}" for cell, threshold in serialized_missing.items()
        )
        additional_rule_counts.update(
            f"{cell}={threshold}" for cell, threshold in serialized_additional.items()
        )
        fold_reports.append(
            {
                "fold": fold.index,
                "n_train_paragraphs": len(train_ids),
                "n_validation_paragraphs": len(val_ids),
                "missing_rules": serialized_missing,
                "missing_search_trace": missing_trace,
                "additional_rules": serialized_additional,
                "additional_search_trace": additional_trace,
                "validation_base_scores": compact_scores(
                    score_slices(val_gold, val_base, val_records)
                ),
                "validation_missing_scores": compact_scores(
                    score_slices(val_gold, val_missing, val_records)
                ),
                "validation_final_scores": compact_scores(
                    score_slices(val_gold, val_final, val_records)
                ),
                "validation_missing_stats": missing_stats,
                "validation_additional_stats": additional_stats,
            }
        )

    if set(final_oof) != set(pids) or set(missing_oof) != set(pids):
        raise AssertionError("OOF predictions do not cover every paragraph exactly once")

    report = {
        "method": "paragraph-grouped cross-fit of missing then additional rescue searches",
        "config": {
            "base": str(args.base),
            "t1": str(args.t1),
            "encoder_scores": str(args.encoder_scores),
            "gold": str(args.gold),
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "replay_report": str(args.replay_report) if args.replay_report else None,
        },
        "n_paragraphs": len(pids),
        "base_scores": compact_scores(score_slices(gold, base, records)),
        "missing_oof_scores": compact_scores(score_slices(gold, missing_oof, records)),
        "final_oof_scores": compact_scores(score_slices(gold, final_oof, records)),
        "changed_span_count_vs_base": changed_span_count(base, final_oof),
        "rule_selection_frequency": {
            "missing": dict(sorted(missing_rule_counts.items())),
            "additional": dict(sorted(additional_rule_counts.items())),
        },
        "heldout_prefix_scores": {
            "missing_from_base": {
                str(limit): compact_scores(score_slices(gold, pred, records))
                for limit, pred in missing_prefix_oof.items()
            },
            "additional_after_all_missing": {
                str(limit): compact_scores(score_slices(gold, pred, records))
                for limit, pred in additional_prefix_oof.items()
            },
        },
        "fold_reports": fold_reports,
    }
    if args.output:
        write_predictions(args.output, final_oof, records)
        report["output"] = str(args.output)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
