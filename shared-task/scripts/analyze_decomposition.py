"""Measure the proposal/atomization oracle before spending optimizer calls.

This is a read-only analysis command.  It takes a Task 2 prediction JSONL from
the frozen quote extractor, scores that proposal set, then replaces semantic
decisions with gold-derived oracle roles at two boundary resolutions:

1. original broad quote proposals;
2. quote proposals atomized at deterministic clause/connective boundaries.

The gap from (1) to (2) quantifies the value of boundary atomization; the gap
from the real proposal score to (2) is the end-to-end headroom available to the
independently compiled role stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.candidates import (  # noqa: E402
    CandidateAtom,
    assemble_role_spans,
    atomize_spans,
    gold_roles_for_atom,
)
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records  # noqa: E402
from daleel.io import read_jsonl  # noqa: E402
from daleel.metrics import Span, span_partial_f1, task1_macro_f1  # noqa: E402
from daleel.splits import clean_task1_gold, clean_task2_gold  # noqa: E402
from daleel.submission import validate_records_against_source  # noqa: E402


def _prediction_spans(rows: list[dict]) -> dict[int, list[Span]]:
    return {
        row["paragraph_id"]: [
            Span(item["start_offset"], item["end_offset"], item["label"])
            for item in row["labels"]
        ]
        for row in rows
    }


def _round_score(result: dict) -> dict:
    return {
        "precision": round(result["precision"], 6),
        "recall": round(result["recall"], 6),
        "f1": round(result["f1"], 6),
        "per_label_f1": {
            label: round(values["f1"], 6)
            for label, values in result["per_label"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preds", type=Path, required=True,
                        help="Task 2 quote-extractor predictions JSONL")
    parser.add_argument("--granularity", default="connective",
                        choices=("sentence", "clause", "connective"))
    parser.add_argument("--containment-threshold", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON report path")
    args = parser.parse_args()

    rows = read_jsonl(args.preds)
    t1 = load_records(TRAIN_TASK1)
    t2 = load_records(TRAIN_TASK2)
    source_by_id = {record["paragraph_id"]: record for record in t1}
    try:
        source = [source_by_id[row["paragraph_id"]] for row in rows]
    except KeyError as exc:
        raise SystemExit(f"prediction contains unknown training ID {exc.args[0]}") from exc
    problems = validate_records_against_source(rows, source, "task_2")
    if problems:
        raise SystemExit("invalid proposal file:\n- " + "\n- ".join(problems[:20]))

    proposal_by_id = _prediction_spans(rows)
    ids = [row["paragraph_id"] for row in rows]
    gold_all = clean_task2_gold(t2)
    gold = {pid: gold_all[pid] for pid in ids}

    coarse_oracle: dict[int, list[Span]] = {}
    atom_oracle: dict[int, list[Span]] = {}
    atom_counts = Counter()
    for record in source:
        pid = record["paragraph_id"]
        text = record["text"]

        coarse_atoms = [
            CandidateAtom(
                proposal.start,
                proposal.end,
                text[proposal.start : proposal.end],
                (proposal.label,),
            )
            for proposal in proposal_by_id[pid]
        ]
        coarse_decisions = [
            (
                gold_roles_for_atom(
                    atom,
                    gold[pid],
                    containment_threshold=args.containment_threshold,
                ),
                True,
            )
            for atom in coarse_atoms
        ]
        coarse_oracle[pid], _ = assemble_role_spans(coarse_atoms, coarse_decisions)

        atoms = atomize_spans(text, proposal_by_id[pid], args.granularity)
        decisions = []
        for atom in atoms:
            roles = gold_roles_for_atom(
                atom,
                gold[pid],
                containment_threshold=args.containment_threshold,
            )
            decisions.append((roles, True))
            atom_counts["total"] += 1
            atom_counts["none" if not roles else "positive"] += 1
            atom_counts["multi_role"] += len(roles) > 1
        atom_oracle[pid], _ = assemble_role_spans(atoms, decisions)

    proposal_score = span_partial_f1(gold, proposal_by_id)
    coarse_score = span_partial_f1(gold, coarse_oracle)
    atom_score = span_partial_f1(gold, atom_oracle)
    task1_gold_all = clean_task1_gold(t1, t2)
    task1_gold = {pid: task1_gold_all[pid] for pid in ids}
    task1_from_atom = {
        pid: {span.label for span in atom_oracle[pid]} for pid in ids
    }
    t1_oracle = task1_macro_f1(task1_gold, task1_from_atom)

    report = {
        "proposal_file": str(args.preds),
        "n_paragraphs": len(ids),
        "granularity": args.granularity,
        "containment_threshold": args.containment_threshold,
        "counts": {
            "gold_spans": sum(len(gold[pid]) for pid in ids),
            "proposal_spans": sum(len(proposal_by_id[pid]) for pid in ids),
            "coarse_oracle_spans": sum(len(coarse_oracle[pid]) for pid in ids),
            "atom_oracle_spans": sum(len(atom_oracle[pid]) for pid in ids),
            **dict(atom_counts),
        },
        "task2_proposals": _round_score(proposal_score),
        "task2_coarse_role_oracle": _round_score(coarse_score),
        "task2_atomized_role_oracle": _round_score(atom_score),
        "task2_headroom": round(atom_score["f1"] - proposal_score["f1"], 6),
        "task1_from_atomized_oracle": {
            "macro_f1": round(t1_oracle["macro_f1"], 6),
            "micro_f1": round(t1_oracle["micro_f1"], 6),
            "per_label_f1": {
                label: round(values["f1"], 6)
                for label, values in t1_oracle["per_label"].items()
            },
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
