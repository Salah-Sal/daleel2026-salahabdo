"""Corpus constants quoted in the paper, recomputed from the organizer files.

Reproducible artifact for the descriptive numbers in paper/daleel2026_salahabdo.tex
(§2 data paragraph and the §3 S1 motivation): training size and genre split,
gold span count, character coverage, Task 1 label priors, the AN base rate,
and the dev/test input sizes with their genre splits. Counts and rates only;
no dataset text is written. No LLM calls; stdlib + the daleel package.

Definitions (each chosen to match the figure previously recorded in the
paper audit, paper/README.md "Audit 2026-08-06"):

- char_coverage: union of the gold span character ranges per paragraph,
  summed over paragraphs, divided by the total number of paragraph
  characters (train_task_2 texts). This is the definition behind the
  recorded 90.73% (NON_LLM_NON_DSPY_REPORT.md §2.3); overlapping gold
  spans are counted once. The summed-span-length variant is reported
  alongside for transparency (it differs because a few adjacent gold
  spans overlap). Whitespace-only spans (the two paragraph-964 defects
  dropped by daleel.splits.clean_task2_gold) do not change the union.
- an_base_rate: AN spans as a fraction of ALL raw gold spans in the
  training release (train_task_2, 2,975-span pool). This is the
  unconditional AN share that the paper's 11.5% denotes ("an AN span
  follows another AN 39% of the time, against an 11.5% base rate");
  paper/README.md records it as recomputed from gold without a
  metrics.json, so the pool is fixed here. The 39% is the conditional
  P(next span = AN | current span = AN) over consecutive spans ordered by
  start offset within each paragraph; both are emitted, plus the
  transition-pool alternative P(next = AN) over all consecutive pairs.
- task1 label rates: paragraphs carrying the label / 612, raw official
  Task 1 labels (no cleaning), e.g. CO 36/612 = scripts/route_task1.py
  CO_PRIOR.

Test per-label gold supports are NOT computable locally (test labels are
held by the organizers) and are deliberately absent.

Example (from shared-task/):
  uv run python scripts/paper_corpus_counts.py
  uv run python scripts/paper_corpus_counts.py --data-root ../resources/repos/Daleel2026/data
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel.artifacts import atomic_write_json, file_sha256
from daleel.constants import GENRES, LABELS
from daleel.data import OFFICIAL_CLONE, load_records
from daleel.splits import clean_task2_gold

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = OFFICIAL_CLONE / "data"
DEFAULT_OUT = REPO / "paper" / "corpus_counts.json"

SOURCE_FILES = {
    "train_task_1": Path("train") / "train_task_1.jsonl",
    "train_task_2": Path("train") / "train_task_2.jsonl",
    "dev_in": Path("dev") / "dev_in.jsonl",
    "test_in": Path("test") / "test_in.jsonl",
}


def genre_split(records: list[dict]) -> dict[str, int]:
    counts = Counter(r["type"] for r in records)
    unknown = set(counts) - set(GENRES)
    if unknown:
        raise ValueError(f"unexpected genre values: {sorted(unknown)}")
    return {g: counts[g] for g in GENRES}


def raw_spans(t2_records: list[dict]) -> dict[int, list[tuple[int, int, str]]]:
    return {
        r["paragraph_id"]: [
            (s["start_offset"], s["end_offset"], s["label"]) for s in r["labels"]
        ]
        for r in t2_records
    }


def char_coverage(
    texts: dict[int, str], spans: dict[int, list[tuple[int, int, str]]]
) -> dict:
    """Union-of-ranges coverage (the recorded definition) plus the summed-length variant."""
    total = sum(len(t) for t in texts.values())
    union = summed = overlapping_pairs = 0
    for pid, ss in spans.items():
        covered: set[int] = set()
        for start, end, _ in ss:
            if start < 0 or end > len(texts[pid]):
                raise ValueError(f"paragraph {pid}: span [{start}, {end}) out of bounds")
            covered.update(range(start, end))
            summed += end - start
        union += len(covered)
        ordered = sorted(ss)
        overlapping_pairs += sum(1 for a, b in zip(ordered, ordered[1:]) if b[0] < a[1])
    return {
        "definition": "union of gold span character ranges / total paragraph characters",
        "n_paragraph_chars": total,
        "n_covered_chars": union,
        "char_coverage": union / total,
        "char_coverage_sum_of_span_lengths": summed / total,
        "n_overlapping_adjacent_gold_span_pairs": overlapping_pairs,
    }


def an_rates(spans: dict[int, list[tuple[int, int, str]]]) -> dict:
    label_counts = Counter(label for ss in spans.values() for _, _, label in ss)
    n_spans = sum(label_counts.values())
    transitions: Counter[tuple[str, str]] = Counter()
    for ss in spans.values():
        ordered = sorted(ss)  # by start offset (then end) within the paragraph
        for a, b in zip(ordered, ordered[1:]):
            transitions[(a[2], b[2])] += 1
    n_transitions = sum(transitions.values())
    from_an = sum(v for (a, _), v in transitions.items() if a == "AN")
    to_an = sum(v for (_, b), v in transitions.items() if b == "AN")
    return {
        "definition": "AN spans / all raw gold spans (train_task_2)",
        "n_an_spans": label_counts["AN"],
        "n_gold_spans": n_spans,
        "an_base_rate": label_counts["AN"] / n_spans,
        "span_label_counts": {label: label_counts[label] for label in LABELS},
        "n_consecutive_span_pairs": n_transitions,
        "an_to_an_given_an": transitions[("AN", "AN")] / from_an,
        "next_is_an_over_all_pairs": to_an / n_transitions,
    }


def task1_rates(t1_records: list[dict]) -> dict:
    n = len(t1_records)
    per_label = {}
    for label in LABELS:
        k = sum(1 for r in t1_records if label in r["labels"])
        per_label[label] = {"count": k, "rate": k / n}
    return {
        "definition": "paragraphs carrying the label / all training paragraphs (raw Task 1 labels)",
        "n_paragraphs": n,
        "per_label": per_label,
        "n_paragraphs_without_labels": sum(1 for r in t1_records if not r["labels"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="organizer data directory (default: resources/repos/Daleel2026/data)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="JSON output path (default: paper/corpus_counts.json)",
    )
    args = ap.parse_args()

    paths = {key: args.data_root / rel for key, rel in SOURCE_FILES.items()}
    t1 = load_records(paths["train_task_1"])
    t2 = load_records(paths["train_task_2"])
    dev = load_records(paths["dev_in"])
    test = load_records(paths["test_in"])

    if [r["paragraph_id"] for r in t1] != [r["paragraph_id"] for r in t2] or any(
        a["text"] != b["text"] for a, b in zip(t1, t2)
    ):
        raise AssertionError("train_task_1 and train_task_2 disagree on ids or texts")

    texts = {r["paragraph_id"]: r["text"] for r in t2}
    spans = raw_spans(t2)
    n_gold_spans = sum(len(ss) for ss in spans.values())
    n_gold_spans_cleaned = sum(len(ss) for ss in clean_task2_gold(t2).values())
    t1_equals_t2 = all(
        set(r["labels"]) == {label for _, _, label in spans[r["paragraph_id"]]} for r in t1
    )

    result = {
        "n_train_paragraphs": len(t1),
        "train_genre": genre_split(t1),
        "n_gold_spans": n_gold_spans,
        "n_gold_spans_cleaned": n_gold_spans_cleaned,
        "task1_label_set_equals_task2_span_labels_everywhere": t1_equals_t2,
        "coverage": char_coverage(texts, spans),
        "task1_gold": task1_rates(t1),
        "an": an_rates(spans),
        "n_dev_paragraphs": len(dev),
        "dev_genre": genre_split(dev),
        "n_test_paragraphs": len(test),
        "test_genre": genre_split(test),
        "source": {
            key: {"file": str(SOURCE_FILES[key]), "sha256": file_sha256(path)}
            for key, path in paths.items()
        },
    }
    print(json.dumps(result, indent=2))
    atomic_write_json(args.out, result)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
