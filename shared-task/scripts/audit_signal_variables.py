"""Ceiling audit for the (source, form, stance) signal-variable design.

Design-phase instrument for the Task 1 redesign (2026-07-11):
before building a classifier that conditions on intermediate
signal variables, measure how much label signal the variables carry at
best. The extractor is deliberately LABEL-BLIND — it never sees the six
ADU labels — so any label information recovered downstream provably
arrived through the three axes:

  source: who asserts the unit's content?   author | attributed | procedural
  form:   what kind of content is it?       generic | instance | quantitative | question-or-filler
  stance: the author's argumentative move?  argued | presupposed | none

Units come from the deterministic segmenter (connective granularity,
segmentation oracle 0.920), so offsets are exact and unit-level gold can
be read off the cleaned Task 2 spans by character overlap. The audit runs
on the 430 train-side paragraphs only — the frozen val stays untouched.

Three mappings are scored at paragraph level (macro-F1, official metric):
- theory: the pre-registered deterministic cell->label mapping;
- fitted greedy mapping (optimistic ceiling — fit and scored on the same
  paragraphs);
- 5-fold cross-fitted greedy mapping (the honest estimate).

Artifacts land in experiments/<stamp>-t1-signal-audit-<model>-<n>/:
config.json + metrics.json (committed; enum tokens and counts only, no
dataset text) and predictions/extractions.jsonl (gitignored — text).

Examples (from shared-task/):
  uv run scripts/audit_signal_variables.py --model gemma-4-31b-paid --limit 8
  uv run scripts/audit_signal_variables.py --model gemma-4-31b-paid
"""

import argparse
import concurrent.futures
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from daleel import runtime as _runtime  # noqa: E402,F401 -- before dspy import

import dspy
import pydantic

from daleel.artifacts import atomic_write_json, create_experiment_dir, file_sha256
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.io import write_jsonl
from daleel.metrics import task1_macro_f1
from daleel.models import SPECS, make_lm
from daleel.policy import TASK_CONTEXT
from daleel.runtime import EXPERIMENTS_DIR
from daleel.segment import segment_paragraph
from daleel.splits import clean_task1_gold, clean_task2_gold, train_val_ids

AXES: dict[str, tuple[str, ...]] = {
    "source": ("author", "attributed", "procedural"),
    "form": ("generic", "instance", "quantitative", "question-or-filler"),
    "stance": ("argued", "presupposed", "none"),
}

# Containment net for near-miss tokens; anything else becomes "?" and is
# counted, never guessed.
SYNONYMS: dict[str, dict[str, str]] = {
    "source": {
        "quoted": "attributed", "reported": "attributed", "other": "attributed",
        "speaker": "author", "writer": "author",
        "moderator": "procedural", "nobody": "procedural", "none": "procedural",
    },
    "form": {
        "claim": "generic", "generic-claim": "generic",
        "specific": "instance", "specific-instance": "instance",
        "event": "instance", "example": "instance",
        "statistic": "quantitative", "statistics": "quantitative",
        "number": "quantitative", "numbers": "quantitative",
        "question": "question-or-filler", "filler": "question-or-filler",
        "procedural": "question-or-filler",
    },
    "stance": {
        "argued-for": "argued", "arguing": "argued", "argue": "argued",
        "shared": "presupposed", "presupposed-shared": "presupposed",
        "background": "none", "no-argumentative-work": "none",
    },
}


def norm(axis: str, raw: str) -> str:
    token = str(raw or "").strip().lower().replace("_", "-").replace(" ", "-")
    if token in AXES[axis]:
        return token
    return SYNONYMS[axis].get(token, "?")


class UnitSignal(pydantic.BaseModel):
    """The three signal variables for one unit."""

    # str, not Literal: nested Literals are validated case-sensitively by
    # TypeAdapter and one near-miss would fail the whole paragraph (see the
    # ADUQuote note in dspy_programs). norm() contains instead.
    source: str = pydantic.Field(
        description="exactly one of: author, attributed, procedural"
    )
    form: str = pydantic.Field(
        description="exactly one of: generic, instance, quantitative, question-or-filler"
    )
    stance: str = pydantic.Field(
        description="exactly one of: argued, presupposed, none"
    )


class AnnotateUnits(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

The paragraph has been pre-segmented into numbered clause-level units,
given in order. Annotate EVERY unit on three independent axes — one
annotation object per unit, same order and count as `units`.

source — whose voice asserts the unit's content:
- author: the writer or speaker asserts it in their own voice.
- attributed: the content is quoted or reported from an identifiable
  other source — a named person, official, expert, study, or the
  opposing team.
- procedural: nobody asserts content — greetings, introducing speakers,
  announcing the structure of the speech, pure fillers.

form — what kind of content the unit carries:
- generic: a general claim, judgment, evaluation, prediction or proposal
  about how things are or should be.
- instance: a concrete specific event, case, personal experience or
  historical example — particular actors, places, times.
- quantitative: explicit numbers, percentages, survey or study results.
- question-or-filler: a question, exclamation, transition, or other
  discourse management with no propositional content of its own.

stance — the argumentative move the author makes with the unit:
- argued: advanced as a point the author needs to establish or defend —
  the audience might dispute it.
- presupposed: presented as already shared, accepted background that
  nobody in the audience would dispute — definitions, proverbs, settled
  facts, settled legal or procedural background.
- none: no argumentative work is being done with this unit."""

    text: str = dspy.InputField(desc="the full Arabic paragraph, for context")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    units: list[str] = dspy.InputField(desc="the paragraph's units, in order")
    signals: list[UnitSignal] = dspy.OutputField(
        desc="exactly one annotation per unit, same order and count as `units`"
    )


class SignalProgram(dspy.Module):
    """Label-blind annotator: one call per paragraph, one triple per unit."""

    def __init__(self, cot: bool = True):
        super().__init__()
        self.annotate = (dspy.ChainOfThought if cot else dspy.Predict)(AnnotateUnits)

    def forward(self, text: str, genre: str) -> dspy.Prediction:
        segments = segment_paragraph(text, "connective")
        if not segments:
            return dspy.Prediction(segments=[], triples=[], length_mismatch=False)
        result = self.annotate(text=text, genre=genre, units=[s.text for s in segments])
        raw = result.signals if isinstance(result.signals, list) else []

        def field(item, key: str) -> str:
            got = item.get(key, "") if isinstance(item, dict) else getattr(item, key, "")
            return str(got or "")

        triples = []
        for i in range(len(segments)):
            item = raw[i] if i < len(raw) else None
            triples.append(
                tuple(norm(axis, field(item, axis)) for axis in AXES)
                if item is not None
                else ("?", "?", "?")
            )
        return dspy.Prediction(
            segments=segments,
            triples=triples,
            length_mismatch=len(raw) != len(segments),
        )


def theory_label(source: str, form: str, stance: str) -> str | None:
    """The pre-registered deterministic mapping (registered 2026-07-11).

    Precedence encodes the label policy: quantitative outranks attribution
    (ST 'even when embedded inside reported speech'); attribution outranks
    everything else (TE); procedural voice or a no-work stance is OT;
    concrete instances are AN; presupposed generics are CO; what remains —
    the author arguing generic content, including rhetorical questions —
    is AS.
    """
    if "?" in (source, form, stance):
        return None
    if form == "quantitative":
        return "ST"
    if source == "attributed":
        return "TE"
    if source == "procedural" or stance == "none":
        return "OT"
    if form == "instance":
        return "AN"
    if stance == "presupposed":
        return "CO"
    return "AS"


def _binary_f1(pred_pos: set[int], gold_pos: set[int]) -> float:
    tp = len(pred_pos & gold_pos)
    if not pred_pos or not gold_pos:
        return 0.0
    p, r = tp / len(pred_pos), tp / len(gold_pos)
    return 2 * p * r / (p + r) if p + r else 0.0


def greedy_cells(
    cells_by_pid: dict[int, set[tuple]],
    gold_pos: set[int],
    candidates: list[tuple],
    max_cells: int = 25,
) -> list[tuple]:
    """Greedily pick the cell set whose union-fire maximizes one label's F1."""
    fires: dict[tuple, set[int]] = {
        c: {pid for pid, cells in cells_by_pid.items() if c in cells} for c in candidates
    }
    chosen: list[tuple] = []
    covered: set[int] = set()
    best = 0.0
    while len(chosen) < max_cells:
        gain_cell, gain_f1 = None, best
        for c in candidates:
            if c in chosen:
                continue
            f1 = _binary_f1(covered | fires[c], gold_pos)
            if f1 > gain_f1 + 1e-9:
                gain_cell, gain_f1 = c, f1
        if gain_cell is None:
            break
        chosen.append(gain_cell)
        covered |= fires[gain_cell]
        best = gain_f1
    return chosen


def fit_mapping(
    pids: list[int],
    cells_by_pid: dict[int, set[tuple]],
    gold: dict[int, set[str]],
    candidates: list[tuple],
) -> dict[str, list[tuple]]:
    subset = {pid: cells_by_pid[pid] for pid in pids}
    return {
        label: greedy_cells(subset, {pid for pid in pids if label in gold[pid]}, candidates)
        for label in LABELS
    }


def apply_mapping(
    pids: list[int],
    cells_by_pid: dict[int, set[tuple]],
    mapping: dict[str, list[tuple]],
) -> dict[int, set[str]]:
    return {
        pid: {
            label
            for label, cells in mapping.items()
            if cells_by_pid[pid] & set(cells)
        }
        for pid in pids
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(SPECS), default="gemma-4-31b-paid")
    ap.add_argument("--limit", type=int, default=None,
                    help="deterministic subsample of the train side (smoke test)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-cot", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4000)
    args = ap.parse_args()

    spec = SPECS[args.model]
    t1, t2 = load_records(TRAIN_TASK1), load_records(TRAIN_TASK2)
    train_ids, _ = train_val_ids(t1, t2)
    keep = set(train_ids)
    records = [r for r in t1 if r["paragraph_id"] in keep]
    if args.limit:
        records = sorted(
            random.Random(0).sample(records, args.limit),
            key=lambda r: r["paragraph_id"],
        )
    gold_labels = {
        pid: labels
        for pid, labels in clean_task1_gold(t1, t2).items()
        if pid in {r["paragraph_id"] for r in records}
    }
    gold_spans = clean_task2_gold(t2)

    resolved_config = {
        "script": "audit_signal_variables.py",
        "argv": sys.argv[1:],
        "model": spec.__dict__,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "dspy_version": dspy.__version__,
        "adapter": "ChatAdapter",
        "cot": not args.no_cot,
        "granularity": "connective",
        "axes": {axis: list(values) for axis, values in AXES.items()},
        "selection": f"train{':' + str(args.limit) if args.limit else ''}",
        "source_path": str(TRAIN_TASK1),
        "source_sha256": file_sha256(TRAIN_TASK1),
        "n_paragraphs": len(records),
    }
    exp_dir = create_experiment_dir(
        EXPERIMENTS_DIR,
        f"t1-signal-audit-{spec.key}-n{len(records)}",
        resolved_config,
    )
    atomic_write_json(exp_dir / "config.json", resolved_config)
    dspy.configure(lm=make_lm(spec, max_tokens=args.max_tokens), adapter=dspy.ChatAdapter())
    program = SignalProgram(cot=not args.no_cot)

    def annotate_one(r: dict):
        last = None
        for attempt in range(4):
            try:
                return program(text=r["text"], genre=r["type"]), None
            except Exception as e:  # containment: never drop a row
                last = f"{r['paragraph_id']}: {type(e).__name__}: {e}"
                if not dspy.is_retryable_lm_error(e) or attempt == 3:
                    break
                time.sleep(min(75, 25 * (attempt + 1)))
        return None, last

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(annotate_one, records))
    wall = time.time() - t0

    # ---- collect: unit-level rows + paragraph-level fired-cell sets
    out_rows, errors = [], []
    n_parse_fail = n_mismatch = 0
    unknown_by_axis = Counter()
    n_units = 0
    cells_by_pid: dict[int, set[tuple]] = {}
    cell_gold = defaultdict(Counter)  # cell -> unit-gold-label counts
    axis_gold = {axis: defaultdict(Counter) for axis in AXES}
    for r, (pred, err) in zip(records, results):
        pid = r["paragraph_id"]
        if err:
            errors.append(err)
        n_parse_fail += pred is None
        n_mismatch += bool(pred and pred.length_mismatch)
        units_out, fired = [], set()
        for seg, triple in zip(pred.segments, pred.triples) if pred else ():
            n_units += 1
            for axis, value in zip(AXES, triple):
                unknown_by_axis[axis] += value == "?"
            best_label, best_ov = "NONE", 0
            for sp in gold_spans[pid]:
                ov = min(seg.end, sp.end) - max(seg.start, sp.start)
                if ov > best_ov:
                    best_ov, best_label = ov, sp.label
            cell_gold[triple][best_label] += 1
            for axis, value in zip(AXES, triple):
                axis_gold[axis][value][best_label] += 1
            fired.add(triple)
            units_out.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "source": triple[0],
                    "form": triple[1],
                    "stance": triple[2],
                    "unit_gold": best_label,
                }
            )
        cells_by_pid[pid] = fired
        out_rows.append(
            {
                "paragraph_id": pid,
                "type": r["type"],
                "gold_labels": sorted(gold_labels[pid]),
                "units": units_out,
            }
        )

    pids = [r["paragraph_id"] for r in records]
    candidates = sorted(cell_gold, key=lambda c: -sum(cell_gold[c].values()))
    complete = [c for c in candidates if "?" not in c]

    # ---- theory mapping (pre-registered)
    theory_pred = {
        pid: {
            label
            for label in (theory_label(*c) for c in cells_by_pid[pid])
            if label is not None
        }
        for pid in pids
    }
    unit_theory_hits = unit_theory_total = 0
    for cell, gold_counts in cell_gold.items():
        label = theory_label(*cell)
        if label is None:
            continue
        covered = {k: v for k, v in gold_counts.items() if k != "NONE"}
        unit_theory_total += sum(covered.values())
        unit_theory_hits += covered.get(label, 0)

    # ---- fitted greedy mapping (optimistic ceiling)
    fitted = fit_mapping(pids, cells_by_pid, gold_labels, complete)
    fitted_pred = apply_mapping(pids, cells_by_pid, fitted)

    # ---- 5-fold cross-fitted mapping (honest estimate)
    genre_of = {r["paragraph_id"]: r["type"] for r in records}
    ordered = sorted(pids, key=lambda pid: (genre_of[pid], pid))
    folds = [ordered[i::5] for i in range(5)]
    cross_pred: dict[int, set[str]] = {}
    for i, fold in enumerate(folds):
        fit_pids = [pid for j, other in enumerate(folds) if j != i for pid in other]
        mapping = fit_mapping(fit_pids, cells_by_pid, gold_labels, complete)
        cross_pred.update(apply_mapping(fold, cells_by_pid, mapping))

    def score(pred: dict[int, set[str]]) -> dict:
        official = task1_macro_f1(gold_labels, pred)
        return {
            "macro_f1": round(official["macro_f1"], 4),
            "per_label_f1": {
                label: round(v["f1"], 4) for label, v in official["per_label"].items()
            },
        }

    metrics = {
        "model": spec.key,
        "n_paragraphs": len(records),
        "n_units": n_units,
        "wall_seconds": round(wall, 1),
        "n_program_errors": len(errors),
        "n_parse_failures": n_parse_fail,
        "n_length_mismatch": n_mismatch,
        "unknown_rate_by_axis": {
            axis: round(unknown_by_axis[axis] / max(n_units, 1), 4) for axis in AXES
        },
        "errors_sample": errors[:5],
        "reference": {"champion_local_val": 0.7024, "champion_dev": 0.6159},
        "theory_mapping": score(theory_pred)
        | {
            "unit_accuracy_on_gold_units": round(
                unit_theory_hits / max(unit_theory_total, 1), 4
            )
        },
        "fitted_mapping_optimistic": score(fitted_pred)
        | {"cells": {label: [list(c) for c in cells] for label, cells in fitted.items()}},
        "crossfit_mapping": score(cross_pred),
        "cell_contingency": {
            "|".join(cell): dict(gold_counts.most_common())
            for cell, gold_counts in sorted(
                cell_gold.items(), key=lambda kv: -sum(kv[1].values())
            )
        },
        "axis_contingency": {
            axis: {
                value: dict(counts.most_common())
                for value, counts in sorted(table.items())
            }
            for axis, table in axis_gold.items()
        },
    }

    write_jsonl(exp_dir / "predictions" / "extractions.jsonl", out_rows)
    atomic_write_json(exp_dir / "metrics.json", metrics)
    summary = {
        k: metrics[k]
        for k in (
            "n_paragraphs",
            "n_units",
            "n_parse_failures",
            "n_length_mismatch",
            "unknown_rate_by_axis",
            "theory_mapping",
            "fitted_mapping_optimistic",
            "crossfit_mapping",
        )
    }
    summary["fitted_mapping_optimistic"] = {
        k: v for k, v in summary["fitted_mapping_optimistic"].items() if k != "cells"
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nfull metrics: {exp_dir / 'metrics.json'}")
    print(f"extractions:  {exp_dir / 'predictions' / 'extractions.jsonl'}")


if __name__ == "__main__":
    main()
