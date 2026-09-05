"""Guideline-native two-stage program.

Mirrors the annotator workflow documented in the official annotation
guidelines (§3.3): (1) ``SegmentUnits`` brackets every argumentative unit
verbatim; (2) ``CategorizeUnits`` assigns one or two categories per unit.
Offsets are recovered by the same staged string matching as QuoteProgram
(the LLM never emits offsets), via a unit-index-preserving twin of
``align.align_adus`` so categories attach to spans AFTER alignment.

Task 2 = aligned spans carrying each unit's primary category. Task 1 = the
union of all unit categories — presence does not require recovered offsets
(QuoteProgram's rule). Multi-category units are official (guidelines §2,
§3.2 ex. 7 'ST/TE'); emitting the secondary label as a second same-offset
span is behind ``emit_dual_spans`` (default False) until the
overlapping-gold dual hypothesis is checked (design doc question 2).

All instruction text comes from daleel.guideline_policy (single-source
rule — see that module's docstring).
"""

import re
from typing import Literal

from . import runtime  # noqa: F401  — cache/env setup before dspy import

import dspy

from .align import _trim_to_text, find_quote
from .constants import LABELS
from .guideline_policy import (
    DECISION_PROCEDURE,
    GUIDELINE_CONTEXT,
    GUIDELINE_LABEL_POLICY,
    UNIT_RULES,
)
from .metrics import Span

_POLICY_BLOCK = "\n".join(f"- {GUIDELINE_LABEL_POLICY[label]}" for label in LABELS)

_LABEL_SPLIT = re.compile(r"[/,+&\s]+")


class SegmentUnits(dspy.Signature):
    __doc__ = f"""{GUIDELINE_CONTEXT}

{UNIT_RULES}"""

    text: str = dspy.InputField(desc="the Arabic paragraph")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    units: list[str] = dspy.OutputField(
        desc="every argumentative unit copied verbatim, in reading order; "
        "[] if the paragraph contains none"
    )


class CategorizeUnits(dspy.Signature):
    __doc__ = f"""{GUIDELINE_CONTEXT}

The paragraph has been divided into argumentative units, listed in reading
order. Assign each unit its category — usually one, more than one only
when several genuinely apply.

Categories:
{_POLICY_BLOCK}

{DECISION_PROCEDURE}"""

    text: str = dspy.InputField(desc="the full Arabic paragraph, for context")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    units: list[str] = dspy.InputField(
        desc="the paragraph's argumentative units, in reading order"
    )
    unit_labels: list[str] = dspy.OutputField(
        desc="one entry per unit, same order and count: a category code "
        "(CO, AS, TE, ST, AN, OT), or two codes joined by '/' when both "
        "genuinely apply (e.g. 'ST/TE')"
    )


def parse_unit_labels(raw: object) -> list[str]:
    """One model answer -> ordered valid label codes ('st/te' -> ['ST','TE']).

    Tolerates any of the separators models actually emit (/ , + & space),
    uppercases, drops unknown codes, dedupes preserving order. An answer
    with no valid code yields [] — the unit stays unlabeled (no span, no
    Task-1 contribution) rather than crashing the paragraph.
    """
    out: list[str] = []
    for part in _LABEL_SPLIT.split(str(raw).strip().upper()):
        if part in LABELS and part not in out:
            out.append(part)
    return out


def align_units(text: str, units: list[str]) -> tuple[list[tuple[int, Span]], dict]:
    """Unit-index-preserving twin of ``align.align_adus``.

    Same staged matching (find_quote), same reading-order cursor, same
    whitespace trim, same non-overlap policy (later spans trimmed, fully
    covered ones dropped) — but returns (unit_index, span) pairs so the
    caller can attach categories after alignment. align_adus itself cannot
    report which item produced which span once items are dropped or
    re-sorted, which is exactly what the two-stage design needs.
    """
    stats = {"exact": 0, "ws": 0, "norm": 0, "fuzzy": 0, "unaligned": 0, "dropped_overlap": 0}
    raw: list[tuple[int, Span]] = []
    cursor = 0
    for idx, unit in enumerate(units):
        found = find_quote(text, unit, cursor)
        if found is None:
            stats["unaligned"] += 1
            continue
        stats[found.stage] += 1
        start, end = _trim_to_text(text, found.span.start, found.span.end)
        if start >= end:
            stats["unaligned"] += 1
            continue
        raw.append((idx, Span(start, end, "")))
        cursor = end

    aligned: list[tuple[int, Span]] = []
    last_end = 0
    for idx, s in sorted(raw, key=lambda p: (p[1].start, p[1].end)):
        start = max(s.start, last_end)
        if start >= s.end:
            stats["dropped_overlap"] += 1
            continue
        if start != s.start:
            start, end = _trim_to_text(text, start, s.end)
            if start >= end:
                stats["dropped_overlap"] += 1
                continue
            s = Span(start, end, s.label)
        aligned.append((idx, s))
        last_end = s.end
    return aligned, stats


class GuidelineProgram(dspy.Module):
    """Segment-then-categorize, one LLM call per stage per paragraph.

    The prediction carries `spans` (Task 2), `adu_labels` (Task 1 union —
    includes labels of units whose quote failed to align), plus telemetry:
    `align_stats`, `length_mismatch` (categorizer returned the wrong number
    of answers; missing tail left unlabeled, excess truncated — degradation,
    never a crash), `n_units`, `n_multi` (units given 2+ categories) and
    `n_unlabeled` (units whose answer parsed to no valid code).
    """

    def __init__(self, cot: bool = True, emit_dual_spans: bool = False,
                 lm: dspy.LM | None = None):
        super().__init__()
        self.emit_dual_spans = emit_dual_spans
        predictor = dspy.ChainOfThought if cot else dspy.Predict
        self.segment = predictor(SegmentUnits)
        self.categorize = predictor(CategorizeUnits)
        # Module-owned LM (fresh instance per module, never implicitly
        # shared across modules); None falls back to the global configure.
        if lm is not None:
            self.set_lm(lm)

    def forward(self, text: str, genre: str) -> dspy.Prediction:
        seg = self.segment(text=text, genre=genre)
        raw_units = seg.units if isinstance(seg.units, list) else []
        units = [str(u).strip() for u in raw_units if str(u).strip()]
        if not units:
            return dspy.Prediction(
                spans=[], adu_labels=[], align_stats=None,
                length_mismatch=False, n_units=0, n_multi=0, n_unlabeled=0,
            )

        cat = self.categorize(text=text, genre=genre, units=units)
        raw_labels = cat.unit_labels if isinstance(cat.unit_labels, list) else []
        parsed = [parse_unit_labels(answer) for answer in raw_labels]
        length_mismatch = len(parsed) != len(units)
        parsed = (parsed + [[]] * len(units))[: len(units)]

        # Alignment runs over ALL units (labeled or not): segmentation owns
        # the geometry, so an unlabeled unit still advances the cursor and
        # blocks its neighbours from absorbing its text.
        aligned, align_stats = align_units(text, units)
        spans: list[Span] = []
        for idx, span in aligned:
            labels = parsed[idx]
            if not labels:
                continue
            spans.append(Span(span.start, span.end, labels[0]))
            if self.emit_dual_spans:
                spans.extend(Span(span.start, span.end, extra) for extra in labels[1:])

        return dspy.Prediction(
            spans=spans,
            adu_labels=sorted({label for labels in parsed for label in labels}),
            align_stats=align_stats,
            length_mismatch=length_mismatch,
            n_units=len(units),
            n_multi=sum(len(labels) >= 2 for labels in parsed),
            n_unlabeled=sum(not labels for labels in parsed),
        )
