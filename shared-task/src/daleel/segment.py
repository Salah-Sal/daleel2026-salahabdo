"""Deterministic Arabic segmenter for Task 2 (D3: segment-then-classify).

LLMs cannot emit reliable character offsets, so offsets never come from the
model: this module cuts each paragraph into clause/sentence segments whose
offsets are correct by construction, a DSPy program labels each segment
(or NONE), and labeled segments become the predicted spans.

The official pair-sum metric makes this near-optimal: over-segmentation is
roughly neutral and label accuracy over covered text is what scores
(metrics notes). `oracle_spans`/`oracle_ceiling` measure the
ceiling this segmenter imposes — every segment labeled perfectly from the
gold — which must be measured BEFORE any prompt work (D3) and is a paper
table row.

Debate transcripts contain speaker markers like
`*المتحدث الأول موالاة: (ذكر)/*` whose inner `:` would otherwise trigger a
split; they are protected and become segments of their own.
"""

import re
from dataclasses import dataclass

from .metrics import Span, span_partial_f1

# Matches *…* including the observed *…/* variant (the '/' is an ordinary
# inner character); never spans a newline.
SPEAKER_MARKER = re.compile(r"\*[^*\n]{1,80}\*")

SENTENCE_BOUNDARIES = frozenset(".؟!؛:…\n")
CLAUSE_BOUNDARIES = SENTENCE_BOUNDARIES | {"،"}

# Debate speech chains clauses with bare connectives instead of punctuation;
# cutting before a standalone connective lifts the oracle ceiling from 0.858
# to 0.920 (measured on cleaned training gold, 2026-07-09). Inventory from
# the theory docs' الروابط families.
CONNECTIVE_CUT = re.compile(r"(?<=\s)(?:و|أو|لكن|لأن|كما|حيث|إذن|بل|ثم|فإن)(?=\s)")

# A segment must contain at least one word character (letter/digit); this
# drops punctuation-only fragments produced by consecutive boundary chars.
_HAS_WORD = re.compile(r"\w")


@dataclass(frozen=True)
class Segment:
    """A candidate span; offsets index the ORIGINAL text, end exclusive."""

    start: int
    end: int
    text: str


def segment_paragraph(text: str, granularity: str = "connective") -> list[Segment]:
    """Cut a paragraph at Arabic punctuation / newlines / speaker markers.

    granularity — measured oracle ceilings on cleaned training gold:
    "sentence" (0.764) splits on sentence punctuation only; "clause"
    (0.858) adds the Arabic comma `،`; "connective" (0.920, default) adds
    cuts before standalone connectives. Boundary characters stay attached
    to the segment they close; segment edges are trimmed of whitespace
    (offset-adjusted, never by rewriting text — 271 gold spans forbid text
    normalization).
    """
    if granularity in ("clause", "connective"):
        boundaries = CLAUSE_BOUNDARIES
    elif granularity == "sentence":
        boundaries = SENTENCE_BOUNDARIES
    else:
        raise ValueError(f"unknown granularity: {granularity!r}")

    protected = [m.span() for m in SPEAKER_MARKER.finditer(text)]

    def unprotected(i: int) -> bool:
        return not any(s <= i < e for s, e in protected)

    # Cut positions: after each boundary char outside protected regions,
    # at both edges of every protected region, and (connective granularity)
    # before each standalone connective.
    cuts = {0, len(text)}
    for i, ch in enumerate(text):
        if ch in boundaries and unprotected(i):
            cuts.add(i + 1)
    for s, e in protected:
        cuts.update((s, e))
    if granularity == "connective":
        for m in CONNECTIVE_CUT.finditer(text):
            if unprotected(m.start()):
                cuts.add(m.start())

    positions = sorted(cuts)
    segments = []
    for start, end in zip(positions, positions[1:]):
        piece = text[start:end]
        lstrip = len(piece) - len(piece.lstrip())
        rstrip = len(piece) - len(piece.rstrip())
        start, end = start + lstrip, end - rstrip
        if start >= end or not _HAS_WORD.search(text[start:end]):
            continue
        segments.append(Segment(start, end, text[start:end]))
    return segments


def oracle_label(segment: Segment, gold_spans: list[Span]) -> str | None:
    """The best single label for a segment: argmax char overlap with the
    gold spans, ties broken toward the earlier gold span; None if the
    segment overlaps no gold span."""
    overlap_by_label: dict[str, int] = {}
    for gs in gold_spans:
        o = max(0, min(segment.end, gs.end) - max(segment.start, gs.start))
        if o:
            overlap_by_label[gs.label] = overlap_by_label.get(gs.label, 0) + o
    if not overlap_by_label:
        return None
    return max(overlap_by_label, key=lambda l: overlap_by_label[l])


def oracle_spans(text: str, gold_spans: list[Span], granularity: str = "connective") -> list[Span]:
    """The spans a perfect classifier would produce over this segmentation."""
    out = []
    for seg in segment_paragraph(text, granularity):
        label = oracle_label(seg, gold_spans)
        if label is not None:
            out.append(Span(seg.start, seg.end, label))
    return out


def oracle_ceiling(
    texts: dict[int, str], gold: dict[int, list[Span]], granularity: str = "connective"
) -> dict:
    """Official Task 2 score of the oracle-labeled segmentation — the
    ceiling segment-then-classify can reach with this segmenter."""
    pred = {
        pid: oracle_spans(texts[pid], gold[pid], granularity) for pid in gold
    }
    result = span_partial_f1(gold, pred)
    result["n_segments"] = sum(
        len(segment_paragraph(texts[pid], granularity)) for pid in gold
    )
    return result
