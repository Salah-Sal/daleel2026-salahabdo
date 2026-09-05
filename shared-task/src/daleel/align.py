"""Quote-to-offset alignment for quote-then-classify span prediction (D3).

Alternative to the deterministic segmenter: the LLM emits (label, quote)
pairs where each quote should be a verbatim substring of the paragraph,
and this module recovers character offsets by string matching. Measured
oracle ceiling on cleaned training gold (perfect quoter + greedy exact
matching): 0.999 vs the connective segmenter's 0.920 — the ceiling moves
into the model's quoting fidelity, which the pair-sum partial-overlap
metric is tolerant of.

Matching is staged, strictest first, and reports which stage fired so
runs can track fidelity drift:

1. exact      — str.find, reading-order cursor with global fallback
2. ws         — whitespace-flexible (models re-wrap lines/spaces)
3. norm       — Arabic-normalized (diacritics, tatweel, alef/ya/ta-marbuta
                variants) via an index map back to original offsets
4. fuzzy      — difflib over the normalized texts, accepted only above a
                match-ratio floor (paraphrase drift, small word drops)

Predicted spans are de-overlapped (later spans trimmed, never the metric
quirk) — gold does contain 7 overlapping-span paragraphs, but we commit
to non-overlapping predictions (metrics.py policy).
"""

import difflib
import re
from dataclasses import dataclass

from .metrics import Span

# Decorations a model may add around a quote; stripped before matching.
_QUOTE_DECORATIONS = "«»\"'“”‘’`[]() \t\n…"

_DIACRITICS = frozenset("ًٌٍَُِّْٰ")
_TATWEEL = "ـ"
_ALEF_VARIANTS = frozenset("أإآٱ")

FUZZY_MIN_RATIO = 0.70  # fraction of normalized quote chars that must match


@dataclass(frozen=True)
class AlignedQuote:
    """A recovered span plus the matching stage that produced it."""

    span: Span
    stage: str  # "exact" | "ws" | "norm" | "fuzzy"


def _norm_char(ch: str) -> str:
    if ch in _DIACRITICS or ch == _TATWEEL:
        return ""
    if ch in _ALEF_VARIANTS:
        return "ا"
    if ch == "ى":
        return "ي"
    if ch == "ة":
        return "ه"
    if ch.isspace():
        return " "
    return ch


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalized text + per-char index into the ORIGINAL text.

    Whitespace runs collapse to one space so the map stays monotonic and
    normalized positions translate back to original offsets exactly.
    """
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        c = _norm_char(ch)
        if not c:
            continue
        if c == " " and chars and chars[-1] == " ":
            continue
        chars.append(c)
        index.append(i)
    return "".join(chars), index


def _find_ws_flexible(text: str, quote: str, cursor: int) -> tuple[int, int] | None:
    tokens = quote.split()
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(t) for t in tokens))
    m = pattern.search(text, cursor) or pattern.search(text)
    return (m.start(), m.end()) if m else None


def _find_normalized(text: str, quote: str, cursor: int) -> tuple[int, int] | None:
    norm_text, index = _normalize_with_map(text)
    norm_quote, _ = _normalize_with_map(quote)
    norm_quote = norm_quote.strip()
    if not norm_quote or not norm_text:
        return None
    # translate the original-text cursor into normalized coordinates
    norm_cursor = next((k for k, i in enumerate(index) if i >= cursor), len(index))
    pos = norm_text.find(norm_quote, norm_cursor)
    if pos == -1:
        pos = norm_text.find(norm_quote)
    if pos == -1:
        return None
    return index[pos], index[pos + len(norm_quote) - 1] + 1


def _find_fuzzy(text: str, quote: str) -> tuple[int, int] | None:
    norm_text, index = _normalize_with_map(text)
    norm_quote, _ = _normalize_with_map(quote)
    norm_quote = norm_quote.strip()
    if not norm_quote or not norm_text:
        return None
    matcher = difflib.SequenceMatcher(None, norm_quote, norm_text, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    if matched / len(norm_quote) < FUZZY_MIN_RATIO:
        return None
    start = index[blocks[0].b]
    end = index[blocks[-1].b + blocks[-1].size - 1] + 1
    if end <= start:
        return None
    return start, end


def find_quote(text: str, quote: str, cursor: int = 0) -> AlignedQuote | None:
    """Locate one quote in the paragraph; offsets index the original text.

    `cursor` biases the search to at-or-after the previous match (models
    emit ADUs in reading order — this resolves the 0.7% of gold quotes
    that occur more than once in their paragraph); every stage falls back
    to a global search before the next, looser stage is tried.
    """
    q = quote.strip(_QUOTE_DECORATIONS)
    if not q:
        return None

    pos = text.find(q, cursor)
    if pos == -1:
        pos = text.find(q)
    if pos != -1:
        return AlignedQuote(Span(pos, pos + len(q), ""), "exact")

    for stage, finder in (("ws", _find_ws_flexible), ("norm", _find_normalized)):
        found = finder(text, q, cursor)
        if found:
            return AlignedQuote(Span(found[0], found[1], ""), stage)

    found = _find_fuzzy(text, q)
    if found:
        return AlignedQuote(Span(found[0], found[1], ""), "fuzzy")
    return None


def _trim_to_text(text: str, start: int, end: int) -> tuple[int, int]:
    piece = text[start:end]
    start += len(piece) - len(piece.lstrip())
    end -= len(piece) - len(piece.rstrip())
    return start, end


def align_adus(text: str, items: list[tuple[str, str]]) -> tuple[list[Span], dict]:
    """Align (label, quote) pairs to non-overlapping spans.

    Returns (spans, stats) where stats counts matches per stage plus
    "unaligned" and "dropped_overlap" — the run-level fidelity telemetry
    (the quote-mode analogue of the segmenter's length_mismatch).
    """
    stats = {"exact": 0, "ws": 0, "norm": 0, "fuzzy": 0, "unaligned": 0, "dropped_overlap": 0}
    raw: list[Span] = []
    cursor = 0
    for label, quote in items:
        found = find_quote(text, quote, cursor)
        if found is None:
            stats["unaligned"] += 1
            continue
        stats[found.stage] += 1
        start, end = _trim_to_text(text, found.span.start, found.span.end)
        if start >= end:
            stats["unaligned"] += 1
            continue
        raw.append(Span(start, end, label))
        cursor = end

    spans: list[Span] = []
    last_end = 0
    for s in sorted(raw, key=lambda s: (s.start, s.end)):
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
        spans.append(s)
        last_end = s.end
    return spans, stats
