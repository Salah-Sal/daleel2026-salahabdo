"""Deterministic candidate atomization for the decomposed Daleel program.

The stage-0 quote extractor has excellent text coverage but often emits one
broad span across several differently annotated units.  This module keeps the
extractor as a high-recall proposal source and intersects its boundaries with
the existing Arabic clause/connective segmenter.  The resulting *atoms* are
small, exact substrings that can be classified one at a time by DSPy.

All offsets index the untouched Python string and use half-open intervals.
Cross-label overlap is preserved when role decisions are assembled; exact
same-label duplicates are removed because deliberately duplicating predictions
would exploit the official all-pairs scorer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .constants import LABELS
from .metrics import Span
from .segment import segment_paragraph

_HAS_WORD = re.compile(r"\w")
_LABEL_ORDER = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class CandidateAtom:
    """One exact candidate substring sent to the role classifier.

    ``draft_roles`` records every proposal label covering the atom.  It is
    telemetry and a parse-failure fallback, not a default LM input: the first
    relabeling experiment showed that exposing the draft can anchor Gemma.
    """

    start: int
    end: int
    text: str
    draft_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty or negative atom: [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start


def overlap_length(start: int, end: int, span: Span) -> int:
    """Character intersection between ``[start, end)`` and ``span``."""

    return max(0, min(end, span.end) - max(start, span.start))


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    piece = text[start:end]
    start += len(piece) - len(piece.lstrip())
    end -= len(piece) - len(piece.rstrip())
    return start, end


def _ordered_roles(roles: Iterable[str]) -> tuple[str, ...]:
    unique = set(roles)
    return tuple(sorted(unique, key=lambda x: (_LABEL_ORDER.get(x, len(LABELS)), x)))


def atomize_spans(
    text: str,
    proposals: Sequence[Span],
    granularity: str = "connective",
) -> list[CandidateAtom]:
    """Intersect proposal coverage with deterministic clause boundaries.

    Proposal boundaries, rule-segment boundaries, and overlap boundaries all
    become cut points.  Intervals outside every proposal are discarded.  The
    function therefore cannot invent text coverage, while broad proposals can
    be split into independently typable units.  Punctuation-only fragments and
    whitespace are discarded without rewriting the source string.

    Unlike :func:`daleel.align.align_adus`, this representation does not assume
    that one label owns every character.  If proposals overlap, the atom keeps
    all covering draft roles and downstream assembly may emit the same offsets
    under different official labels.
    """

    if not proposals or not text:
        return []

    checked: list[Span] = []
    for proposal in proposals:
        if proposal.start < 0 or proposal.end > len(text):
            raise ValueError(
                f"proposal [{proposal.start}, {proposal.end}) outside text length {len(text)}"
            )
        checked.append(proposal)

    rule_segments = segment_paragraph(text, granularity)
    cuts = {point for p in checked for point in (p.start, p.end)}
    for seg in rule_segments:
        # Adding every rule edge is harmless; intervals outside proposal
        # coverage are removed below.  It also handles overlapping proposals
        # without a quadratic edge-filtering pass.
        cuts.update((seg.start, seg.end))

    atoms: list[CandidateAtom] = []
    positions = sorted(cuts)
    for raw_start, raw_end in zip(positions, positions[1:]):
        if raw_start >= raw_end:
            continue
        covering = [
            p for p in checked if overlap_length(raw_start, raw_end, p) > 0
        ]
        if not covering:
            continue
        start, end = _trim(text, raw_start, raw_end)
        if start >= end or not _HAS_WORD.search(text[start:end]):
            continue
        atoms.append(
            CandidateAtom(
                start=start,
                end=end,
                text=text[start:end],
                draft_roles=_ordered_roles(
                    p.label for p in covering if p.label in LABELS
                ),
            )
        )

    # Whitespace trimming can make two different raw intervals collapse to the
    # same atom.  Combine their drafts deterministically.
    combined: dict[tuple[int, int], CandidateAtom] = {}
    for atom in atoms:
        key = (atom.start, atom.end)
        prior = combined.get(key)
        roles = atom.draft_roles if prior is None else _ordered_roles(
            prior.draft_roles + atom.draft_roles
        )
        combined[key] = CandidateAtom(atom.start, atom.end, atom.text, roles)
    return [combined[key] for key in sorted(combined)]


def gold_roles_for_atom(
    atom: CandidateAtom,
    gold_spans: Sequence[Span],
    containment_threshold: float = 0.80,
) -> tuple[str, ...]:
    """Derive the supervised role set for one proposal-shaped atom.

    Most atoms map to exactly one label.  Ordinary candidates qualify a role
    only when gold covers ``containment_threshold`` of the *atom*.  Using the
    smaller interval here would incorrectly turn one broad atom containing two
    consecutive gold units into a multi-label target.  Multiple roles are kept
    only when each role covers most of the atom (for example a genuine
    same-offset TE+ST annotation).  A short nested statistic is not projected
    over an entire broad testimony atom; it needs its own candidate boundary.
    If nothing qualifies but the atom overlaps gold, the maximum-overlap label
    is kept.  No overlap is internal NONE.
    """

    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError("containment_threshold must lie in (0, 1]")

    overlap_by_label: dict[str, int] = {}
    qualifying: set[str] = set()
    for gold in gold_spans:
        intersection = overlap_length(atom.start, atom.end, gold)
        if not intersection:
            continue
        overlap_by_label[gold.label] = overlap_by_label.get(gold.label, 0) + intersection
        if intersection / len(atom) >= containment_threshold:
            qualifying.add(gold.label)

    if not overlap_by_label:
        return ()
    if not qualifying:
        qualifying.add(
            max(
                overlap_by_label,
                key=lambda label: (overlap_by_label[label], -_LABEL_ORDER.get(label, 999)),
            )
        )
    return _ordered_roles(qualifying)


def normalize_roles(raw_roles: Iterable[object]) -> tuple[str, ...]:
    """Normalize, filter, deduplicate, and canonically order role tokens."""

    return _ordered_roles(
        str(role).strip().upper()
        for role in raw_roles
        if str(role).strip().upper() in LABELS
    )


def assemble_role_spans(
    atoms: Sequence[CandidateAtom],
    decisions: Sequence[tuple[Iterable[object], bool]],
    *,
    fallback_to_draft: bool = True,
) -> tuple[list[Span], dict[str, int]]:
    """Turn per-atom decisions into scorer-ready spans.

    Each decision is ``(roles, parsed_ok)``.  A parsed empty role list is the
    legitimate internal NONE decision and drops the atom.  Only a malformed
    decision falls back to extractor drafts.  Cross-label identical or nested
    spans survive; exact same-label duplicates do not.
    """

    if len(atoms) != len(decisions):
        raise ValueError(
            f"atom/decision length mismatch: {len(atoms)} != {len(decisions)}"
        )

    stats = {
        "atoms": len(atoms),
        "parse_failures": 0,
        "fallback_atoms": 0,
        "none_atoms": 0,
        "duplicate_same_label": 0,
    }
    raw: list[Span] = []
    for atom, (roles_raw, parsed_ok) in zip(atoms, decisions):
        roles = normalize_roles(roles_raw)
        if not parsed_ok:
            stats["parse_failures"] += 1
            if fallback_to_draft:
                roles = normalize_roles(atom.draft_roles)
                stats["fallback_atoms"] += 1
        if not roles:
            stats["none_atoms"] += 1
            continue
        raw.extend(Span(atom.start, atom.end, role) for role in roles)

    seen: set[tuple[int, int, str]] = set()
    spans: list[Span] = []
    for span in sorted(
        raw,
        key=lambda s: (s.start, s.end, _LABEL_ORDER.get(s.label, len(LABELS))),
    ):
        key = (span.start, span.end, span.label)
        if key in seen:
            stats["duplicate_same_label"] += 1
            continue
        seen.add(key)
        spans.append(span)
    return spans, stats


def marked_context(
    text: str,
    start: int,
    end: int,
    context_chars: int = 500,
) -> str:
    """Return a bounded, immutable-text view with the target visibly marked."""

    if not 0 <= start < end <= len(text):
        raise ValueError(f"invalid target [{start}, {end}) for text length {len(text)}")
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    prefix = "…" if left else ""
    suffix = "…" if right < len(text) else ""
    return (
        prefix
        + text[left:start]
        + "<TARGET>"
        + text[start:end]
        + "</TARGET>"
        + text[end:right]
        + suffix
    )
