"""DSPy programs for both tasks (design axes D2, D3, D4).

Task 1: one multi-label ChainOfThought predictor (D2 option 1) —
`text, genre -> labels`, empty list legal.

Task 2, option 1: segment-then-classify (D3). Deterministic segmentation
(daleel.segment, connective granularity, oracle ceiling 0.920) produces
the offsets; the LLM only labels each segment jointly in one call; labeled
segments become spans, so offsets are correct by construction and spans
are non-overlapping by construction (the recall>1 metric quirk is
unreachable).

Task 2, option 2: quote-then-align (D3, `QuoteProgram`). The LLM emits
(label, verbatim quote) pairs and daleel.align recovers offsets by staged
string matching (oracle ceiling 0.999 — the rule segmenter's 0.920 ceiling
disappears, at the price of depending on the model's quoting fidelity).
One extraction serves BOTH tasks: the label set is Task 1, the aligned
quotes are Task 2 — which also removes the DALEEL-tailored segmenter from
the method's story.

Seed instructions are D4 seed (b): official definitions + the audit's
operational corrections, one canonical phrasing shared with the metric
feedback via daleel.policy. Genre is an explicit input field. These seeds
are the frozen baseline that optimizer-evolved instructions get diffed
against (a headline paper figure) — change them only deliberately.

Every program module accepts an optional `lm`: a module-OWNED dspy.LM set
on all of its predictors in __init__ (via Module.set_lm). Pass a fresh
make_lm(...) instance per module so no module inherits another module's
LM object (dspy.LM instances carry mutable state — history, kwargs,
rollout ids). `lm=None` preserves the legacy behavior exactly: predictors
fall back to the global dspy.configure LM.
"""

from typing import Literal

from . import runtime  # noqa: F401  — cache/env setup before dspy import

import dspy
import pydantic

from .align import align_adus
from .constants import LABELS
from .metrics import Span
from .policy import LABEL_POLICY, SPURIOUS_HINTS, TASK_CONTEXT
from .segment import segment_paragraph

LabelCode = Literal["CO", "AS", "TE", "ST", "AN", "OT"]
SegmentLabel = Literal["CO", "AS", "TE", "ST", "AN", "OT", "NONE"]

_POLICY_BLOCK = "\n".join(f"- {LABEL_POLICY[label]}" for label in LABELS)
_HINTS_BLOCK = "\n".join(f"- {SPURIOUS_HINTS[label]}" for label in LABELS)


class ClassifyParagraph(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

Identify every type of argumentative discourse unit (ADU) that occurs
anywhere in the paragraph. Several types usually co-occur; return an
empty list when the paragraph contains no ADU at all.

Label inventory:
{_POLICY_BLOCK}"""

    # NOTE the field name: dspy.Example/Prediction define a built-in METHOD
    # `.labels()` (primitives/example.py:273), so a field named `labels`
    # would be shadowed and attribute access would return the bound method.
    text: str = dspy.InputField(desc="the Arabic paragraph")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    adu_labels: list[LabelCode] = dspy.OutputField(
        desc="ADU types present anywhere in the paragraph; [] if none"
    )


class ClassifySegments(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

The paragraph has been pre-segmented into numbered clause-level segments,
given in order. Assign EXACTLY ONE label to each segment: the ADU type
that covers most of it, or NONE if the segment carries no ADU content at
all. Most segments do carry a label; use NONE sparingly (isolated
fragments, stray connectives). Procedural or greeting content is OT, not
NONE.

Label inventory:
{_POLICY_BLOCK}"""

    text: str = dspy.InputField(desc="the full Arabic paragraph, for context")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    segments: list[str] = dspy.InputField(desc="the paragraph's segments, in order")
    segment_labels: list[SegmentLabel] = dspy.OutputField(
        desc="exactly one label per segment, same order and count as `segments`"
    )


class ADUQuote(pydantic.BaseModel):
    """One argumentative discourse unit found in the paragraph."""

    # NOTE `label` is a str, not Literal: a Literal nested inside a pydantic
    # item is validated case-sensitively by TypeAdapter with none of the
    # adapter's top-level-Literal leniency (dspy adapters/utils.py
    # parse_value), so one lowercase label would fail the WHOLE paragraph
    # with AdapterParseError. QuoteProgram normalizes and filters instead.
    label: str = pydantic.Field(
        description="the ADU type: exactly one of CO, AS, TE, ST, AN, OT"
    )
    quote: str = pydantic.Field(
        description="the ADU text copied VERBATIM from the paragraph — "
        "character-for-character, no paraphrase, no ellipsis"
    )


class ExtractADUs(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

List every argumentative discourse unit (ADU) in the paragraph, in reading
order. An ADU is one contiguous stretch of text; report its type plus its
text copied VERBATIM from the paragraph — character-for-character,
including punctuation, with no paraphrase, no ellipsis, no words added or
dropped. ADUs must not overlap. The same type may occur several times;
together the ADUs usually cover most of the paragraph. Return an empty
list when the paragraph contains no ADU at all.

Label inventory:
{_POLICY_BLOCK}"""

    text: str = dspy.InputField(desc="the Arabic paragraph")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    adus: list[ADUQuote] = dspy.OutputField(
        desc="every ADU in reading order; [] when the paragraph has none"
    )


class RelabelSpans(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

The paragraph's argumentative discourse units (ADUs) have already been
extracted; they are listed in reading order, each with a preliminary
type. The segmentation is trustworthy — ONLY the types may be wrong.
Re-examine every segment against the definitions and return the corrected
type for each, in the same order. Keep the preliminary type unless
another definition clearly fits better; most segments are already
correct.

The usual mistakes to check for:
{_HINTS_BLOCK}

Label inventory:
{_POLICY_BLOCK}"""

    text: str = dspy.InputField(desc="the full Arabic paragraph, for context")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    segments: list[str] = dspy.InputField(
        desc="the extracted ADU segments, in reading order"
    )
    draft_labels: list[str] = dspy.InputField(
        desc="the preliminary type of each segment, same order"
    )
    corrected_labels: list[LabelCode] = dspy.OutputField(
        desc="exactly one corrected type per segment, same order and count "
        "as `segments`"
    )


class RelabelSpansBlind(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

The paragraph's argumentative discourse units (ADUs) have already been
extracted; they are listed in reading order. The segmentation is
trustworthy. Decide the type of each segment independently against the
definitions and return one type per segment, in the same order.

The usual mistakes to check for:
{_HINTS_BLOCK}

Label inventory:
{_POLICY_BLOCK}"""

    text: str = dspy.InputField(desc="the full Arabic paragraph, for context")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    segments: list[str] = dspy.InputField(
        desc="the extracted ADU segments, in reading order"
    )
    corrected_labels: list[LabelCode] = dspy.OutputField(
        desc="exactly one type per segment, same order and count as `segments`"
    )


class RelabelStage(dspy.Module):
    """Second-stage relabeler: boundaries stay fixed, only types re-decided.

    Motivated by the 2026-07-10 error anatomy of the val champion: 0% of
    gold span mass is unpredicted (extraction is solved) while 33% of AN
    gold mass carries an AS prediction and 75% of predicted CO mass sits
    on gold text of other labels — the oracle relabel-only ceiling is
    0.879 vs the 0.6805 baseline. `blind=True` withholds the extractor's
    draft labels (anchoring ablation). Containment mirrors Task2Program:
    a missing/invalid answer keeps the draft label, never crashes.
    """

    def __init__(self, cot: bool = True, blind: bool = False, lm: dspy.LM | None = None):
        super().__init__()
        self.blind = blind
        signature = RelabelSpansBlind if blind else RelabelSpans
        self.relabel = (dspy.ChainOfThought if cot else dspy.Predict)(signature)
        if lm is not None:
            self.set_lm(lm)

    def forward(self, text: str, genre: str, spans: list[Span]) -> dspy.Prediction:
        if not spans:
            return dspy.Prediction(spans=[], n_changed=0, length_mismatch=False)
        inputs = dict(
            text=text,
            genre=genre,
            segments=[text[s.start : s.end] for s in spans],
        )
        if not self.blind:
            inputs["draft_labels"] = [s.label for s in spans]
        result = self.relabel(**inputs)
        raw = result.corrected_labels if isinstance(result.corrected_labels, list) else []
        answers = [str(label).strip().upper() for label in raw]
        fixed = [
            answers[i] if i < len(answers) and answers[i] in LABELS else span.label
            for i, span in enumerate(spans)
        ]
        new_spans = [Span(s.start, s.end, label) for s, label in zip(spans, fixed)]
        return dspy.Prediction(
            spans=new_spans,
            n_changed=sum(s.label != n.label for s, n in zip(spans, new_spans)),
            length_mismatch=len(answers) != len(spans),
        )


class VerifyLabel(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

Decide whether one specific ADU type is genuinely present anywhere in the
paragraph. Apply the definition strictly; the caution names the common
mistake for this type. Presence requires at least one stretch of text
that the definition actually covers — a resemblance or a borderline
reading is not enough.

Label inventory, for contrast with the type under review:
{_POLICY_BLOCK}"""

    text: str = dspy.InputField(desc="the Arabic paragraph")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    label: str = dspy.InputField(desc="the ADU type code under review")
    definition: str = dspy.InputField(desc="the official definition of that type")
    caution: str = dspy.InputField(desc="the common mistake made with that type")
    present: bool = dspy.OutputField(
        desc="True only if the type is genuinely present somewhere in the paragraph"
    )


class JudgeStage(dspy.Module):
    """Second-stage label verification for Task 1 (v2 propose-then-verify).

    Motivated by the 2026-07-10 error anatomy of the val champion: recall
    is nearly saturated (ST 1.0, CO 0.9) while precision sinks the macro
    (CO 20 FP vs 9 TP, ST 7 FP, OT 31 FP) and AN is the lone recall sink
    (26 FN). Judges therefore run drop-only on fired precision-sink
    labels, and AN gets an authoritative both-ways verdict. The oracle
    FP-judge ceiling is 0.8607 vs the 0.7024 baseline. One generic
    predictor serves every label: the definition and caution arrive as
    inputs (daleel.policy single-source), so evolving its instructions
    stays one optimization problem.
    """

    def __init__(self, cot: bool = True,
                 drop_labels: tuple = ("CO", "ST", "OT"),
                 arbiter_labels: tuple = ("AN",),
                 lm: dspy.LM | None = None):
        super().__init__()
        self.drop_labels = drop_labels
        self.arbiter_labels = arbiter_labels
        self.verify = (dspy.ChainOfThought if cot else dspy.Predict)(VerifyLabel)
        if lm is not None:
            self.set_lm(lm)

    def _present(self, text: str, genre: str, label: str) -> bool:
        result = self.verify(
            text=text, genre=genre, label=label,
            definition=LABEL_POLICY[label], caution=SPURIOUS_HINTS[label],
        )
        return bool(result.present)

    def forward(self, text: str, genre: str, proposed: set[str]) -> dspy.Prediction:
        final, dropped, added = set(proposed), [], []
        for label in self.drop_labels:
            if label in final and not self._present(text, genre, label):
                final.discard(label)
                dropped.append(label)
        for label in self.arbiter_labels:
            verdict = self._present(text, genre, label)
            if verdict and label not in final:
                final.add(label)
                added.append(label)
            elif not verdict and label in final:
                final.discard(label)
                dropped.append(label)
        return dspy.Prediction(adu_labels=sorted(final), dropped=dropped, added=added)


class VerifyOne(dspy.Module):
    """Compile-time student for the judge: one decision per forward call.

    Deliberately holds its predictor under the SAME attribute name as
    JudgeStage (`verify`), so a state file compiled against VerifyOne
    loads directly into JudgeStage — the decision-level optimizer output
    drops into the paragraph-level pipeline unchanged.
    """

    def __init__(self, cot: bool = True, lm: dspy.LM | None = None):
        super().__init__()
        self.verify = (dspy.ChainOfThought if cot else dspy.Predict)(VerifyLabel)
        if lm is not None:
            self.set_lm(lm)

    def forward(self, text: str, genre: str, label: str,
                definition: str, caution: str) -> dspy.Prediction:
        return self.verify(text=text, genre=genre, label=label,
                           definition=definition, caution=caution)


class Task1Program(dspy.Module):
    """Single multi-label predictor; `cot=False` is the D2 Predict ablation."""

    def __init__(self, cot: bool = True, lm: dspy.LM | None = None):
        super().__init__()
        self.classify = (dspy.ChainOfThought if cot else dspy.Predict)(ClassifyParagraph)
        if lm is not None:
            self.set_lm(lm)

    def forward(self, text: str, genre: str) -> dspy.Prediction:
        return self.classify(text=text, genre=genre)


class Task2Program(dspy.Module):
    """Segment-then-classify: deterministic offsets, LLM labels only.

    The prediction carries `spans` (list[daleel.metrics.Span]) for the
    metric/submission layer, plus `segment_labels` and a `length_mismatch`
    flag (model returned the wrong number of labels; missing tail padded
    with NONE, excess truncated — degradation, never a crash).
    """

    def __init__(self, cot: bool = True, granularity: str = "connective",
                 lm: dspy.LM | None = None):
        super().__init__()
        self.granularity = granularity
        self.classify = (dspy.ChainOfThought if cot else dspy.Predict)(ClassifySegments)
        if lm is not None:
            self.set_lm(lm)

    def forward(self, text: str, genre: str) -> dspy.Prediction:
        segments = segment_paragraph(text, self.granularity)
        if not segments:
            return dspy.Prediction(spans=[], segment_labels=[], length_mismatch=False)
        result = self.classify(
            text=text, genre=genre, segments=[s.text for s in segments]
        )
        raw = result.segment_labels if isinstance(result.segment_labels, list) else []
        labels = [str(label).strip().upper() for label in raw]
        length_mismatch = len(labels) != len(segments)
        labels = (labels + ["NONE"] * len(segments))[: len(segments)]
        spans = [
            Span(seg.start, seg.end, label)
            for seg, label in zip(segments, labels)
            if label in LABELS
        ]
        return dspy.Prediction(
            spans=spans, segment_labels=labels, length_mismatch=length_mismatch
        )


class QuoteProgram(dspy.Module):
    """Quote-then-align: one extraction call serves both tasks.

    The prediction carries `spans` (aligned, non-overlapping — Task 2),
    `adu_labels` (distinct types the model extracted, pre-alignment —
    Task 1: label presence does not require recovered offsets), plus
    fidelity telemetry: `align_stats` (per-stage match counts,
    "unaligned", "dropped_overlap" — the quote-mode analogue of
    length_mismatch) and `unknown_labels`.
    """

    def __init__(self, cot: bool = True, lm: dspy.LM | None = None):
        super().__init__()
        self.extract = (dspy.ChainOfThought if cot else dspy.Predict)(ExtractADUs)
        if lm is not None:
            self.set_lm(lm)

    def forward(self, text: str, genre: str) -> dspy.Prediction:
        result = self.extract(text=text, genre=genre)
        raw = result.adus if isinstance(result.adus, list) else []

        def field(item, key: str) -> str:
            got = item.get(key, "") if isinstance(item, dict) else getattr(item, key, "")
            return str(got or "")

        items, unknown = [], []
        for item in raw:
            label = field(item, "label").strip().upper()
            quote = field(item, "quote")
            if label in LABELS and quote.strip():
                items.append((label, quote))
            elif label:
                unknown.append(label)
        spans, align_stats = align_adus(text, items)
        return dspy.Prediction(
            spans=spans,
            adu_labels=sorted({label for label, _ in items}),
            align_stats=align_stats,
            unknown_labels=unknown,
        )


def task1_examples(records: list[dict], gold: dict[int, set[str]]) -> list[dspy.Example]:
    """dspy.Example objects with D9-cleaned gold labels.

    The gold field is `adu_labels`, matching the signature output field so
    labeled-few-shot demo rendering finds it (`labels` would be shadowed by
    the Example.labels() method)."""
    return [
        dspy.Example(
            paragraph_id=r["paragraph_id"],
            text=r["text"],
            genre=r["type"],
            adu_labels=sorted(gold[r["paragraph_id"]]),
        ).with_inputs("text", "genre")
        for r in records
    ]


def task2_examples(records: list[dict], gold: dict[int, list[Span]]) -> list[dspy.Example]:
    """dspy.Example objects with D9-cleaned gold spans."""
    return [
        dspy.Example(
            paragraph_id=r["paragraph_id"],
            text=r["text"],
            genre=r["type"],
            spans=list(gold[r["paragraph_id"]]),
        ).with_inputs("text", "genre")
        for r in records
    ]
