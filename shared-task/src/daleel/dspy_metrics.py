"""DSPy optimization metrics for both tasks (design item D8).

Design decided 2026-07-09 from the contract audit + alignment study:

- One GRADED metric per task serves every consumer (Evaluate, BFRS,
  MIPROv2, SIMBA, GEPA, bootstrap). Strict demo gating comes from
  `metric_threshold=1.0` at BootstrapFewShot* call sites — NEVER from the
  `trace is not None -> bool` idiom, which silently corrupts GEPA's graded
  scoring (its scoring path also captures traces).
- Task 1 proxy: unweighted per-example set-F1 (best measured alignment
  with the official corpus macro-F1; rarity weighting mis-ranks).
- Task 2 proxy: the official pair-sum formula on a single-paragraph dict
  (rho = 0.990 vs the corpus score).
- Empty gold + empty prediction scores 1.0 — 60/612 paragraphs are
  legitimately empty and predict-nothing must stay reachable.
- A parse failure scores 0.0 and gets format feedback; it must never look
  like a legitimate empty prediction.
- GEPA reflection path (pred_name is not None) returns
  dspy.Prediction(score=..., feedback=...) with template feedback: label
  diff, canonical policy lines, and — unique asset — verbatim quotes of
  the gold Task 2 span(s) for each missed label (Task 1 gold equals the
  Task 2 span-label set, so the evidence text is always available).

Metrics consume the D9-CLEANED gold (paragraph 964's phantom ST/CO would
otherwise feed GEPA false feedback on the two rarest labels).
"""

import re

from . import runtime  # noqa: F401  — cache/env setup before dspy import

import dspy

from .constants import LABELS
from .metrics import span_partial_f1
from .policy import LABEL_POLICY, SPURIOUS_HINTS

MAX_QUOTE_CHARS = 160

# Tokens a model may reasonably emit to mean "no labels" — not parse errors.
_EMPTY_TOKENS = {"", "NONE", "EMPTY", "NO LABELS", "لا شيء"}


def parse_label_set(pred) -> tuple[set[str], bool, list[str]]:
    """Extract a label set from a prediction: (labels, parsed_ok, unknown).

    parsed_ok is False only for hard failures (no usable `adu_labels`
    field — e.g. an adapter FailedPrediction). Unknown tokens are dropped
    but the output still counts as parsed; they come back for feedback.
    (The field is `adu_labels`, not `labels`: dspy.Prediction has a
    built-in .labels() METHOD that shadows any field of that name.)
    """
    raw = getattr(pred, "adu_labels", None)
    if isinstance(raw, str):
        raw = [t for t in re.split(r"[,\s؛،]+", raw) if t]
    if not isinstance(raw, (list, tuple, set)):
        return set(), False, []
    valid, unknown = set(), []
    for token in raw:
        t = str(token).strip().upper()
        if t in LABELS:
            valid.add(t)
        elif t not in _EMPTY_TOKENS:
            unknown.append(str(token))
    return valid, True, unknown


def _set_f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    return 2 * len(gold & pred) / ((len(gold) + len(pred)) or 1)


_FORMAT_FEEDBACK = (
    "The output could not be parsed. Return `adu_labels` as a list of "
    "label codes drawn only from CO, AS, TE, ST, AN, OT; an empty list is "
    "a valid answer when the paragraph contains no ADU."
)


class Task1Metric:
    """Graded set-F1 everywhere; 5-arg signature satisfies GEPA.

    `gold_spans` ({paragraph_id: [Span]}, D9-cleaned) and `texts`
    ({paragraph_id: str}) enable span-grounded feedback quotes; without
    them feedback degrades gracefully to the label diff + policy lines.
    """

    def __init__(self, gold_spans=None, texts=None):
        self.gold_spans = gold_spans or {}
        self.texts = texts or {}

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        g = set(gold.adu_labels)
        p, parsed, unknown = parse_label_set(pred)
        score = 0.0 if not parsed else _set_f1(g, p)
        if pred_name is None:
            return score
        return dspy.Prediction(
            score=score, feedback=self._feedback(gold, g, p, parsed, unknown)
        )

    def _gold_quote(self, paragraph_id, label) -> str | None:
        spans = [s for s in self.gold_spans.get(paragraph_id, []) if s.label == label]
        text = self.texts.get(paragraph_id)
        if not spans or text is None:
            return None
        quote = text[spans[0].start : spans[0].end]
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS] + "…"
        more = f" (+{len(spans) - 1} more {label} span(s))" if len(spans) > 1 else ""
        return f"«{quote}»{more}"

    def _feedback(self, gold, g, p, parsed, unknown) -> str:
        if not parsed:
            return _FORMAT_FEEDBACK
        parts = []
        if unknown:
            parts.append(
                f"Unknown label token(s) {unknown!r} were ignored — emit only "
                "CO, AS, TE, ST, AN, OT."
            )
        pid = getattr(gold, "paragraph_id", None)
        for label in [l for l in LABELS if l in g - p]:
            line = f"Missed {label}. {LABEL_POLICY[label]}"
            quote = self._gold_quote(pid, label)
            if quote:
                line += f" The gold {label} evidence in this paragraph: {quote}"
            parts.append(line)
        for label in [l for l in LABELS if l in p - g]:
            parts.append(f"Spurious {label}: {SPURIOUS_HINTS[label]}")
        if not parts:
            return (
                "Correct: the paragraph contains no ADU labels."
                if not g
                else f"Correct label set: {sorted(g)}."
            )
        return " ".join(parts)


class JudgeMetric:
    """Binary correctness of one label-presence decision; 5-arg for GEPA.

    Examples carry (paragraph_id, label, present); predictions carry
    `present`. Unlike the corpus macro-F1, this metric decomposes exactly
    per example — the property whose absence broke optimizer-internal
    selection in the D7 campaign. Feedback quotes the gold span as
    evidence on missed positives (the corpus-conventions signal that
    zero-shot definitions failed to convey, 2026-07-10 stage tests).
    """

    def __init__(self, gold_spans=None, texts=None):
        self.gold_spans = gold_spans or {}
        self.texts = texts or {}

    @staticmethod
    def _parse(pred) -> tuple[bool | None, bool]:
        raw = getattr(pred, "present", None)
        if isinstance(raw, bool):
            return raw, True
        if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
            return raw.strip().lower() == "true", True
        return None, False

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        want = bool(gold.present)
        got, parsed = self._parse(pred)
        score = 1.0 if parsed and got == want else 0.0
        if pred_name is None:
            return score
        return dspy.Prediction(
            score=score, feedback=self._feedback(gold, want, got, parsed)
        )

    def _gold_quote(self, paragraph_id, label) -> str | None:
        spans = [s for s in self.gold_spans.get(paragraph_id, []) if s.label == label]
        text = self.texts.get(paragraph_id)
        if not spans or text is None:
            return None
        quote = text[spans[0].start : spans[0].end]
        if len(quote) > MAX_QUOTE_CHARS:
            quote = quote[:MAX_QUOTE_CHARS] + "…"
        more = f" (+{len(spans) - 1} more)" if len(spans) > 1 else ""
        return f"«{quote}»{more}"

    def _feedback(self, gold, want, got, parsed) -> str:
        label = gold.label
        if not parsed:
            return "The answer must be a boolean `present`: True or False."
        if got == want:
            return f"Correct: {label} is {'present' if want else 'not present'}."
        if want:
            line = (
                f"Wrong: {label} IS present in this paragraph. "
                f"{LABEL_POLICY[label]}"
            )
            quote = self._gold_quote(getattr(gold, "paragraph_id", None), label)
            if quote:
                line += f" The gold {label} evidence: {quote}"
            return line
        return (
            f"Wrong: {label} is NOT present in this paragraph. "
            f"{SPURIOUS_HINTS[label]}"
        )


class Task2Metric:
    """Per-paragraph pair-sum F1 (official formula on a 1-paragraph dict).

    Examples carry `spans` (list of daleel.metrics.Span, D9-cleaned gold);
    predictions carry `spans` from the segment-then-classify program.
    Note for stage 1+: `metric_threshold=1.0` is usually unreachable here
    (segment edges are whitespace-trimmed, gold spans are not) — bootstrap
    call sites should gate at ~0.9 instead.
    """

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        g = list(gold.spans)
        p_raw = getattr(pred, "spans", None)
        parsed = isinstance(p_raw, (list, tuple))
        p = list(p_raw) if parsed else []
        if not parsed:
            score = 0.0
        elif not g and not p:
            score = 1.0
        else:
            score = span_partial_f1({0: g}, {0: p})["f1"]
        if pred_name is None:
            return score
        return dspy.Prediction(
            score=score, feedback=self._feedback(gold, g, p, parsed, pred)
        )

    def _feedback(self, gold, g, p, parsed, pred) -> str:
        if not parsed:
            return (
                "The output could not be parsed into segment labels. Return "
                "one label per segment, in order, drawn from CO, AS, TE, ST, "
                "AN, OT, NONE."
            )
        parts = []
        if getattr(pred, "length_mismatch", False):
            parts.append(
                "The number of labels returned did not match the number of "
                "segments — return exactly one label per numbered segment, "
                "in the same order."
            )
        g_labels = {s.label for s in g}
        p_labels = {s.label for s in p}
        text = getattr(gold, "text", None)
        for label in [l for l in LABELS if l in g_labels - p_labels]:
            line = f"No segment was labeled {label}. {LABEL_POLICY[label]}"
            if text is not None:
                first = next(s for s in g if s.label == label)
                quote = text[first.start : first.end]
                if len(quote) > MAX_QUOTE_CHARS:
                    quote = quote[:MAX_QUOTE_CHARS] + "…"
                line += f" The gold {label} text here: «{quote}»"
            parts.append(line)
        for label in [l for l in LABELS if l in p_labels - g_labels]:
            parts.append(f"Spurious {label} segment(s): {SPURIOUS_HINTS[label]}")
        if not parts:
            return (
                "Correct: no ADU spans in this paragraph."
                if not g
                else "Label set correct; remaining loss is boundary overlap."
            )
        return " ".join(parts)
