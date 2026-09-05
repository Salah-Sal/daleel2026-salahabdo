"""DSPy v3: proposal atomization plus independently optimized role decisions.

The optimizer-facing unit in this module is **one candidate atom**, never one
paragraph containing a list of decisions.  That distinction is essential:
DSPy's bootstrap and GEPA trace logic retain only one invocation when the same
predictor is called repeatedly inside one program example.  Standalone atom
examples give every rare label, hard negative, and confusion its own score and
feedback.

At deployment the frozen quote extractor supplies high-recall spans,
``daleel.candidates.atomize_spans`` splits broad proposals at deterministic
clause boundaries, and the compiled ``SpanRoleDecision`` assigns zero, one, or
(rarely) several official roles.  An empty role list is an internal NONE
decision; OT remains the real shared-task discourse-management label.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from . import runtime  # noqa: F401 -- cache/env setup before dspy import

import dspy
from sklearn.feature_extraction.text import TfidfVectorizer

from .candidates import (
    CandidateAtom,
    assemble_role_spans,
    atomize_spans,
    gold_roles_for_atom,
    marked_context,
    normalize_roles,
)
from .constants import LABELS
from .discourse_forest import (
    DEFAULT_SHUFFLE_SEED,
    DiscourseForest,
    STRUCTURE_MODES,
    render_structural_context,
)
from .dspy_programs import QuoteProgram
from .metrics import Span
from .policy import LABEL_POLICY, SPURIOUS_HINTS, TASK_CONTEXT

LabelCode = Literal["CO", "AS", "TE", "ST", "AN", "OT"]

_POLICY_BLOCK = "\n".join(f"- {LABEL_POLICY[label]}" for label in LABELS)
_EMPTY_ROLE_TOKENS = {"", "NONE", "NO LABEL", "NO LABELS", "EMPTY", "[]", "لا شيء"}

# Pair-specific distinctions are more useful GEPA feedback than repeating all
# six definitions after every error.  They are rules/annotation guidance, not
# manually invented labeled examples; examples themselves always come from the
# organizer-provided annotations.
ROLE_CONTRASTS: dict[tuple[str, str], str] = {
    ("CO", "AS"): (
        "CO presents background as already accepted or settled; AS advances the "
        "speaker's own contestable judgment, proposal, or conclusion."
    ),
    ("AN", "AS"): (
        "AN narrates or identifies a concrete case, incident, historical event, "
        "or illustrative scenario; AS states a general claim about such cases."
    ),
    ("AN", "TE"): (
        "AN is selected for the concrete event/example itself; TE requires that "
        "the relevant unit be presented as words or a statement attributed to a source."
    ),
    ("TE", "AS"): (
        "TE needs an identifiable attributed speaker/source (including an opposing "
        "debate team); an unattributed assertion is AS."
    ),
    ("ST", "TE"): (
        "ST marks explicit quantitative evidence and may legitimately be nested in "
        "or share offsets with a sourced TE statement."
    ),
    ("ST", "AS"): (
        "ST requires an explicit quantity, percentage, count, range, or reported "
        "study result; a merely empirical-sounding claim is AS."
    ),
    ("OT", "AS"): (
        "OT performs discourse management (greeting, roadmap, transition, speaker "
        "procedure, closing); rhetorical content that advances a position is AS."
    ),
}


class ClassifyOneAtom(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

Classify only the text enclosed by <TARGET>...</TARGET>.  The target came from
a high-recall extractor, so it may be a genuine ADU or a false/broad proposal.
Return:

- exactly one role in the normal case;
- [] when the target itself is not an annotated ADU (internal NONE);
- more than one role only when the same target characters genuinely carry
  nested or same-offset roles, which is rare but legal in the Daleel gold.

Judge the highlighted target in its surrounding context.  Do not label nearby
text outside the markers.  OT is a real annotated discourse-management unit;
[] is reserved for material that should be deleted from the proposal set.

Label inventory:
{_POLICY_BLOCK}"""

    paragraph_context: str = dspy.InputField(
        desc="an exact source-text window containing one <TARGET>...</TARGET> pair"
    )
    genre: Literal["editorial", "debate"] = dspy.InputField()
    target: str = dspy.InputField(desc="the exact candidate substring between the markers")
    adu_roles: list[str] = dspy.OutputField(
        desc="normally [one of CO, AS, TE, ST, AN, OT]; [] for internal NONE; "
        "multiple only for a genuine same-character multi-role annotation"
    )


class ClassifyOneAtomStructured(dspy.Signature):
    __doc__ = f"""{TASK_CONTEXT}

Classify only the text enclosed by <TARGET>...</TARGET>. The separate
structural-context field gives a compact, frozen view of neighboring atoms.
It always includes reading-order neighbors; in typed conditions it also gives
the target's validated parent, children, or siblings. Treat the unchanged
marked paragraph text as authoritative when a proposed relation is uncertain.
Never label neighboring text outside the target markers.

Return exactly one role in the normal case; [] when the target itself is not
an annotated ADU; and more than one role only for a genuine same-character
multi-role annotation. OT is a real discourse-management unit, whereas []
deletes a false or broad extractor proposal.

Label inventory:
{_POLICY_BLOCK}"""

    paragraph_context: str = dspy.InputField(
        desc="an exact source-text window containing one <TARGET>...</TARGET> pair"
    )
    structural_context: str = dspy.InputField(
        desc="a compact local atom neighborhood; typed relations may be withheld"
    )
    genre: Literal["editorial", "debate"] = dspy.InputField()
    target: str = dspy.InputField(desc="the exact candidate substring between the markers")
    adu_roles: list[str] = dspy.OutputField(
        desc="normally [one of CO, AS, TE, ST, AN, OT]; [] for internal NONE; "
        "multiple only for a genuine same-character multi-role annotation"
    )


def parse_role_set(prediction) -> tuple[tuple[str, ...], bool, list[str]]:
    """Decode ``adu_roles`` as ``(valid_roles, parsed_ok, unknown_tokens)``."""

    raw = getattr(prediction, "adu_roles", None)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.upper() in _EMPTY_ROLE_TOKENS:
            raw = []
        else:
            raw = [token for token in re.split(r"[,\s؛،]+", stripped) if token]
    if not isinstance(raw, (list, tuple, set)):
        return (), False, []

    valid: list[str] = []
    unknown: list[str] = []
    for item in raw:
        token = str(item).strip().upper()
        if token in LABELS:
            valid.append(token)
        elif token not in _EMPTY_ROLE_TOKENS:
            unknown.append(str(item))
    normalized = normalize_roles(valid)
    # [] is a legitimate NONE decision.  Any unknown token makes the complete
    # decision malformed: accepting ["AS", "BAD"] as a perfect AS prediction
    # would hide format drift from GEPA and candidate eligibility checks.
    parsed_ok = not unknown
    return normalized, parsed_ok, unknown


def _role_key(roles: Iterable[object]) -> str:
    normalized = normalize_roles(roles)
    if not normalized:
        return "NONE"
    # Rare/diagnostic roles get ownership of multi-role examples so balanced
    # sampling does not bury TE+ST under the common TE class.
    priority = ("ST", "CO", "AN", "TE", "OT", "AS")
    return next(role for role in priority if role in normalized)


class BalancedRoleDemoSelector:
    """Deterministic one-per-class nearest demos from official decisions.

    The old BFRS campaign selected four already-easy paragraph examples and
    omitted CO/ST entirely.  This selector enforces class coverage first, then
    uses Arabic character n-gram similarity and a small same-genre preference.
    It never uses an external embedding model.  The pool must be constructed
    only from the training IDs legal for the current fold/setting.
    """

    def __init__(
        self,
        examples: Sequence[dspy.Example],
        *,
        max_demos: int = 7,
        context_chars: int = 350,
        same_genre_bonus: float = 0.05,
        include_structure: bool = False,
    ) -> None:
        if max_demos < 0:
            raise ValueError("max_demos must be non-negative")
        self.max_demos = max_demos
        self.context_chars = context_chars
        self.same_genre_bonus = same_genre_bonus
        self.include_structure = include_structure
        self.examples = list(examples)
        self._groups: dict[str, list[int]] = defaultdict(list)
        documents: list[str] = []
        for index, example in enumerate(self.examples):
            self._groups[_role_key(example.adu_roles)].append(index)
            target = example.text[example.start : example.end]
            documents.append(target + " " + marked_context(
                example.text,
                example.start,
                example.end,
                min(context_chars, 180),
            ))

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if documents:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
                norm="l2",
            )
            self.matrix = self.vectorizer.fit_transform(documents)

    def __len__(self) -> int:
        return len(self.examples)

    def _demo(self, example: dspy.Example) -> dspy.Example:
        fields = dict(
            paragraph_context=marked_context(
                example.text, example.start, example.end, self.context_chars
            ),
            genre=example.genre,
            target=example.text[example.start : example.end],
            adu_roles=list(normalize_roles(example.adu_roles)),
        )
        input_fields = ["paragraph_context", "genre", "target"]
        if self.include_structure:
            structural_context = getattr(example, "structural_context", None)
            if not isinstance(structural_context, str) or not structural_context:
                raise ValueError("structured demo example has no structural_context")
            fields["structural_context"] = structural_context
            input_fields.insert(1, "structural_context")
        return dspy.Example(**fields).with_inputs(*input_fields)

    def select(
        self,
        *,
        paragraph_id: object,
        text: str,
        genre: str,
        start: int,
        end: int,
        structural_context: str | None = None,
    ) -> list[dspy.Example]:
        if not self.examples or self.max_demos == 0:
            return []

        scores = [0.0] * len(self.examples)
        if self.vectorizer is not None and self.matrix is not None:
            query = text[start:end] + " " + marked_context(
                text, start, end, min(self.context_chars, 180)
            )
            query_vector = self.vectorizer.transform([query])
            scores = (self.matrix @ query_vector.T).toarray().ravel().tolist()
        for i, example in enumerate(self.examples):
            if example.genre == genre:
                scores[i] += self.same_genre_bonus

        selected: list[int] = []
        # Keep a stable inventory order; NONE is an equally important class
        # because extractor over-coverage is a principal precision error.
        for key in (*LABELS, "NONE"):
            candidates = [
                i
                for i in self._groups.get(key, [])
                if self.examples[i].paragraph_id != paragraph_id and i not in selected
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda i: (
                    scores[i],
                    -int(self.examples[i].paragraph_id)
                    if isinstance(self.examples[i].paragraph_id, int)
                    else 0,
                    -i,
                ),
            )
            selected.append(best)
            if len(selected) >= self.max_demos:
                break
        return [self._demo(self.examples[i]) for i in selected]


class SpanRoleDecision(dspy.Module):
    """Optimizer-facing program: exactly one atom decision per forward call."""

    def __init__(
        self,
        *,
        cot: bool = False,
        demo_selector: BalancedRoleDemoSelector | None = None,
        context_chars: int = 500,
        structured: bool = False,
    ) -> None:
        super().__init__()
        self.context_chars = context_chars
        self.demo_selector = demo_selector
        self.structured = structured
        predictor = dspy.ChainOfThought if cot else dspy.Predict
        signature = ClassifyOneAtomStructured if structured else ClassifyOneAtom
        self.classify_span = predictor(signature)

    def forward(
        self,
        paragraph_id: object,
        text: str,
        genre: str,
        start: int,
        end: int,
        structural_context: str | None = None,
    ) -> dspy.Prediction:
        inputs = {
            "paragraph_context": marked_context(
                text, start, end, context_chars=self.context_chars
            ),
            "genre": genre,
            "target": text[start:end],
        }
        if self.structured:
            if not isinstance(structural_context, str) or not structural_context:
                raise ValueError("structured role decision requires structural_context")
            inputs["structural_context"] = structural_context
        if self.demo_selector is not None:
            inputs["demos"] = self.demo_selector.select(
                paragraph_id=paragraph_id,
                text=text,
                genre=genre,
                start=start,
                end=end,
                structural_context=structural_context,
            )
        # Deterministic retry ladder: a rare degenerate greedy completion
        # (empty output, tag-repetition loop) would otherwise be replayed
        # verbatim from the cache and permanently disqualify every candidate
        # under zero-tolerance eligibility. rollout_id only busts the cache
        # key, and each retry is itself cached, so reruns stay reproducible.
        last_error: Exception | None = None
        for rollout in range(3):
            try:
                if rollout == 0:
                    return self.classify_span(**inputs)
                base_lm = dspy.settings.lm
                if base_lm is None:
                    break
                retry_lm = base_lm.copy(rollout_id=rollout, temperature=1.0)
                with dspy.context(lm=retry_lm):
                    return self.classify_span(**inputs)
            except Exception as error:
                last_error = error
        assert last_error is not None
        raise last_error


class DecomposedSpanProgram(dspy.Module):
    """End-to-end inference composition for both official tasks.

    Compile ``SpanRoleDecision`` independently and then load its state through
    :meth:`load_role_state`.  Do not pass this looping parent module directly to
    GEPA/BFRS: repeated predictor invocations have ambiguous trace credit.
    """

    def __init__(
        self,
        *,
        extractor_cot: bool = True,
        role_cot: bool = False,
        demo_selector: BalancedRoleDemoSelector | None = None,
        granularity: str = "connective",
        context_chars: int = 500,
    ) -> None:
        super().__init__()
        self.extractor = QuoteProgram(cot=extractor_cot)
        self.role_stage = SpanRoleDecision(
            cot=role_cot,
            demo_selector=demo_selector,
            context_chars=context_chars,
        )
        self.granularity = granularity

    def load_role_state(self, path: str | Path) -> None:
        self.role_stage.load(path)

    def forward(self, paragraph_id: object, text: str, genre: str) -> dspy.Prediction:
        proposed = self.extractor(text=text, genre=genre)
        proposal_spans = list(getattr(proposed, "spans", []) or [])
        atoms = atomize_spans(text, proposal_spans, self.granularity)
        parsed_decisions: list[tuple[tuple[str, ...], bool]] = []
        decision_rows: list[dict] = []
        unknown_count = 0
        for atom in atoms:
            result = self.role_stage(
                paragraph_id=paragraph_id,
                text=text,
                genre=genre,
                start=atom.start,
                end=atom.end,
            )
            roles, parsed, unknown = parse_role_set(result)
            unknown_count += len(unknown)
            parsed_decisions.append((roles, parsed))
            decision_rows.append(
                {
                    "start": atom.start,
                    "end": atom.end,
                    "draft_roles": list(atom.draft_roles),
                    "roles": list(roles),
                    "parsed": parsed,
                    "unknown": unknown,
                }
            )

        spans, assembly = assemble_role_spans(atoms, parsed_decisions)
        return dspy.Prediction(
            spans=spans,
            adu_labels=sorted({span.label for span in spans}),
            atoms=atoms,
            decisions=decision_rows,
            assembly_stats=assembly,
            proposal_align_stats=getattr(proposed, "align_stats", {}),
            n_unknown_role_tokens=unknown_count,
        )


def span_role_examples(
    records: Sequence[dict],
    gold_spans: Mapping[object, Sequence[Span]],
    *,
    proposals: Mapping[object, Sequence[Span]] | None = None,
    granularity: str = "connective",
    containment_threshold: float = 0.80,
) -> list[dspy.Example]:
    """Build proposal-shaped standalone DSPy decisions.

    With ``proposals`` supplied, examples match deployment boundaries and
    include extractor false positives as empty-role hard negatives.  Without
    proposals, a whole-paragraph proposal is atomized as a deterministic,
    network-free fallback suitable for plumbing and initial demo pools.
    """

    examples: list[dspy.Example] = []
    for record in records:
        pid = record["paragraph_id"]
        text = record["text"]
        if not text:
            continue
        if proposals is None:
            candidate_spans = [Span(0, len(text), "")]
            source = "deterministic_full_text"
        else:
            if pid not in proposals:
                raise KeyError(f"proposal file has no paragraph_id {pid}")
            candidate_spans = list(proposals[pid])
            source = "frozen_quote_proposal"
        atoms = atomize_spans(text, candidate_spans, granularity)
        for atom in atoms:
            roles = gold_roles_for_atom(
                atom,
                gold_spans.get(pid, ()),
                containment_threshold=containment_threshold,
            )
            examples.append(
                dspy.Example(
                    paragraph_id=pid,
                    text=text,
                    genre=record["type"],
                    start=atom.start,
                    end=atom.end,
                    draft_roles=list(atom.draft_roles),
                    source=source,
                    adu_roles=list(roles),
                ).with_inputs("paragraph_id", "text", "genre", "start", "end")
            )
    return examples


def structurize_role_examples(
    examples: Sequence[dspy.Example],
    forests: Mapping[object, DiscourseForest],
    *,
    mode: str,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> list[dspy.Example]:
    """Attach one frozen local control view to each proposal-shaped example."""

    if mode not in STRUCTURE_MODES or mode == "baseline":
        raise ValueError(f"structural examples require flat/forest/shuffled, got {mode!r}")
    rendered: list[dspy.Example] = []
    for example in examples:
        forest = forests.get(example.paragraph_id)
        if forest is None:
            raise KeyError(f"no frozen forest for paragraph {example.paragraph_id!r}")
        node_id = forest.node_id_for(example.start, example.end)
        structural_context = render_structural_context(
            forest,
            node_id,
            mode,
            shuffle_seed=shuffle_seed,
        )
        rendered.append(
            dspy.Example(
                paragraph_id=example.paragraph_id,
                text=example.text,
                genre=example.genre,
                start=example.start,
                end=example.end,
                draft_roles=list(getattr(example, "draft_roles", ())),
                source=getattr(example, "source", "frozen_quote_proposal"),
                structural_context=structural_context,
                adu_roles=list(example.adu_roles),
            ).with_inputs(
                "paragraph_id",
                "text",
                "genre",
                "start",
                "end",
                "structural_context",
            )
        )
    return rendered


def balance_role_examples(
    examples: Sequence[dspy.Example],
    *,
    per_class: int,
    seed: int = 20260710,
) -> list[dspy.Example]:
    """Return an equal-size, deterministic sample for six roles plus NONE.

    Scarce groups are cycled after every unique example is used.  Repetition is
    intentional weighting (LM calls remain cache hits); candidate selection
    must still be done on the untouched natural-distribution fold.
    """

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    groups: dict[str, list[dspy.Example]] = defaultdict(list)
    for example in examples:
        groups[_role_key(example.adu_roles)].append(example)

    rng = random.Random(seed)
    balanced: list[dspy.Example] = []
    for key in (*LABELS, "NONE"):
        group = list(groups.get(key, ()))
        if not group:
            continue
        rng.shuffle(group)
        balanced.extend(group[i % len(group)] for i in range(per_class))
    rng.shuffle(balanced)
    return balanced


def _set_f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    return 2 * len(gold & pred) / ((len(gold) + len(pred)) or 1)


def _contrast(expected: str, predicted: str) -> str | None:
    return ROLE_CONTRASTS.get((expected, predicted)) or ROLE_CONTRASTS.get(
        (predicted, expected)
    )


class SpanRoleMetric:
    """Exact, decomposable atom-role metric with contrastive GEPA feedback."""

    def __call__(self, gold, pred, trace=None, pred_name=None, pred_trace=None):
        expected = set(normalize_roles(gold.adu_roles))
        got_roles, parsed, unknown = parse_role_set(pred)
        got = set(got_roles)
        score = 0.0 if not parsed else _set_f1(expected, got)
        if pred_name is None:
            return score
        return dspy.Prediction(
            score=score,
            feedback=self._feedback(gold, expected, got, parsed, unknown),
        )

    def _feedback(self, gold, expected, got, parsed, unknown) -> str:
        target = gold.text[gold.start : gold.end]
        if len(target) > 220:
            target = target[:220] + "…"
        if not parsed:
            detail = f" Unknown tokens: {unknown!r}." if unknown else ""
            return (
                "Unparseable output. Return `adu_roles` as [] or a list drawn "
                "only from CO, AS, TE, ST, AN, OT."
                + detail
                + " Target: «"
                + target
                + "»"
            )
        parts: list[str] = []
        if expected == got:
            return f"Correct role decision {sorted(expected)} for target «{target}»."
        if not expected:
            parts.append(
                "This extractor atom is internal NONE: output [] rather than an "
                "official label. OT is only for genuine discourse-management units."
            )
        for role in [label for label in LABELS if label in expected - got]:
            parts.append(f"Missed {role}. {LABEL_POLICY[role]}")
        for role in [label for label in LABELS if label in got - expected]:
            parts.append(f"Spurious {role}. {SPURIOUS_HINTS[role]}")
        if len(expected) == 1 and len(got) == 1:
            hint = _contrast(next(iter(expected)), next(iter(got)))
            if hint:
                parts.append("Key distinction: " + hint)
        parts.append(f"Classify only the marked target: «{target}»")
        return " ".join(parts)


def role_diagnostics(
    examples: Sequence[dspy.Example],
    predictions: Sequence,
) -> dict:
    """Natural-distribution accuracy/set-F1/confusions for an evaluation run."""

    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal length")
    exact = parse_failures = unknown_token_count = mixed_unknown_outputs = 0
    set_scores: list[float] = []
    confusion: Counter[tuple[str, str]] = Counter()
    for gold, pred in zip(examples, predictions):
        expected = set(normalize_roles(gold.adu_roles))
        roles, parsed, unknown = parse_role_set(pred)
        got = set(roles)
        parse_failures += not parsed
        unknown_token_count += len(unknown)
        mixed_unknown_outputs += bool(unknown and roles)
        exact += parsed and expected == got
        set_scores.append(0.0 if not parsed else _set_f1(expected, got))
        ekey = "+".join(sorted(expected)) if expected else "NONE"
        gkey = "+".join(sorted(got)) if got else ("NONE" if parsed else "PARSE_FAIL")
        confusion[(ekey, gkey)] += 1
    n = len(examples)
    return {
        "n": n,
        "exact_accuracy": exact / n if n else 0.0,
        "mean_set_f1": sum(set_scores) / n if n else 0.0,
        "parse_failures": parse_failures,
        "unknown_token_count": unknown_token_count,
        "mixed_unknown_outputs": mixed_unknown_outputs,
        "class_counts": dict(Counter(_role_key(ex.adu_roles) for ex in examples)),
        "confusions": {
            f"{expected}->{got}": count
            for (expected, got), count in sorted(confusion.items())
        },
    }


def assemble_example_predictions(
    examples: Sequence[dspy.Example],
    predictions: Sequence,
    *,
    fallback_to_draft: bool = True,
) -> tuple[dict[object, list[Span]], dict[object, set[str]], dict[str, int]]:
    """Reassemble flat atom-evaluation outputs at paragraph level.

    This is the bridge used for *external* candidate reranking: DSPy optimizes
    the decomposable atom metric, while every retained program is finally
    ranked with the official corpus Task 1/Task 2 scorers after this assembly.
    """

    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal length")
    grouped: dict[object, list[tuple[CandidateAtom, tuple[tuple[str, ...], bool]]]] = (
        defaultdict(list)
    )
    for example, prediction in zip(examples, predictions):
        atom = CandidateAtom(
            example.start,
            example.end,
            example.text[example.start : example.end],
            normalize_roles(getattr(example, "draft_roles", ())),
        )
        roles, parsed, _ = parse_role_set(prediction)
        grouped[example.paragraph_id].append((atom, (roles, parsed)))

    span_predictions: dict[object, list[Span]] = {}
    label_predictions: dict[object, set[str]] = {}
    totals: Counter[str] = Counter()
    # Preserve paragraphs that have examples but whose every atom becomes NONE.
    for pid, rows in grouped.items():
        rows.sort(key=lambda row: (row[0].start, row[0].end))
        spans, stats = assemble_role_spans(
            [row[0] for row in rows],
            [row[1] for row in rows],
            fallback_to_draft=fallback_to_draft,
        )
        span_predictions[pid] = spans
        label_predictions[pid] = {span.label for span in spans}
        totals.update(stats)
    return span_predictions, label_predictions, dict(totals)
