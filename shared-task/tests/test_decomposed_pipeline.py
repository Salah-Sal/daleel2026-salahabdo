"""No-network tests for the proposal-atom-role DSPy architecture."""

import dspy
import pytest
from dspy.utils import DummyLM

from daleel.candidates import (
    CandidateAtom,
    assemble_role_spans,
    atomize_spans,
    gold_roles_for_atom,
    marked_context,
)
from daleel.constants import LABELS
from daleel.discourse_forest import DiscourseForest, ForestEdge, forest_nodes
from daleel.dspy_span_roles import (
    BalancedRoleDemoSelector,
    DecomposedSpanProgram,
    SpanRoleDecision,
    SpanRoleMetric,
    balance_role_examples,
    parse_role_set,
    span_role_examples,
    structurize_role_examples,
)
from daleel.metrics import Span


def test_atomize_splits_broad_proposal_and_preserves_offsets():
    text = "هذه مقدمة، لكن المثال وقع سنة 2020. ولذلك يجب أن نتصرف."
    proposal = Span(0, len(text), "AS")
    atoms = atomize_spans(text, [proposal])
    assert len(atoms) >= 3
    assert all(text[atom.start : atom.end] == atom.text for atom in atoms)
    assert all(atom.text == atom.text.strip() for atom in atoms)
    assert all(atom.draft_roles == ("AS",) for atom in atoms)
    assert all(a.end <= b.start for a, b in zip(atoms, atoms[1:]))


def test_atomize_combines_overlapping_draft_roles():
    text = "قال الخبير إن 70٪ من السكان وافقوا."
    proposals = [
        Span(0, len(text), "TE"),
        Span(text.index("70"), text.index(" وافقوا"), "ST"),
    ]
    atoms = atomize_spans(text, proposals)
    assert any(set(atom.draft_roles) == {"TE", "ST"} for atom in atoms)


def test_gold_roles_support_none_single_and_same_offset_multi_role():
    text = "قال الخبير إن 70٪ وافقوا"
    atom = CandidateAtom(0, len(text), text, ("TE",))
    gold = [
        Span(0, len(text), "TE"),
        Span(text.index("70"), text.index(" وافقوا"), "ST"),
    ]
    # The short nested statistic is not projected over the whole testimony.
    assert gold_roles_for_atom(atom, gold) == ("TE",)
    same_offsets = [Span(0, len(text), "TE"), Span(0, len(text), "ST")]
    assert gold_roles_for_atom(atom, same_offsets) == ("TE", "ST")
    assert gold_roles_for_atom(CandidateAtom(100, 103, "نص"), gold) == ()
    claim = CandidateAtom(0, 8, text[:8])
    assert gold_roles_for_atom(claim, [Span(0, 8, "AS")]) == ("AS",)


def test_gold_roles_do_not_conflate_consecutive_units_inside_broad_atom():
    atom = CandidateAtom(0, 10, "أ" * 10)
    roles = gold_roles_for_atom(atom, [Span(0, 5, "AS"), Span(5, 10, "TE")])
    assert len(roles) == 1


def test_assemble_distinguishes_none_from_parse_failure_and_keeps_overlap():
    atoms = [
        CandidateAtom(0, 5, "أ" * 5, ("AS",)),
        CandidateAtom(5, 10, "ب" * 5, ("TE",)),
        CandidateAtom(10, 15, "ج" * 5, ("CO",)),
    ]
    spans, stats = assemble_role_spans(
        atoms,
        [(["TE", "ST", "TE"], True), ([], True), ([], False)],
    )
    assert [(s.start, s.end, s.label) for s in spans] == [
        (0, 5, "TE"),
        (0, 5, "ST"),
        (10, 15, "CO"),
    ]
    assert stats["none_atoms"] == 1
    assert stats["parse_failures"] == 1
    assert stats["fallback_atoms"] == 1


def test_marked_context_is_bounded_and_exact():
    text = "أ" * 30 + "الهدف" + "ب" * 30
    start = text.index("الهدف")
    rendered = marked_context(text, start, start + len("الهدف"), context_chars=8)
    assert "<TARGET>الهدف</TARGET>" in rendered
    assert rendered.startswith("…") and rendered.endswith("…")
    with pytest.raises(ValueError):
        marked_context(text, -1, 3)


def test_parse_role_set_tolerates_case_and_empty_but_flags_failure():
    assert parse_role_set(dspy.Prediction(adu_roles=["as", "ST", "AS"])) == (
        ("AS", "ST"),
        True,
        [],
    )
    assert parse_role_set(dspy.Prediction(adu_roles=[])) == ((), True, [])
    assert parse_role_set(dspy.Prediction(adu_roles="NONE")) == ((), True, [])
    roles, ok, unknown = parse_role_set(dspy.Prediction(adu_roles=["AS", "BAD"]))
    assert roles == ("AS",) and not ok and unknown == ["BAD"]
    assert parse_role_set(dspy.Prediction(adu_roles=["BAD"])) == ((), False, ["BAD"])
    assert parse_role_set(dspy.Prediction(other=True))[1] is False


def _decision(pid, role, genre="editorial"):
    label_text = role if role != "NONE" else "فراغ"
    text = f"سياق {pid} {label_text} ونهاية"
    start = text.index(label_text)
    roles = [] if role == "NONE" else [role]
    return dspy.Example(
        paragraph_id=pid,
        text=text,
        genre=genre,
        start=start,
        end=start + len(label_text),
        adu_roles=roles,
    ).with_inputs("paragraph_id", "text", "genre", "start", "end")


def test_balanced_demo_selector_covers_every_available_class_and_excludes_self():
    pool = [_decision(i + 1, role, "debate" if i % 2 else "editorial")
            for i, role in enumerate((*LABELS, "NONE"))]
    # Add an alternate CO so a CO query can exclude its own paragraph.
    pool.append(_decision(99, "CO"))
    selector = BalancedRoleDemoSelector(pool, max_demos=7)
    query = pool[0]
    demos = selector.select(
        paragraph_id=query.paragraph_id,
        text=query.text,
        genre=query.genre,
        start=query.start,
        end=query.end,
    )
    assert demos
    assert all(set(demo.inputs().keys()) == {"paragraph_context", "genre", "target"}
               for demo in demos)
    demo_roles = {tuple(demo.adu_roles) for demo in demos}
    assert ("CO",) in demo_roles
    assert () in demo_roles
    assert all("<TARGET>" in demo.paragraph_context for demo in demos)


def test_balance_role_examples_equalizes_groups():
    examples = [_decision(1, "AS"), _decision(2, "AS"), _decision(3, "CO"),
                _decision(4, "NONE")]
    got = balance_role_examples(examples, per_class=4, seed=7)
    counts = {key: 0 for key in ("AS", "CO", "NONE")}
    for example in got:
        key = example.adu_roles[0] if example.adu_roles else "NONE"
        counts[key] += 1
    assert counts == {"AS": 4, "CO": 4, "NONE": 4}


def test_span_role_examples_include_proposal_none_hard_negative():
    text = "تحية طيبة. الاقتصاد ينمو."
    records = [{"paragraph_id": 1, "text": text, "type": "editorial"}]
    proposals = {1: [Span(0, len(text), "AS")]}
    gold = {1: [Span(text.index("الاقتصاد"), len(text), "AS")]}
    examples = span_role_examples(records, gold, proposals=proposals)
    assert examples
    assert any(not ex.adu_roles for ex in examples)
    assert any(ex.adu_roles == ["AS"] for ex in examples)
    assert all(set(ex.inputs().keys()) == {"paragraph_id", "text", "genre", "start", "end"}
               for ex in examples)


def test_span_role_decision_passes_dynamic_demos():
    pool = [_decision(i + 1, role) for i, role in enumerate((*LABELS, "NONE"))]
    selector = BalancedRoleDemoSelector(pool)
    program = SpanRoleDecision(demo_selector=selector)
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return dspy.Prediction(adu_roles=["AS"])

    program.classify_span = fake
    query = _decision(100, "AS")
    pred = program(**query.inputs())
    assert pred.adu_roles == ["AS"]
    assert captured["target"] == query.text[query.start : query.end]
    assert captured["demos"]


def test_structured_role_decision_and_demos_receive_frozen_local_context():
    text = "مقدمة، ثم ادعاء واضح."
    records = [{"paragraph_id": 1, "text": text, "type": "editorial"}]
    proposals = {1: [Span(0, len(text), "AS")]}
    base = span_role_examples(records, {1: [Span(0, len(text), "AS")]},
                              proposals=proposals)
    atoms = atomize_spans(text, proposals[1])
    nodes = forest_nodes(atoms)
    forest = DiscourseForest(
        1,
        nodes,
        (ForestEdge(nodes[0].node_id, nodes[1].node_id, "SEQUENCE"),),
    )
    structured = structurize_role_examples(base, {1: forest}, mode="forest")
    selector = BalancedRoleDemoSelector(
        structured, max_demos=1, include_structure=True
    )
    program = SpanRoleDecision(demo_selector=selector, structured=True)
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return dspy.Prediction(adu_roles=["AS"])

    program.classify_span = fake
    query = _decision(99, "AS")
    prediction = program(
        **query.inputs(), structural_context=structured[0].structural_context
    )
    assert prediction.adu_roles == ["AS"]
    assert captured["structural_context"].startswith("STRUCTURE_CONTROL=TYPED")
    assert captured["demos"][0].structural_context
    assert set(captured["demos"][0].inputs()) == {
        "paragraph_context", "structural_context", "genre", "target"
    }


def test_structured_role_decision_rejects_missing_context():
    program = SpanRoleDecision(structured=True)
    query = _decision(1, "AS")
    with pytest.raises(ValueError, match="requires structural_context"):
        program(**query.inputs())


def test_real_predict_accepts_dynamic_demos_with_dummy_lm():
    pool = [_decision(i + 1, role) for i, role in enumerate((*LABELS, "NONE"))]
    program = SpanRoleDecision(demo_selector=BalancedRoleDemoSelector(pool))
    query = _decision(100, "AS")
    with dspy.context(lm=DummyLM([{"adu_roles": ["AS"]}])):
        prediction = program(**query.inputs())
    assert prediction.adu_roles == ["AS"]


def test_span_role_decision_retries_degenerate_completions_deterministically():
    seen_lms = []

    class FlakyPredictor:
        def __call__(self, **kwargs):
            seen_lms.append(dspy.settings.lm)
            if len(seen_lms) < 3:
                raise ValueError("degenerate completion")
            return dspy.Prediction(adu_roles=["AS"])

    program = SpanRoleDecision()
    program.classify_span = FlakyPredictor()
    query = _decision(1, "AS")
    with dspy.context(lm=DummyLM([{"adu_roles": ["AS"]}])):
        prediction = program(**query.inputs())

    assert prediction.adu_roles == ["AS"]
    assert len(seen_lms) == 3
    # Retries must be cache-busted rollouts of the session LM, in a fixed
    # ladder, so a rerun replays the identical attempt sequence.
    assert seen_lms[1].kwargs.get("rollout_id") == 1
    assert seen_lms[2].kwargs.get("rollout_id") == 2
    assert seen_lms[1].kwargs.get("temperature") == 1.0

    class AlwaysDegenerate:
        def __call__(self, **kwargs):
            raise ValueError("degenerate completion")

    exhausted = SpanRoleDecision()
    exhausted.classify_span = AlwaysDegenerate()
    with dspy.context(lm=DummyLM([{"adu_roles": ["AS"]}])):
        with pytest.raises(ValueError, match="degenerate completion"):
            exhausted(**query.inputs())


def test_span_role_state_roundtrip(tmp_path):
    program = SpanRoleDecision()
    evolved = program.classify_span.signature.with_instructions("EVOLVED ATOM INSTRUCTIONS")
    program.classify_span.signature = evolved
    path = tmp_path / "role.json"
    program.save(path, save_program=False)

    fresh = SpanRoleDecision()
    fresh.load(path)
    assert fresh.classify_span.signature.instructions == "EVOLVED ATOM INSTRUCTIONS"


def test_span_role_metric_scores_and_feedback():
    metric = SpanRoleMetric()
    gold = _decision(1, "AN")
    assert metric(gold, dspy.Prediction(adu_roles=["AN"])) == 1.0
    assert metric(gold, dspy.Prediction(adu_roles=[])) == 0.0
    feedback = metric(
        gold,
        dspy.Prediction(adu_roles=["AS"]),
        pred_name="classify_span",
    )
    assert feedback.score == 0.0
    assert "Key distinction" in feedback.feedback
    assert "AN" in feedback.feedback and "AS" in feedback.feedback
    negative = _decision(2, "NONE")
    assert metric(negative, dspy.Prediction(adu_roles=[])) == 1.0
    mixed = metric(
        _decision(3, "AS"),
        dspy.Prediction(adu_roles=["AS", "BAD"]),
        pred_name="classify_span",
    )
    assert mixed.score == 0.0
    assert "BAD" in mixed.feedback


def test_decomposed_program_assembles_task1_and_task2(monkeypatch):
    text = "مقدمة، ثم ادعاء واضح."
    program = DecomposedSpanProgram()
    program.extractor = lambda **kw: dspy.Prediction(
        spans=[Span(0, len(text), "AS")], align_stats={"exact": 1}
    )

    def role(**kwargs):
        target = kwargs["text"][kwargs["start"] : kwargs["end"]]
        return dspy.Prediction(adu_roles=[] if "مقدمة" in target else ["AS"])

    program.role_stage = role
    pred = program(paragraph_id=1, text=text, genre="editorial")
    assert pred.adu_labels == ["AS"]
    assert pred.spans and all(span.label == "AS" for span in pred.spans)
    assert pred.assembly_stats["none_atoms"] >= 1
