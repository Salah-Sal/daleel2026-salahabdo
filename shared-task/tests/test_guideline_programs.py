"""Unit tests for the guideline-native program stack: policy seeds, the
official-examples demo bank, label parsing, index-preserving alignment,
and GuidelineProgram decode/containment logic. No LLM calls anywhere."""

import dspy
import pytest

from daleel.constants import LABELS
from daleel.guideline_demos import categorize_demos, segment_demos
from daleel.guideline_programs import (
    CategorizeUnits,
    GuidelineProgram,
    SegmentUnits,
    align_units,
    parse_unit_labels,
)


# ---- policy seeds ---------------------------------------------------------


def test_signature_seeds_carry_the_official_razors():
    seg_doc = SegmentUnits.__doc__
    assert "VERBATIM" in seg_doc
    assert "sentence boundary" in seg_doc
    assert "said that" in seg_doc  # U9: attribution + content = one unit
    cat_doc = CategorizeUnits.__doc__
    assert "IDENTIFIABLE" in cat_doc  # TE razor
    assert "WITHOUT supporting evidence" in cat_doc  # CO audience test
    assert "personal experience" in cat_doc  # AN/ST razor
    assert "ST/TE" in cat_doc  # multi-category sanction


def test_signature_seeds_have_no_arabic_bytes():
    # The guideline-native modules are deliberately Arabic-free so the
    # repo's mechanical staged-diff sweep stays meaningful.
    for doc in (SegmentUnits.__doc__, CategorizeUnits.__doc__):
        assert not any(0x0600 <= ord(ch) <= 0x06FF for ch in doc)


# ---- demo bank ------------------------------------------------------------


def test_segment_demos_are_verbatim_and_complete():
    demos = segment_demos()
    assert len(demos) == 8  # guidelines §3.1 has exactly 8 worked examples
    for demo in demos:
        assert demo.units, demo.text
        for unit in demo.units:
            assert unit in demo.text  # verbatim substring, aligner-recoverable
        assert set(demo.inputs().keys()) == {"text", "genre"}


def test_categorize_demos_are_consistent_and_parseable():
    demos = categorize_demos()
    assert len(demos) == 8  # §3.2's seven + the TE-definition debate example
    genres = set()
    dual_seen = False
    for demo in demos:
        genres.add(demo.genre)
        assert len(demo.unit_labels) == len(demo.units)
        for unit in demo.units:
            assert unit in demo.text
        for answer in demo.unit_labels:
            parsed = parse_unit_labels(answer)
            assert parsed, answer  # every official answer must parse
            dual_seen = dual_seen or len(parsed) == 2
        assert set(demo.inputs().keys()) == {"text", "genre", "units"}
    assert dual_seen  # the ST/TE dual (§3.2 ex. 7) must survive round-trip
    assert genres == {"editorial", "debate"}


def test_demo_label_inventory_covers_all_six():
    seen = {
        label
        for demo in categorize_demos()
        for answer in demo.unit_labels
        for label in parse_unit_labels(answer)
    }
    assert seen == set(LABELS)


# ---- label parsing --------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("CO", ["CO"]),
        (" co ", ["CO"]),
        ("st/te", ["ST", "TE"]),
        ("ST, TE", ["ST", "TE"]),
        ("ST+TE", ["ST", "TE"]),
        ("TE/TE", ["TE"]),
        ("QQ", []),
        ("", []),
        (None, []),
        ("TE and ST", ["TE", "ST"]),
    ],
)
def test_parse_unit_labels(raw, expected):
    assert parse_unit_labels(raw) == expected


# ---- index-preserving alignment -------------------------------------------


def test_align_units_maps_indices_in_order():
    text = "We should demolish the building, It is full of asbestos."
    units = ["We should demolish the building", "It is full of asbestos"]
    aligned, stats = align_units(text, units)
    assert [idx for idx, _ in aligned] == [0, 1]
    assert stats["exact"] == 2 and stats["unaligned"] == 0
    for (_, span), unit in zip(aligned, units):
        assert text[span.start : span.end] == unit


def test_align_units_drops_fully_covered_overlap_but_keeps_mapping():
    text = "alpha beta gamma delta."
    aligned, stats = align_units(text, ["alpha beta gamma", "beta gamma"])
    assert stats["dropped_overlap"] == 1
    assert [idx for idx, _ in aligned] == [0]


def test_align_units_trims_partial_overlap():
    text = "alpha beta gamma delta."
    aligned, stats = align_units(text, ["alpha beta", "beta gamma delta"])
    assert stats["dropped_overlap"] == 0
    assert [idx for idx, _ in aligned] == [0, 1]
    first, second = aligned[0][1], aligned[1][1]
    assert second.start >= first.end  # non-overlap policy preserved
    assert text[second.start : second.end].startswith("gamma")


def test_align_units_counts_unmatchable_quote_as_unaligned():
    text = "alpha beta gamma delta."
    aligned, stats = align_units(text, ["quantum blockchain synergy"])
    assert aligned == [] and stats["unaligned"] == 1


# ---- GuidelineProgram decode/containment (stubbed predictors) -------------


def make_program(units, unit_labels, **kwargs):
    program = GuidelineProgram(cot=False, **kwargs)
    program.segment = lambda **kw: dspy.Prediction(units=units)
    program.categorize = lambda **kw: dspy.Prediction(unit_labels=unit_labels)
    return program


TEXT = "We should demolish the building, It is full of asbestos."


def test_forward_normal_flow():
    program = make_program(
        ["We should demolish the building", "It is full of asbestos"],
        ["CO", "as"],
    )
    pred = program(text=TEXT, genre="editorial")
    assert [s.label for s in pred.spans] == ["CO", "AS"]
    assert TEXT[pred.spans[0].start : pred.spans[0].end] == "We should demolish the building"
    assert pred.adu_labels == ["AS", "CO"]
    assert pred.n_units == 2 and not pred.length_mismatch
    assert pred.n_multi == 0 and pred.n_unlabeled == 0


def test_forward_dual_category_primary_only_by_default():
    program = make_program(["We should demolish the building"], ["ST/TE"])
    pred = program(text=TEXT, genre="editorial")
    assert [s.label for s in pred.spans] == ["ST"]  # primary only
    assert pred.adu_labels == ["ST", "TE"]  # Task 1 union keeps both
    assert pred.n_multi == 1


def test_forward_dual_category_emits_same_offset_span_when_enabled():
    program = make_program(
        ["We should demolish the building"], ["ST/TE"], emit_dual_spans=True
    )
    pred = program(text=TEXT, genre="editorial")
    assert [s.label for s in pred.spans] == ["ST", "TE"]
    assert pred.spans[0].start == pred.spans[1].start
    assert pred.spans[0].end == pred.spans[1].end


def test_forward_length_mismatch_leaves_tail_unlabeled():
    program = make_program(
        ["We should demolish the building", "It is full of asbestos"],
        ["CO"],  # one answer for two units
    )
    pred = program(text=TEXT, genre="editorial")
    assert pred.length_mismatch and pred.n_unlabeled == 1
    assert [s.label for s in pred.spans] == ["CO"]


def test_forward_invalid_answer_never_crashes():
    program = make_program(["We should demolish the building"], ["QQ"])
    pred = program(text=TEXT, genre="editorial")
    assert pred.spans == [] and pred.adu_labels == []
    assert pred.n_unlabeled == 1


def test_forward_unaligned_unit_still_feeds_task1_union():
    # Presence does not require recovered offsets (QuoteProgram's rule).
    program = make_program(["quantum blockchain synergy"], ["TE"])
    pred = program(text=TEXT, genre="editorial")
    assert pred.spans == []
    assert pred.adu_labels == ["TE"]
    assert pred.align_stats["unaligned"] == 1


def test_forward_empty_units_is_a_legal_empty_prediction():
    program = make_program([], [])
    pred = program(text=TEXT, genre="editorial")
    assert pred.spans == [] and pred.adu_labels == []
    assert pred.n_units == 0 and not pred.length_mismatch


def test_forward_non_list_outputs_are_contained():
    program = GuidelineProgram(cot=False)
    program.segment = lambda **kw: dspy.Prediction(units="not a list")
    pred = program(text=TEXT, genre="editorial")
    assert pred.spans == [] and pred.adu_labels == []


# ---- module-owned LM ------------------------------------------------------


def test_module_owned_lm_reaches_every_predictor():
    lm = dspy.LM("openai/module-owned-test", api_key="unused")
    program = GuidelineProgram(cot=True, lm=lm)
    predictors = list(program.named_predictors())
    assert predictors  # CoT wraps: segment.predict / categorize.predict
    for _, predictor in predictors:
        assert predictor.lm is lm


def test_default_construction_keeps_global_fallback():
    # lm=None must preserve the legacy behavior exactly: predictor.lm stays
    # None and resolution falls through to the dspy.configure global.
    program = GuidelineProgram(cot=False)
    for _, predictor in program.named_predictors():
        assert predictor.lm is None


def test_two_modules_never_share_an_lm_instance():
    from daleel.dspy_programs import QuoteProgram, Task1Program

    lms = [dspy.LM("openai/module-owned-test", api_key="unused") for _ in range(3)]
    programs = [
        GuidelineProgram(cot=True, lm=lms[0]),
        Task1Program(cot=True, lm=lms[1]),
        QuoteProgram(cot=True, lm=lms[2]),
    ]
    owned = [{id(p.lm) for _, p in program.named_predictors()} for program in programs]
    for lm, ids in zip(lms, owned):
        assert ids == {id(lm)}  # each module: exactly its own instance
    assert len(set().union(*owned)) == 3  # and no instance crosses modules


# ---- mlflow gate ----------------------------------------------------------


def test_mlflow_is_disabled_under_pytest_but_defaults_on():
    import os

    from daleel import runtime

    # conftest.py hard-sets the opt-out before daleel imports…
    assert os.environ["DALEEL_MLFLOW"] == "0"
    assert runtime.MLFLOW_ENABLED is False
    # …and the gate itself defaults ON for any other value.
    assert runtime.enable_mlflow.__defaults__ is None  # env-driven, no arg
    for off in ("0", "false", "OFF"):
        os.environ["DALEEL_MLFLOW"] = off
        assert runtime.enable_mlflow() is False
    os.environ["DALEEL_MLFLOW"] = "0"
