"""Unit tests for the stage-0 pipeline: splits, segmenter, DSPy metrics,
and program decode logic. No LLM calls anywhere; data-dependent tests use
the official clone like the parity tests do."""

import dspy
import pytest

from daleel.align import align_adus, find_quote
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records
from daleel.dspy_metrics import Task1Metric, Task2Metric, parse_label_set
from daleel.dspy_metrics import JudgeMetric
from daleel.dspy_programs import (
    JudgeStage,
    QuoteProgram,
    RelabelStage,
    Task1Program,
    Task2Program,
    VerifyOne,
    task1_examples,
)
from daleel.metrics import Span, span_partial_f1
from daleel.segment import oracle_ceiling, oracle_label, Segment, segment_paragraph
from daleel.splits import (
    FORCED_TRAIN_IDS,
    clean_task1_gold,
    clean_task2_gold,
    filter_setting,
    optimizer_split,
    train_val_ids,
)


@pytest.fixture(scope="module")
def t1():
    return load_records(TRAIN_TASK1)


@pytest.fixture(scope="module")
def t2():
    return load_records(TRAIN_TASK2)


# ---------------------------------------------------------------- splits


def test_cleaning_drops_964_phantoms(t1, t2):
    spans = clean_task2_gold(t2)
    assert all(s.label not in ("ST", "CO") for s in spans[964])
    labels = clean_task1_gold(t1, t2)
    assert "ST" not in labels[964] and "CO" not in labels[964]
    # everywhere else the derivation equals the official Task 1 labels
    official = {r["paragraph_id"]: set(r["labels"]) for r in t1}
    diffs = [pid for pid in official if labels[pid] != official[pid]]
    assert diffs == [964]


def test_split_fixed_and_stratified(t1, t2):
    train, val = train_val_ids(t1, t2)
    train2, val2 = train_val_ids(t1, t2)
    assert (train, val) == (train2, val2), "split must be deterministic"
    assert not set(train) & set(val)
    assert len(train) + len(val) == len(t1)
    assert FORCED_TRAIN_IDS <= set(train)
    assert 0.25 <= len(val) / len(t1) <= 0.35
    gold = clean_task1_gold(t1, t2)
    for rare in ("ST", "CO"):
        n_total = sum(rare in gold[pid] for pid in gold)
        n_val = sum(rare in gold[pid] for pid in val)
        assert 0.2 <= n_val / n_total <= 0.4, f"{rare} not proportionally in val"


def test_filter_setting(t1):
    assert len(filter_setting(t1, "both")) == len(t1)
    editorial = filter_setting(t1, "editorial")
    assert editorial and all(r["type"] == "editorial" for r in editorial)


def test_compiled_state_roundtrip(tmp_path):
    """What compile_program.py saves, run_zero_shot.py --compiled must restore."""
    program = Task1Program()
    predictor = next(p for _, p in program.named_predictors())
    predictor.signature = predictor.signature.with_instructions("EVOLVED INSTRUCTIONS")
    predictor.demos = [
        dspy.Example(
            text="فقرة تجريبية", genre="editorial", reasoning="...", adu_labels=["AS"]
        ).with_inputs("text", "genre")
    ]
    path = tmp_path / "state.json"
    program.save(path, save_program=False)

    fresh = Task1Program()
    fresh_pred = next(p for _, p in fresh.named_predictors())
    assert fresh_pred.signature.instructions != "EVOLVED INSTRUCTIONS"
    fresh.load(path)
    assert fresh_pred.signature.instructions == "EVOLVED INSTRUCTIONS"
    assert len(fresh_pred.demos) == 1
    demo = fresh_pred.demos[0]
    get = demo.get if hasattr(demo, "get") else demo.__getitem__
    assert get("adu_labels") == ["AS"]


def test_optimizer_split_stays_inside_train_side(t1, t2):
    train, val = train_val_ids(t1, t2)
    opt_train, opt_val = optimizer_split(t1, t2)
    assert (opt_train, opt_val) == optimizer_split(t1, t2), "must be deterministic"
    assert not set(opt_train) & set(opt_val)
    assert set(opt_train) | set(opt_val) == set(train), "frozen val must stay untouched"
    assert not set(opt_val) & set(val)
    assert FORCED_TRAIN_IDS <= set(opt_train)
    gold = clean_task1_gold(t1, t2)
    for rare in ("ST", "CO"):
        n_train_side = sum(rare in gold[pid] for pid in train)
        n_opt_val = sum(rare in gold[pid] for pid in opt_val)
        assert n_opt_val >= 1, f"opt_val must contain some {rare}"
        assert 0.2 <= n_opt_val / n_train_side <= 0.5


# -------------------------------------------------------------- segmenter


def test_segment_offsets_are_exact():
    text = "قال الرئيس: الوضع جيد، والاقتصاد ينمو بسرعة. هذا تقدم كبير!"
    segments = segment_paragraph(text)
    assert segments
    for seg in segments:
        assert text[seg.start : seg.end] == seg.text
        assert seg.text == seg.text.strip()
    starts = [s.start for s in segments]
    assert starts == sorted(starts)
    for a, b in zip(segments, segments[1:]):
        assert a.end <= b.start, "segments must not overlap"


def test_speaker_marker_is_protected():
    text = "*المتحدث الأول موالاة: (ذكر)/* شكرا جزيلا"
    segments = segment_paragraph(text)
    assert segments[0].text == "*المتحدث الأول موالاة: (ذكر)/*"


def test_connective_granularity_cuts_wa():
    text = "ذهب الولد إلى السوق و اشترى خبزا طازجا"
    assert len(segment_paragraph(text, "connective")) == 2
    assert len(segment_paragraph(text, "clause")) == 1


def test_oracle_label_majority():
    seg = Segment(0, 10, "x" * 10)
    gold = [Span(0, 3, "OT"), Span(3, 10, "AS")]
    assert oracle_label(seg, gold) == "AS"
    assert oracle_label(Segment(50, 60, "y" * 10), gold) is None


def test_oracle_ceiling_holds(t2):
    texts = {r["paragraph_id"]: r["text"] for r in t2}
    gold = clean_task2_gold(t2)
    assert oracle_ceiling(texts, gold, "connective")["f1"] > 0.90


# ---------------------------------------------------------------- metrics


def _ex(**kw):
    return dspy.Example(**kw)


def test_parse_label_set_variants():
    assert parse_label_set(dspy.Prediction(adu_labels=["AS", "TE"])) == ({"AS", "TE"}, True, [])
    labels, ok, unknown = parse_label_set(dspy.Prediction(adu_labels=["AS", "STAT"]))
    assert labels == {"AS"} and ok and unknown == ["STAT"]
    assert parse_label_set(dspy.Prediction(adu_labels="AS, ST"))[0] == {"AS", "ST"}
    assert parse_label_set(dspy.Prediction(adu_labels=["NONE"])) == (set(), True, [])
    assert parse_label_set(dspy.Prediction(other=1))[1] is False
    assert parse_label_set(None)[1] is False


def test_task1_metric_scores():
    m = Task1Metric()
    assert m(_ex(adu_labels=[]), dspy.Prediction(adu_labels=[])) == 1.0
    assert m(_ex(adu_labels=["AS", "TE"]), dspy.Prediction(adu_labels=["TE", "AS"])) == 1.0
    assert m(_ex(adu_labels=["AS", "TE"]), dspy.Prediction(adu_labels=["AS"])) == pytest.approx(2 / 3)
    # parse failure on empty gold must NOT look like a correct empty answer
    assert m(_ex(adu_labels=[]), dspy.Prediction(nothing=True)) == 0.0


def test_task1_metric_gepa_feedback_quotes_gold_span():
    text = "أظهرت دراسة أن 85٪ من المشاركين يؤيدون القرار"
    m = Task1Metric(gold_spans={7: [Span(0, 30, "ST")]}, texts={7: text})
    out = m(
        _ex(paragraph_id=7, adu_labels=["ST"]),
        dspy.Prediction(adu_labels=["AS"]),
        pred_name="classify",
    )
    assert isinstance(out, dspy.Prediction)
    assert out.score == 0.0
    assert "Missed ST" in out.feedback and "«" + text[:30] + "»" in out.feedback
    assert "Spurious AS" in out.feedback


def test_task2_metric_scores():
    m = Task2Metric()
    assert m(_ex(spans=[]), dspy.Prediction(spans=[])) == 1.0
    gold = [Span(0, 10, "AS"), Span(10, 20, "TE")]
    assert m(_ex(spans=gold), dspy.Prediction(spans=list(gold))) == pytest.approx(1.0)
    assert m(_ex(spans=gold), dspy.Prediction(spans=[])) == 0.0
    assert m(_ex(spans=gold), dspy.Prediction(nothing=True)) == 0.0
    partial = m(_ex(spans=gold), dspy.Prediction(spans=[Span(0, 10, "AS")]))
    assert 0.0 < partial < 1.0
    fb = m(
        _ex(spans=gold, text="أ" * 20),
        dspy.Prediction(spans=[Span(0, 10, "AS")]),
        pred_name="classify",
    )
    assert "No segment was labeled TE" in fb.feedback


# --------------------------------------------------------------- programs


def test_task2_program_decode(monkeypatch):
    program = Task2Program()
    text = "هذا جيد، وهذا أفضل. النهاية!"
    segments = segment_paragraph(text)
    fake = dspy.Prediction(segment_labels=["AS", "NONE", "OT"][: len(segments)])
    program.classify = lambda **kw: fake
    pred = program(text=text, genre="editorial")
    assert not pred.length_mismatch
    assert [s.label for s in pred.spans] == [
        l for l in fake.segment_labels if l != "NONE"
    ]
    for span in pred.spans:
        assert text[span.start : span.end].strip()


def test_task2_program_pads_short_output(monkeypatch):
    program = Task2Program()
    text = "هذا جيد، وهذا أفضل. النهاية!"
    program.classify = lambda **kw: dspy.Prediction(segment_labels=["AS"])
    pred = program(text=text, genre="editorial")
    assert pred.length_mismatch
    assert len(pred.segment_labels) == len(segment_paragraph(text))


def test_relabel_stage_applies_and_contains():
    text = "هذا جيد، وهذا أفضل. النهاية!"
    spans = [Span(0, 8, "AS"), Span(9, 19, "CO"), Span(20, 28, "OT")]
    stage = RelabelStage()
    stage.relabel = lambda **kw: dspy.Prediction(corrected_labels=["AS", "AN"])
    pred = stage(text=text, genre="editorial", spans=spans)
    # short answer: positions covered by the reply are applied, tail keeps drafts
    assert [s.label for s in pred.spans] == ["AS", "AN", "OT"]
    assert pred.length_mismatch and pred.n_changed == 1
    # invalid labels never replace a draft
    stage.relabel = lambda **kw: dspy.Prediction(corrected_labels=["XX", "as", "OT"])
    pred = stage(text=text, genre="editorial", spans=spans)
    assert [s.label for s in pred.spans] == ["AS", "AS", "OT"]
    # boundaries are never touched
    assert [(s.start, s.end) for s in pred.spans] == [(0, 8), (9, 19), (20, 28)]


def test_relabel_stage_blind_withholds_drafts():
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return dspy.Prediction(corrected_labels=["AS"])

    stage = RelabelStage(blind=True)
    stage.relabel = spy
    stage(text="نص", genre="debate", spans=[Span(0, 2, "CO")])
    assert "draft_labels" not in seen
    assert "segments" in seen


def test_judge_stage_drops_and_arbitrates():
    verdicts = {"CO": False, "ST": True, "OT": False, "AN": True}
    stage = JudgeStage()
    stage.verify = lambda **kw: dspy.Prediction(present=verdicts[kw["label"]])
    pred = stage(text="نص", genre="debate", proposed={"CO", "ST", "OT", "AS"})
    # CO/OT dropped by their judges, ST kept, AN added by the arbiter,
    # AS untouched (never judged)
    assert pred.adu_labels == ["AN", "AS", "ST"]
    assert sorted(pred.dropped) == ["CO", "OT"] and pred.added == ["AN"]
    # arbiter also removes a fired AN when absent
    verdicts["AN"] = False
    pred = stage(text="نص", genre="debate", proposed={"AN", "AS"})
    assert pred.adu_labels == ["AS"] and pred.dropped == ["AN"]


def test_verify_one_state_loads_into_judge_stage(tmp_path):
    student = VerifyOne()
    evolved = student.verify.predict.signature.with_instructions(
        "EVOLVED JUDGE INSTRUCTIONS"
    )
    student.verify.predict.signature = evolved
    path = tmp_path / "judge-gepa.json"
    student.save(path, save_program=False)
    stage = JudgeStage()
    stage.load(path)
    assert stage.verify.predict.signature.instructions == "EVOLVED JUDGE INSTRUCTIONS"


def test_judge_metric_scores_and_feedback():
    gold_spans = {7: [Span(0, 6, "CO")]}
    texts = {7: "مثل شائع يقبله الجميع"}
    metric = JudgeMetric(gold_spans, texts)
    gold = dspy.Example(paragraph_id=7, label="CO", present=True)
    assert metric(gold, dspy.Prediction(present=True)) == 1.0
    assert metric(gold, dspy.Prediction(present="false")) == 0.0
    assert metric(gold, dspy.Prediction()) == 0.0  # unparseable
    fb = metric(gold, dspy.Prediction(present=False), pred_name="verify")
    assert fb.score == 0.0
    assert "CO IS present" in fb.feedback and "«مثل ش" in fb.feedback
    neg = dspy.Example(paragraph_id=7, label="ST", present=False)
    fb = metric(neg, dspy.Prediction(present=True), pred_name="verify")
    assert "ST is NOT present" in fb.feedback


def test_task1_examples_shape(t1, t2):
    gold = clean_task1_gold(t1, t2)
    ex = task1_examples(t1[:3], gold)[0]
    assert set(ex.inputs().keys()) == {"text", "genre"}
    assert isinstance(ex.adu_labels, list)


# ------------------------------------------------------------ quote-align


def test_find_quote_exact_offsets():
    text = "قال الوزير إن الاقتصاد ينمو بسرعة كبيرة هذا العام."
    q = "الاقتصاد ينمو بسرعة"
    found = find_quote(text, q)
    assert found.stage == "exact"
    assert text[found.span.start : found.span.end] == q


def test_find_quote_cursor_resolves_repeats():
    text = "نعم صحيح. لكن نعم صحيح أيضا."
    first = find_quote(text, "نعم صحيح", cursor=0)
    second = find_quote(text, "نعم صحيح", cursor=first.span.end)
    assert first.span.start == 0
    assert second.span.start > first.span.end


def test_find_quote_strips_decorations_and_whitespace_drift():
    text = "هذا مثال واضح\nعلى الفكرة المطروحة هنا."
    found = find_quote(text, "«مثال واضح على الفكرة»")
    assert found.stage == "ws"
    assert text[found.span.start : found.span.end] == "مثال واضح\nعلى الفكرة"


def test_find_quote_arabic_normalization():
    text = "أكد الرئيس أن المسألة محسومة نهائياً."
    # model drops the hamza and the final diacritic
    found = find_quote(text, "اكد الرئيس ان المسألة محسومة نهائيا")
    assert found is not None and found.stage == "norm"
    assert found.span.start == 0


def test_find_quote_fuzzy_survives_small_drift():
    text = "المشكلة الحقيقية هي غياب التخطيط طويل الأمد في المؤسسات الحكومية."
    # one word paraphrased ("انعدام" for "غياب") — exact/norm fail, fuzzy hits
    found = find_quote(text, "المشكلة الحقيقية هي انعدام التخطيط طويل الأمد")
    assert found is not None and found.stage == "fuzzy"
    assert found.span.start == 0


def test_find_quote_rejects_hallucination():
    text = "الاقتصاد الوطني في تحسن مستمر."
    assert find_quote(text, "الطقس اليوم ممطر في الشمال") is None


def test_align_adus_nonoverlapping_and_stats():
    text = "الفقر يتزايد بوضوح. لذلك يجب التدخل فورا. هذا رأي الخبراء."
    items = [
        ("AS", "الفقر يتزايد بوضوح"),
        ("CO", "الفقر يتزايد بوضوح. لذلك يجب التدخل"),  # overlaps the first
        ("TE", "نص غير موجود إطلاقا في أي مكان"),
    ]
    spans, stats = align_adus(text, items)
    assert stats["unaligned"] == 1
    ordered = sorted(spans, key=lambda s: s.start)
    assert all(a.end <= b.start for a, b in zip(ordered, ordered[1:]))
    for s in spans:
        assert text[s.start : s.end].strip()


def test_quote_align_oracle_ceiling(t2):
    """A perfect quoter + the aligner must beat the segmenter's 0.920."""
    gold = clean_task2_gold(t2)
    texts = {r["paragraph_id"]: r["text"] for r in t2}
    pred = {}
    for pid, spans in gold.items():
        items = [
            (s.label, texts[pid][s.start : s.end])
            for s in sorted(spans, key=lambda s: s.start)
        ]
        pred[pid], _ = align_adus(texts[pid], items)
    assert span_partial_f1(gold, pred)["f1"] > 0.99


def test_quote_program_decode():
    program = QuoteProgram()
    text = "الفقر يتزايد بوضوح. لذلك يجب التدخل فورا."
    fake = dspy.Prediction(
        adus=[
            {"label": "as", "quote": "الفقر يتزايد بوضوح"},  # case-normalized
            {"label": "CO", "quote": "لذلك يجب التدخل فورا"},
            {"label": "XX", "quote": "الفقر"},  # unknown label dropped
        ]
    )
    program.extract = lambda **kw: fake
    pred = program(text=text, genre="editorial")
    assert pred.adu_labels == ["AS", "CO"]
    assert [s.label for s in pred.spans] == ["AS", "CO"]
    assert pred.unknown_labels == ["XX"]
    for s in pred.spans:
        assert text[s.start : s.end].strip()
