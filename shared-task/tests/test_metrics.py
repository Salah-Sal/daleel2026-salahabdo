import pytest

from daleel.metrics import Span, span_partial_f1, task1_macro_f1


def test_task1_perfect_prediction_all_labels_present():
    gold = {1: {"AS", "TE"}, 2: {"CO", "ST"}, 3: {"AN", "OT"}}
    result = task1_macro_f1(gold, gold)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["micro_f1"] == pytest.approx(1.0)


def test_task1_absent_labels_still_count_in_macro():
    # Official convention (zero_division=0): labels missing from both gold
    # and pred score F1=0 and stay in the macro — perfect prediction on a
    # slice without all six labels is NOT 1.0.
    gold = {1: {"AS"}}
    result = task1_macro_f1(gold, gold)
    assert result["macro_f1"] == pytest.approx(1 / 6)


def test_task1_by_hand():
    gold = {1: {"AS"}, 2: {"AS", "ST"}}
    pred = {1: {"AS", "ST"}, 2: {"AS"}}
    result = task1_macro_f1(gold, pred)
    # AS: perfect. ST: tp=0, fp=1, fn=1 -> f1=0. Other four: absent -> 0.
    assert result["per_label"]["AS"]["f1"] == pytest.approx(1.0)
    assert result["per_label"]["ST"]["f1"] == 0.0
    assert result["per_label"]["ST"]["support"] == 1
    assert result["macro_f1"] == pytest.approx(1 / 6)


def test_task1_missing_paragraph_counts_against_recall():
    gold = {1: {"AS"}, 2: {"AS"}}
    pred = {1: {"AS"}}  # paragraph 2 absent
    result = task1_macro_f1(gold, pred)
    assert result["per_label"]["AS"]["recall"] == pytest.approx(0.5)
    assert result["per_label"]["AS"]["precision"] == pytest.approx(1.0)


def test_task1_predictions_for_unknown_ids_are_ignored():
    # Official scorer iterates over gold ids only.
    gold = {1: {"AS"}}
    pred = {1: {"AS"}, 99: {"ST", "AS"}}
    result = task1_macro_f1(gold, pred)
    assert result["per_label"]["AS"]["precision"] == pytest.approx(1.0)
    assert result["per_label"]["ST"]["f1"] == 0.0


def test_span_rejects_empty():
    with pytest.raises(ValueError):
        Span(5, 5, "AS")


def test_span_perfect_and_disjoint():
    gold = {1: [Span(0, 10, "AS")]}
    assert span_partial_f1(gold, gold)["f1"] == pytest.approx(1.0)
    assert span_partial_f1(gold, {1: [Span(20, 30, "AS")]})["f1"] == 0.0


def test_span_partial_overlap_by_hand():
    gold = {1: [Span(0, 10, "AS")]}
    pred = {1: [Span(5, 10, "AS")]}
    result = span_partial_f1(gold, pred)
    # precision: overlap 5 / pred length 5 = 1.0; recall: 5 / 10 = 0.5
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["per_label"]["AS"]["f1"] == pytest.approx(2 / 3)


def test_span_label_mismatch_earns_nothing():
    gold = {1: [Span(0, 10, "AS")]}
    pred = {1: [Span(0, 10, "TE")]}
    assert span_partial_f1(gold, pred)["f1"] == 0.0


def test_span_credit_sums_over_all_pairs_not_best_match():
    # One pred span covering two same-label gold spans earns credit twice.
    gold = {1: [Span(0, 10, "AS"), Span(10, 20, "AS")]}
    pred = {1: [Span(0, 20, "AS")]}
    result = span_partial_f1(gold, pred)
    # precision: 10/20 + 10/20 = 1.0 over 1 pred span
    # recall: 10/10 + 10/10 = 2.0 over 2 gold spans
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_span_duplicate_predictions_inflate_recall():
    # Documented quirk of the official formula: recall can exceed 1.0.
    # Keep predictions non-overlapping; do not exploit this.
    gold = {1: [Span(0, 10, "AS")]}
    pred = {1: [Span(0, 10, "AS"), Span(0, 10, "AS")]}
    result = span_partial_f1(gold, pred)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(2.0)


def test_span_unknown_gold_paragraph_hits_precision_only():
    gold = {1: [Span(0, 10, "AS")]}
    pred = {1: [Span(0, 10, "AS")], 99: [Span(0, 10, "AS")]}
    result = span_partial_f1(gold, pred)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
