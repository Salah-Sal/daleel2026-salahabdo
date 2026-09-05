"""Unit contracts for the non-generative, non-DSPy encoder baseline."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from daleel.constants import LABELS
from daleel.encoder_baseline import (
    NONE_LABEL,
    ParagraphCollator,
    ParagraphEncoderClassifier,
    ParagraphExample,
    SEGMENT_LABELS,
    SegmentCollator,
    SegmentEncoderClassifier,
    SegmentExample,
    binary_f1,
    build_paragraph_examples,
    build_segment_examples,
    cross_fitted_task1_predictions,
    render_segment_input,
    score_records,
    segment_prediction_map,
    task1_prediction_map,
    task1_records,
    task2_records,
    tune_binary_threshold,
    tune_task1_thresholds,
)
from daleel.metrics import Span


class FakeTokenizer:
    """Small deterministic tokenizer implementing the two collator contracts."""

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            max_length = kwargs["max_length"]
            stride = kwargs["stride"]
            payload = list(range(10, 10 + len(text.split())))
            width = max_length - 2
            windows = []
            cursor = 0
            while cursor < max(1, len(payload)):
                body = payload[cursor : cursor + width]
                windows.append([101, *body, 102])
                if cursor + width >= len(payload):
                    break
                cursor += width - stride
            return {
                "input_ids": windows,
                "attention_mask": [[1] * len(window) for window in windows],
                "token_type_ids": [[0] * len(window) for window in windows],
            }

        rows = []
        for value in text:
            length = min(kwargs["max_length"], max(2, len(value.split()) + 2))
            rows.append([101, *([7] * (length - 2)), 102])
        padded = self.pad(
            [{"input_ids": row, "attention_mask": [1] * len(row)} for row in rows],
            pad_to_multiple_of=kwargs.get("pad_to_multiple_of"),
            return_tensors=kwargs.get("return_tensors"),
        )
        return padded

    def pad(
        self,
        rows,
        *,
        padding=True,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        assert padding is True
        length = max(len(row["input_ids"]) for row in rows)
        if pad_to_multiple_of:
            length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        output = {}
        for key in rows[0]:
            values = []
            for row in rows:
                pad_value = 0
                values.append(row[key] + [pad_value] * (length - len(row[key])))
            output[key] = torch.tensor(values, dtype=torch.long)
        return output


class FakeEncoder(nn.Module):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        del attention_mask, token_type_ids
        values = input_ids.float()
        hidden = torch.stack((values, values + 1.0, values + 2.0), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def paragraph(pid=1, text="واحد اثنان ثلاثة", labels=(1, 0, 0, 0, 0, 0)):
    return ParagraphExample(pid, text, "editorial", labels)


def segment(pid=1, start=0, end=4, label="AS"):
    text = "دليل ثم نتيجة"
    return SegmentExample(
        paragraph_id=pid,
        paragraph_text=text,
        genre="debate",
        start=start,
        end=end,
        text=text[start:end],
        label=label,
        position_bucket=0,
    )


def test_example_validation_and_builders_preserve_source_order():
    with pytest.raises(ValueError, match="expected 6 targets"):
        ParagraphExample(1, "نص", "editorial", (1.0,))
    with pytest.raises(ValueError, match="unknown paragraph genre"):
        paragraph(labels=(0,) * 6).__class__(1, "نص", "unknown", (0,) * 6)
    with pytest.raises(ValueError, match="does not match source"):
        SegmentExample(1, "abcd", "debate", 0, 2, "zz", "AS", 0)

    records = [
        {"paragraph_id": 2, "text": "ب", "type": "debate"},
        {"paragraph_id": 1, "text": "أ", "type": "editorial"},
    ]
    built = build_paragraph_examples(records, {1: {"AS"}, 2: {"OT"}}, {1, 2})
    assert [example.paragraph_id for example in built] == [2, 1]
    assert built[0].labels[LABELS.index("OT")] == 1.0
    with pytest.raises(ValueError, match="missing requested IDs"):
        build_paragraph_examples(records[:1], {1: set(), 2: set()}, {1, 2})


def test_segment_builder_offsets_and_rendered_context_are_source_bound():
    text = "دليل، ولذلك نتيجة."
    records = [{"paragraph_id": 7, "text": text, "type": "editorial"}]
    examples = build_segment_examples(
        records,
        {7: [Span(0, len(text), "AS")]},
        {7},
        granularity="connective",
    )
    assert examples
    assert all(text[item.start : item.end] == item.text for item in examples)
    rendered = render_segment_input(examples[0], context_chars=5)
    assert rendered.startswith(examples[0].text)
    assert "<المقطع>" in rendered
    assert "نوع النص:" in rendered
    with pytest.raises(ValueError, match="non-negative"):
        render_segment_input(examples[0], context_chars=-1)


def test_paragraph_collator_flattens_overflow_windows_and_tracks_owners():
    collator = ParagraphCollator(
        FakeTokenizer(), max_length=8, stride=2, pad_to_multiple_of=None
    )
    batch = collator(
        [
            paragraph(1, "أ ب ج د هـ و ز ح ط ي"),
            paragraph(2, "أ ب"),
        ]
    )
    assert batch["paragraph_ids"] == [1, 2]
    assert batch["window_counts"] == [2, 1]
    assert batch["window_to_paragraph"].tolist() == [0, 0, 1]
    assert tuple(batch["labels"].shape) == (2, len(LABELS))
    assert tuple(batch["inputs"]["input_ids"].shape) == (3, 8)
    with pytest.raises(ValueError, match="empty paragraph batch"):
        collator([])


def test_segment_collator_emits_indices_and_exact_examples():
    examples = [segment(label="AS"), segment(start=5, end=8, label=NONE_LABEL)]
    batch = SegmentCollator(
        FakeTokenizer(), max_length=32, context_chars=4, pad_to_multiple_of=8
    )(examples)
    assert batch["labels"].tolist() == [SEGMENT_LABELS.index("AS"), SEGMENT_LABELS.index(NONE_LABEL)]
    assert batch["examples"] == examples
    assert batch["inputs"]["input_ids"].shape[1] % 8 == 0


def test_encoder_heads_have_expected_shapes_and_window_pooling():
    inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
        "attention_mask": torch.ones((3, 2), dtype=torch.long),
    }
    owners = torch.tensor([0, 0, 1])
    paragraph_model = ParagraphEncoderClassifier(
        FakeEncoder(), hidden_size=3, dropout=0.0, window_pool="max"
    )
    paragraph_logits = paragraph_model(inputs, owners, n_paragraphs=2)
    assert tuple(paragraph_logits.shape) == (2, len(LABELS))

    segment_model = SegmentEncoderClassifier(
        FakeEncoder(), hidden_size=3, dropout=0.0
    )
    assert tuple(segment_model(inputs).shape) == (3, len(SEGMENT_LABELS))
    with pytest.raises(ValueError, match="no encoded window"):
        paragraph_model(inputs, torch.tensor([0, 0, 0]), n_paragraphs=2)


def test_threshold_tuning_is_exact_and_conservative_on_ties():
    threshold, score = tune_binary_threshold([0, 1, 1], [0.1, 0.2, 0.9])
    assert threshold == pytest.approx(0.2)
    assert score == 1.0
    all_negative_threshold, all_negative_f1 = tune_binary_threshold(
        [0, 0], [0.1, 0.2]
    )
    assert all_negative_threshold > 0.2
    assert all_negative_f1 == 0.0
    assert binary_f1(np.array([1, 0]), np.array([1, 1])) == pytest.approx(2 / 3)


def test_task1_thresholds_and_cross_fit_do_not_use_heldout_fold_labels():
    gold = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
        ]
    )
    scores = np.array(
        [
            [0.8, 0.8, 0.2, 0.1, 0.7, 0.2],
            [0.2, 0.9, 0.8, 0.2, 0.3, 0.7],
            [0.7, 0.7, 0.1, 0.8, 0.2, 0.1],
            [0.1, 0.9, 0.7, 0.1, 0.8, 0.2],
        ]
    )
    thresholds, per_label = tune_task1_thresholds(gold, scores)
    assert thresholds.shape == (len(LABELS),)
    assert set(per_label) == set(LABELS)
    decisions, by_fold = cross_fitted_task1_predictions(gold, scores, [0, 0, 1, 1])
    assert decisions.shape == gold.shape
    assert set(by_fold) == {0, 1}
    with pytest.raises(ValueError, match="at least two folds"):
        cross_fitted_task1_predictions(gold, scores, [0, 0, 0, 0])


def test_prediction_and_record_conversions_are_deterministic():
    decisions = np.zeros((2, len(LABELS)), dtype=bool)
    decisions[0, [1, 5]] = True
    mapping = task1_prediction_map([2, 1], decisions)
    assert mapping == {2: {"AS", "OT"}, 1: set()}

    examples = [segment(2, label="AS"), segment(2, label="AS")]
    spans = segment_prediction_map(
        examples,
        [SEGMENT_LABELS.index("AS"), SEGMENT_LABELS.index("AS")],
        [1, 2],
    )
    assert spans[1] == []
    assert spans[2] == [Span(0, 4, "AS")]

    source = {
        1: {"paragraph_id": 1, "text": "أ", "type": "editorial"},
        2: {"paragraph_id": 2, "text": "دليل ثم نتيجة", "type": "debate"},
    }
    rows1 = task1_records(source, mapping, [2, 1])
    assert rows1[0]["labels"] == ["AS", "OT"]
    rows2 = task2_records(source, spans, [1, 2])
    assert rows2[1]["labels"] == [
        {"label": "AS", "start_offset": 0, "end_offset": 4}
    ]
    serialized_scores = score_records([1], np.zeros((1, len(LABELS))))
    assert list(serialized_scores[0]["scores"]) == list(LABELS)

