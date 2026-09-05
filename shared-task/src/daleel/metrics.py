"""Evaluation metrics for the two Daleel 2026 subtasks.

Both functions are faithful ports of the OFFICIAL scoring scripts published
in the organizers' repository (https://github.com/Argmining/Daleel2026,
cloned at ../resources/repos/Daleel2026/evaluation/): `task1_scoring.py` and
`task2_scoring.py`. The parity tests in tests/test_official_parity.py check
this module against the official scripts directly.

Official conventions preserved here (they bite on small dev slices):

- Task 1 macro-F1 averages over ALL six labels with zero_division=0, so a
  label absent from both gold and predictions still contributes F1=0 to the
  macro. A "perfect" prediction on a slice that lacks some label does NOT
  score 1.0.
- Task 1 iterates over gold paragraph ids only; predictions for unknown ids
  are ignored (not counted as false positives).
- Task 2 credit accumulates over ALL overlapping same-label gold-pred span
  pairs, not the best match. Duplicate or overlapping predictions can push
  recall above 1.0 — treat that as a property of the official formula, keep
  predictions non-overlapping, and do not try to exploit it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from .constants import LABELS


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def task1_macro_f1(
    gold: Mapping[object, Collection[str]],
    pred: Mapping[object, Collection[str]],
    labels: Sequence[str] = LABELS,
) -> dict:
    """Label-macro F1 over paragraph label sets keyed by paragraph_id.

    Port of official task1_scoring.py (sklearn MultiLabelBinarizer +
    precision_recall_fscore_support(average="macro", zero_division=0)).
    Only paragraphs present in `gold` are scored; a paragraph missing from
    `pred` counts as predicting no labels. Every label in `labels` enters
    the macro average, with F1=0 when it never occurs. Returns
    {"macro_f1", "micro_f1", "per_label": {label: {precision, recall, f1,
    support}}}.
    """
    ids = list(gold)
    per_label = {}
    tp_all = fp_all = fn_all = 0
    for label in labels:
        tp = fp = fn = 0
        for i in ids:
            in_gold = label in gold[i]
            in_pred = label in pred.get(i, ())
            tp += in_gold and in_pred
            fp += in_pred and not in_gold
            fn += in_gold and not in_pred
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "support": tp + fn,
        }
        tp_all += tp
        fp_all += fp
        fn_all += fn
    macro = sum(v["f1"] for v in per_label.values()) / len(per_label)
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    return {"macro_f1": macro, "micro_f1": _f1(micro_p, micro_r), "per_label": per_label}


@dataclass(frozen=True)
class Span:
    """A typed character span; `end` is exclusive."""

    start: int
    end: int
    label: str

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(f"empty or negative span: [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def span_partial_f1(
    gold: Mapping[object, Sequence[Span]],
    pred: Mapping[object, Sequence[Span]],
    labels: Sequence[str] = LABELS,
) -> dict:
    """Official Task 2 partial-match span F1 (port of task2_scoring.py).

    Every overlapping same-label (pred, gold) span pair contributes
    intersection/len(pred) to the precision numerator and
    intersection/len(gold) to the recall numerator; denominators are the
    total span counts. Paragraphs in `pred` but not in `gold` contribute to
    the precision denominator only. Returns {"precision", "recall", "f1",
    "per_label": {label: {precision, recall, f1}}}.
    """
    total_pred = sum(len(v) for v in pred.values())
    total_gold = sum(len(v) for v in gold.values())
    cum_p = cum_r = 0.0
    label_p = dict.fromkeys(labels, 0.0)
    label_r = dict.fromkeys(labels, 0.0)
    for i, pred_spans in pred.items():
        if i not in gold:
            continue
        for ps in pred_spans:
            for gs in gold[i]:
                if ps.label != gs.label:
                    continue
                intersection = _overlap(ps, gs)
                if intersection == 0:
                    continue
                cum_p += intersection / len(ps)
                cum_r += intersection / len(gs)
                label_p[gs.label] += intersection / len(ps)
                label_r[gs.label] += intersection / len(gs)

    def _prf(p_num, p_den, r_num, r_den):
        p = p_num / p_den if p_den else 0.0
        r = r_num / r_den if r_den else 0.0
        return {"precision": p, "recall": r, "f1": _f1(p, r)}

    def _freq(annotations, label):
        return sum(1 for spans in annotations.values() for s in spans if s.label == label)

    per_label = {
        label: _prf(label_p[label], _freq(pred, label), label_r[label], _freq(gold, label))
        for label in labels
    }
    return _prf(cum_p, total_pred, cum_r, total_gold) | {"per_label": per_label}
