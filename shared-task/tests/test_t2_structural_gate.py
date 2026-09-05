import importlib.util
import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

spec = importlib.util.spec_from_file_location(
    "t2_structural_gate", SCRIPTS / "t2_structural_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

from daleel.metrics import Span

L = gate.LABELS


def uniform():
    return {c: 1.0 / len(L) for c in L}


def onehotish(label, p=0.9):
    rest = (1.0 - p) / (len(L) - 1)
    return {c: p if c == label else rest for c in L}


def flat_model():
    lp = math.log(1.0 / len(L))
    start = {c: lp for c in L}
    trans = {a: {b: lp for b in L} for a in L}
    return start, trans


def test_viterbi_follows_emissions_under_flat_transitions():
    start, trans = flat_model()
    emissions = [onehotish("AS"), onehotish("AN"), onehotish("TE")]
    assert gate.viterbi(emissions, start, trans) == ["AS", "AN", "TE"]


def test_viterbi_transitions_break_emission_ties():
    start, trans = flat_model()
    # strongly prefer staying in AN once entered
    for b in L:
        trans["AN"][b] = math.log(0.02)
    trans["AN"]["AN"] = math.log(0.9)
    emissions = [onehotish("AN"), uniform(), onehotish("TE")]
    seq = gate.viterbi(emissions, start, trans)
    assert seq[0] == "AN" and seq[1] == "AN"


def test_span_encoder_dist_weights_by_overlap():
    segs = [(0, 10, onehotish("AS", 1.0 - 1e-9)),
            (10, 40, onehotish("AN", 1.0 - 1e-9))]
    dist = gate.span_encoder_dist(segs, Span(0, 40, "CO"))
    assert dist["AN"] > dist["AS"] > 0
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_span_encoder_dist_uniform_when_no_overlap():
    dist = gate.span_encoder_dist([(0, 5, uniform())], Span(100, 120, "CO"))
    assert all(abs(v - 1.0 / len(L)) < 1e-9 for v in dist.values())


def test_calibrate_renormalizes():
    weights = {c: (2.0 if c == "AN" else 1.0) for c in L}
    out = gate.calibrate(uniform(), weights)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["AN"] > out["AS"]


def test_rule_a_flips_only_unconfirmed_co():
    spans = [Span(0, 10, "CO"), Span(10, 20, "CO"), Span(20, 30, "AS")]
    quote = [Span(0, 10, "TE"), Span(10, 20, "CO")]
    out, changed = gate.rule_a(spans, quote)
    assert [s.label for s in out] == ["TE", "CO", "AS"]
    assert changed == 1


def test_rule_a_keeps_co_without_quote_overlap():
    out, changed = gate.rule_a([Span(0, 10, "CO")], [])
    assert [s.label for s in out] == ["CO"] and changed == 0


def test_dedup_spans_drops_exact_duplicates_only():
    spans = [Span(0, 10, "TE"), Span(0, 10, "TE"), Span(0, 10, "AS"),
             Span(10, 20, "TE")]
    out = gate.dedup_spans(spans)
    assert [(s.start, s.end, s.label) for s in out] == [
        (0, 10, "TE"), (0, 10, "AS"), (10, 20, "TE")]


def test_fit_sequence_model_prefers_observed_transitions():
    gold = {1: [Span(0, 5, "AN"), Span(5, 9, "AN"), Span(9, 12, "AS")]}
    model = gate.fit_sequence_model([1], gold, {1: "debate"})
    start_lp, trans_lp = model["debate"]
    assert start_lp["AN"] > start_lp["AS"]
    assert trans_lp["AN"]["AN"] > trans_lp["AN"]["TE"]
    assert "__global__" in model
