"""No-network tests for frozen discourse forests and structural controls."""

import copy

import pytest

from daleel.artifacts import canonical_json_sha256
from daleel.candidates import atomize_spans
from daleel.discourse_forest import (
    DiscourseForest,
    ForestEdge,
    build_forest_artifact,
    forest_nodes,
    forest_record,
    load_forest_artifact,
    render_structural_context,
    shuffled_edges,
    validate_forest_edges,
)
from daleel.metrics import Span
from daleel.provenance import ComplianceError


def _forest():
    text = "مقدمة واضحة، لأن المثال يشرحها، لكن النتيجة مختلفة."
    proposals = [Span(0, len(text), "AS")]
    nodes = forest_nodes(atomize_spans(text, proposals))
    assert len(nodes) >= 3
    edges, stats = validate_forest_edges(
        nodes,
        [
            {"parent": nodes[0].node_id, "child": nodes[1].node_id,
             "relation": "elaboration"},
            {"parent": nodes[0].node_id, "child": nodes[2].node_id,
             "relation": "contrast"},
        ],
    )
    assert stats["dropped_edges"] == 0
    return text, proposals, DiscourseForest(1, nodes, edges)


def test_edge_validation_drops_invalid_second_parent_cycle_and_duplicate():
    _, _, forest = _forest()
    nodes = forest.nodes
    raw = [
        {"parent": nodes[0].node_id, "child": nodes[1].node_id,
         "relation": "ELABORATION"},
        {"parent": nodes[0].node_id, "child": nodes[1].node_id,
         "relation": "ELABORATION"},
        {"parent": nodes[2].node_id, "child": nodes[1].node_id,
         "relation": "CONTRAST"},
        {"parent": nodes[1].node_id, "child": nodes[0].node_id,
         "relation": "CAUSE_REASON"},
        {"parent": nodes[0].node_id, "child": nodes[0].node_id,
         "relation": "SEQUENCE"},
        {"parent": "BAD", "child": nodes[2].node_id, "relation": "SEQUENCE"},
        {"parent": nodes[0].node_id, "child": nodes[2].node_id,
         "relation": "ADU_AS"},
    ]
    edges, stats = validate_forest_edges(nodes, raw)
    assert edges == (ForestEdge(nodes[0].node_id, nodes[1].node_id, "ELABORATION"),)
    assert stats["dropped_by_reason"] == {
        "invalid_endpoint": 1,
        "self_edge": 1,
        "invalid_relation": 1,
        "duplicate": 1,
        "second_parent": 1,
        "cycle": 1,
    }
    assert stats["unknown_relations"] == ["ADU_AS"]


def test_flat_forest_and_shuffled_render_distinct_controls():
    _, _, forest = _forest()
    target = forest.nodes[1].node_id
    flat = render_structural_context(forest, target, "flat")
    related = render_structural_context(forest, target, "forest")
    shuffled = render_structural_context(forest, target, "shuffled")
    assert "WITHHELD_FOR_FLAT_CONTROL" in flat
    assert "PARENT relation=ELABORATION" in related
    assert shuffled != related
    assert "<TARGET>" not in flat + related + shuffled
    # The unchanged marked source remains a separate role-signature input.
    assert forest.nodes[1].text in related


def test_shuffle_preserves_graph_statistics_and_is_deterministic():
    _, _, forest = _forest()
    first = shuffled_edges(forest, seed=9)
    second = shuffled_edges(forest, seed=9)
    assert first == second
    assert first != forest.edges
    assert sorted(edge.relation for edge in first) == sorted(
        edge.relation for edge in forest.edges
    )
    assert len({edge.child for edge in first}) == len(first)
    validated, stats = validate_forest_edges(forest.nodes, first)
    assert validated == tuple(sorted(
        first,
        key=lambda edge: (edge.parent, edge.child,
                          ["ELABORATION", "CAUSE_REASON", "CONTRAST", "ATTRIBUTION",
                           "SEQUENCE", "META_COMMENT", "OTHER_COHERENCE"].index(edge.relation)),
    ))
    assert stats["dropped_edges"] == 0


def test_frozen_artifact_revalidates_exact_nodes_edges_and_contract():
    text, proposals, forest = _forest()
    record = {"paragraph_id": 1, "text": text, "type": "editorial"}
    contract = {
        "proposal_sha256": "a" * 64,
        "granularity": "connective",
        "source_sha256": "b" * 64,
    }
    payload = build_forest_artifact(
        contract=contract,
        records=[
            forest_record(
                forest,
                validation={"accepted_edges": len(forest.edges), "dropped_edges": 0},
                parser_status="ok",
            )
        ],
        telemetry={"parser_calls": 1},
    )
    assert payload["contract_sha256"] == canonical_json_sha256(contract)
    loaded, loaded_contract = load_forest_artifact(
        payload,
        source_records=[record],
        proposals={1: proposals},
        expected_proposal_sha256="a" * 64,
        expected_granularity="connective",
    )
    assert loaded[1] == forest
    assert loaded_contract == contract

    tampered = copy.deepcopy(payload)
    tampered["records"][0]["nodes"][0]["text"] += "تلاعب"
    with pytest.raises(ComplianceError, match="atomization"):
        load_forest_artifact(
            tampered,
            source_records=[record],
            proposals={1: proposals},
            expected_proposal_sha256="a" * 64,
            expected_granularity="connective",
        )

    wrong_lineage = copy.deepcopy(payload)
    wrong_lineage["contract"]["proposal_sha256"] = "c" * 64
    wrong_lineage["contract_sha256"] = canonical_json_sha256(wrong_lineage["contract"])
    with pytest.raises(ComplianceError, match="proposal SHA-256"):
        load_forest_artifact(
            wrong_lineage,
            source_records=[record],
            proposals={1: proposals},
            expected_proposal_sha256="a" * 64,
            expected_granularity="connective",
        )
