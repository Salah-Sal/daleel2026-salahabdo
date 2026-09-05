"""Validated, label-blind discourse forests over proposal-derived atoms.

The structural experiment deliberately reuses :class:`CandidateAtom` nodes.
It does not retokenize text, expose the quote extractor's draft ADU labels, or
invent offsets. A model may propose typed parent->child relations, but this
module owns validation, canonicalization, deterministic shuffling, and local
neighborhood rendering.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import canonical_json_sha256
from .candidates import CandidateAtom, atomize_spans
from .metrics import Span
from .provenance import ComplianceError


FOREST_SCHEMA_VERSION = 1
FOREST_ARCHITECTURE = "proposal-atom-discourse-forest-v1"
STRUCTURAL_ROLE_ARCHITECTURE = "proposal-atom-role-v4-structure"
STRUCTURE_MODES = ("baseline", "flat", "forest", "shuffled")
DEFAULT_SHUFFLE_SEED = 20260711

# These are discourse/coherence relations, not Daleel ADU types.
RELATIONS = (
    "ELABORATION",
    "CAUSE_REASON",
    "CONTRAST",
    "ATTRIBUTION",
    "SEQUENCE",
    "META_COMMENT",
    "OTHER_COHERENCE",
)
_RELATION_ORDER = {relation: index for index, relation in enumerate(RELATIONS)}


@dataclass(frozen=True)
class ForestNode:
    """One exact atom node in reading order."""

    node_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if not self.node_id or self.end <= self.start:
            raise ValueError(f"invalid forest node {self!r}")


@dataclass(frozen=True)
class ForestEdge:
    """A directed discourse relation from parent to child."""

    parent: str
    child: str
    relation: str


@dataclass(frozen=True)
class DiscourseForest:
    """A validated forest whose nodes exactly equal one atomization."""

    paragraph_id: object
    nodes: tuple[ForestNode, ...]
    edges: tuple[ForestEdge, ...]

    def node_map(self) -> dict[str, ForestNode]:
        return {node.node_id: node for node in self.nodes}

    def node_id_for(self, start: int, end: int) -> str:
        matches = [
            node.node_id for node in self.nodes if node.start == start and node.end == end
        ]
        if len(matches) != 1:
            raise KeyError(
                f"forest {self.paragraph_id!r} has {len(matches)} nodes for "
                f"target [{start}, {end})"
            )
        return matches[0]


def forest_nodes(atoms: Sequence[CandidateAtom]) -> tuple[ForestNode, ...]:
    """Assign stable reading-order IDs to exact atoms."""

    return tuple(
        ForestNode(f"A{index:03d}", atom.start, atom.end, atom.text)
        for index, atom in enumerate(atoms)
    )


def atom_inventory(nodes: Sequence[ForestNode]) -> list[str]:
    """Compact, exact node inventory supplied to the label-blind parser."""

    return [
        f"{node.node_id} [{node.start}:{node.end}] {node.text}" for node in nodes
    ]


def _field(item: object, name: str) -> str:
    value = item.get(name, "") if isinstance(item, Mapping) else getattr(item, name, "")
    return str(value or "").strip()


def _would_cycle(parent_by_child: Mapping[str, str], parent: str, child: str) -> bool:
    cursor = parent
    seen = {child}
    while cursor in parent_by_child:
        if cursor in seen:
            return True
        seen.add(cursor)
        cursor = parent_by_child[cursor]
    return cursor in seen


def validate_forest_edges(
    nodes: Sequence[ForestNode],
    raw_edges: Iterable[object],
) -> tuple[tuple[ForestEdge, ...], dict[str, Any]]:
    """Normalize proposed edges into a deterministic, acyclic forest.

    Invalid edges are dropped rather than repaired with invented endpoints or
    relations. At most one incoming edge is accepted per node. Competing
    parents are resolved in canonical lexical order.
    """

    node_ids = {node.node_id for node in nodes}
    candidates: list[ForestEdge] = []
    dropped = {
        "invalid_endpoint": 0,
        "self_edge": 0,
        "invalid_relation": 0,
        "duplicate": 0,
        "second_parent": 0,
        "cycle": 0,
    }
    unknown_relations: set[str] = set()
    raw_count = 0
    for item in raw_edges:
        raw_count += 1
        parent = _field(item, "parent").upper()
        child = _field(item, "child").upper()
        relation = _field(item, "relation").upper().replace("-", "_").replace(" ", "_")
        if parent not in node_ids or child not in node_ids:
            dropped["invalid_endpoint"] += 1
            continue
        if parent == child:
            dropped["self_edge"] += 1
            continue
        if relation not in _RELATION_ORDER:
            dropped["invalid_relation"] += 1
            if relation:
                unknown_relations.add(relation)
            continue
        candidates.append(ForestEdge(parent, child, relation))

    node_index = {node.node_id: index for index, node in enumerate(nodes)}
    # Prefer ordinary forward discourse attachments, then the nearest parent.
    # This is still independent of model output order, but avoids allowing a
    # later backward edge to consume a child's one-parent slot before a local
    # reading-order attachment is considered.
    candidates.sort(
        key=lambda edge: (
            node_index[edge.parent] > node_index[edge.child],
            node_index[edge.child],
            abs(node_index[edge.parent] - node_index[edge.child]),
            node_index[edge.parent],
            _RELATION_ORDER[edge.relation],
        )
    )
    accepted: list[ForestEdge] = []
    seen: set[ForestEdge] = set()
    parent_by_child: dict[str, str] = {}
    for edge in candidates:
        if edge in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(edge)
        if edge.child in parent_by_child:
            dropped["second_parent"] += 1
            continue
        if _would_cycle(parent_by_child, edge.parent, edge.child):
            dropped["cycle"] += 1
            continue
        parent_by_child[edge.child] = edge.parent
        accepted.append(edge)

    accepted.sort(
        key=lambda edge: (edge.parent, edge.child, _RELATION_ORDER[edge.relation])
    )
    stats = {
        "raw_edges": raw_count,
        "accepted_edges": len(accepted),
        "dropped_edges": sum(dropped.values()),
        "dropped_by_reason": dropped,
        "unknown_relations": sorted(unknown_relations),
        "roots": len(nodes) - len(parent_by_child),
    }
    return tuple(accepted), stats


def _stable_seed(seed: int, paragraph_id: object) -> int:
    digest = hashlib.sha256(f"{seed}:{paragraph_id!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def shuffled_edges(
    forest: DiscourseForest,
    *,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> tuple[ForestEdge, ...]:
    """Misalign a forest from text while preserving its exact graph shape."""

    ids = [node.node_id for node in forest.nodes]
    if len(ids) < 2:
        return forest.edges
    permuted = list(ids)
    rng = random.Random(_stable_seed(seed, forest.paragraph_id))
    rng.shuffle(permuted)
    if permuted == ids:
        permuted = permuted[1:] + permuted[:1]
    mapping = dict(zip(ids, permuted))
    return tuple(
        ForestEdge(mapping[edge.parent], mapping[edge.child], edge.relation)
        for edge in forest.edges
    )


def _snippet(text: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("snippet limit must be positive")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _node_line(prefix: str, node: ForestNode, *, limit: int) -> str:
    rendered = json.dumps(_snippet(node.text, limit), ensure_ascii=False)
    return f"{prefix} {node.node_id} [{node.start}:{node.end}] {rendered}"


def render_structural_context(
    forest: DiscourseForest,
    target_id: str,
    mode: str,
    *,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    sequential_radius: int = 2,
    max_relational_neighbors: int = 6,
    snippet_chars: int = 180,
) -> str:
    """Render a compact target-local flat/forest/shuffled control input."""

    if mode not in STRUCTURE_MODES or mode == "baseline":
        raise ValueError(f"structural renderer requires flat/forest/shuffled, got {mode!r}")
    if sequential_radius < 0 or max_relational_neighbors < 0:
        raise ValueError("neighborhood sizes must be non-negative")
    node_by_id = forest.node_map()
    if target_id not in node_by_id:
        raise KeyError(f"unknown target node {target_id!r}")
    order = [node.node_id for node in forest.nodes]
    target_index = order.index(target_id)
    target = node_by_id[target_id]
    # Forest and shuffled controls must be presentation-identical. Revealing
    # SHUFFLED would let the classifier consciously ignore the negative
    # control, invalidating the test of attachment correctness.
    control_name = "FLAT" if mode == "flat" else "TYPED"
    lines = [
        f"STRUCTURE_CONTROL={control_name}",
        _node_line("TARGET", target, limit=snippet_chars),
        "SEQUENTIAL_NEIGHBORHOOD:",
    ]
    sequential = []
    for index in range(
        max(0, target_index - sequential_radius),
        min(len(order), target_index + sequential_radius + 1),
    ):
        if index == target_index:
            continue
        direction = "PREVIOUS" if index < target_index else "NEXT"
        distance = abs(index - target_index)
        sequential.append(
            _node_line(f"- {direction}_{distance}", node_by_id[order[index]], limit=snippet_chars)
        )
    lines.extend(sequential or ["- NONE"])
    if mode == "flat":
        lines.append("TYPED_RELATIONS: WITHHELD_FOR_FLAT_CONTROL")
        return "\n".join(lines)

    edges = forest.edges if mode == "forest" else shuffled_edges(
        forest, seed=shuffle_seed
    )
    parent_edge = next((edge for edge in edges if edge.child == target_id), None)
    child_edges = sorted(
        (edge for edge in edges if edge.parent == target_id),
        key=lambda edge: (edge.child, _RELATION_ORDER[edge.relation]),
    )
    sibling_rows: list[tuple[ForestEdge, ForestEdge]] = []
    if parent_edge is not None:
        for sibling in edges:
            if sibling.parent == parent_edge.parent and sibling.child != target_id:
                sibling_rows.append((parent_edge, sibling))
    relation_lines: list[str] = []
    if parent_edge is not None:
        relation_lines.append(
            _node_line(
                f"- PARENT relation={parent_edge.relation}",
                node_by_id[parent_edge.parent],
                limit=snippet_chars,
            )
        )
    for edge in child_edges:
        relation_lines.append(
            _node_line(
                f"- CHILD relation={edge.relation}",
                node_by_id[edge.child],
                limit=snippet_chars,
            )
        )
    for shared_parent, sibling in sorted(
        sibling_rows,
        key=lambda pair: (pair[1].child, _RELATION_ORDER[pair[1].relation]),
    ):
        relation_lines.append(
            _node_line(
                "- SIBLING "
                f"via_parent={shared_parent.parent} "
                f"target_relation={shared_parent.relation} "
                f"sibling_relation={sibling.relation}",
                node_by_id[sibling.child],
                limit=snippet_chars,
            )
        )
    lines.append("TYPED_RELATIONAL_NEIGHBORHOOD:")
    lines.extend(relation_lines[:max_relational_neighbors] or ["- NONE"])
    if len(relation_lines) > max_relational_neighbors:
        lines.append(f"- TRUNCATED_RELATIONS={len(relation_lines) - max_relational_neighbors}")
    return "\n".join(lines)


def forest_record(
    forest: DiscourseForest,
    *,
    validation: Mapping[str, Any],
    parser_status: str,
    parser_error: str | None = None,
) -> dict[str, Any]:
    """Convert one validated forest to its frozen JSON representation."""

    return {
        "paragraph_id": forest.paragraph_id,
        "nodes": [
            {"id": node.node_id, "start": node.start, "end": node.end, "text": node.text}
            for node in forest.nodes
        ],
        "edges": [
            {"parent": edge.parent, "child": edge.child, "relation": edge.relation}
            for edge in forest.edges
        ],
        "parser_status": parser_status,
        "parser_error": parser_error,
        "validation": dict(validation),
    }


def build_forest_artifact(
    *,
    contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    contract_dict = dict(contract)
    return {
        "forest_schema_version": FOREST_SCHEMA_VERSION,
        "architecture": FOREST_ARCHITECTURE,
        "relation_inventory": list(RELATIONS),
        "contract": contract_dict,
        "contract_sha256": canonical_json_sha256(contract_dict),
        "records": list(records),
        "telemetry": dict(telemetry),
    }


def load_forest_artifact(
    payload: Mapping[str, Any],
    *,
    source_records: Sequence[Mapping[str, Any]],
    proposals: Mapping[object, Sequence[Span]],
    expected_proposal_sha256: str,
    expected_granularity: str,
) -> tuple[dict[object, DiscourseForest], dict[str, Any]]:
    """Strictly validate a frozen artifact against source text and proposals."""

    if payload.get("forest_schema_version") != FOREST_SCHEMA_VERSION:
        raise ComplianceError(
            f"unsupported forest schema {payload.get('forest_schema_version')!r}"
        )
    if payload.get("architecture") != FOREST_ARCHITECTURE:
        raise ComplianceError(f"invalid forest architecture {payload.get('architecture')!r}")
    if payload.get("relation_inventory") != list(RELATIONS):
        raise ComplianceError("forest relation inventory does not match executable schema")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ComplianceError("forest contract must be an object")
    if payload.get("contract_sha256") != canonical_json_sha256(dict(contract)):
        raise ComplianceError("forest contract SHA-256 mismatch")
    if contract.get("proposal_sha256") != expected_proposal_sha256:
        raise ComplianceError("forest proposal SHA-256 does not match requested proposals")
    if contract.get("granularity") != expected_granularity:
        raise ComplianceError(
            f"forest granularity {contract.get('granularity')!r} does not match "
            f"{expected_granularity!r}"
        )

    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ComplianceError("forest records must be a list")
    row_by_id: dict[object, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or "paragraph_id" not in row:
            raise ComplianceError("invalid forest record")
        pid = row["paragraph_id"]
        if pid in row_by_id:
            raise ComplianceError(f"duplicate forest paragraph_id {pid!r}")
        row_by_id[pid] = row

    expected_ids = {record["paragraph_id"] for record in source_records}
    if set(row_by_id) != expected_ids:
        raise ComplianceError(
            "forest paragraph IDs differ from source: "
            f"missing={sorted(expected_ids - set(row_by_id))[:20]}, "
            f"extra={sorted(set(row_by_id) - expected_ids)[:20]}"
        )

    forests: dict[object, DiscourseForest] = {}
    for source in source_records:
        pid = source["paragraph_id"]
        if pid not in proposals:
            raise ComplianceError(f"proposal map has no paragraph_id {pid!r}")
        atoms = atomize_spans(source["text"], list(proposals[pid]), expected_granularity)
        expected_nodes = forest_nodes(atoms)
        row = row_by_id[pid]
        raw_nodes = row.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ComplianceError(f"forest {pid!r} nodes must be a list")
        got_nodes: list[ForestNode] = []
        for node in raw_nodes:
            if not isinstance(node, Mapping):
                raise ComplianceError(f"forest {pid!r} contains invalid node")
            try:
                got_nodes.append(
                    ForestNode(
                        str(node["id"]),
                        int(node["start"]),
                        int(node["end"]),
                        str(node["text"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ComplianceError(f"forest {pid!r} contains invalid node: {exc}") from exc
        if tuple(got_nodes) != expected_nodes:
            raise ComplianceError(f"forest {pid!r} nodes do not match proposal atomization")
        raw_edges = row.get("edges")
        if not isinstance(raw_edges, list):
            raise ComplianceError(f"forest {pid!r} edges must be a list")
        edges, stats = validate_forest_edges(expected_nodes, raw_edges)
        canonical_raw = [
            {"parent": edge.parent, "child": edge.child, "relation": edge.relation}
            for edge in edges
        ]
        if raw_edges != canonical_raw or stats["dropped_edges"]:
            raise ComplianceError(f"forest {pid!r} stores non-canonical/invalid edges")
        forests[pid] = DiscourseForest(pid, expected_nodes, edges)
    return forests, dict(contract)


__all__ = [
    "DEFAULT_SHUFFLE_SEED",
    "DiscourseForest",
    "FOREST_ARCHITECTURE",
    "FOREST_SCHEMA_VERSION",
    "ForestEdge",
    "ForestNode",
    "RELATIONS",
    "STRUCTURAL_ROLE_ARCHITECTURE",
    "STRUCTURE_MODES",
    "atom_inventory",
    "build_forest_artifact",
    "forest_nodes",
    "forest_record",
    "load_forest_artifact",
    "render_structural_context",
    "shuffled_edges",
    "validate_forest_edges",
]
