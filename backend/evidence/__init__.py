"""
Stage 3A — the evidence graph.

Grows the packet / claim / entity skeleton produced by
`ClaimSet.to_graph_seed()` with evidence nodes and stance edges. Data
layer only: retrieval (3B) and the verdict (4) live elsewhere.
"""

from .graph import EvidenceGraph
from .schema import (
    EDGE_TYPES,
    EVIDENCE_TYPES,
    NODE_KINDS,
    RELATION_EDGES,
    RELATIONS,
    Edge,
    Evidence,
    EvidenceRelation,
    EvidenceSource,
    Node,
    evidence_id,
)

__all__ = [
    "EvidenceGraph",
    "Evidence",
    "EvidenceSource",
    "EvidenceRelation",
    "Node",
    "Edge",
    "evidence_id",
    "NODE_KINDS",
    "EDGE_TYPES",
    "EVIDENCE_TYPES",
    "RELATIONS",
    "RELATION_EDGES",
]
