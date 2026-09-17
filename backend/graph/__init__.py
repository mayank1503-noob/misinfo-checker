"""
Stage 3a — the evidence graph.

One NetworkX multigraph per packet, holding claims, entities, images,
retrieved evidence, sources, dates and verdicts. Built from stage 2's
`ClaimSet`, grown by stages 3b / 4 / 5, read by the verdict.
"""

from .schema import (
    EDGE_TYPES,
    NODE_TYPES,
    STANCE_EDGES,
    STANCES,
    date_id,
    image_id,
    normalize_stance,
    source_id,
    verdict_id,
)
from .store import DATE_MISMATCH_DAYS, EvidenceGraph

__all__ = [
    "EvidenceGraph",
    "NODE_TYPES",
    "EDGE_TYPES",
    "STANCES",
    "STANCE_EDGES",
    "DATE_MISMATCH_DAYS",
    "normalize_stance",
    "image_id",
    "source_id",
    "date_id",
    "verdict_id",
]
