"""
Compatibility shim: the evidence graph moved to `backend/graph/`.

An earlier iteration of this project kept a pydantic evidence graph here,
alongside retrieval. The two are different jobs — one is a data
structure, the other talks to the internet — and the graph outgrew
pydantic once it needed traversal (shared entities, timelines, duplicate
images), so it is now a NetworkX multigraph in `backend.graph`.

This module exists so that an import written against the old layout still
resolves. New code should import from `backend.graph`:

    from backend.graph import EvidenceGraph

The old `Evidence` / `EvidenceSource` / `EvidenceRelation` models are not
re-exported: retrieval now produces `EvidenceCandidate`
(`backend.evidence.schema`), and stance lives on graph edges rather than
in a separate relation object.
"""

import warnings

from ..graph.store import EvidenceGraph


warnings.warn(
    "backend.evidence.graph has moved to backend.graph; "
    "import EvidenceGraph from backend.graph instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["EvidenceGraph"]
