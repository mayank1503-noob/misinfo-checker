"""
Stage 3b — text evidence retrieval.

Finds things that bear on each check-worthy claim, normalises them into
`EvidenceCandidate`s, and writes them into the evidence graph. It does
not decide what any of it *means*: every candidate leaves here with
`stance=None`, and stage 4 (`backend.stance`) reads the text and judges.

    from backend.evidence import collect_evidence

    report = collect_evidence(claimset, graph)

Runs with no API keys: the local seed index answers offline, and the two
network retrievers log once and return nothing when their key is unset.
"""

from .collector import CollectionReport, collect_evidence, retrieve_for_claim
from .normalize import normalize_candidates, normalize_rating, tier_for, weight_for
from .queries import Query, build_queries, queries_for
from .translit import devanagari, looks_romanised
from .schema import (
    RATINGS,
    SOURCE_TYPES,
    EvidenceCandidate,
    canonical_url,
    domain_of,
)

__all__ = [
    "EvidenceCandidate",
    "SOURCE_TYPES",
    "RATINGS",
    "canonical_url",
    "domain_of",
    "Query",
    "build_queries",
    "queries_for",
    "devanagari",
    "looks_romanised",
    "normalize_candidates",
    "normalize_rating",
    "tier_for",
    "weight_for",
    "collect_evidence",
    "retrieve_for_claim",
    "CollectionReport",
]
