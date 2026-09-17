"""
The three text retrievers.

Each exposes the same two functions — `search(claim_id, query)` and
`search_many(claim_id, queries)` — plus `available()`, and each returns a
list of `EvidenceCandidate` objects, or an empty list when its key is
missing or its call failed. None of them raises, and none of them decides
what the evidence means.
"""

from . import factcheck, seed_index, web

#  name -> module, in the order the collector runs them
RETRIEVERS = {
    "factcheck": factcheck,
    "seed_index": seed_index,
    "web": web,
}

__all__ = ["factcheck", "seed_index", "web", "RETRIEVERS"]
