"""
Stage 3b entry point: run every retriever for every check-worthy claim.

    collect_evidence(claimset, graph) -> CollectionReport

For each check-worthy claim this builds its queries (`queries.py`), asks
all three retrievers, normalises and dedupes what comes back
(`normalize.py`), fetches full text for the most promising few
(`fetch.py`), and writes the survivors into the graph with
`graph.add_evidence` — which creates the evidence node and its
FROM_SOURCE and PUBLISHED_ON edges. Each claim's `evidence_ids` is filled
in as well, so a serialised `ClaimSet` still knows what was found for it
without the graph.

Retrieval runs in a thread pool, because the whole stage is IO-bound: a
dozen claims times three retrievers is a lot of waiting, and the
embedding-based seed index is the only part that uses the CPU. Threads
(not processes) because the work is HTTP and the shared cache and model
are both in this process.

Nothing raises. A retriever that fails contributes nothing and is
recorded in the report's `errors`; the pipeline continues on whatever the
others found. Only check-worthy claims are searched — stage 2 already
decided that a greeting is not worth an API call.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, NamedTuple

from . import fetch, normalize, queries as query_builder
from .retrievers import RETRIEVERS


log = logging.getLogger(__name__)


MAX_WORKERS = 8
MAX_PER_CLAIM = normalize.MAX_PER_CLAIM
FETCH_TOP = fetch.MAX_FETCH


class CollectionReport(NamedTuple):
    """
    What one retrieval pass did — for logs, tests and the API response.

    `by_retriever` counts what each source contributed *before* dedupe, so
    a retriever that is returning nothing is visible even when the others
    made up for it.
    """

    claims: int
    searched: int
    candidates: int
    kept: int
    decisive: int
    by_retriever: dict
    errors: List[str]

    def as_dict(self):
        return {
            "claims": self.claims,
            "searched": self.searched,
            "candidates": self.candidates,
            "kept": self.kept,
            "decisive": self.decisive,
            "by_retriever": dict(self.by_retriever),
            "errors": list(self.errors),
        }


def _run_retriever(name, module, claim_id, claim_queries):
    """One retriever's results for one claim, or [] plus an error string."""
    try:
        if not module.available():
            log.debug("retriever %s is unavailable; skipping", name)
            return name, [], None

        return name, module.search_many(claim_id, claim_queries), None
    except Exception as error:                       # fail soft, never raise
        message = f"{name} failed for {claim_id}: {type(error).__name__}: {error}"
        log.warning(message)

        return name, [], message


def retrieve_for_claim(claim, max_queries=query_builder.MAX_QUERIES,
                       retrievers=None, workers=MAX_WORKERS):
    """
    Every candidate any retriever found for one claim, normalised.

    Returns (candidates, per-retriever counts, errors). Nothing is
    written to a graph here, so this is also the function to call when
    you want to look at raw retrieval without building one.
    """
    retrievers = retrievers or RETRIEVERS
    claim_queries = query_builder.build_queries(claim, max_queries=max_queries)

    if not claim_queries:
        return [], {}, []

    found = []
    counts = {}
    errors = []

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(retrievers)))) as pool:
        futures = [
            pool.submit(_run_retriever, name, module, claim.id, claim_queries)
            for name, module in retrievers.items()
        ]

        for future in futures:
            name, candidates, error = future.result()

            counts[name] = counts.get(name, 0) + len(candidates)
            found.extend(candidates)

            if error:
                errors.append(error)

    candidates = normalize.normalize_candidates(
        found, claim_text=claim.text, cap=MAX_PER_CLAIM
    )

    return candidates, counts, errors


def collect_evidence(claimset, graph=None, max_queries=query_builder.MAX_QUERIES,
                     fetch_top=FETCH_TOP, retrievers=None, workers=MAX_WORKERS):
    """
    Retrieve evidence for every check-worthy claim and write it to `graph`.

    `graph` is optional: without one this still fills each claim's
    `evidence_ids` and returns the report, which is what the retrieval
    smoke script wants. Candidates arrive with `stance=None` — deciding
    what they mean is stage 4's job.
    """
    claims = [claim for claim in claimset.claims if claim.check_worthy]

    report = {
        "claims": len(claimset.claims),
        "searched": len(claims),
        "candidates": 0,
        "kept": 0,
        "decisive": 0,
        "by_retriever": {},
        "errors": [],
    }

    if not claims:
        return CollectionReport(**report)

    # Claims are searched in parallel too: with three retrievers each
    # already running concurrently, the pool is mostly waiting on sockets.
    with ThreadPoolExecutor(max_workers=min(workers, len(claims))) as pool:
        results = list(
            pool.map(
                lambda claim: (
                    claim,
                    *retrieve_for_claim(
                        claim,
                        max_queries=max_queries,
                        retrievers=retrievers,
                        workers=workers,
                    ),
                ),
                claims,
            )
        )

    for claim, candidates, counts, errors in results:
        for name, count in counts.items():
            report["by_retriever"][name] = report["by_retriever"].get(name, 0) + count

        report["errors"].extend(errors)
        report["candidates"] += sum(counts.values())

        if fetch_top:
            fetch.enrich(candidates, limit=fetch_top)

        evidence_ids = []

        for candidate in candidates:
            if graph is not None:
                try:
                    graph.add_evidence(claim.id, candidate)
                except Exception as error:           # fail soft, never raise
                    message = (
                        f"could not add {candidate.id} to the graph: "
                        f"{type(error).__name__}: {error}"
                    )
                    log.warning(message)
                    report["errors"].append(message)
                    continue

            evidence_ids.append(candidate.id)

            if candidate.decisive:
                report["decisive"] += 1

        claim.evidence_ids = evidence_ids
        report["kept"] += len(evidence_ids)

    log.info(
        "evidence: %s candidates over %s claims -> %s kept (%s decisive) via %s",
        report["candidates"], report["searched"], report["kept"],
        report["decisive"], report["by_retriever"],
    )

    return CollectionReport(**report)
