"""
Stage 4's one connection to the evidence graph.

`classify.py` is deliberately pure: it takes claims and flat evidence
text and returns `StanceResult`s. Something still has to read the graph,
hand it that text, and write the answers back as SUPPORTS / REFUTES /
NEUTRAL edges. That is this module, and keeping it separate is what lets
the stance logic be tested with no graph at all.

    from backend.stance.graph_adapter import apply_stances

    report = apply_stances(graph)          # every claim with evidence

Two details worth knowing:

  * **The rating passed to stage 4 is the publisher's own words**
    (`rating_raw`), not the normalised one. Stage 3b's normalisation is a
    lossy summary — "Pants on Fire" and "Fake" both become `false` — and
    the stance layer's own rating table is better at the original. The
    normalised value is the fallback for candidates that only have it.

  * **Vocabularies differ on purpose.** Stage 4 says `support`, the graph
    says `supports`; `graph.set_stance` normalises, so neither side had to
    change to suit the other (DECISIONS.md D6).

Nothing here raises. Stage 4 already fails soft to neutral, and a write
that the graph rejects is logged and counted rather than thrown.
"""

import logging
from typing import List, NamedTuple

from .classify import classify_stance
from .rank import RELEVANCE_CUTOFF, TOP_K


log = logging.getLogger(__name__)


class StanceReport(NamedTuple):
    """
    What one stance pass decided — for logs, tests and the API response.

    `misleading` counts the case this stage exists for: evidence whose
    text reads as support because it quotes the claim, while its
    publisher's rating refutes it.
    """

    claims: int
    evidence: int
    supports: int
    refutes: int
    neutral: int
    misleading: int
    by_method: dict
    errors: List[str]

    def as_dict(self):
        return {
            "claims": self.claims,
            "evidence": self.evidence,
            "supports": self.supports,
            "refutes": self.refutes,
            "neutral": self.neutral,
            "misleading": self.misleading,
            "by_method": dict(self.by_method),
            "errors": list(self.errors),
        }


# graph stance name -> the report field that counts it
_COUNTER = {"supports": "supports", "refutes": "refutes", "neutral": "neutral"}


def _evidence_inputs(graph, claim_id):
    """The flat text shape stage 4 consumes, read off the graph."""
    items = []

    for item in graph.evidence_for(claim_id):
        items.append(
            {
                "id": item["id"],
                "claim_id": claim_id,
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "text": item.get("text"),
                # The publisher's exact verdict where there is one; the
                # normalised value only as a fallback.
                "rating": item.get("rating_raw") or item.get("rating"),
                "weight": float(item.get("source_weight") or 0.5),
            }
        )

    return items


def apply_stances(graph, claim_ids=None, top_k=TOP_K, cutoff=RELEVANCE_CUTOFF):
    """
    Decide what every piece of evidence in the graph says about its claim,
    and write it back as a stance edge.

    Claims are processed one at a time because `classify_stance` batches
    per claim already — one embedding call and one NLI call per claim,
    however many documents it has. `claim_ids` narrows the pass to a
    subset (a second retrieval round only needs to re-score what it added).
    """
    claim_nodes = dict(graph.claims())

    if claim_ids is not None:
        wanted = set(claim_ids)
        claim_nodes = {cid: attrs for cid, attrs in claim_nodes.items() if cid in wanted}

    report = {
        "claims": 0,
        "evidence": 0,
        "supports": 0,
        "refutes": 0,
        "neutral": 0,
        "misleading": 0,
        "by_method": {},
        "errors": [],
    }

    for claim_id, attrs in claim_nodes.items():
        evidence = _evidence_inputs(graph, claim_id)

        if not evidence:
            continue

        report["claims"] += 1

        claim = {"id": claim_id, "text": attrs.get("text") or ""}

        try:
            results = classify_stance([claim], evidence, top_k=top_k, cutoff=cutoff)
        except Exception as error:                   # fail soft, never raise
            message = (
                f"stance classification failed for {claim_id}: "
                f"{type(error).__name__}: {error}"
            )
            log.warning(message)
            report["errors"].append(message)
            continue

        for result in results:
            report["evidence"] += 1
            report["by_method"][result.method] = report["by_method"].get(result.method, 0) + 1

            if result.misleading:
                report["misleading"] += 1

            try:
                stance = graph.set_stance(
                    result.evidence_id,
                    result.claim_id,
                    result.stance,
                    score=result.score,
                    relevance=result.relevance,
                    method=result.method,
                    best_passage=result.best_passage,
                    misleading=result.misleading,
                    note=result.note,
                )
            except Exception as error:               # fail soft, never raise
                message = (
                    f"could not write the stance for {result.evidence_id}: "
                    f"{type(error).__name__}: {error}"
                )
                log.warning(message)
                report["errors"].append(message)
                continue

            report[_COUNTER[stance]] += 1

    log.info(
        "stance: %s items over %s claims -> %s supports / %s refutes / %s neutral "
        "(%s misleading) via %s",
        report["evidence"], report["claims"], report["supports"], report["refutes"],
        report["neutral"], report["misleading"], report["by_method"],
    )

    return StanceReport(**report)
