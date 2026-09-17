"""
The verification agent: put it all together and let the rules decide.

This is the agent most likely to be handed to a language model in a
system built the other way round, and it is the one where that would do
the most damage. The labels here come from `backend.verdict.decide` —
the same rule cascade the linear pipeline uses, in the same order:
decisive fact-check, then recycled media, then weighted stance, then
`unverified`. Every label comes with the evidence that produced it. A
model that could overrule that could produce a verdict nobody can
explain, which is worth less than no verdict.

So the agent's job is to assemble, not to judge:

  1. run stage 4 (`stance.apply`) over the claims this round touched —
     the evidence agent handed over exactly which ones;
  2. read the graph per claim (`graph.totals`, `graph.decisive`) so its
     structured output carries the arithmetic behind each label;
  3. fold in what the media agent found, by claim id;
  4. call `verdict.decide`, unchanged, and report what it said.

Then the part that makes it an agent rather than a function call: it
**abstains**. When the verdict comes back `unverified` — nothing decisive,
no stance either way — it returns `abstained` rather than `ok`, which is
what lets the orchestrator send the evidence agent back out for a second
round before anyone writes an explanation. The label is still
`unverified` and still returned; abstaining is about whether this is a
finished answer, not about hiding it.
"""

from .base import Agent


# Labels that mean "we have an answer". `disputed` is one of them: sources
# disagreeing is a finding, not a gap.
SETTLED = ("false", "misleading", "true", "disputed")


class VerificationAgent(Agent):
    agent = "verification"
    description = "Combine evidence, stance, the graph and media into a verdict."
    tools = ("stance.apply", "graph.totals", "graph.decisive", "graph.open_claims",
             "verdict.decide")

    def run(self, context, step):
        if context.graph is None:
            return self.fail("there is no graph to verify against")

        stance = None

        if context.option("stance", True):
            claim_ids = step.args.get("claim_ids") or context.data(
                "evidence", "rescore"
            )

            report, call = context.call(
                "stance.apply",
                context.graph,
                claim_ids=claim_ids,
                _args={"claims": len(claim_ids) if claim_ids else None},
            )

            if report is None:
                context.notes.append(call.error or "the stance pass returned nothing")
            else:
                stance = report.as_dict()

        verdict, call = context.call(
            "verdict.decide", context.graph, _args={"write": True}
        )

        if verdict is None:
            return self.fail(
                call.error or "the verdict stage returned nothing",
                data={"stance": stance},
            )

        per_claim = [self._claim_row(context, claim) for claim in verdict["claims"]]

        open_claims = context.open_claims()
        media = context.result("media")

        data = {
            "verdict": verdict,
            "label": verdict["label"],
            "counts": verdict.get("counts", {}),
            "per_claim": per_claim,
            "open_claims": open_claims,
            "unsettled": [
                row["claim_id"] for row in per_claim if row["label"] == "unverified"
            ],
            "stance": stance,
            "media": {
                "status": media.status if media is not None else "not run",
                "recycled": media.data.get("recycled", []) if media is not None else [],
                "mismatched": media.data.get("mismatched", []) if media is not None else [],
            },
            "evidence": {
                "kept": context.data("evidence", "kept", 0),
                "decisive": context.data("evidence", "decisive", 0),
                "retrievers": context.data("evidence", "retrievers", []),
                "escalated": bool(context.data("evidence", "escalated", False)),
            },
        }

        #  A verdict the graph did not support is still the verdict; what
        #  changes is whether this agent calls the job done.
        if verdict["label"] not in SETTLED:
            return self.abstain(
                "the evidence is not sufficient to settle this message either way",
                data=data,
                handoffs=["explanation"],
            )

        notes = [
            f"{verdict['label']} at {verdict['confidence']:.2f} "
            f"over {len(per_claim)} claim(s)"
        ]

        if data["unsettled"]:
            notes.append(f"{len(data['unsettled'])} claim(s) remain unverified")

        if data["media"]["recycled"]:
            notes.append("a recycled picture contributed to this verdict")

        return self.ok(
            data=data,
            confidence=float(verdict["confidence"]),
            notes=notes,
            handoffs=["explanation"],
        )

    def _claim_row(self, context, claim):
        """
        One claim's verdict with the arithmetic that produced it.

        The totals and the decisive hits are read through the registry so
        they appear in the trace: they are the reason for the label, and a
        reader asking "why false?" should not have to open the graph.
        """
        claim_id = claim["claim_id"]

        totals, _call = context.call(
            "graph.totals", context.graph, claim_id, _args={"claim_id": claim_id}
        )
        decisive, _call = context.call(
            "graph.decisive", context.graph, claim_id, _args={"claim_id": claim_id}
        )

        return {
            "claim_id": claim_id,
            "claim": claim.get("claim"),
            "label": claim["label"],
            "confidence": claim["confidence"],
            "explanation": claim["explanation"],
            "reasons": claim.get("reasons", []),
            "evidence_ids": claim.get("evidence_ids", []),
            "demo_only": claim.get("demo_only", False),
            "totals": totals or {},
            "decisive": [
                {
                    "evidence_id": hit["id"],
                    "publisher": hit.get("publisher") or hit.get("domain"),
                    "rating": hit.get("rating"),
                    "stance": hit.get("stance"),
                }
                for hit in (decisive or [])
            ],
        }
