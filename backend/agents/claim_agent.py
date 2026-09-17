"""
The claim agent: what is worth checking here, and in what order?

Extraction is stage 2's job and this agent does not repeat a line of it —
`claims.extract` is `backend.claims.extract_claims`. What the agent adds
is the decision stage 2 deliberately does not make: stage 2 scores every
sentence for check-worthiness, but a message with nine check-worthy
sentences and three retrievers is nine times three API calls, and the
sentence most worth spending them on is not simply the one with the
highest confidence.

So each claim gets a priority: its own confidence, blended with how much
harm its *kind* does when it spreads. A forward-to-get-money chain offer
and a cure-for-a-disease claim are the two that cost people money and
health, and they are also the two a fact-checker most likely has already
written about; a generic factual sentence is neither. The blend is 60%
what stage 2 thought of this sentence, 40% what type it is, so the type
tilts the order without ever overruling a confident extraction.

Abstains when nothing is check-worthy. That is the personal-chat case,
and the honest answer to "kal shaam ko ghar aa raha hoon" is not a
retrieval budget.
"""

from .base import Agent


# How much a kind of claim matters when it travels. Not a probability of
# being false - a measure of what it costs if it is.
RISK = {
    "chain_offer": 1.0,     # forward-to-get: costs people money directly
    "health":      0.95,    # cures and vaccines: costs people their health
    "money":       0.9,
    "policy":      0.8,
    "event":       0.7,
    "statistic":   0.65,
    "attribution": 0.6,
    "causal":      0.6,
    "prediction":  0.5,
    "generic":     0.4,
}

DEFAULT_RISK = 0.5

CONFIDENCE_WEIGHT = 0.6

# Enough for any real forward; a cap on what retrieval can be asked to do.
MAX_CLAIMS = 8


def priority_of(claim):
    """Blended priority for one claim, in [0, 1]."""
    risk = RISK.get(claim.claim_type, DEFAULT_RISK)

    return round(
        CONFIDENCE_WEIGHT * float(claim.confidence)
        + (1 - CONFIDENCE_WEIGHT) * risk,
        4,
    )


class ClaimAgent(Agent):
    agent = "claim"
    description = "Extract check-worthy claims and put them in priority order."
    tools = ("claims.extract",)

    def run(self, context, step):
        claimset, call = context.call(
            "claims.extract",
            context.packet,
            today=context.option("today"),
            backend=context.option("backend", "auto"),
            _args={"backend": context.option("backend", "auto")},
        )

        if claimset is None:
            return self.fail(
                call.error or "claim extraction returned nothing",
                data={"claims": [], "priority": [], "check_worthy": 0},
            )

        context.claimset = claimset

        worthy = sorted(
            claimset.check_worthy(), key=priority_of, reverse=True
        )
        limit = int(step.args.get("max_claims", MAX_CLAIMS))
        ranked = worthy[:limit]

        context.priority = [claim.id for claim in ranked]

        data = {
            "backend": claimset.backend,
            "claims": len(claimset.claims),
            "check_worthy": len(worthy),
            "dropped": len(claimset.dropped),
            "deferred": [claim.id for claim in worthy[limit:]],
            "priority": [
                {
                    "claim_id": claim.id,
                    "text": claim.text,
                    "claim_type": claim.claim_type,
                    "confidence": claim.confidence,
                    "priority": priority_of(claim),
                    "language": claim.language,
                    "entities": [entity.text for entity in claim.entities][:6],
                }
                for claim in ranked
            ],
        }

        if not worthy:
            return self.abstain(
                "nothing in this message reads as a checkable claim", data=data
            )

        notes = [
            f"{len(worthy)} check-worthy claim(s) from {claimset.backend}; "
            f"checking {len(ranked)} in priority order"
        ]

        if data["deferred"]:
            notes.append(
                f"{len(data['deferred'])} lower-priority claim(s) left unchecked "
                f"(budget is {limit})"
            )

        #  Confidence here is confidence in the *extraction*, so the top
        #  claim's own score is the honest number.
        return self.ok(
            data=data,
            confidence=float(ranked[0].confidence),
            notes=notes,
            handoffs=["evidence"],
        )
