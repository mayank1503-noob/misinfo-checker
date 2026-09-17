"""
The explanation agent: say what was found, and nothing more.

It has no tools. That is deliberate: everything it needs is already in
the other agents' structured output, and an explainer that could go and
look something up could also explain something the verdict was not based
on. Its only inputs are the verification agent's per-claim rows (label,
confidence, reasons, the weighted totals) and the media agent's findings.

Two rules it will not break.

**Nothing new.** Every line it emits is traceable to an item of evidence,
a named publisher, a date, or the absence of all three. "No evidence was
found" is a sentence it is allowed to write; "this looks like a scam" is
not, because nothing in the graph says that.

**An abstention is stated, not hidden.** When the verdict is
`unverified` the explanation says what was searched and what came back
empty, so a reader can tell "we asked and found nothing" from "we could
not ask". Those read identically in a summary that only reports the
label, and they mean completely different things to someone deciding
whether to forward a message.

The prose is generated from a template. A `writer` can be passed in —
any callable taking the same structured brief and returning a string —
which is where an LLM goes when one is configured. If the writer raises
or returns nothing, the template stands; the explanation is never allowed
to be the thing that fails a run.
"""

import logging

from .base import Agent


log = logging.getLogger(__name__)


MAX_BULLETS = 4

OPENINGS = {
    "false": "This claim is false.",
    "misleading": "This is misleading.",
    "true": "This checks out.",
    "disputed": "Credible sources disagree about this.",
    "unverified": "This could not be verified.",
}


def verification_unsettled(rows):
    """Claims that came back `unverified`, by id."""
    return [row["claim_id"] for row in rows if row.get("label") == "unverified"]


class ExplanationAgent(Agent):
    agent = "explanation"
    description = "Write a short, evidence-based explanation of the final verdict."
    tools = ()

    def __init__(self, writer=None):
        #  Provider-agnostic by construction: a callable, not a client.
        self.writer = writer

    def run(self, context, step):
        verification = context.result("verification")

        if verification is None or not verification.data.get("verdict"):
            return self.abstain(
                "there is no verdict to explain",
                data={"label": "unverified", "explanation": "", "bullets": []},
            )

        verdict = verification.data["verdict"]
        rows = verification.data.get("per_claim", [])
        media = verification.data.get("media", {})

        brief = self._brief(context, verdict, rows, media)
        bullets = self._bullets(context, verdict, rows, media)

        text = self._write(brief, bullets)

        data = {
            "label": verdict["label"],
            "confidence": verdict["confidence"],
            "explanation": text,
            "bullets": bullets,
            "sources": brief["sources"],
            "abstained": verdict["label"] == "unverified",
            "brief": brief,
        }

        if verdict["label"] == "unverified":
            #  Reported, in full, and still an abstention.
            return self.abstain(
                "explained why this could not be settled", data=data
            )

        return self.ok(
            data=data,
            confidence=float(verdict["confidence"]),
            notes=[f"explained a {verdict['label']} verdict from {len(bullets)} point(s)"],
        )

    #  --- the brief -----------------------------------------------------

    def _brief(self, context, verdict, rows, media):
        """
        The structured account a writer (template or model) turns into prose.

        Kept JSON-safe and small for the same reason the planner's
        observation is: this is what would go into a prompt.
        """
        evidence = context.data("evidence", "kept", 0)
        retrievers = context.data("evidence", "retrievers", []) or []
        top = rows[0] if rows else None

        sources = []

        for row in rows:
            for hit in row.get("decisive", []):
                publisher = hit.get("publisher")

                if publisher and publisher not in sources:
                    sources.append(publisher)

        return {
            "label": verdict["label"],
            "confidence": verdict["confidence"],
            "claims": len(rows),
            "claim": top.get("claim") if top else None,
            "counts": verdict.get("counts", {}),
            "evidence": evidence,
            "retrievers": retrievers,
            "searched": bool(retrievers),
            "recycled": media.get("recycled", []),
            "mismatched": media.get("mismatched", []),
            "unsettled": len(verification_unsettled(rows)),
            "demo_only": bool(top.get("demo_only")) if top else False,
            "sources": sources,
        }

    def _bullets(self, context, verdict, rows, media):
        """One line per piece of the argument, evidence first."""
        bullets = []

        for row in rows[:2]:
            for reason in row.get("reasons", [])[:2]:
                bullets.append(reason)

        for picture in media.get("recycled", [])[:1]:
            bullets.append(
                f"the picture {picture.get('path') or picture.get('image')} was "
                f"already online on {picture.get('first_seen')}, before this message"
            )

        for mismatch in media.get("mismatched", [])[:1]:
            if mismatch.get("reason"):
                bullets.append(mismatch["reason"])

        if not bullets:
            evidence = context.data("evidence", "kept", 0)
            retrievers = context.data("evidence", "retrievers", []) or []

            if not retrievers:
                bullets.append(
                    "no source could be searched: no fact-check API key is "
                    "configured and no local index is available"
                )
            elif evidence:
                bullets.append(
                    f"{evidence} related item(s) were found via "
                    f"{', '.join(retrievers)}, but none of them takes a side"
                )
            else:
                bullets.append(
                    f"{', '.join(retrievers)} were searched and returned nothing "
                    "about these claims"
                )

        return bullets[:MAX_BULLETS]

    #  --- the prose -----------------------------------------------------

    def _write(self, brief, bullets):
        """The writer if there is one, the template otherwise."""
        if self.writer is not None:
            try:
                written = self.writer(brief, bullets)

                if written and str(written).strip():
                    return str(written).strip()

                log.warning("the explanation writer returned nothing; using the template")
            except Exception as error:               # fail soft, never raise
                log.warning(
                    "the explanation writer failed (%s): %s",
                    type(error).__name__, error,
                )

        return self._template(brief, bullets)

    def _template(self, brief, bullets):
        line = OPENINGS.get(brief["label"], OPENINGS["unverified"])

        if brief["claims"] > 1:
            line += f" {brief['claims']} claims were checked"
            counts = ", ".join(
                f"{count} {label}" for label, count in sorted(brief["counts"].items())
            )
            line += f" ({counts})." if counts else "."

        if bullets:
            line += " " + bullets[0].rstrip(".") + "."

        if brief["label"] == "unverified" and brief["unsettled"]:
            line += (
                f" {brief['unsettled']} claim(s) are reported as unverified rather "
                "than guessed at."
            )

        if brief["demo_only"]:
            line += " This rests on the demo evidence index, not on live fact-checks."

        return line

