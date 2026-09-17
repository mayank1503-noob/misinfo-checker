"""
The evidence agent: which sources are worth asking, and is one pass enough?

Stage 3b already knows *how* to retrieve — build the queries, ask three
retrievers in parallel, normalise, dedupe, fetch the promising few, write
them into the graph. This agent does not touch any of that. It makes the
two decisions the stage leaves to its caller:

**Which retrievers.** Not all three, always. A fact-check API answers
"has a publisher already ruled on this exact claim", the seed index
answers the same question offline, and a web search answers "is there
anything out there about this", which is a much broader and much noisier
question. So the first pass asks the two precise sources and holds the
broad one back — unless neither precise source is configured, in which
case a noisy answer beats no answer and web goes in the first pass.

**Whether to go again.** After the first pass the graph says which
claims still have nothing on either side (`graph.open_claims`). If any
are still open and a source was held back, the agent escalates: the held
back retrievers, wider queries, and only the claims that are actually
open — a second round that costs a few calls rather than a repeat of the
whole pass. The escalated round works claim by claim through
`retrieve_for_claim`, fetches text for what it found, and attaches it
with `graph.add_evidence`, all stage 3b functions.

It abstains when nothing was kept, and it says whether that was because
every source was unavailable (offline, no keys) or because the sources
were asked and had nothing. Those are very different facts about a claim
and the explanation needs to tell them apart.
"""

from .base import Agent
from .context import RETRIEVAL_TOOLS


# The precise sources: they answer "has this exact claim been ruled on".
PRECISE = ("factcheck", "seed_index")

# The broad one: held back for a second pass.
BROAD = ("web",)

# A wider net on the second pass - more queries per claim, because the
# obvious phrasing has already failed.
ESCALATED_QUERIES = 4

# Ceilings for the escalated round: it is the expensive one.
MAX_ESCALATED_CLAIMS = 4
ESCALATED_FETCH = 2


class EvidenceAgent(Agent):
    agent = "evidence"
    description = "Decide which retrievers to ask, and escalate when claims stay open."
    tools = RETRIEVAL_TOOLS + (
        "evidence.collect", "evidence.fetch_text", "evidence.attach",
        "graph.open_claims",
    )

    def run(self, context, step):
        if not context.option("retrieve", True):
            return self.skip("retrieval is switched off for this run")

        claims = context.priority_claims()

        if not claims:
            return self.skip("there are no check-worthy claims to search for")

        if context.graph is None:
            return self.fail("there is no graph to write evidence into")

        if step.args.get("escalate"):
            #  A second round replaces this agent's result, so it carries
            #  the first round's tool calls with it - otherwise the
            #  trace on the agent would show only the escalation.
            previous = context.result(self.agent)
            result = self._escalate(context, step, claims)

            if previous is not None:
                result.tool_calls = list(previous.tool_calls) + result.tool_calls

            return result

        return self._first_pass(context, step, claims)

    #  --- pass one ------------------------------------------------------

    def _route(self, context):
        """
        Which retrievers to ask now, and which to keep in reserve.

        Returns (chosen, held_back, reasons) in retriever-name terms.
        """
        usable = {
            tool.split(".", 1)[1]
            for tool in context.available(*RETRIEVAL_TOOLS)
        }

        chosen = [name for name in PRECISE if name in usable]
        held_back = [name for name in BROAD if name in usable]
        reasons = []

        if chosen:
            reasons.append(
                "asking the sources that rule on a claim directly: "
                + ", ".join(chosen)
            )
        elif held_back:
            #  Nothing precise is configured. A broad search now beats an
            #  abstention we could have avoided.
            chosen, held_back = held_back, []
            reasons.append(
                "no fact-check source is configured, so the web search runs first"
            )

        if held_back:
            reasons.append(
                "holding back " + ", ".join(held_back) + " for claims that stay open"
            )

        unavailable = sorted(
            {tool.split(".", 1)[1] for tool in RETRIEVAL_TOOLS} - usable
        )

        if unavailable:
            reasons.append("unavailable here: " + ", ".join(unavailable))

        return chosen, held_back, reasons

    def _first_pass(self, context, step, claims):
        chosen, held_back, reasons = self._route(context)

        data = {
            "retrievers": chosen,
            "held_back": held_back,
            "routing": reasons,
            "rounds": [],
            "candidates": 0,
            "kept": 0,
            "decisive": 0,
            "escalated": False,
            "errors": [],
        }

        if not chosen:
            data["open_claims"] = [claim.id for claim in claims]

            return self.abstain(
                "no retriever is available: no API keys and no local index, so "
                "nothing could be asked about these claims",
                data=data,
                handoffs=["verification"],
            )

        # Only the prioritised claims are searched. The claim set is
        # copied rather than edited: stage 3b writes `evidence_ids` onto
        # the claim objects, and a shallow copy shares them, so the real
        # claims still learn what was found for them.
        scoped = context.claimset.model_copy(
            update={"claims": claims}
        )
        modules = {
            name: module
            for name, module in context.tools.retrievers.items()
            if name in chosen
        }

        report, call = context.call(
            "evidence.collect",
            scoped,
            context.graph,
            retrievers=modules,
            _args={"retrievers": chosen, "claims": len(claims)},
        )

        if report is None:
            data["errors"].append(call.error or "the collector returned nothing")

            return self.fail(
                call.error or "evidence collection failed", data=data
            )

        summary = report.as_dict()
        data["rounds"].append({"round": 1, "retrievers": chosen, **summary})
        data["candidates"] += summary["candidates"]
        data["kept"] += summary["kept"]
        data["decisive"] += summary["decisive"]
        data["by_retriever"] = summary["by_retriever"]
        data["errors"].extend(summary["errors"])
        data["open_claims"] = context.open_claims()

        return self._finish(context, data, held_back, asked=chosen)

    #  --- pass two ------------------------------------------------------

    def _escalate(self, context, step, claims):
        """
        A second, targeted round: the held-back sources on the open claims.

        Everything here is stage 3b — `retrieve_for_claim` per claim per
        retriever, `fetch.enrich` on what came back, `graph.add_evidence`
        to attach it. The agent's contribution is the *scope*: which
        claims, which sources, how many queries.
        """
        previous = context.result("evidence")
        data = dict(previous.data) if previous is not None else {}

        data.setdefault("rounds", [])
        data.setdefault("errors", [])
        data.setdefault("candidates", 0)
        data.setdefault("kept", 0)
        data.setdefault("decisive", 0)

        wanted = step.args.get("retrievers") or data.get("held_back") or []
        tools = context.available(*[f"evidence.{name}" for name in wanted])

        open_ids = set(step.args.get("claim_ids") or context.open_claims())
        targets = [claim for claim in claims if claim.id in open_ids]
        targets = targets[:MAX_ESCALATED_CLAIMS]

        if not tools or not targets:
            data["escalated"] = False
            data["open_claims"] = sorted(open_ids)

            reason = (
                "no retriever is left to try" if not tools
                else "no claim is still open"
            )

            if previous is not None and previous.status == "ok":
                #  The first pass stands; there was simply nothing more to do.
                return self.ok(
                    data=data, confidence=previous.confidence,
                    notes=list(previous.notes) + [f"no second pass: {reason}"],
                )

            return self.abstain(f"no second pass was possible: {reason}", data=data)

        max_queries = int(step.args.get("max_queries", ESCALATED_QUERIES))
        round_number = len(data["rounds"]) + 1
        found = 0
        attached = 0

        for claim in targets:
            for tool in tools:
                result, call = context.call(
                    tool,
                    claim,
                    max_queries=max_queries,
                    _args={"claim_id": claim.id, "max_queries": max_queries},
                )

                if result is None:
                    data["errors"].append(call.error or f"{tool} returned nothing")
                    continue

                candidates = result["candidates"]
                data["errors"].extend(result["errors"])
                found += len(candidates)

                if not candidates:
                    continue

                context.call(
                    "evidence.fetch_text",
                    candidates,
                    limit=ESCALATED_FETCH,
                    _args={"candidates": len(candidates)},
                )

                written, _call = context.call(
                    "evidence.attach",
                    context.graph,
                    claim,
                    candidates,
                    _args={"claim_id": claim.id, "candidates": len(candidates)},
                )

                if written is not None:
                    attached += len(written["attached"])
                    data["errors"].extend(written["errors"])

        data["rounds"].append(
            {
                "round": round_number,
                "retrievers": [tool.split(".", 1)[1] for tool in tools],
                "claims": [claim.id for claim in targets],
                "max_queries": max_queries,
                "candidates": found,
                "kept": attached,
            }
        )
        data["candidates"] += found
        data["kept"] += attached
        data["escalated"] = True
        data["open_claims"] = context.open_claims()

        return self._finish(
            context, data, held_back=[],
            asked=[tool.split(".", 1)[1] for tool in tools],
            escalated=True,
        )

    #  --- shared ending -------------------------------------------------

    def _finish(self, context, data, held_back, asked, escalated=False):
        """
        One ending for both passes: hand the claim ids on, or abstain.

        `rescore` is the handoff. The verification agent only needs to
        re-run stage 4 over the claims this round touched, and telling it
        which ones is cheaper and clearer than letting it re-score the
        whole graph.
        """
        data["escalated"] = data.get("escalated", escalated)
        data["rescore"] = [claim.id for claim in context.priority_claims()]

        if not data["kept"]:
            return self.abstain(
                "the sources that were available ("
                + ", ".join(asked)
                + ") returned nothing about these claims",
                data=data,
                handoffs=["verification"],
            )

        notes = [
            f"{data['kept']} item(s) of evidence from {', '.join(asked)}"
            + (f", {data['decisive']} decisive" if data.get("decisive") else "")
        ]

        if data.get("open_claims"):
            notes.append(f"{len(data['open_claims'])} claim(s) still open")

        if held_back:
            notes.append("in reserve: " + ", ".join(held_back))

        #  Confidence in the *retrieval*, not in any claim: a decisive
        #  fact-check is a good day, a handful of loose articles is not.
        confidence = 0.9 if data.get("decisive") else min(0.75, 0.3 + 0.1 * data["kept"])

        return self.ok(
            data=data,
            confidence=round(confidence, 4),
            notes=notes,
            handoffs=["verification"],
        )
