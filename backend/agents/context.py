"""
The blackboard every agent reads and writes.

Agents do not message each other. They share one `AgentContext`: the
packet, the claim set stage 2 produced, the graph every later stage grows,
and the results of whoever ran before them. A handoff is therefore a fact
in the context plus a named step in the plan, not a conversation — which
is why the whole layer is testable without a model and why an agent can
be swapped for an LLM-backed one without changing anyone else.

`observe()` is the one method the planner uses. It is a small, flat,
JSON-safe dict: what kind of input this is, how many claims are worth
checking, which tools are actually usable, what is still unresolved. A
rule planner reads it directly; an LLM planner puts it in a prompt. That
is the whole reason the context exposes a summary rather than itself.
"""

import logging
from dataclasses import dataclass, field


log = logging.getLogger(__name__)


# Retrieval tools, in the order a first pass should prefer them: a
# publisher who checked this claim beats a search engine that found
# something near it, and the local index answers with no key at all.
RETRIEVAL_TOOLS = ("evidence.factcheck", "evidence.seed_index", "evidence.web")

MEDIA_TOOLS = ("images.reverse_search", "images.consistency")


@dataclass
class AgentContext:
    """
    Shared state for one agentic run.

    `options` carries the same switches `backend.pipeline.analyze` takes
    (`backend`, `today`, `retrieve`, `images`, `stance`), so the agentic
    path honours a caller who turned retrieval off for latency exactly as
    the linear path does.
    """

    packet: dict
    tools: object
    options: dict = field(default_factory=dict)

    claimset: object = None
    graph: object = None

    results: dict = field(default_factory=dict)
    priority: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    #  the agent currently running, so `call` can file its tool calls
    current: object = None

    def option(self, name, default=None):
        return self.options.get(name, default)

    def call(self, tool, *args, _args=None, _agent=None, **kwargs):
        """
        Run a tool on behalf of the running agent.

        The `ToolCall` record lands on that agent's result as well as in
        the registry's log, so an agent's own output says what it did
        without the reader having to cross-reference a global trace.
        """
        value, record = self.tools.call(
            tool,
            *args,
            _agent=_agent or getattr(self.current, "agent", None),
            _args=_args,
            **kwargs,
        )

        if self.current is not None:
            self.current.tool_calls.append(record)

        return value, record

    def available(self, *tools):
        """The subset of these tools that can actually run."""
        return [tool for tool in tools if self.tools.available(tool)]

    def claims(self):
        return list(getattr(self.claimset, "claims", None) or [])

    def check_worthy(self):
        return [claim for claim in self.claims() if claim.check_worthy]

    def claim(self, claim_id):
        for claim in self.claims():
            if claim.id == claim_id:
                return claim

        return None

    def priority_claims(self):
        """
        The claims the claim agent put in front, as `Claim` objects.

        Falls back to every check-worthy claim, so an agent still works
        if the claim agent failed or was skipped.
        """
        ordered = [self.claim(claim_id) for claim_id in self.priority]
        ordered = [claim for claim in ordered if claim is not None]

        return ordered or self.check_worthy()

    def images(self):
        return list((self.packet or {}).get("images") or [])

    def has_media(self):
        return bool(self.images())

    def open_claims(self):
        """
        Check-worthy claims with no stance either way, read off the graph.

        Read through the tool registry rather than the graph directly:
        this is the signal an escalation decision turns on, so it belongs
        in the trace like any other tool call.
        """
        if self.graph is None:
            return [claim.id for claim in self.check_worthy()]

        open_ids, _record = self.call("graph.open_claims", self.graph)

        return list(open_ids or [])

    def result(self, agent):
        return self.results.get(agent)

    def data(self, agent, key, default=None):
        """One field of another agent's structured output."""
        result = self.results.get(agent)

        if result is None:
            return default

        return result.data.get(key, default)

    def observe(self, round=0):
        """
        The flat, JSON-safe picture a planner decides from.

        Everything a routing decision could need and nothing that cannot
        be put in a prompt: no graphs, no claim objects, no embeddings.
        """
        claims = self.check_worthy()
        still_open = None

        if self.graph is not None:
            try:
                #  Read directly, not through the registry: an
                #  observation is a snapshot for a planner, and a
                #  snapshot should not show up in the trace as work.
                still_open = len(self.graph.open_claims())
            except Exception as error:               # fail soft, never raise
                log.debug("could not count open claims: %s", error)

        retrieval = {
            tool.split(".", 1)[1]: self.tools.available(tool)
            for tool in RETRIEVAL_TOOLS
        }
        #  Only probed when there is media to probe *for*: the CLIP
        #  check's own probe imports torch, and a text forward must not
        #  pay tens of seconds to be told about a tool it cannot use.
        media = {
            tool.split(".", 1)[1]: self.tools.available(tool) if self.images() else None
            for tool in MEDIA_TOOLS
        }

        return {
            "round": round,
            "input_type": (self.packet or {}).get("input_type"),
            "has_text": bool((self.packet or {}).get("text")),
            "images": len(self.images()),
            "claims": len(self.claims()),
            "check_worthy": len(claims),
            "claim_types": sorted({claim.claim_type for claim in claims}),
            "languages": sorted({claim.language for claim in claims}),
            "retrievers": retrieval,
            "media_tools": media,
            "offline": not any(retrieval.values()),
            "open_claims": still_open,
            "options": {
                key: self.options.get(key)
                for key in ("retrieve", "images", "stance", "backend", "max_rounds")
            },
            "statuses": {name: result.status for name, result in self.results.items()},
        }
