"""
Stages 1-5, as tools an agent can pick up.

Nothing in this file decides anything and nothing in it re-implements a
stage. Each `Tool` is a name, a sentence of description, a probe that
says whether it can run at all, and a call straight through to the
function that already exists:

    claims.extract       backend.claims.extract_claims
    graph.build          backend.graph.EvidenceGraph.from_claimset
    evidence.collect     backend.evidence.collect_evidence
    evidence.factcheck   backend.evidence.retrieve_for_claim, factcheck only
    evidence.seed_index  ... the local seed index only
    evidence.web         ... web search only
    evidence.fetch_text  backend.evidence.fetch.enrich
    evidence.attach      graph.add_evidence, over a list of candidates
    images.keyframes     backend.images.select_keyframes
    images.collect       backend.images.collect_image_evidence
    stance.apply         backend.stance.apply_stances
    graph.totals         graph.stance_totals
    graph.open_claims    graph.open_claims
    graph.decisive       graph.decisive_hits
    graph.timeline       graph.timeline
    verdict.decide       backend.verdict.decide

Two properties matter more than the list.

**A tool knows whether it can run.** `available()` is the retriever's or
the model's own probe - `factcheck.available()` is false without an API
key - so an agent routes on the real configuration rather than on a call
that will come back empty. This is what keeps the offline path honest:
with no keys and no models the registry reports one usable retrieval
path, and the agents plan around that instead of pretending.

**A tool never raises.** `ToolRegistry.call` returns `(value, ToolCall)`,
and a failure is a `ToolCall` with `ok=False` and the exception text. The
stages already fail soft; this keeps that property at the agent layer,
where one dead retriever must not take a verdict down with it.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .schema import ToolCall


log = logging.getLogger(__name__)


@dataclass
class Tool:
    """One callable stage function, named and probed."""

    name: str
    description: str
    run: Callable
    probe: Optional[Callable] = None
    summary: Optional[Callable] = None

    #  the probe's answer, remembered for the life of this tool object
    _probed: Optional[bool] = None

    def available(self):
        """
        Whether this tool can do anything right now.

        No probe means "always": `graph.totals` reads a dict in memory and
        cannot be unavailable. A probe that raises counts as unavailable,
        because a probe is the cheap check and if even that failed the
        tool itself will not work.

        The answer is remembered. Some probes are not cheap at all — the
        CLIP one imports `sentence_transformers`, which means importing
        torch — and an agent run asks the same question several times
        while it routes. Tools are created per run (`default_registry`),
        so nothing is cached across runs and a test that switches a model
        off still gets a fresh probe.
        """
        if self.probe is None:
            return True

        if self._probed is not None:
            return self._probed

        try:
            self._probed = bool(self.probe())
        except Exception as error:                   # fail soft, never raise
            log.debug(
                "probe for %s failed (%s): %s", self.name, type(error).__name__, error
            )
            self._probed = False

        return self._probed


@dataclass
class ToolRegistry:
    """
    The tools an orchestrator hands its agents.

    Built per run, so a test can swap one tool for a stub without
    touching module state and the call log belongs to the run rather than
    to the process.
    """

    tools: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    #  name -> retriever module, for the tools that take a retriever set
    #  (stage 3b's collector is given one, rather than reading the global
    #  RETRIEVERS itself). Kept here so a run can be pointed at a
    #  different set of sources without patching module state.
    retrievers: dict = field(default_factory=dict)

    def register(self, tool):
        self.tools[tool.name] = tool

        return tool

    def get(self, name):
        if name not in self.tools:
            raise KeyError(f"no such tool: {name!r}")

        return self.tools[name]

    def names(self):
        return sorted(self.tools)

    def available(self, name):
        """False for an unknown tool as well as for an unusable one."""
        return name in self.tools and self.tools[name].available()

    def usable(self, names):
        return [name for name in names if self.available(name)]

    def call(self, name, *args, _agent=None, _args=None, **kwargs):
        """
        Run one tool. Returns `(value, ToolCall)`, and never raises.

        `_args` is what the trace should record about the arguments. The
        real ones are packets, claim sets and graphs, none of which
        belong in a log line, so the caller passes the small version.
        """
        started = time.perf_counter()
        record = ToolCall(tool=name, agent=_agent, args=dict(_args or {}))

        try:
            tool = self.get(name)
            value = tool.run(*args, **kwargs)

            if tool.summary is not None:
                try:
                    record.summary = dict(tool.summary(value) or {})
                except Exception as error:        # a summary is never load-bearing
                    log.debug("summarising %s failed: %s", name, error)
        except Exception as error:                   # fail soft, never raise
            log.warning("tool %s failed (%s): %s", name, type(error).__name__, error)
            record.ok = False
            record.error = f"{type(error).__name__}: {error}"
            value = None

        record.ms = round((time.perf_counter() - started) * 1000, 1)
        self.calls.append(record)

        return value, record

    def called(self, name=None):
        """The call log, optionally for one tool. For traces and for tests."""
        return [call for call in self.calls if name is None or call.tool == name]


def attach_candidates(graph, claim, candidates):
    """
    Write retrieved candidates into the graph and onto the claim.

    The one piece of glue in this file, and it is glue rather than logic:
    `graph.add_evidence` does the work. It exists because a targeted
    second retrieval round works claim by claim, while stage 3b's own
    collector works over a whole `ClaimSet`. Failures are counted, not
    raised.
    """
    attached = []
    errors = []

    for candidate in candidates:
        try:
            if graph is not None:
                graph.add_evidence(claim.id, candidate)

            attached.append(candidate.id)
        except Exception as error:                   # fail soft, never raise
            errors.append(f"{candidate.id}: {type(error).__name__}: {error}")

    known = list(getattr(claim, "evidence_ids", None) or [])
    claim.evidence_ids = known + [eid for eid in attached if eid not in known]

    return {"attached": attached, "errors": errors}


def _retriever_tool(name, module):
    """
    One retriever as a tool: `retrieve_for_claim` scoped to it alone.

    Stage 3b documents `retrieve_for_claim` as the way to look at raw
    retrieval without building a graph, which is exactly what an agent
    doing a targeted second round wants. A one-entry `retrievers` dict is
    how that function already supports being asked for a single source.
    """
    from ..evidence import retrieve_for_claim

    def run(claim, max_queries=3):
        candidates, counts, errors = retrieve_for_claim(
            claim, max_queries=max_queries, retrievers={name: module}
        )

        return {"candidates": candidates, "counts": counts, "errors": errors}

    return Tool(
        name=f"evidence.{name}",
        description=f"Retrieve evidence for one claim from {name} alone.",
        run=run,
        probe=module.available,
        summary=lambda result: {
            "candidates": len((result or {}).get("candidates") or []),
            "errors": len((result or {}).get("errors") or []),
        },
    )


def default_registry(retrievers=None):
    """
    Every stage function an agent is allowed to call.

    `retrievers` overrides `backend.evidence.retrievers.RETRIEVERS`, which
    is how a test hands the evidence agent a fake source without patching
    module globals.
    """
    from ..claims import extract_claims
    from ..evidence import collect_evidence, fetch
    from ..evidence.retrievers import RETRIEVERS
    from ..graph import EvidenceGraph
    from ..images import collect_image_evidence, select_keyframes
    from ..images import consistency, reverse_search
    from ..stance import apply_stances
    from ..verdict import decide

    registry = ToolRegistry(retrievers=dict(retrievers or RETRIEVERS))

    registry.register(Tool(
        name="claims.extract",
        description="Stage 2: read a packet and return its claims, scored and typed.",
        run=extract_claims,
        summary=lambda claimset: {
            "claims": len(claimset.claims) if claimset else 0,
            "check_worthy": len(claimset.check_worthy()) if claimset else 0,
            "backend": getattr(claimset, "backend", None),
        },
    ))

    registry.register(Tool(
        name="graph.build",
        description="Stage 3a: build the evidence graph from a claim set and its packet.",
        run=EvidenceGraph.from_claimset,
        summary=lambda graph: {"nodes": len(graph) if graph is not None else 0},
    ))

    registry.register(Tool(
        name="evidence.collect",
        description=(
            "Stage 3b: retrieve evidence for every check-worthy claim and write it "
            "into the graph."
        ),
        run=collect_evidence,
        summary=lambda report: report.as_dict() if report is not None else {},
    ))

    for name, module in registry.retrievers.items():
        registry.register(_retriever_tool(name, module))

    registry.register(Tool(
        name="evidence.fetch_text",
        description="Fetch the full article text for the most promising candidates.",
        run=fetch.enrich,
        probe=fetch.available,
        summary=lambda fetched: {"fetched": int(fetched or 0)},
    ))

    registry.register(Tool(
        name="evidence.attach",
        description="Write retrieved candidates into the graph as evidence nodes.",
        run=attach_candidates,
        summary=lambda result: {
            "attached": len((result or {}).get("attached") or []),
            "errors": len((result or {}).get("errors") or []),
        },
    ))

    registry.register(Tool(
        name="images.keyframes",
        description="Drop near-identical frames and cap how many images are checked.",
        run=select_keyframes,
        summary=lambda kept: {"kept": len(kept or [])},
    ))

    registry.register(Tool(
        name="images.collect",
        description=(
            "Stage 5: index match, reverse image search and the CLIP consistency "
            "check, written into the graph."
        ),
        run=collect_image_evidence,
        summary=lambda report: report.as_dict() if report is not None else {},
    ))

    registry.register(Tool(
        name="images.reverse_search",
        description="Ask the open web where else this picture has appeared.",
        run=reverse_search.search,
        probe=reverse_search.available,
        summary=lambda matches: {"matches": len(matches or [])},
    ))

    registry.register(Tool(
        name="images.consistency",
        description="Does the picture show what the claim says it shows?",
        run=consistency.check_image,
        probe=consistency.available,
        summary=lambda findings: {"findings": len(findings or [])},
    ))

    registry.register(Tool(
        name="stance.apply",
        description="Stage 4: decide what each piece of evidence says about its claim.",
        run=apply_stances,
        summary=lambda report: report.as_dict() if report is not None else {},
    ))

    registry.register(Tool(
        name="graph.totals",
        description="Weighted support vs. refutation for one claim.",
        run=lambda graph, claim_id: graph.stance_totals(claim_id),
        summary=lambda totals: {
            "supports": (totals or {}).get("supports"),
            "refutes": (totals or {}).get("refutes"),
        },
    ))

    registry.register(Tool(
        name="graph.open_claims",
        description="Check-worthy claims nothing has taken a side on yet.",
        run=lambda graph: graph.open_claims(),
        summary=lambda open_ids: {"open": len(open_ids or [])},
    ))

    registry.register(Tool(
        name="graph.decisive",
        description="Fact-checks that match a claim closely enough to settle it.",
        run=lambda graph, claim_id=None: graph.decisive_hits(claim_id),
        summary=lambda hits: {"decisive": len(hits or [])},
    ))

    registry.register(Tool(
        name="graph.timeline",
        description="Everything the graph knows a date for, oldest first.",
        run=lambda graph: graph.timeline(),
        summary=lambda events: {"events": len(events or [])},
    ))

    registry.register(Tool(
        name="verdict.decide",
        description="Stage 6: decide every claim and the packet, by the existing rules.",
        run=decide,
        summary=lambda verdict: {
            "label": (verdict or {}).get("label"),
            "confidence": (verdict or {}).get("confidence"),
        },
    ))

    return registry
