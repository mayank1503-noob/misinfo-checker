"""
The agentic entry point, shaped exactly like the linear one.

    from backend.agents import run_agentic

    result = run_agentic(packet)
    result["verdict"]["label"]     # false / misleading / true / disputed / unverified

`run_agentic` returns the same dict `backend.pipeline.analyze` returns —
`packet`, `claims`, `dropped`, `verdict`, `graph`, `open_claims`,
`timeline`, `stages`, `ms` — plus two additive keys, `agents` and
`explanation`. That is not a courtesy to the API; it is the point. The
bot, the API and the tests already read that shape, and an architecture
change that forces every caller to learn a second one has failed
regardless of how good the architecture is.

`stages` is the interesting part of the compatibility. The linear
pipeline reports one entry per stage, with `ms` and `error`, and that is
what `/health`-adjacent tooling and the API response contract expose. The
agents did the same work through the same functions, so the same report
is reconstructed from the tool trace: `claims.extract` becomes the
`claims` stage, the six `evidence.*` tools become the `evidence` stage,
and so on. Each entry also carries which agent made the call, which the
linear pipeline could not have told you.

The labels are untouched. `verdict` here is the dict
`backend.verdict.decide` returned, with the explanation agent's prose
added as `verdict["explanation"]` alongside — never in place of —
`verdict["summary"]`, because `summary` is what the bot sends and the
rules that generate it are the explainable ones.
"""

import logging
import time

from .orchestrator import Orchestrator


log = logging.getLogger(__name__)


# Which tool calls belong to which stage of the linear pipeline, so an
# agentic run reports the same `stages` block.
STAGE_TOOLS = {
    "claims": ("claims.extract",),
    "graph": ("graph.build",),
    "evidence": (
        "evidence.collect", "evidence.factcheck", "evidence.seed_index",
        "evidence.web", "evidence.fetch_text", "evidence.attach",
    ),
    "images": ("images.keyframes", "images.collect"),
    "stance": ("stance.apply",),
    "verdict": ("verdict.decide",),
}

# What the evidence agent's own aggregate is authoritative about: after an
# escalated round the per-call summaries each describe one round, and the
# agent is the only thing that knows the total.
EVIDENCE_TOTALS = ("candidates", "kept", "decisive", "by_retriever", "errors")

UNVERIFIED = {
    "label": "unverified",
    "confidence": 0.0,
    "summary": "The message could not be checked.",
    "counts": {},
    "claims": [],
}


def run_agentic(packet, backend="auto", today=None, retrieve=True, images=True,
                stance=True, graph_json=False, max_text=None, planner=None,
                agents=None, tools=None, writer=None, max_rounds=None,
                orchestrator=None):
    """
    Run the agent layer over a packet and return a pipeline-shaped result.

    Every keyword the linear `analyze` takes means the same thing here.
    The extra ones are the seams: `planner` (a routing policy — see
    `policy.LLMPlanner`), `agents` (swap one specialist), `tools` (swap
    one stage function), `writer` (a callable that writes the final
    explanation), `orchestrator` (replace the loop entirely).
    """
    started = time.perf_counter()

    packet = dict(packet or {})

    if max_text and packet.get("text"):
        packet["text"] = packet["text"][:max_text]

    runner = orchestrator or Orchestrator(
        planner=planner,
        agents=agents,
        tools=tools,
        writer=writer,
        **({"max_rounds": max_rounds} if max_rounds is not None else {}),
    )

    options = {
        "backend": backend,
        "today": today,
        "retrieve": retrieve,
        "images": images,
        "stance": stance,
    }

    if max_rounds is not None:
        options["max_rounds"] = max_rounds

    context, report = runner.run(packet, **options)

    verdict = _verdict_of(context)
    explanation = context.data("explanation", "explanation")

    if explanation:
        #  Added beside `summary`, never over it: `summary` comes from the
        #  rule cascade and is what the bot sends.
        verdict = dict(verdict)
        verdict["explanation"] = explanation

    claims = []
    dropped = []

    if context.claimset is not None:
        serialised = context.claimset.model_dump(mode="json")
        claims = serialised["claims"]
        dropped = context.claimset.dropped

    graph = None

    if context.graph is not None:
        graph = context.graph.to_dict() if graph_json else context.graph.summary(2000)

    return {
        "packet": _public_packet(packet),
        "claims": claims,
        "dropped": dropped,
        "verdict": verdict,
        "graph": graph,
        "open_claims": context.graph.open_claims() if context.graph is not None else [],
        "timeline": context.graph.timeline() if context.graph is not None else [],
        "stages": _stages(report),
        "agents": report,
        "explanation": context.result("explanation").data
        if context.result("explanation") is not None else {},
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _verdict_of(context):
    """
    The verdict the verification agent produced, or an honest placeholder.

    A missing verdict means verification failed outright — a broken
    graph, a stage that raised. `unverified` with a reason is the right
    answer to that, and the agent trace carries the detail.
    """
    verification = context.result("verification")

    if verification is not None and verification.data.get("verdict"):
        return verification.data["verdict"]

    placeholder = dict(UNVERIFIED)

    if verification is not None and verification.notes:
        placeholder["summary"] = (
            f"The message could not be checked: {verification.notes[0]}"
        )
    elif context.result("claim") is not None and context.result("claim").abstained:
        placeholder["summary"] = (
            "The message could not be read as any checkable claim."
        )

    return placeholder


def _stages(report):
    """
    The linear pipeline's `stages` block, rebuilt from the tool trace.

    A stage appears when one of its tools was called, with the summed
    timing, the first error if any, and whichever agent made the call.
    Stages the plan never reached are absent, exactly as they are absent
    from `analyze(retrieve=False)`.
    """
    calls = report.get("trace") or []
    results = report.get("results") or {}

    stages = {}

    for stage, tools in STAGE_TOOLS.items():
        relevant = [call for call in calls if call["tool"] in tools]

        if not relevant:
            continue

        errors = [call["error"] for call in relevant if call.get("error")]
        entry = {
            "ms": round(sum(call["ms"] for call in relevant), 1),
            "error": errors[0] if errors else None,
            "agent": next(
                (call["agent"] for call in relevant if call.get("agent")), None
            ),
            "calls": len(relevant),
        }

        for call in relevant:
            entry.update(call.get("summary") or {})

        stages[stage] = entry

    evidence = (results.get("evidence") or {}).get("data") or {}

    if "evidence" in stages and evidence:
        for key in EVIDENCE_TOTALS:
            if key in evidence:
                stages["evidence"][key] = evidence[key]

        stages["evidence"]["rounds"] = len(evidence.get("rounds") or [])
        stages["evidence"]["escalated"] = bool(evidence.get("escalated"))

    return stages


def _public_packet(packet):
    """The packet without the embeddings — the linear pipeline's own version."""
    from ..pipeline import _public_packet as strip

    return strip(packet)
