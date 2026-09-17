"""
The orchestrator: dispatch, shared state, handoffs, and a hard budget.

It does four things and refuses to do a fifth.

**It dispatches.** The planner says which agent runs next and why; this
loop calls it with the shared context and files the result. It does not
decide the route (that is `policy.py`) and it does not decide anything
about misinformation (that is the agents and, underneath them, stages
1-5).

**It owns the shared state.** The graph belongs to the run, not to an
agent, so the orchestrator builds it — `graph.build` — the moment the
claim agent produces a claim set, and every later agent grows the same
one. Agents therefore never hand each other objects; they read and write
one blackboard, which is what makes any of them replaceable.

**It carries handoffs.** An agent's `handoffs` and the planner's
`revise()` are the two halves of the same mechanism: an agent says who
should look at this next, the planner decides whether that is worth
another round, and the loop appends the steps. The commonest one in
practice is the evidence agent's second pass after verification abstains.

**It stops.** `max_rounds` bounds re-planning and `max_steps` bounds the
queue, because the failure mode of an agent loop is not a wrong answer,
it is an unbounded bill. When a step's agent is missing or a claim set
was never produced, the loop skips the step and records why rather than
raising.

A short-circuit worth naming: when the claim agent finds nothing
check-worthy, the retrieval and media steps are dropped. Not an
optimisation — searching three retrievers for "kal shaam ko ghar aa raha
hoon" is how a checker turns into a machine that has an opinion about
someone's dinner.
"""

import logging
import time

from .base import Agent
from .claim_agent import ClaimAgent
from .context import AgentContext
from .evidence_agent import EvidenceAgent
from .explanation_agent import ExplanationAgent
from .media_agent import MediaAgent
from .policy import MAX_ROUNDS, RulePlanner
from .schema import AgentResult
from .tools import default_registry
from .verification_agent import VerificationAgent


log = logging.getLogger(__name__)


# A ceiling on the queue, not a target: the opening plan is five steps and
# one escalation adds three.
MAX_STEPS = 16

# Agents that only make sense once there is something to check.
NEEDS_CLAIMS = ("evidence", "media")


def default_agents(writer=None):
    """One instance of each specialist, by name."""
    return {
        agent.agent: agent
        for agent in (
            ClaimAgent(),
            EvidenceAgent(),
            MediaAgent(),
            VerificationAgent(),
            ExplanationAgent(writer=writer),
        )
    }


class Orchestrator:
    """
    Runs one packet through the agents a planner chooses.

    Everything is injectable — planner, agents, tool registry — because
    that is what "provider agnostic" has to mean in practice: swapping the
    routing policy for a model, or one specialist for an LLM-backed
    version, must not require touching this file.
    """

    def __init__(self, planner=None, agents=None, tools=None, writer=None,
                 max_rounds=MAX_ROUNDS, max_steps=MAX_STEPS):
        self.planner = planner or RulePlanner()
        self.agents = agents if agents is not None else default_agents(writer=writer)
        self.tools = tools
        self.max_rounds = max_rounds
        self.max_steps = max_steps

    def run(self, packet, **options):
        """
        Run the agents over one packet.

        Returns the trace: the plan, every agent's structured result, the
        tool calls in order, and the context the caller needs to build a
        response (`backend.agents.runner` does that part).
        """
        started = time.perf_counter()

        options.setdefault("max_rounds", self.max_rounds)

        context = AgentContext(
            packet=dict(packet or {}),
            tools=self.tools if self.tools is not None else default_registry(),
            options=dict(options),
        )

        plan = self.planner.plan(context.observe(round=0))
        queue = list(plan.steps)
        history = list(plan.steps)

        dispatched = 0
        skipped = []
        handoffs = []
        rounds = 0

        while queue:
            if dispatched >= self.max_steps:
                context.notes.append(
                    f"stopped after {self.max_steps} steps: the step budget is spent"
                )
                break

            step = queue.pop(0)
            agent = self.agents.get(step.agent)

            if agent is None:
                skipped.append({"agent": step.agent, "why": "no such agent"})
                continue

            if step.agent in NEEDS_CLAIMS and not context.check_worthy():
                skipped.append(
                    {
                        "agent": step.agent,
                        "why": "there is nothing check-worthy in this message",
                    }
                )
                continue

            result = agent(context, step)
            dispatched += 1

            for target in result.handoffs:
                handoffs.append({"from": step.agent, "to": target})

            #  The graph is the run's, not the claim agent's. Built here,
            #  once, as soon as there is a claim set to build it from.
            if context.claimset is not None and context.graph is None:
                graph, call = context.call(
                    "graph.build",
                    context.claimset,
                    context.packet,
                    _agent="orchestrator",
                    _args={"claims": len(context.claimset.claims)},
                )

                if graph is None:
                    context.notes.append(
                        call.error or "the graph could not be built"
                    )
                else:
                    context.graph = graph

            if not queue:
                rounds += 1
                more = self._revise(context, rounds)

                if more:
                    queue.extend(more)
                    history.extend(more)

        report = {
            "planner": plan.planner,
            "plan": plan.as_dict(),
            "steps": [step.as_dict() for step in history],
            "dispatched": dispatched,
            "skipped": skipped,
            "handoffs": handoffs,
            "rounds": rounds,
            "results": {name: result.as_dict() for name, result in context.results.items()},
            "abstained": sorted(
                name for name, result in context.results.items() if result.abstained
            ),
            "failed": sorted(
                name for name, result in context.results.items()
                if result.status == "failed"
            ),
            "trace": [call.as_dict() for call in context.tools.calls],
            "notes": context.notes,
            "observation": context.observe(round=rounds),
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }

        return context, report

    def _revise(self, context, rounds):
        """Ask the planner for another round, and never let it raise."""
        if rounds >= context.option("max_rounds", self.max_rounds):
            return []

        try:
            return list(
                self.planner.revise(context.observe(round=rounds), context.results, rounds)
                or []
            )
        except Exception as error:                   # fail soft, never raise
            log.warning(
                "the planner could not revise (%s): %s", type(error).__name__, error
            )
            context.notes.append(
                f"re-planning failed ({type(error).__name__}: {error}); "
                "the run ends with what it has"
            )

            return []


__all__ = ["Orchestrator", "default_agents", "Agent", "AgentResult", "MAX_STEPS"]
