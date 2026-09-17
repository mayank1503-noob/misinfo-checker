"""
What every agent is, and the three things the base class guarantees.

**It returns an `AgentResult`, always.** `__call__` wraps `run`, so an
agent that raises produces `status="failed"` with the exception text in
its notes rather than an exception the orchestrator has to catch. The
stages below fail soft; so does this layer, for the same reason: one
broken specialist must not cost the caller a verdict.

**It is timed.** `ms` on the result, per agent, next to the per-tool
timings in the trace.

**It can say no.** `abstain()` is a first-class outcome, not an error
path. An agent with insufficient evidence returns `abstained`, the
orchestrator may route around it, and the explanation says so — the
alternative is a confident label built out of nothing, which is the one
failure mode this whole project exists to avoid.

Writing an agent means subclassing this and implementing `run`:

    class MyAgent(Agent):
        agent = "my"
        description = "..."

        def run(self, context, step):
            value, _call = context.call("some.tool", ...)

            if value is None:
                return self.abstain("the tool had nothing")

            return self.ok(data={"value": value}, confidence=0.6)

`run` gets the shared context and the `PlanStep` that dispatched it, so
the same agent class can behave differently on a second, escalated pass
(`step.args`) without the orchestrator needing a second class.
"""

import logging
import time

from .schema import AgentResult


log = logging.getLogger(__name__)


class Agent:
    """Base class: timing, fail-soft, and the four result constructors."""

    agent = "agent"
    description = ""

    #  Tools this agent may reach for. Advisory: it documents the agent
    #  and lets a planner name tools per step, but nothing enforces it -
    #  the registry is the real boundary.
    tools = ()

    def run(self, context, step):
        raise NotImplementedError

    def __call__(self, context, step):
        started = time.perf_counter()
        previous, context.current = context.current, None

        result = AgentResult(agent=self.agent)
        context.current = result

        try:
            produced = self.run(context, step)

            if produced is not None:
                #  `run` builds its result with the helpers below, which
                #  return a fresh object; carry over the tool calls that
                #  were filed against the placeholder while it ran. Calls
                #  the agent put on its own result come first: those are
                #  history it chose to carry (an earlier round of its
                #  own), and the placeholder's are what just happened.
                produced.tool_calls = produced.tool_calls + result.tool_calls
                result = produced
        except Exception as error:                   # fail soft, never raise
            log.warning(
                "agent %s failed (%s): %s", self.agent, type(error).__name__, error
            )
            result.status = "failed"
            result.confidence = 0.0
            result.notes.append(f"{type(error).__name__}: {error}")

        result.agent = self.agent
        result.ms = round((time.perf_counter() - started) * 1000, 1)
        context.current = previous
        context.results[self.agent] = result

        return result

    #  --- result constructors ------------------------------------------

    def ok(self, data=None, confidence=0.5, notes=(), handoffs=()):
        return AgentResult(
            agent=self.agent,
            status="ok",
            confidence=confidence,
            data=dict(data or {}),
            notes=list(notes),
            handoffs=list(handoffs),
        )

    def abstain(self, reason, data=None, handoffs=()):
        """
        Ran, could not conclude, and says why.

        Confidence is zero by definition: an abstention that carried a
        number would invite someone to threshold on it.
        """
        return AgentResult(
            agent=self.agent,
            status="abstained",
            confidence=0.0,
            data=dict(data or {}),
            notes=[reason],
            handoffs=list(handoffs),
        )

    def skip(self, reason, data=None):
        """Not applicable: no media to check, retrieval switched off."""
        return AgentResult(
            agent=self.agent,
            status="skipped",
            confidence=0.0,
            data=dict(data or {}),
            notes=[reason],
        )

    def fail(self, reason, data=None):
        """A failure the agent noticed itself, rather than one it raised."""
        return AgentResult(
            agent=self.agent,
            status="failed",
            confidence=0.0,
            data=dict(data or {}),
            notes=[reason],
        )
