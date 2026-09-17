"""
Who decides which agents run: the routing policy, kept separate on purpose.

The orchestrator dispatches; it does not choose. Choosing happens here,
behind an interface with exactly two methods:

    plan(observation)                    -> Plan        the opening route
    revise(observation, results, round)  -> [PlanStep]  what to do next

`observation` is `AgentContext.observe()`: a flat, JSON-safe dict — input
type, how many claims are check-worthy and of what type, which retrievers
and media tools are actually usable, how many claims are still open. A
rule engine reads it as a dict; a model reads it as a prompt. That is the
whole reason routing was given an interface instead of an `if` ladder
inside the orchestrator.

`RulePlanner` is the default and is deterministic, which matters more
here than sophistication: with no API keys and no models installed this
project still has to produce a well-formed, explainable answer, and a
route that depends on a model being reachable would not. Its decisions
are the obvious ones — do not search when there is nothing check-worthy
to search for, do not run the media agent on a text forward, and do go
back out for more evidence when verification abstained and a source was
held in reserve.

`LLMPlanner` is the seam for a model, and it is written defensively on
purpose: it takes any callable that maps a prompt to JSON, validates what
comes back against the same `Plan` schema, refuses anything naming an
agent that does not exist, and falls back to `RulePlanner` on any
failure. A planner that hallucinated a step would otherwise be able to
skip verification.
"""

import json
import logging

from .schema import AGENT_NAMES, Plan, PlanStep


log = logging.getLogger(__name__)


MAX_ROUNDS = 2


class RulePlanner:
    """The default route: deterministic, offline, and explainable."""

    name = "rules"

    def plan(self, observation):
        """
        The opening route.

        The claim agent always runs: without claims there is nothing to
        check and no honest way to say anything about the message.
        Everything after it is conditional on what the observation says
        exists — and the verification and explanation agents are *not*
        conditional on evidence being found, because "nothing was found"
        is a verdict that still has to be reached and explained.
        """
        steps = [
            PlanStep(
                agent="claim",
                reason="every run starts by asking what is actually being claimed",
                tools=["claims.extract"],
            )
        ]
        notes = []

        if observation["options"].get("retrieve", True):
            steps.append(
                PlanStep(
                    agent="evidence",
                    reason="find out whether anyone credible has addressed these claims",
                    tools=["evidence.collect"],
                )
            )
        else:
            notes.append("retrieval is switched off: no evidence will be gathered")

        if observation["images"]:
            steps.append(
                PlanStep(
                    agent="media",
                    reason=(
                        f"{observation['images']} picture(s) came with this message; "
                        "a recycled photograph is the commonest kind of misinformation"
                    ),
                    tools=["images.keyframes", "images.collect"],
                )
            )

        steps.append(
            PlanStep(
                agent="verification",
                reason="combine the evidence, the stance, the graph and the media",
                tools=["stance.apply", "verdict.decide"],
            )
        )
        steps.append(
            PlanStep(
                agent="explanation",
                reason="say what was found, in one or two sentences",
            )
        )

        if observation["offline"]:
            notes.append(
                "no retriever is configured: this run will fall back to whatever "
                "is available locally and abstain rather than guess"
            )

        return Plan(planner=self.name, steps=steps, notes=notes)

    def revise(self, observation, results, round):
        """
        What to do after a plan has run through.

        One escalation and one only: verification abstained, claims are
        still open, and the evidence agent kept a source in reserve. Then
        it is worth another retrieval round on just those claims,
        followed by a re-verification and a re-explanation.

        Everything else returns nothing, which ends the run. An
        orchestrator that keeps trying because it is unhappy with the
        answer is a system that eventually reports whatever it wanted to
        find.
        """
        if round >= observation["options"].get("max_rounds", MAX_ROUNDS):
            return []

        verification = results.get("verification")

        if verification is None or not verification.abstained:
            return []

        evidence = results.get("evidence")

        if evidence is None or evidence.status == "skipped":
            return []

        held_back = evidence.data.get("held_back") or []
        open_claims = evidence.data.get("open_claims") or observation.get("open_claims")

        if not held_back or not open_claims:
            return []

        return [
            PlanStep(
                agent="evidence",
                reason=(
                    "verification could not settle this and "
                    + ", ".join(held_back)
                    + " has not been asked yet"
                ),
                tools=[f"evidence.{name}" for name in held_back],
                args={
                    "escalate": True,
                    "retrievers": held_back,
                    "claim_ids": list(open_claims)
                    if isinstance(open_claims, list) else None,
                },
                round=round + 1,
            ),
            PlanStep(
                agent="verification",
                reason="re-score the claims the second round touched",
                tools=["stance.apply", "verdict.decide"],
                round=round + 1,
            ),
            PlanStep(
                agent="explanation",
                reason="explain the verdict the second round produced",
                round=round + 1,
            ),
        ]


PROMPT = """You route a misinformation-checking pipeline.

Available agents:
  claim         extract and prioritise check-worthy claims (always run first)
  evidence      retrieve evidence for those claims
  media         verify images and video frames (only useful when images > 0)
  verification  combine evidence, stance, graph and media into a verdict
  explanation   write the final explanation (run last)

Observation:
{observation}

Reply with JSON only, in this shape:
{{"steps": [{{"agent": "claim", "reason": "why", "tools": []}}]}}
"""


class LLMPlanner:
    """
    Routing by a model, with the rules underneath it.

    `complete` is any callable `(prompt: str) -> str | dict`. That is the
    entire provider contract: no SDK, no client object, no streaming, no
    tool-calling protocol. An Anthropic, OpenAI, local-Ollama or
    HTTP-endpoint wrapper all fit it in a few lines, and the tests fit a
    lambda in one.

    Anything the model returns that is not a valid `Plan` over known
    agents is discarded and the rule planner answers instead, which is
    recorded in `Plan.notes` so a trace never hides that the fallback
    ran.
    """

    def __init__(self, complete, fallback=None, name="llm"):
        self.complete = complete
        self.fallback = fallback or RulePlanner()
        self.name = name

    def plan(self, observation):
        try:
            raw = self.complete(PROMPT.format(observation=json.dumps(observation, indent=2)))
            plan = self._parse(raw)

            if plan is not None:
                return plan

            reason = "the planner returned no usable steps"
        except Exception as error:                   # fail soft, never raise
            reason = f"{type(error).__name__}: {error}"
            log.warning("the LLM planner failed (%s); falling back to rules", reason)

        plan = self.fallback.plan(observation)
        plan.notes.append(f"fell back to the {self.fallback.name} planner: {reason}")

        return plan

    def revise(self, observation, results, round):
        """
        Escalation stays with the rules.

        A model is useful for choosing a route; deciding to spend another
        round of API calls because it is dissatisfied with an answer is
        exactly the loop a budget cannot survive.
        """
        return self.fallback.revise(observation, results, round)

    def _parse(self, raw):
        """A `Plan` from whatever the model said, or None."""
        if isinstance(raw, str):
            raw = json.loads(_json_block(raw))

        if not isinstance(raw, dict):
            return None

        steps = []

        for item in raw.get("steps") or []:
            if not isinstance(item, dict):
                continue

            agent = item.get("agent")

            if agent not in AGENT_NAMES:
                log.warning("the planner asked for an unknown agent %r; dropped", agent)
                continue

            steps.append(
                PlanStep(
                    agent=agent,
                    reason=str(item.get("reason") or "")[:300],
                    tools=[str(tool) for tool in (item.get("tools") or [])],
                    args=dict(item.get("args") or {}),
                )
            )

        if not steps:
            return None

        #  A route that never verifies cannot produce a verdict, and a
        #  route that never explains produces one nobody can read. Both
        #  are appended rather than rejected: the model's ordering of the
        #  specialists is worth keeping even when it forgets the ending.
        for required in ("verification", "explanation"):
            if required not in [step.agent for step in steps]:
                steps.append(
                    PlanStep(
                        agent=required,
                        reason=f"{required} is required on every route",
                    )
                )

        return Plan(planner=self.name, steps=steps, notes=[])


def _json_block(text):
    """The outermost JSON object in a model's reply."""
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("no JSON object in the planner's reply")

    return text[start:end + 1]
