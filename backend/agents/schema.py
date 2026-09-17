"""
The shapes agents speak in.

Every agent returns an `AgentResult` and nothing else — no bare dicts, no
tuples, no "it depends what happened". The orchestrator, the tests and
the API all read the same four fields (`status`, `confidence`, `data`,
`tool_calls`), which is what makes an agent replaceable: a rule-based
`EvidenceAgent` and an LLM-driven one are interchangeable as long as both
return this.

`status` carries the thing a verdict system needs most and a chat model
is worst at: the difference between *an answer* and *no answer*.

    ok         the agent did its job and its data can be used
    abstained  the agent ran, found the evidence insufficient, and says so
    skipped    the agent was not applicable (no media, retrieval off)
    failed     the agent raised; the trace holds the exception

`abstained` is not a failure and must never be read as one. An evidence
agent that found nothing has told you something true about the world; an
evidence agent that guessed would not have.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


AGENT_NAMES = ("claim", "evidence", "media", "verification", "explanation")

STATUSES = ("ok", "abstained", "skipped", "failed")


class ToolCall(BaseModel):
    """
    One call an agent made into stages 1-5.

    `args` and `summary` are deliberately small dicts rather than the real
    objects: a `ClaimSet` or a graph in a trace makes the trace unusable,
    and what a reader (or a test) wants is "which tool, with what, and
    what came back".
    """

    tool: str
    agent: Optional[str] = None
    args: Dict[str, Any] = {}
    ok: bool = True
    error: Optional[str] = None
    ms: float = 0.0
    summary: Dict[str, Any] = {}

    def as_dict(self):
        return self.model_dump(mode="json")


class AgentResult(BaseModel):
    """What one agent did. See the module docstring for `status`."""

    agent: str
    status: str = "ok"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data: Dict[str, Any] = {}
    notes: List[str] = []
    tool_calls: List[ToolCall] = []
    handoffs: List[str] = []
    ms: float = 0.0

    @property
    def abstained(self):
        return self.status == "abstained"

    @property
    def usable(self):
        return self.status == "ok"

    def as_dict(self):
        return self.model_dump(mode="json")


class PlanStep(BaseModel):
    """
    One dispatch: which agent, why, and with what.

    `reason` exists because a plan nobody can read is a plan nobody can
    debug. It goes into the trace verbatim.
    """

    agent: str
    reason: str = ""
    tools: List[str] = []
    args: Dict[str, Any] = {}
    round: int = 0

    def as_dict(self):
        return self.model_dump(mode="json")


class Plan(BaseModel):
    """
    The steps to run, and who decided them.

    `planner` names the policy that produced this (`rules`, `llm:...`), so
    a trace says whether a model or the fallback chose the route.
    """

    planner: str = "rules"
    steps: List[PlanStep] = []
    notes: List[str] = []

    def as_dict(self):
        return self.model_dump(mode="json")

    def agents(self):
        return [step.agent for step in self.steps]
