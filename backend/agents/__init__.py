"""
The agent layer — a coordinator over stages 1-5, not a replacement for them.

Stages 1-5 are unchanged and remain the only code that extracts a claim,
retrieves evidence, scores a stance, checks an image or decides a label.
This package adds the thing a fixed pipeline cannot do: decide what is
worth doing for *this* message, notice when the answer is not good
enough, and go back out for more.

    from backend.agents import run_agentic

    result = run_agentic(packet)          # same shape as pipeline.analyze
    result["verdict"]["label"]
    result["agents"]["results"]["evidence"]["status"]   # ok / abstained / ...

Six agents, each with one job:

    orchestrator   coordinates: plans, dispatches, carries handoffs, stops
    claim          stage 2, plus which claims are worth the retrieval budget
    evidence       which retrievers to ask, and whether one pass was enough
    media          stage 5, when there is media and the tools for it exist
    verification   stage 4 + the graph + media findings -> stage 6's rules
    explanation    a short account of what was actually found

Four properties the design is built around:

  * **Every stage function is a tool** (`tools.py`), named and probed. No
    stage logic is duplicated here.
  * **Every agent returns an `AgentResult`** (`schema.py`) — a status, a
    confidence, structured data and its tool calls.
  * **Abstention is a first-class outcome.** An agent that cannot
    conclude says so; the verdict stays `unverified` rather than becoming
    a guess.
  * **Nothing is provider-bound.** Routing is a `Planner` (rules by
    default, `LLMPlanner` for a model) and the explanation writer is a
    callable. With no keys, no models and no network, the rule planner
    and the template writer run the whole layer offline.
"""

from .base import Agent
from .claim_agent import ClaimAgent
from .context import AgentContext
from .evidence_agent import EvidenceAgent
from .explanation_agent import ExplanationAgent
from .media_agent import MediaAgent
from .orchestrator import MAX_STEPS, Orchestrator, default_agents
from .policy import MAX_ROUNDS, LLMPlanner, RulePlanner
from .runner import run_agentic
from .schema import AGENT_NAMES, STATUSES, AgentResult, Plan, PlanStep, ToolCall
from .tools import Tool, ToolRegistry, default_registry
from .verification_agent import VerificationAgent


AGENTS = (
    ClaimAgent,
    EvidenceAgent,
    MediaAgent,
    VerificationAgent,
    ExplanationAgent,
)


__all__ = [
    "run_agentic",
    "Orchestrator",
    "default_agents",
    "AGENTS",
    "AGENT_NAMES",
    "STATUSES",
    "Agent",
    "AgentContext",
    "AgentResult",
    "Plan",
    "PlanStep",
    "ToolCall",
    "Tool",
    "ToolRegistry",
    "default_registry",
    "RulePlanner",
    "LLMPlanner",
    "MAX_ROUNDS",
    "MAX_STEPS",
    "ClaimAgent",
    "EvidenceAgent",
    "MediaAgent",
    "VerificationAgent",
    "ExplanationAgent",
]
