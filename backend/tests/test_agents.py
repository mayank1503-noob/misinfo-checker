"""
The agent layer: routing, tool calls, handoffs, failures, aggregation.

Everything here runs offline. Retrievers are fakes handed to the tool
registry — `default_registry(retrievers=...)` exists for exactly that, so
no test patches module globals — and the models are switched off the same
way the stage 4 and pipeline tests switch them off. What is under test is
the coordination: which agents a plan reaches for, which tools they
actually call, what happens when one of them fails, whether an
abstention really does send the evidence agent back out, and whether the
final result is still the shape every existing caller reads.

No test here asserts anything about *how* a stage works. Stages 1-5 have
their own suites and this layer is not allowed to have opinions about
their internals.
"""

from datetime import date

import pytest

from backend.agents import (
    AgentResult,
    ClaimAgent,
    EvidenceAgent,
    ExplanationAgent,
    LLMPlanner,
    MediaAgent,
    Orchestrator,
    RulePlanner,
    VerificationAgent,
    default_registry,
    run_agentic,
)
from backend.agents.claim_agent import priority_of
from backend.agents.schema import STATUSES, Plan, PlanStep
from backend.agents.tools import Tool, ToolRegistry
from backend.evidence import EvidenceCandidate
from backend.pipeline import analyze, analyze_text
from backend.verdict import LABELS


TODAY = date(2026, 9, 17)

SCAM = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your reward."
)


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """No keys, an isolated cache, no image index, the regex claim backend."""
    for name in ("GOOGLE_FACTCHECK_KEY", "TAVILY_API_KEY", "SERPAPI_KEY"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("IMAGE_INDEX", str(tmp_path / "no-index.json"))
    monkeypatch.setenv("CLAIM_BACKEND", "heuristic")

    from backend.images import local_index

    local_index.reset()

    yield

    local_index.reset()


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    """
    The models unavailable, which is the configuration that must work.

    Stage 4 still decides a stance from a publisher's rating with no
    model at all, so an offline run is not a run with nothing in it.
    """
    monkeypatch.setattr("backend.stance.rank.available", lambda: False)
    monkeypatch.setattr("backend.stance.nli.available", lambda: False)
    monkeypatch.setattr("backend.images.consistency.available", lambda: False)

    from backend.evidence.retrievers import seed_index

    seed_index.reset()

    yield

    seed_index.reset()


def packet(text=SCAM, images=None, source_date=None, input_type="text"):
    return {
        "input_type": input_type,
        "text": text,
        "source_date": source_date,
        "images": images or [],
    }


def image(path="flood.jpg", description="a flooded street"):
    return {
        "path": path, "ocr_text": "", "description": description,
        "embedding": [0.1, 0.2, 0.3], "exif_date": None,
        "ai_generated_score": None, "frame_time": None,
    }


def factcheck_candidate(claim_id, url="https://www.altnews.in/sbi-cashback"):
    """A fact-check that refutes the scam claim, with no model needed."""
    return EvidenceCandidate(
        claim_id=claim_id,
        source_type="factcheck",
        url=url,
        title="No, SBI is not giving Rs 5,000 cashback to customers",
        snippet=(
            "SBI is giving Rs 5,000 cashback to every customer, claims a viral "
            "message. There is no such scheme, the bank confirmed."
        ),
        publisher="Alt News",
        rating_raw="False",
        published_date="2024-03-11",
    )


class FakeRetriever:
    """
    A retriever module, as the registry and stage 3b expect one.

    `available`, `search` and `search_many` are the whole contract
    (`backend/evidence/retrievers/__init__.py`), so a fake needs nothing
    more than this to be indistinguishable from the real three.
    """

    def __init__(self, candidates=None, up=True, explode=False):
        self.candidates = candidates
        self.up = up
        self.explode = explode
        self.queries = []

    def available(self):
        return self.up

    def search(self, claim_id, query):
        return self.search_many(claim_id, [query])

    def search_many(self, claim_id, queries):
        self.queries.append((claim_id, list(queries)))

        if self.explode:
            raise RuntimeError("this retriever is on fire")

        if self.candidates is None:
            return []

        return [self.candidates(claim_id)]


def registry_with(**retrievers):
    return default_registry(retrievers=retrievers)


# --- routing ----------------------------------------------------------------


def test_the_rule_planner_routes_a_plain_text_message():
    observation = {
        "images": 0, "check_worthy": 2, "claims": 2, "offline": False,
        "options": {"retrieve": True},
    }

    plan = RulePlanner().plan(observation)

    assert plan.agents() == ["claim", "evidence", "verification", "explanation"]
    assert all(step.reason for step in plan.steps)


def test_the_rule_planner_adds_the_media_agent_when_a_picture_came_with_it():
    plan = RulePlanner().plan({
        "images": 2, "check_worthy": 1, "claims": 1, "offline": False,
        "options": {"retrieve": True},
    })

    assert plan.agents() == [
        "claim", "evidence", "media", "verification", "explanation"
    ]


def test_the_rule_planner_leaves_retrieval_out_when_it_is_switched_off():
    plan = RulePlanner().plan({
        "images": 0, "check_worthy": 1, "claims": 1, "offline": True,
        "options": {"retrieve": False},
    })

    assert "evidence" not in plan.agents()
    assert "verification" in plan.agents()          # a verdict is still reached
    assert any("switched off" in note for note in plan.notes)


def test_the_rule_planner_says_so_when_nothing_is_configured():
    plan = RulePlanner().plan({
        "images": 0, "check_worthy": 1, "claims": 1, "offline": True,
        "options": {"retrieve": True},
    })

    assert any("no retriever is configured" in note for note in plan.notes)


def test_a_message_with_no_checkable_claim_is_not_searched():
    """
    The personal-chat case: nothing check-worthy, so no retrieval budget.

    The claim agent abstains, the orchestrator drops the steps that need
    claims, and the answer is `unverified` rather than an accusation.
    """
    result = run_agentic(packet("Hi. Ok."), backend="heuristic")

    report = result["agents"]

    assert report["results"]["claim"]["status"] == "abstained"
    assert [entry["agent"] for entry in report["skipped"]] == ["evidence"]
    assert result["verdict"]["label"] == "unverified"
    assert not [call for call in report["trace"] if call["tool"] == "evidence.collect"]


def test_the_media_agent_never_runs_on_a_text_message():
    result = run_agentic(packet(), backend="heuristic")

    assert "media" not in result["agents"]["results"]
    assert "media" not in result["agents"]["plan"]["steps"][0]["agent"]


def test_retrieval_off_is_honoured_by_the_agents_too():
    result = run_agentic(packet(), backend="heuristic", retrieve=False)

    assert "evidence" not in result["agents"]["results"]
    assert result["verdict"]["label"] == "unverified"


# --- tools ------------------------------------------------------------------


def test_every_tool_an_agent_names_exists_in_the_registry():
    registry = default_registry()
    agents = (
        ClaimAgent(), EvidenceAgent(), MediaAgent(),
        VerificationAgent(), ExplanationAgent(),
    )

    for agent in agents:
        for tool in agent.tools:
            assert tool in registry.names(), f"{agent.agent} names a missing {tool}"


def test_the_registry_reports_what_is_actually_usable():
    """With no keys, the network retrievers are not usable and say so."""
    registry = default_registry()

    assert registry.available("evidence.factcheck") is False
    assert registry.available("evidence.web") is False
    assert registry.available("graph.totals") is True      # no probe: always on
    assert registry.available("nope.nothing") is False


def test_a_tool_that_raises_is_recorded_rather_than_raised():
    registry = ToolRegistry()
    registry.register(Tool(
        name="boom",
        description="always fails",
        run=lambda: (_ for _ in ()).throw(RuntimeError("no")),
    ))

    value, call = registry.call("boom")

    assert value is None
    assert call.ok is False
    assert "RuntimeError: no" in call.error
    assert registry.called("boom") == [call]


def test_a_probe_is_asked_once_per_run():
    """
    Some probes import torch. Asking one four times while routing is not
    free, and a registry lives for exactly one run.
    """
    asked = []

    registry = ToolRegistry()
    registry.register(Tool(
        name="expensive",
        description="its probe costs real seconds",
        run=lambda: 1,
        probe=lambda: asked.append(1) or True,
    ))

    assert [registry.available("expensive") for _ in range(4)] == [True] * 4
    assert len(asked) == 1


def test_a_text_message_never_probes_the_image_models():
    """
    The CLIP probe imports torch; a text forward must not pay for it.

    Asserted on the observation the planner sees, because that is where
    the probe would be triggered from.
    """
    probed = []

    def never(*args, **kwargs):
        probed.append(1)

        return True

    tools = default_registry()
    tools.register(Tool(
        name="images.consistency", description="stub", run=lambda *a, **k: [],
        probe=never,
    ))
    tools.register(Tool(
        name="images.reverse_search", description="stub", run=lambda *a, **k: [],
        probe=never,
    ))

    report = run_agentic(packet(), backend="heuristic", tools=tools)["agents"]

    assert probed == []
    assert report["observation"]["media_tools"] == {
        "reverse_search": None, "consistency": None,
    }


def test_a_probe_that_raises_makes_its_tool_unavailable():
    registry = ToolRegistry()
    registry.register(Tool(
        name="flaky",
        description="its probe explodes",
        run=lambda: 1,
        probe=lambda: (_ for _ in ()).throw(OSError("no model")),
    ))

    assert registry.available("flaky") is False


def test_the_claim_agent_calls_stage_two_and_puts_the_scam_first():
    result = run_agentic(
        packet(
            "The RBI reduced the repo rate by 25 basis points on 07/06/2024. "
            "SBI is giving Rs 5,000 cashback, forward this to 10 people."
        ),
        backend="heuristic", today=TODAY,
    )

    claim = result["agents"]["results"]["claim"]

    assert [call["tool"] for call in claim["tool_calls"]] == ["claims.extract"]
    assert claim["status"] == "ok"

    priority = claim["data"]["priority"]

    assert len(priority) >= 2
    assert priority[0]["claim_type"] == "chain_offer"
    assert priority[0]["priority"] >= priority[1]["priority"]


def test_priority_prefers_the_kinds_of_claim_that_cost_people_money():
    class Fake:
        def __init__(self, claim_type, confidence):
            self.claim_type = claim_type
            self.confidence = confidence

    assert priority_of(Fake("chain_offer", 0.6)) > priority_of(Fake("generic", 0.6))
    assert priority_of(Fake("health", 0.6)) > priority_of(Fake("prediction", 0.6))
    #  ... but a confident extraction is never overruled by type alone
    assert priority_of(Fake("generic", 1.0)) > priority_of(Fake("chain_offer", 0.3))


def test_the_evidence_agent_asks_the_precise_sources_and_holds_the_web_back():
    tools = registry_with(
        factcheck=FakeRetriever(), seed_index=FakeRetriever(up=False),
        web=FakeRetriever(factcheck_candidate),
    )

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    evidence = result["agents"]["results"]["evidence"]

    assert evidence["data"]["retrievers"] == ["factcheck"]
    assert evidence["data"]["held_back"] == ["web"]

    collect = [
        call for call in evidence["tool_calls"] if call["tool"] == "evidence.collect"
    ]

    assert len(collect) == 1
    assert collect[0]["args"]["retrievers"] == ["factcheck"]


def test_the_evidence_agent_searches_the_web_when_no_fact_checker_is_configured():
    web = FakeRetriever(factcheck_candidate)
    tools = registry_with(
        factcheck=FakeRetriever(up=False), seed_index=FakeRetriever(up=False), web=web,
    )

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    evidence = result["agents"]["results"]["evidence"]

    assert evidence["data"]["retrievers"] == ["web"]
    assert evidence["data"]["held_back"] == []
    assert web.queries, "the web retriever should have been asked"
    assert any(
        "no fact-check source" in reason for reason in evidence["data"]["routing"]
    )


def test_the_evidence_agent_abstains_when_no_source_can_be_asked():
    tools = registry_with(
        factcheck=FakeRetriever(up=False), seed_index=FakeRetriever(up=False),
        web=FakeRetriever(up=False),
    )

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    evidence = result["agents"]["results"]["evidence"]

    assert evidence["status"] == "abstained"
    assert "no retriever is available" in evidence["notes"][0]
    assert result["verdict"]["label"] == "unverified"
    #  the explanation has to tell "could not ask" from "asked, found nothing"
    assert "no fact-check API key" in result["explanation"]["bullets"][0]


def test_the_evidence_agent_abstains_when_the_sources_had_nothing():
    tools = registry_with(factcheck=FakeRetriever(), web=FakeRetriever(up=False))

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    evidence = result["agents"]["results"]["evidence"]

    assert evidence["status"] == "abstained"
    assert "returned nothing" in evidence["notes"][0]
    assert "factcheck" in evidence["notes"][0]


def test_an_unavailable_media_tool_is_switched_off_rather_than_called(monkeypatch):
    monkeypatch.setattr("backend.images.reverse_search.available", lambda: False)

    result = run_agentic(
        packet(images=[image()], input_type="image"), backend="heuristic"
    )
    media = result["agents"]["results"]["media"]

    assert media["data"]["checks"] == {
        "local_index": True, "reverse_search": False, "consistency": False,
    }

    collect = [
        call for call in media["tool_calls"] if call["tool"] == "images.collect"
    ][0]

    assert collect["args"]["reverse_search"] is False
    assert collect["args"]["consistency"] is False


# --- media ------------------------------------------------------------------


def test_the_media_agent_reports_a_recycled_picture_and_it_reaches_the_verdict(
        monkeypatch):
    def search(embedding, top_k=3, floor=0.0, path=None):
        return [{
            "similarity": 0.98,
            "first_seen": "2015-12-02",
            "context": "Chennai floods, December 2015",
            "source_url": "https://example.org/chennai-2015",
            "domain": "example.org",
            "publisher": "Local image index",
            "demo": True,
        }]

    monkeypatch.setattr("backend.images.local_index.search", search)

    result = run_agentic(
        packet(
            "Chennai is flooded today and the whole city is under water.",
            images=[image()], source_date="2026-09-01", input_type="image",
        ),
        backend="heuristic", today=TODAY,
    )
    media = result["agents"]["results"]["media"]

    assert media["status"] == "ok"
    assert media["data"]["recycled"][0]["first_seen"] == "2015-12-02"
    assert media["data"]["index_matches"] >= 1

    assert result["verdict"]["label"] == "misleading"
    assert "2015-12-02" in result["explanation"]["explanation"]


def test_the_media_agent_abstains_when_nothing_is_known_about_the_picture():
    result = run_agentic(
        packet(images=[image()], input_type="image"), backend="heuristic"
    )
    media = result["agents"]["results"]["media"]

    assert media["status"] == "abstained"
    assert "nothing is known about them" in media["notes"][0]
    assert media["data"]["images"] == 1


def test_near_identical_frames_are_dropped_before_anything_is_checked():
    frames = [image(path=f"frame{index}.jpg") for index in range(4)]

    result = run_agentic(
        packet("Floods everywhere today in Chennai city.", images=frames,
               input_type="video"),
        backend="heuristic",
    )
    media = result["agents"]["results"]["media"]

    #  identical embeddings: one frame survives
    assert media["data"]["keyframes"] == 1
    assert media["data"]["dropped_frames"] == 3


# --- handoffs and escalation ------------------------------------------------


def test_an_abstaining_verification_sends_the_evidence_agent_back_out():
    """
    The handoff this layer exists for.

    The fact-check source is configured but has nothing; the web search
    is held back. Verification cannot settle the claim, abstains, and the
    planner spends the reserve on exactly the claims that are still open.
    """
    web = FakeRetriever(factcheck_candidate)
    tools = registry_with(factcheck=FakeRetriever(), web=web)

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    report = result["agents"]

    assert report["rounds"] == 2
    assert [step["agent"] for step in report["steps"]][-3:] == [
        "evidence", "verification", "explanation"
    ]

    evidence = report["results"]["evidence"]

    assert evidence["data"]["escalated"] is True
    assert evidence["status"] == "ok"
    assert len(evidence["data"]["rounds"]) == 2

    #  the second round replaces the agent's result; both rounds' calls survive
    tools_called = [call["tool"] for call in evidence["tool_calls"]]

    assert tools_called[0] == "evidence.collect"
    assert tools_called.count("evidence.web") >= 1
    assert "evidence.attach" in tools_called

    assert web.queries, "the reserve source was never actually asked"
    assert result["verdict"]["label"] == "false"
    assert "Alt News" in result["explanation"]["explanation"]


def test_handoffs_are_recorded_in_the_trace():
    tools = registry_with(factcheck=FakeRetriever(factcheck_candidate))

    report = run_agentic(packet(), backend="heuristic", tools=tools)["agents"]
    pairs = {(hop["from"], hop["to"]) for hop in report["handoffs"]}

    assert ("claim", "evidence") in pairs
    assert ("evidence", "verification") in pairs
    assert ("verification", "explanation") in pairs


def test_there_is_never_a_third_retrieval_round():
    """A loop that keeps going until it likes the answer is not a checker."""
    web = FakeRetriever()
    tools = registry_with(factcheck=FakeRetriever(), web=web)

    report = run_agentic(packet(), backend="heuristic", tools=tools)["agents"]

    assert report["rounds"] == 2
    assert len(report["results"]["evidence"]["data"]["rounds"]) == 2
    assert {claim_id for claim_id, _queries in web.queries}, "the reserve was tried"
    assert report["results"]["verification"]["status"] == "abstained"


def test_the_escalated_round_only_touches_claims_that_are_still_open():
    web = FakeRetriever()
    tools = registry_with(factcheck=FakeRetriever(), web=web)

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    second = result["agents"]["results"]["evidence"]["data"]["rounds"][1]
    open_after_first = result["agents"]["results"]["evidence"]["data"]["open_claims"]

    assert second["round"] == 2
    assert second["claims"]
    assert set(second["claims"]) <= set(open_after_first)
    assert second["max_queries"] > 3                # a wider net the second time


def test_no_escalation_when_there_is_nothing_left_in_reserve():
    tools = registry_with(factcheck=FakeRetriever())

    report = run_agentic(packet(), backend="heuristic", tools=tools)["agents"]

    assert report["rounds"] == 1
    assert report["results"]["evidence"]["data"]["escalated"] is False


# --- failures ---------------------------------------------------------------


def test_a_claim_extractor_that_explodes_does_not_take_the_run_down():
    tools = default_registry()
    tools.register(Tool(
        name="claims.extract",
        description="broken",
        run=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("extractor is down")),
    ))

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    report = result["agents"]

    assert report["results"]["claim"]["status"] == "failed"
    assert "extractor is down" in report["results"]["claim"]["notes"][0]
    assert result["verdict"]["label"] == "unverified"
    assert result["claims"] == []
    assert "extractor is down" in result["stages"]["claims"]["error"]


def test_a_retriever_that_explodes_becomes_an_abstention():
    tools = registry_with(factcheck=FakeRetriever(explode=True))

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    evidence = result["agents"]["results"]["evidence"]

    assert evidence["status"] == "abstained"
    assert any("on fire" in error for error in evidence["data"]["errors"])
    assert result["verdict"]["label"] == "unverified"


def test_a_verdict_stage_that_explodes_leaves_an_honest_placeholder():
    tools = default_registry()
    tools.register(Tool(
        name="verdict.decide",
        description="broken",
        run=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("verdict is down")),
    ))

    result = run_agentic(packet(), backend="heuristic", tools=tools)

    assert result["agents"]["results"]["verification"]["status"] == "failed"
    assert result["verdict"]["label"] == "unverified"
    assert "verdict is down" in result["verdict"]["summary"]
    assert result["agents"]["results"]["explanation"]["status"] == "abstained"


def test_an_agent_that_raises_is_reported_as_failed_not_propagated():
    class Exploding(EvidenceAgent):
        def run(self, context, step):
            raise RuntimeError("this agent is on fire")

    agents = Orchestrator().agents | {"evidence": Exploding()}

    result = run_agentic(packet(), backend="heuristic", agents=agents)

    assert result["agents"]["results"]["evidence"]["status"] == "failed"
    assert result["agents"]["failed"] == ["evidence"]
    assert result["verdict"]["label"] in LABELS


def test_a_plan_naming_an_agent_that_does_not_exist_skips_it():
    class Ghosts:
        name = "ghosts"

        def plan(self, observation):
            return Plan(planner=self.name, steps=[
                PlanStep(agent="claim", reason="real"),
                PlanStep(agent="poltergeist", reason="not real"),
                PlanStep(agent="verification", reason="real"),
                PlanStep(agent="explanation", reason="real"),
            ])

        def revise(self, observation, results, round):
            return []

    report = run_agentic(packet(), backend="heuristic", planner=Ghosts())["agents"]

    assert {"agent": "poltergeist", "why": "no such agent"} in report["skipped"]
    assert report["results"]["verification"]["data"]["label"] in LABELS


def test_a_planner_that_cannot_revise_ends_the_run_cleanly():
    class Fragile(RulePlanner):
        def revise(self, observation, results, round):
            raise RuntimeError("the planner broke")

    report = run_agentic(packet(), backend="heuristic", planner=Fragile())["agents"]

    assert any("re-planning failed" in note for note in report["notes"])
    assert "explanation" in report["results"]


def test_a_failing_explanation_writer_falls_back_to_the_template():
    def writer(brief, bullets):
        raise RuntimeError("the model is down")

    result = run_agentic(packet(), backend="heuristic", writer=writer)

    assert result["explanation"]["explanation"]
    assert result["explanation"]["explanation"].startswith("This could not be verified")


# --- aggregation and the response contract ----------------------------------


def test_the_agentic_result_has_the_shape_every_caller_already_reads():
    linear = analyze(packet(), backend="heuristic")
    agentic = run_agentic(packet(), backend="heuristic")

    assert set(agentic) >= set(linear)
    assert set(agentic) - set(linear) == {"agents", "explanation"}
    assert agentic["verdict"]["label"] in LABELS
    assert set(agentic["verdict"]) >= {"label", "confidence", "summary", "counts", "claims"}


def test_the_stages_block_is_rebuilt_from_the_tool_trace():
    tools = registry_with(factcheck=FakeRetriever(factcheck_candidate))

    result = run_agentic(packet(), backend="heuristic", tools=tools)
    stages = result["stages"]

    assert set(stages) >= {"claims", "graph", "evidence", "stance", "verdict"}

    for name, stage in stages.items():
        assert "ms" in stage and "error" in stage, name

    assert stages["claims"]["agent"] == "claim"
    assert stages["graph"]["agent"] == "orchestrator"
    assert stages["evidence"]["kept"] >= 1
    assert stages["evidence"]["rounds"] == 1


def test_the_same_evidence_gives_the_same_label_either_way(monkeypatch):
    """
    The agent layer must not move a label.

    Same stubbed fact-check, same rules underneath: the linear pipeline
    and the agentic one have to agree, or the labels are not the existing
    labels any more.
    """
    class Fake(FakeRetriever):
        pass

    monkeypatch.setattr(
        "backend.evidence.collector.RETRIEVERS",
        {"factcheck": Fake(factcheck_candidate)},
    )
    linear = analyze_text(SCAM, backend="heuristic")

    tools = registry_with(factcheck=Fake(factcheck_candidate))
    agentic = run_agentic(packet(), backend="heuristic", tools=tools)

    assert linear["verdict"]["label"] == agentic["verdict"]["label"] == "false"
    assert linear["verdict"]["summary"] == agentic["verdict"]["summary"]


def test_the_explanation_is_added_beside_the_summary_never_over_it():
    tools = registry_with(factcheck=FakeRetriever(factcheck_candidate))

    verdict = run_agentic(packet(), backend="heuristic", tools=tools)["verdict"]

    assert verdict["summary"].startswith("This message contains a false claim")
    assert verdict["explanation"] != verdict["summary"]
    assert "Alt News" in verdict["explanation"]


def test_every_agent_result_is_a_structured_agent_result():
    tools = registry_with(factcheck=FakeRetriever(factcheck_candidate))

    report = run_agentic(packet(), backend="heuristic", tools=tools)["agents"]

    for name, row in report["results"].items():
        restored = AgentResult(**row)

        assert restored.agent == name
        assert restored.status in STATUSES
        assert 0.0 <= restored.confidence <= 1.0
        assert isinstance(restored.data, dict)


def test_the_trace_names_the_agent_behind_every_tool_call():
    tools = registry_with(factcheck=FakeRetriever(factcheck_candidate))

    report = run_agentic(packet(), backend="heuristic", tools=tools)["agents"]

    assert report["trace"]
    assert all(call["agent"] for call in report["trace"])
    assert all(call["ms"] >= 0 for call in report["trace"])


def test_nothing_found_stays_unverified_and_says_which_kind_of_nothing():
    result = run_agentic(packet(), backend="heuristic")

    assert result["verdict"]["label"] == "unverified"
    assert result["verdict"]["confidence"] < 0.5
    assert result["explanation"]["abstained"] is True
    assert result["agents"]["abstained"] == [
        "evidence", "explanation", "verification"
    ]


def test_the_five_samples_all_run_through_the_agents():
    import os

    directory = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "samples")

    for name in ("fake_upi", "hinglish_rumor", "personal_chat",
                 "recycled_rumor", "true_claim"):
        with open(os.path.join(directory, f"{name}.txt"), encoding="utf-8") as handle:
            text = handle.read().strip()

        result = run_agentic(packet(text), backend="heuristic")

        assert result["verdict"]["label"] in LABELS
        assert result["verdict"]["summary"]
        assert result["agents"]["results"]["claim"]["status"] in STATUSES


def test_the_pipeline_can_delegate_to_the_agents():
    result = analyze(packet(), backend="heuristic", agentic=True)

    assert "agents" in result
    assert result["verdict"]["label"] in LABELS


def test_the_graph_can_still_be_returned_whole():
    from backend.graph import EvidenceGraph

    result = run_agentic(packet(), backend="heuristic", graph_json=True)
    restored = EvidenceGraph.from_dict(result["graph"])

    assert restored.packet_id == result["graph"]["packet_id"]


# --- the API ----------------------------------------------------------------


@pytest.fixture
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    from backend.api.main import app

    return fastapi_testclient.TestClient(app)


def test_the_agent_layer_is_opt_in_on_the_api(client):
    plain = client.post("/check/text", json={"text": SCAM}).json()
    agentic = client.post("/check/text?agentic=true", json={"text": SCAM}).json()

    assert "agents" not in plain
    assert "explanation" not in plain

    assert agentic["verdict"]["label"] in LABELS
    assert agentic["verdict"]["summary"]
    assert agentic["results"] == [] or set(agentic["results"][0]) >= {
        "claim", "label", "confidence", "explanation", "reasons"
    }
    assert agentic["agents"]["plan"]["steps"]
    assert agentic["explanation"]["explanation"]
    assert set(agentic) >= set(plain)


def test_an_agentic_request_still_validates_its_input(client):
    assert client.post(
        "/check/text?agentic=true", json={"text": "  "}
    ).status_code == 422


# --- the planner seam -------------------------------------------------------


def test_a_model_can_choose_the_route():
    def complete(prompt):
        assert "Observation" in prompt

        return """Sure, here is the plan:
        {"steps": [{"agent": "claim", "reason": "start here", "tools": []},
                   {"agent": "evidence", "reason": "look it up"}]}"""

    plan = LLMPlanner(complete, name="llm:test").plan({"a": 1})

    assert plan.planner == "llm:test"
    assert plan.agents() == ["claim", "evidence", "verification", "explanation"]
    assert plan.steps[0].reason == "start here"


def test_a_model_route_that_forgets_to_verify_is_completed_not_trusted():
    plan = LLMPlanner(lambda prompt: {"steps": [{"agent": "claim"}]}).plan({})

    assert plan.agents()[-2:] == ["verification", "explanation"]


def test_an_unknown_agent_from_a_model_is_dropped():
    plan = LLMPlanner(
        lambda prompt: {"steps": [
            {"agent": "claim"}, {"agent": "rm -rf", "reason": "trust me"},
        ]}
    ).plan({})

    assert "rm -rf" not in plan.agents()
    assert plan.agents()[0] == "claim"


@pytest.mark.parametrize(
    "complete",
    [
        lambda prompt: (_ for _ in ()).throw(RuntimeError("the model is down")),
        lambda prompt: "I would rather not.",
        lambda prompt: {"steps": []},
        lambda prompt: {"steps": [{"agent": "nonsense"}]},
    ],
)
def test_a_misbehaving_model_falls_back_to_the_rules(complete):
    observation = {
        "images": 0, "check_worthy": 1, "claims": 1, "offline": True,
        "options": {"retrieve": True},
    }

    plan = LLMPlanner(complete).plan(observation)

    assert plan.planner == "rules"
    assert plan.agents() == ["claim", "evidence", "verification", "explanation"]
    assert any("fell back to the rules planner" in note for note in plan.notes)


def test_escalation_stays_with_the_rules_even_behind_a_model():
    planner = LLMPlanner(lambda prompt: {"steps": [{"agent": "claim"}]})
    web = FakeRetriever(factcheck_candidate)
    tools = registry_with(factcheck=FakeRetriever(), web=web)

    result = run_agentic(
        packet(), backend="heuristic", tools=tools, planner=planner
    )

    #  The model's route skipped evidence entirely, so nothing is held in
    #  reserve and there is nothing to escalate - and the run still ends
    #  with a verdict and an explanation.
    assert result["verdict"]["label"] in LABELS
    assert result["agents"]["results"]["explanation"]["status"] in STATUSES
    assert web.queries == []


def test_a_writer_can_replace_the_explanation_prose():
    seen = {}

    def writer(brief, bullets):
        seen["brief"] = brief
        seen["bullets"] = bullets

        return "Checked against nothing at all."

    tools = registry_with(factcheck=FakeRetriever(factcheck_candidate))
    result = run_agentic(packet(), backend="heuristic", tools=tools, writer=writer)

    assert result["explanation"]["explanation"] == "Checked against nothing at all."
    assert seen["brief"]["label"] == "false"
    assert seen["brief"]["sources"] or seen["bullets"]
    #  the brief must be promptable: JSON-safe, no graphs, no claim objects
    import json

    json.dumps(seen["brief"])
