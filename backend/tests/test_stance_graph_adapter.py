"""
Stage 4's graph adapter: read evidence off the graph, decide, write edges.

The models are stubbed the same way `test_stance.py` stubs them — a
bag-of-words embedder and a scripted NLI table — so the real ranking,
batching and rating-override code runs without downloading anything.
"""

import re
import zlib
from datetime import date

import pytest

from backend.claims import extract_claims
from backend.evidence import EvidenceCandidate
from backend.graph import EvidenceGraph
from backend.stance import apply_stances


TODAY = date(2026, 9, 17)

CLAIM_TEXT = "SBI is giving Rs 5,000 cashback to every customer today."

# Worded the way a real fact-check is: it repeats enough of the claim to
# clear the relevance cutoff, then contradicts it.
REFUTING = (
    "SBI has confirmed that no cashback scheme exists. "
    "The message telling every customer they will get Rs 5,000 today is a scam."
)

SUPPORTING = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "The bank confirmed the offer in a statement on Monday."
)

OFF_TOPIC = (
    "Monsoon rainfall in Kerala was above average this week. "
    "Farmers in Wayanad district reported a good paddy harvest."
)


# --- stubs ------------------------------------------------------------------

STOP = {"the", "a", "an", "of", "in", "is", "to", "this", "has", "and", "that", "on"}
BUCKETS = 512


def _tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


@pytest.fixture
def models(monkeypatch):
    """A hashing embedder and an NLI model scripted by keyword."""
    def embed(texts):
        vectors = []

        for text in texts:
            vector = [0.0] * BUCKETS

            for token in _tokens(text):
                vector[zlib.crc32(token.encode("utf-8")) % BUCKETS] += 1.0

            vectors.append(vector)

        return vectors

    def distribution(label, top=0.9):
        rest = round((1.0 - top) / 2, 4)

        return {
            name: (top if name == label else rest)
            for name in ("entailment", "neutral", "contradiction")
        }

    def score_pairs(premises, hypothesis, batch_size=8):
        out = []

        for premise in premises:
            lowered = premise.lower()

            if "no cashback scheme exists" in lowered or "is a scam" in lowered:
                out.append(distribution("contradiction"))
            elif "confirmed the offer" in lowered or "giving rs 5,000" in lowered:
                out.append(distribution("entailment"))
            else:
                out.append(distribution("neutral"))

        return out

    monkeypatch.setattr("backend.stance.rank.embed", embed)
    monkeypatch.setattr("backend.stance.rank.available", lambda: True)
    monkeypatch.setattr("backend.stance.nli.available", lambda: True)
    monkeypatch.setattr("backend.stance.nli.score_pairs", score_pairs)

    yield


# --- fixtures ---------------------------------------------------------------


def packet(text=CLAIM_TEXT):
    return {"input_type": "text", "text": text, "source_date": None, "images": []}


def build_graph(text=CLAIM_TEXT):
    claims = extract_claims(packet(text), today=TODAY, backend="heuristic")
    graph = EvidenceGraph.from_claimset(claims, packet(text))

    return claims, graph


def add(graph, claim_id, evidence_id, text, **overrides):
    fields = {
        "claim_id": claim_id,
        "source_type": "web",
        "url": f"https://www.thehindu.com/{evidence_id}",
        "title": "Story",
        "snippet": text,
        "source_weight": 0.8,
    }
    fields.update(overrides)

    candidate = EvidenceCandidate(**fields)
    candidate.id = evidence_id

    return graph.add_evidence(claim_id, candidate)


def only_claim(claims):
    return claims.claims[0].id


# --- the pass ---------------------------------------------------------------


def test_refuting_evidence_becomes_a_refutes_edge(models):
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(graph, claim_id, "ev_refute", REFUTING)

    report = apply_stances(graph)

    assert report.refutes == 1
    assert report.supports == 0
    assert graph.stance_of("ev_refute", claim_id) == "refutes"
    assert graph.g.has_edge("ev_refute", claim_id, "REFUTES")

    edge = graph.g.edges["ev_refute", claim_id, "REFUTES"]

    assert edge["method"] == "nli"
    assert edge["best_passage"]
    assert edge["relevance"] > 0


def test_supporting_evidence_becomes_a_supports_edge(models):
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(graph, claim_id, "ev_support", SUPPORTING)

    report = apply_stances(graph)

    assert report.supports == 1
    assert graph.stance_of("ev_support", claim_id) == "supports"


def test_off_topic_evidence_becomes_neutral(models):
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(graph, claim_id, "ev_off", OFF_TOPIC)

    report = apply_stances(graph)

    assert report.neutral == 1
    assert graph.stance_of("ev_off", claim_id) == "neutral"
    assert report.by_method.get("off_topic") == 1


def test_a_publishers_rating_overrides_the_text(models):
    """
    The trap this stage exists for: a fact-check that quotes the false
    claim in its lead reads as *support* to an entailment model, while
    its own rating refutes it.
    """
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(
        graph, claim_id, "ev_factcheck",
        SUPPORTING,                       # text the stub model calls entailment
        source_type="factcheck",
        url="https://www.altnews.in/sbi-cashback",
        rating="false",
        rating_raw="False",
        source_weight=1.0,
    )

    report = apply_stances(graph)

    assert report.refutes == 1
    assert report.misleading == 1
    assert report.by_method.get("rating") == 1

    assert graph.stance_of("ev_factcheck", claim_id) == "refutes"

    edge = graph.g.edges["ev_factcheck", claim_id, "REFUTES"]

    assert edge["misleading"] is True
    assert "quotes the claim" in (edge.get("note") or "")
    assert graph.node("ev_factcheck")["misleading"] is True


def test_the_raw_rating_is_what_stage_four_reads(models):
    """'Pants on Fire' must reach the rating table, not just 'false'."""
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(
        graph, claim_id, "ev_pof", OFF_TOPIC,
        source_type="factcheck",
        rating="false",
        rating_raw="Pants on Fire",
        source_weight=1.0,
    )

    apply_stances(graph)

    edge = graph.g.edges["ev_pof", claim_id, "REFUTES"]

    assert "Pants on Fire" in (edge.get("note") or "")


def test_stance_totals_come_out_weighted(models):
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(graph, claim_id, "ev_refute", REFUTING,
        source_type="factcheck", rating_raw="False", source_weight=1.0)
    add(graph, claim_id, "ev_support", SUPPORTING, source_weight=0.3)

    apply_stances(graph)

    totals = graph.stance_totals(claim_id)

    assert totals["refutes"] == pytest.approx(1.0)        # rating: score 1.0 x weight 1.0
    assert 0 < totals["supports"] <= 0.3
    assert totals["margin"] < 0
    assert claim_id not in graph.open_claims()


def test_a_second_pass_replaces_rather_than_stacks(models):
    claims, graph = build_graph()
    claim_id = only_claim(claims)

    add(graph, claim_id, "ev_one", SUPPORTING)
    apply_stances(graph)

    assert graph.stance_of("ev_one", claim_id) == "supports"

    edges_before = graph.g.number_of_edges()

    # The publisher's rating arrives on a later pass (a fetch filled it in).
    graph.g.nodes["ev_one"]["rating_raw"] = "False"
    apply_stances(graph)

    assert graph.stance_of("ev_one", claim_id) == "refutes"
    assert not graph.g.has_edge("ev_one", claim_id, "SUPPORTS")
    assert graph.g.number_of_edges() == edges_before


def test_claim_ids_narrows_the_pass(models):
    claims, graph = build_graph(
        "SBI is giving Rs 5,000 cashback to every customer today. "
        "RBI announced a 2% cut in repo rate on 12/08/2026."
    )
    first, second = sorted(cid for cid, _ in graph.claims())[:2]

    add(graph, first, "ev_a", REFUTING)
    add(graph, second, "ev_b", REFUTING)

    report = apply_stances(graph, claim_ids=[first])

    assert report.claims == 1
    assert graph.stance_of("ev_a", first) == "refutes"
    assert graph.stance_of("ev_b", second) is None


def test_claims_without_evidence_are_not_counted(models):
    claims, graph = build_graph()

    report = apply_stances(graph)

    assert report.claims == 0
    assert report.evidence == 0
    assert graph.open_claims()


# --- failing soft -----------------------------------------------------------


def test_missing_models_leave_everything_neutral(monkeypatch):
    monkeypatch.setattr("backend.stance.rank.available", lambda: False)

    claims, graph = build_graph()
    claim_id = only_claim(claims)
    add(graph, claim_id, "ev_x", REFUTING)

    report = apply_stances(graph)

    assert report.neutral == 1
    assert report.by_method.get("unavailable") == 1
    assert graph.stance_of("ev_x", claim_id) == "neutral"
    assert report.errors == []          # abstaining is not an error


def test_a_raising_classifier_is_caught(monkeypatch, models):
    def explode(*args, **kwargs):
        raise RuntimeError("model went away")

    monkeypatch.setattr("backend.stance.graph_adapter.classify_stance", explode)

    claims, graph = build_graph()
    claim_id = only_claim(claims)
    add(graph, claim_id, "ev_x", REFUTING)

    report = apply_stances(graph)

    assert report.evidence == 0
    assert any("model went away" in error for error in report.errors)
    assert graph.stance_of("ev_x", claim_id) is None       # nothing written


def test_report_is_serialisable(models):
    claims, graph = build_graph()
    claim_id = only_claim(claims)
    add(graph, claim_id, "ev_x", REFUTING)

    data = apply_stances(graph).as_dict()

    assert data["refutes"] == 1
    assert data["by_method"] == {"nli": 1}
    assert data["errors"] == []
