import json
from datetime import date

import pytest

from backend.claims import extract_claims
from backend.evidence import (
    Evidence,
    EvidenceGraph,
    EvidenceSource,
    evidence_id,
)


TODAY = date(2026, 9, 17)

SCAM_TEXT = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your SBI reward. "
    "RBI announced a 2% cut in repo rate on 12/08/2026."
)


def text_packet(text):
    return {
        "input_type": "text",
        "text": text,
        "source_date": None,
        "images": [],
    }


def build_claimset(text=SCAM_TEXT):
    return extract_claims(
        text_packet(text), today=TODAY, backend="heuristic"
    )


def build_graph(text=SCAM_TEXT):
    return EvidenceGraph.from_claimset(build_claimset(text))


def fact_check(url="https://www.altnews.in/sbi-cashback-scam", relation="refutes",
               relevance=0.9, credibility=0.9, text="No such SBI cashback scheme exists."):
    return Evidence(
        evidence_type="fact_check",
        text=text,
        relation=relation,
        relevance=relevance,
        credibility=credibility,
        source=EvidenceSource(
            url=url,
            title="No, SBI is not giving Rs 5,000 cashback",
            published="2026-09-10",
            retrieved="2026-09-17",
            rating="False",
        ),
    )


def first_claim(graph):
    return sorted(graph.claims)[0]


# --- construction from a ClaimSet -------------------------------------------


def test_from_claimset_carries_packet_and_claim_nodes():
    claimset = build_claimset()
    graph = EvidenceGraph.from_claimset(claimset)

    assert graph.packet_id == claimset.packet_id
    assert graph.input_type == "text"
    assert graph.nodes[graph.packet_id].kind == "packet"

    assert len(graph.claims) == len(claimset.claims)

    for claim in claimset.claims:
        node = graph.nodes[claim.id]
        assert node.kind == "claim"
        assert node.get("text") == claim.text
        assert node.get("claim_type") == claim.claim_type


def test_from_claimset_matches_the_seed_exactly():
    """The skeleton is consumed, not re-derived: no nodes/edges invented."""
    claimset = build_claimset()
    seed = claimset.to_graph_seed()
    graph = EvidenceGraph.from_claimset(claimset)

    assert [n["id"] for n in seed["nodes"]] == list(graph.nodes)
    assert [e["type"] for e in seed["edges"]] == [e.type for e in graph.edges]


def test_entity_nodes_and_mentions_edges():
    claimset = build_claimset()
    graph = EvidenceGraph.from_claimset(claimset)

    assert graph.entities, "the scam text should yield entities (SBI, Rs 5,000, ...)"

    for node in graph.entities.values():
        assert node.kind == "entity"
        assert node.get("label")

    for claim in claimset.claims:
        mentioned = {node.id for node in graph.get_entities(claim.id)}
        assert mentioned == {e.id for e in claim.entities}


def test_packet_has_claim_edges_cover_every_claim():
    graph = build_graph()

    targets = {
        edge.dst
        for edge in graph.edges_of_type("HAS_CLAIM")
        if edge.src == graph.packet_id
    }

    assert targets == set(graph.claims)


def test_shares_entity_links_claims_that_name_sbi():
    graph = build_graph()

    shared = graph.edges_of_type("SHARES_ENTITY")

    assert shared, "both SBI sentences should be linked"

    for edge in shared:
        assert edge.src in graph.claims
        assert edge.dst in graph.claims
        assert edge.get("via") in graph.entities


# --- adding evidence --------------------------------------------------------


def test_add_evidence_creates_node_and_edges():
    graph = build_graph()
    claim_id = first_claim(graph)

    stored = graph.add_evidence(claim_id, fact_check())

    assert stored.id in graph.evidence
    assert graph.nodes[stored.id].kind == "evidence"
    assert graph.nodes[stored.id].get("publisher") == "altnews.in"

    has_evidence = [
        edge for edge in graph.edges_of_type("HAS_EVIDENCE") if edge.src == claim_id
    ]
    assert [edge.dst for edge in has_evidence] == [stored.id]

    refutes = graph.edges_of_type("REFUTES")
    assert [(edge.src, edge.dst) for edge in refutes] == [(stored.id, claim_id)]


def test_add_evidence_rejects_unknown_claim():
    graph = build_graph()

    with pytest.raises(KeyError):
        graph.add_evidence("clm_does_not_exist", fact_check())


def test_add_evidence_rejects_a_non_claim_node():
    graph = build_graph()

    with pytest.raises(ValueError):
        graph.add_evidence(graph.packet_id, fact_check())


def test_supports_relation():
    graph = build_graph()
    claim_id = first_claim(graph)

    stored = graph.add_evidence(
        claim_id,
        Evidence(
            evidence_type="official",
            text="RBI cut the repo rate by 2% on 12 August 2026.",
            relation="supports",
            relevance=0.95,
            credibility=1.0,
            source=EvidenceSource(url="https://rbi.org.in/press/repo-2026"),
        ),
    )

    assert [e.dst for e in graph.edges_of_type("SUPPORTS")] == [claim_id]
    assert graph.get_claim_evidence(claim_id, relation="supports") == [stored]
    assert graph.get_claim_evidence(claim_id, relation="refutes") == []


def test_refutes_relation():
    graph = build_graph()
    claim_id = first_claim(graph)

    stored = graph.add_evidence(claim_id, fact_check())

    assert [e.dst for e in graph.edges_of_type("REFUTES")] == [claim_id]
    assert graph.get_claim_evidence(claim_id, relation="refutes") == [stored]


def test_uncertain_relation():
    graph = build_graph()
    claim_id = first_claim(graph)

    stored = graph.add_evidence(
        claim_id,
        Evidence(
            evidence_type="news",
            text="Banks have run cashback promotions before.",
            relation="uncertain",
            source=EvidenceSource(url="https://example.com/cashback"),
        ),
    )

    assert [e.dst for e in graph.edges_of_type("UNCERTAIN_FOR")] == [claim_id]
    assert graph.get_claim_evidence(claim_id, relation="uncertain") == [stored]


def test_add_relation_replaces_the_previous_stance():
    graph = build_graph()
    claim_id = first_claim(graph)
    stored = graph.add_evidence(claim_id, fact_check(relation="uncertain"))

    graph.add_relation(claim_id, stored.id, "refutes", note="rating: False")

    assert graph.edges_of_type("UNCERTAIN_FOR") == []
    assert len(graph.edges_of_type("REFUTES")) == 1
    assert graph.nodes[stored.id].get("relation") == "refutes"

    records = graph.get_claim_relations(claim_id)
    assert len(records) == 1
    assert records[0].relation == "refutes"
    assert records[0].note == "rating: False"


def test_add_relation_rejects_an_unknown_relation():
    graph = build_graph()
    claim_id = first_claim(graph)
    stored = graph.add_evidence(claim_id, fact_check())

    with pytest.raises(ValueError):
        graph.add_relation(claim_id, stored.id, "maybe")


def test_evidence_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        Evidence(evidence_type="tarot_reading", text="x")


# --- multiple evidence items ------------------------------------------------


def test_multiple_evidence_items_for_one_claim_are_ranked():
    graph = build_graph()
    claim_id = first_claim(graph)

    weak = graph.add_evidence(
        claim_id,
        Evidence(
            evidence_type="news",
            text="Readers report a cashback message.",
            relation="uncertain",
            relevance=0.4,
            credibility=0.5,
            source=EvidenceSource(url="https://example.com/a"),
        ),
    )
    strong = graph.add_evidence(claim_id, fact_check())
    middling = graph.add_evidence(
        claim_id,
        Evidence(
            evidence_type="rumor_index",
            text="Known UPI cashback chain message.",
            relation="refutes",
            relevance=0.8,
            credibility=0.7,
            source=EvidenceSource(publisher="local-index", title="upi-cashback-2024"),
        ),
    )

    assert graph.get_claim_evidence(claim_id) == [strong, middling, weak]
    assert len(graph.edges_of_type("HAS_EVIDENCE")) == 3
    assert len(graph.get_claim_relations(claim_id)) == 3
    assert graph.summary()["by_kind"]["evidence"] == 3


def test_the_same_evidence_on_two_claims_is_one_node():
    graph = build_graph()
    claim_ids = sorted(graph.claims)[:2]

    assert len(claim_ids) == 2

    for claim_id in claim_ids:
        graph.add_evidence(claim_id, fact_check())

    assert len(graph.evidence) == 1
    assert len(graph.nodes_of_kind("evidence")) == 1
    assert len(graph.edges_of_type("HAS_EVIDENCE")) == 2
    assert len(graph.edges_of_type("REFUTES")) == 2


def test_re_adding_the_same_evidence_is_idempotent():
    graph = build_graph()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, fact_check())
    before = len(graph.edges)
    graph.add_evidence(claim_id, fact_check())

    assert len(graph.edges) == before
    assert len(graph.evidence) == 1
    assert len(graph.get_claim_relations(claim_id)) == 1


# --- deterministic ids ------------------------------------------------------


def test_graph_ids_are_deterministic_across_runs():
    first = build_graph()
    second = build_graph()

    assert list(first.nodes) == list(second.nodes)
    assert [e.key for e in first.edges] == [e.key for e in second.edges]


def test_evidence_ids_are_deterministic_and_url_keyed():
    assert fact_check().id == fact_check().id

    # retrieval date is not part of the identity
    later = fact_check()
    later.source.retrieved = "2027-01-01"

    assert evidence_id(later.evidence_type, later.source, later.text) == fact_check().id

    other = fact_check(url="https://www.altnews.in/some-other-story")

    assert other.id != fact_check().id


def test_evidence_without_a_url_is_identified_by_publisher_and_text():
    def make(text):
        return Evidence(
            evidence_type="rumor_index",
            text=text,
            source=EvidenceSource(publisher="local-index", title="upi-cashback-2024"),
        )

    assert make("same").id == make("same").id
    assert make("same").id != make("different").id


def test_whole_graph_is_reproducible_including_evidence():
    def built():
        graph = build_graph()
        claim_id = first_claim(graph)
        graph.add_evidence(claim_id, fact_check())
        return graph.to_dict()

    assert built() == built()


# --- serialisation ----------------------------------------------------------


def test_to_dict_is_json_serialisable_and_complete():
    graph = build_graph()
    claim_id = first_claim(graph)
    stored = graph.add_evidence(claim_id, fact_check())

    data = graph.to_dict()
    round_tripped = json.loads(json.dumps(data))

    assert round_tripped == data
    assert data["packet_id"] == graph.packet_id
    assert data["input_type"] == "text"

    assert {n["id"] for n in data["claims"]} == set(graph.claims)
    assert {n["id"] for n in data["entities"]} == set(graph.entities)
    assert [e["id"] for e in data["evidence"]] == [stored.id]
    assert data["relations"][0] == {
        "claim_id": claim_id,
        "evidence_id": stored.id,
        "relation": "refutes",
        "relevance": 0.9,
        "credibility": 0.9,
        "note": None,
    }

    edge = next(e for e in data["edges"] if e["type"] == "REFUTES")
    assert edge["from"] == stored.id and edge["to"] == claim_id

    assert data["summary"]["by_kind"]["claim"] == len(graph.claims)


def test_from_dict_round_trips():
    graph = build_graph()
    claim_id = first_claim(graph)
    graph.add_evidence(claim_id, fact_check())

    restored = EvidenceGraph.from_dict(json.loads(json.dumps(graph.to_dict())))

    assert restored.to_dict() == graph.to_dict()
    assert restored.get_claim_evidence(claim_id) == graph.get_claim_evidence(claim_id)


def test_apply_to_claimset_writes_evidence_ids_back():
    claimset = build_claimset()
    graph = EvidenceGraph.from_claimset(claimset)
    claim_id = claimset.claims[0].id
    stored = graph.add_evidence(claim_id, fact_check())

    graph.apply_to_claimset(claimset)

    assert claimset.claims[0].evidence_ids == [stored.id]
    assert claimset.model_dump(mode="json")["claims"][0]["evidence_ids"] == [stored.id]


def test_stage_two_output_is_untouched():
    """Stage 3A must not mutate the ClaimSet it was built from."""
    claimset = build_claimset()
    before = claimset.model_dump(mode="json")

    graph = EvidenceGraph.from_claimset(claimset)
    graph.add_evidence(claimset.claims[0].id, fact_check())

    assert claimset.model_dump(mode="json") == before
