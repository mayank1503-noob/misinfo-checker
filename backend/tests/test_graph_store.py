"""
Stage 3a (evidence graph) tests.

Nothing here touches the network or a model: the graph is a pure data
structure, and the claims it is built from come from the regex backend.
"""

import json
from datetime import date

import pytest

from backend.claims import extract_claims
from backend.graph import EvidenceGraph
from backend.graph.schema import date_id, image_id, normalize_stance, source_id
from backend.graph.store import _iso


TODAY = date(2026, 9, 17)

SCAM_TEXT = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your SBI reward. "
    "RBI announced a 2% cut in repo rate on 12/08/2026."
)


def text_packet(text=SCAM_TEXT, source_date=None):
    return {
        "input_type": "text",
        "text": text,
        "source_date": source_date,
        "images": [],
    }


def image_packet(caption="Flood in Chennai today", ocr_text="SBI is giving Rs 5,000 cashback",
                 description="a flooded street with cars", exif_date="2019-08-12",
                 frame_time=None, path="flood.jpg"):
    return {
        "input_type": "image",
        "text": caption,
        "source_date": "2026-09-01",
        "images": [
            {
                "path": path,
                "ocr_text": ocr_text,
                "description": description,
                "embedding": [0.1] * 768,
                "exif_date": exif_date,
                "ai_generated_score": 0.12,
                "frame_time": frame_time,
            }
        ],
    }


def build(packet=None):
    packet = packet or text_packet()
    claimset = extract_claims(packet, today=TODAY, backend="heuristic")

    return claimset, EvidenceGraph.from_claimset(claimset, packet)


def candidate(evidence_id="ev_altnews", **overrides):
    item = {
        "id": evidence_id,
        "claim_id": None,
        "source_type": "factcheck",
        "query": "SBI cashback",
        "url": "https://www.altnews.in/sbi-cashback-scam",
        "domain": "altnews.in",
        "title": "No, SBI is not giving Rs 5,000 cashback",
        "snippet": "The viral message is a scam.",
        "text": "The viral message claiming SBI gives Rs 5,000 cashback is a scam.",
        "publisher": "Alt News",
        "rating": "false",
        "rating_raw": "False",
        "published_date": "2026-09-10",
        "retrieved_at": "2026-09-17",
        "source_weight": 1.0,
        "decisive": True,
    }
    item.update(overrides)

    return item


def first_claim(graph):
    return sorted(cid for cid, _ in graph.claims())[0]


# --- construction -----------------------------------------------------------


def test_from_claimset_loads_the_stage_two_seed():
    claimset, graph = build()
    seed = claimset.to_graph_seed()

    assert graph.packet_id == claimset.packet_id
    assert graph.input_type == "text"
    assert graph.node(graph.packet_id)["kind"] == "packet"

    assert {cid for cid, _ in graph.claims()} == {c.id for c in claimset.claims}
    assert {eid for eid, _ in graph.entities()} == {
        n["id"] for n in seed["nodes"] if n["kind"] == "entity"
    }

    for claim in claimset.claims:
        assert graph.node(claim.id)["text"] == claim.text
        assert graph.g.has_edge(graph.packet_id, claim.id, "HAS_CLAIM")


def test_mentions_and_shares_entity_edges_survive_the_load():
    claimset, graph = build()

    for claim in claimset.claims:
        for entity in claim.entities:
            assert graph.g.has_edge(claim.id, entity.id, "MENTIONS")

    shared = [
        (src, dst, attrs)
        for src, dst, attrs in graph.g.edges(data=True)
        if attrs.get("type") == "SHARES_ENTITY"
    ]

    assert shared, "both SBI sentences should be linked"

    for src, dst, attrs in shared:
        assert graph.node(src)["kind"] == "claim"
        assert graph.node(dst)["kind"] == "claim"
        assert graph.node(attrs["via"])["kind"] == "entity"


def test_parallel_shares_entity_edges_are_kept_apart_by_via():
    """Two claims sharing two entities is two edges, not one overwritten."""
    claimset, graph = build(text_packet(
        "SBI told RBI the scheme is closed. RBI asked SBI for the scheme details."
    ))

    keys = [
        key
        for src, dst, key, attrs in graph.g.edges(keys=True, data=True)
        if attrs.get("type") == "SHARES_ENTITY"
    ]

    assert len(keys) == len(set(keys))
    assert all(key.startswith("SHARES_ENTITY:") for key in keys)


def test_images_become_nodes_with_exif_dates_and_frame_times():
    packet = image_packet(frame_time=12.5)
    _claimset, graph = build(packet)

    images = graph.images()

    assert len(images) == 1

    node_id, attrs = images[0]

    assert node_id == image_id(graph.packet_id, "flood.jpg", 12.5)
    assert attrs["exif_date"] == "2019-08-12"
    assert attrs["frame_time"] == 12.5
    assert attrs["ai_generated_score"] == 0.12
    assert graph.g.has_edge(graph.packet_id, node_id, "HAS_IMAGE")
    assert graph.g.has_edge(node_id, date_id("2019-08-12"), "CAPTURED_ON")


def test_no_embeddings_are_stored():
    _claimset, graph = build(image_packet())

    blob = graph.to_json()

    assert "embedding" not in blob

    for _node_id, attrs in graph.g.nodes(data=True):
        assert "embedding" not in attrs


def test_ocr_claims_link_back_to_their_image():
    packet = image_packet()
    claimset, graph = build(packet)

    ocr_claims = [c for c in claimset.claims if c.source.field in ("ocr", "caption")]

    assert ocr_claims, "the OCR text should yield a claim"

    node_id = graph.images()[0][0]

    for claim in ocr_claims:
        assert graph.g.has_edge(claim.id, node_id, "EXTRACTED_FROM")
        assert graph.g.edges[claim.id, node_id, "EXTRACTED_FROM"]["field"] == claim.source.field

    for claim in claimset.claims:
        if claim.source.field == "text":
            assert not graph.g.has_edge(claim.id, node_id, "EXTRACTED_FROM")


def test_video_keyframes_get_one_node_each():
    packet = {
        "input_type": "video",
        "text": "Watch this",
        "source_date": None,
        "images": [
            {"path": "f.jpg", "ocr_text": "", "description": "", "embedding": [],
             "exif_date": None, "ai_generated_score": None, "frame_time": time}
            for time in (0.0, 5.0, 10.0)
        ],
    }

    _claimset, graph = build(packet)

    assert len(graph.images()) == 3
    assert len({node_id for node_id, _ in graph.images()}) == 3


# --- evidence ---------------------------------------------------------------


def test_add_evidence_writes_source_and_date_nodes():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    evidence_id = graph.add_evidence(claim_id, candidate())

    assert evidence_id == "ev_altnews"
    assert graph.node(evidence_id)["kind"] == "evidence"
    assert graph.g.has_edge(claim_id, evidence_id, "HAS_EVIDENCE")

    src = source_id("altnews.in")
    assert graph.node(src)["publisher"] == "Alt News"
    assert graph.node(src)["weight"] == 1.0
    assert graph.g.has_edge(evidence_id, src, "FROM_SOURCE")

    assert graph.g.has_edge(evidence_id, date_id("2026-09-10"), "PUBLISHED_ON")
    assert graph.node(date_id("2026-09-10"))["date"] == "2026-09-10"


def test_add_evidence_without_a_stance_leaves_the_claim_open():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate())

    assert graph.stance_of("ev_altnews", claim_id) is None
    assert graph.stance_totals(claim_id)["counts"]["unscored"] == 1
    assert claim_id in graph.open_claims()


def test_add_evidence_is_idempotent():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate())
    nodes, edges = len(graph.g.nodes), len(graph.g.edges)

    graph.add_evidence(claim_id, candidate())

    assert (len(graph.g.nodes), len(graph.g.edges)) == (nodes, edges)


def test_one_evidence_item_can_serve_two_claims():
    _claimset, graph = build()
    claim_ids = sorted(cid for cid, _ in graph.claims())[:2]

    for claim_id in claim_ids:
        graph.add_evidence(claim_id, candidate())

    assert len(graph.nodes_of("evidence")) == 1
    assert all(graph.g.has_edge(cid, "ev_altnews", "HAS_EVIDENCE") for cid in claim_ids)


def test_two_articles_from_one_publisher_share_a_source_node():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate("ev_one"))
    graph.add_evidence(claim_id, candidate("ev_two", url="https://www.altnews.in/other"))

    assert len(graph.nodes_of("source")) == 1


def test_add_evidence_rejects_an_unknown_claim():
    _claimset, graph = build()

    with pytest.raises(KeyError):
        graph.add_evidence("clm_nope", candidate())


def test_add_evidence_rejects_a_non_claim_node():
    _claimset, graph = build()

    with pytest.raises(ValueError):
        graph.add_evidence(graph.packet_id, candidate())


# --- stance -----------------------------------------------------------------


def test_set_stance_adds_one_directed_edge():
    _claimset, graph = build()
    claim_id = first_claim(graph)
    graph.add_evidence(claim_id, candidate())

    graph.set_stance("ev_altnews", claim_id, "refutes", score=0.9, relevance=0.8,
                     method="rating")

    assert graph.g.has_edge("ev_altnews", claim_id, "REFUTES")
    assert graph.stance_of("ev_altnews", claim_id) == "refutes"

    edge = graph.g.edges["ev_altnews", claim_id, "REFUTES"]
    assert edge["score"] == 0.9
    assert edge["method"] == "rating"


def test_set_stance_accepts_the_stage_four_vocabulary():
    _claimset, graph = build()
    claim_id = first_claim(graph)
    graph.add_evidence(claim_id, candidate())

    graph.set_stance("ev_altnews", claim_id, "refute")

    assert graph.stance_of("ev_altnews", claim_id) == "refutes"
    assert normalize_stance("support") == "supports"

    with pytest.raises(ValueError):
        normalize_stance("maybe")


def test_set_stance_replaces_the_previous_edge():
    _claimset, graph = build()
    claim_id = first_claim(graph)
    graph.add_evidence(claim_id, candidate(), stance="supports", score=0.7)

    assert graph.g.has_edge("ev_altnews", claim_id, "SUPPORTS")

    graph.set_stance("ev_altnews", claim_id, "refutes", score=1.0)

    assert not graph.g.has_edge("ev_altnews", claim_id, "SUPPORTS")
    assert graph.g.has_edge("ev_altnews", claim_id, "REFUTES")
    assert graph.stance_totals(claim_id)["supports"] == 0.0


def test_stance_totals_are_weighted_by_source_credibility():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(
        claim_id,
        candidate("ev_pib", domain="pib.gov.in", publisher="PIB", source_weight=1.0),
        stance="refutes",
        score=1.0,
    )
    graph.add_evidence(
        claim_id,
        candidate("ev_blog", domain="blog.example", publisher="A blog",
                  source_weight=0.3, decisive=False),
        stance="supports",
        score=1.0,
    )

    totals = graph.stance_totals(claim_id)

    assert totals["refutes"] == 1.0
    assert totals["supports"] == 0.3
    assert totals["margin"] == -0.7
    assert totals["counts"] == {"supports": 1, "refutes": 1, "neutral": 0, "unscored": 0}
    assert totals["decisive"] == 1


def test_evidence_for_filters_and_ranks():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate("ev_weak", source_weight=0.3, decisive=False),
                       stance="refutes", score=0.5)
    graph.add_evidence(claim_id, candidate("ev_strong", source_weight=1.0),
                       stance="refutes", score=1.0)
    graph.add_evidence(claim_id, candidate("ev_side", source_weight=0.8, decisive=False),
                       stance="neutral", score=0.4)

    assert [item["id"] for item in graph.evidence_for(claim_id)] == [
        "ev_strong", "ev_side", "ev_weak",
    ]
    assert [item["id"] for item in graph.evidence_for(claim_id, stance="refutes")] == [
        "ev_strong", "ev_weak",
    ]
    assert [item["id"] for item in graph.evidence_for(claim_id, decisive_only=True)] == [
        "ev_strong",
    ]


def test_decisive_hits_ignores_undecided_and_neutral_items():
    _claimset, graph = build()
    claim_ids = sorted(cid for cid, _ in graph.claims())[:2]

    graph.add_evidence(claim_ids[0], candidate("ev_hit"), stance="refutes", score=1.0)
    graph.add_evidence(claim_ids[1], candidate("ev_open"))
    graph.add_evidence(claim_ids[1], candidate("ev_side", decisive=False),
                       stance="neutral")

    hits = graph.decisive_hits()

    assert [hit["id"] for hit in hits] == ["ev_hit"]
    assert hits[0]["claim_id"] == claim_ids[0]
    assert graph.decisive_hits(claim_ids[1]) == []


def test_open_claims_shrinks_as_stances_arrive():
    _claimset, graph = build()
    claim_ids = sorted(cid for cid, _ in graph.claims())

    assert set(graph.open_claims()) == set(claim_ids)

    graph.add_evidence(claim_ids[0], candidate(), stance="refutes")

    assert claim_ids[0] not in graph.open_claims()
    assert claim_ids[1] in graph.open_claims()


def test_neutral_evidence_does_not_close_a_claim():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate(), stance="neutral")

    assert claim_id in graph.open_claims()


# --- images, duplicates, dates ----------------------------------------------


def test_add_duplicate_links_an_image_to_the_older_post():
    _claimset, graph = build(image_packet())
    claim_id = first_claim(graph)
    image_node = graph.images()[0][0]

    graph.add_evidence(
        claim_id,
        candidate("ev_lens", source_type="reverse_image", published_date="2019-08-20"),
    )
    graph.add_duplicate(image_node, "ev_lens", similarity=0.97, first_seen="2019-08-20",
                        context="Chennai floods, 2019")

    edge = graph.g.edges[image_node, "ev_lens", "DUPLICATE_OF"]

    assert edge["similarity"] == 0.97
    assert edge["context"] == "Chennai floods, 2019"
    assert graph.g.has_edge(image_node, date_id("2019-08-20"), "FIRST_SEEN_ON")
    assert graph.node(image_node)["first_seen"] == "2019-08-20"


def test_flag_date_mismatch_fires_when_the_picture_predates_the_message():
    _claimset, graph = build(image_packet())
    image_node = graph.images()[0][0]

    flagged = graph.flag_date_mismatch(image_node, "2019-08-20")

    assert flagged == "2019-08-20"

    edge = graph.g.edges[image_node, date_id("2019-08-20"), "DATE_MISMATCH"]

    assert edge["claimed_date"] == "2026-09-01"
    assert edge["gap_days"] > 2000
    assert graph.node(image_node)["date_mismatch"] is True


def test_flag_date_mismatch_ignores_a_small_gap_or_a_newer_image():
    _claimset, graph = build(image_packet())
    image_node = graph.images()[0][0]

    assert graph.flag_date_mismatch(image_node, "2026-08-25") is None   # 7 days
    assert graph.flag_date_mismatch(image_node, "2026-09-30") is None   # newer
    assert graph.flag_date_mismatch(image_node, None) is None
    assert not graph.node(image_node).get("date_mismatch")


def test_timeline_is_ordered_and_names_the_mismatch():
    _claimset, graph = build(image_packet())
    claim_id = first_claim(graph)
    image_node = graph.images()[0][0]

    graph.add_evidence(claim_id, candidate())
    graph.flag_date_mismatch(image_node, "2019-08-20")

    events = graph.timeline()
    dates = [event["date"] for event in events]

    assert dates == sorted(dates)
    assert "2019-08-12" in dates      # EXIF capture
    assert "2026-09-01" in dates      # the message itself
    assert "2026-09-10" in dates      # the fact-check
    assert any(event["type"] == "DATE_MISMATCH" for event in events)
    assert any("days older than claimed" in event["label"] for event in events)


def test_iso_normalises_the_date_shapes_sources_use():
    assert _iso("2024-06-08T10:00:00Z") == "2024-06-08"
    assert _iso("2024/06/08") == "2024-06-08"
    assert _iso(date(2024, 6, 8)) == "2024-06-08"
    assert _iso("08-06-2024") == "2024-06-08"
    assert _iso("not a date") is None
    assert _iso(None) is None


# --- verdicts, summary, serialisation ---------------------------------------


def test_add_verdict_writes_a_node_and_marks_the_claim():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    node_id = graph.add_verdict(claim_id, "false", confidence=0.9,
                                explanation="Alt News rated it false")

    assert graph.node(node_id)["label"] == "false"
    assert graph.node(claim_id)["verdict"] == "false"
    assert graph.g.has_edge(claim_id, node_id, "HAS_VERDICT")


def test_summary_quotes_the_evidence_and_respects_max_chars():
    _claimset, graph = build(image_packet())
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate(), stance="refutes", score=1.0)
    graph.add_verdict(claim_id, "false", confidence=0.9)
    graph.flag_date_mismatch(graph.images()[0][0], "2019-08-20")

    text = graph.summary(max_chars=4000)

    assert "Alt News" in text
    assert "[false]" in text
    assert "-> false" in text
    assert "recycled image" in text

    short = graph.summary(max_chars=120)

    assert len(short) <= 124
    assert short.endswith("...")


def test_summary_marks_demo_evidence_as_demo():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate(demo=True), stance="refutes")

    assert "(demo data)" in graph.summary()


def test_to_json_and_from_json_round_trip(tmp_path):
    _claimset, graph = build(image_packet())
    claim_id = first_claim(graph)

    graph.add_evidence(claim_id, candidate(), stance="refutes", score=1.0)
    graph.flag_date_mismatch(graph.images()[0][0], "2019-08-20")
    graph.add_verdict(claim_id, "false", confidence=0.9)

    path = tmp_path / "graph.json"
    text = graph.to_json(path=str(path))

    restored = EvidenceGraph.from_json(text)

    assert restored.to_dict() == graph.to_dict()
    assert restored.stance_totals(claim_id) == graph.stance_totals(claim_id)
    assert restored.timeline() == graph.timeline()

    from_file = EvidenceGraph.from_json(str(path))

    assert from_file.to_dict() == graph.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["packet_id"] == graph.packet_id


def test_graph_ids_are_deterministic_across_runs():
    _first_claims, first = build(image_packet())
    _second_claims, second = build(image_packet())

    claim_id = first_claim(first)

    for graph in (first, second):
        graph.add_evidence(claim_id, candidate(), stance="refutes", score=1.0)

    assert first.to_dict() == second.to_dict()


def test_export_html_writes_a_file(tmp_path):
    pyvis = pytest.importorskip("pyvis")           # optional dependency
    assert pyvis

    _claimset, graph = build()
    graph.add_evidence(first_claim(graph), candidate(), stance="refutes")

    path = tmp_path / "graph.html"
    written = graph.export_html(str(path))

    assert written == str(path)
    assert path.exists() and path.stat().st_size > 0


def test_export_html_fails_soft_without_pyvis(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_pyvis(name, *args, **kwargs):
        if name.startswith("pyvis"):
            raise ImportError("no pyvis")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyvis)

    _claimset, graph = build()

    assert graph.export_html(str(tmp_path / "graph.html")) is None


def test_bad_node_and_edge_types_are_rejected():
    _claimset, graph = build()
    claim_id = first_claim(graph)

    with pytest.raises(ValueError):
        graph.add_node("x", "not_a_kind")

    with pytest.raises(ValueError):
        graph.add_edge(graph.packet_id, claim_id, "NOT_AN_EDGE")

    with pytest.raises(KeyError):
        graph.add_edge(graph.packet_id, "missing", "HAS_CLAIM")
