"""
Stage 5 (image evidence) tests.

No model is loaded and no socket is opened. DINOv2 embeddings are just
lists of floats, so the keyframe and index tests use synthetic vectors
and exercise the real cosine / FAISS / NumPy code; CLIP is stubbed at
`consistency.score_texts_against_image`; SerpAPI is stubbed at
`backend.common.http.get_json`.
"""

import json
from datetime import date

import pytest

from backend.claims import extract_claims
from backend.graph import EvidenceGraph
from backend.images import collect_image_evidence, select_keyframes
from backend.images import consistency, local_index, reverse_search
from backend.images.keyframes import cosine, dedupe_packet


TODAY = date(2026, 9, 17)

# Orthogonal unit vectors stand in for DINOv2 embeddings of different
# pictures; scaled copies stand in for the same picture re-encoded.
DIM = 16


def vector(index, noise=0.0):
    values = [0.0] * DIM
    values[index % DIM] = 1.0

    if noise:
        values[(index + 1) % DIM] = noise

    return values


def image(path="a.jpg", embedding=None, frame_time=None, exif_date=None,
          description="a flooded street with cars", ocr_text="", **extra):
    return {
        "path": path,
        "ocr_text": ocr_text,
        "description": description,
        "embedding": embedding if embedding is not None else vector(0),
        "exif_date": exif_date,
        "ai_generated_score": None,
        "frame_time": frame_time,
        **extra,
    }


def packet(text="Flood in Chennai today, whole city is under water.",
           images=None, source_date="2026-09-01", input_type="image"):
    return {
        "input_type": input_type,
        "text": text,
        "source_date": source_date,
        "images": images if images is not None else [image()],
    }


def build(pkt=None):
    pkt = pkt or packet()
    claims = extract_claims(pkt, today=TODAY, backend="heuristic")

    return claims, EvidenceGraph.from_claimset(claims, pkt), pkt


# --- keyframes --------------------------------------------------------------


def test_near_identical_frames_are_dropped():
    frames = [
        image("f0.jpg", vector(0), frame_time=0.0),
        image("f1.jpg", vector(0), frame_time=5.0),        # the same shot
        image("f2.jpg", vector(3), frame_time=10.0),       # a different shot
    ]

    kept = select_keyframes(frames)

    assert [frame["path"] for frame in kept] == ["f0.jpg", "f2.jpg"]


def test_the_first_frame_of_a_run_is_the_one_kept():
    frames = [image(f"f{i}.jpg", vector(0), frame_time=float(i)) for i in range(4)]

    assert [frame["path"] for frame in select_keyframes(frames)] == ["f0.jpg"]


def test_slightly_different_frames_survive():
    frames = [
        image("f0.jpg", vector(0)),
        image("f1.jpg", vector(0, noise=0.9)),     # cosine well under 0.95
    ]

    assert len(select_keyframes(frames)) == 2


def test_images_are_capped():
    frames = [image(f"f{i}.jpg", vector(i)) for i in range(12)]

    assert len(select_keyframes(frames)) == 8
    assert len(select_keyframes(frames, cap=3)) == 3


def test_frames_without_embeddings_are_kept():
    frames = [image("f0.jpg", []), image("f1.jpg", [])]

    assert len(select_keyframes(frames)) == 2


def test_dedupe_packet_does_not_mutate_the_original():
    original = packet(images=[image("f0.jpg", vector(0)), image("f1.jpg", vector(0))])

    trimmed = dedupe_packet(original)

    assert len(original["images"]) == 2
    assert len(trimmed["images"]) == 1


def test_cosine_edges():
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# --- the local index --------------------------------------------------------


INDEX_ENTRIES = [
    {
        "file": "chennai_flood_2015.png",
        "context": "Flooding in Chennai during the December 2015 storms.",
        "first_seen": "2015-12-02",
        "source_url": "https://example-demo.invalid/archive/chennai-floods-2015",
        "publisher": "Demo Image Archive",
        "domain": "boomlive.in",
        "demo": True,
        "demo_note": "DEMO DATA - a generated placeholder image.",
        "embedding": vector(0),
    },
    {
        "file": "army_convoy_2019.png",
        "context": "An army convoy at a Republic Day rehearsal, January 2019.",
        "first_seen": "2019-01-20",
        "source_url": "https://example-demo.invalid/archive/republic-day-2019",
        "publisher": "Demo Image Archive",
        "domain": "factchecker.in",
        "demo": True,
        "demo_note": "DEMO DATA - a generated placeholder image.",
        "embedding": vector(5),
    },
]


@pytest.fixture
def image_index(tmp_path, monkeypatch):
    path = tmp_path / "image_index.json"
    path.write_text(json.dumps({"entries": INDEX_ENTRIES}), encoding="utf-8")
    monkeypatch.setenv("IMAGE_INDEX", str(path))
    local_index.reset()

    yield path

    local_index.reset()


def test_index_finds_the_same_picture(image_index):
    matches = local_index.search(vector(0))

    assert matches
    assert matches[0]["file"] == "chennai_flood_2015.png"
    assert matches[0]["first_seen"] == "2015-12-02"
    assert matches[0]["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert matches[0]["demo"] is True


def test_index_ignores_a_different_picture(image_index):
    assert local_index.search(vector(9)) == []


def test_index_is_empty_without_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_INDEX", str(tmp_path / "missing.json"))
    local_index.reset()

    assert local_index.available() is False
    assert local_index.search(vector(0)) == []

    local_index.reset()


def test_index_survives_a_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("IMAGE_INDEX", str(path))
    local_index.reset()

    assert local_index.search(vector(0)) == []

    local_index.reset()


def test_index_rejects_an_embedding_of_the_wrong_size(image_index):
    assert local_index.search([0.1, 0.2, 0.3]) == []


def test_index_falls_back_to_numpy_without_faiss(image_index, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("no faiss")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_faiss)
    local_index.reset()

    matches = local_index.search(vector(0))

    assert matches
    assert matches[0]["backend"] == "numpy"


def test_search_with_no_embedding_is_empty(image_index):
    assert local_index.search([]) == []
    assert local_index.search(None) == []


# --- CLIP consistency -------------------------------------------------------


@pytest.fixture
def clip(monkeypatch):
    """A scripted CLIP: floods match flood pictures, nothing else does."""
    def score(path, texts):
        scores = []

        for text in texts:
            lowered = text.lower()

            if "flood" in lowered or "water" in lowered:
                scores.append(0.31)
            else:
                scores.append(0.07)

        return scores

    monkeypatch.setattr(consistency, "score_texts_against_image", score)
    monkeypatch.setattr(consistency, "score_texts", lambda a, b: 0.11)
    monkeypatch.setattr(consistency, "available", lambda: True)

    yield


def test_a_matching_caption_is_not_flagged(clip):
    findings = consistency.check_image(
        "a.jpg", [("clm_1", "Chennai is flooded and the whole city is under water.")]
    )

    assert findings[0]["mismatch"] is False
    assert findings[0]["similarity"] == 0.31


def test_a_mismatched_caption_is_flagged(clip):
    findings = consistency.check_image(
        "a.jpg",
        [("clm_1", "The army has moved tanks to the border.")],
        description="a flooded street with cars",
    )

    finding = findings[0]

    assert finding["mismatch"] is True
    assert finding["similarity"] == 0.07
    assert finding["description_similarity"] == 0.11
    assert "below" in finding["reason"]
    assert "flooded street" in finding["reason"]


def test_the_threshold_is_configurable(clip, monkeypatch):
    monkeypatch.setenv("CLIP_MIN_SIMILARITY", "0.5")

    findings = consistency.check_image(
        "a.jpg", [("clm_1", "Chennai is flooded, the city is under water.")]
    )

    assert findings[0]["mismatch"] is True      # 0.31 now fails


def test_an_unavailable_model_is_no_opinion_rather_than_an_accusation(monkeypatch):
    monkeypatch.setattr(consistency, "available", lambda: False)

    findings = consistency.check_image("a.jpg", [("clm_1", "anything at all")])

    assert findings[0]["similarity"] is None
    assert findings[0]["mismatch"] is False     # never accuse when we could not look
    assert "could not be scored" in findings[0]["reason"]


def test_a_missing_file_is_no_opinion(monkeypatch):
    # `available` is stubbed only to keep `import sentence_transformers`
    # (fifteen seconds of torch) out of the suite: the real file-opening
    # path below it is what is under test, and it returns before any
    # encoder is touched.
    monkeypatch.setattr(consistency, "available", lambda: True)

    assert consistency.score_texts_against_image("no/such/file.jpg", ["x"]) is None


def test_no_texts_is_no_opinion(monkeypatch):
    monkeypatch.setattr(consistency, "available", lambda: True)

    assert consistency.score_texts_against_image("a.jpg", []) is None
    assert consistency.score_texts("", "something") is None


# --- reverse image search ---------------------------------------------------


LENS_PAYLOAD = {
    "visual_matches": [
        {
            "title": "Chennai floods: in pictures",
            "link": "https://www.thehindu.com/news/chennai-floods-2015",
            "source": "The Hindu",
            "snippet": "Published Dec 2, 2015",
            "thumbnail": "https://example.com/thumb.jpg",
        },
        {
            "title": "Chennai under water again",
            "link": "https://www.ndtv.com/chennai-rain-2023",
            "source": "NDTV",
            "date": "2023-12-06",
        },
    ]
}


def test_reverse_search_is_skipped_without_a_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)

    assert reverse_search.available() is False
    assert reverse_search.search("clm_1", "https://example.com/x.jpg") == []


def test_reverse_search_parses_matches_oldest_first(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    monkeypatch.setattr("backend.common.http.get_json", lambda *a, **k: LENS_PAYLOAD)

    results = reverse_search.search("clm_1", "https://example.com/x.jpg")

    assert [item.published_date for item in results] == ["2015-12-02", "2023-12-06"]
    assert results[0].source_type == "reverse_image"
    assert results[0].publisher == "The Hindu"
    assert reverse_search.first_seen(results) == "2015-12-02"


def test_reverse_search_needs_a_public_url(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "test-key")

    assert reverse_search.search("clm_1", None) == []
    assert reverse_search.image_url_of({"path": "/local/file.jpg"}) is None
    assert reverse_search.image_url_of({"path": "https://x.com/a.jpg"}) == "https://x.com/a.jpg"
    assert reverse_search.image_url_of({"url": "https://x.com/b.jpg"}) == "https://x.com/b.jpg"


def test_reverse_search_fails_soft(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    monkeypatch.setattr("backend.common.http.get_json", lambda *a, **k: None)

    assert reverse_search.search("clm_1", "https://example.com/x.jpg") == []


# --- the collector ----------------------------------------------------------


def test_a_recycled_image_becomes_evidence_and_a_date_mismatch(image_index, clip):
    claims, graph, pkt = build(packet(images=[image("flood.jpg", vector(0))]))

    report = collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert report.images == 1
    assert report.index_matches >= 1
    assert report.date_mismatches >= 1

    image_node = graph.images()[0][0]

    assert graph.node(image_node)["first_seen"] == "2015-12-02"
    assert graph.node(image_node)["date_mismatch"] is True

    duplicates = [
        (src, dst, attrs)
        for src, dst, attrs in graph.g.edges(data=True)
        if attrs.get("type") == "DUPLICATE_OF"
    ]

    assert duplicates
    assert duplicates[0][2]["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert "Chennai" in duplicates[0][2]["context"]

    evidence = graph.evidence_for(claims.check_worthy()[0].id)

    assert any(item["source_type"] == "reverse_image" for item in evidence)
    assert any(item["demo"] for item in evidence)

    events = graph.timeline()

    assert any(event["type"] == "DATE_MISMATCH" for event in events)
    assert events[0]["date"] == "2015-12-02"        # oldest first


def test_an_unknown_image_produces_no_index_evidence(image_index, clip):
    claims, graph, pkt = build(packet(images=[image("new.jpg", vector(9))]))

    report = collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert report.index_matches == 0
    assert report.date_mismatches == 0


def test_a_mismatched_caption_becomes_a_refutes_edge(image_index, clip):
    """The picture is of a flood; the claim is about the army."""
    pkt = packet(
        text="The army has moved tanks to the border this morning.",
        images=[image("flood.jpg", vector(9))],     # not in the index
    )
    claims, graph, pkt = build(pkt)

    report = collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert report.mismatches >= 1

    claim_id = claims.check_worthy()[0].id
    evidence = graph.evidence_for(claim_id, stance="refutes")

    assert evidence
    assert evidence[0]["source_type"] == "image_caption"
    assert evidence[0]["method"] == "clip"
    assert evidence[0]["misleading"] is True

    totals = graph.stance_totals(claim_id)

    assert totals["refutes"] > 0
    assert claim_id not in graph.open_claims()


def test_a_matching_caption_produces_no_mismatch(image_index, clip):
    pkt = packet(
        text="Chennai is flooded today and the whole city is under water.",
        images=[image("flood.jpg", vector(9))],
    )
    claims, graph, pkt = build(pkt)

    report = collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert report.mismatches == 0
    assert graph.open_claims()


def test_ocr_claims_get_the_evidence_of_their_own_image(image_index, clip):
    pkt = packet(
        text="See this",
        images=[
            image("flood.jpg", vector(0),
                  ocr_text="Chennai is under water, the whole city is flooded today."),
        ],
    )
    claims, graph, pkt = build(pkt)

    ocr_claims = [c for c in claims.claims if c.source.field == "ocr"]

    assert ocr_claims

    collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    for claim in ocr_claims:
        assert graph.evidence_for(claim.id)


def test_near_duplicate_frames_are_only_checked_once(image_index, clip):
    frames = [
        image("f0.jpg", vector(0), frame_time=0.0),
        image("f1.jpg", vector(0), frame_time=5.0),
        image("f2.jpg", vector(0), frame_time=10.0),
    ]
    claims, graph, pkt = build(packet(images=frames, input_type="video"))

    report = collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert report.images == 1
    assert report.dropped_frames == 2


def test_reverse_search_results_reach_the_graph(image_index, clip, monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    monkeypatch.setattr("backend.common.http.get_json", lambda *a, **k: LENS_PAYLOAD)

    pkt = packet(images=[image("flood.jpg", vector(9), url="https://example.com/x.jpg")])
    claims, graph, pkt = build(pkt)

    report = collect_image_evidence(claims, graph, pkt)

    assert report.reverse_matches >= 1
    assert report.date_mismatches >= 1

    image_node = graph.images()[0][0]

    assert graph.node(image_node)["first_seen"] == "2015-12-02"


def test_a_packet_with_no_images_is_a_no_op():
    claims, graph, pkt = build(packet(images=[], input_type="text"))

    report = collect_image_evidence(claims, graph, pkt)

    assert report.as_dict() == {
        "images": 0, "dropped_frames": 0, "index_matches": 0, "reverse_matches": 0,
        "mismatches": 0, "date_mismatches": 0, "errors": [],
    }


def test_everything_failing_is_survivable(image_index, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("model went away")

    monkeypatch.setattr(local_index, "search", explode)
    monkeypatch.setattr(consistency, "check_image", explode)

    claims, graph, pkt = build()

    report = collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert report.index_matches == 0
    assert report.mismatches == 0
    assert len(report.errors) >= 2
    assert all("model went away" in error for error in report.errors)


def test_no_embeddings_reach_the_graph(image_index, clip):
    claims, graph, pkt = build(packet(images=[image("flood.jpg", vector(0))]))

    collect_image_evidence(claims, graph, pkt, run_reverse_search=False)

    assert "embedding" not in graph.to_json()


def test_the_shipped_image_metadata_is_marked_demo():
    metadata = local_index.load_metadata()

    if not metadata:
        pytest.skip("run scripts/build_image_index.py to create the demo images")

    assert all(record.get("demo") for record in metadata.values())
    assert all("DEMO DATA" in (record.get("demo_note") or "")
               for record in metadata.values())
    assert all(record.get("first_seen") for record in metadata.values())
