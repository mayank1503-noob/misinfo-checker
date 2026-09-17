"""
Integration tests: the verdict rules, the pipeline, the API and the bot.

Everything runs offline. Retrieval is either stubbed or left to find
nothing (no API keys are set), and the stance models are stubbed the same
way the stage 4 tests stub them. The point of these tests is the wiring
and the rules on top of it, not the models underneath.
"""

import re
import zlib
from datetime import date

import pytest

from backend.claims import extract_claims
from backend.evidence import EvidenceCandidate
from backend.graph import EvidenceGraph
from backend.pipeline import analyze, analyze_text
from backend.verdict import decide, verdict_for_claim


TODAY = date(2026, 9, 17)

SCAM = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your reward."
)


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    """No keys, an isolated cache, and no image index."""
    for name in ("GOOGLE_FACTCHECK_KEY", "TAVILY_API_KEY", "SERPAPI_KEY"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("IMAGE_INDEX", str(tmp_path / "no-index.json"))

    # The API calls the pipeline with backend="auto", which resolves to
    # the transformer backend whenever `transformers` is installed. That
    # would load a model inside one of the API's worker threads - slow,
    # and on Windows a hard crash. These tests are about the wiring.
    monkeypatch.setenv("CLAIM_BACKEND", "heuristic")

    from backend.images import local_index

    local_index.reset()

    yield

    local_index.reset()


@pytest.fixture
def no_models(monkeypatch):
    """The seed index and stance models unavailable: the barest configuration."""
    monkeypatch.setattr("backend.stance.rank.available", lambda: False)
    monkeypatch.setattr("backend.stance.nli.available", lambda: False)
    monkeypatch.setattr("backend.images.consistency.available", lambda: False)

    from backend.evidence.retrievers import seed_index

    seed_index.reset()

    yield

    seed_index.reset()


@pytest.fixture
def stance_models(monkeypatch):
    """A hashing embedder and a keyword-scripted NLI model."""
    STOP = {"the", "a", "an", "of", "in", "is", "to", "this", "has", "and", "that"}
    BUCKETS = 512

    def embed(texts):
        vectors = []

        for text in texts:
            vector = [0.0] * BUCKETS

            for token in re.findall(r"\w+", text.lower()):
                if token not in STOP:
                    vector[zlib.crc32(token.encode("utf-8")) % BUCKETS] += 1.0

            vectors.append(vector)

        return vectors

    def score_pairs(premises, hypothesis, batch_size=8):
        out = []

        for premise in premises:
            lowered = premise.lower()
            top = 0.9
            rest = round((1 - top) / 2, 4)

            if "no such scheme" in lowered or "is a scam" in lowered:
                label = "contradiction"
            elif "confirmed the offer" in lowered:
                label = "entailment"
            else:
                label = "neutral"

            out.append({
                name: (top if name == label else rest)
                for name in ("entailment", "neutral", "contradiction")
            })

        return out

    monkeypatch.setattr("backend.stance.rank.embed", embed)
    monkeypatch.setattr("backend.stance.rank.available", lambda: True)
    monkeypatch.setattr("backend.stance.nli.available", lambda: True)
    monkeypatch.setattr("backend.stance.nli.score_pairs", score_pairs)

    yield


def packet(text=SCAM, images=None, source_date=None, input_type="text"):
    return {
        "input_type": input_type,
        "text": text,
        "source_date": source_date,
        "images": images or [],
    }


def graph_with(text=SCAM, **packet_kwargs):
    pkt = packet(text, **packet_kwargs)
    claims = extract_claims(pkt, today=TODAY, backend="heuristic")

    return claims, EvidenceGraph.from_claimset(claims, pkt), pkt


def add_evidence(graph, claim_id, evidence_id, stance=None, **overrides):
    fields = {
        "claim_id": claim_id,
        "source_type": "factcheck",
        "url": f"https://www.altnews.in/{evidence_id}",
        "title": "No, SBI is not giving Rs 5,000 cashback",
        "snippet": "There is no such scheme, the bank confirmed.",
        "publisher": "Alt News",
        "rating": "false",
        "rating_raw": "False",
        "published_date": "2024-03-11",
        "source_weight": 1.0,
        "decisive": False,
    }
    fields.update(overrides)

    candidate = EvidenceCandidate(**fields)
    candidate.id = evidence_id

    graph.add_evidence(claim_id, candidate)

    if stance:
        graph.set_stance(evidence_id, claim_id, stance, score=1.0, method="rating")

    return evidence_id


# --- the verdict rules ------------------------------------------------------


def test_a_decisive_fact_check_decides_on_its_own():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_pib", stance="refutes", decisive=True)

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["label"] == "false"
    assert verdict["confidence"] == pytest.approx(0.9)
    assert "Alt News" in verdict["reasons"][0]
    assert "already checked" in verdict["explanation"]


def test_a_decisive_misleading_rating_gives_misleading():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_mis", stance="refutes", decisive=True,
                 rating="misleading", rating_raw="Missing context")

    assert verdict_for_claim(graph, claim_id)["label"] == "misleading"


def test_one_decisive_hit_outweighs_several_weak_supporters():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    for index in range(4):
        add_evidence(
            graph, claim_id, f"ev_blog{index}", stance="supports",
            url=f"https://blog{index}.wordpress.com/x", source_weight=0.3,
            rating=None, rating_raw=None, source_type="web",
        )

    add_evidence(graph, claim_id, "ev_pib", stance="refutes", decisive=True)

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["label"] == "false"
    assert graph.stance_totals(claim_id)["supports"] > 0   # they were counted
    assert verdict["confidence"] == pytest.approx(0.9)     # and outranked anyway


def test_nothing_found_is_unverified_not_false():
    claims, graph, _pkt = graph_with()

    verdict = verdict_for_claim(graph, claims.claims[0].id)

    assert verdict["label"] == "unverified"
    assert verdict["confidence"] < 0.5
    assert "Nothing was found" in verdict["explanation"]


def test_related_but_uncommitted_evidence_is_still_unverified():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_side", stance="neutral", decisive=False)

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["label"] == "unverified"
    assert "none of it takes a side" in verdict["explanation"]


def test_weighted_refutation_without_a_decisive_hit():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_news", stance="refutes", decisive=False,
                 url="https://www.thehindu.com/x", source_type="web",
                 rating=None, rating_raw=None)

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["label"] == "false"
    assert 0.5 < verdict["confidence"] < 0.9
    assert "weight of the evidence" in verdict["explanation"]


def test_a_single_weak_source_is_not_enough():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_blog", stance="refutes", decisive=False,
                 url="https://someblog.wordpress.com/x", source_type="web",
                 rating=None, rating_raw=None,
                 # what normalize.py would assign this domain
                 source_weight=0.3)

    # 0.3 is below the floor: one blog is not a verdict
    assert verdict_for_claim(graph, claim_id)["label"] == "unverified"


def test_evidence_on_both_sides_is_disputed():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_for", stance="supports", decisive=False,
                 url="https://www.thehindu.com/a", source_type="web",
                 rating=None, rating_raw=None)
    add_evidence(graph, claim_id, "ev_against", stance="refutes", decisive=False,
                 url="https://www.ndtv.com/b", source_type="web",
                 rating=None, rating_raw=None)

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["label"] == "disputed"
    assert "disagree" in verdict["explanation"]


def test_support_gives_true():
    claims, graph, _pkt = graph_with(
        "The RBI reduced the repo rate by 25 basis points on 07/06/2024."
    )
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_rbi", stance="supports", decisive=False,
                 url="https://rbi.org.in/press", source_type="web",
                 rating=None, rating_raw=None)

    assert verdict_for_claim(graph, claim_id)["label"] == "true"


def test_a_recycled_picture_makes_a_defensible_claim_misleading():
    image = {
        "path": "flood.jpg", "ocr_text": "", "description": "a flooded street",
        "embedding": [], "exif_date": None, "ai_generated_score": None,
        "frame_time": None,
    }
    claims, graph, _pkt = graph_with(
        "Chennai is flooded today and the whole city is under water.",
        images=[image], source_date="2026-09-01", input_type="image",
    )
    claim_id = claims.claims[0].id
    image_node = graph.images()[0][0]

    graph.flag_date_mismatch(image_node, "2015-12-02")

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["label"] == "misleading"
    assert verdict["confidence"] == pytest.approx(0.75)
    assert "2015-12-02" in verdict["explanation"]
    assert "older than the story" in verdict["explanation"]


def test_demo_only_evidence_is_flagged_and_discounted():
    claims, graph, _pkt = graph_with()
    claim_id = claims.claims[0].id

    add_evidence(graph, claim_id, "ev_demo", stance="refutes", decisive=True, demo=True)

    verdict = verdict_for_claim(graph, claim_id)

    assert verdict["demo_only"] is True
    assert "demo evidence index" in verdict["explanation"]
    assert verdict["confidence"] < 0.9       # discounted for resting on demo data


def test_decide_writes_verdict_nodes_and_takes_the_worst_label():
    claims, graph, _pkt = graph_with(
        "SBI is giving Rs 5,000 cashback to every customer today. "
        "The RBI reduced the repo rate by 25 basis points on 07/06/2024."
    )
    scam_claim, rate_claim = sorted(cid for cid, _ in graph.claims())[:2]

    add_evidence(graph, scam_claim, "ev_false", stance="refutes", decisive=True)
    add_evidence(graph, rate_claim, "ev_true", stance="supports", decisive=False,
                 url="https://rbi.org.in/press", source_type="web",
                 rating=None, rating_raw=None)

    result = decide(graph)

    assert result["label"] == "false"                   # the worst claim wins
    assert result["counts"]["false"] == 1
    assert len(result["claims"]) == 2
    assert result["claims"][0]["claim_id"] == scam_claim

    assert graph.node(scam_claim)["verdict"] == "false"
    assert graph.nodes_of("verdict")


def test_a_packet_with_no_check_worthy_claims_is_unverified():
    claims, graph, _pkt = graph_with("Hi. Ok.")

    result = decide(graph)

    assert result["label"] == "unverified"
    assert "No check-worthy claims" in result["summary"]


# --- the pipeline -----------------------------------------------------------


def test_pipeline_runs_with_no_keys_and_no_models(no_models):
    result = analyze_text(SCAM, backend="heuristic")

    assert result["verdict"]["label"] == "unverified"
    assert result["claims"]
    assert result["stages"]["claims"]["check_worthy"] >= 1
    assert all(stage.get("error") is None for stage in result["stages"].values())
    assert result["ms"] >= 0


def test_pipeline_reports_every_stage(no_models):
    result = analyze_text(SCAM, backend="heuristic")

    assert set(result["stages"]) >= {"claims", "graph", "evidence", "stance", "verdict"}
    assert "ms" in result["stages"]["claims"]


def test_pipeline_finds_and_uses_evidence(stance_models, monkeypatch):
    """A stubbed retriever's fact-check should reach the verdict."""
    class Fake:
        def available(self):
            return True

        def search_many(self, claim_id, queries):
            return [
                EvidenceCandidate(
                    claim_id=claim_id,
                    source_type="factcheck",
                    url="https://www.altnews.in/sbi-cashback",
                    title="No, SBI is not giving Rs 5,000 cashback to customers",
                    snippet=(
                        "SBI is giving Rs 5,000 cashback to every customer, claims a "
                        "viral message. There is no such scheme, the bank confirmed."
                    ),
                    publisher="Alt News",
                    rating_raw="False",
                    published_date="2024-03-11",
                )
            ]

    monkeypatch.setattr(
        "backend.evidence.collector.RETRIEVERS", {"factcheck": Fake()}
    )

    result = analyze_text(SCAM, backend="heuristic")

    assert result["verdict"]["label"] == "false"
    assert result["verdict"]["counts"]["false"] >= 1

    top = result["verdict"]["claims"][0]

    assert "Alt News" in " ".join(top["reasons"])
    assert top["evidence_ids"]
    assert result["stages"]["evidence"]["kept"] >= 1
    assert result["stages"]["stance"]["refutes"] >= 1


def test_pipeline_strips_embeddings_from_its_output(no_models):
    image = {
        "path": "x.jpg", "ocr_text": "", "description": "a street",
        "embedding": [0.1] * 768, "exif_date": None, "ai_generated_score": None,
        "frame_time": None,
    }

    result = analyze(packet(images=[image], input_type="image"), backend="heuristic")

    assert result["packet"]["images"][0].get("embedding") is None
    assert "embedding" not in str(result["graph"])


def test_pipeline_can_return_the_whole_graph(no_models):
    result = analyze(packet(), backend="heuristic", graph_json=True)

    assert isinstance(result["graph"], dict)
    assert result["graph"]["nodes"]

    restored = EvidenceGraph.from_dict(result["graph"])

    assert restored.packet_id == result["graph"]["packet_id"]


def test_pipeline_survives_a_stage_that_explodes(no_models, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("retrieval is down")

    monkeypatch.setattr("backend.pipeline.collect_evidence", explode)

    result = analyze_text(SCAM, backend="heuristic")

    assert result["verdict"]["label"] == "unverified"
    assert "retrieval is down" in result["stages"]["evidence"]["error"]


def test_pipeline_survives_claim_extraction_failing(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("extractor is down")

    monkeypatch.setattr("backend.pipeline.extract_claims", explode)

    result = analyze_text(SCAM)

    assert result["verdict"]["label"] == "unverified"
    assert result["claims"] == []
    assert "extractor is down" in result["stages"]["claims"]["error"]


def test_retrieval_can_be_turned_off(no_models):
    result = analyze_text(SCAM, backend="heuristic", retrieve=False, stance=False)

    assert "evidence" not in result["stages"]
    assert "stance" not in result["stages"]
    assert result["verdict"]["label"] == "unverified"


def test_the_five_samples_all_run(no_models):
    import os

    directory = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "samples")

    for name in ("fake_upi", "hinglish_rumor", "personal_chat",
                 "recycled_rumor", "true_claim"):
        with open(os.path.join(directory, f"{name}.txt"), encoding="utf-8") as handle:
            text = handle.read().strip()

        assert text, f"{name}.txt is empty"

        result = analyze_text(text, backend="heuristic")

        assert result["verdict"]["label"] in (
            "false", "misleading", "true", "disputed", "unverified"
        )
        assert result["verdict"]["summary"]


def test_personal_chat_is_never_called_false(no_models):
    """
    Personal chat must come back `unverified`, never `false`.

    Note that stage 2's regex backend does treat "Kal shaam ko ghar aa
    raha hoon" as check-worthy - it has a time reference and a subject -
    so this is not a no-claims case. What matters downstream is that
    finding no evidence about someone's dinner plans produces an
    abstention rather than an accusation (see DECISIONS.md, open issues).
    """
    result = analyze_text(
        "Hi bhai, kaise ho? Kal shaam ko ghar aa raha hoon.", backend="heuristic"
    )

    assert result["verdict"]["label"] == "unverified"
    assert result["verdict"]["confidence"] < 0.5


# --- the API ----------------------------------------------------------------


@pytest.fixture
def client():
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    from backend.api.main import app

    return fastapi_testclient.TestClient(app)


def test_health_reports_what_is_switched_on(client, monkeypatch):
    # The availability probes are stubbed only because the real ones
    # `import sentence_transformers`, which costs the suite forty
    # seconds of torch import. What is under test here is that /health
    # reports them, not what they answer on this machine.
    monkeypatch.setattr("backend.images.consistency.available", lambda: False)
    monkeypatch.setattr("backend.stance.rank.available", lambda: True)
    monkeypatch.setattr("backend.stance.nli.available", lambda: True)

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert set(body["retrievers"]) == {"factcheck", "seed_index", "web"}
    assert body["retrievers"]["factcheck"] is False        # no key in the tests
    assert body["models"] == {"embedder": True, "nli": True, "clip": False}


def test_check_text_returns_a_verdict_not_an_empty_list(client, no_models):
    body = client.post("/check/text", json={"text": SCAM}).json()

    assert body["verdict"]["label"] in (
        "false", "misleading", "true", "disputed", "unverified"
    )
    assert body["verdict"]["summary"]
    assert isinstance(body["results"], list)
    assert body["results"], "results must no longer be hard-coded empty"

    first = body["results"][0]

    assert set(first) >= {"claim", "label", "confidence", "explanation", "reasons"}
    assert body["packet"]["input_type"] == "text"


def test_check_text_validates_input(client):
    assert client.post("/check/text", json={"text": "   "}).status_code == 422
    assert client.post("/check/text", json={"text": "x" * 10001}).status_code == 413


def test_check_link_validates_the_scheme(client):
    assert client.post("/check/link", json={"url": "notaurl"}).status_code == 422
    assert client.post("/check/link", json={"url": ""}).status_code == 422


def test_check_image_and_video_need_a_real_file(client):
    assert client.post("/check/image", json={"path": "no/such.jpg"}).status_code == 404
    assert client.post("/check/video", json={"path": "no/such.mp4"}).status_code == 404
    assert client.post("/check/image", json={"path": ""}).status_code == 422


def test_check_packet_accepts_a_prebuilt_packet(client, no_models):
    body = client.post("/check/packet", json=packet()).json()

    assert body["verdict"]["label"]
    assert body["claims"]

    assert client.post("/check/packet", json={}).status_code == 422


def test_graph_is_opt_in(client, no_models):
    without = client.post("/check/text", json={"text": SCAM}).json()
    with_graph = client.post("/check/text?graph=true", json={"text": SCAM}).json()

    assert "graph" not in without
    assert isinstance(with_graph["graph"], dict)


# --- the bot ----------------------------------------------------------------


def test_bot_replies_with_a_verdict(no_models):
    from backend.bot.guardian_bot import process_message

    reply = process_message(SCAM, analyze=lambda text, **kwargs: analyze_text(
        text, backend="heuristic"))

    assert reply["status"] == "checked"
    assert reply["input_type"] == "text"
    assert reply["label"] == "unverified"
    assert "couldn't verify" in reply["reply"]


def test_bot_reply_names_the_source_when_there_is_one():
    from backend.bot.guardian_bot import format_reply

    reply = format_reply({
        "verdict": {
            "label": "false",
            "summary": "...",
            "claims": [
                {
                    "claim": "SBI is giving Rs 5,000 cashback.",
                    "label": "false",
                    "explanation": "A fact-checker has already checked this claim.",
                    "reasons": ["Alt News rated it false on 2024-03-11: No, SBI is not"],
                    "demo_only": False,
                }
            ],
        }
    })

    assert reply.startswith("❌")
    assert "Don't forward this" in reply
    assert "Alt News" in reply


def test_bot_says_when_it_is_running_on_demo_data():
    from backend.bot.guardian_bot import format_reply

    reply = format_reply({
        "verdict": {
            "label": "false",
            "claims": [{
                "claim": "x", "label": "false", "explanation": "y",
                "reasons": [], "demo_only": True,
            }],
        }
    })

    assert "demo evidence index" in reply


def test_bot_ignores_an_empty_message():
    from backend.bot.guardian_bot import process_message

    result = process_message("   ")

    assert result["status"] == "ignored"
    assert result["reply"]


def test_bot_routes_a_link_and_normalises_a_bare_www():
    from backend.bot.guardian_bot import process_message

    seen = {}

    def fake(url, **kwargs):
        seen["url"] = url

        return {"verdict": {"label": "unverified", "claims": []}}

    result = process_message("check this www.example.com/story please", analyze=fake)

    assert result["input_type"] == "link"
    assert seen["url"] == "https://www.example.com/story"


def test_bot_survives_the_pipeline_failing():
    from backend.bot.guardian_bot import process_message

    def explode(text, **kwargs):
        raise RuntimeError("everything is on fire")

    result = process_message("something", analyze=explode)

    assert result["status"] == "error"
    assert "everything is on fire" in result["reason"]
    assert "went wrong" in result["reply"]
