from datetime import date

import pytest

from backend.claims import extract_claims
from backend.claims.entities import extract_entities, extract_time_refs
from backend.claims.segment import split_sentences


TODAY = date(2026, 9, 17)


def text_packet(text):
    return {
        "input_type": "text",
        "text": text,
        "source_date": None,
        "images": [],
    }


def image_packet(caption, ocr_text, description, frame_time=None):
    return {
        "input_type": "image",
        "text": caption,
        "source_date": None,
        "images": [
            {
                "path": "x.jpg",
                "ocr_text": ocr_text,
                "description": description,
                "embedding": [],
                "exif_date": None,
                "ai_generated_score": None,
                "frame_time": frame_time,
            }
        ],
    }


# --- segmentation -----------------------------------------------------------


def test_split_sentences_keeps_offsets_and_abbreviations():
    text = "Dr. Sharma said this. Second one!\nThird line"

    sentences = split_sentences(text)

    assert [s[0] for s in sentences] == [
        "Dr. Sharma said this.",
        "Second one!",
        "Third line",
    ]

    for sentence, start, end in sentences:
        assert text[start:end].strip() == sentence


def test_split_sentences_handles_devanagari_danda():
    sentences = split_sentences("पहला वाक्य। दूसरा वाक्य।")

    assert len(sentences) == 2


# --- entities ---------------------------------------------------------------


def test_money_percent_and_org_entities():
    entities = extract_entities(
        "According to the Reserve Bank of India, inflation was 6.5% and reserves hit Rs 2 lakh crore."
    )

    by_label = {e.label: e for e in entities}

    assert by_label["ORG"].text == "Reserve Bank of India"
    assert by_label["PERCENT"].normalized == "6.5%"
    assert by_label["MONEY"].text.startswith("Rs 2 lakh")


def test_upi_and_phone_entities():
    entities = extract_entities("Send Rs 10 to sbirewards@ybl or call 9876543210")

    labels = {e.label for e in entities}

    assert {"UPI", "PHONE", "MONEY"} <= labels


def test_absolute_and_relative_time_refs():
    refs = extract_time_refs("It happened on 8 June 2024 and again yesterday.", today=TODAY)

    kinds = {r.kind: r for r in refs}

    assert kinds["absolute"].normalized == "2024-06-08"
    assert kinds["relative"].normalized == "2026-09-16"


def test_year_inside_money_is_not_a_time_ref():
    claim = extract_claims(text_packet("Govt announces Rs 2000 for every farmer."), today=TODAY, backend="heuristic").claims[0]

    assert all(r.text != "2000" for r in claim.time_refs)
    assert any(e.label == "MONEY" for e in claim.entities)


# --- check-worthiness -------------------------------------------------------


def test_personal_chat_yields_no_claims():
    packet = text_packet("Hey bro kaise ho? I think we should meet tomorrow. Good night!")

    result = extract_claims(packet, today=TODAY, backend="heuristic")

    assert result.claims == []
    reasons = {d["reason"] for d in result.dropped}
    assert {"question", "opinion", "greeting"} <= reasons


def test_fake_upi_message_is_split_into_separate_chain_offer_claims():
    packet = text_packet(
        "Congratulations! You have won Rs 5,000 cashback from SBI. "
        "Send Rs 10 to sbirewards@ybl to claim now. Forward this to 10 people."
    )

    result = extract_claims(packet, today=TODAY, backend="heuristic")

    texts = [c.text for c in result.claims]

    assert "You have won Rs 5,000 cashback from SBI." in texts
    assert "Send Rs 10 to sbirewards@ybl to claim now." in texts
    assert all(c.claim_type == "chain_offer" for c in result.claims)
    assert all(c.check_worthy for c in result.claims)


def test_attribution_claim_records_speaker_and_type():
    packet = text_packet(
        "According to the Reserve Bank of India, the repo rate was kept at 6.5% on 8 June 2024."
    )

    claim = extract_claims(packet, today=TODAY, backend="heuristic").claims[0]

    assert claim.claim_type == "attribution"
    assert claim.attributed_to == "Reserve Bank of India"
    assert "6.5%" in claim.numbers
    assert claim.time_refs[0].normalized == "2024-06-08"


def test_health_and_policy_types():
    health = extract_claims(text_packet("Drinking hot water with lemon cures cancer in 7 days."), today=TODAY, backend="heuristic")
    policy = extract_claims(text_packet("Government has banned 500 rupee notes from 1 October."), today=TODAY, backend="heuristic")

    assert health.claims[0].claim_type == "health"
    assert policy.claims[0].claim_type == "policy"
    assert policy.claims[0].time_refs[0].normalized == "2026-10-01"


def test_hinglish_language_detection():
    packet = text_packet("Bharat sarkar ne kaha hai ki 500 rupaye ke note band ho jayenge.")

    claim = extract_claims(packet, today=TODAY, backend="heuristic").claims[0]

    assert claim.language == "hinglish"
    assert claim.check_worthy


# --- multi-source packets ---------------------------------------------------


def test_image_packet_uses_ocr_and_tracks_source():
    packet = image_packet(
        caption="Look at this",
        ocr_text="Modi announces Rs 2000 for every farmer from January 2025",
        description="a poster with text and a photo of a man",
    )

    result = extract_claims(packet, today=TODAY, backend="heuristic")

    assert len(result.claims) == 1

    claim = result.claims[0]

    assert claim.source.field == "ocr"
    assert claim.source.image_index == 0
    assert claim.attributed_to == "Modi"

    dropped_fields = {d["field"] for d in result.dropped}
    assert dropped_fields == {"text", "caption"}


def test_duplicate_claim_across_caption_and_ocr_is_collapsed():
    sentence = "WHO confirms 5000 new dengue cases in Delhi this week."

    packet = image_packet(
        caption=sentence,
        ocr_text=sentence.upper(),
        description="",
    )

    result = extract_claims(packet, today=TODAY, backend="heuristic")

    assert len(result.claims) == 1
    assert result.claims[0].source.field == "text"


def test_video_frame_time_is_preserved():
    packet = image_packet(
        caption="",
        ocr_text="Earthquake kills 200 people in Nepal today",
        description="",
        frame_time=12.5,
    )
    packet["input_type"] = "video"

    claim = extract_claims(packet, today=TODAY, backend="heuristic").claims[0]

    assert claim.source.frame_time == 12.5
    assert claim.claim_type == "event"


# --- graph seed -------------------------------------------------------------


def test_graph_seed_links_claims_sharing_an_entity():
    packet = text_packet(
        "RBI has banned 2000 rupee notes from 1 October. "
        "RBI also confirmed that 500 rupee notes are safe."
    )

    result = extract_claims(packet, today=TODAY, backend="heuristic")
    graph = result.to_graph_seed()

    kinds = {n["kind"] for n in graph["nodes"]}
    assert kinds == {"packet", "claim", "entity"}

    types = {e["type"] for e in graph["edges"]}
    assert {"HAS_CLAIM", "MENTIONS", "SHARES_ENTITY"} <= types

    shared = [e for e in graph["edges"] if e["type"] == "SHARES_ENTITY"]
    rbi_id = next(n["id"] for n in graph["nodes"] if n.get("normalized") == "rbi")
    assert any(e["via"] == rbi_id for e in shared)


def test_claim_ids_are_stable_across_runs():
    packet = text_packet("ISRO launched Chandrayaan-3 on 14 July 2023.")

    first = extract_claims(packet, today=TODAY, backend="heuristic")
    second = extract_claims(packet, today=TODAY, backend="heuristic")

    assert first.packet_id == second.packet_id
    assert [c.id for c in first.claims] == [c.id for c in second.claims]


def test_claimset_serialises_to_json():
    packet = text_packet("ISRO launched Chandrayaan-3 on 14 July 2023.")

    payload = extract_claims(packet, today=TODAY, backend="heuristic").model_dump(mode="json")

    assert payload["claims"][0]["evidence_ids"] == []
    assert payload["claims"][0]["verdict"] is None
