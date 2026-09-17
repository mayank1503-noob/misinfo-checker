"""
Transformer backend tests.

The unit tests below stub the model calls so they run without downloading
anything. The final integration test uses the real models and only runs
when CLAIM_TESTS_WITH_MODELS=1 is set.
"""

import os
from datetime import date

import pytest

from backend.claims import extract_claims, transformer
from backend.claims.extractor import _merge_entities, resolve_backend
from backend.claims.schema import Entity, stable_id


TODAY = date(2026, 9, 17)


def text_packet(text):
    return {"input_type": "text", "text": text, "source_date": None, "images": []}


def entity(label, text, start):
    return Entity(
        id=stable_id("ent", label, text.lower()),
        text=text,
        label=label,
        normalized=text.lower(),
        start=start,
        end=start + len(text),
    )


@pytest.fixture
def fake_models(monkeypatch):
    """
    Replace the three model entry points with scripted responses.
    Tests set fake.ner / fake.worthy / fake.types per sentence.
    """

    class Fake:
        ner = {}
        worthy = {}
        types = {}
        calls = []

    monkeypatch.setattr(transformer, "available", lambda: True)

    def ner_entities(sentence):
        Fake.calls.append(("ner", sentence))
        return [entity(l, t, sentence.index(t)) for l, t in Fake.ner.get(sentence, [])]

    def worthiness(sentence):
        Fake.calls.append(("worthy", sentence))
        return Fake.worthy.get(sentence, 0.5)

    def claim_type(sentence):
        Fake.calls.append(("type", sentence))
        return Fake.types.get(sentence, (None, 0.0, 0.0))

    monkeypatch.setattr(transformer, "ner_entities", ner_entities)
    monkeypatch.setattr(transformer, "worthiness", worthiness)
    monkeypatch.setattr(transformer, "claim_type", claim_type)

    return Fake


# --- backend selection ------------------------------------------------------


def test_auto_falls_back_to_heuristic_when_transformers_missing(monkeypatch):
    monkeypatch.setattr(transformer, "available", lambda: False)

    assert resolve_backend("auto") == "heuristic"

    with pytest.raises(RuntimeError):
        resolve_backend("transformer")


def test_auto_picks_transformer_when_available(monkeypatch):
    monkeypatch.setattr(transformer, "available", lambda: True)

    assert resolve_backend("auto") == "transformer"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        resolve_backend("gpt")


# --- entity merging ---------------------------------------------------------


def test_model_ner_replaces_regex_guesses_but_keeps_patterns():
    sentence = "Massive quake hits Kathmandu, Rs 5 crore aid announced"

    regex = [
        entity("PROPER", "Massive", 0),          # bad regex guess
        entity("MONEY", "Rs 5 crore", 30),        # pattern: must survive
    ]
    model = [entity("GPE", "Kathmandu", 19)]

    merged = _merge_entities(regex, model)

    labels = [(e.label, e.text) for e in merged]

    assert ("GPE", "Kathmandu") in labels
    assert ("MONEY", "Rs 5 crore") in labels
    # regex guesses that don't overlap the model are still kept
    assert ("PROPER", "Massive") in labels


def test_regex_named_entity_dropped_when_model_covers_the_span():
    regex = [entity("PERSON", "Reserve Bank", 0)]
    model = [entity("ORG", "Reserve Bank of India", 0)]

    merged = _merge_entities(regex, model)

    assert [(e.label, e.text) for e in merged] == [("ORG", "Reserve Bank of India")]


# --- scoring / typing blend -------------------------------------------------


def test_model_worthiness_is_blended_into_confidence(fake_models):
    sentence = "Kathmandu recorded 300 deaths after the quake."
    fake_models.worthy[sentence] = 1.0

    result = extract_claims(text_packet(sentence), today=TODAY, backend="transformer")
    claim = result.claims[0]

    assert result.backend == "transformer"
    assert claim.signals["backend"] == "transformer"
    assert claim.signals["model_worthy"] == 1.0

    heuristic = claim.signals["heuristic_confidence"]
    assert claim.confidence == pytest.approx(0.5 * heuristic + 0.5 * 1.0, abs=1e-3)
    assert claim.confidence > heuristic


def test_model_can_demote_a_sentence_below_threshold(fake_models):
    # Heuristically this passes (number + named entity); model says it's chit-chat.
    sentence = "Ramesh ordered 3 pizzas for the party."
    fake_models.worthy[sentence] = 0.0

    result = extract_claims(text_packet(sentence), today=TODAY, backend="transformer")

    assert result.claims == []
    assert result.dropped[0]["reason"] == "model_not_claim"


def test_model_type_overrides_heuristic_when_confident(fake_models):
    sentence = "The bridge will collapse by 2030 according to engineers."
    fake_models.worthy[sentence] = 0.9
    fake_models.types[sentence] = ("prediction", 0.7, 0.4)

    claim = extract_claims(text_packet(sentence), today=TODAY, backend="transformer").claims[0]

    assert claim.signals["heuristic_type"] == "attribution"
    assert claim.claim_type == "prediction"


def test_model_type_ignored_when_margin_is_low(fake_models):
    sentence = "The bridge will collapse by 2030 according to engineers."
    fake_models.worthy[sentence] = 0.9
    fake_models.types[sentence] = (None, 0.3, 0.02)

    claim = extract_claims(text_packet(sentence), today=TODAY, backend="transformer").claims[0]

    assert claim.claim_type == "attribution"


def test_chain_offer_type_is_sticky_against_model(fake_models):
    sentence = "Send Rs 10 to sbirewards@ybl to claim your cashback now."
    fake_models.worthy[sentence] = 0.9
    fake_models.types[sentence] = ("money", 0.8, 0.5)

    claim = extract_claims(text_packet(sentence), today=TODAY, backend="transformer").claims[0]

    assert claim.claim_type == "chain_offer"
    assert claim.signals["model_type"] == "money"


def test_model_ner_entities_show_up_in_graph(fake_models):
    sentence = "Sunita Williams returned to Earth after 9 months in orbit."
    fake_models.ner[sentence] = [("PERSON", "Sunita Williams")]
    fake_models.worthy[sentence] = 0.9

    result = extract_claims(text_packet(sentence), today=TODAY, backend="transformer")
    graph = result.to_graph_seed()

    persons = [n for n in graph["nodes"] if n["kind"] == "entity" and n["label"] == "PERSON"]

    assert [p["text"] for p in persons] == ["Sunita Williams"]


def test_hard_rejects_skip_the_model(fake_models):
    result = extract_claims(text_packet("Good night!"), today=TODAY, backend="transformer")

    assert result.claims == []
    assert not any(kind in ("worthy", "type") for kind, _ in fake_models.calls)


# --- real models (opt-in) ---------------------------------------------------


@pytest.mark.skipif(
    os.getenv("CLAIM_TESTS_WITH_MODELS") != "1",
    reason="set CLAIM_TESTS_WITH_MODELS=1 to run against the real HF models",
)
def test_real_models_end_to_end():
    packet = text_packet(
        "Hey bro kaise ho? "
        "Narendra Modi announced Rs 2000 for every farmer in Punjab on 8 June 2024. "
        "I think it is a good idea."
    )

    result = extract_claims(packet, today=TODAY, backend="transformer")

    assert result.backend == "transformer"
    assert len(result.claims) == 1

    claim = result.claims[0]

    labels = {(e.label, e.text) for e in claim.entities}

    assert ("PERSON", "Narendra Modi") in labels
    assert ("GPE", "Punjab") in labels
    assert any(l == "MONEY" for l, _ in labels)
    assert claim.signals["model_worthy"] > 0.5
