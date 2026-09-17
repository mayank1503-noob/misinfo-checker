"""
Stage 4 (stance) tests.

The unit tests stub the two models so they run without downloading
anything. The embedder is replaced by a bag-of-words vectoriser, which
keeps the real cosine / top-k / cutoff code under test while making
similarity predictable; the NLI model is replaced by a scripted
premise -> label table. The integration test at the bottom uses the real
models and only runs when STANCE_TESTS_WITH_MODELS=1 is set.
"""

import os
import re

import pytest

from backend.stance import classify_stance, nli, rank
from backend.stance.classify import rating_stance
from backend.stance.passages import split_passages
from backend.stance.schema import EvidenceInput, StanceResult


CLAIM_ID = "clm_pop"
HINDI_CLAIM_ID = "clm_hi"


# --- fakes ------------------------------------------------------------------

# Words too common to mean two texts are about the same thing.
STOP = {
    "the", "a", "an", "of", "in", "on", "is", "are", "was", "were", "to",
    "and", "that", "this", "it", "for", "has", "have", "had", "by", "at",
    "से", "का", "के", "की", "है", "हैं", "और", "में", "को", "ने",
}


def _tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


def _bag_of_words(texts):
    """Term-count vectors over the vocabulary of this batch."""
    vocabulary = sorted({token for text in texts for token in _tokens(text)})
    index = {token: position for position, token in enumerate(vocabulary)}

    vectors = []

    for text in texts:
        vector = [0.0] * len(vocabulary)

        for token in _tokens(text):
            vector[index[token]] += 1.0

        vectors.append(vector)

    return vectors


def distribution(label, top=0.9):
    """A three-way NLI distribution peaked on `label`."""
    rest = round((1.0 - top) / 2, 4)

    return {
        name: (top if name == label else rest)
        for name in ("entailment", "neutral", "contradiction")
    }


@pytest.fixture
def models(monkeypatch):
    """
    Both models available and stubbed.

    Tests fill `models.script` with {substring of a passage: nli label};
    passages matching nothing come back neutral. `models.embed_calls` and
    `models.nli_calls` record what each model was asked.
    """

    class Fake:
        script = {}
        embed_calls = []
        nli_calls = []

    def embed(texts):
        texts = list(texts)
        Fake.embed_calls.append(texts)

        return _bag_of_words(texts)

    def score_pairs(premises, hypothesis, batch_size=nli.BATCH_SIZE):
        premises = list(premises)
        Fake.nli_calls.append((premises, hypothesis))

        results = []

        for premise in premises:
            label = "neutral"

            for needle, value in Fake.script.items():
                if needle in premise:
                    label = value
                    break

            results.append(distribution(label))

        return results

    monkeypatch.setattr(rank, "available", lambda: True)
    monkeypatch.setattr(rank, "embed", embed)
    monkeypatch.setattr(nli, "available", lambda: True)
    monkeypatch.setattr(nli, "score_pairs", score_pairs)

    return Fake


CLAIMS = [
    {"id": CLAIM_ID, "text": "India's population crossed 1.4 billion people in 2023."},
    {
        "id": HINDI_CLAIM_ID,
        "text": "रिज़र्व बैंक ने 500 रुपये के नोट बंद कर दिए हैं।",
    },
]


def evidence(**kwargs):
    base = {"id": "ev_1", "claim_id": CLAIM_ID, "weight": 0.8}
    base.update(kwargs)

    return base


# --- passage chunking -------------------------------------------------------


def test_chunks_are_two_to_three_sentences():
    text = "A one. A two. A three. A four."

    assert split_passages(text) == ["A one. A two.", "A three. A four."]


def test_tail_never_left_as_a_lone_sentence():
    text = " ".join("Sentence number {}.".format(n) for n in range(1, 8))

    passages = split_passages(text)

    assert [len(p.split(".")) - 1 for p in passages] == [3, 2, 2]
    assert " ".join(passages) == text


def test_short_document_is_one_passage():
    assert split_passages("Only one sentence here.") == ["Only one sentence here."]
    assert split_passages("") == []
    assert split_passages(None) == []


def test_devanagari_danda_splits_sentences():
    text = "यह पहला वाक्य है। यह दूसरा है। यह तीसरा है। यह चौथा है।"

    passages = split_passages(text)

    assert len(passages) == 2
    assert passages[0] == "यह पहला वाक्य है। यह दूसरा है।"


def test_unterminated_blob_is_broken_on_words():
    text = " ".join(["scraped"] * 600)      # ~4200 chars, no terminator

    passages = split_passages(text)

    assert len(passages) > 1
    assert all(passage.strip() for passage in passages)
    assert max(len(passage) for passage in passages) < 2500


def test_body_drops_parts_contained_in_fuller_ones():
    item = EvidenceInput(
        id="ev",
        claim_id=CLAIM_ID,
        title="Population crosses 1.4 billion",
        snippet="India's population crossed 1.4 billion",
        text="India's population crossed 1.4 billion people in 2023, the UN said.",
    )

    # The snippet is a prefix of the text, so only the title and text survive.
    assert item.body == (
        "Population crosses 1.4 billion\n"
        "India's population crossed 1.4 billion people in 2023, the UN said."
    )


# --- ranking ----------------------------------------------------------------


def test_ranking_keeps_top_three_above_the_cutoff(models):
    claim = "India population crossed billion"

    passages = [
        "India population crossed billion people.",     # near-identical
        "India population grew last year.",
        "India census counted people nationwide.",
        "Cricket scores from yesterday match.",         # nothing in common
    ]

    ranking = rank.rank_passages(claim, passages)

    assert len(ranking.passages) <= rank.TOP_K
    assert ranking.passages[0].text == passages[0]
    assert all(scored.score >= rank.RELEVANCE_CUTOFF for scored in ranking.passages)
    assert "Cricket" not in " ".join(scored.text for scored in ranking.passages)
    assert ranking.best == ranking.passages[0].score


def test_one_encode_call_per_claim_for_all_its_evidence(models):
    items = [
        evidence(
            id="ev_1",
            text="India's population crossed 1.4 billion people. The UN confirmed it. "
                 "Growth has slowed since.",
        ),
        evidence(
            id="ev_2",
            text="Population in India crossed 1.4 billion. China is now second. "
                 "The figures are for 2023.",
        ),
    ]

    classify_stance(CLAIMS, items)

    assert len(models.embed_calls) == 1
    assert len(models.nli_calls) == 1


# --- stance -----------------------------------------------------------------


def test_support(models):
    models.script = {"UN confirmed": "entailment"}

    item = evidence(
        title="India population crossed 1.4 billion",
        text="India's population crossed 1.4 billion people in 2023. "
             "The UN confirmed the 1.4 billion population figure for India. "
             "Growth has slowed since the 1990s.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert isinstance(result, StanceResult)
    assert result.stance == "support"
    assert result.method == "nli"
    assert result.score == pytest.approx(0.9)
    assert result.relevance >= rank.RELEVANCE_CUTOFF
    assert "UN confirmed" in result.best_passage
    assert result.misleading is False
    # the retriever's weight rides along for the verdict stage
    assert result.weight == 0.8
    assert result.weighted_score == pytest.approx(0.72)


def test_refute(models):
    models.script = {"never crossed": "contradiction"}

    item = evidence(
        text="India's population figures were revised this year. "
             "The population never crossed 1.4 billion people, officials said. "
             "The earlier estimate was withdrawn.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "refute"
    assert result.method == "nli"
    assert "never crossed" in result.best_passage


def test_strongest_non_neutral_wins_over_neutral_passages(models):
    models.script = {"never crossed": "contradiction"}

    item = evidence(
        text="India's population was discussed at a conference. "
             "Delegates reviewed the population data for 2023. "
             "The population never crossed 1.4 billion people, officials said. "
             "The session ended at noon.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "refute"
    assert "never crossed" in result.best_passage


def test_strongest_stance_prefers_the_most_confident_side():
    distributions = [
        distribution("neutral"),
        distribution("entailment", top=0.6),
        distribution("contradiction", top=0.95),
    ]

    assert nli.strongest_stance(distributions) == ("refute", 0.95, 2)


def test_strongest_stance_is_neutral_when_no_side_clears_the_floor():
    distributions = [{"entailment": 0.4, "neutral": 0.35, "contradiction": 0.25}]

    stance, score, index = nli.strongest_stance(distributions)

    assert (stance, index) == ("neutral", 0)
    assert score == pytest.approx(0.35)


def test_off_topic_evidence_is_neutral_and_never_reaches_the_model(models):
    item = evidence(
        title="Cricket board announces new schedule",
        text="The board released the tour schedule yesterday. "
             "Three matches will be played in October. "
             "Ticket sales open next week.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "neutral"
    assert result.method == "off_topic"
    assert result.best_passage is None
    assert result.relevance < rank.RELEVANCE_CUTOFF
    assert models.nli_calls == []


def test_hindi_claim_and_hindi_evidence(models):
    models.script = {"वापस ले लिया": "contradiction"}

    item = evidence(
        claim_id=HINDI_CLAIM_ID,
        title="500 रुपये के नोट बंद नहीं हुए",
        text="रिज़र्व बैंक ने 500 रुपये के नोट बंद करने की खबर पर सफाई दी। "
             "बैंक ने कहा कि ऐसा कोई फैसला वापस ले लिया गया है। "
             "नोट पूरी तरह वैध हैं।",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "refute"
    assert result.method == "nli"
    assert result.relevance >= rank.RELEVANCE_CUTOFF

    # the hypothesis handed to the model is the Hindi claim itself
    _premises, hypothesis = models.nli_calls[0]
    assert hypothesis == CLAIMS[1]["text"]


# --- ratings ----------------------------------------------------------------


def test_rating_table():
    assert rating_stance("False")[0] == "refute"
    assert rating_stance("Pants on fire!")[0] == "refute"
    assert rating_stance("भ्रामक")[0] == "refute"
    assert rating_stance("Mostly true")[0] == "support"
    assert rating_stance("Half true")[0] == "neutral"
    assert rating_stance("Partly false")[0] == "neutral"
    assert rating_stance(None) == (None, None)
    assert rating_stance("weird publisher verdict") == (None, None)


def test_rating_overrides_the_model(models):
    # The model reads this article as refuting the claim; the publisher rated
    # the claim true. The rating wins.
    models.script = {"population": "contradiction"}

    item = evidence(
        rating="Mostly true",
        text="India's population crossed 1.4 billion people in 2023. "
             "The population estimate comes from UN projections. "
             "A census is still pending.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "support"
    assert result.method == "rating"
    assert result.score == 1.0
    assert result.misleading is False
    assert "overrides nli 'refute'" in result.note


def test_fact_check_quoting_the_false_claim_is_flagged_misleading(models):
    # A fact-check headline restates the false claim verbatim, so entailment
    # reads as support. The "False" rating decides, and the conflict is
    # flagged so a verdict never counts this text as support.
    models.script = {"crossed 1.4 billion": "entailment"}

    item = evidence(
        title="Did India's population cross 1.4 billion in 2023?",
        snippet="A viral post claims India's population crossed 1.4 billion people.",
        text="A viral post claims India's population crossed 1.4 billion people. "
             "We checked the official figures. "
             "The claim is false: no such figure was published.",
        rating="False",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "refute"
    assert result.method == "rating"
    assert result.misleading is True
    assert "quotes the claim" in result.note
    # the quoted passage is still reported, so the verdict can show it
    assert result.best_passage is not None


def test_mixed_rating_is_neutral_with_a_soft_score(models):
    models.script = {"population": "entailment"}

    item = evidence(
        rating="Mixture",
        text="India's population crossed 1.4 billion people in 2023. "
             "Some states dispute the population projection. "
             "A census is still pending.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "neutral"
    assert result.method == "rating"
    assert result.score == 0.5


def test_rating_still_applies_when_the_text_is_off_topic(models):
    item = evidence(rating="False", text="Cricket tickets go on sale next week.")

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "refute"
    assert result.method == "rating"
    assert result.misleading is False


# --- failing soft -----------------------------------------------------------


def test_missing_nli_keeps_relevance_but_returns_neutral(models, monkeypatch):
    monkeypatch.setattr(nli, "available", lambda: False)

    item = evidence(
        text="India's population crossed 1.4 billion people in 2023. "
             "The UN confirmed the milestone. "
             "Growth has slowed.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert result.stance == "neutral"
    assert result.method == "unavailable"
    assert result.note == "nli unavailable"
    assert result.relevance >= rank.RELEVANCE_CUTOFF
    assert result.best_passage is not None


def test_missing_embedder_returns_neutral(models, monkeypatch):
    monkeypatch.setattr(rank, "available", lambda: False)

    result = classify_stance(CLAIMS, [evidence(text="Anything at all here.")])[0]

    assert (result.stance, result.method) == ("neutral", "unavailable")
    assert result.note == "embedder unavailable"


def test_model_exception_does_not_propagate(models, monkeypatch):
    def boom(premises, hypothesis, batch_size=nli.BATCH_SIZE):
        raise RuntimeError("no weights on disk")

    monkeypatch.setattr(nli, "score_pairs", boom)

    item = evidence(
        text="India's population crossed 1.4 billion people in 2023. "
             "The UN confirmed it. "
             "Growth has slowed.",
    )

    result = classify_stance(CLAIMS, [item])[0]

    assert (result.stance, result.method) == ("neutral", "unavailable")
    assert result.note == "nli failed: RuntimeError"


def test_empty_and_unknown_evidence_is_neutral(models):
    items = [
        evidence(id="ev_empty", text="", snippet=None, title=None),
        evidence(id="ev_orphan", claim_id="clm_missing", text="Some text here."),
    ]

    results = classify_stance(CLAIMS, items)

    assert [r.method for r in results] == ["unavailable", "unavailable"]
    assert [r.stance for r in results] == ["neutral", "neutral"]
    assert results[0].note == "no evidence text"
    assert "no text" in results[1].note


def test_empty_document_alongside_a_real_one_is_unavailable(models):
    models.script = {"UN confirmed": "entailment"}

    items = [
        evidence(
            id="ev_full",
            text="India's population crossed 1.4 billion people in 2023. "
                 "The UN confirmed the 1.4 billion population figure. "
                 "Growth has slowed.",
        ),
        evidence(id="ev_blank", title="   ", snippet=None, text=None),
    ]

    results = classify_stance(CLAIMS, items)

    assert results[0].stance == "support"
    assert (results[1].stance, results[1].method) == ("neutral", "unavailable")
    assert results[1].note == "no evidence text"


def test_results_come_back_one_per_input_in_order(models):
    models.script = {"UN confirmed": "entailment"}

    items = [
        evidence(id="ev_a", claim_id=HINDI_CLAIM_ID, text="नोट पूरी तरह वैध हैं।"),
        evidence(
            id="ev_b",
            text="India's population crossed 1.4 billion people. "
                 "The UN confirmed it. "
                 "Growth has slowed.",
        ),
        evidence(id="ev_c", text="Cricket tickets go on sale next week."),
    ]

    results = classify_stance(CLAIMS, items)

    assert [r.evidence_id for r in results] == ["ev_a", "ev_b", "ev_c"]
    assert [r.claim_id for r in results] == [HINDI_CLAIM_ID, CLAIM_ID, CLAIM_ID]
    assert results[1].stance == "support"


def test_accepts_claim_and_evidence_objects(models):
    class Claim:
        id = CLAIM_ID
        text = CLAIMS[0]["text"]

    models.script = {"UN confirmed": "entailment"}

    item = EvidenceInput(
        id="ev_obj",
        claim_id=CLAIM_ID,
        text="India's population crossed 1.4 billion people in 2023. "
             "The UN confirmed the milestone. "
             "Growth has slowed.",
    )

    result = classify_stance([Claim()], [item])[0]

    assert result.stance == "support"


# --- real models (opt-in) ---------------------------------------------------


@pytest.mark.skipif(
    os.getenv("STANCE_TESTS_WITH_MODELS") != "1",
    reason="set STANCE_TESTS_WITH_MODELS=1 to run against the real HF models",
)
def test_real_models_end_to_end():
    claims = [
        {"id": "c_true", "text": "Paris is the capital of France."},
        {"id": "c_false", "text": "Paris is the capital of Germany."},
    ]

    article = (
        "Paris is the capital and largest city of France. "
        "The French government and parliament sit in the city. "
        "Berlin, by contrast, is the capital of Germany."
    )

    items = [
        {"id": "ev_true", "claim_id": "c_true", "text": article, "weight": 0.9},
        {"id": "ev_false", "claim_id": "c_false", "text": article, "weight": 0.9},
        {
            "id": "ev_off",
            "claim_id": "c_true",
            "text": "The monsoon arrived over Kerala this week. "
                    "Rainfall was above average. "
                    "Farmers began sowing.",
            "weight": 0.9,
        },
    ]

    by_id = {r.evidence_id: r for r in classify_stance(claims, items)}

    assert by_id["ev_true"].stance == "support"
    assert by_id["ev_true"].method == "nli"
    assert by_id["ev_true"].relevance >= rank.RELEVANCE_CUTOFF

    assert by_id["ev_false"].stance == "refute"

    assert by_id["ev_off"].stance == "neutral"
    assert by_id["ev_off"].method == "off_topic"
