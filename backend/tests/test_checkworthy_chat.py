"""
Stage 2 precision on ordinary messages (DECISIONS.md O2).

Personal chat used to clear the check-worthiness threshold, which cost a
retrieval round per message. Two defects fed each other:

  * a lone capitalised word at the start of a sentence was read as a
    name, which is how "Kal", "Namak" and "Aaj" each bought +0.15 — an
    artifact of orthography, and one that hits romanised Hindi hardest
    because its everyday words are in no English vocabulary; and
  * the personal-chat penalty was switched off by the presence of a
    name, so that artifact also disabled the rule meant to catch it.

These tests pin both directions: chat stays out, and the rumours that
look superficially like chat — a vocative, a first-person verb — stay in.
"""

import pytest

from backend.claims import extract_claims
from backend.claims.checkworthy import score
from backend.claims.entities import extract_entities


def worthy(text):
    result = extract_claims({"input_type": "text", "text": text},
                            backend="heuristic")

    return any(claim.check_worthy for claim in result.claims)


CHAT = [
    "Kal shaam ko ghar aa raha hoon.",
    "Amma ne poocha tha ki tum dinner ke liye rukoge ya nahi.",
    "Namak lekar aana ghar aate waqt, khatam ho gaya hai.",
    "Happy birthday bhai, bahut bahut badhai ho aapko.",
    "Aaj office mein bahut kaam tha, isliye reply nahi kar paya.",
    "Please send me the meeting notes from yesterday before the standup.",
]

CLAIMS = [
    # A vocative opening is the commonest disguise a forward wears.
    "Bhai suno, WHO ne kaha hai ki garam paani peene se corona theek ho jata hai.",
    "Sarkari camp mein di jane wali polio drops se bachchon mein bandhyapan ho jata hai.",
    "SBI is giving Rs 5,000 cashback to every customer today.",
    "Modi announces Rs 2000 for every farmer from January 2025",
]

# Pre-existing misses, unrelated to O2 and unchanged by it. Recorded as
# xfail so they stay visible instead of being quietly absent: this one
# trips the OPINION rule on "the best", which is an opinion marker in
# "this is the best phone" and part of the claim in "declared it the
# best national anthem".
KNOWN_MISSES = [
    "UNESCO has declared Jana Gana Mana the best national anthem in the world.",
]


@pytest.mark.parametrize("text", CHAT)
def test_personal_chat_is_not_check_worthy(text):
    assert not worthy(text)


@pytest.mark.parametrize("text", CLAIMS)
def test_claims_are_still_check_worthy(text):
    assert worthy(text)


@pytest.mark.xfail(reason="pre-existing: OPINION matches 'the best'", strict=True)
@pytest.mark.parametrize("text", KNOWN_MISSES)
def test_known_misses(text):
    assert worthy(text)


def test_a_sentence_initial_capital_alone_is_not_a_name():
    """The artifact behind most of O2's false positives."""
    entities = extract_entities("Namak lekar aana ghar aate waqt.")

    assert not [e for e in entities if e.text == "Namak"]


def test_a_name_in_subject_position_survives():
    """
    The guard must not eat real names.

    "Modi announces ..." is a capitalised word doing a verb, which is
    corroboration independent of where it sits in the sentence.
    """
    entities = extract_entities(
        "Modi announces Rs 2000 for every farmer from January 2025"
    )

    assert [e for e in entities if e.text == "Modi"]


def test_an_acronym_or_place_still_survives():
    assert [e for e in extract_entities("UNESCO has declared it the best.")
            if e.text == "UNESCO"]
    assert [e for e in extract_entities("Mumbai mein kal baarish hui thi.")
            if e.text == "Mumbai"]


def test_a_repeated_capital_survives():
    """Capitalised again mid-sentence, where the sentence start cannot explain it."""
    entities = extract_entities("Sharma ne kaha ki Sharma aayega.")

    assert [e for e in entities if e.text == "Sharma"]


def test_a_name_does_not_cancel_the_chat_penalty():
    """
    The second defect, pinned directly at the scorer.

    A message addressed to somebody stays chat even with a name in it;
    only a quantity earns the exemption.
    """
    sentence = "Rahul, call me when you reach home."
    entities = extract_entities(sentence)

    confidence, check_worthy, signals = score(sentence, entities, [])

    assert signals["personal"]
    assert not check_worthy


def test_a_quantity_still_overrides_the_chat_penalty():
    """"They credited me Rs 5,000" is personal in phrasing and checkable in fact."""
    sentence = "I received Rs 5,000 cashback from SBI today."
    entities = extract_entities(sentence)

    _confidence, check_worthy, signals = score(sentence, entities, [])

    assert signals["quantity"]
    assert check_worthy
