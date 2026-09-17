"""
Romanised Hindi -> Devanagari (backend/evidence/translit.py).

The module exists to raise Hinglish retrieval, but the tests that matter
most here are the negative ones: an English query and a Devanagari query
must come out of `devanagari()` as None, because that None is what
guarantees the rest of the pipeline scores them exactly as it did before
transliteration existed.
"""

import pytest

from backend.evidence import translit


# --- the gate ---------------------------------------------------------


ENGLISH = [
    "Drinking hot water kills the coronavirus",
    "RBI cut the repo rate by 25 basis points at its policy meeting",
    "Please send me the meeting notes from yesterday",
    "The cricket match was postponed due to rain in Mumbai",
    # "bank", "vote", "do" and "the" are all in the lexicon; none of them
    # is native Hindi, so none of them may open the gate.
    "The bank will do the vote count",
    "Meeting agenda attached, please review before tomorrow",
]


@pytest.mark.parametrize("text", ENGLISH)
def test_english_is_never_transliterated(text):
    assert translit.devanagari(text) is None
    assert translit.variants(text) == [text]


@pytest.mark.parametrize("text", [
    "गर्म पानी पीने "
    "से कोरोना ठीक "
    "होता है",
    "SBI हर ग्राहक को "
    "5,000 रुपये",
])
def test_devanagari_is_left_alone(text):
    """Already in script: nothing to do, and no second vector to embed."""
    assert translit.devanagari(text) is None
    assert translit.variants(text) == [text]


def test_one_hindi_word_in_an_english_sentence_is_not_enough():
    assert translit.devanagari("The sarkar announced a new policy today") is None


def test_empty_input_yields_no_variants():
    assert translit.variants("") == []
    assert translit.variants(None) == []
    assert translit.devanagari("") is None


# --- what it actually rewrites ----------------------------------------


def test_the_hot_water_rumour_becomes_devanagari():
    out = translit.devanagari("garam paani peene se corona theek ho jata hai")

    assert out == (
        "गरम पानी पीने "
        "से कोरोना ठीक "
        "हो जाता है"
    )


def test_variants_puts_the_original_first():
    variants = translit.variants("gomutra cancer theek karta hai")

    assert len(variants) == 2
    assert variants[0] == "gomutra cancer theek karta hai"
    assert translit.is_devanagari(variants[1])


@pytest.mark.parametrize("a,b", [
    ("garam paani", "garam pani"),
    ("theek", "thik"),
    ("nahi", "nahin"),
    ("zaroor", "jaroor"),
    ("bahut jaldi", "bohut jaldi"),
    ("jhooth", "jhuth"),
    ("woh", "voh"),
    ("bahut", "bahuuut"),                            # leaning on the key
])
def test_spelling_variants_land_on_the_same_devanagari(a, b):
    """paani/pani, theek/thik — the same rumour, a different typist."""
    assert translit.transliterate(a) == translit.transliterate(b)


def test_acronyms_and_numbers_survive():
    out = translit.devanagari("SBI har grahak ko 5000 rupaye de raha hai")

    assert "SBI" in out
    assert "5000" in out


def test_an_unknown_word_is_left_in_latin_script():
    """
    Coverage is the lexicon's limit, and the failure is the safe one: an
    unrecognised word is left exactly as the retriever would have seen it.
    """
    out = translit.devanagari("yeh jalebi bahut accha hai")

    assert "jalebi" in out
    assert translit.is_devanagari(out)


def test_loanwords_are_rewritten_once_the_gate_is_open():
    out = translit.devanagari("kal tak bank khata band ho jayega")

    assert "bank" not in out
    assert "बैंक" in out                  # बैंक


# --- the gate's own arithmetic ----------------------------------------


def test_hindi_ratio_counts_only_unambiguous_hindi():
    _ratio, hits = translit.hindi_ratio("Please send me the meeting notes")

    assert hits == 0


def test_hindi_ratio_counts_native_words():
    _ratio, hits = translit.hindi_ratio("sarkar ne kaha hai")

    assert hits >= 2


@pytest.mark.parametrize("text", [
    "sarkar sabhi students ko free laptop de rahi hai",
    "polio drops se bachchon mein bandhyapan hota hai",
    "jio 3 mahine ka free recharge de raha hai",
])
def test_real_hinglish_rumours_open_the_gate(text):
    assert translit.looks_romanised(text)
