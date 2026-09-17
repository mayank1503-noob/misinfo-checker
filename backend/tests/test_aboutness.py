"""
The relevance gate: is a piece of evidence about the claim it was
retrieved for? (DECISIONS.md O6.)

The three pairs at the top are the defect this module was written for,
taken verbatim from data/eval_cases.jsonl and data/seed_factchecks.jsonl,
so if the seed corpus is reworded these tests are what notices. The rest
pin the properties that make the gate safe to act on: it abstains across
languages, it abstains on texts too short to measure, and it never says
"about" for something it has not actually compared.
"""

import pytest

from backend import aboutness
from backend.aboutness import ABOUT, NOT_ABOUT, UNCLEAR, judge, judge_any


# --- the O6 pairs -----------------------------------------------------------

# (claim, the claim the retrieved fact-check actually reviews)
O6_PAIRS = [
    (
        "Boiling water before drinking it reduces the risk of waterborne disease.",
        "Drinking hot water every 15 minutes kills the coronavirus before it "
        "reaches the lungs.",
    ),
    (
        "COVID-19 vaccines were tested in clinical trials before being approved.",
        "COVID-19 vaccines contain a microchip that tracks the person who "
        "received the dose.",
    ),
    (
        "COVID-19 vaccines were tested in clinical trials before being approved.",
        "5G tower radiation causes COVID-19 symptoms and is the real reason for "
        "the second wave.",
    ),
    (
        "You should never share your UPI PIN with anyone, including bank staff.",
        "You must enter your UPI PIN to receive the refund credited to your "
        "account.",
    ),
]


@pytest.mark.parametrize("claim,reviewed", O6_PAIRS)
def test_a_neighbouring_rumour_is_not_about_a_benign_claim(claim, reviewed):
    judgement = judge(claim, reviewed)

    assert judgement.answer == NOT_ABOUT
    assert judgement.blocks_rating is True
    # the words that decided it are reported, so a log or a reply can
    # name what the fact-check was actually about
    assert judgement.evidence_only


def test_the_note_names_what_the_evidence_was_about():
    judgement = judge(*O6_PAIRS[1])

    assert "microchip" in judgement.note


# --- the rumours themselves must still get through --------------------------

MATCHING_PAIRS = [
    # the rumour as forwarded, and as the fact-checker recorded it
    (
        "Drinking hot water every 15 minutes kills the coronavirus before it "
        "reaches the lungs.",
        "Drinking hot water every 15 minutes kills the coronavirus before it "
        "reaches the lungs.",
    ),
    (
        "COVID-19 vaccines contain a microchip that tracks the person who "
        "received it.",
        "COVID-19 vaccines contain a microchip that tracks the person who "
        "received the dose.",
    ),
    # the same claim told shorter and longer: one direction carries it
    (
        "The RBI has banned Rs 500 notes.",
        "The Reserve Bank of India has banned Rs 500 notes from midnight.",
    ),
    # a Hindi claim against a Hindi fact-check
    (
        "गाय के मूत्र से कैंसर ठीक हो जाता है और डॉक्टर इसे छिपा रहे हैं।",
        "गाय के मूत्र से कैंसर ठीक होता है और डॉक्टर यह बात छिपा रहे हैं।",
    ),
]


@pytest.mark.parametrize("claim,reviewed", MATCHING_PAIRS)
def test_a_fact_check_of_the_same_claim_is_about_it(claim, reviewed):
    judgement = judge(claim, reviewed)

    assert judgement.answer == ABOUT
    assert judgement.blocks_rating is False


def test_one_direction_is_enough():
    """
    A short claim inside a long reviewed claim covers little of it, and a
    long claim covers a short one badly. Either direction clearing the
    floor means the same thing is being talked about, so both are tried.
    """
    claim = "Aadhaar-PAN linking is compulsory by tomorrow."
    reviewed = (
        "Aadhaar-PAN linking is compulsory by tomorrow or your bank account "
        "will be frozen permanently by the income tax department."
    )

    judgement = judge(claim, reviewed)

    assert judgement.focus > judgement.coverage
    assert judgement.answer == ABOUT


# --- abstaining -------------------------------------------------------------


def test_different_languages_are_unclear_not_not_about():
    """
    A Hinglish claim and an English fact-check of the same rumour share
    only their loanwords, which is indistinguishable from sharing only a
    topic. Guessing here would downgrade real rumours, so the gate
    abstains and stage 4 carries on exactly as it did before.
    """
    judgement = judge(
        "Garam paani peene se corona theek ho jata hai, WHO ne kaha hai.",
        "Drinking hot water every 15 minutes kills the coronavirus before it "
        "reaches the lungs.",
    )

    assert judgement.answer == UNCLEAR
    assert judgement.blocks_rating is False
    assert "different languages" in judgement.note


def test_devanagari_and_english_are_unclear():
    judgement = judge(
        "हर 15 मिनट में गर्म पानी पीने से कोरोना वायरस मर जाता है।",
        "Boiling water reduces the risk of waterborne disease.",
    )

    assert judgement.answer == UNCLEAR


def test_a_claim_too_short_to_measure_is_unclear():
    judgement = judge("Modi resigned.", "The Prime Minister has not resigned.")

    assert judgement.answer == UNCLEAR
    assert "too few content words" in judgement.note


def test_nothing_to_compare_is_unclear():
    for left, right in (("", "anything"), ("anything", ""), (None, None)):
        assert judge(left, right).answer == UNCLEAR


def test_shared_topic_words_alone_do_not_make_it_about():
    """
    The failure mode in one line: every content word the two sentences
    share is the topic, and nothing they share is the assertion.
    """
    judgement = judge(
        "Boiling water before drinking it reduces the risk of waterborne disease.",
        "Drinking hot water every 15 minutes kills the coronavirus before it "
        "reaches the lungs.",
    )

    assert set(judgement.shared) <= {"water", "drink"}
    assert "coronaviru" in judgement.evidence_only


# --- tokenising -------------------------------------------------------------


def test_inflections_of_one_word_meet_at_one_token():
    assert aboutness.stem("vaccines") == aboutness.stem("vaccine")
    assert aboutness.stem("trials") == aboutness.stem("trial")
    assert aboutness.stem("kills") == aboutness.stem("kill")
    assert aboutness.stem("boiling") == aboutness.stem("boil")


def test_devanagari_words_survive_tokenising_whole():
    """
    `\\w` does not match the dependent vowel signs or the virama, so a
    bare `\\w+` splits "गर्म" into "गर" and "म" and two Hindi sentences
    then overlap on fragments. This is what that regression looks like.
    """
    terms = aboutness.content_terms("हर 15 मिनट में गर्म पानी पीने से कोरोना")

    assert "गर्म" in terms
    assert "पानी" in terms
    assert "गर" not in terms


def test_the_danda_is_not_part_of_a_word():
    assert "जाता" in aboutness.content_terms("मर जाता है।")


def test_stop_words_cannot_carry_a_judgement():
    terms = aboutness.content_terms("The next new thing is that it will be here.")

    assert terms == {"thing"}


# --- several tellings of what the evidence is about --------------------------


def test_judge_any_takes_the_most_favourable_answer():
    """
    Retrievers put the claim under review in different fields — the seed
    index in `title`, the Google Fact Check API in `snippet` — so both
    are offered and the best answer wins. Here the debunk's prose shares
    little with the claim while the reviewed claim is the claim.
    """
    claim = "Cow urine cures cancer and doctors are hiding the evidence."

    reviewed = "Cow urine cures cancer and doctors are hiding the evidence."
    prose = (
        "No trial has shown any anti-tumour effect in humans. Oncology bodies "
        "in India have repeatedly warned that abandoning treatment for this "
        "costs lives."
    )

    assert judge(claim, prose).answer == NOT_ABOUT
    assert judge_any(claim, reviewed, prose).answer == ABOUT


def test_judge_any_gates_only_when_every_telling_agrees():
    claim = "Boiling water before drinking it reduces the risk of waterborne disease."

    reviewed = (
        "Drinking hot water every 15 minutes kills the coronavirus before it "
        "reaches the lungs."
    )
    prose = (
        "Drinking water cannot reach the respiratory tract where the virus "
        "replicates, and water hot enough to kill a virus would scald the "
        "throat."
    )

    assert judge_any(claim, reviewed, prose).answer == NOT_ABOUT


def test_judge_any_with_nothing_to_read_is_unclear():
    assert judge_any("a claim with several content words", None, "").answer == UNCLEAR
