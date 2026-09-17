"""
Is this piece of evidence actually *about* this claim?

Retrieval answers a weaker question than the verdict needs. A multilingual
embedder asked for "Boiling water before drinking it reduces the risk of
waterborne disease" returns the hot-water-cures-COVID debunk, and it is
right to: both sentences are about drinking water, and 0.47 cosine is an
honest score for that. What is not honest is what happened next — the
debunk carries a publisher rating of `false`, the rating settled the
claim, and the system told someone that boiling their water is
misinformation (DECISIONS.md O6, the worst failure class the system has).

The first attempt at a fix asked the entailment model to notice. It does
not: on those pairs the model takes a side rather than abstaining, so
there was no abstention for a rule to defer to, and the one thing the
change did achieve was downgrading a real rumour. The measurement that
came out of it said where the defect actually is, and this module is that
place — **a relevance gate keyed on something other than the embedding
similarity that retrieved the evidence**, sitting between retrieval and
the stance model:

    claim -> retrieve -> [ is the evidence about this claim? ] -> stance
                              no -> neutral, rating withheld

## What it compares, and why lexically

It compares the claim with the evidence's *claim under review* — a
fact-check's headline claim, not its debunking prose. The prose is
written about the rumour and shares the rumour's vocabulary whichever
claim you arrived from, so it cannot tell the two apart; the reviewed
claim can.

The test is lexical on purpose. The failure being fixed is an embedding
being confidently wrong, and a second embedding of the same two sentences
is the same evidence twice. Word overlap is a genuinely independent
signal: "microchip", "coronavirus", "lungs" are simply absent from the
benign sentence, and no amount of topical closeness puts them there.

## The rule

Two directions are measured over stemmed content words:

  * `focus`    — how much of the *claim's* own content the evidence
                 claim accounts for;
  * `coverage` — how much of the *evidence claim's* content the claim
                 accounts for.

If either clears `ABOUT_FLOOR` the two texts are about the same thing
(one may be a shorter or longer telling of it, which is what a paraphrase
looks like from each side). Only when **neither** text accounts for half
of the other is the answer `NOT_ABOUT`: a shared topic with two different
assertions inside it, which is exactly the O6 shape.

## Three answers, and only one of them acts

`UNCLEAR` is not a polite `NOT_ABOUT`; it is the answer whenever a
lexical comparison would be meaningless, and it leaves behaviour exactly
as it was before this module existed:

  * **Different languages.** A Hinglish claim against an English
    fact-check shares its loanwords and nothing else — measured over the
    eval set, true Hinglish matches score 0.21-0.37 and the English
    benign mismatches score 0.25-0.35, so the two are not separable and
    any threshold would cost real rumours. Transliterating one side to
    meet the other does not rescue it either; see `comparable`. Hindi
    against Hindi and Hinglish against Hinglish are compared normally.
  * **Too few words to judge.** Under `MIN_TERMS` content words on
    either side, one word decides the ratio, so nothing is decided.

Because only NOT_ABOUT acts, and it acts by withholding an accusation,
the cost of a wrong answer here is one rumour reported `unverified`
instead of `false` — never a true statement called false.
"""

import re
from typing import NamedTuple, Tuple

from . import translit


# Either direction clearing this means the two texts are about the same
# thing. Measured against data/eval_cases.jsonl: true claim/fact-check
# pairs score 0.67-1.00 on at least one direction, while the benign
# statements that O6 is about reach 0.43 on neither.
ABOUT_FLOOR = 0.5

# Below this many content words on a side, the ratios are noise.
MIN_TERMS = 4

ABOUT = "about"
NOT_ABOUT = "not_about"
UNCLEAR = "unclear"

# Devanagari first: `\w` does not match the dependent vowel signs and the
# virama, so a bare `\w+` shreds "गर्म" into "गर" and "म". The class is the
# one `backend.translit` tokenises Devanagari with, danda and double
# danda excluded — they are sentence punctuation, not letters.
_WORD = re.compile(r"[ऀ-ॣ०-ॿ]+|\w+", re.UNICODE)

# Words that two texts can share without being about the same thing.
# Deliberately larger than `backend.evidence.normalize._STOP`, which
# measures how much of a claim's vocabulary a whole article repeats; this
# module compares two single claims, where one shared filler word is a
# much larger share of a much smaller set.
STOP = frozenset("""
the a an of in on at by for from to and or but so as if then than that this these those it its
is are was were be been being am has have had do does did will would can could should may might
must no not never you your yours we our us they them their he him his she her i me my mine
there here when while after before during until since under over above about into onto off out
up down again also very just only even still yet such each both other others same own any all
every some more most much many few less least next last new old
हैं है था थी थे का की के को में से और या पर यह वह ये वे कि भी ही तो एक नहीं होता होती होते
करने करता करती किया गया गई गए लिए हो रहा रही रहे अपने इस उस
""".split())


class Judgement(NamedTuple):
    """
    Why the gate answered the way it did, not just what it answered.

    `focus` and `coverage` are the two directions; `shared`,
    `claim_only` and `evidence_only` are the words behind them, so a
    verdict, a log line or a test can name the term that was missing.
    """

    answer: str                     # ABOUT | NOT_ABOUT | UNCLEAR
    focus: float
    coverage: float
    shared: Tuple[str, ...]
    claim_only: Tuple[str, ...]
    evidence_only: Tuple[str, ...]
    note: str

    @property
    def blocks_rating(self):
        """Whether a publisher rating on this item may settle the claim."""
        return self.answer == NOT_ABOUT


def stem(word):
    """
    A crude, dependency-free stem, enough to align the inflections two
    tellings of one claim differ by: plurals, -ing/-ed, and a silent
    final -e so "vaccines" and "vaccine" meet at the same token.

    Deliberately not a Porter stemmer: this runs on every evidence item
    in the bot's hot path, it has to behave identically on Devanagari
    (which it leaves alone, having no ASCII suffix to strip), and an
    over-eager stemmer invents overlap, which is the one error this
    module must not make. Hence the minimum stem lengths: without them
    "thing" stems to "th" and every -ing word in the language matches it.
    """
    def strip(suffix, keep, replacement=""):
        if not word.endswith(suffix) or len(word) - len(suffix) < keep:
            return None

        return word[:-len(suffix)] + replacement

    word = (
        strip("ies", 3, "y")
        or strip("sses", 3, "ss")
        or (None if word.endswith("ss") else strip("s", 3))
        or word
    )

    word = strip("ing", 4) or strip("ed", 4) or word

    return strip("e", 4) or word


def content_terms(text):
    """The stemmed content words of `text`, as a set."""
    return {
        stem(token)
        for token in _WORD.findall((text or "").lower())
        if len(token) > 1 and token not in STOP
    }


def _script(text):
    """
    Which vocabulary `text` is written in: Devanagari, romanised Hindi,
    or Latin. Two texts are only comparable word by word when these agree
    — "paani" and "water" are the same word and share no letters.
    """
    if translit.is_devanagari(text):
        return "deva"

    if translit.looks_romanised(text):
        return "hinglish"

    return "latin"


def comparable(claim_text, evidence_text):
    """
    Whether the two texts are written in the same vocabulary, and so can
    be compared word by word at all.

    `backend.translit` could put a Hinglish claim into Devanagari and
    compare it with a Hindi fact-check, and that was tried: the lexicon
    is deliberately partial, so a rewritten claim keeps whatever words it
    does not know, and the overlap that comes back measures the
    lexicon's coverage as much as the claim's content. Measured over the
    eval set it mis-gated four of the eight Hinglish pairs — a low score
    there means "not transliterated" at least as often as it means "not
    about". So the gate reads what is actually written and abstains
    across languages, which costs it nothing it had before.
    """
    return _script(claim_text) == _script(evidence_text)


def _measure(claim_text, evidence_text):
    """(focus, coverage, shared, claim_only, evidence_only) or None."""
    claim_terms = content_terms(claim_text)
    evidence_terms = content_terms(evidence_text)

    if len(claim_terms) < MIN_TERMS or len(evidence_terms) < MIN_TERMS:
        return None

    shared = claim_terms & evidence_terms

    return (
        len(shared) / len(claim_terms),
        len(shared) / len(evidence_terms),
        shared,
        claim_terms - evidence_terms,
        evidence_terms - claim_terms,
    )


def _unclear(note):
    return Judgement(UNCLEAR, 0.0, 0.0, (), (), (), note)


def judge(claim_text, evidence_claim_text):
    """
    Whether `evidence_claim_text` — the claim a piece of evidence is
    about — is the same claim as `claim_text`.

    Never raises and never guesses: anything it cannot measure comes back
    UNCLEAR, which callers must treat as "carry on as before".
    """
    claim_text = (claim_text or "").strip()
    evidence_claim_text = (evidence_claim_text or "").strip()

    if not claim_text or not evidence_claim_text:
        return _unclear("nothing to compare")

    if not comparable(claim_text, evidence_claim_text):
        return _unclear("the claim and the evidence are in different languages")

    measured = _measure(claim_text, evidence_claim_text)

    if measured is None:
        return _unclear("too few content words to compare")

    focus, coverage, shared, claim_only, evidence_only = measured

    words = (
        tuple(sorted(shared)), tuple(sorted(claim_only)), tuple(sorted(evidence_only)),
    )

    if max(focus, coverage) >= ABOUT_FLOOR:
        return Judgement(
            ABOUT, round(focus, 4), round(coverage, 4), *words,
            "shares {} content words with the claim".format(len(shared)),
        )

    return Judgement(
        NOT_ABOUT, round(focus, 4), round(coverage, 4), *words,
        "same topic, different claim: the evidence is about {}".format(
            ", ".join(sorted(evidence_only)[:4]) or "something else"
        ),
    )


# Best answer first: one text saying the evidence is about the claim
# outweighs another that could not tell.
_PREFERENCE = {ABOUT: 0, UNCLEAR: 1, NOT_ABOUT: 2}


def judge_any(claim_text, *evidence_texts):
    """
    Judge `claim_text` against several tellings of what the evidence is
    about, and return the most favourable answer.

    Retrievers disagree about where the claim under review lives: the
    seed index puts it in `title` (and its debunk summary in `snippet`),
    while the Google Fact Check API puts the review's headline in `title`
    and the claim as the publisher recorded it in `snippet`. Rather than
    teach this module every retriever's layout, it is handed both and
    NOT_ABOUT has to survive all of them — a gate that fires on the one
    field that happened to be prose would be a gate that fires by
    accident.
    """
    judgements = [
        judge(claim_text, text) for text in evidence_texts if (text or "").strip()
    ]

    if not judgements:
        return _unclear("nothing to compare")

    return min(judgements, key=lambda judgement: _PREFERENCE[judgement.answer])
