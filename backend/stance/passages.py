"""
Split an evidence document into passages small enough to be judged.

An entailment model reads one premise at a time, and a whole article is a
poor premise: the sentence that bears on the claim gets averaged away with
everything else. Single sentences are the opposite problem — "He denied
it." means nothing alone. So passages are 2-3 sentences: enough for a
pronoun to find its antecedent, short enough that the model is judging one
thought.

Sentence splitting is reused from the claim stage
(`backend.claims.segment`), so Devanagari dandas and "Rs." behave the same
way here as they do there.
"""

from ..claims.segment import clean, split_sentences


MIN_SENTENCES = 2
MAX_SENTENCES = 3

# A "sentence" with no terminator (scraped nav text, a wall of OCR) can be
# arbitrarily long; past this many characters it is broken up on words so a
# single blob cannot swamp a passage.
MAX_CHARS = 800


def _split_long(sentence, max_chars):
    """Break an over-long unterminated sentence on word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]

    pieces = []
    current = []
    size = 0

    for word in sentence.split():
        if current and size + 1 + len(word) > max_chars:
            pieces.append(" ".join(current))
            current = []
            size = 0

        current.append(word)
        size += len(word) + (1 if size else 0)

    if current:
        pieces.append(" ".join(current))

    return pieces


def split_passages(
    text,
    min_sentences=MIN_SENTENCES,
    max_sentences=MAX_SENTENCES,
    max_chars=MAX_CHARS,
):
    """
    Return the text as a list of 2-3 sentence passages.

    Groups are `max_sentences` long; if the tail would be left as a lone
    sentence it borrows from the group before it, so no passage is shorter
    than `min_sentences` — unless the whole document is that short, in
    which case it comes back as one passage of whatever there is.
    """
    sentences = []

    for sentence, _start, _end in split_sentences(clean(text)):
        sentences.extend(_split_long(sentence, max_chars))

    if not sentences:
        return []

    groups = [
        sentences[i:i + max_sentences]
        for i in range(0, len(sentences), max_sentences)
    ]

    while (
        len(groups) > 1
        and len(groups[-1]) < min_sentences
        and len(groups[-2]) > min_sentences
    ):
        groups[-1].insert(0, groups[-2].pop())

    return [" ".join(group) for group in groups]
