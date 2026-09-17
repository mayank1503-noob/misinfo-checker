"""
Natural language inference: does this passage entail or contradict the claim?

The model is the same multilingual NLI checkpoint the claim stage already
uses for zero-shot check-worthiness
(MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), reached through that stage's
`lru_cache`d pipeline so both stages share one set of weights in memory
rather than loading a second copy. Only the model and tokenizer are
borrowed — the zero-shot pipeline builds its own hypotheses from a
template, whereas here the hypothesis *is* the claim:

    premise    = a passage of the evidence
    hypothesis = the claim under test

Pairs are scored in batches, and the strongest non-neutral verdict across
a document's passages wins: one passage flatly contradicting the claim is
a refutation even when the four around it are neutral filler.
"""

from functools import lru_cache

from ..claims import transformer as claims_transformer


LABELS = ("entailment", "neutral", "contradiction")

STANCE_BY_LABEL = {
    "entailment": "support",
    "contradiction": "refute",
}

# A winning label below this probability is not worth a side.
MIN_STANCE_SCORE = 0.50

BATCH_SIZE = 8
MAX_LENGTH = 512


def available():
    return claims_transformer.available()


def _canonical(label):
    text = str(label).lower()

    if "entail" in text:
        return "entailment"

    if "contra" in text:
        return "contradiction"

    if "neutral" in text:
        return "neutral"

    return text


@lru_cache
def nli_model():
    """
    (model, tokenizer, id -> label) from the cached mDeBERTa pipeline in
    backend/claims. Label order differs between checkpoints, so it is read
    off the config rather than assumed.
    """
    pipeline = claims_transformer.zsc_pipeline()

    id2label = {
        int(index): _canonical(label)
        for index, label in pipeline.model.config.id2label.items()
    }

    return pipeline.model, pipeline.tokenizer, id2label


def score_pairs(premises, hypothesis, batch_size=BATCH_SIZE):
    """
    Score every premise against one hypothesis.

    Returns one {label: probability} dict per premise, in order.
    """
    premises = list(premises)

    if not premises:
        return []

    import torch

    model, tokenizer, id2label = nli_model()

    distributions = []

    for start in range(0, len(premises), batch_size):
        batch = premises[start:start + batch_size]

        encoded = tokenizer(
            batch,
            [hypothesis] * len(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            logits = model(**encoded).logits

        for row in logits.softmax(dim=-1).tolist():
            distributions.append(
                {
                    id2label.get(index, str(index)): float(probability)
                    for index, probability in enumerate(row)
                }
            )

    return distributions


def strongest_stance(distributions, min_score=MIN_STANCE_SCORE):
    """
    Reduce a document's per-passage distributions to one verdict.

    Returns (stance, score, index): the highest-scoring passage whose top
    label is entailment or contradiction, or neutral with the passage that
    was most confidently neutral when no passage takes a side.
    """
    if not distributions:
        return "neutral", 0.0, None

    winner = None

    for index, distribution in enumerate(distributions):
        if not distribution:
            continue

        label = max(distribution, key=distribution.get)
        stance = STANCE_BY_LABEL.get(label)

        if stance is None:
            continue

        score = float(distribution[label])

        if score < min_score:
            continue

        if winner is None or score > winner[1]:
            winner = (stance, round(score, 4), index)

    if winner is not None:
        return winner

    neutral_index, neutral_score = max(
        enumerate(float(d.get("neutral", 0.0)) for d in distributions),
        key=lambda pair: pair[1],
    )

    return "neutral", round(neutral_score, 4), neutral_index
