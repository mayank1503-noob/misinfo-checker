"""
Transformer-backed signals for claim extraction.

Two models, both loaded lazily and cached like backend/analyzers/models.py:

    NER        token-classification  -> PERSON / ORG / GPE spans
    zero-shot  NLI                   -> check-worthiness + claim type

Everything here returns plain Python values; the heuristic layer decides
how to blend them. If `transformers` is not installed, `available()` is
False and the extractor silently falls back to regex-only mode.

Override the models with env vars:

    CLAIM_NER_MODEL   default dslim/bert-base-NER
                      (try ai4bharat/IndicNER for Hindi / Hinglish)
    CLAIM_ZSC_MODEL   default MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
                      (multilingual, covers Hindi)
"""

import os
from functools import lru_cache

from .schema import CLAIM_TYPES, Entity, stable_id


DEFAULT_NER_MODEL = "dslim/bert-base-NER"
DEFAULT_ZSC_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

NER_LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "GPE",
    "GPE": "GPE",
    "MISC": "PROPER",
}

NER_MIN_SCORE = 0.60

# Hypotheses for the check-worthiness decision.
WORTHY_LABELS = {
    "claim": "a factual claim that can be verified",
    "chat": "personal conversation, opinion, or greeting",
}

# Natural-language descriptions of CLAIM_TYPES for zero-shot typing.
TYPE_LABELS = {
    "statistic": "a statistic or number",
    "money": "an amount of money, a fee, or a prize",
    "event": "a news event that happened",
    "attribution": "a statement attributed to a person or organisation",
    "policy": "a government policy, law, or scheme",
    "health": "a health, medical, or disease claim",
    "prediction": "a prediction about the future",
    "causal": "one thing causes another",
    "chain_offer": "a forward-this-message offer, giveaway, or scam",
    "generic": "a general factual statement",
}

TYPE_MIN_MARGIN = 0.10


def available():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False

    return True


@lru_cache
def ner_pipeline():
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=os.getenv("CLAIM_NER_MODEL", DEFAULT_NER_MODEL),
        aggregation_strategy="first",
    )


@lru_cache
def zsc_pipeline():
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=os.getenv("CLAIM_ZSC_MODEL", DEFAULT_ZSC_MODEL),
    )


def ner_entities(sentence):
    """
    Model NER -> list[Entity] with PERSON / ORG / GPE / PROPER labels.
    """
    entities = []

    for hit in ner_pipeline()(sentence):
        label = NER_LABEL_MAP.get(hit["entity_group"])

        if not label or hit["score"] < NER_MIN_SCORE:
            continue

        text = sentence[hit["start"]:hit["end"]].strip()

        if len(text) < 2:
            continue

        entities.append(
            Entity(
                id=stable_id("ent", label, text.lower()),
                text=text,
                label=label,
                normalized=text.lower(),
                start=hit["start"],
                end=hit["start"] + len(text),
            )
        )

    return entities


def worthiness(sentence):
    """
    Probability (0..1) that the sentence is a verifiable factual claim.
    """
    result = zsc_pipeline()(
        sentence,
        candidate_labels=list(WORTHY_LABELS.values()),
        hypothesis_template="This text is {}.",
    )

    scores = dict(zip(result["labels"], result["scores"]))

    return float(scores.get(WORTHY_LABELS["claim"], 0.0))


def claim_type(sentence):
    """
    Return (type, score, margin). `type` is None when the model is not
    confident enough (top-2 margin below TYPE_MIN_MARGIN).
    """
    labels = list(TYPE_LABELS.values())

    result = zsc_pipeline()(
        sentence,
        candidate_labels=labels,
        hypothesis_template="This text is {}.",
    )

    inverse = {v: k for k, v in TYPE_LABELS.items()}

    ranked = list(zip(result["labels"], result["scores"]))
    top_label, top_score = ranked[0]
    margin = top_score - (ranked[1][1] if len(ranked) > 1 else 0.0)

    top = inverse[top_label]

    if top not in CLAIM_TYPES or margin < TYPE_MIN_MARGIN:
        return None, float(top_score), float(margin)

    return top, float(top_score), float(margin)
