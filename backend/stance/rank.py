"""
Find the passages of an evidence document that are actually about the claim.

Running entailment over every passage of every retrieved article is both
slow and noisy — most passages have nothing to do with the claim, and an
NLI model asked about an unrelated premise still returns *some* label.
So passages are first ranked by embedding similarity to the claim, the
top few go to the model, and a document whose best passage is below
`RELEVANCE_CUTOFF` is treated as off-topic and never scored at all.

The embedder is multilingual (paraphrase-multilingual-MiniLM-L12-v2), so a
Hindi claim matches Hindi evidence, and an English claim matches a Hindi
article about it. It is loaded lazily and cached, like the models in
backend/analyzers/models.py; `available()` is False when
sentence-transformers is not installed and the caller fails soft.

Override with:

    STANCE_EMBED_MODEL   default sentence-transformers/
                                 paraphrase-multilingual-MiniLM-L12-v2
"""

import math
import os
from functools import lru_cache
from typing import List, NamedTuple


DEFAULT_EMBED_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3
RELEVANCE_CUTOFF = 0.3


class ScoredPassage(NamedTuple):
    text: str
    score: float


class Ranking(NamedTuple):
    """
    `passages` are the top-k that cleared the cutoff (possibly empty);
    `best` is the top raw similarity either way, so an off-topic document
    can still report how far off it was.
    """

    passages: List[ScoredPassage]
    best: float


def available():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False

    return True


@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    from ..torch_runtime import prepare

    return prepare(
        SentenceTransformer(os.getenv("STANCE_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    )


def embed(texts):
    """Encode a batch of strings -> list of plain float vectors."""
    texts = list(texts)

    if not texts:
        return []

    from ..torch_runtime import inference

    # Stage 3b calls this from inside two nested thread pools, so this is
    # the forward pass that used to run several times over. See
    # backend/torch_runtime.py.
    with inference():
        vectors = embedder().encode(texts)

    return [[float(value) for value in vector] for vector in vectors]


def cosine(left, right):
    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))

    if norm == 0.0:
        return 0.0

    return dot / norm


def rank_batch(claim_text, passage_groups, top_k=TOP_K, cutoff=RELEVANCE_CUTOFF):
    """
    Rank several documents' passages against one claim in a single encode
    call. `passage_groups` is a list of passage lists (one per document);
    returns one Ranking per group, in the same order.
    """
    groups = [list(group) for group in passage_groups]
    flat = [passage for group in groups for passage in group]

    if not flat:
        return [Ranking([], 0.0) for _ in groups]

    vectors = embed([claim_text] + flat)
    claim_vector, passage_vectors = vectors[0], vectors[1:]

    rankings = []
    offset = 0

    for group in groups:
        scored = [
            ScoredPassage(
                passage,
                round(max(0.0, cosine(claim_vector, passage_vectors[offset + index])), 4),
            )
            for index, passage in enumerate(group)
        ]

        offset += len(group)

        scored.sort(key=lambda item: item.score, reverse=True)

        rankings.append(
            Ranking(
                [item for item in scored if item.score >= cutoff][:top_k],
                scored[0].score if scored else 0.0,
            )
        )

    return rankings


def rank_passages(claim_text, passages, top_k=TOP_K, cutoff=RELEVANCE_CUTOFF):
    """Rank one document's passages against a claim."""
    return rank_batch(claim_text, [passages], top_k=top_k, cutoff=cutoff)[0]
