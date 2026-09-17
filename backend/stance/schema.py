"""
Data model for stance detection (Stage 4).

Stage 3 hands over retrieved items; this stage decides what each one says
*about* the claim it was retrieved for: does it support it, refute it, or
take no side. The input is deliberately flat — `EvidenceInput` is whatever
a retriever can always produce (a title, a snippet, maybe the full page,
maybe the publisher's own rating) rather than the richer graph model, so
the stance layer stays a pure function of text and can be unit-tested
without building a graph.

One `StanceResult` comes back per evidence item, carrying not just the
stance but *why*: the passage that decided it, how relevant that passage
was to the claim, and which method produced the answer. A caller that
wants to weigh stances uses `weighted_score`, which tempers the model's
confidence by the retriever's own weight for the item.
"""

from typing import Optional

from pydantic import BaseModel, Field


STANCES = (
    "support",    # the evidence asserts the claim
    "refute",     # the evidence contradicts the claim
    "neutral",    # relevant but takes no side, off-topic, or undecidable
)

METHODS = (
    "nli",          # decided by the entailment model
    "rating",       # decided by the publisher's own verdict (overrides nli)
    "off_topic",    # no passage cleared the relevance cutoff
    "unavailable",  # a model was missing or failed — failed soft to neutral
)


class EvidenceInput(BaseModel):
    """
    One retrieved item, as flat text plus what the source says about it.

    `title` / `snippet` / `text` are all optional and often overlap (a
    snippet is usually a prefix of the text); `body` joins whichever of
    them carry new information. `rating` is the publisher's own verdict
    string when the item is a fact-check — it beats anything the model
    reads out of the text. `weight` is the retriever's confidence in the
    source, passed through untouched for the verdict stage to use.
    """

    id: str
    claim_id: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    text: Optional[str] = None
    rating: Optional[str] = None
    weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @property
    def body(self):
        """
        Title, snippet and text joined by newlines, dropping any part that
        is already contained in a later, fuller one. Newlines keep a title
        with no full stop from running into the first sentence.
        """
        parts = [p for p in (self.title, self.snippet, self.text) if (p or "").strip()]
        parts = [p.strip() for p in parts]

        kept = []

        for index, part in enumerate(parts):
            if any(part in longer and part != longer for longer in parts[index + 1:]):
                continue

            if part in kept:
                continue

            kept.append(part)

        return "\n".join(kept)


class StanceResult(BaseModel):
    """
    What one evidence item says about one claim.

    `score` is confidence in the stance (an entailment probability, or 1.0
    for an explicit publisher rating), `relevance` is how close the best
    passage was to the claim, and `best_passage` is that passage — so a
    verdict can quote the sentence it turned on.

    `misleading` marks the trap this stage exists to avoid: a fact-check
    whose text reads as *supporting* the claim because it quotes the false
    claim verbatim, while its own rating refutes it. Downstream must never
    count such text as support.
    """

    evidence_id: str
    claim_id: str
    stance: str = "neutral"                                  # one of STANCES
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    best_passage: Optional[str] = None
    method: str = "unavailable"                              # one of METHODS
    misleading: bool = False
    weight: float = Field(default=1.0, ge=0.0, le=1.0)       # passed through from the input
    note: Optional[str] = None                               # why, for debugging / eval

    def model_post_init(self, _context):
        if self.stance not in STANCES:
            raise ValueError(
                f"unknown stance {self.stance!r}; expected one of {STANCES}"
            )

        if self.method not in METHODS:
            raise ValueError(
                f"unknown method {self.method!r}; expected one of {METHODS}"
            )

    @property
    def weighted_score(self):
        """Confidence tempered by the retriever's weight for the source."""
        return round(self.score * self.weight, 4)
