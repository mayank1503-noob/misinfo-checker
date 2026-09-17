"""
Data model for extracted claims.

A ClaimSet is the hand-off between claim extraction (this stage) and the
evidence graph that will be built on top of it later. Every Claim and Entity
carries a stable id so later stages can attach evidence nodes and edges
without re-parsing the packet.
"""

import hashlib
from typing import List, Optional

from pydantic import BaseModel, Field


CLAIM_TYPES = (
    "statistic",     # numbers, percentages, counts
    "money",         # amounts, cashback, prizes, fees
    "event",         # something happened / will happen at a time or place
    "attribution",   # X said / announced / confirmed Y
    "policy",        # government / official rule or scheme
    "health",        # cure, disease, vaccine, medicine
    "prediction",    # will / going to / about to
    "causal",        # X causes / leads to Y
    "chain_offer",   # forward-to-get, free gift, lucky draw style
    "generic",       # factual sentence with no stronger signal
)

SOURCE_FIELDS = (
    "text",          # packet["text"] (raw text, article body, caption + transcript)
    "ocr",           # images[i]["ocr_text"]
    "caption",       # images[i]["description"] (model-generated caption)
)


def stable_id(prefix, *parts):
    digest = hashlib.sha1(
        "|".join(str(p) for p in parts).encode("utf-8")
    ).hexdigest()[:12]

    return f"{prefix}_{digest}"


class Entity(BaseModel):
    id: str
    text: str
    label: str            # PERSON, ORG, GPE, MONEY, PERCENT, NUMBER, DATE, URL, PHONE, UPI, HASHTAG
    normalized: Optional[str] = None
    start: int
    end: int


class TimeRef(BaseModel):
    text: str
    kind: str             # absolute | relative | recurring
    normalized: Optional[str] = None   # ISO date if resolvable, else None
    start: int
    end: int


class SourceSpan(BaseModel):
    field: str            # one of SOURCE_FIELDS
    image_index: Optional[int] = None
    frame_time: Optional[float] = None
    start: int            # char offsets inside that field's text
    end: int


class Claim(BaseModel):
    id: str
    text: str
    normalized_text: str
    claim_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    check_worthy: bool

    entities: List[Entity] = []
    time_refs: List[TimeRef] = []
    numbers: List[str] = []
    attributed_to: Optional[str] = None

    source: SourceSpan
    language: str = "en"  # en | hi | hinglish | unknown
    signals: dict = {}    # feature flags that drove the score (for debugging / eval)

    # slots to be filled by later stages; kept here so the shape is stable
    evidence_ids: List[str] = []
    verdict: Optional[str] = None


class ClaimSet(BaseModel):
    packet_id: str
    input_type: str
    source_date: Optional[str] = None
    backend: str = "heuristic"   # heuristic | transformer | ollama
    claims: List[Claim] = []
    dropped: List[dict] = []   # sentences rejected as not check-worthy, with reason

    def check_worthy(self):
        return [c for c in self.claims if c.check_worthy]

    def to_graph_seed(self):
        """
        Produce the node/edge skeleton the evidence graph will grow from.

        Nodes: packet, claims, entities.
        Edges: packet -HAS_CLAIM-> claim
               claim  -MENTIONS->  entity
               claim  -SHARES_ENTITY-> claim   (same normalized entity)
        Evidence nodes and SUPPORTS / REFUTES edges are added later.
        """
        nodes = [
            {
                "id": self.packet_id,
                "kind": "packet",
                "input_type": self.input_type,
                "source_date": self.source_date,
            }
        ]
        edges = []

        entity_index = {}

        for claim in self.claims:
            nodes.append(
                {
                    "id": claim.id,
                    "kind": "claim",
                    "text": claim.text,
                    "claim_type": claim.claim_type,
                    "confidence": claim.confidence,
                    "check_worthy": claim.check_worthy,
                    "source_field": claim.source.field,
                }
            )

            edges.append(
                {
                    "from": self.packet_id,
                    "to": claim.id,
                    "type": "HAS_CLAIM",
                }
            )

            for entity in claim.entities:
                key = entity.id

                if key not in entity_index:
                    entity_index[key] = []

                    nodes.append(
                        {
                            "id": entity.id,
                            "kind": "entity",
                            "label": entity.label,
                            "text": entity.text,
                            "normalized": entity.normalized,
                        }
                    )

                edges.append(
                    {
                        "from": claim.id,
                        "to": entity.id,
                        "type": "MENTIONS",
                    }
                )

                for other in entity_index[key]:
                    if other != claim.id:
                        edges.append(
                            {
                                "from": other,
                                "to": claim.id,
                                "type": "SHARES_ENTITY",
                                "via": entity.id,
                            }
                        )

                if claim.id not in entity_index[key]:
                    entity_index[key].append(claim.id)

        return {"nodes": nodes, "edges": edges}
