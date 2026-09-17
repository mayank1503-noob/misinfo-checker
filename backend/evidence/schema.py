"""
Data model for the evidence graph (Stage 3A).

Stage 2 hands over a ClaimSet; `ClaimSet.to_graph_seed()` turns it into a
flat {"nodes": [...], "edges": [...]} skeleton of packet / claim / entity
nodes. This stage grows that skeleton: a retriever produces `Evidence`
objects, which become `evidence` nodes joined to claims by HAS_EVIDENCE
and SUPPORTS / REFUTES / UNCERTAIN_FOR edges.

Stage 3A is the data layer only — nothing here touches the network. The
retriever (3B) builds `Evidence` objects; the verdict stage (4) reads the
graph back out.

Nodes and edges stay flat dicts on the wire (the same shape the seed
uses), so a graph round-trips through JSON without a schema migration,
while the Node / Edge models give the graph code typed access to `id`,
`kind`, `type` and the free-form rest (`attrs`).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..claims.schema import stable_id


NODE_KINDS = (
    "packet",     # the ingested media packet (stage 1)
    "claim",      # an extracted claim (stage 2)
    "entity",     # a person / org / amount / date mentioned by a claim
    "evidence",   # a retrieved document or media match (stage 3)
)

EDGE_TYPES = (
    "HAS_CLAIM",       # packet   -> claim
    "MENTIONS",        # claim    -> entity
    "SHARES_ENTITY",   # claim    -> claim      (via a common entity)
    "HAS_EVIDENCE",    # claim    -> evidence   (retrieved for this claim)
    "SUPPORTS",        # evidence -> claim
    "REFUTES",         # evidence -> claim
    "UNCERTAIN_FOR",   # evidence -> claim      (relevant, takes no side)
)

EVIDENCE_TYPES = (
    "fact_check",     # Google Fact Check Tools, IFCN publishers
    "news",           # a news article covering the claim
    "rumor_index",    # local index of known scams / recycled rumours
    "reverse_image",  # DINOv2 embedding match against a known image
    "date_check",     # source_date / exif_date vs. the claimed date
    "official",       # government / company statement: RBI, PIB, ...
    "other",
)

RELATIONS = ("supports", "refutes", "uncertain")

# relation -> the evidence->claim edge that carries it
RELATION_EDGES = {
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "uncertain": "UNCERTAIN_FOR",
}


def _domain(url):
    """Host of a URL, without scheme or `www.` — used as the publisher."""
    if not url:
        return None

    rest = url.split("://", 1)[-1]
    host = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    host = host.split("@")[-1].split(":")[0].lower()

    if host.startswith("www."):
        host = host[4:]

    return host or None


class EvidenceSource(BaseModel):
    """
    Where a piece of evidence came from.

    `published` is the date the source itself states (often missing);
    `retrieved` is when we fetched it. Only `published` matters for
    recycled-media checks, so the two are kept apart rather than merged.
    """

    url: Optional[str] = None
    title: Optional[str] = None
    publisher: Optional[str] = None     # falls back to the URL's domain
    domain: Optional[str] = None        # derived from `url`
    published: Optional[str] = None     # ISO date, if the source states one
    retrieved: Optional[str] = None     # ISO date we fetched it (never in the id)
    rating: Optional[str] = None        # publisher's own verdict text, if any

    def model_post_init(self, _context):
        if self.domain is None:
            self.domain = _domain(self.url)

        if self.publisher is None:
            self.publisher = self.domain

    @property
    def fingerprint(self):
        """The identifying part of a source — never the retrieval date."""
        return self.url or f"{self.publisher or ''}|{self.title or ''}"


def evidence_id(evidence_type, source=None, text=""):
    """
    Deterministic id for a piece of evidence: the URL identifies it when
    there is one (the same fact-check retrieved twice is one node), else
    the publisher + title + text. The retrieval date is deliberately
    excluded so re-running retrieval does not fork the graph.
    """
    fingerprint = source.fingerprint if source is not None else ""

    return stable_id("ev", evidence_type, fingerprint, (text or "")[:200])


class Evidence(BaseModel):
    """
    One retrieved item that bears on a claim.

    `relevance` is how well the item matches the claim; `credibility` is
    how much the source is worth trusting. They stay separate so stage 4
    can weigh "a spot-on match from a random blog" differently from "a
    loose match from PIB". `match` records *why* the retriever returned
    this item (query string, cosine distance, matched pattern) so a
    verdict can be explained rather than asserted.
    """

    id: str = ""
    evidence_type: str = "other"            # one of EVIDENCE_TYPES
    text: str = ""                          # snippet / quote bearing on the claim
    source: EvidenceSource = Field(default_factory=EvidenceSource)

    relation: str = "uncertain"             # one of RELATIONS
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    credibility: float = Field(default=0.5, ge=0.0, le=1.0)

    match: dict = {}                        # why this was retrieved
    meta: dict = {}

    def model_post_init(self, _context):
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(
                f"unknown evidence_type {self.evidence_type!r}; "
                f"expected one of {EVIDENCE_TYPES}"
            )

        if self.relation not in RELATIONS:
            raise ValueError(
                f"unknown relation {self.relation!r}; expected one of {RELATIONS}"
            )

        if not self.id:
            self.id = evidence_id(self.evidence_type, self.source, self.text)

    @property
    def weight(self):
        """Relevance tempered by credibility — a convenience for stage 4."""
        return round(self.relevance * self.credibility, 4)

    def to_node(self):
        """The `evidence` graph node for this item."""
        return Node(
            id=self.id,
            kind="evidence",
            attrs={
                "evidence_type": self.evidence_type,
                "text": self.text,
                "relation": self.relation,
                "relevance": self.relevance,
                "credibility": self.credibility,
                "url": self.source.url,
                "title": self.source.title,
                "publisher": self.source.publisher,
                "domain": self.source.domain,
                "published": self.source.published,
                "retrieved": self.source.retrieved,
                "rating": self.source.rating,
                "match": self.match,
                "meta": self.meta,
            },
        )


class EvidenceRelation(BaseModel):
    """A claim <-> evidence verdict link, kept alongside the edge it produced."""

    claim_id: str
    evidence_id: str
    relation: str                       # one of RELATIONS
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    note: Optional[str] = None

    def model_post_init(self, _context):
        if self.relation not in RELATIONS:
            raise ValueError(
                f"unknown relation {self.relation!r}; expected one of {RELATIONS}"
            )

    @property
    def edge_type(self):
        return RELATION_EDGES[self.relation]

    @property
    def key(self):
        return (self.claim_id, self.evidence_id)


class Node(BaseModel):
    id: str
    kind: str          # one of NODE_KINDS
    attrs: dict = {}   # everything else, flattened on the wire

    def get(self, key, default=None):
        return self.attrs.get(key, default)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, **self.attrs}

    @classmethod
    def from_dict(cls, data):
        attrs = dict(data)
        node_id = attrs.pop("id")
        kind = attrs.pop("kind")

        return cls(id=node_id, kind=kind, attrs=attrs)


class Edge(BaseModel):
    """
    A directed edge. `src` / `dst` serialise as "from" / "to" to match the
    seed produced by ClaimSet.to_graph_seed().
    """

    model_config = ConfigDict(populate_by_name=True)

    src: str = Field(alias="from")
    dst: str = Field(alias="to")
    type: str          # one of EDGE_TYPES
    attrs: dict = {}   # via, relation, relevance, ...

    @property
    def key(self):
        """Identity of an edge: the same triple + `via` is one edge."""
        return (self.src, self.dst, self.type, self.attrs.get("via"))

    def get(self, key, default=None):
        return self.attrs.get(key, default)

    def to_dict(self):
        return {"from": self.src, "to": self.dst, "type": self.type, **self.attrs}

    @classmethod
    def from_dict(cls, data):
        attrs = dict(data)
        src = attrs.pop("from", None) or attrs.pop("src")
        dst = attrs.pop("to", None) or attrs.pop("dst")
        edge_type = attrs.pop("type")

        return cls(src=src, dst=dst, type=edge_type, attrs=attrs)
