"""
The evidence graph (Stage 3A) — the data layer only, no retrieval.

    from backend.claims import extract_claims
    from backend.evidence import Evidence, EvidenceGraph, EvidenceSource

    claims = extract_claims(packet)
    graph  = EvidenceGraph.from_claimset(claims)      # packet/claim/entity skeleton

    graph.add_evidence(claim_id, Evidence(...))       # grows an evidence node
    graph.get_claim_evidence(claim_id)                # -> [Evidence, ...]
    graph.to_dict()                                   # JSON-serialisable

The skeleton is *not* rebuilt here: `ClaimSet.to_graph_seed()` already
emits packet / claim / entity nodes with deterministic ids, and this
module consumes that output as-is. Everything added on top is keyed by
those same ids, so the graph for a given input is byte-identical run to
run — the test suite leans on that.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .schema import (
    EDGE_TYPES,
    RELATION_EDGES,
    RELATIONS,
    Edge,
    Evidence,
    EvidenceRelation,
    Node,
)


class EvidenceGraph(BaseModel):
    """
    Nodes and edges, plus typed views of the three node kinds that later
    stages actually ask for (claims, entities, evidence).

    `nodes` is the single source of truth and holds every node kind;
    `claims` / `entities` / `evidence_nodes` are filtered views of it, so
    they can never drift out of sync. `evidence` keeps the full Evidence
    objects (scores, source, match) that the flat nodes only mirror.
    """

    packet_id: Optional[str] = None
    input_type: Optional[str] = None
    source_date: Optional[str] = None

    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)
    evidence: Dict[str, Evidence] = Field(default_factory=dict)
    relations: List[EvidenceRelation] = Field(default_factory=list)

    # --- construction -------------------------------------------------

    @classmethod
    def from_claimset(cls, claimset):
        """Build the skeleton from a Stage 2 ClaimSet via to_graph_seed()."""
        graph = cls.from_seed(claimset.to_graph_seed())
        graph.input_type = claimset.input_type
        graph.source_date = claimset.source_date

        return graph

    # kept as an alias: `from_claim_set` reads better next to ClaimSet
    from_claim_set = from_claimset

    @classmethod
    def from_seed(cls, seed):
        """Build from the raw {"nodes": [...], "edges": [...]} skeleton."""
        graph = cls()

        for node in seed.get("nodes", []):
            graph.add_node(Node.from_dict(node))

        for edge in seed.get("edges", []):
            graph.add_edge(Edge.from_dict(edge))

        packet = graph.nodes_of_kind("packet")

        if packet:
            graph.packet_id = packet[0].id
            graph.input_type = packet[0].get("input_type")
            graph.source_date = packet[0].get("source_date")

        return graph

    # --- primitives ---------------------------------------------------

    def add_node(self, node):
        """Insert a node, or merge attrs into the one already there."""
        existing = self.nodes.get(node.id)

        if existing is None:
            self.nodes[node.id] = node
            return node

        existing.attrs.update({k: v for k, v in node.attrs.items() if v is not None})

        return existing

    def add_edge(self, edge):
        """Insert an edge unless the same (from, to, type, via) is present."""
        for existing in self.edges:
            if existing.key == edge.key:
                existing.attrs.update(edge.attrs)
                return existing

        self.edges.append(edge)

        return edge

    def has_node(self, node_id):
        return node_id in self.nodes

    def nodes_of_kind(self, kind):
        return [node for node in self.nodes.values() if node.kind == kind]

    def edges_of_type(self, edge_type):
        return [edge for edge in self.edges if edge.type == edge_type]

    # --- typed views --------------------------------------------------

    @property
    def claims(self):
        return {node.id: node for node in self.nodes_of_kind("claim")}

    @property
    def entities(self):
        return {node.id: node for node in self.nodes_of_kind("entity")}

    @property
    def evidence_nodes(self):
        return {node.id: node for node in self.nodes_of_kind("evidence")}

    # --- growing the graph --------------------------------------------

    def add_evidence(self, claim_id, evidence, relation=None):
        """
        Attach a piece of evidence to a claim.

        Adds the evidence node (once — ids are content-derived, so the
        same item retrieved for two claims is one node), the
        claim -HAS_EVIDENCE-> evidence edge, and the relation edge
        carrying the stance. `relation` overrides `evidence.relation`.

        Returns the stored Evidence.
        """
        self._require_claim(claim_id)

        stored = self.evidence.get(evidence.id)

        if stored is None:
            stored = evidence
            self.evidence[evidence.id] = stored

        self.add_node(stored.to_node())

        self.add_edge(
            Edge(
                src=claim_id,
                dst=stored.id,
                type="HAS_EVIDENCE",
                attrs={"evidence_type": stored.evidence_type},
            )
        )

        self.add_relation(
            claim_id,
            stored.id,
            relation or stored.relation,
            relevance=stored.relevance,
            credibility=stored.credibility,
        )

        return stored

    def add_relation(self, claim_id, evidence_id, relation, relevance=None,
                     credibility=None, note=None):
        """
        Record how a piece of evidence bears on a claim: `supports`,
        `refutes` or `uncertain`.

        Re-stating a relation replaces the previous one (a re-scored item
        must not leave a stale SUPPORTS edge behind), and the HAS_EVIDENCE
        edge is ensured so a relation can be set without add_evidence.
        """
        if relation not in RELATIONS:
            raise ValueError(
                f"unknown relation {relation!r}; expected one of {RELATIONS}"
            )

        self._require_claim(claim_id)

        if evidence_id not in self.nodes:
            raise KeyError(f"unknown evidence node {evidence_id!r}")

        item = self.evidence.get(evidence_id)

        if relevance is None:
            relevance = item.relevance if item else 0.5

        if credibility is None:
            credibility = item.credibility if item else 0.5

        record = EvidenceRelation(
            claim_id=claim_id,
            evidence_id=evidence_id,
            relation=relation,
            relevance=relevance,
            credibility=credibility,
            note=note,
        )

        # drop any earlier stance for this pair, edge and record alike
        stale = set(RELATION_EDGES.values())
        self.edges = [
            edge
            for edge in self.edges
            if not (
                edge.type in stale
                and edge.src == evidence_id
                and edge.dst == claim_id
            )
        ]
        self.relations = [r for r in self.relations if r.key != record.key]

        self.relations.append(record)

        if item is not None and item.relation != relation:
            item.relation = relation
            node = self.nodes.get(evidence_id)

            if node is not None:
                node.attrs["relation"] = relation

        self.add_edge(
            Edge(
                src=claim_id,
                dst=evidence_id,
                type="HAS_EVIDENCE",
                attrs={"evidence_type": item.evidence_type if item else None},
            )
        )

        self.add_edge(
            Edge(
                src=evidence_id,
                dst=claim_id,
                type=record.edge_type,
                attrs={
                    "relation": relation,
                    "relevance": relevance,
                    "credibility": credibility,
                    **({"note": note} if note else {}),
                },
            )
        )

        return record

    # --- reading it back ----------------------------------------------

    def get_claim_evidence(self, claim_id, relation=None):
        """
        Evidence attached to a claim, strongest first (relevance ×
        credibility, ties broken by id so the order is deterministic).
        """
        self._require_claim(claim_id)

        if relation is not None and relation not in RELATIONS:
            raise ValueError(
                f"unknown relation {relation!r}; expected one of {RELATIONS}"
            )

        by_pair = {r.key: r for r in self.relations}
        found = []

        for edge in self.edges_of_type("HAS_EVIDENCE"):
            if edge.src != claim_id:
                continue

            item = self.evidence.get(edge.dst)

            if item is None:
                continue

            record = by_pair.get((claim_id, edge.dst))
            stance = record.relation if record else item.relation

            if relation is not None and stance != relation:
                continue

            found.append(item)

        found.sort(key=lambda e: (-e.weight, e.id))

        return found

    def get_claim_relations(self, claim_id):
        """The EvidenceRelation records for one claim, in insertion order."""
        self._require_claim(claim_id)

        return [r for r in self.relations if r.claim_id == claim_id]

    def get_entities(self, claim_id):
        """Entity nodes a claim MENTIONS."""
        self._require_claim(claim_id)

        return [
            self.nodes[edge.dst]
            for edge in self.edges_of_type("MENTIONS")
            if edge.src == claim_id and edge.dst in self.nodes
        ]

    def related_claims(self, claim_id):
        """Claim ids joined to this one by SHARES_ENTITY, either direction."""
        self._require_claim(claim_id)

        related = []

        for edge in self.edges_of_type("SHARES_ENTITY"):
            other = None

            if edge.src == claim_id:
                other = edge.dst
            elif edge.dst == claim_id:
                other = edge.src

            if other and other not in related:
                related.append(other)

        return related

    def neighbors(self, node_id, edge_type=None):
        """(edge, node) pairs leaving `node_id`, optionally of one type."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node {node_id!r}")

        out = []

        for edge in self.edges:
            if edge.src != node_id:
                continue

            if edge_type is not None and edge.type != edge_type:
                continue

            target = self.nodes.get(edge.dst)

            if target is not None:
                out.append((edge, target))

        return out

    def summary(self):
        """Node counts by kind and edge counts by type — for logs and tests."""
        kinds = {}
        types = {}

        for node in self.nodes.values():
            kinds[node.kind] = kinds.get(node.kind, 0) + 1

        for edge in self.edges:
            types[edge.type] = types.get(edge.type, 0) + 1

        return {
            "packet_id": self.packet_id,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "by_kind": kinds,
            "by_type": {t: types[t] for t in EDGE_TYPES if t in types},
        }

    def apply_to_claimset(self, claimset):
        """
        Write the evidence ids back onto the ClaimSet's claims, so a
        serialised ClaimSet carries its evidence without the graph.
        Returns the same ClaimSet (mutated in place).
        """
        for claim in claimset.claims:
            if claim.id in self.nodes:
                claim.evidence_ids = [e.id for e in self.get_claim_evidence(claim.id)]

        return claimset

    # --- serialisation ------------------------------------------------

    def to_dict(self):
        """
        Flat, JSON-serialisable form. `nodes` / `edges` are the graph;
        `claims` / `entities` / `evidence` are the views a consumer of the
        API would otherwise have to filter out for itself.
        """
        return {
            "packet_id": self.packet_id,
            "input_type": self.input_type,
            "source_date": self.source_date,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "claims": [node.to_dict() for node in self.nodes_of_kind("claim")],
            "entities": [node.to_dict() for node in self.nodes_of_kind("entity")],
            "evidence": [
                item.model_dump(mode="json") for item in self.evidence.values()
            ],
            "relations": [r.model_dump(mode="json") for r in self.relations],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data):
        """Inverse of to_dict(); the views are rebuilt from `nodes`."""
        graph = cls.from_seed(
            {"nodes": data.get("nodes", []), "edges": data.get("edges", [])}
        )

        graph.packet_id = data.get("packet_id", graph.packet_id)
        graph.input_type = data.get("input_type", graph.input_type)
        graph.source_date = data.get("source_date", graph.source_date)

        for item in data.get("evidence", []):
            evidence = Evidence(**item)
            graph.evidence[evidence.id] = evidence

        for record in data.get("relations", []):
            graph.relations.append(EvidenceRelation(**record))

        return graph

    def merge(self, other):
        """Fold another graph into this one (same packet or a related one)."""
        for node in other.nodes.values():
            self.add_node(node)

        for edge in other.edges:
            self.add_edge(edge)

        for evidence_id, item in other.evidence.items():
            self.evidence.setdefault(evidence_id, item)

        known = {r.key for r in self.relations}

        for record in other.relations:
            if record.key not in known:
                self.relations.append(record)
                known.add(record.key)

        return self

    # --- internals ----------------------------------------------------

    def _require_claim(self, claim_id):
        node = self.nodes.get(claim_id)

        if node is None:
            raise KeyError(f"unknown claim {claim_id!r}")

        if node.kind != "claim":
            raise ValueError(f"node {claim_id!r} is a {node.kind}, not a claim")

        return node
