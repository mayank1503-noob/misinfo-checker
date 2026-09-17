"""
The evidence graph itself (Stage 3a) — a NetworkX MultiDiGraph.

    from backend.claims import extract_claims
    from backend.graph import EvidenceGraph

    claims = extract_claims(packet)
    graph  = EvidenceGraph.from_claimset(claims, packet)

    graph.add_evidence(claim_id, candidate, stance="refutes")
    graph.stance_totals(claim_id)      # weighted support vs refute
    graph.summary(1200)                # what a bot reply can quote
    graph.export_html("graph.html")    # pyvis, for a demo

Why a multigraph: two nodes can be joined for more than one reason at
once — an image is both `DUPLICATE_OF` an older post and `FIRST_SEEN_ON`
the date of that post, and two claims can share three different entities.
Parallel edges are keyed by type (and by `via` for SHARES_ENTITY), which
is also what makes every `add_*` idempotent: re-running retrieval updates
edges in place instead of stacking copies of them.

The graph holds no embeddings. Vectors belong to the index that searches
them (`backend/evidence/retrievers/seed_index.py`,
`backend/images/local_index.py`); keeping them here would multiply the
size of a serialised graph by a thousand and put a float array into every
bot reply.
"""

import json
import logging
import os
from datetime import date, datetime

import networkx as nx

from .schema import (
    EDGE_TYPES,
    NODE_TYPES,
    STANCE_EDGES,
    STANCE_EDGE_TYPES,
    STANCE_FROM_EDGE,
    date_id,
    image_id,
    normalize_stance,
    source_id,
    verdict_id,
)


log = logging.getLogger(__name__)


# Attribute names that must never reach the graph: vectors are big,
# unreadable, and already live in the index that produced them.
_DROPPED_ATTRS = ("embedding", "embeddings", "vector", "vectors")

# A source with no credibility weight of its own is treated as an unknown
# blog rather than as trustworthy.
DEFAULT_SOURCE_WEIGHT = 0.5

# How far apart a picture's first appearance and its claimed date have to
# be before it is worth flagging as recycled.
DATE_MISMATCH_DAYS = 30


def _clean_attrs(attrs):
    """Drop embeddings and Nones, and make values JSON-safe."""
    cleaned = {}

    for key, value in (attrs or {}).items():
        if key in _DROPPED_ATTRS or value is None:
            continue

        if isinstance(value, (datetime, date)):
            value = value.isoformat()

        cleaned[key] = value

    return cleaned


def _as_dict(item):
    """Accept a pydantic model, a plain dict, or a simple object."""
    if item is None:
        return {}

    if isinstance(item, dict):
        return dict(item)

    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")

    if hasattr(item, "_asdict"):
        return dict(item._asdict())

    return dict(vars(item))


def _iso(value):
    """Normalise whatever a source called a date into `YYYY-MM-DD`, or None."""
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip()

    if not text:
        return None

    # ISO timestamps, "2024-06-08T10:00:00Z" -> "2024-06-08"
    head = text.replace("/", "-")[:10]

    try:
        return datetime.strptime(head, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass

    for pattern in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(head[:len(pattern) + 2], pattern).date().isoformat()
        except ValueError:
            continue

    return None


def _days_between(left, right):
    try:
        return abs(
            (datetime.strptime(left, "%Y-%m-%d") - datetime.strptime(right, "%Y-%m-%d")).days
        )
    except (TypeError, ValueError):
        return None


class EvidenceGraph:
    """
    Nodes and edges for one packet, plus the queries stages 4-6 ask of it.

    Construction is normally `from_claimset`, which consumes stage 2's
    `to_graph_seed()` rather than re-deriving claims and entities: stage 2
    stays the single owner of those ids.
    """

    def __init__(self, packet_id=None, input_type=None, source_date=None):
        self.g = nx.MultiDiGraph()
        self.packet_id = packet_id
        self.input_type = input_type
        self.source_date = _iso(source_date)

        if packet_id:
            self.add_node(
                packet_id,
                "packet",
                input_type=input_type,
                source_date=self.source_date,
            )

    # --- primitives -------------------------------------------------------

    def add_node(self, node_id, kind, **attrs):
        """
        Insert a node, or merge attributes into the one already there.

        Merging (rather than replacing) is what lets stages run in any
        order: stage 5 can flag an image before or after stage 3b has
        attached evidence to the claim it came from.
        """
        if kind not in NODE_TYPES:
            raise ValueError(f"unknown node type {kind!r}; expected one of {NODE_TYPES}")

        attrs = _clean_attrs(attrs)

        if self.g.has_node(node_id):
            existing = self.g.nodes[node_id]

            if existing.get("kind") != kind:
                raise ValueError(
                    f"node {node_id!r} is already a {existing.get('kind')!r}, not a {kind!r}"
                )

            existing.update(attrs)
        else:
            self.g.add_node(node_id, kind=kind, **attrs)

        return node_id

    def add_edge(self, src, dst, edge_type, via=None, **attrs):
        """
        Insert an edge keyed by (type, via), updating it if it exists.

        Both endpoints must already be nodes — an edge to a node that was
        never added is a bug in the caller, and silently creating an
        attribute-less node would hide it until the verdict read garbage.
        """
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {edge_type!r}; expected one of {EDGE_TYPES}")

        for node_id in (src, dst):
            if not self.g.has_node(node_id):
                raise KeyError(f"unknown node {node_id!r}")

        key = edge_type if via is None else f"{edge_type}:{via}"
        attrs = _clean_attrs(attrs)

        if self.g.has_edge(src, dst, key):
            self.g.edges[src, dst, key].update(attrs)
        else:
            self.g.add_edge(src, dst, key=key, type=edge_type, via=via, **attrs)

        return key

    # --- construction -----------------------------------------------------

    @classmethod
    def from_claimset(cls, claimset, packet=None):
        """
        Build the skeleton from stage 2's `ClaimSet`, plus the images the
        packet carried.

        `to_graph_seed()` supplies packet / claim / entity nodes and the
        HAS_CLAIM / MENTIONS / SHARES_ENTITY edges. On top of that, every
        image (or video keyframe) in the packet becomes an `image` node
        carrying its EXIF date, frame time and AI-generated score, and any
        claim that was read *out of* an image — OCR text or a generated
        caption — is linked back to it with EXTRACTED_FROM. That link is
        what lets stage 5 say "the sentence that makes this claim is
        printed on a photo that has been circulating since 2019".
        """
        seed = claimset.to_graph_seed()

        graph = cls(
            packet_id=claimset.packet_id,
            input_type=claimset.input_type,
            source_date=claimset.source_date,
        )

        for node in seed.get("nodes", []):
            attrs = {k: v for k, v in node.items() if k not in ("id", "kind")}
            graph.add_node(node["id"], node["kind"], **attrs)

        for edge in seed.get("edges", []):
            attrs = {
                k: v for k, v in edge.items() if k not in ("from", "to", "type", "via")
            }
            graph.add_edge(
                edge["from"], edge["to"], edge["type"], via=edge.get("via"), **attrs
            )

        if graph.source_date:
            graph.add_date(graph.source_date)

        images = (packet or {}).get("images") or []
        image_ids = graph.add_images(images)

        for claim in claimset.claims:
            index = claim.source.image_index

            if index is None or index >= len(image_ids):
                continue

            graph.add_edge(
                claim.id,
                image_ids[index],
                "EXTRACTED_FROM",
                field=claim.source.field,
                frame_time=claim.source.frame_time,
            )

        return graph

    def add_images(self, images):
        """Add one `image` node per analysed image; returns their ids in order."""
        ids = []

        for index, image in enumerate(images or []):
            image = _as_dict(image)
            node_id = image_id(self.packet_id, image.get("path"), image.get("frame_time"))

            captured = _iso(image.get("exif_date"))

            self.add_node(
                node_id,
                "image",
                path=image.get("path"),
                index=index,
                frame_time=image.get("frame_time"),
                exif_date=captured,
                ai_generated_score=image.get("ai_generated_score"),
                description=image.get("description"),
                ocr_text=image.get("ocr_text"),
            )

            if self.packet_id:
                self.add_edge(self.packet_id, node_id, "HAS_IMAGE", index=index)

            if captured:
                self.add_date(captured)
                self.add_edge(node_id, date_id(captured), "CAPTURED_ON")

            ids.append(node_id)

        return ids

    def add_date(self, iso_date):
        """Add (or reuse) the shared node for one calendar date."""
        iso_date = _iso(iso_date)

        if not iso_date:
            return None

        return self.add_node(date_id(iso_date), "date", date=iso_date)

    # --- evidence ---------------------------------------------------------

    def add_evidence(self, claim_id, candidate, stance=None, score=None, relevance=None):
        """
        Attach one retrieved item to a claim.

        Writes the `evidence` node, the claim -HAS_EVIDENCE-> evidence
        edge, a shared `source` node for the publisher (FROM_SOURCE) and,
        when the item has a publication date, a shared `date` node
        (PUBLISHED_ON). `stance` is optional: stage 3b retrieves without
        judging, and stage 4 fills the stance in later via `set_stance`.

        Returns the evidence node id. Calling it twice with the same item
        updates that node rather than adding a second one.
        """
        self._require(claim_id, "claim")

        item = _as_dict(candidate)
        evidence_id = item.get("id")

        if not evidence_id:
            raise ValueError("evidence candidate has no id")

        published = _iso(item.get("published_date") or item.get("published"))
        domain = (item.get("domain") or item.get("publisher") or "unknown").lower()
        weight = item.get("source_weight")

        self.add_node(
            evidence_id,
            "evidence",
            source_type=item.get("source_type"),
            title=item.get("title"),
            url=item.get("url"),
            snippet=item.get("snippet"),
            text=item.get("text"),
            publisher=item.get("publisher"),
            domain=domain,
            rating=item.get("rating"),
            rating_raw=item.get("rating_raw"),
            query=item.get("query"),
            published_date=published,
            retrieved_at=item.get("retrieved_at"),
            source_weight=DEFAULT_SOURCE_WEIGHT if weight is None else float(weight),
            decisive=bool(item.get("decisive")),
            demo=bool(item.get("demo")),
            method=item.get("method"),
            similarity=item.get("similarity"),
        )

        self.add_edge(claim_id, evidence_id, "HAS_EVIDENCE")

        source_node = source_id(domain)
        self.add_node(
            source_node,
            "source",
            domain=domain,
            publisher=item.get("publisher") or domain,
            weight=DEFAULT_SOURCE_WEIGHT if weight is None else float(weight),
        )
        self.add_edge(evidence_id, source_node, "FROM_SOURCE")

        if published:
            self.add_date(published)
            self.add_edge(evidence_id, date_id(published), "PUBLISHED_ON")

        if stance is not None:
            self.set_stance(
                evidence_id, claim_id, stance, score=score, relevance=relevance
            )

        return evidence_id

    def set_stance(self, evidence_id, claim_id, stance, score=None, relevance=None,
                   method=None, best_passage=None, misleading=False, note=None):
        """
        Record what a piece of evidence says about a claim.

        Re-stating a stance *replaces* the previous one: a re-scored item
        must not leave a stale SUPPORTS edge behind next to its new
        REFUTES edge, or the totals would count it twice, on both sides.
        """
        self._require(evidence_id, "evidence")
        self._require(claim_id, "claim")

        stance = normalize_stance(stance)

        for edge_type in STANCE_EDGE_TYPES:
            if self.g.has_edge(evidence_id, claim_id, edge_type):
                self.g.remove_edge(evidence_id, claim_id, edge_type)

        self.add_edge(
            evidence_id,
            claim_id,
            STANCE_EDGES[stance],
            score=1.0 if score is None else round(float(score), 4),
            relevance=None if relevance is None else round(float(relevance), 4),
            method=method,
            best_passage=best_passage,
            misleading=bool(misleading),
            note=note,
        )

        node = self.g.nodes[evidence_id]
        node["stance"] = stance

        if misleading:
            node["misleading"] = True

        return stance

    def add_duplicate(self, image_node, evidence_id, similarity=None, first_seen=None,
                      context=None):
        """
        Mark that an image is a re-post of something already known.

        The match itself is an `evidence` node (it was retrieved, it has a
        source and a date), and DUPLICATE_OF is the claim that *this*
        picture is that picture. `first_seen` additionally pins the image
        to a date node, which is what makes recycled media show up in
        `timeline()` without any special casing.
        """
        self._require(image_node, "image")
        self._require(evidence_id, "evidence")

        self.add_edge(
            image_node,
            evidence_id,
            "DUPLICATE_OF",
            similarity=None if similarity is None else round(float(similarity), 4),
            context=context,
        )

        first_seen = _iso(first_seen)

        if first_seen:
            self.add_date(first_seen)
            self.add_edge(image_node, date_id(first_seen), "FIRST_SEEN_ON")
            self.g.nodes[image_node]["first_seen"] = first_seen

        return evidence_id

    def flag_date_mismatch(self, image_node, first_seen, claimed_date=None, note=None):
        """
        Flag a picture that predates the event it is being shown as.

        `claimed_date` defaults to the packet's own date — for a forward,
        "now" is the implied claim. Nothing is flagged when the gap is
        under `DATE_MISMATCH_DAYS`, because a photo published a week
        before the story is just a photo published a week before the
        story.
        """
        self._require(image_node, "image")

        first_seen = _iso(first_seen)
        claimed = _iso(claimed_date) or self.source_date

        if not first_seen:
            return None

        gap = _days_between(claimed, first_seen) if claimed else None

        if claimed and first_seen >= claimed:
            return None

        if gap is not None and gap < DATE_MISMATCH_DAYS:
            return None

        self.add_date(first_seen)
        self.add_edge(
            image_node,
            date_id(first_seen),
            "DATE_MISMATCH",
            first_seen=first_seen,
            claimed_date=claimed,
            gap_days=gap,
            note=note,
        )

        self.g.nodes[image_node]["first_seen"] = first_seen
        self.g.nodes[image_node]["date_mismatch"] = True

        return first_seen

    def add_verdict(self, claim_id, label, confidence=None, explanation=None, **attrs):
        """Attach the conclusion reached about a claim (stage 6)."""
        self._require(claim_id, "claim")

        node_id = verdict_id(claim_id)

        self.add_node(
            node_id,
            "verdict",
            label=label,
            confidence=None if confidence is None else round(float(confidence), 4),
            explanation=explanation,
            claim_id=claim_id,
            **attrs,
        )
        self.add_edge(claim_id, node_id, "HAS_VERDICT")

        self.g.nodes[claim_id]["verdict"] = label

        return node_id

    # --- reading it back --------------------------------------------------

    def nodes_of(self, kind):
        """[(id, attrs)] for one node kind, in insertion order."""
        return [
            (node_id, attrs)
            for node_id, attrs in self.g.nodes(data=True)
            if attrs.get("kind") == kind
        ]

    def node(self, node_id):
        return dict(self.g.nodes[node_id]) if self.g.has_node(node_id) else None

    def claims(self):
        return self.nodes_of("claim")

    def images(self):
        return self.nodes_of("image")

    def entities(self):
        return self.nodes_of("entity")

    def stance_of(self, evidence_id, claim_id):
        """The stance recorded for one pair, or None if nothing decided yet."""
        for edge_type in STANCE_EDGE_TYPES:
            if self.g.has_edge(evidence_id, claim_id, edge_type):
                return STANCE_FROM_EDGE[edge_type]

        return None

    def evidence_for(self, claim_id, stance=None, decisive_only=False):
        """
        Every evidence item attached to a claim, strongest first.

        Each entry is the evidence node's attributes plus the `stance`,
        `stance_score` and `weighted` figure the totals are built from, so
        a caller can render a verdict's reasons without walking the graph
        itself.
        """
        self._require(claim_id, "claim")

        if stance is not None:
            stance = normalize_stance(stance)

        found = []

        for _claim, evidence_id, key in self.g.out_edges(claim_id, keys=True):
            if key != "HAS_EVIDENCE":
                continue

            attrs = dict(self.g.nodes[evidence_id])
            recorded = self.stance_of(evidence_id, claim_id)

            edge = None

            if recorded is not None:
                edge = self.g.edges[evidence_id, claim_id, STANCE_EDGES[recorded]]

            if stance is not None and recorded != stance:
                continue

            if decisive_only and not attrs.get("decisive"):
                continue

            score = float((edge or {}).get("score") or 0.0)
            weight = float(attrs.get("source_weight") or DEFAULT_SOURCE_WEIGHT)

            found.append(
                {
                    "id": evidence_id,
                    **attrs,
                    "stance": recorded,
                    "stance_score": score,
                    "relevance": (edge or {}).get("relevance"),
                    "best_passage": (edge or {}).get("best_passage"),
                    "method": (edge or {}).get("method") or attrs.get("method"),
                    "misleading": bool((edge or {}).get("misleading")),
                    "weighted": round(score * weight, 4),
                }
            )

        # Decisive first, then the weighted stance, then the credibility of
        # the source. The last two tiers matter before stage 4 has run:
        # every stance is None then, so every weighted score is 0, and
        # without them the listing would fall back to id order and bury
        # the fact-check that settles the claim under a loose match.
        found.sort(
            key=lambda item: (
                0 if item.get("decisive") else 1,
                -item["weighted"],
                -float(item.get("source_weight") or 0.0),
                item["id"],
            )
        )

        return found

    def stance_totals(self, claim_id):
        """
        Weighted support vs. refutation for one claim.

        Each item contributes `stance score x source credibility`, so a
        PIB fact-check outweighs three blogs, and `margin` is what a
        verdict thresholds on. `decisive` counts the fact-checks that
        matched the claim closely enough to settle it on their own.
        """
        totals = {"supports": 0.0, "refutes": 0.0, "neutral": 0.0}
        counts = {"supports": 0, "refutes": 0, "neutral": 0, "unscored": 0}
        decisive = 0

        for item in self.evidence_for(claim_id):
            stance = item["stance"]

            if stance is None:
                counts["unscored"] += 1
                continue

            totals[stance] += item["weighted"]
            counts[stance] += 1

            if item.get("decisive") and stance in ("supports", "refutes"):
                decisive += 1

        return {
            "claim_id": claim_id,
            "supports": round(totals["supports"], 4),
            "refutes": round(totals["refutes"], 4),
            "neutral": round(totals["neutral"], 4),
            "margin": round(totals["supports"] - totals["refutes"], 4),
            "counts": counts,
            "evidence": sum(counts.values()),
            "decisive": decisive,
        }

    def decisive_hits(self, claim_id=None):
        """
        Fact-checks that match a claim closely enough to settle it.

        A verdict consults these before it looks at any weighted sum: one
        IFCN publisher rating the exact claim "false" is worth more than
        an aggregate of loosely related articles.
        """
        claim_ids = [claim_id] if claim_id else [cid for cid, _ in self.claims()]
        hits = []

        for cid in claim_ids:
            for item in self.evidence_for(cid, decisive_only=True):
                if item["stance"] in ("supports", "refutes"):
                    hits.append({"claim_id": cid, **item})

        hits.sort(key=lambda item: -item["weighted"])

        return hits

    def open_claims(self):
        """
        Check-worthy claims nothing has taken a side on yet.

        These are what a bot reply has to hedge about, and what a second
        retrieval pass should target.
        """
        open_ids = []

        for claim_id, attrs in self.claims():
            if not attrs.get("check_worthy", True):
                continue

            totals = self.stance_totals(claim_id)

            if totals["counts"]["supports"] == 0 and totals["counts"]["refutes"] == 0:
                open_ids.append(claim_id)

        return open_ids

    def timeline(self):
        """
        Everything the graph knows a date for, oldest first.

        Recycled media is a *chronology* problem — the picture is older
        than the story it illustrates — so the dates are pulled together
        into one ordered list: EXIF capture dates, evidence publication
        dates, first-seen dates and mismatch flags.
        """
        events = []

        if self.source_date:
            events.append(
                {
                    "date": self.source_date,
                    "type": "packet",
                    "node": self.packet_id,
                    "label": f"message dated {self.source_date}",
                }
            )

        labels = {
            "CAPTURED_ON": "image captured",
            "PUBLISHED_ON": "evidence published",
            "FIRST_SEEN_ON": "image first seen",
            "DATE_MISMATCH": "date mismatch",
        }

        for src, dst, key, attrs in self.g.edges(keys=True, data=True):
            edge_type = attrs.get("type")

            if edge_type not in labels:
                continue

            node = self.g.nodes[src]
            when = self.g.nodes[dst].get("date")

            if not when:
                continue

            label = labels[edge_type]
            title = node.get("title") or node.get("path") or src

            if edge_type == "DATE_MISMATCH":
                gap = attrs.get("gap_days")
                detail = f"{label}: {title} is {gap} days older than claimed" if gap else f"{label}: {title}"
            else:
                detail = f"{label}: {title}"

            events.append(
                {"date": when, "type": edge_type, "node": src, "label": detail}
            )

        events.sort(key=lambda event: (event["date"], event["type"], event["node"]))

        return events

    def summary(self, max_chars=1200):
        """
        A compact, human-readable account of the graph.

        This is what a bot reply or a log line quotes, so it leads with the
        claims that have a side taken on them, names the publisher behind
        each one, and stops cleanly at `max_chars` rather than mid-word.
        """
        lines = [
            f"packet {self.packet_id} ({self.input_type or 'unknown'})"
            + (f", dated {self.source_date}" if self.source_date else "")
        ]

        claims = self.claims()
        images = self.images()

        counts = {kind: len(self.nodes_of(kind)) for kind in NODE_TYPES}
        lines.append(
            "{claims} claims, {images} images, {evidence} evidence, {sources} sources".format(
                claims=counts["claim"],
                images=counts["image"],
                evidence=counts["evidence"],
                sources=counts["source"],
            )
        )

        ranked = sorted(
            claims,
            key=lambda pair: -abs(self.stance_totals(pair[0])["margin"]),
        )

        for claim_id, attrs in ranked:
            totals = self.stance_totals(claim_id)
            text = (attrs.get("text") or "").strip()

            verdict = attrs.get("verdict")
            head = f"- {text[:160]}"

            if verdict:
                head += f" -> {verdict}"

            lines.append(head)
            lines.append(
                "  supports {supports} / refutes {refutes} over {evidence} items".format(**totals)
            )

            for item in self.evidence_for(claim_id)[:2]:
                stance = item["stance"] or "unscored"
                publisher = item.get("publisher") or item.get("domain") or "unknown"
                rating = f" [{item['rating']}]" if item.get("rating") else ""
                demo = " (demo data)" if item.get("demo") else ""
                lines.append(
                    f"    {stance}: {publisher}{rating}{demo} — {(item.get('title') or '')[:80]}"
                )

        for image_node, attrs in images:
            if attrs.get("date_mismatch"):
                lines.append(
                    f"- recycled image: {attrs.get('path')} first seen {attrs.get('first_seen')}"
                )

        text = "\n".join(lines)

        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + " ..."

        return text

    # --- serialisation ----------------------------------------------------

    def to_dict(self):
        return {
            "packet_id": self.packet_id,
            "input_type": self.input_type,
            "source_date": self.source_date,
            "nodes": [
                {"id": node_id, **attrs} for node_id, attrs in self.g.nodes(data=True)
            ],
            "edges": [
                {
                    "from": src,
                    "to": dst,
                    **{k: v for k, v in attrs.items()},
                }
                for src, dst, attrs in self.g.edges(data=True)
            ],
        }

    def to_json(self, path=None, indent=2):
        """Serialise to a JSON string, and to `path` as well when given."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

        if path:
            directory = os.path.dirname(os.path.abspath(path))

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

        return text

    @classmethod
    def from_dict(cls, data):
        graph = cls(
            packet_id=data.get("packet_id"),
            input_type=data.get("input_type"),
            source_date=data.get("source_date"),
        )

        for node in data.get("nodes", []):
            attrs = {k: v for k, v in node.items() if k not in ("id", "kind")}
            graph.add_node(node["id"], node["kind"], **attrs)

        for edge in data.get("edges", []):
            attrs = {
                k: v for k, v in edge.items() if k not in ("from", "to", "type", "via")
            }
            graph.add_edge(
                edge["from"], edge["to"], edge["type"], via=edge.get("via"), **attrs
            )

        return graph

    @classmethod
    def from_json(cls, text):
        """Rebuild from a JSON string, or from a path to one."""
        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")

        stripped = text.strip()

        if not stripped.startswith("{"):
            with open(text, "r", encoding="utf-8") as handle:
                stripped = handle.read()

        return cls.from_dict(json.loads(stripped))

    # --- presentation -----------------------------------------------------

    COLOURS = {
        "packet": "#6b7280",
        "claim": "#2563eb",
        "entity": "#0891b2",
        "image": "#7c3aed",
        "evidence": "#ca8a04",
        "source": "#4b5563",
        "date": "#059669",
        "verdict": "#dc2626",
    }

    def export_html(self, path, height="800px", notebook=False):
        """
        Write an interactive pyvis view of the graph.

        Presentation only — it is for showing a judge or a reviewer why a
        verdict was reached. Missing pyvis is not an error: it logs and
        returns None, like every other optional dependency here.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            log.warning("pyvis is not installed; skipping HTML export of %s", path)
            return None

        try:
            # cdn_resources="in_line" embeds vis.js in the file itself: one
            # self-contained page that works offline. The default ("local")
            # copies a `lib/` tree into the *working* directory, which
            # pollutes the repo and leaves the written file pointing at
            # scripts that are not next to it.
            network = Network(
                height=height,
                directed=True,
                notebook=notebook,
                cdn_resources="in_line",
            )

            for node_id, attrs in self.g.nodes(data=True):
                kind = attrs.get("kind", "packet")
                label = (
                    attrs.get("text")
                    or attrs.get("title")
                    or attrs.get("date")
                    or attrs.get("path")
                    or node_id
                )

                network.add_node(
                    node_id,
                    label=f"{kind}: {str(label)[:40]}",
                    title=json.dumps(attrs, ensure_ascii=False, indent=2)[:2000],
                    color=self.COLOURS.get(kind, "#9ca3af"),
                )

            for src, dst, attrs in self.g.edges(data=True):
                network.add_edge(src, dst, label=attrs.get("type"), title=attrs.get("type"))

            directory = os.path.dirname(os.path.abspath(path))

            if directory:
                os.makedirs(directory, exist_ok=True)

            # generate_html + our own write, rather than write_html():
            # show() opens a browser (wrong on a server, hangs a test run)
            # and write_html() opens the file with the platform's default
            # encoding, which throws UnicodeEncodeError on Windows the
            # moment a node label contains Devanagari or an em dash.
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(network.generate_html(notebook=notebook))

            return path
        except Exception as error:                     # fail soft, never raise
            log.warning("pyvis export failed (%s): %s", type(error).__name__, error)
            return None

    # --- internals --------------------------------------------------------

    def _require(self, node_id, kind):
        if not self.g.has_node(node_id):
            raise KeyError(f"unknown {kind} {node_id!r}")

        actual = self.g.nodes[node_id].get("kind")

        if actual != kind:
            raise ValueError(f"node {node_id!r} is a {actual}, not a {kind}")

        return self.g.nodes[node_id]

    def __len__(self):
        return self.g.number_of_nodes()

    def __repr__(self):
        return (
            f"<EvidenceGraph packet={self.packet_id} "
            f"nodes={self.g.number_of_nodes()} edges={self.g.number_of_edges()}>"
        )
