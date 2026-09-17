"""
The vocabulary of the evidence graph (Stage 3a).

Everything the pipeline learns about one forwarded message ends up in a
single directed multigraph: the packet it arrived as, the claims pulled
out of it, the entities those claims name, the images that came with it,
the evidence retrieved about them, where that evidence came from, the
dates involved, and the verdict reached.

Node and edge *types* live here, apart from the store, because they are
the contract between stages: stage 3b writes `evidence` and `source`
nodes, stage 4 writes stance edges, stage 5 writes `DUPLICATE_OF` and
`DATE_MISMATCH`, and the verdict reads all of it back. Ids are derived
from content (never from a counter or a timestamp), so the same input
rebuilds the same graph and a re-run merges instead of duplicating.
"""

from ..claims.schema import stable_id


NODE_TYPES = (
    "packet",     # the ingested message (stage 1)
    "claim",      # a check-worthy assertion pulled out of it (stage 2)
    "entity",     # a person / org / amount / date a claim names (stage 2)
    "image",      # an attached image, or a video keyframe (stage 1)
    "evidence",   # a retrieved document or media match (stages 3b / 5)
    "source",     # the publisher an evidence item came from (stage 3b)
    "date",       # a calendar date something is pinned to
    "verdict",    # the conclusion reached about a claim (stage 6)
)

EDGE_TYPES = (
    "HAS_CLAIM",       # packet   -> claim
    "MENTIONS",        # claim    -> entity
    "SHARES_ENTITY",   # claim    -> claim     (they name the same entity)
    "HAS_IMAGE",       # packet   -> image
    "EXTRACTED_FROM",  # claim    -> image     (claim came from OCR / caption)
    "CAPTURED_ON",     # image    -> date      (EXIF capture date)
    "HAS_EVIDENCE",    # claim    -> evidence
    "FROM_SOURCE",     # evidence -> source
    "PUBLISHED_ON",    # evidence -> date
    "SUPPORTS",        # evidence -> claim
    "REFUTES",         # evidence -> claim
    "NEUTRAL",         # evidence -> claim     (relevant, takes no side)
    "DUPLICATE_OF",    # image    -> evidence  (this picture is older than the story)
    "FIRST_SEEN_ON",   # image    -> date      (earliest known appearance)
    "DATE_MISMATCH",   # image    -> date      (first seen long before it is claimed)
    "HAS_VERDICT",     # claim    -> verdict
)

STANCES = ("supports", "refutes", "neutral")

# stance -> the evidence->claim edge that carries it
STANCE_EDGES = {
    "supports": "SUPPORTS",
    "refutes": "REFUTES",
    "neutral": "NEUTRAL",
}

STANCE_EDGE_TYPES = tuple(STANCE_EDGES.values())

EDGE_FROM_STANCE = STANCE_EDGES
STANCE_FROM_EDGE = {edge: stance for stance, edge in STANCE_EDGES.items()}

# Stage 4 speaks singular ("support"), the graph speaks plural
# ("supports"). Neither vocabulary is worth churning, so they are mapped
# in one place, here.
STANCE_ALIASES = {
    "support": "supports",
    "supports": "supports",
    "refute": "refutes",
    "refutes": "refutes",
    "neutral": "neutral",
    "": "neutral",
    None: "neutral",
}


def normalize_stance(stance):
    """Accept either vocabulary; raise on anything else."""
    try:
        resolved = STANCE_ALIASES[stance if stance is None else str(stance).lower()]
    except KeyError:
        raise ValueError(
            f"unknown stance {stance!r}; expected one of {STANCES}"
        ) from None

    return resolved


# --- deterministic ids ------------------------------------------------------
#
# Claim, entity and packet ids are minted by stage 2 and reused verbatim;
# only the node kinds this stage introduces need id helpers.


def image_id(packet_id, path, frame_time=None):
    """
    One id per image *within a packet*. The same file forwarded twice in
    two different messages is two image nodes (they have different
    provenance) but the same keyframe re-analysed is one.
    """
    return stable_id("img", packet_id, path or "", "" if frame_time is None else round(float(frame_time), 2))


def source_id(domain):
    """Publishers are identified by domain, so two articles from the same
    outlet share one source node and its credibility weight."""
    return stable_id("src", (domain or "unknown").lower())


def date_id(iso_date):
    """
    Date nodes are shared across the whole graph: if an image was captured
    on the day a fact-check was published, both point at one node, and the
    timeline falls out of the graph rather than being assembled by hand.
    """
    return f"date_{iso_date}"


def verdict_id(claim_id):
    return stable_id("vrd", claim_id)


def evidence_node_id(candidate_id):
    """Evidence keeps the id its retriever minted (already content-derived)."""
    return candidate_id
