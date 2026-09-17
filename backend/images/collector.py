"""
Stage 5 entry point: what do the pictures say?

    collect_image_evidence(claimset, graph, packet) -> ImageReport

Three checks per image, each answering a different question:

  * **The local index** — have we seen this exact picture before, and
    when? A match becomes an `evidence` node, a DUPLICATE_OF edge, and,
    when the picture turns out to predate the message by more than a
    month, a DATE_MISMATCH. This is the recycled-media finding.
  * **Reverse image search** — the same question asked of the open web,
    for images that have a public URL.
  * **The CLIP consistency check** — does the picture actually show what
    the claim says it shows? A mismatch is evidence *against* the claim,
    written with stance `refutes` and the `misleading` flag.

Which claims does an image's evidence attach to? The ones stage 3a linked
to it with EXTRACTED_FROM — the claims that were read off this picture's
OCR text or caption. An image with no claims of its own falls back to the
packet's check-worthy claims, because a photograph attached to a message
is being used to support that message whether or not it carries text.

Only the first two checks add `evidence` nodes with no stance; the CLIP
check writes its own stance, because "this picture does not show that"
is a judgement, not a retrieval. Nothing here raises.
"""

import logging
from datetime import date
from typing import List, NamedTuple

from ..evidence.schema import EvidenceCandidate
from . import consistency, local_index, reverse_search
from .keyframes import MAX_IMAGES, SIMILARITY_THRESHOLD, select_keyframes


log = logging.getLogger(__name__)


# A local-index match is a statement by a curated corpus, so it is
# weighted like a government source rather than like a random web page.
# It is not 1.0: the corpus is demo data, and saying so in the weight is
# more honest than saying it only in a footnote.
INDEX_WEIGHT = 0.9


class ImageReport(NamedTuple):
    images: int
    dropped_frames: int
    index_matches: int
    reverse_matches: int
    mismatches: int
    date_mismatches: int
    errors: List[str]

    def as_dict(self):
        return {
            "images": self.images,
            "dropped_frames": self.dropped_frames,
            "index_matches": self.index_matches,
            "reverse_matches": self.reverse_matches,
            "mismatches": self.mismatches,
            "date_mismatches": self.date_mismatches,
            "errors": list(self.errors),
        }


def claims_for_image(graph, image_node, fallback):
    """
    The claims an image's evidence should attach to.

    EXTRACTED_FROM first (claims read off this picture), then the
    packet's check-worthy claims as a fallback.
    """
    linked = [
        src
        for src, dst, key in graph.g.in_edges(image_node, keys=True)
        if key == "EXTRACTED_FROM"
    ]

    return linked or list(fallback)


def _index_candidate(claim_id, match, image_node):
    """A local-index match, as an evidence candidate."""
    context = match.get("context") or match.get("caption") or ""
    first_seen = match.get("first_seen")

    title = f"This image has been published before: {context}" if context \
        else "This image matches one already in the index"

    snippet = context

    if first_seen:
        snippet = f"{context} First seen {first_seen}.".strip()

    return EvidenceCandidate(
        claim_id=claim_id,
        source_type="reverse_image",
        query=f"image:{image_node}",
        url=match.get("source_url"),
        domain=match.get("domain"),
        title=title,
        snippet=snippet,
        text=match.get("description") or snippet,
        publisher=match.get("publisher") or "Local image index",
        published_date=first_seen,
        retrieved_at=date.today().isoformat(),
        source_weight=INDEX_WEIGHT,
        score=match.get("similarity"),
        demo=bool(match.get("demo", True)),
        meta={
            "image_node": image_node,
            "similarity": match.get("similarity"),
            "first_seen": first_seen,
            "context": context,
            "backend": match.get("backend"),
            "demo_note": match.get("demo_note"),
        },
    )


def _mismatch_candidate(claim_id, finding, image_node, image):
    """A CLIP caption mismatch, as an evidence candidate."""
    description = image.get("description") or ""

    return EvidenceCandidate(
        claim_id=claim_id,
        source_type="image_caption",
        query=f"image:{image_node}",
        title="The picture does not show what this claim says",
        snippet=finding["reason"],
        text=(
            f"{finding['reason']}"
            + (f" The image was described as: {description}" if description else "")
        ),
        publisher="Image/caption consistency check",
        retrieved_at=date.today().isoformat(),
        # The check is a model's opinion about this packet, not a
        # published source, so it carries a middling weight.
        source_weight=0.6,
        score=finding.get("similarity"),
        meta={
            "image_node": image_node,
            "similarity": finding.get("similarity"),
            "description_similarity": finding.get("description_similarity"),
            "description": description,
            "method": "clip",
        },
    )


def collect_image_evidence(claimset, graph, packet, threshold=None,
                           index_floor=local_index.SIMILARITY_FLOOR,
                           run_reverse_search=True, run_consistency=True):
    """
    Run every image check and write the findings into the graph.

    The packet's images are expected to be the ones already in the graph
    (stage 3a put them there); `keyframes.select_keyframes` is applied
    first so a long video's near-identical frames are not all checked.
    """
    report = {
        "images": 0,
        "dropped_frames": 0,
        "index_matches": 0,
        "reverse_matches": 0,
        "mismatches": 0,
        "date_mismatches": 0,
        "errors": [],
    }

    images = (packet or {}).get("images") or []

    if not images:
        return ImageReport(**report)

    kept = select_keyframes(images, threshold=SIMILARITY_THRESHOLD, cap=MAX_IMAGES)
    report["dropped_frames"] = len(images) - len(kept)

    # The graph's image nodes, in packet order, so a packet image can be
    # matched to the node stage 3a created for it.
    nodes = {attrs.get("index"): node_id for node_id, attrs in graph.images()}

    fallback_claims = [
        claim.id for claim in claimset.claims
        if claim.check_worthy and claim.id in dict(graph.claims())
    ]

    for position, image in enumerate(images):
        if image not in kept:
            continue

        image_node = nodes.get(position)

        if image_node is None:
            continue

        report["images"] += 1
        claim_ids = claims_for_image(graph, image_node, fallback_claims)

        if not claim_ids:
            continue

        _check_index(graph, image, image_node, claim_ids, index_floor, report)

        if run_reverse_search:
            _check_reverse(graph, image, image_node, claim_ids, report)

        if run_consistency:
            _check_consistency(graph, image, image_node, claim_ids, threshold, report)

    log.info(
        "images: %s checked (%s frames dropped) -> %s index matches, "
        "%s reverse matches, %s caption mismatches, %s date mismatches",
        report["images"], report["dropped_frames"], report["index_matches"],
        report["reverse_matches"], report["mismatches"], report["date_mismatches"],
    )

    return ImageReport(**report)


def _attach(graph, claim_id, candidate, report, what):
    """add_evidence, with failures recorded rather than raised."""
    try:
        return graph.add_evidence(claim_id, candidate)
    except Exception as error:                       # fail soft, never raise
        message = f"could not add {what} for {claim_id}: {type(error).__name__}: {error}"
        log.warning(message)
        report["errors"].append(message)

        return None


def _check_index(graph, image, image_node, claim_ids, floor, report):
    """Local index: is this a picture we already know, and how old is it?"""
    try:
        matches = local_index.search(image.get("embedding"), floor=floor)
    except Exception as error:                       # fail soft, never raise
        message = f"image index search failed: {type(error).__name__}: {error}"
        log.warning(message)
        report["errors"].append(message)
        return

    for match in matches:
        first_seen = match.get("first_seen")

        for claim_id in claim_ids:
            candidate = _index_candidate(claim_id, match, image_node)
            evidence_id = _attach(graph, claim_id, candidate, report, "an index match")

            if evidence_id is None:
                continue

            report["index_matches"] += 1

            try:
                graph.add_duplicate(
                    image_node, evidence_id,
                    similarity=match.get("similarity"),
                    first_seen=first_seen,
                    context=match.get("context"),
                )
            except Exception as error:               # fail soft, never raise
                message = f"could not link {evidence_id}: {type(error).__name__}: {error}"
                log.warning(message)
                report["errors"].append(message)

    # One picture, one date-mismatch finding, using the *earliest* match.
    # A picture that matches three archive entries has not been recycled
    # three times, and the oldest appearance is the one that matters.
    _flag_earliest(graph, image_node, [m.get("first_seen") for m in matches], report)


def _check_reverse(graph, image, image_node, claim_ids, report):
    """Reverse image search, for images with a public URL."""
    image_url = reverse_search.image_url_of(image)

    if not image_url or not reverse_search.available():
        return

    for claim_id in claim_ids:
        try:
            matches = reverse_search.search(claim_id, image_url)
        except Exception as error:                   # fail soft, never raise
            message = f"reverse image search failed: {type(error).__name__}: {error}"
            log.warning(message)
            report["errors"].append(message)
            return

        earliest = reverse_search.first_seen(matches)

        for candidate in matches:
            candidate.meta["image_node"] = image_node
            evidence_id = _attach(graph, claim_id, candidate, report, "a reverse match")

            if evidence_id is None:
                continue

            report["reverse_matches"] += 1

            try:
                graph.add_duplicate(
                    image_node, evidence_id,
                    first_seen=candidate.published_date,
                    context=candidate.title,
                )
            except Exception as error:               # fail soft, never raise
                report["errors"].append(str(error))

        _flag_earliest(graph, image_node, [earliest], report)


def _flag_earliest(graph, image_node, dates, report):
    """
    Flag one image as recycled, once, from the oldest date found for it.

    Counted per image rather than per match, so the report says how many
    pictures are recycled rather than how many archive entries they hit.
    """
    known = [date for date in dates if date]

    if not known:
        return

    already = bool((graph.node(image_node) or {}).get("date_mismatch"))

    try:
        flagged = graph.flag_date_mismatch(image_node, min(known))
    except Exception as error:                       # fail soft, never raise
        message = f"could not flag a date mismatch: {type(error).__name__}: {error}"
        log.warning(message)
        report["errors"].append(message)
        return

    if flagged and not already:
        report["date_mismatches"] += 1


def _check_consistency(graph, image, image_node, claim_ids, threshold, report):
    """CLIP: does the picture show what the claims say it shows?"""
    claim_nodes = dict(graph.claims())
    claims = [
        (claim_id, claim_nodes[claim_id].get("text") or "")
        for claim_id in claim_ids
        if claim_id in claim_nodes
    ]

    if not claims:
        return

    try:
        findings = consistency.check_image(
            image.get("path"),
            claims,
            description=image.get("description"),
            minimum=threshold,
        )
    except Exception as error:                       # fail soft, never raise
        message = f"the CLIP check failed: {type(error).__name__}: {error}"
        log.warning(message)
        report["errors"].append(message)
        return

    for finding in findings:
        if not finding["mismatch"]:
            continue

        claim_id = finding["claim_id"]
        candidate = _mismatch_candidate(claim_id, finding, image_node, image)
        evidence_id = _attach(graph, claim_id, candidate, report, "a caption mismatch")

        if evidence_id is None:
            continue

        report["mismatches"] += 1

        try:
            # This one *is* a judgement, so it carries its own stance
            # rather than waiting for stage 4 to read the text back.
            graph.set_stance(
                evidence_id, claim_id, "refutes",
                score=1.0 - float(finding.get("similarity") or 0.0),
                relevance=finding.get("similarity"),
                method="clip",
                misleading=True,
                note=finding["reason"],
            )
        except Exception as error:                   # fail soft, never raise
            message = f"could not set the CLIP stance: {type(error).__name__}: {error}"
            log.warning(message)
            report["errors"].append(message)
