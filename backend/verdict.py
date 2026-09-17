"""
Turning a graph full of evidence into something a person can act on.

Everything before this stage gathers and labels. This one decides, and it
decides by rules rather than by a model, for a reason: a verdict that
cannot be explained is not worth giving. Every label here comes with the
specific evidence that produced it, so the reply can say "PIB rated this
exact claim false in March 2024" instead of "our model is 87% confident".

The order of the rules is the whole design:

  1. **A decisive fact-check wins outright.** A publisher who checked
     *this* claim and rated it false has done the work properly; no
     weighted sum of loosely-related articles should be able to outvote
     that.
  2. **A recycled picture makes a claim misleading**, even when the
     sentence is defensible. "Floods in Chennai" over a 2015 photograph
     is misinformation about *today*, and the graph knows the picture is
     older than the message.
  3. **Otherwise, weighted stance decides**, and only when one side
     clearly outweighs the other. Anything closer than that is
     `disputed`, and no evidence at all is `unverified`.

`unverified` is a real answer and the honest one for most forwards. A
system that guesses "false" because it found nothing is a system that
cries wolf, and the fastest way to make people stop reading the warnings.
"""

import logging


log = logging.getLogger(__name__)


LABELS = (
    "false",        # the claim is contradicted by credible evidence
    "misleading",   # true-ish words, false impression (recycled media, missing context)
    "true",         # supported by credible evidence
    "disputed",     # credible evidence on both sides
    "unverified",   # nothing found either way
)

# Worst-first, for summarising a packet from its claims.
SEVERITY = {"false": 0, "misleading": 1, "disputed": 2, "unverified": 3, "true": 4}

# The weighted stance one side needs before it counts as settled: both an
# absolute floor (one blog agreeing is not evidence) and a ratio (it has
# to actually outweigh the other side).
MIN_WEIGHT = 0.5
DOMINANCE = 2.0

DECISIVE_CONFIDENCE = 0.9
RECYCLED_CONFIDENCE = 0.75
MAX_REASONS = 3


def _describe(item):
    """One evidence item as a phrase a reply can print."""
    publisher = item.get("publisher") or item.get("domain") or "an unnamed source"
    title = (item.get("title") or "").strip()
    rating = item.get("rating")

    parts = [publisher]

    if rating and rating != "unknown":
        parts.append(f"rated it {rating}")

    if item.get("published_date"):
        parts.append(f"on {item['published_date']}")

    phrase = " ".join(parts)

    if title:
        phrase += f": {title[:120]}"

    if item.get("demo"):
        phrase += "  [demo data]"

    return phrase


def _describe_other_claim(item):
    """
    A fact-check that stage 4 gated as being about a different claim.

    Described *without* its rating, deliberately. "Rated it false" next
    to a claim the rating was never applied to is the O6 accusation in
    prose rather than in a label, and a reader would take it the same
    way.
    """
    publisher = item.get("publisher") or item.get("domain") or "A fact-checker"
    title = (item.get("title") or "").strip()

    phrase = f"{publisher} has checked a similar-sounding claim"

    if title:
        phrase += f" ({title[:120]})"

    phrase += ", which is not this one"

    if item.get("demo"):
        phrase += "  [demo data]"

    return phrase


def _recycled_images(graph, claim_id):
    """
    Pictures behind this claim that turned out to predate the message.

    EXTRACTED_FROM covers the claim that was read *off* a picture — OCR
    text or a generated caption. But the commonest case is the other way
    round: the claim is the message's own words and the photograph is
    attached to it, with no edge between them. So a claim with no picture
    of its own falls back to any flagged image in the same packet, which
    is exactly the set stage 5 attached this claim's image evidence to.
    """
    flagged = []

    for _claim, image_node, key in graph.g.out_edges(claim_id, keys=True):
        if key != "EXTRACTED_FROM":
            continue

        attrs = graph.node(image_node) or {}

        if attrs.get("date_mismatch"):
            flagged.append((image_node, attrs))

    if flagged:
        return flagged

    own_images = any(
        key == "EXTRACTED_FROM"
        for _claim, _image, key in graph.g.out_edges(claim_id, keys=True)
    )

    if own_images:
        return []

    return [
        (image_node, attrs)
        for image_node, attrs in graph.images()
        if attrs.get("date_mismatch")
    ]


def _clip_mismatches(graph, claim_id):
    """Evidence saying the picture does not show what the claim says."""
    return [
        item
        for item in graph.evidence_for(claim_id, stance="refutes")
        if item.get("method") == "clip"
    ]


def verdict_for_claim(graph, claim_id):
    """
    Decide one claim.

    Returns {label, confidence, explanation, reasons, evidence_ids,
    totals} — `reasons` being the specific items that decided it, so the
    caller never has to re-derive the argument.
    """
    totals = graph.stance_totals(claim_id)
    evidence = graph.evidence_for(claim_id)

    reasons = []
    demo_only = bool(evidence) and all(item.get("demo") for item in evidence)

    # 1. a decisive fact-check
    decisive = [item for item in graph.decisive_hits(claim_id)]

    if decisive:
        top = decisive[0]
        label = "misleading" if top.get("rating") == "misleading" else (
            "true" if top["stance"] == "supports" else "false"
        )
        reasons = [_describe(item) for item in decisive[:MAX_REASONS]]

        return _result(
            label, DECISIVE_CONFIDENCE, totals, evidence, reasons,
            f"A fact-checker has already checked this claim. {reasons[0]}",
            demo_only,
        )

    # 2. a recycled or mismatched picture
    recycled = _recycled_images(graph, claim_id)
    mismatches = _clip_mismatches(graph, claim_id)

    if recycled:
        image_node, attrs = recycled[0]
        first_seen = attrs.get("first_seen")
        reasons = [
            f"the image {attrs.get('path') or image_node} has been in circulation "
            f"since {first_seen}, before this message"
        ]
        reasons += [_describe(item) for item in evidence[:MAX_REASONS - 1]]

        return _result(
            "misleading", RECYCLED_CONFIDENCE, totals, evidence, reasons,
            "The picture used here is older than the story it is being shown with: "
            f"it was already online on {first_seen}.",
            demo_only,
        )

    if mismatches:
        reasons = [item.get("best_passage") or item.get("snippet") or _describe(item)
                   for item in mismatches[:MAX_REASONS]]

        return _result(
            "misleading", 0.6, totals, evidence, reasons,
            "The picture does not appear to show what this claim describes.",
            demo_only,
        )

    # 3. weighted stance
    supports = totals["supports"]
    refutes = totals["refutes"]

    if refutes >= MIN_WEIGHT and refutes >= supports * DOMINANCE:
        label, confidence = "false", min(0.85, 0.5 + refutes / 4)
        side = "refutes"
    elif supports >= MIN_WEIGHT and supports >= refutes * DOMINANCE:
        label, confidence = "true", min(0.85, 0.5 + supports / 4)
        side = "supports"
    elif supports > 0 and refutes > 0:
        label, confidence, side = "disputed", 0.4, None
    else:
        # Evidence stage 4 gated: retrieved on topic, reviewing a
        # different claim (DECISIONS.md O6). Saying so is the whole
        # point — "nothing found" would hide that a fact-check of the
        # neighbouring rumour is sitting right there, and the old
        # behaviour was to convict on it.
        gated = [item for item in evidence if item.get("method") == "not_about"]

        if gated:
            explanation = (
                "No fact-check of this claim was found. "
                f"{_describe_other_claim(gated[0])}."
            )
            reasons = [_describe_other_claim(item) for item in gated[:MAX_REASONS]]
        elif evidence:
            explanation = (
                "Some related material was found, but none of it takes a side "
                "on this claim."
            )
        else:
            explanation = "Nothing was found that settles this claim either way."

        return _result(
            "unverified", 0.2, totals, evidence, reasons, explanation, demo_only,
        )

    if side:
        top = graph.evidence_for(claim_id, stance=side)[:MAX_REASONS]
        reasons = [_describe(item) for item in top]
        explanation = (
            f"The weight of the evidence {side} this claim"
            + (f". {reasons[0]}" if reasons else ".")
        )
    else:
        reasons = [_describe(item) for item in evidence[:MAX_REASONS]]
        explanation = (
            "Credible sources disagree about this claim, so it cannot be called "
            "either way from what was found."
        )

    return _result(label, confidence, totals, evidence, reasons, explanation, demo_only)


def _result(label, confidence, totals, evidence, reasons, explanation, demo_only):
    if demo_only and label != "unverified":
        explanation += (
            "  (This rests on the demo evidence index, not on live fact-checks.)"
        )
        confidence = round(confidence * 0.8, 4)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "explanation": explanation,
        "reasons": reasons,
        "evidence_ids": [item["id"] for item in evidence],
        "totals": totals,
        "demo_only": demo_only,
    }


def decide(graph, write=True):
    """
    Decide every claim in the graph, and the packet as a whole.

    With `write=True` each claim gets a `verdict` node and a HAS_VERDICT
    edge, so the conclusion is part of the graph rather than something
    the caller has to carry separately.

    The packet's own label is the worst of its check-worthy claims: one
    false claim in a message makes the message false, whatever else it
    says.
    """
    claims = []

    for claim_id, attrs in graph.claims():
        if not attrs.get("check_worthy", True):
            continue

        verdict = verdict_for_claim(graph, claim_id)
        verdict["claim_id"] = claim_id
        verdict["claim"] = attrs.get("text")

        claims.append(verdict)

        if write:
            try:
                graph.add_verdict(
                    claim_id,
                    verdict["label"],
                    confidence=verdict["confidence"],
                    explanation=verdict["explanation"],
                )
            except Exception as error:               # fail soft, never raise
                log.warning(
                    "could not write the verdict for %s (%s): %s",
                    claim_id, type(error).__name__, error,
                )

    claims.sort(key=lambda verdict: (SEVERITY[verdict["label"]], -verdict["confidence"]))

    counts = {}

    for verdict in claims:
        counts[verdict["label"]] = counts.get(verdict["label"], 0) + 1

    overall = claims[0]["label"] if claims else "unverified"

    return {
        "label": overall,
        "confidence": claims[0]["confidence"] if claims else 0.0,
        "summary": _packet_summary(overall, claims),
        "counts": counts,
        "claims": claims,
    }


def _packet_summary(label, claims):
    """One sentence for a bot reply."""
    if not claims:
        return "No check-worthy claims were found in this message."

    worst = claims[0]

    openings = {
        "false": "This message contains a false claim.",
        "misleading": "This message is misleading.",
        "true": "The claims in this message check out.",
        "disputed": "The claims in this message are disputed.",
        "unverified": "Nothing was found to confirm or deny this message.",
    }

    line = openings.get(label, openings["unverified"])

    if len(claims) > 1:
        line += f" ({len(claims)} claims checked.)"

    return f"{line} {worst['explanation']}"
