"""
Stage 4 entry point: what does each piece of evidence say about its claim?

    classify_stance(claims, evidence) -> list[StanceResult]

The pipeline per claim is: chunk each document into 2-3 sentence passages,
rank them against the claim with a multilingual embedder, send the top few
to the entailment model in one batch, and let the strongest non-neutral
verdict win.

Two rules sit on top of the model:

  * A publisher's own rating beats the text. Fact-check articles quote the
    false claim in their headline and lead ("Did the RBI ban Rs 500 notes?
    ... the claim is false"), so an entailment model reading those
    passages frequently calls it *support*. When the source states a
    rating, that rating decides the stance, and a support-vs-refute
    conflict is flagged `misleading` so a verdict can say why it ignored
    the text.

  * Nothing here raises. A missing model, a failed download or an empty
    document all come back as neutral with method="unavailable", because
    a stage that crashes takes the whole verdict down with it, while a
    stage that abstains only costs the claim one piece of evidence.

This module is pure: it reads claims and evidence and returns results. It
neither builds nor touches the evidence graph.
"""

from . import nli, rank
from .passages import split_passages
from .rank import RELEVANCE_CUTOFF, TOP_K
from .schema import EvidenceInput, StanceResult


# Publisher verdict strings, matched as substrings of a lowercased rating.
# Checked mixed-first so "half true" and "partly false" do not read as a
# clean "true" / "false".
RATING_NEUTRAL = (
    "mixture", "mixed", "half true", "half-true", "partly", "partially",
    "unproven", "unverified", "unclear", "outdated", "lacks context",
    "missing context", "no verdict",
)

RATING_REFUTE = (
    "false", "fake", "hoax", "scam", "misleading", "miscaptioned", "altered",
    "manipulated", "doctored", "debunked", "no evidence", "incorrect",
    "untrue", "not true", "misinformation", "disinformation", "satire",
    "pants on fire", "legend", "rumour", "rumor",
    "भ्रामक", "फर्जी", "झूठ",
)

RATING_SUPPORT = (
    "mostly true", "true", "correct", "accurate", "verified", "genuine",
    "confirmed", "legit",
    "सही", "सच",
)

RATING_GROUPS = (
    ("neutral", RATING_NEUTRAL),
    ("refute", RATING_REFUTE),
    ("support", RATING_SUPPORT),
)

# An explicit publisher verdict is categorical, not a probability.
RATING_SCORE = 1.0
RATING_NEUTRAL_SCORE = 0.5


def rating_stance(rating):
    """
    (stance, matched pattern) for a publisher rating string, or (None, None)
    when the rating is absent or unrecognised.
    """
    text = (rating or "").strip().lower()

    if not text:
        return None, None

    for stance, patterns in RATING_GROUPS:
        for pattern in patterns:
            if pattern in text:
                return stance, pattern

    return None, None


def as_evidence_input(item):
    """Accept an EvidenceInput, a plain dict, or anything with the fields."""
    if isinstance(item, EvidenceInput):
        return item

    if isinstance(item, dict):
        return EvidenceInput(**item)

    return EvidenceInput(
        id=getattr(item, "id"),
        claim_id=getattr(item, "claim_id"),
        title=getattr(item, "title", None),
        snippet=getattr(item, "snippet", None),
        text=getattr(item, "text", None),
        rating=getattr(item, "rating", None),
        weight=getattr(item, "weight", 0.5),
    )


def _claim_texts(claims):
    index = {}

    for claim in claims:
        if isinstance(claim, dict):
            claim_id, text = claim.get("id"), claim.get("text")
        else:
            claim_id, text = getattr(claim, "id", None), getattr(claim, "text", None)

        if claim_id:
            index[claim_id] = text or ""

    return index


def _rank_group(claim_text, passage_groups, top_k, cutoff):
    """Rankings for one claim's documents, or (None, note) on failure."""
    if not any(passage_groups):
        return None, "no evidence text"

    if not rank.available():
        return None, "embedder unavailable"

    try:
        rankings = rank.rank_batch(
            claim_text, passage_groups, top_k=top_k, cutoff=cutoff
        )
    except Exception as error:                      # fail soft, never raise
        return None, "ranking failed: {}".format(type(error).__name__)

    return rankings, None


def _score_group(claim_text, rankings):
    """
    Entailment distributions for every kept passage of every document of one
    claim, in a single batched model call.

    Returns (list of per-document distribution lists, note).
    """
    pairs = []

    for position, ranking in enumerate(rankings):
        for scored in ranking.passages:
            pairs.append((position, scored.text))

    empty = [[] for _ in rankings]

    if not pairs:
        return empty, None

    if not nli.available():
        return empty, "nli unavailable"

    try:
        distributions = nli.score_pairs([text for _, text in pairs], claim_text)
    except Exception as error:                      # fail soft, never raise
        return empty, "nli failed: {}".format(type(error).__name__)

    if len(distributions) != len(pairs):
        return empty, "nli returned a short batch"

    grouped = [[] for _ in rankings]

    for (position, _text), distribution in zip(pairs, distributions):
        grouped[position].append(distribution)

    return grouped, None


def _result(item, stance, score, relevance, best_passage, method, misleading, note):
    return StanceResult(
        evidence_id=item.id,
        claim_id=item.claim_id,
        stance=stance,
        score=round(min(1.0, max(0.0, score)), 4),
        relevance=round(min(1.0, max(0.0, relevance)), 4),
        best_passage=best_passage,
        method=method,
        misleading=misleading,
        weight=item.weight,
        note=note,
    )


def _classify_item(item, ranking, distributions, note):
    """Decide one evidence item, then let its rating have the last word."""
    top = ranking.passages[0] if ranking is not None and ranking.passages else None
    best = ranking.best if ranking is not None else 0.0

    if ranking is None:
        stance, score, relevance, passage, method = (
            "neutral", 0.0, 0.0, None, "unavailable",
        )
    elif top is None:
        stance, score, relevance, passage, method = (
            "neutral", 0.0, best, None, "off_topic",
        )
        note = note or "no passage cleared the relevance cutoff"
    elif not distributions:
        stance, score, relevance, passage, method = (
            "neutral", 0.0, best, top.text, "unavailable",
        )
    else:
        stance, score, index = nli.strongest_stance(distributions)
        decider = ranking.passages[index] if index is not None else top
        relevance, passage, method = decider.score, decider.text, "nli"

    rating, pattern = rating_stance(item.rating)

    if rating is None:
        return _result(item, stance, score, relevance, passage, method, False, note)

    misleading = rating == "refute" and stance == "support"

    rating_note = "rating {!r} matched {!r}".format(item.rating, pattern)

    if misleading:
        rating_note += "; text reads as support (quotes the claim)"
    elif method == "nli" and stance != rating:
        rating_note += "; overrides nli {!r}".format(stance)

    return _result(
        item,
        rating,
        RATING_NEUTRAL_SCORE if rating == "neutral" else RATING_SCORE,
        relevance,
        # The most relevant passage, not whichever one the model reacted to.
        top.text if top is not None else None,
        "rating",
        misleading,
        rating_note,
    )


def classify_stance(claims, evidence, top_k=TOP_K, cutoff=RELEVANCE_CUTOFF):
    """
    Classify every evidence item against the claim it was retrieved for.

    `claims` is any iterable of objects or dicts carrying `id` and `text`
    (a stage-2 `Claim` works as-is); `evidence` is EvidenceInputs or dicts.
    Returns one StanceResult per evidence item, in input order. Evidence
    whose `claim_id` is not in `claims` comes back neutral rather than
    being dropped, so counts stay comparable across runs.
    """
    claim_texts = _claim_texts(claims)
    items = [as_evidence_input(item) for item in evidence]

    groups = {}

    for position, item in enumerate(items):
        groups.setdefault(item.claim_id, []).append((position, item))

    results = [None] * len(items)

    for claim_id, group in groups.items():
        claim_text = (claim_texts.get(claim_id) or "").strip()

        if not claim_text:
            note = "claim {!r} has no text".format(claim_id)

            for position, item in group:
                results[position] = _classify_item(item, None, [], note)

            continue

        passage_groups = [split_passages(item.body) for _position, item in group]

        rankings, rank_note = _rank_group(claim_text, passage_groups, top_k, cutoff)

        if rankings is None:
            for position, item in group:
                results[position] = _classify_item(item, None, [], rank_note)

            continue

        grouped, nli_note = _score_group(claim_text, rankings)

        for index, (position, item) in enumerate(group):
            if not passage_groups[index]:
                # Nothing to read — not the same thing as reading it and
                # finding it off-topic.
                results[position] = _classify_item(
                    item, None, [], "no evidence text"
                )

                continue

            results[position] = _classify_item(
                item, rankings[index], grouped[index], nli_note
            )

    return results
