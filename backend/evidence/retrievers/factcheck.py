"""
Google Fact Check Tools — the claims that professional fact-checkers have
already published a verdict on.

This is the highest-value retriever in the pipeline, because what it
returns is not an article that might bear on the claim but a *rating* of
a claim, by a publisher, with a date. Most viral Indian forwards are
recycled: the UPI cashback scam, the UNESCO anthem claim and the salt
shortage have each been checked many times, so a single hit here often
settles the question outright.

English and Hindi are queried separately (`languageCode` takes one
locale, and the Indian fact-check corpus is substantially Hindi), and the
results are merged.

No key, no problem: with `GOOGLE_FACTCHECK_KEY` unset the retriever logs
once and returns nothing, and the pipeline runs on its other sources.
"""

import logging
from datetime import date

from ...common import http
from ..schema import EvidenceCandidate


log = logging.getLogger(__name__)


ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
CACHE_NAMESPACE = "factcheck"
KEY_ENV = "GOOGLE_FACTCHECK_KEY"

DEFAULT_LANGUAGES = ("en", "hi")
PAGE_SIZE = 10


def available():
    return http.api_key(KEY_ENV) is not None


def _publisher(review):
    publisher = review.get("publisher") or {}

    return publisher.get("name") or publisher.get("site") or None


def _candidates_from(payload, claim_id, query, language):
    """Flatten the API's claim -> claimReview[] shape into candidates."""
    candidates = []
    today = date.today().isoformat()

    for claim in (payload or {}).get("claims", []) or []:
        claim_text = claim.get("text") or ""
        claimant = claim.get("claimant")

        for review in claim.get("claimReview", []) or []:
            url = review.get("url")

            if not url:
                continue

            title = review.get("title") or claim_text

            # The claim as the publisher recorded it is the most useful
            # snippet there is: it is what stage 4 compares our claim to.
            snippet = claim_text or review.get("title") or ""

            if claimant and claim_text:
                snippet = f"{claim_text} (claimed by {claimant})"

            candidates.append(
                EvidenceCandidate(
                    claim_id=claim_id,
                    source_type="factcheck",
                    query=query,
                    url=url,
                    title=title,
                    snippet=snippet,
                    publisher=_publisher(review),
                    rating_raw=review.get("textualRating"),
                    published_date=(review.get("reviewDate") or claim.get("claimDate") or "")[:10] or None,
                    retrieved_at=today,
                    language=review.get("languageCode") or language,
                    meta={"claimant": claimant} if claimant else {},
                )
            )

    return candidates


def search(claim_id, query, languages=DEFAULT_LANGUAGES, page_size=PAGE_SIZE):
    """
    Search the fact-check corpus for one query.

    Returns `[]` — never raises, never None — when the key is missing or
    the API fails, so a caller can always iterate the result.
    """
    key = http.api_key(KEY_ENV)

    if not key:
        log.info("%s is not set; skipping the fact-check retriever", KEY_ENV)
        return []

    text = (query or "").strip()

    if not text:
        return []

    found = []

    for language in languages or DEFAULT_LANGUAGES:
        payload = http.get_json(
            ENDPOINT,
            params={
                "query": text,
                "languageCode": language,
                "pageSize": page_size,
                "key": key,
            },
            namespace=CACHE_NAMESPACE,
            # The key is deliberately left out of the cache key: two
            # people with different keys should share one cache, and a
            # key does not belong in a filename.
            cache_parts=["factcheck", text, language, page_size],
        )

        if not payload:
            continue

        found.extend(_candidates_from(payload, claim_id, text, language))

    return found


def search_many(claim_id, queries, languages=DEFAULT_LANGUAGES):
    """Run several queries for one claim and concatenate the results."""
    results = []

    for query in queries:
        text = getattr(query, "text", query)
        language = getattr(query, "language", None)

        # A Hindi claim is still worth checking against English
        # fact-checks (and vice versa), so both locales are always asked;
        # `language` only decides which comes first.
        locales = (
            (language,) + tuple(l for l in languages if l != language)
            if language else languages
        )

        results.extend(search(claim_id, text, languages=locales))

    return results
