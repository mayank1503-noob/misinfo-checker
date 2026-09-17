"""
General web search, via Tavily.

The fact-check corpus only covers claims someone has already checked.
Most claims arrive before that: a rumour about a scheme announced this
morning, a statistic from a press conference, an attribution to a
minister. For those the question is not "has this been debunked" but
"what does the record say", and that means a web search — scoped, where
the claim type allows it, to the sources that can actually answer it
(`Query.sites` becomes Tavily's `include_domains`).

Tavily is used rather than a raw search engine because it returns page
content with each result, so the pipeline usually does not need a second
fetch to have something for stage 4 to read.

With `TAVILY_API_KEY` unset this retriever logs once and returns nothing.
"""

import logging
from datetime import date

from ...common import http
from ..schema import EvidenceCandidate


log = logging.getLogger(__name__)


ENDPOINT = "https://api.tavily.com/search"
CACHE_NAMESPACE = "web"
KEY_ENV = "TAVILY_API_KEY"

MAX_RESULTS = 5
SEARCH_DEPTH = "basic"

# Tavily caps include_domains; a scope longer than this is the same as no
# scope at all, and sending it wastes the call.
MAX_DOMAINS = 10


def available():
    return http.api_key(KEY_ENV) is not None


def _candidates_from(payload, claim_id, query):
    candidates = []
    today = date.today().isoformat()

    for result in (payload or {}).get("results", []) or []:
        url = result.get("url")

        if not url:
            continue

        published = (result.get("published_date") or "")[:10] or None

        candidates.append(
            EvidenceCandidate(
                claim_id=claim_id,
                source_type="web",
                query=query,
                url=url,
                title=result.get("title"),
                snippet=result.get("content"),
                text=result.get("raw_content"),
                published_date=published,
                retrieved_at=today,
                score=result.get("score"),
            )
        )

    return candidates


def search(claim_id, query, sites=(), max_results=MAX_RESULTS):
    """
    One web search. Returns `[]` on a missing key or any failure.

    `sites` scopes the search to a set of domains; it is dropped when it
    is longer than the API will honour.
    """
    key = http.api_key(KEY_ENV)

    if not key:
        log.info("%s is not set; skipping the web retriever", KEY_ENV)
        return []

    text = (query or "").strip()

    if not text:
        return []

    domains = list(sites or ())[:MAX_DOMAINS]

    body = {
        "api_key": key,
        "query": text,
        "max_results": max_results,
        "search_depth": SEARCH_DEPTH,
        "include_answer": False,
    }

    if domains:
        body["include_domains"] = domains

    payload = http.post_json(
        ENDPOINT,
        json_body=body,
        namespace=CACHE_NAMESPACE,
        cache_parts=["tavily", text, domains, max_results, SEARCH_DEPTH],
    )

    return _candidates_from(payload, claim_id, text)


def search_many(claim_id, queries, max_results=MAX_RESULTS):
    """Run several queries for one claim, honouring each query's scope."""
    results = []

    for query in queries:
        results.extend(
            search(
                claim_id,
                getattr(query, "text", query),
                sites=getattr(query, "sites", ()),
                max_results=max_results,
            )
        )

    return results
