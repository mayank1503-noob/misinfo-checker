"""
Turning a claim into the two or three questions worth asking about it.

A retriever is only as good as what it is asked, and the useful question
depends on what kind of claim it is. "SBI is giving Rs 5,000 cashback" is
not answered by a news search — it is answered by asking a fact-checker
whether that scam is known. "The RBI cut the repo rate by 2%" is not
answered by a fact-checker — it is answered by the RBI's own site.

So each claim produces:

  1. **the claim itself**, near-verbatim, for a semantic search;
  2. **a keyword query** of its entities and numbers, for a keyword
     search that would drown in a full sentence;
  3. **a routed query**, scoped by `claim_type` to the sources that can
     actually settle that kind of claim — fact-checkers and PIB for chain
     offers, WHO / ICMR / MoHFW for health, PIB / RBI / NPCI for policy
     and money, recent news for events, the speaker for attributions.

Hindi and Hinglish claims get their routed query in Hindi as well: the
Indian fact-check corpus is substantially Hindi, and the English query
alone will not reach it.

Nothing here performs a search. It returns `Query` objects, and the
retrievers decide what to do with `sites` — the fact-check API ignores
them, a web search turns them into a site: filter.
"""

import re
from typing import List, NamedTuple, Optional


# Where each kind of claim can actually be settled.
FACT_CHECKERS = (
    "altnews.in", "boomlive.in", "factchecker.in", "newschecker.in",
    "vishvasnews.com", "factcrescendo.com", "thequint.com",
)

PIB = ("pib.gov.in", "factcheck.pib.gov.in")

HEALTH_SOURCES = ("who.int", "icmr.gov.in", "mohfw.gov.in")

MONEY_SOURCES = ("pib.gov.in", "rbi.org.in", "npci.org.in")

NEWS_SOURCES = (
    "thehindu.com", "indianexpress.com", "ndtv.com", "hindustantimes.com",
    "reuters.com",
)

ROUTES = {
    # claim_type -> (extra keywords, sites to scope to)
    "chain_offer": (("scam", "fake", "fact check"), FACT_CHECKERS + PIB),
    "health":      (("fact check", "evidence"), HEALTH_SOURCES + FACT_CHECKERS),
    "policy":      (("official", "notification"), MONEY_SOURCES),
    "money":       (("official", "fact check"), MONEY_SOURCES + FACT_CHECKERS),
    "event":       (("news", "latest"), NEWS_SOURCES),
    "attribution": (("said", "statement"), NEWS_SOURCES + FACT_CHECKERS),
    "statistic":   (("data", "official figures"), MONEY_SOURCES + NEWS_SOURCES),
    "causal":      (("study", "evidence"), HEALTH_SOURCES + NEWS_SOURCES),
    "prediction":  (("news", "announcement"), NEWS_SOURCES),
    "generic":     (("fact check",), FACT_CHECKERS),
}

# Hindi equivalents of the routed keywords, for hi / hinglish claims.
HINDI_KEYWORDS = {
    "chain_offer": "फर्जी मैसेज सच्चाई",
    "health":      "स्वास्थ्य दावा सच्चाई",
    "policy":      "सरकारी आदेश सच्चाई",
    "money":       "फर्जी ऑफर सच्चाई",
    "event":       "खबर सच्चाई",
    "attribution": "बयान सच्चाई",
    "statistic":   "आंकड़े सच्चाई",
    "causal":      "अध्ययन सच्चाई",
    "prediction":  "घोषणा सच्चाई",
    "generic":     "फैक्ट चेक",
}

DEFAULT_ROUTE = ROUTES["generic"]

MAX_QUERIES = 3
MAX_QUERY_CHARS = 200

# Entity labels worth putting in a keyword query. URLs, phone numbers and
# UPI ids identify the scam but are useless as search terms — no
# fact-check article contains the particular number in *this* forward.
KEYWORD_LABELS = ("PERSON", "ORG", "GPE", "MONEY", "PERCENT", "NUMBER", "DATE", "PROPER")

_WS = re.compile(r"\s+")
_URLISH = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.IGNORECASE)


class Query(NamedTuple):
    """
    One question to ask.

    `kind` says which of the three roles it plays, `sites` is the domain
    scope a web search should apply (empty means unscoped), and
    `language` steers which API locale to ask in.
    """

    text: str
    kind: str                  # verbatim | keywords | routed
    language: str = "en"
    sites: tuple = ()
    claim_type: Optional[str] = None


def _clean(text):
    return _WS.sub(" ", _URLISH.sub(" ", text or "")).strip()


def _truncate(text, limit=MAX_QUERY_CHARS):
    text = _clean(text)

    if len(text) <= limit:
        return text

    return text[:limit].rsplit(" ", 1)[0]


def _entity_terms(claim):
    """Entity texts and numbers, de-duplicated, in the order they appear."""
    terms = []

    for entity in getattr(claim, "entities", []) or []:
        if entity.label not in KEYWORD_LABELS:
            continue

        term = _clean(entity.text)

        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)

    for number in getattr(claim, "numbers", []) or []:
        term = _clean(str(number)).replace("INR ", "Rs ")

        if term and term.lower() not in {t.lower() for t in terms}:
            terms.append(term)

    return terms


def _time_terms(claim):
    """
    Resolved dates a time-scoped query can use.

    A relative reference ("today", "kal") is only useful once stage 2 has
    resolved it to a date — the word itself matches every article ever
    written.
    """
    terms = []

    for ref in getattr(claim, "time_refs", []) or []:
        value = ref.normalized or ""

        if value and value not in terms:
            terms.append(value)

    return terms


def is_indic(claim):
    return getattr(claim, "language", "en") in ("hi", "hinglish")


def build_queries(claim, max_queries=MAX_QUERIES):
    """
    Two or three `Query` objects for one claim, most specific last.

    The verbatim query always comes first (it is the one a semantic index
    answers best), and duplicates are dropped — a short claim whose
    keyword query is the claim itself yields two queries, not two copies
    of one.
    """
    text = _clean(getattr(claim, "text", "") or "")

    if not text:
        return []

    claim_type = getattr(claim, "claim_type", "generic") or "generic"
    keywords, sites = ROUTES.get(claim_type, DEFAULT_ROUTE)
    language = "hi" if is_indic(claim) else "en"

    queries = [
        Query(_truncate(text), "verbatim", language, (), claim_type),
    ]

    terms = _entity_terms(claim)
    time_terms = _time_terms(claim)

    if terms:
        # Entities and numbers, plus the dates the claim pins itself to,
        # so a keyword engine scopes to the right week rather than to
        # every article that ever mentioned the same scheme.
        keyword_text = " ".join(terms[:6] + time_terms[:1])
        queries.append(
            Query(_truncate(keyword_text), "keywords", language, (), claim_type)
        )

    routed_text = " ".join([_truncate(text, 90)] + list(keywords))
    queries.append(Query(_truncate(routed_text), "routed", language, tuple(sites), claim_type))

    if is_indic(claim):
        hindi_text = " ".join([_truncate(text, 90), HINDI_KEYWORDS.get(claim_type, "फैक्ट चेक")])
        queries.append(Query(_truncate(hindi_text), "routed", "hi", tuple(sites), claim_type))

    seen = set()
    unique = []

    for query in queries:
        key = (query.text.lower(), query.language)

        if not query.text or key in seen:
            continue

        seen.add(key)
        unique.append(query)

    # The verbatim query is worth keeping whatever the cap; the routed one
    # is what actually reaches the fact-checkers, so trim from the middle.
    if len(unique) > max_queries:
        unique = [unique[0]] + unique[-(max_queries - 1):]

    return unique


def queries_for(claims, max_queries=MAX_QUERIES):
    """{claim_id: [Query, ...]} for an iterable of claims."""
    return {
        claim.id: build_queries(claim, max_queries=max_queries)
        for claim in claims
    }
