"""
Making candidates from different retrievers comparable.

Three retrievers, three vocabularies: Google Fact Check returns whatever
string the publisher typed ("Pants on Fire", "मिथ्या दावा", "Miscaptioned"),
a web search returns no rating at all, and the seed index returns its own.
Before any of it can be weighed together it has to be normalised:

  * **ratings** collapse to `false` / `misleading` / `true` / `unknown`,
    with the publisher's exact words kept in `rating_raw` — stage 4 shows
    the original, and a mapping error stays visible rather than silently
    becoming the verdict;
  * **domains** get a credibility weight from `config/sources.yaml`;
  * **duplicates** collapse by canonical URL, keeping the richest copy;
  * and the list is **capped**, so one prolific domain cannot crowd out
    the one article that settles the claim.

`decisive` is set here and nowhere else. It marks the narrow case a
verdict may act on alone — a fact-checker whose rating is `false` or
`misleading` *and* whose text overlaps the claim enough that it is
plainly about the same thing. Everything else has to win on weight.

The overlap test here is one-directional (how much of the claim the
article repeats) and reads the whole article, which is enough to keep an
unrelated page out but not enough to tell one rumour from its neighbour:
the benign "boiling water reduces waterborne disease" scores 0.44
against the hot-water-cures-COVID debunk, a hair under the 0.45 floor
(DECISIONS.md O6). So `backend.aboutness` is asked as well, and a
fact-check that reviews a *different* claim in the same topic cannot be
decisive however its wording scores. Stage 4 applies the same gate to
the rating itself; this is the half of it that keeps the verdict's
first rule from firing.
"""

import functools
import logging
import os
import re

from .. import aboutness
from .schema import RATINGS, canonical_url, domain_of


log = logging.getLogger(__name__)


CONFIG_PATH = os.getenv(
    "SOURCES_CONFIG",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "config", "sources.yaml"),
)

# Used when the config file is missing or unreadable, so the pipeline
# still runs (with everything treated as an unknown source).
FALLBACK_TIERS = {
    "factchecker": 1.0, "government": 0.9, "news": 0.8, "other": 0.5, "blog": 0.3,
}

MAX_PER_CLAIM = 10

# How much of the claim's wording a fact-check has to share before its
# rating is allowed to settle the claim by itself.
DECISIVE_OVERLAP = 0.45

# Publisher verdict strings. Mixed verdicts are matched first so that
# "half true" and "partly false" do not read as a clean "true"/"false".
RATING_PATTERNS = (
    ("misleading", (
        "misleading", "miscaptioned", "missing context", "lacks context",
        "out of context", "partly", "partially", "half true", "half-true",
        "mixture", "mixed", "exaggerat", "distorted", "altered", "edited",
        "manipulated", "doctored", "old video", "old photo", "old post",
        "unproven", "unverified", "unsubstantiated", "overstated",
        "भ्रामक", "अधूरा सच", "गुमराह",
    )),
    ("false", (
        "false", "fake", "hoax", "scam", "debunked", "no evidence",
        "incorrect", "untrue", "not true", "baseless", "misinformation",
        "disinformation", "pants on fire", "fabricated", "satire",
        "फर्जी", "झूठ", "गलत", "मिथ्या", "अफवाह",
    )),
    ("true", (
        "mostly true", "true", "correct", "accurate", "verified", "genuine",
        "confirmed", "legit", "authentic",
        "सही", "सच",
    )),
)

_WORD = re.compile(r"\w+", re.UNICODE)

# Words too common to say two texts are about the same thing.
_STOP = {
    "the", "a", "an", "of", "in", "on", "is", "are", "was", "were", "to", "and",
    "that", "this", "it", "for", "has", "have", "had", "by", "at", "from",
    "will", "be", "been", "no", "not", "you", "your", "with", "as",
    "है", "हैं", "का", "की", "के", "को", "में", "से", "और", "पर", "यह", "ने",
}


def normalize_rating(raw):
    """
    (normalised rating, matched pattern) for a publisher's verdict string.

    Returns ("unknown", None) for an absent or unrecognised rating rather
    than guessing: an unrecognised verdict must not quietly become
    "true".
    """
    text = (raw or "").strip().lower()

    if not text:
        return "unknown", None

    for rating, patterns in RATING_PATTERNS:
        for pattern in patterns:
            if pattern in text:
                return rating, pattern

    log.debug("unrecognised rating %r", raw)

    return "unknown", None


@functools.lru_cache(maxsize=1)
def _config(path=None):
    """The parsed sources.yaml, cached; falls back to the built-in tiers."""
    path = path or CONFIG_PATH

    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as error:                        # fail soft, never raise
        log.warning("could not read %s (%s); using fallback tiers", path, error)
        data = {}

    tiers = {**FALLBACK_TIERS, **(data.get("tiers") or {})}
    default_tier = data.get("default_tier", "other")

    # domain -> tier, so a lookup is one dict hit rather than a scan.
    index = {}

    for tier, domains in (data.get("domains") or {}).items():
        for domain in domains or []:
            index[str(domain).strip().lower()] = tier

    return {"tiers": tiers, "default_tier": default_tier, "index": index}


def reload_config():
    """Drop the cached config — used by the tests, and after editing it."""
    _config.cache_clear()


def tier_for(domain_or_url):
    """
    The credibility tier of a domain.

    A subdomain inherits its parent's tier (`factcheck.pib.gov.in` ->
    `pib.gov.in` -> factchecker) unless it is listed in its own right, so
    the config does not have to enumerate every ministry's subdomain.
    """
    config = _config()
    domain = (domain_of(domain_or_url) or str(domain_or_url or "")).strip().lower()

    if domain.startswith("www."):
        domain = domain[4:]

    if not domain:
        return config["default_tier"]

    index = config["index"]

    # Longest match first: a path-qualified entry ("thequint.com/webqoof")
    # is more specific than the bare domain.
    for listed, tier in sorted(index.items(), key=lambda pair: -len(pair[0])):
        host = listed.split("/", 1)[0]

        if domain == host or domain.endswith("." + host):
            return tier

    return config["default_tier"]


def weight_for(domain_or_url):
    """The credibility weight (0-1) of a domain."""
    config = _config()
    tier = tier_for(domain_or_url)

    return float(config["tiers"].get(tier, config["tiers"].get("other", 0.5)))


def _tokens(text):
    return {
        token for token in _WORD.findall((text or "").lower())
        if token not in _STOP and len(token) > 1
    }


def claim_overlap(claim_text, candidate):
    """
    How much of the claim's vocabulary the candidate repeats, 0-1.

    Measured against the claim (not the article) on purpose: a long
    fact-check that covers the claim in one paragraph still scores high,
    while a short snippet about something else does not.
    """
    claim_tokens = _tokens(claim_text)

    if not claim_tokens:
        return 0.0

    text = " ".join(
        part for part in (candidate.title, candidate.snippet, candidate.text) if part
    )
    candidate_tokens = _tokens(text)

    if not candidate_tokens:
        return 0.0

    return round(len(claim_tokens & candidate_tokens) / len(claim_tokens), 4)


def about_the_claim(candidate, claim_text):
    """
    `backend.aboutness`'s answer for this candidate, as a plain string.

    Both the fields a retriever might put the claim under review in are
    offered; see `aboutness.judge_any`.
    """
    try:
        return aboutness.judge_any(
            claim_text, candidate.title, candidate.snippet
        ).answer
    except Exception as error:                       # fail soft, never raise
        log.warning("aboutness failed (%s): %s", type(error).__name__, error)

        return aboutness.UNCLEAR


def is_decisive(candidate, claim_text, overlap=None, about=None):
    """
    Whether this candidate can settle the claim on its own.

    Four conditions, all required: it is a fact-check, its publisher
    rated the claim `false` or `misleading`, its text is close enough to
    the claim that the rating is plainly about *this* claim, and the
    claim it says it reviews is not a different claim in the same topic.

    `overlap` and `about` are accepted already-computed, because
    `normalize_candidates` records both on the candidate anyway.
    """
    if candidate.source_type not in ("factcheck", "seed_index"):
        return False

    if candidate.rating not in ("false", "misleading"):
        return False

    if weight_for(candidate.domain or "") < 0.9:
        return False

    if overlap is None:
        overlap = claim_overlap(claim_text, candidate)

    if overlap < DECISIVE_OVERLAP:
        return False

    if about is None:
        about = about_the_claim(candidate, claim_text)

    return about != aboutness.NOT_ABOUT


def _richness(candidate):
    """How much a copy of a document carries — used to pick the survivor."""
    return (
        len(candidate.text or ""),
        len(candidate.snippet or ""),
        1 if candidate.rating and candidate.rating != "unknown" else 0,
        1 if candidate.published_date else 0,
    )


def dedupe(candidates):
    """
    Collapse candidates that point at the same document.

    Keyed by canonical URL, falling back to (claim, title) for items that
    have none. The richest copy survives, but it inherits anything the
    others had and it lacked — a snippet from the web search, a rating
    from the fact-check API — because two retrievers finding the same
    article usually each know something the other does not.
    """
    survivors = {}
    order = []

    for candidate in candidates:
        key = (
            candidate.claim_id,
            canonical_url(candidate.url) or f"title:{(candidate.title or '').lower()}",
        )

        if key not in survivors:
            survivors[key] = candidate
            order.append(key)
            continue

        kept = survivors[key]
        loser = candidate

        if _richness(candidate) > _richness(kept):
            kept, loser = candidate, kept
            survivors[key] = kept

        for field in ("text", "snippet", "title", "publisher", "published_date",
                      "rating_raw", "language"):
            if not getattr(kept, field, None) and getattr(loser, field, None):
                setattr(kept, field, getattr(loser, field))

        if (kept.rating in (None, "unknown")) and loser.rating not in (None, "unknown"):
            kept.rating = loser.rating

        kept.decisive = kept.decisive or loser.decisive
        kept.demo = kept.demo or loser.demo

        # Keep a trace of the other ways this document was found: useful
        # when a verdict has to explain why it trusted it.
        also = kept.meta.setdefault("also_found_by", [])

        if loser.source_type not in also and loser.source_type != kept.source_type:
            also.append(loser.source_type)

    return [survivors[key] for key in order]


def rank_key(candidate):
    """
    Sort key for what to keep when there are too many candidates.

    Decisive fact-checks first, then source credibility, then the
    retriever's own relevance, then having a full text at all.
    """
    return (
        0 if candidate.decisive else 1,
        -float(candidate.source_weight or 0.0),
        -float(candidate.score or 0.0),
        0 if candidate.text else 1,
        candidate.id,
    )


def normalize_candidates(candidates, claim_text="", cap=MAX_PER_CLAIM):
    """
    Normalise, dedupe, mark decisive and cap one claim's candidates.

    Returns a new list, strongest first. Every retriever's output goes
    through this before it reaches the graph, so the graph never sees a
    publisher's raw rating string or two copies of one article.
    """
    prepared = []

    for candidate in candidates:
        if candidate.url:
            candidate.url = canonical_url(candidate.url)
            candidate.domain = candidate.domain or domain_of(candidate.url)

        if candidate.rating_raw and (
            candidate.rating is None or candidate.rating == "unknown"
        ):
            candidate.rating, _pattern = normalize_rating(candidate.rating_raw)

        if candidate.rating is not None and candidate.rating not in RATINGS:
            candidate.rating = "unknown"

        candidate.source_weight = weight_for(candidate.domain or candidate.url or "")

        prepared.append(candidate)

    unique = dedupe(prepared)

    for candidate in unique:
        overlap = claim_overlap(claim_text, candidate)
        candidate.meta["claim_overlap"] = overlap
        about = about_the_claim(candidate, claim_text)
        candidate.meta["aboutness"] = about
        candidate.decisive = is_decisive(
            candidate, claim_text, overlap=overlap, about=about
        )

    unique.sort(key=rank_key)

    return unique[:cap] if cap else unique
