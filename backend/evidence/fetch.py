"""
Getting the full text of the most promising results.

A search snippet is two sentences chosen to sell a click. That is enough
to rank a result and often enough to read a rating off, but it is thin
evidence for an entailment model: the sentence that actually refutes the
claim is usually in the third paragraph, not the teaser.

So the top few candidates per claim get fetched and run through
trafilatura, which is already a dependency (stage 1 uses it for link
ingestion) and which is good at throwing away navigation, cookie banners
and comment sections. It also reads the publication date out of the page,
which matters here: a "breaking" story dated 2019 is the whole finding
for recycled content.

Only the top `MAX_FETCH` are fetched, because this is the slowest step in
the pipeline and the tail of a result list rarely changes a verdict.
Everything is cached and fails soft — a page that will not load leaves
the candidate with its snippet and nothing else.
"""

import logging

from ..common import cache, http


log = logging.getLogger(__name__)


CACHE_NAMESPACE = "article"

MAX_FETCH = 3
MIN_TEXT_CHARS = 200

# Long enough for stage 4 to find the relevant passage, short enough that
# a graph carrying a dozen of them stays a reasonable size.
MAX_TEXT_CHARS = 20000


def available():
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        return False

    return True


def _extract(html, url):
    """(text, date) from a downloaded page, either of which may be None."""
    try:
        import trafilatura
    except ImportError:                              # fail soft, never raise
        log.info("trafilatura is not installed; skipping full-text fetch")
        return None, None

    text = None
    published = None

    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except Exception as error:                       # fail soft, never raise
        log.warning("extraction failed for %s (%s): %s", url, type(error).__name__, error)

    try:
        from htmldate import find_date

        published = find_date(html, url=url, outputformat="%Y-%m-%d")
    except ImportError:
        log.debug("htmldate is not installed; no publication date for %s", url)
    except Exception as error:                       # fail soft, never raise
        log.debug("date extraction failed for %s (%s): %s", url, type(error).__name__, error)

    return text, published


def fetch_article(url):
    """
    {"text", "published_date"} for one URL, or None.

    The *extracted* result is cached rather than the raw HTML: it is a
    twentieth of the size, and re-extracting on every run would undo the
    point of caching.
    """
    if not url:
        return None

    def download():
        html = http.get_text(url, namespace=None)

        if not html:
            return None

        text, published = _extract(html, url)

        if not text or len(text) < MIN_TEXT_CHARS:
            log.debug("no usable article text at %s", url)
            return None

        return {"text": text[:MAX_TEXT_CHARS], "published_date": published}

    return cache.memoize(CACHE_NAMESPACE, ["article", url], download)


def enrich(candidates, limit=MAX_FETCH):
    """
    Fetch full text for the first `limit` candidates that lack it.

    Mutates and returns the same list — order is the caller's ranking,
    and re-sorting here would quietly change which evidence the verdict
    sees first. Candidates that already carry text (Tavily can return it)
    are left alone and do not count against the limit.
    """
    fetched = 0

    for candidate in candidates:
        if fetched >= limit:
            break

        if candidate.text and len(candidate.text) >= MIN_TEXT_CHARS:
            continue

        if not candidate.url:
            continue

        article = fetch_article(candidate.url)

        fetched += 1

        if not article:
            continue

        candidate.text = article["text"]

        if not candidate.published_date and article.get("published_date"):
            candidate.published_date = article["published_date"]

        candidate.meta["fetched"] = True

    return candidates
