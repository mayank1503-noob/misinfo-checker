"""
Reverse image search, via SerpAPI's Google Lens endpoint.

The local index only knows the pictures we put in it. A real forward
usually carries a picture nobody has catalogued, and the question is
still the same: where else has this appeared, and when was it first
published? That is what a reverse image search answers, and the *dates*
on the results are the finding — a photograph whose earliest match is
from 2015 is not a photograph of this week's flood.

SerpAPI needs a publicly reachable image URL; it cannot see a local file.
So this retriever only runs for images the packet knows a URL for
(`image["url"]` or `image["source_url"]`, set when a link or a social post
was ingested). A local upload with no URL is left to the local index and
the CLIP check — noted here because it is the common case for a WhatsApp
forward, and silently returning nothing would look like a bug.

With `SERPAPI_KEY` unset it logs once and returns nothing. Everything is
cached and fails soft.
"""

import logging
import re
from datetime import date

from ..common import http
from ..evidence.schema import EvidenceCandidate


log = logging.getLogger(__name__)


ENDPOINT = "https://serpapi.com/search"
CACHE_NAMESPACE = "reverse_image"
KEY_ENV = "SERPAPI_KEY"

ENGINE = "google_lens"
MAX_RESULTS = 8

# Dates as Lens tends to report them, in free text: "Aug 12, 2019",
# "12 Aug 2019", "2019-08-12".
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTHS = {
    month: index + 1
    for index, month in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
_TEXT_DATE = re.compile(
    r"\b(\d{1,2})\s+([a-z]{3})[a-z]*\.?,?\s+(\d{4})\b|"
    r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)


def available():
    return http.api_key(KEY_ENV) is not None


def parse_date(text):
    """An ISO date out of whatever a result called one, or None."""
    if not text:
        return None

    text = str(text)

    match = _ISO.search(text)

    if match:
        return match.group(0)

    match = _TEXT_DATE.search(text)

    if not match:
        return None

    if match.group(1):
        day, month, year = match.group(1), match.group(2), match.group(3)
    else:
        month, day, year = match.group(4), match.group(5), match.group(6)

    number = _MONTHS.get(month.lower()[:3])

    if not number:
        return None

    try:
        return date(int(year), number, int(day)).isoformat()
    except ValueError:
        return None


def _candidates_from(payload, claim_id, image_url, max_results=MAX_RESULTS):
    """Google Lens visual matches as evidence candidates, oldest first."""
    matches = (payload or {}).get("visual_matches") or []
    candidates = []
    today = date.today().isoformat()

    for match in matches[:max_results]:
        link = match.get("link")

        if not link:
            continue

        published = (
            parse_date(match.get("date"))
            or parse_date(match.get("snippet"))
            or parse_date(match.get("title"))
        )

        candidates.append(
            EvidenceCandidate(
                claim_id=claim_id,
                source_type="reverse_image",
                query=image_url,
                url=link,
                title=match.get("title"),
                snippet=match.get("snippet") or match.get("source"),
                publisher=match.get("source"),
                published_date=published,
                retrieved_at=today,
                meta={
                    "thumbnail": match.get("thumbnail"),
                    "image_url": image_url,
                    "engine": ENGINE,
                },
            )
        )

    # Oldest first: the earliest appearance is the one that matters, and
    # a caller that takes the first result should get the useful one.
    candidates.sort(key=lambda item: (item.published_date or "9999-99-99", item.id))

    return candidates


def search(claim_id, image_url, max_results=MAX_RESULTS):
    """
    Reverse-search one image URL. Returns `[]` on a missing key or failure.
    """
    key = http.api_key(KEY_ENV)

    if not key:
        log.info("%s is not set; skipping reverse image search", KEY_ENV)
        return []

    if not image_url:
        log.debug("no public URL for this image; reverse search needs one")
        return []

    payload = http.get_json(
        ENDPOINT,
        params={"engine": ENGINE, "url": image_url, "api_key": key},
        namespace=CACHE_NAMESPACE,
        # The key stays out of the cache key, as everywhere else.
        cache_parts=[ENGINE, image_url, max_results],
    )

    return _candidates_from(payload, claim_id, image_url, max_results=max_results)


def image_url_of(image):
    """
    The publicly reachable URL of a packet image, if it has one.

    A local path is not one: SerpAPI fetches the image itself, so a
    filesystem path would just fail on their side.
    """
    image = image or {}

    for field in ("url", "source_url", "remote_url"):
        value = image.get(field)

        if value and str(value).startswith(("http://", "https://")):
            return value

    path = image.get("path")

    if path and str(path).startswith(("http://", "https://")):
        return path

    return None


def first_seen(candidates):
    """The earliest publication date among matches, or None."""
    dates = [item.published_date for item in candidates if item.published_date]

    return min(dates) if dates else None
