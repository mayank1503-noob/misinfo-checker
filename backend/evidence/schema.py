"""
What a retriever returns (Stage 3b).

An `EvidenceCandidate` is one thing we found that might bear on one claim.
"Might" is the important word: this stage retrieves and normalises, and
deliberately does not judge. `stance` stays None until stage 4 reads the
text and decides, and the two jobs are kept apart because a retriever
that also guessed at stance would bury its guess inside a relevance
score, where nothing could audit it.

Two fields do come back from retrieval, because they are facts about the
*source* rather than readings of the text:

  * `rating` / `rating_raw` — a fact-check publisher's own verdict, kept
    both normalised and verbatim. Stage 4 lets this override the model.
  * `source_weight` — how much credibility the domain gets, from
    `config/sources.yaml`.

Ids are derived from the claim plus the canonical URL, so the same
article retrieved by two different queries, or by two different
retrievers, is one candidate with one id.
"""

import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field

from ..claims.schema import stable_id


SOURCE_TYPES = (
    "factcheck",      # Google Fact Check Tools / a fact-check publisher
    "web",            # a general web search result
    "seed_index",     # the local demo index of known rumours
    "reverse_image",  # an image match (stage 5)
    "image_caption",  # a caption/image consistency check (stage 5)
)

RATINGS = ("false", "misleading", "true", "unknown")


# Tracking parameters carry no meaning and would split one article into
# several candidates, so they are stripped before an id is minted.
_TRACKING_PARAMS = re.compile(
    r"^(utm_|fbclid$|gclid$|igshid$|mc_cid$|mc_eid$|ref$|ref_src$|s$|_ga$)",
    re.IGNORECASE,
)

_AMP_SUFFIX = re.compile(r"/amp/?$|\.amp$", re.IGNORECASE)


def canonical_url(url):
    """
    Reduce a URL to the thing that identifies the document.

    Lowercases the host, drops `www.`, the scheme's default port, the
    fragment, tracking parameters and an `/amp` suffix, normalises the
    scheme to https, and sorts what query parameters remain. Two links to
    the same article that differ only in how they were shared collapse to
    one string — which is what the dedupe in `normalize.py` and the
    candidate id both depend on.

    The scheme is normalised rather than preserved because `http://` and
    `https://` on the same host are the same document in every case that
    matters here, and keeping both would split one fact-check into two
    pieces of evidence that then "agree" with each other.
    """
    if not url:
        return ""

    text = str(url).strip()

    if not text:
        return ""

    if "//" not in text:
        text = "https://" + text

    try:
        parts = urlsplit(text)
    except ValueError:
        return text.lower()

    host = (parts.hostname or "").lower()

    if host.startswith("www."):
        host = host[4:]

    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"

    path = _AMP_SUFFIX.sub("", parts.path or "")

    if path.endswith("/") and len(path) > 1:
        path = path[:-1]

    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not _TRACKING_PARAMS.match(key)
        )
    )

    scheme = "https" if parts.scheme.lower() in ("http", "https", "") else parts.scheme.lower()

    return urlunsplit((scheme, host, path, query, ""))


def domain_of(url):
    """The registrable-ish host of a URL, without `www.`, or ''."""
    if not url:
        return ""

    try:
        host = (urlsplit(canonical_url(url)).hostname or "").lower()
    except ValueError:
        return ""

    return host


def candidate_id(claim_id, source_type, url=None, title=None, snippet=None):
    """
    Deterministic id for one candidate.

    Keyed on the claim as well as the document: the same article
    retrieved for two different claims is two candidates, because the
    stance stage will judge it separately against each. Where there is no
    URL (the seed index, an image match) the title and snippet stand in.
    """
    fingerprint = canonical_url(url) or f"{title or ''}|{(snippet or '')[:200]}"

    return stable_id("ev", claim_id or "", source_type, fingerprint)


class EvidenceCandidate(BaseModel):
    """
    One retrieved item, before anyone has decided what it means.

    `decisive` marks the narrow case the verdict trusts on its own: a
    fact-checker's rating of a claim that closely matches this one. It is
    set in `normalize.py`, never by a retriever.
    """

    id: str = ""
    claim_id: str
    source_type: str                              # one of SOURCE_TYPES
    query: Optional[str] = None                   # what we asked to find it

    url: Optional[str] = None
    domain: Optional[str] = None                  # derived from `url`
    title: Optional[str] = None
    snippet: Optional[str] = None
    text: Optional[str] = None                    # full article body, if fetched

    publisher: Optional[str] = None
    rating: Optional[str] = None                  # normalised: one of RATINGS
    rating_raw: Optional[str] = None              # exactly what the publisher said
    published_date: Optional[str] = None          # ISO, when known
    retrieved_at: Optional[str] = None            # ISO date we fetched it

    source_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    decisive: bool = False
    stance: Optional[str] = None                  # stage 4 fills this in

    language: Optional[str] = None
    score: Optional[float] = None                 # retriever's own relevance, if any
    demo: bool = False                            # from the demo seed index
    meta: dict = {}

    def model_post_init(self, _context):
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"unknown source_type {self.source_type!r}; expected one of {SOURCE_TYPES}"
            )

        if self.rating is not None and self.rating not in RATINGS:
            raise ValueError(
                f"unknown rating {self.rating!r}; expected one of {RATINGS}"
            )

        if self.url and not self.domain:
            self.domain = domain_of(self.url)

        if self.url:
            self.url = canonical_url(self.url)

        if not self.id:
            self.id = candidate_id(
                self.claim_id, self.source_type,
                url=self.url, title=self.title, snippet=self.snippet,
            )

    @property
    def body(self):
        """Title, snippet and full text joined, longest-wins, for stage 4."""
        parts = [p.strip() for p in (self.title, self.snippet, self.text) if (p or "").strip()]
        kept = []

        for index, part in enumerate(parts):
            if any(part in longer and part != longer for longer in parts[index + 1:]):
                continue

            if part not in kept:
                kept.append(part)

        return "\n".join(kept)

    def to_stance_input(self):
        """The flat shape `backend.stance` consumes."""
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "title": self.title,
            "snippet": self.snippet,
            "text": self.text,
            "rating": self.rating_raw or self.rating,
            "weight": self.source_weight,
        }
