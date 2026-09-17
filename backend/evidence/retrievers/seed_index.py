"""
A local semantic index of rumours we already know about.

This is the retriever that works with no API keys, no network and no
budget — which makes it the one the demo actually runs on, and the one
the tests exercise. It holds a small corpus of known Indian
misinformation (`data/seed_factchecks.jsonl`), embeds each entry once,
and answers a claim with the closest entries above a similarity floor.

It is also the only retriever that can answer in milliseconds, which
matters for a bot: a forwarded UPI scam is recognised before the web
search has finished connecting.

Two things are deliberate:

  * **The embedder is the one stage 4 already loads.** It comes from
    `backend.stance.rank.embedder()`, which is `lru_cache`d — so the
    multilingual MiniLM lives in memory once, whether it was the ranker
    or this index that asked for it first. The model is multilingual, so
    a Hinglish claim matches an English seed entry.
  * **FAISS is optional.** With `faiss` installed the search uses an
    inner-product index; without it, the same normalised vectors are
    compared with NumPy. At thirty entries the difference is invisible,
    and the fallback keeps the demo installable anywhere.

Everything it returns is marked `demo=True` and carries the entry's
`demo_note`, so a verdict built on it can say out loud that it is
standing on demo data.
"""

import json
import logging
import os
from datetime import date
from functools import lru_cache

from ..schema import EvidenceCandidate


log = logging.getLogger(__name__)


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "seed_factchecks.jsonl",
)

CACHE_NAMESPACE = "seed_index"

TOP_K = 3

# Below this cosine similarity the "match" is noise. The multilingual
# MiniLM puts unrelated Indian news in the 0.2-0.35 band, so the floor
# sits above that.
SIMILARITY_FLOOR = 0.45


def index_path():
    return os.getenv("SEED_FACTCHECKS", DEFAULT_PATH)


def available():
    """True when the corpus is readable and an embedder can be loaded."""
    if not os.path.exists(index_path()):
        return False

    try:
        from ...stance import rank
    except ImportError:                              # fail soft, never raise
        return False

    return rank.available()


@lru_cache(maxsize=4)
def load_entries(path=None):
    """
    The seed corpus as a tuple of dicts, cached per path.

    A malformed line is skipped with a warning rather than taking the
    loader down — a corpus is data, and data has typos.
    """
    path = path or index_path()
    entries = []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as error:
                    log.warning("%s line %s is not valid JSON: %s", path, number, error)
    except OSError as error:                         # fail soft, never raise
        log.warning("could not read the seed index at %s: %s", path, error)
        return ()

    demo = sum(1 for entry in entries if entry.get("demo"))

    if demo:
        log.info(
            "seed index: %s entries loaded from %s, %s marked DEMO DATA "
            "(synthetic records for offline testing)",
            len(entries), path, demo,
        )

    return tuple(entries)


def _entry_text(entry):
    """
    What an entry is indexed *by*.

    The claim as people phrase it, plus its Hindi phrasing and its
    topics — not the summary, which is written about the claim and would
    pull in the vocabulary of debunking rather than of the rumour.
    """
    parts = [entry.get("claim") or "", entry.get("claim_hi") or ""]
    parts.extend(entry.get("topics") or [])

    return " ".join(part for part in parts if part).strip()


@lru_cache(maxsize=4)
def _vectors(path=None):
    """
    (entries, matrix) with the corpus embedded once per process.

    Returns (entries, None) when no embedder is available, which is the
    signal to skip the retriever rather than to fail.
    """
    entries = load_entries(path)

    if not entries:
        return (), None

    try:
        from ...stance import rank            # the shared, cached embedder

        if not rank.available():
            return entries, None

        vectors = rank.embed([_entry_text(entry) for entry in entries])
    except Exception as error:                       # fail soft, never raise
        log.warning("seed index embedding failed (%s): %s", type(error).__name__, error)
        return entries, None

    try:
        import numpy as np
    except ImportError:
        log.warning("numpy is not installed; the seed index is unavailable")
        return entries, None

    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return entries, matrix / norms


@lru_cache(maxsize=4)
def _faiss_index(path=None):
    """A FAISS inner-product index over the corpus, or None if unavailable."""
    _entries, matrix = _vectors(path)

    if matrix is None:
        return None

    try:
        import faiss
    except ImportError:
        log.debug("faiss is not installed; the seed index will use NumPy")
        return None

    try:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        return index
    except Exception as error:                       # fail soft, never raise
        log.warning("faiss index build failed (%s): %s", type(error).__name__, error)
        return None


def _query_vector(text):
    from ...stance import rank

    import numpy as np

    vector = np.asarray(rank.embed([text])[0], dtype="float32")
    norm = float(np.linalg.norm(vector)) or 1.0

    return (vector / norm).reshape(1, -1)


def _entry_to_candidate(entry, claim_id, query, similarity):
    # The entry's `domain` is the publisher it stands in for, and it is
    # what the credibility lookup keys on. The demo URLs deliberately
    # point at `example-demo.invalid`, so without this the whole corpus
    # would score as an unknown source and nothing could ever be
    # decisive. The `demo` flag travels with it, so nothing downstream
    # can mistake the record for the real publication.
    return EvidenceCandidate(
        claim_id=claim_id,
        source_type="seed_index",
        query=query,
        url=entry.get("url"),
        domain=entry.get("domain"),
        title=entry.get("claim"),
        snippet=entry.get("summary"),
        text=" ".join(
            part for part in (entry.get("claim"), entry.get("summary")) if part
        ),
        publisher=entry.get("publisher"),
        rating_raw=entry.get("rating_raw") or entry.get("rating"),
        published_date=entry.get("published_date"),
        retrieved_at=date.today().isoformat(),
        score=round(float(similarity), 4),
        language="hi" if entry.get("claim_hi") else "en",
        demo=bool(entry.get("demo")),
        meta={
            "seed_id": entry.get("id"),
            "topics": entry.get("topics") or [],
            "demo_note": entry.get("demo_note"),
            "backend": "faiss" if _faiss_index() is not None else "numpy",
        },
    )


def search(claim_id, query, top_k=TOP_K, floor=SIMILARITY_FLOOR, path=None):
    """
    The closest seed entries to a query, as candidates.

    Returns `[]` when the corpus is missing, the embedder is unavailable,
    or nothing clears the similarity floor.
    """
    text = (query or "").strip()

    if not text:
        return []

    entries, matrix = _vectors(path)

    if matrix is None or not entries:
        log.info("seed index unavailable (no embedder or no corpus); skipping")
        return []

    try:
        vector = _query_vector(text)
    except Exception as error:                       # fail soft, never raise
        log.warning("seed index query failed (%s): %s", type(error).__name__, error)
        return []

    wanted = min(top_k, len(entries))
    index = _faiss_index(path)

    if index is not None:
        scores, positions = index.search(vector, wanted)
        pairs = list(zip(positions[0].tolist(), scores[0].tolist()))
    else:
        import numpy as np

        similarities = (matrix @ vector[0])
        order = np.argsort(-similarities)[:wanted]
        pairs = [(int(position), float(similarities[position])) for position in order]

    found = []

    for position, similarity in pairs:
        if position < 0 or similarity < floor:
            continue

        found.append(
            _entry_to_candidate(entries[position], claim_id, text, similarity)
        )

    return found


def search_many(claim_id, queries, top_k=TOP_K, floor=SIMILARITY_FLOOR):
    """Run several queries for one claim and concatenate the results."""
    results = []

    for query in queries:
        results.extend(
            search(claim_id, getattr(query, "text", query), top_k=top_k, floor=floor)
        )

    return results


def reset():
    """Drop the cached corpus and vectors — used by the tests."""
    load_entries.cache_clear()
    _vectors.cache_clear()
    _faiss_index.cache_clear()
