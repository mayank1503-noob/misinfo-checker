"""
A local index of pictures we have seen before.

Recycled media is the cheapest misinformation there is: take a real
photograph of a real disaster and re-caption it as today's. Catching it
needs only one thing — knowing that this picture is older than the story.

So this is a nearest-neighbour index over DINOv2 embeddings of known
images, each with the metadata that makes the finding usable:

    source_url       where it was published
    first_seen       the earliest date we know it existed
    context          what it actually shows

A packet image that matches an entry above `SIMILARITY_FLOOR` is the same
picture, and the entry's `first_seen` is then evidence about the claim's
date — which stage 5's collector turns into a DUPLICATE_OF edge and, when
the gap is wide enough, a DATE_MISMATCH.

Three things this deliberately does not do:

  * **It does not embed packet images.** Stage 1 already computed a
    DINOv2 vector for every image and keyframe; this module consumes
    those. The only time it loads DINOv2 is when *building* the index
    from files on disk, and it does that through
    `backend.analyzers.models.dinov2()` — the same `lru_cache`d loader
    stage 1 uses, so there is never a second copy in memory.
  * **It does not require FAISS.** With `faiss` installed the search uses
    an inner-product index; without it, NumPy does the same arithmetic.
  * **It does not pretend its contents are real.** Everything in
    `data/seed_images/` is demo data, and every match carries the flag.

Build it with `scripts/build_image_index.py`.
"""

import json
import logging
import os
from functools import lru_cache


log = logging.getLogger(__name__)


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_IMAGE_DIR = os.path.join(ROOT, "data", "seed_images")
DEFAULT_INDEX_PATH = os.path.join(ROOT, "data", "image_index.json")

METADATA_NAME = "metadata.jsonl"

# DINOv2 cosine for the same picture stays very high even after it has
# been recompressed on its way through three chat apps: measured at 0.98
# for a JPEG re-encode at quality 70. Distinct images - even synthetic
# ones built to look alike - peak around 0.92. The floor sits in that
# gap, and is deliberately nearer the top of it: a missed match costs one
# finding, while a false match tells someone a real photograph is
# recycled, which is the more expensive mistake.
SIMILARITY_FLOOR = 0.95

TOP_K = 3


def image_dir():
    return os.getenv("SEED_IMAGE_DIR", DEFAULT_IMAGE_DIR)


def index_path():
    return os.getenv("IMAGE_INDEX", DEFAULT_INDEX_PATH)


def metadata_path(directory=None):
    return os.path.join(directory or image_dir(), METADATA_NAME)


def load_metadata(directory=None):
    """{filename: record} from the seed directory's metadata.jsonl."""
    path = metadata_path(directory)
    records = {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    log.warning("%s line %s is not valid JSON: %s", path, number, error)
                    continue

                if record.get("file"):
                    records[record["file"]] = record
    except OSError as error:                         # fail soft, never raise
        log.info("no image metadata at %s (%s)", path, error)

    return records


def embed_file(path):
    """
    The DINOv2 embedding of one file on disk, or None.

    Used only when building the index. It goes through
    `backend.analyzers.image.embed`, which uses the cached loader in
    `backend.analyzers.models`, so the weights are shared with stage 1.
    """
    try:
        from ..analyzers import image as image_analyzer

        return image_analyzer.embed(path)
    except Exception as error:                       # fail soft, never raise
        log.warning("could not embed %s (%s): %s", path, type(error).__name__, error)
        return None


def build(directory=None, output=None):
    """
    Embed every image in `directory` and write the index.

    Returns the number of entries written. Files with no metadata record
    are still indexed — a match on an unlabelled picture is less useful
    than a labelled one, but it is not nothing.
    """
    directory = directory or image_dir()
    output = output or index_path()

    if not os.path.isdir(directory):
        log.warning("no image directory at %s; nothing to index", directory)
        return 0

    metadata = load_metadata(directory)
    entries = []

    for name in sorted(os.listdir(directory)):
        if name == METADATA_NAME or name.startswith("."):
            continue

        path = os.path.join(directory, name)

        if not os.path.isfile(path):
            continue

        vector = embed_file(path)

        if not vector:
            continue

        record = dict(metadata.get(name) or {})
        record.setdefault("file", name)
        record.setdefault("demo", True)
        record["embedding"] = [float(value) for value in vector]

        entries.append(record)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle, ensure_ascii=False)

    log.info("image index: %s entries written to %s", len(entries), output)

    reset()

    return len(entries)


@lru_cache(maxsize=4)
def _index(path=None):
    """(entries without vectors, normalised matrix) or ((), None)."""
    path = path or index_path()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError:
        log.info("no image index at %s; run scripts/build_image_index.py", path)
        return (), None
    except json.JSONDecodeError as error:            # fail soft, never raise
        log.warning("image index at %s is corrupt: %s", path, error)
        return (), None

    raw = data.get("entries") or []

    if not raw:
        return (), None

    try:
        import numpy as np
    except ImportError:
        log.warning("numpy is not installed; the image index is unavailable")
        return (), None

    vectors = [entry.get("embedding") or [] for entry in raw]
    width = max((len(vector) for vector in vectors), default=0)

    entries = []
    kept = []

    for entry, vector in zip(raw, vectors):
        if len(vector) != width or not width:
            log.warning("skipping %s: embedding is the wrong size", entry.get("file"))
            continue

        entries.append({k: v for k, v in entry.items() if k != "embedding"})
        kept.append(vector)

    if not kept:
        return (), None

    matrix = np.asarray(kept, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    demo = sum(1 for entry in entries if entry.get("demo"))

    if demo:
        log.info(
            "image index: %s entries, %s marked DEMO DATA (synthetic or "
            "placeholder images for offline testing)",
            len(entries), demo,
        )

    return tuple(entries), matrix / norms


@lru_cache(maxsize=4)
def _faiss_index(path=None):
    _entries, matrix = _index(path)

    if matrix is None:
        return None

    try:
        import faiss
    except ImportError:
        log.debug("faiss is not installed; the image index will use NumPy")
        return None

    try:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        return index
    except Exception as error:                       # fail soft, never raise
        log.warning("faiss image index build failed (%s): %s", type(error).__name__, error)
        return None


def available(path=None):
    _entries, matrix = _index(path)

    return matrix is not None


def search(embedding, top_k=TOP_K, floor=SIMILARITY_FLOOR, path=None):
    """
    Entries matching an image embedding, closest first.

    `embedding` is the vector stage 1 already put on the packet image —
    nothing is encoded here. Returns a list of the metadata records with
    a `similarity` added, or `[]`.
    """
    if not embedding:
        return []

    entries, matrix = _index(path)

    if matrix is None:
        return []

    try:
        import numpy as np

        vector = np.asarray(embedding, dtype="float32")

        if vector.shape[0] != matrix.shape[1]:
            log.warning(
                "image embedding is %s-dimensional but the index is %s-dimensional; "
                "was the index built with a different model?",
                vector.shape[0], matrix.shape[1],
            )
            return []

        norm = float(np.linalg.norm(vector)) or 1.0
        vector = (vector / norm).reshape(1, -1)
    except Exception as error:                       # fail soft, never raise
        log.warning("image search failed (%s): %s", type(error).__name__, error)
        return []

    wanted = min(top_k, len(entries))
    index = _faiss_index(path)

    if index is not None:
        scores, positions = index.search(vector, wanted)
        pairs = list(zip(positions[0].tolist(), scores[0].tolist()))
        backend = "faiss"
    else:
        similarities = matrix @ vector[0]
        order = np.argsort(-similarities)[:wanted]
        pairs = [(int(position), float(similarities[position])) for position in order]
        backend = "numpy"

    matches = []

    for position, similarity in pairs:
        if position < 0 or similarity < floor:
            continue

        matches.append(
            {**entries[position], "similarity": round(float(similarity), 4),
             "backend": backend}
        )

    return matches


def reset():
    """Drop the cached index — used by the tests and after a rebuild."""
    _index.cache_clear()
    _faiss_index.cache_clear()
