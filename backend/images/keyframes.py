"""
Thinning a video down to the frames worth checking.

Stage 1 samples a keyframe every five seconds. For a talking-head clip or
a still image set to music, that produces six near-identical pictures,
and every later stage then pays six times over: six reverse-image
lookups, six CLIP passes, six sets of near-duplicate evidence nodes
saying the same thing.

So frames are deduplicated before any of that happens, using the DINOv2
embedding stage 1 *already computed* — nothing is re-encoded here, and no
model is loaded. A frame is dropped when its cosine similarity to a frame
already kept is above `SIMILARITY_THRESHOLD`; what survives is a set of
visually distinct frames, and the count is capped so a long video cannot
flood the pipeline.

The first frame of each run is the one kept, because a forward's first
frame is the one that gets screenshotted and re-shared.
"""

import logging
import math


log = logging.getLogger(__name__)


# Above this cosine the two frames are the same shot. DINOv2 embeddings
# of genuinely different scenes sit well below it.
SIMILARITY_THRESHOLD = 0.95

MAX_IMAGES = 8


def cosine(left, right):
    if not left or not right:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))

    if norm == 0.0:
        return 0.0

    return dot / norm


def select_keyframes(images, threshold=SIMILARITY_THRESHOLD, cap=MAX_IMAGES):
    """
    The visually distinct images from a packet, in order.

    Images with no embedding are kept as-is: an image that could not be
    embedded may still have OCR text and an EXIF date worth checking, and
    silently dropping it would lose that. Returns a new list; the input
    is not modified.
    """
    kept = []
    kept_vectors = []
    dropped = 0

    for image in images or []:
        if len(kept) >= cap:
            dropped += 1
            continue

        vector = (image or {}).get("embedding") or []

        if not vector:
            kept.append(image)
            continue

        duplicate = False

        for existing in kept_vectors:
            if cosine(vector, existing) > threshold:
                duplicate = True
                break

        if duplicate:
            dropped += 1
            continue

        kept.append(image)
        kept_vectors.append(vector)

    if dropped:
        log.info(
            "keyframes: %s of %s frames dropped as near-duplicates or over the cap of %s",
            dropped, len(images or []), cap,
        )

    return kept


def dedupe_packet(packet, threshold=SIMILARITY_THRESHOLD, cap=MAX_IMAGES):
    """
    The same packet with its near-duplicate keyframes removed.

    Returns a shallow copy so the caller's packet is untouched — the
    original frame list is sometimes worth keeping for debugging, and a
    function that quietly mutates its input is a bad neighbour to a
    pipeline that hands the same packet to several stages.
    """
    images = (packet or {}).get("images") or []

    if not images:
        return packet

    trimmed = dict(packet)
    trimmed["images"] = select_keyframes(images, threshold=threshold, cap=cap)

    return trimmed
