"""
Does the picture actually show what the message says it shows?

The commonest form of visual misinformation is not a forged image — it is
a real photograph with a false caption. A 2015 flood photo captioned as
today's flood, a crowd at a religious festival captioned as a protest, a
burning building in another country captioned as a local riot. Nothing
about the pixels is fake; the lie is the pairing.

So this module scores the pairing directly, with CLIP, which embeds
images and text into one space:

    clip-ViT-B-32                    the image side
    clip-ViT-B-32-multilingual-v1    the text side, so a Hindi or
                                     Hinglish claim can be scored too

A claim whose CLIP similarity to the image it was attached to falls below
`MIN_SIMILARITY` is evidence *against* that claim: the picture being used
to sell it does not depict it. That becomes an evidence node with stance
`refutes` and the `misleading` flag, method `clip`.

The BLIP description stage 1 already produced is used as a second
opinion. Where CLIP is weak (dense text, screenshots), a claim that has
nothing in common with what the captioner saw is the more legible signal,
and it costs nothing — the description is already on the packet.

Both models are lazy and `lru_cache`d, and every failure path returns
"no opinion" rather than a false accusation: an unreadable image, a
missing model or an exception all yield no evidence at all. Accusing a
truthful caption of being miscaptioned is worse than staying quiet.
"""

import logging
import os
from functools import lru_cache


log = logging.getLogger(__name__)


IMAGE_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "clip-ViT-B-32-multilingual-v1"

# CLIP similarities are compressed: a good caption/image pair sits around
# 0.25-0.35, an unrelated pair near 0.1. The floor is deliberately low,
# because this flag accuses a caption of lying.
MIN_SIMILARITY = 0.2

# How unlike the model's own description of the picture a claim has to be
# before that counts as corroboration of a mismatch.
MIN_DESCRIPTION_SIMILARITY = 0.25


def image_model_name():
    return os.getenv("CLIP_IMAGE_MODEL", IMAGE_MODEL)


def text_model_name():
    return os.getenv("CLIP_TEXT_MODEL", TEXT_MODEL)


def threshold():
    try:
        return float(os.getenv("CLIP_MIN_SIMILARITY", MIN_SIMILARITY))
    except ValueError:
        return MIN_SIMILARITY


def available():
    try:
        import sentence_transformers  # noqa: F401
        import PIL                    # noqa: F401
    except ImportError:
        return False

    return True


@lru_cache
def image_encoder():
    """CLIP's image tower — cached, so one copy serves the whole process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(image_model_name())


@lru_cache
def text_encoder():
    """
    The multilingual text tower.

    A separate checkpoint from the image one: `clip-ViT-B-32-multilingual-v1`
    is trained to put fifty languages into the *same* space as
    `clip-ViT-B-32`'s images, which is what makes a Hindi claim scorable
    against an English-trained image encoder.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(text_model_name())


def _cosine(left, right):
    import numpy as np

    left = np.asarray(left, dtype="float32")
    right = np.asarray(right, dtype="float32")

    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))

    if denominator == 0.0:
        return 0.0

    return float(np.dot(left, right) / denominator)


def score_texts_against_image(path, texts):
    """
    CLIP similarity of each text to the image at `path`.

    Returns a list of floats in the same order, or `None` — meaning "no
    opinion" — when the models or the file are unavailable. `None` is
    distinct from a list of low scores: one means we could not look, the
    other means we looked and they do not match.
    """
    texts = [text for text in (texts or []) if (text or "").strip()]

    if not path or not texts:
        return None

    if not available():
        log.info("sentence-transformers or Pillow is missing; skipping the CLIP check")
        return None

    try:
        from PIL import Image

        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image_vector = image_encoder().encode(image)

        text_vectors = text_encoder().encode(texts)
    except FileNotFoundError:
        log.info("image %s is gone; skipping the CLIP check", path)
        return None
    except Exception as error:                       # fail soft, never raise
        log.warning("CLIP scoring failed for %s (%s): %s", path, type(error).__name__, error)
        return None

    return [round(_cosine(image_vector, vector), 4) for vector in text_vectors]


def score_texts(left, right):
    """
    Similarity between two pieces of text in CLIP's text space.

    Used to compare a claim with the model-generated description of the
    picture — both are text, so only the text tower is needed.
    """
    if not (left or "").strip() or not (right or "").strip():
        return None

    if not available():
        return None

    try:
        vectors = text_encoder().encode([left, right])
    except Exception as error:                       # fail soft, never raise
        log.warning("CLIP text scoring failed (%s): %s", type(error).__name__, error)
        return None

    return round(_cosine(vectors[0], vectors[1]), 4)


def check_image(path, claims, description=None, minimum=None):
    """
    Score one image against the claims it is being used to support.

    `claims` is [(claim_id, text)]. Returns one finding per claim:

        {claim_id, similarity, description_similarity, mismatch, reason}

    `mismatch` is True only when CLIP actually looked and the similarity
    came back below the floor. An unavailable model produces findings
    with `similarity=None` and `mismatch=False`, so a caller can tell
    "checked and fine" from "could not check".
    """
    minimum = threshold() if minimum is None else minimum
    texts = [text for _claim_id, text in claims]

    similarities = score_texts_against_image(path, texts)
    findings = []

    for index, (claim_id, text) in enumerate(claims):
        similarity = similarities[index] if similarities else None
        description_similarity = None

        if similarity is not None and similarity < minimum and description:
            # Only worth the second encode when CLIP has already flagged
            # it: this is corroboration, not a first opinion.
            description_similarity = score_texts(text, description)

        mismatch = similarity is not None and similarity < minimum

        if mismatch:
            reason = (
                f"CLIP similarity between the claim and the image is {similarity}, "
                f"below {minimum}"
            )

            if description_similarity is not None:
                reason += (
                    f"; the picture was described as {description!r} "
                    f"(similarity {description_similarity})"
                )
        elif similarity is None:
            reason = "the image could not be scored"
        else:
            reason = f"CLIP similarity {similarity} is at or above {minimum}"

        findings.append(
            {
                "claim_id": claim_id,
                "similarity": similarity,
                "description_similarity": description_similarity,
                "mismatch": mismatch,
                "reason": reason,
            }
        )

    return findings
