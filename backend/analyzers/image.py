import logging

from PIL import Image
from . import models


log = logging.getLogger(__name__)


def ocr(path):
    reader = models.ocr_reader()
    results = reader.readtext(path)

    texts = []

    for result in results:
        if len(result) >= 2:
            texts.append(result[1])

    return " ".join(texts).strip()


def trocr_ocr(path):
    processor, model = models.trocr()

    image = Image.open(path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    generated_ids = model.generate(
        **inputs
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return text.strip()


def describe(path):
    image = Image.open(path).convert("RGB")

    # The `image-text-to-text` pipeline is the only captioning task left in
    # transformers 5, and it refuses an image on its own ("You must provide
    # text for this pipeline"). An empty prompt is how BLIP is asked for an
    # unconditional caption, and it keeps `generated_text` free of a prefix
    # we would then have to strip back off.
    result = models.blip()(image, text="")

    if not result:
        return ""

    return (result[0].get("generated_text") or "").strip()


def embed(path):
    import torch

    processor, model = models.dinov2()

    image = Image.open(path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    vector = outputs.last_hidden_state[:, 0, :]

    vector = torch.nn.functional.normalize(
        vector,
        p=2,
        dim=1
    )

    return vector[0].tolist()


def exif_date(path):
    image = Image.open(path)

    exif = image.getexif()

    if not exif:
        return None

    return (
        exif.get(36867)
        or exif.get(306)
    )


def ai_score(path):
    detector = models.ai_detector()

    if detector is None:
        return None

    results = detector(
        Image.open(path).convert("RGB")
    )

    if not results:
        return None

    return float(results[0]["score"])


def _soft(signal, function, path, fallback):
    """
    Read one image signal, or give up on just that signal.

    Each of these needs an optional model — easyocr, BLIP, DINOv2, the
    AI-image detector — and any of them may be missing or may choke on a
    particular file. That is not a reason to refuse to build the packet:
    an image with no OCR text is still worth checking against the known
    -image index, and the downstream stages already treat every field
    here as optional. So a failure costs one field, not the packet.
    """
    try:
        return function(path)
    except Exception as error:
        log.warning("image %s unavailable for %s: %s", signal, path, error)

        return fallback


def analyze(path, frame_time=None):
    return {
        "path": path,
        "ocr_text": _soft("ocr", ocr, path, ""),
        "description": _soft("description", describe, path, ""),
        "embedding": _soft("embedding", embed, path, None),
        "exif_date": _soft("exif date", exif_date, path, None),
        "ai_generated_score": _soft("ai score", ai_score, path, None),
        "frame_time": frame_time,
    }