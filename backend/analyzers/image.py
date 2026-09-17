from PIL import Image
from . import models


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

    result = models.blip()(image)

    if not result:
        return ""

    return result[0]["generated_text"]


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


def analyze(path, frame_time=None):
    return {
        "path": path,
        "ocr_text": ocr(path),
        "description": describe(path),
        "embedding": embed(path),
        "exif_date": exif_date(path),
        "ai_generated_score": ai_score(path),
        "frame_time": frame_time,
    }