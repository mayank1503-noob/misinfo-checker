import os
from functools import lru_cache


@lru_cache
def whisper_model():
    import whisper

    return whisper.load_model("base")


@lru_cache
def ocr_reader():
    import easyocr

    return easyocr.Reader(["en", "hi"])


@lru_cache
def trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    name = "microsoft/trocr-base-printed"

    processor = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name)

    return processor, model


@lru_cache
def blip():
    from transformers import pipeline

    return pipeline(
        "image-text-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@lru_cache
def dinov2():
    from transformers import AutoImageProcessor, AutoModel

    name = "facebook/dinov2-base"

    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()

    return processor, model


@lru_cache
def ai_detector():
    name = os.getenv("AI_DETECTOR_MODEL")

    if not name:
        return None

    from transformers import pipeline

    return pipeline(
        "image-classification",
        model=name
    )