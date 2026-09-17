"""
Turn a media packet into candidate sentences, remembering where each one
came from so a claim can be traced back to its OCR block / caption / transcript.
"""

import re


# Sentence terminators: ., !, ?, Devanagari danda, newlines.
_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")

# Common abbreviations that should not end a sentence.
_ABBREV = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Sr|Jr|Prof|Rs|Govt|Ltd|Inc|vs|St|No|Sh|Smt)\.$",
    re.IGNORECASE,
)

_WS = re.compile(r"\s+")


def clean(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_sentences(text):
    """
    Return [(sentence, start, end)] with char offsets into `text`.
    """
    text = text or ""
    sentences = []

    pos = 0
    pieces = _SPLIT.split(text)

    buffer = None
    buffer_start = 0

    for piece in pieces:
        if piece is None:
            continue

        start = text.find(piece, pos)

        if start < 0:
            start = pos

        end = start + len(piece)
        pos = end

        stripped = piece.strip()

        if not stripped:
            continue

        if buffer is not None:
            stripped = buffer + " " + stripped
            start = buffer_start
            buffer = None

        if _ABBREV.search(stripped):
            buffer = stripped
            buffer_start = start
            continue

        sentences.append((stripped, start, end))

    if buffer is not None:
        sentences.append((buffer, buffer_start, len(text)))

    return sentences


def iter_sources(packet):
    """
    Yield (field, image_index, frame_time, text) for every text-bearing
    part of the packet.

    Note: for video packets, the transcript is already merged into
    packet["text"] by packet.from_video, so it arrives under field="text".
    """
    yield ("text", None, None, clean(packet.get("text")))

    for index, image in enumerate(packet.get("images") or []):
        frame_time = image.get("frame_time")

        yield ("ocr", index, frame_time, clean(image.get("ocr_text")))
        yield ("caption", index, frame_time, clean(image.get("description")))


def candidate_sentences(packet):
    """
    Yield dicts: {text, field, image_index, frame_time, start, end}
    """
    for field, image_index, frame_time, text in iter_sources(packet):
        if not text:
            continue

        for sentence, start, end in split_sentences(text):
            yield {
                "text": _WS.sub(" ", sentence),
                "field": field,
                "image_index": image_index,
                "frame_time": frame_time,
                "start": start,
                "end": end,
            }
