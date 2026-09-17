"""
Stage 5 - image evidence.

Reuses everything stage 1 already computed for each image (the DINOv2
embedding, the BLIP description, the OCR text, the EXIF date, the
AI-generated score, the frame time) and adds three checks on top:

    keyframes.py      drop near-identical video frames, cap the count
    local_index.py    have we seen this picture before, and when?
    reverse_search.py the same question asked of the open web
    consistency.py    does the picture show what the claim says it shows?

    from backend.images import collect_image_evidence

    report = collect_image_evidence(claimset, graph, packet)
"""

from .collector import ImageReport, collect_image_evidence
from .keyframes import dedupe_packet, select_keyframes

__all__ = [
    "collect_image_evidence",
    "ImageReport",
    "select_keyframes",
    "dedupe_packet",
]
