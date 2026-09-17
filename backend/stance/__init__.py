"""
Stage 4 - what does each piece of evidence say about its claim?

`classify_stance` is pure (claims + flat text -> StanceResults);
`apply_stances` is the one function that reads and writes the evidence
graph.
"""

from .classify import classify_stance, rating_stance
from .graph_adapter import StanceReport, apply_stances
from .passages import split_passages
from .schema import METHODS, STANCES, EvidenceInput, StanceResult

__all__ = [
    "METHODS",
    "STANCES",
    "EvidenceInput",
    "StanceResult",
    "StanceReport",
    "classify_stance",
    "apply_stances",
    "rating_stance",
    "split_passages",
]
