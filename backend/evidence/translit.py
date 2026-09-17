"""
Compatibility shim: the lexicon moved to `backend/translit.py`.

It was written for retrieval (stage 3b) and lived here. Stage 2 needs the
same lexicon for the opposite direction — Devanagari claims have to be
read by feature tables written in Latin script (DECISIONS.md O7) — and
stage 4 needs it to put a romanised-Hindi claim in front of an entailment
model that cannot read Latin-script Hindi (O6). A shared linguistic
resource that three stages use is not an evidence concern, so it now sits
at `backend/translit.py`, beside `backend/torch_runtime.py`.

This module re-exports it so `from backend.evidence import translit` and
`from backend.evidence.translit import devanagari` keep working, in the
same way `backend/evidence/graph.py` re-exports the moved graph store
(DECISIONS.md D3).
"""

from ..translit import (                                    # noqa: F401
    AMBIGUOUS,
    DEVANAGARI,
    HINDI,
    LEXICON,
    LOANWORDS,
    MIN_HINDI_RATIO,
    MIN_HINDI_WORDS,
    devanagari,
    hindi_ratio,
    is_devanagari,
    looks_romanised,
    romanised,
    transliterate,
    variants,
)

__all__ = [
    "AMBIGUOUS",
    "DEVANAGARI",
    "HINDI",
    "LEXICON",
    "LOANWORDS",
    "MIN_HINDI_RATIO",
    "MIN_HINDI_WORDS",
    "devanagari",
    "hindi_ratio",
    "is_devanagari",
    "looks_romanised",
    "romanised",
    "transliterate",
    "variants",
]
