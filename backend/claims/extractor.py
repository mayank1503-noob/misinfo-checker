"""
Claim extraction: media packet -> ClaimSet.

    packet = from_text("...")          # backend.analyzers.packet
    claims = extract_claims(packet)    # ClaimSet
    graph  = claims.to_graph_seed()    # nodes/edges for the evidence graph

Pipeline for each text-bearing field in the packet (text, per-image OCR,
per-image caption):

    split into sentences
    -> extract entities + time refs
    -> score check-worthiness
    -> classify claim type
    -> dedupe near-identical claims across fields (OCR often repeats text)

With backend="transformer" (the default when `transformers` is installed)
the entity step is augmented with model NER, and the score / type steps
are blended with a multilingual zero-shot NLI model. See transformer.py.
"""

import os
import re

from . import checkworthy
from . import ollama
from . import transformer
from .entities import extract_entities, extract_time_refs
from .schema import Claim, ClaimSet, Entity, SourceSpan, stable_id
from .segment import candidate_sentences, iter_sources, split_sentences


_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

DEDUPE_JACCARD = 0.8


def normalize(text):
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    text = _WS.sub(" ", text).strip()

    return text


def _tokens(text):
    return set(normalize(text).split())


def _jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _packet_id(packet):
    return stable_id(
        "pkt",
        packet.get("input_type"),
        packet.get("text", "")[:500],
        len(packet.get("images") or []),
    )


NAMED_LABELS = {"PERSON", "ORG", "GPE", "PROPER"}

# Heuristic types that are more precise than the zero-shot model because
# they key off hard patterns (UPI ids, money + "forward", disease words).
STICKY_HEURISTIC_TYPES = {"chain_offer", "health"}

MODEL_WEIGHT = 0.5


BACKENDS = ("heuristic", "transformer", "ollama")


def resolve_backend(backend):
    """
    "auto"        -> $CLAIM_BACKEND if set, else transformer if
                     transformers+torch import, else heuristic.
                     (ollama is never auto-selected: it needs a running
                     server, so opt in explicitly or via CLAIM_BACKEND.)
    "transformer" -> transformer (raises if not installed)
    "ollama"      -> local LLM via Ollama (raises if unreachable)
    "heuristic"   -> regex only
    """
    if backend == "auto":
        backend = os.getenv("CLAIM_BACKEND", "auto")

    if backend == "auto":
        return "transformer" if transformer.available() else "heuristic"

    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS} or 'auto'")

    if backend == "transformer" and not transformer.available():
        raise RuntimeError(
            "backend='transformer' requires `transformers` and `torch` to be installed"
        )

    if backend == "ollama" and not ollama.available():
        raise RuntimeError(
            f"backend='ollama' requires an Ollama server at {ollama.host()} "
            f"with model {ollama.model()!r} pulled"
        )

    return backend


def _overlaps_any(entity, others):
    for other in others:
        if entity.start < other.end and entity.end > other.start:
            return True

    return False


def _merge_entities(regex_entities, model_entities):
    """
    Patterned regex entities (money, dates, UPI, ...) always win. Model
    NER replaces the regex proper-noun guesses, but any regex named entity
    the model did not cover (e.g. "Reserve Bank of India" via ORG_HINT)
    is kept.
    """
    patterned = [e for e in regex_entities if e.label not in NAMED_LABELS]
    guessed = [e for e in regex_entities if e.label in NAMED_LABELS]

    merged = list(patterned)

    for entity in model_entities:
        if not _overlaps_any(entity, merged):
            merged.append(entity)

    for entity in guessed:
        if not _overlaps_any(entity, merged):
            merged.append(entity)

    merged.sort(key=lambda e: e.start)

    return merged


def _build_claim(candidate, packet_id, today=None, backend="heuristic"):
    sentence = candidate["text"]

    entities = extract_entities(sentence)

    if backend == "transformer":
        entities = _merge_entities(entities, transformer.ner_entities(sentence))

    not_dates = [
        (e.start, e.end)
        for e in entities
        if e.label in ("MONEY", "PERCENT", "PHONE", "UPI", "URL")
    ]

    time_refs = extract_time_refs(sentence, today=today, exclude=not_dates)

    confidence, worthy, signals = checkworthy.score(sentence, entities, time_refs)
    claim_type = checkworthy.classify(sentence, signals, entities)

    signals["backend"] = backend

    # Hard rejects (greeting / length) stay hard: confidence == 0 means the
    # heuristic layer refused the sentence outright, so skip the model.
    if backend == "transformer" and confidence > 0.0:
        model_worthy = transformer.worthiness(sentence)
        model_type, type_score, margin = transformer.claim_type(sentence)

        signals["heuristic_confidence"] = confidence
        signals["model_worthy"] = round(model_worthy, 3)
        signals["heuristic_type"] = claim_type
        signals["model_type"] = model_type
        signals["model_type_score"] = round(type_score, 3)
        signals["model_type_margin"] = round(margin, 3)

        confidence = round(
            (1 - MODEL_WEIGHT) * confidence + MODEL_WEIGHT * model_worthy, 3
        )
        worthy = confidence >= checkworthy.CHECK_WORTHY_THRESHOLD

        if model_type and claim_type not in STICKY_HEURISTIC_TYPES:
            claim_type = model_type

    numbers = [
        e.normalized or e.text
        for e in entities
        if e.label in ("NUMBER", "MONEY", "PERCENT")
    ]

    speaker = None

    if claim_type == "attribution":
        speaker = checkworthy.attributed_to(sentence, entities)

    normalized_text = normalize(sentence)

    return Claim(
        id=stable_id("clm", packet_id, normalized_text),
        text=sentence,
        normalized_text=normalized_text,
        claim_type=claim_type,
        confidence=confidence,
        check_worthy=worthy,
        entities=entities,
        time_refs=time_refs,
        numbers=numbers,
        attributed_to=speaker,
        source=SourceSpan(
            field=candidate["field"],
            image_index=candidate["image_index"],
            frame_time=candidate["frame_time"],
            start=candidate["start"],
            end=candidate["end"],
        ),
        language=checkworthy.detect_language(sentence),
        signals=signals,
    )


def _dedupe(claims):
    """
    Keep the first (highest-priority source order: text, then OCR, then
    caption) of any group of near-identical claims. The survivor keeps the
    higher confidence of the pair.
    """
    kept = []
    kept_tokens = []

    for claim in claims:
        tokens = _tokens(claim.text)
        duplicate = None

        for index, existing in enumerate(kept_tokens):
            if _jaccard(tokens, existing) >= DEDUPE_JACCARD:
                duplicate = index
                break

        if duplicate is None:
            kept.append(claim)
            kept_tokens.append(tokens)
            continue

        survivor = kept[duplicate]

        if claim.confidence > survivor.confidence:
            survivor.confidence = claim.confidence
            survivor.check_worthy = claim.check_worthy

    return kept


LLM_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "MONEY", "PERCENT", "NUMBER", "DATE", "URL", "PHONE", "UPI"}


def _locate(claim_text, field_text, sentences):
    """
    Best-effort char span of an LLM claim inside its source field:
    exact (case-insensitive) substring first, else the sentence with the
    highest token overlap, else the whole field.
    """
    index = field_text.lower().find(claim_text.lower())

    if index >= 0:
        return index, index + len(claim_text)

    tokens = _tokens(claim_text)
    best, best_score = None, 0.0

    for sentence, start, end in sentences:
        score = _jaccard(tokens, _tokens(sentence))

        if score > best_score:
            best, best_score = (start, end), score

    if best and best_score >= 0.3:
        return best

    return 0, len(field_text)


def _llm_entities(raw_entities, claim_text, regex_entities):
    """
    Turn the LLM's {text, label} list into Entity objects anchored in the
    claim text, then merge with regex patterned entities (which win on
    overlap, since their normalisation is reliable).
    """
    model_entities = []

    for item in raw_entities:
        label = item["label"]

        if label not in LLM_ENTITY_LABELS:
            continue

        start = claim_text.lower().find(item["text"].lower())

        if start < 0:
            continue

        end = start + len(item["text"])

        model_entities.append(
            Entity(
                id=stable_id("ent", label, item["text"].lower()),
                text=claim_text[start:end],
                label=label,
                normalized=item["text"].lower(),
                start=start,
                end=end,
            )
        )

    patterned = [e for e in regex_entities if e.label not in NAMED_LABELS]
    merged = list(patterned)

    for entity in model_entities:
        if not _overlaps_any(entity, merged):
            merged.append(entity)

    merged.sort(key=lambda e: e.start)

    return merged


def _extract_with_ollama(packet, packet_id, today, include_dropped):
    claims = []
    dropped = []

    for field, image_index, frame_time, text in iter_sources(packet):
        if not text:
            continue

        sentences = split_sentences(text)
        covered = []

        for raw in ollama.extract(text):
            claim_text = raw["text"]
            start, end = _locate(claim_text, text, sentences)
            covered.append((start, end))

            regex_entities = extract_entities(claim_text)
            entities = _llm_entities(raw["entities"], claim_text, regex_entities)

            not_dates = [
                (e.start, e.end)
                for e in entities
                if e.label in ("MONEY", "PERCENT", "PHONE", "UPI", "URL")
            ]
            time_refs = extract_time_refs(claim_text, today=today, exclude=not_dates)

            # Heuristic signals are still computed so the eval harness can
            # compare backends; the LLM's own judgement sets score and type.
            heuristic_conf, _, signals = checkworthy.score(claim_text, entities, time_refs)
            signals["backend"] = "ollama"
            signals["heuristic_confidence"] = heuristic_conf
            signals["heuristic_type"] = checkworthy.classify(claim_text, signals, entities)
            signals["llm_time"] = raw["time"]

            confidence = round(raw["confidence"], 3)
            normalized_text = normalize(claim_text)

            claims.append(
                Claim(
                    id=stable_id("clm", packet_id, normalized_text),
                    text=claim_text,
                    normalized_text=normalized_text,
                    claim_type=raw["claim_type"],
                    confidence=confidence,
                    check_worthy=confidence >= checkworthy.CHECK_WORTHY_THRESHOLD,
                    entities=entities,
                    time_refs=time_refs,
                    numbers=[
                        e.normalized or e.text
                        for e in entities
                        if e.label in ("NUMBER", "MONEY", "PERCENT")
                    ],
                    attributed_to=raw["attributed_to"],
                    source=SourceSpan(
                        field=field,
                        image_index=image_index,
                        frame_time=frame_time,
                        start=start,
                        end=end,
                    ),
                    language=checkworthy.detect_language(claim_text),
                    signals=signals,
                )
            )

        if include_dropped:
            for sentence, start, end in sentences:
                if not any(start < c_end and end > c_start for c_start, c_end in covered):
                    dropped.append(
                        {
                            "text": sentence,
                            "field": field,
                            "confidence": 0.0,
                            "reason": "llm_not_claim",
                        }
                    )

    return claims, dropped


def extract_claims(packet, today=None, include_dropped=True, backend="auto"):
    """
    Extract check-worthy claims from a media packet produced by
    backend.analyzers.packet.

    backend: "auto" (default) honours $CLAIM_BACKEND, else uses the
    transformer models when `transformers` is installed, else regex
    heuristics. Pass "heuristic" to force the dependency-free path,
    "transformer" to require the HF models, or "ollama" to extract
    with a local LLM (needs a running Ollama server).

    Returns a ClaimSet. `claims` holds every sentence that scored above
    the threshold; `dropped` records rejected sentences with the reason
    so the evaluation harness can tune the heuristics.
    """
    backend = resolve_backend(backend)
    packet_id = _packet_id(packet)

    claims = []
    dropped = []

    if backend == "ollama":
        extracted, dropped = _extract_with_ollama(packet, packet_id, today, include_dropped)

        for claim in extracted:
            if claim.check_worthy:
                claims.append(claim)
            elif include_dropped:
                dropped.append(
                    {
                        "text": claim.text,
                        "field": claim.source.field,
                        "confidence": claim.confidence,
                        "reason": "llm_low_confidence",
                    }
                )
    else:
        for candidate in candidate_sentences(packet):
            claim = _build_claim(candidate, packet_id, today=today, backend=backend)

            if claim.check_worthy:
                claims.append(claim)
            elif include_dropped:
                dropped.append(
                    {
                        "text": claim.text,
                        "field": claim.source.field,
                        "confidence": claim.confidence,
                        "reason": _drop_reason(claim.signals),
                    }
                )

    claims = _dedupe(claims)
    claims.sort(key=lambda c: c.confidence, reverse=True)

    return ClaimSet(
        packet_id=packet_id,
        input_type=packet.get("input_type", "unknown"),
        source_date=packet.get("source_date"),
        backend=backend,
        claims=claims,
        dropped=dropped,
    )


def _drop_reason(signals):
    heuristic = signals.get("heuristic_confidence")

    if heuristic is not None and heuristic >= checkworthy.CHECK_WORTHY_THRESHOLD:
        return "model_not_claim"

    if signals.get("greeting"):
        return "greeting"
    if signals.get("tokens", 0) < checkworthy.MIN_TOKENS:
        return "too_short"
    if signals.get("tokens", 0) > checkworthy.MAX_TOKENS:
        return "too_long"
    if signals.get("question"):
        return "question"
    if signals.get("opinion"):
        return "opinion"
    if signals.get("personal"):
        return "personal"
    if signals.get("imperative_only"):
        return "imperative_only"

    return "low_signal"
