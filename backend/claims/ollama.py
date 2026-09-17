"""
Ollama-backed claim extraction (local LLM over HTTP).

Unlike the regex / transformer backends, which score one sentence at a
time, this sends a whole block of text to the model and asks for the
claims as JSON. The model can therefore split compound sentences, resolve
pronouns, and drop chit-chat in one pass. Ollama's structured-output mode
(`format` = JSON schema) keeps the reply parseable.

Env vars:

    OLLAMA_HOST          default http://localhost:11434
    CLAIM_OLLAMA_MODEL   default qwen2.5:7b
    CLAIM_OLLAMA_TIMEOUT seconds, default 120
"""

import json
import os

import requests

from .schema import CLAIM_TYPES


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT = 120

ENTITY_LABELS = ("PERSON", "ORG", "GPE", "MONEY", "PERCENT", "NUMBER", "DATE", "URL", "PHONE", "UPI", "OTHER")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "claim_type": {"type": "string", "enum": list(CLAIM_TYPES)},
                    "confidence": {"type": "number"},
                    "attributed_to": {"type": ["string", "null"]},
                    "time": {"type": ["string", "null"]},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "label": {"type": "string", "enum": list(ENTITY_LABELS)},
                            },
                            "required": ["text", "label"],
                        },
                    },
                },
                "required": ["text", "claim_type", "confidence", "entities"],
            },
        }
    },
    "required": ["claims"],
}

SYSTEM_PROMPT = """You are a fact-checking assistant. Extract every separate, check-worthy factual claim from the message.

A check-worthy claim is a statement about the world that could be verified or refuted with evidence: events, numbers, money, government actions, health effects, quotes, offers that promise money or prizes.

Do NOT extract: greetings, questions, personal opinions ("I think"), personal chat, pure instructions with no factual content.

Rules:
- Copy the claim text from the message as closely as possible (same language, including Hindi or Hinglish). Do not translate.
- If a sentence contains two independent claims, output them separately.
- claim_type is one of: statistic, money, event, attribution, policy, health, prediction, causal, chain_offer, generic.
  chain_offer = forward-this-message offers, cashback/prize/lottery scams.
- confidence is 0.0-1.0: how confident you are that this is a verifiable factual claim.
- entities: named people, organisations, places, money, numbers, dates, phone numbers, UPI ids, URLs in the claim.
- attributed_to: who is quoted or credited, if any, else null.
- time: the date/time expression in the claim, if any, else null.

Return JSON only."""


def host():
    return os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")


def model():
    return os.getenv("CLAIM_OLLAMA_MODEL", DEFAULT_MODEL)


def available():
    try:
        response = requests.get(f"{host()}/api/tags", timeout=2)
    except requests.RequestException:
        return False

    if response.status_code != 200:
        return False

    names = [m.get("name", "") for m in response.json().get("models", [])]
    wanted = model()

    return any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names)


def chat(text):
    """
    One request to /api/chat. Returns the parsed {"claims": [...]} dict.
    Raises requests.RequestException / ValueError on transport or parse
    failure so the caller can decide whether to fall back.
    """
    payload = {
        "model": model(),
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }

    response = requests.post(
        f"{host()}/api/chat",
        json=payload,
        timeout=int(os.getenv("CLAIM_OLLAMA_TIMEOUT", DEFAULT_TIMEOUT)),
    )
    response.raise_for_status()

    content = response.json()["message"]["content"]

    return json.loads(content)


def extract(text):
    """
    Return a list of raw claim dicts for one block of text, validated and
    normalised so the extractor can trust the shape.
    """
    if not text or not text.strip():
        return []

    raw = chat(text).get("claims") or []
    claims = []

    for item in raw:
        claim_text = (item.get("text") or "").strip()

        if not claim_text:
            continue

        claim_type = item.get("claim_type")

        if claim_type not in CLAIM_TYPES:
            claim_type = "generic"

        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        entities = []

        for entity in item.get("entities") or []:
            entity_text = (entity.get("text") or "").strip()
            label = entity.get("label")

            if entity_text and label in ENTITY_LABELS:
                entities.append({"text": entity_text, "label": label})

        claims.append(
            {
                "text": claim_text,
                "claim_type": claim_type,
                "confidence": max(0.0, min(1.0, confidence)),
                "attributed_to": (item.get("attributed_to") or None),
                "time": (item.get("time") or None),
                "entities": entities,
            }
        )

    return claims
