"""
Lightweight, regex-based entity and time extraction.

Intentionally dependency-free: this is the baseline the later model-backed
extractor will be compared against, and it has to run inside the bot with
no GPU. Offsets are relative to the sentence passed in.
"""

import re
from datetime import date, timedelta

from .schema import Entity, TimeRef, stable_id


MONEY = re.compile(
    r"(?:₹|Rs\.?|INR|\$|USD|€)\s?\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|cr|k|million|billion)?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|rupees|rupee|rupaye|dollars)",
    re.IGNORECASE,
)

PERCENT = re.compile(r"\d+(?:\.\d+)?\s?(?:%|percent|per cent)", re.IGNORECASE)

NUMBER = re.compile(
    r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|lakh|crore|k))?(?![\w.])",
    re.IGNORECASE,
)

URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

PHONE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")

UPI = re.compile(
    r"\b[\w.\-]{2,}@(?:upi|ybl|okaxis|oksbi|okicici|okhdfcbank|paytm|apl|ibl|axl)\b",
    re.IGNORECASE,
)

HASHTAG = re.compile(r"#\w+")

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

DATE_ABS = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\.?,?\s+\d{{4}}"
    rf"|(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
    rf"|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|(?:{MONTHS})\s+\d{{4}}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\b"
    rf"|(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b"
    rf"|(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)

_DATE_FORMATS = (
    "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d",
    "%B %Y", "%b %Y", "%Y",
)

_DAY_MONTH_FORMATS = ("%d %B", "%d %b", "%B %d", "%b %d")


def _norm_absolute(text, today):
    from datetime import datetime

    cleaned = re.sub(r"(\d)(?:st|nd|rd|th)", r"\1", text)
    cleaned = cleaned.replace(",", "").replace(".", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue

    # day + month with no year: assume the current year
    for fmt in _DAY_MONTH_FORMATS:
        try:
            return datetime.strptime(f"{cleaned} {today.year}", f"{fmt} %Y").date().isoformat()
        except ValueError:
            continue

    return None

DATE_REL = re.compile(
    r"\b(?:today|yesterday|tomorrow|tonight|last\s+(?:night|week|month|year)|"
    r"next\s+(?:week|month|year)|this\s+(?:week|month|year|morning|evening)|"
    r"aaj|kal|parso|abhi|recently|just now|"
    r"\d+\s+(?:days?|weeks?|months?|years?|hours?)\s+ago)\b",
    re.IGNORECASE,
)

DATE_RECUR = re.compile(
    r"\b(?:every\s+(?:day|week|month|year|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)|daily|weekly|monthly|annually)\b",
    re.IGNORECASE,
)

# Org suffixes / government words that promote a capitalized span to ORG.
ORG_HINT = re.compile(
    r"\b(?:Ltd|Limited|Inc|Corp|Bank|Ministry|Government|Govt|Sarkar|Department|"
    r"University|Hospital|Police|Court|Commission|Board|Authority|Airlines|"
    r"WHO|UNICEF|RBI|SBI|ISRO|NASA|UPI|NPCI|Reserve Bank|Supreme Court)\b"
)

PERSON_TITLE = re.compile(
    r"(?:Dr|Mr|Mrs|Ms|Prof|PM|CM|Minister|President|Chief Minister|Prime Minister|Shri|Smt)\.?\s*$"
)

# Sequence of Capitalized words (allows "of", "and", "&" inside).
PROPER = re.compile(
    r"\b[A-Z][a-zA-Z'’\-]+(?:(?:\s+(?:of|and|&|the|de))?\s+[A-Z][a-zA-Z'’\-]+)*"
)

# Words that look capitalized only because they start the sentence.
STOP_START = {
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "i", "you", "there", "here", "if", "when", "as", "but",
    "and", "or", "so", "please", "forward", "share", "breaking", "urgent",
    "alert", "news", "attention", "important", "note", "warning", "dear",
    "hi", "hello", "good", "yes", "no", "ok", "okay", "according", "in",
    "on", "at", "by", "for", "from", "with", "to", "of", "is", "are", "was",
    "were", "has", "have", "had", "will", "can", "do", "does", "did", "all",
    "every", "free", "get", "send", "click", "join", "today", "yesterday",
    "tomorrow", "govt", "government", "my", "our", "your", "just", "new",
}

_SENTENCE_INITIAL = re.compile(r"^\s*(?:\w+:\s*)?$")
_ADJ_SHAPE = re.compile(r"(?:ing|ive|ed|ly|ous|ful|less|able|ible)$", re.IGNORECASE)


WORD_BOUNDARY = r"\b{}\b"

# A finite verb straight after the word: an English inflected form, an
# auxiliary, or the Hindi ergative-plus-verb that follows a subject
# ("RBI ne kaha", "sarkar ne bataya").
SUBJECT_OF_VERB = re.compile(
    r"\s+(?:\w+\s+)?(?:is|are|was|were|has|have|had|will|can|does|did|do|"
    r"ne|hai|hain|tha|thi|the|kaha|bola|bataya|"
    r"[a-z]+(?:es|ed|ing)|[a-z]{3,}s)\b"
)


def _corroborated(text, sentence, start):
    """
    Is this sentence-initial capitalised word a name on evidence other
    than its position?

    Any one of: it is an acronym, it hints at an organisation or a place,
    a person title precedes it, or it appears capitalised again later in
    the sentence, where the sentence start cannot explain the capital.
    """
    if text.isupper():
        return True

    if ORG_HINT.search(text) or any(w in GPE_HINT for w in text.lower().split()):
        return True

    if PERSON_TITLE.search(sentence[max(0, start - 20):start]):
        return True

    rest = sentence[start + len(text):]

    if re.search(WORD_BOUNDARY.format(re.escape(text)), rest):
        return True

    # Subject position: "Modi announces...", "UNESCO has declared...",
    # "RBI ne kaha...". A capitalised word followed by a finite verb is
    # doing the verb, which is what a name does and what "Kal shaam",
    # "Namak lekar" and "Aaj office" — capital then noun — do not.
    return bool(SUBJECT_OF_VERB.match(rest))


GPE_HINT = {
    "india", "delhi", "mumbai", "bengaluru", "bangalore", "chennai", "kolkata",
    "hyderabad", "pune", "kerala", "tamil", "nadu", "gujarat", "maharashtra",
    "bihar", "punjab", "uttar", "pradesh", "karnataka", "rajasthan", "goa",
    "pakistan", "china", "usa", "america", "uk", "london", "nepal",
    "bangladesh", "sri", "lanka", "europe", "russia", "ukraine", "israel",
}


def _norm_money(text):
    cleaned = text.lower().replace(",", "").replace(" ", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)

    if not match:
        return None

    value = float(match.group())

    if "crore" in cleaned or cleaned.endswith("cr"):
        value *= 1e7
    elif "lakh" in cleaned:
        value *= 1e5
    elif "million" in cleaned:
        value *= 1e6
    elif "billion" in cleaned:
        value *= 1e9
    elif re.search(r"\d+k$", cleaned):
        value *= 1e3

    currency = "INR"

    if "$" in cleaned or "usd" in cleaned or "dollar" in cleaned:
        currency = "USD"
    elif "€" in cleaned:
        currency = "EUR"

    return f"{currency} {value:.0f}"


def _norm_relative(text, today):
    lowered = text.lower().strip()

    if lowered in ("today", "aaj", "tonight", "abhi", "just now"):
        return today.isoformat()

    if lowered == "yesterday":
        return (today - timedelta(days=1)).isoformat()

    if lowered == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    match = re.match(r"(\d+)\s+(day|week|month|year|hour)s?\s+ago", lowered)

    if match:
        n = int(match.group(1))
        unit = match.group(2)

        delta = {
            "hour": timedelta(hours=n),
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=30 * n),
            "year": timedelta(days=365 * n),
        }[unit]

        return (today - delta).isoformat()

    return None


def _overlaps(span, spans):
    for s, e in spans:
        if span[0] < e and span[1] > s:
            return True

    return False


def extract_entities(sentence):
    """
    Return a list of Entity for one sentence. Patterned entities (money,
    percent, url, phone, upi, hashtag, date) win over proper-noun spans.
    """
    entities = []
    taken = []

    def add(label, match, normalized=None):
        span = (match.start(), match.end())

        if _overlaps(span, taken):
            return

        text = match.group().strip()

        entities.append(
            Entity(
                id=stable_id("ent", label, (normalized or text).lower()),
                text=text,
                label=label,
                normalized=normalized,
                start=span[0],
                end=span[1],
            )
        )
        taken.append(span)

    for m in URL.finditer(sentence):
        add("URL", m, m.group().lower())

    for m in UPI.finditer(sentence):
        add("UPI", m, m.group().lower())

    for m in PHONE.finditer(sentence):
        add("PHONE", m, re.sub(r"\D", "", m.group())[-10:])

    for m in HASHTAG.finditer(sentence):
        add("HASHTAG", m, m.group().lower())

    for m in MONEY.finditer(sentence):
        add("MONEY", m, _norm_money(m.group()))

    for m in PERCENT.finditer(sentence):
        add("PERCENT", m, re.sub(r"[^\d.]", "", m.group()) + "%")

    for m in DATE_ABS.finditer(sentence):
        add("DATE", m, _norm_absolute(m.group(), date.today()))

    for m in NUMBER.finditer(sentence):
        add("NUMBER", m, m.group().replace(",", "").lower())

    for m in PROPER.finditer(sentence):
        start = m.start()
        text = m.group().strip()
        words = text.split()

        # drop leading sentence-initial stop words ("The Reserve Bank" -> "Reserve Bank")
        while words and words[0].lower() in STOP_START:
            start += len(words[0]) + 1
            words = words[1:]

        if not words:
            continue

        text = " ".join(words)
        span = (start, start + len(text))

        if len(text) < 3 or _overlaps(span, taken):
            continue

        # A lone capitalised word at the start of a sentence is the
        # weakest possible evidence of a name, because *every* sentence
        # starts with a capital. "Massive earthquake", "Drinking hot
        # water", "Kal shaam ko ghar aa raha hoon", "Namak lekar aana" —
        # the first word is capitalised by orthography, not because it
        # names anything, and romanised Hindi hits this constantly since
        # its everyday words are not in any English vocabulary.
        #
        # So a single sentence-initial word has to corroborate itself
        # some other way: an organisation or place hint, a person title
        # in front of it, an acronym's capitals, or the same word
        # capitalised again later in the sentence where orthography does
        # not explain it. Multi-word spans are unaffected — "Reserve
        # Bank" is not an accident — and so is any word anywhere else in
        # the sentence.
        if len(words) == 1 and _SENTENCE_INITIAL.match(sentence[:start]):
            if not _corroborated(text, sentence, start):
                continue

        lowered = text.lower()
        before = sentence[max(0, start - 20):start]

        if ORG_HINT.search(text):
            label = "ORG"
        elif any(w in GPE_HINT for w in lowered.split()):
            label = "GPE"
        elif PERSON_TITLE.search(before):
            label = "PERSON"
        elif len(words) >= 2:
            label = "PERSON"
        elif text.isupper():
            label = "ORG"
        else:
            label = "PROPER"

        entities.append(
            Entity(
                id=stable_id("ent", label, lowered),
                text=text,
                label=label,
                normalized=lowered,
                start=span[0],
                end=span[1],
            )
        )
        taken.append(span)

    entities.sort(key=lambda e: e.start)

    return entities


def extract_time_refs(sentence, today=None, exclude=()):
    """
    `exclude` is a list of (start, end) spans that must not be read as
    time references, e.g. the "2000" inside "Rs 2000".
    """
    today = today or date.today()
    refs = []
    taken = list(exclude)

    def add(kind, match, normalized=None):
        span = (match.start(), match.end())

        if _overlaps(span, taken):
            return

        refs.append(
            TimeRef(
                text=match.group(),
                kind=kind,
                normalized=normalized,
                start=span[0],
                end=span[1],
            )
        )
        taken.append(span)

    for m in DATE_ABS.finditer(sentence):
        add("absolute", m, _norm_absolute(m.group(), today))

    for m in DATE_REL.finditer(sentence):
        add("relative", m, _norm_relative(m.group(), today))

    for m in DATE_RECUR.finditer(sentence):
        add("recurring", m)

    refs.sort(key=lambda r: r.start)

    return refs
