"""
Decide whether a sentence is a check-worthy factual claim, and what kind.

Everything here is a heuristic feature -> weight table so the behaviour is
inspectable. The `signals` dict returned alongside the score is stored on
the Claim for evaluation later.

## Two scripts, one table

The tables below are written in Latin script, and deliberately mix English
with romanised Hindi: "government|sarkar", "cure|ilaaj", "free|muft". That
is what lets a Hinglish forward score like the English message it is a
translation of. It also meant a *Devanagari* sentence matched nothing at
all: "सरकार सभी छात्रों को मुफ्त लैपटॉप दे रही है" is the same claim as
"Sarkar sabhi students ko free laptop de rahi hai", but every pattern here
missed it, so it scored the bare 0.30 base and was dropped before
retrieval was ever asked (DECISIONS.md O7).

So a Devanagari sentence is also read in its romanised spelling, through
the same lexicon the retrieval side already uses in the other direction,
and a feature counts when *either* spelling shows it. The sentence itself
is never rewritten: what is stored on the Claim, sent to retrieval and
shown to a person is always the text as it arrived — only the feature
tables see the second spelling.

This cuts both ways on purpose. The penalties are read in both spellings
too, so Devanagari chat is still chat: "मैं कल घर आ रहा हूँ" romanises to
"main kal ghar aa raha hoon" and takes the same interpersonal penalty its
Hinglish twin does. Hindi is made *legible* to the table, not exempt from
it.

Nothing about English or romanised Hindi changes: `romanised()` returns
None for both, no second reading is built, and the signals are the ones
this module always produced.
"""

import re

from ..translit import romanised
from .entities import extract_time_refs


CHECK_WORTHY_THRESHOLD = 0.45

MIN_TOKENS = 4
MAX_TOKENS = 60


QUESTION = re.compile(r"\?\s*$|^\s*(?:is|are|was|were|do|does|did|can|could|will|would|kya|kyu|kyun|why|how|what|when|where|who)\b", re.IGNORECASE)

OPINION = re.compile(
    r"\b(?:i\s+(?:think|feel|believe|guess|hope|wish|love|hate|like)|in my opinion|imo|imho|"
    r"personally|honestly|i\s+am\s+(?:sure|not sure)|mujhe lagta|mera manna|lagta hai|shayad|"
    r"maybe|probably|perhaps|should|must|best|worst|amazing|awesome|beautiful|terrible|"
    r"disgusting|great|bad|good|nice|lovely|boring|so cute|omg|lol|lmao)\b",
    re.IGNORECASE,
)

# Romanised Hindi marks the person on the verb, so "aa raha hoon" (I am
# coming) and "kar paya" (I could) are first-person the way "I am" is,
# with no pronoun anywhere in the sentence for PERSONAL to match. A
# message that conjugates itself to a speaker and an addressee is
# interpersonal, whatever nouns it happens to contain.
FIRST_SECOND_PERSON = re.compile(
    r"\b(?:hoon|huun|hun|rahaa?\s+hoon|rahi\s+hoon|karunga|karungi|karunga|"
    r"paya|payi|paaya|sakta\s+hoon|sakti\s+hoon|"
    r"tum|tumhe|tumhara|tumhari|tere|tera|teri|aap|aapko|aapka|aapki|"
    r"rukoge|jaoge|aaoge|karoge|bataoge|milte|milenge|"
    r"mujhe|mujhko|mera|meri|mere|hamara|hamari|humein|apna|apni)\b",
    re.IGNORECASE,
)

PERSONAL = re.compile(
    r"^\s*(?:i|we|my|our|me|mujhe|main|hum|humne|maine|mere|hamare)\b|"
    r"\b(?:call me|meet me|see you|miss you|love you|good morning|good night|happy birthday|"
    r"kaise ho|kya haal|thik hai|theek hai|kal milte|bhai|yaar|dude|bro)\b",
    re.IGNORECASE,
)

GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|namaste|namaskar|good\s+(?:morning|evening|night|afternoon)|thanks|thank you|ok|okay|yes|no|hmm|haan|nahi|acha|accha)\b[\s!.]*$",
    re.IGNORECASE,
)

IMPERATIVE_ONLY = re.compile(
    r"^\s*(?:please\s+)?(?:forward|share|send|click|join|subscribe|like|retweet|spread|"
    r"aage bhejo|forward karo|share karo|bhejo)\b",
    re.IGNORECASE,
)

ATTRIBUTION = re.compile(
    r"\b(?:said|says|announced|announces|confirmed|confirms|claimed|claims|stated|states|"
    r"reported|reports|declared|declares|warned|warns|tweeted|according to|as per|"
    r"kaha|bola|bataya|ghoshna|elaan)\b",
    re.IGNORECASE,
)

POLICY = re.compile(
    r"\b(?:government|govt|sarkar|sarkari|ministry|minister|rbi|sebi|supreme court|high court|"
    r"parliament|lok sabha|rajya sabha|scheme|yojana|policy|law|act|ban|banned|"
    r"mandatory|compulsory|order|notification|circular|rule|rules|tax|gst|aadhaar|"
    r"pan card|voter|election|passport|ration|subsidy|pension)\b",
    re.IGNORECASE,
)

HEALTH = re.compile(
    r"\b(?:cure|cures|cured|vaccine|vaccines|vaccination|virus|covid|corona|cancer|"
    r"diabetes|disease|infection|medicine|drug|tablet|doctor|doctors|hospital|"
    r"who|symptom|symptoms|immunity|dengue|malaria|flu|fever|heart attack|"
    r"deadly|poison|toxic|side effect|side effects|polio|ilaaj|dawai|bimari)\b",
    re.IGNORECASE,
)

CHAIN_OFFER = re.compile(
    r"\b(?:free|cashback|cash back|prize|winner|won|win|lucky|lottery|gift|reward|"
    r"bonus|jackpot|offer|voucher|coupon|recharge|giveaway|claim now|limited time|"
    r"forward (?:this )?(?:to|message)|share (?:this )?(?:with|to)|"
    r"muft|inaam|paise|jeet|jeeto|offer hai)\b",
    re.IGNORECASE,
)

PREDICTION = re.compile(
    r"\b(?:will|going to|about to|shall|by \d{4}|soon|next (?:week|month|year)|"
    r"hoga|hogi|honge|aayega|aayegi|karega|karegi)\b",
    re.IGNORECASE,
)

CAUSAL = re.compile(
    r"\b(?:causes|caused|cause|leads to|led to|results in|resulted in|because of|"
    r"due to|responsible for|triggers|triggered|se hota hai|ki wajah se|karan)\b",
    re.IGNORECASE,
)

EVENT = re.compile(
    r"\b(?:happened|occurred|killed|kills|died|dies|dead|death|arrested|attack|attacked|"
    r"blast|explosion|earthquake|flood|fire|crash|crashed|collapsed|protest|riot|"
    r"strike|shutdown|closed|opened|launched|launches|released|resigned|elected|"
    r"won|lost|defeated|signed|passed|approved|rejected|found|discovered|"
    r"mar gaya|mar gaye|giraftar|hamla|dhamaka)\b",
    re.IGNORECASE,
)

VERB_LIKE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|be|been|being|does|did|do|can|"
    r"hai|hain|tha|thi|the|hoga|hogi|raha|rahi|rahe|kiya|kiye|gaya|gayi|gaye|"
    r"[a-z]+(?:ed|es|ing|s))\b",
    re.IGNORECASE,
)

URGENCY = re.compile(
    r"\b(?:breaking|urgent|alert|attention|warning|must read|viral|shocking|"
    r"exposed|truth|100% true|real news|not fake|zaroor|jaldi|turant|dhyan)\b",
    re.IGNORECASE,
)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

HINGLISH_HINT = re.compile(
    r"\b(?:hai|hain|nahi|nahin|kya|kyu|kyun|aur|ke|ki|ka|ko|se|par|mein|main|"
    r"yeh|ye|woh|wo|sab|log|logo|logon|sarkar|paise|rupaye|karo|kare|karna|"
    r"bhejo|bhai|zaroor|turant|jaldi|abhi|kal|aaj|bahut|bohot)\b",
    re.IGNORECASE,
)


def detect_language(text):
    if DEVANAGARI.search(text):
        return "hi"

    hits = len(HINGLISH_HINT.findall(text))
    tokens = max(1, len(text.split()))

    if hits / tokens >= 0.15:
        return "hinglish"

    return "en"


def readings(sentence):
    """
    The spellings of `sentence` the feature tables should be run over.

    One for English and for romanised Hindi; two for Devanagari, which the
    tables cannot read as it stands. The original is always first, and is
    the only one anything outside this module ever sees.
    """
    second = romanised(sentence)

    return [sentence, second] if second else [sentence]


def _any(pattern, texts, match=False):
    """True when `pattern` fires in any spelling of the sentence."""
    find = pattern.match if match else pattern.search

    return any(bool(find(text)) for text in texts)


def score(sentence, entities, time_refs, texts=None):
    """
    Return (confidence, check_worthy, signals).

    `texts` is the spellings to read the sentence in, defaulting to
    `readings(sentence)` — which adds a romanisation for Devanagari and
    nothing at all for anything else.
    """
    tokens = sentence.split()
    n = len(tokens)

    texts = list(texts) if texts else readings(sentence)

    labels = {e.label for e in entities}
    named = labels & {"PERSON", "ORG", "GPE", "PROPER"}
    quantified = labels & {"NUMBER", "MONEY", "PERCENT"}

    # A date written in Devanagari ("कल से") is a time reference exactly
    # the way "kal se" is. The reference itself is deliberately not kept:
    # its offsets are into the romanisation rather than into the sentence
    # the Claim stores, and only whether there *is* one carries weight.
    dated = bool(time_refs) or any(
        bool(extract_time_refs(text)) for text in texts[1:]
    )

    signals = {
        "tokens": n,
        "question": _any(QUESTION, texts),
        "opinion": _any(OPINION, texts),
        "personal": _any(PERSONAL, texts),
        "person_marked": _any(FIRST_SECOND_PERSON, texts),
        "greeting": _any(GREETING, texts, match=True),
        "imperative_only": _any(IMPERATIVE_ONLY, texts, match=True),
        "has_verb": _any(VERB_LIKE, texts),
        "named_entity": bool(named),
        "quantity": bool(quantified),
        "time_ref": dated,
        "attribution": _any(ATTRIBUTION, texts),
        "policy": _any(POLICY, texts),
        "health": _any(HEALTH, texts),
        "chain_offer": _any(CHAIN_OFFER, texts),
        "prediction": _any(PREDICTION, texts),
        "causal": _any(CAUSAL, texts),
        "event": _any(EVENT, texts),
        "urgency": _any(URGENCY, texts),
        "url": "URL" in labels,
        "contact": bool(labels & {"PHONE", "UPI"}),
    }

    # Hard rejects.
    if signals["greeting"] or n < MIN_TOKENS or n > MAX_TOKENS:
        return 0.0, False, signals

    value = 0.30

    if signals["has_verb"]:
        value += 0.05
    if signals["named_entity"]:
        value += 0.15
    if signals["quantity"]:
        value += 0.15
    if signals["time_ref"]:
        value += 0.10
    if signals["attribution"]:
        value += 0.10
    if signals["policy"] or signals["health"] or signals["event"]:
        value += 0.10
    if signals["causal"]:
        value += 0.05
    if signals["chain_offer"]:
        value += 0.15
    if signals["contact"]:
        value += 0.10
    if signals["urgency"]:
        value += 0.05

    if signals["question"]:
        value -= 0.35
    if signals["opinion"]:
        value -= 0.25
    # A name does not stop a message being chat: "Rahul, call me when you
    # land" is addressed to somebody, not asserted to the world, and the
    # name is the least surprising thing in it. Only a quantity earns the
    # exemption, because "they credited me Rs 5,000" is a factual
    # statement that happens to be phrased personally.
    #
    # This used to read `not (named_entity or quantity)`, which meant any
    # stray capitalised word switched the rule off entirely — and in
    # romanised Hindi the first word of every sentence looked like one.
    # ...but a forward that opens "Bhai suno" and then attributes a health
    # claim to the WHO is a rumour wearing a vocative, and the commonest
    # shape a rumour arrives in. So the chat penalty only applies to a
    # sentence that asserts nothing checkable: no attribution, no policy
    # or health or event subject, no offer, no number. Those are the
    # things a person forwards; "kal ghar aa raha hoon" has none of them.
    asserts_something = (
        signals["attribution"] or signals["policy"] or signals["health"]
        or signals["event"] or signals["chain_offer"] or signals["quantity"]
        or signals["urgency"] or signals["contact"] or signals["url"]
    )

    if (signals["personal"] or signals["person_marked"]) and not asserts_something:
        value -= 0.30
    if signals["imperative_only"] and not (signals["quantity"] or signals["chain_offer"]):
        value -= 0.20

    value = max(0.0, min(1.0, round(value, 3)))

    return value, value >= CHECK_WORTHY_THRESHOLD, signals


def classify(sentence, signals, entities):
    """
    Pick the single most specific claim type. Order matters: earlier
    checks are more specific / more actionable for the fact-checker.
    """
    labels = {e.label for e in entities}

    if signals["chain_offer"] and (signals["contact"] or signals["imperative_only"] or "MONEY" in labels):
        return "chain_offer"

    if signals["health"]:
        return "health"

    if signals["policy"]:
        return "policy"

    if signals["attribution"]:
        return "attribution"

    if "MONEY" in labels:
        return "money"

    if "PERCENT" in labels:
        return "statistic"

    if signals["event"]:
        return "event"

    if "NUMBER" in labels and signals["has_verb"]:
        return "statistic"

    if signals["causal"]:
        return "causal"

    if signals["prediction"]:
        return "prediction"

    if signals["time_ref"]:
        return "event"

    if signals["chain_offer"]:
        return "chain_offer"

    return "generic"


def attributed_to(sentence, entities):
    """
    Best-effort speaker for attribution claims: the nearest named entity
    before the reporting verb, else the phrase after "according to".
    """
    m = re.search(r"(?i:according to|as per)\s+(?i:the\s+)?([A-Z][\w .&]+?)(?:,|\s+(?:the|a|an)\b|$)", sentence)

    if m:
        return m.group(1).strip().rstrip(".,;:!")

    verb = ATTRIBUTION.search(sentence)

    if not verb:
        return None

    candidates = [
        e for e in entities
        if e.label in ("PERSON", "ORG", "GPE", "PROPER") and e.end <= verb.start()
    ]

    if candidates:
        return candidates[-1].text

    return None
