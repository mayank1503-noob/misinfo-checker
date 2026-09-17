"""
Romanised Hindi <-> Devanagari, for every stage that needs one script
read as the other.

The multilingual embedder the seed index and stage 4 share
(paraphrase-multilingual-MiniLM-L12-v2) was trained on Devanagari Hindi
and on English. It was not trained on the Latin-script Hindi that
actually circulates on WhatsApp, so "garam paani peene se corona theek
ho jata hai" lands at 0.32 against the hot-water debunk while its
Devanagari spelling lands at 0.48 — the first is below the 0.45
retrieval floor and the second is above it. The rumour is the same
rumour; only the keyboard was different.

So before a query is embedded it is offered to this module, which
returns a Devanagari rewriting when — and only when — the text really
looks like romanised Hindi. The retriever then embeds both spellings and
scores each corpus entry on whichever matches better. Two consequences
are deliberate:

  * **The floor does not move.** A Hinglish claim is not matched by
    lowering the bar; it is matched by being spelled the way the model
    was trained to read. 0.45 still means 0.45.
  * **English and Devanagari are untouched.** `devanagari()` returns
    None for both, no second vector is built, and their scores are
    bit-for-bit what they were before this module existed.

## Why a lexicon rather than a transliteration scheme

`indic-transliteration` implements ITRANS, HK and friends, which are
*lossless* schemes: they need "paanii", "ThIk", "bachchoM". Nobody types
that. Real romanisation is lossy and inconsistent — paani/pani,
theek/thik, nahi/nahin/nhi — so the mapping has to be phonetic and
forgiving, not a decoder. A lexicon keyed by a normalised spelling
(`_fold`) gives exactly that, and it has the property a
retrieval-critical component needs most: it can only ever change words
it has been told about. There is no rule engine that can quietly mangle
an English sentence.

The cost is coverage: a Hindi word not in the lexicon stays in Latin
script. That is the right failure — it leaves the word exactly as the
retriever would have seen it anyway.

## The gate

Only `HINDI` words — native Hindi with no English homograph — count as
evidence that a sentence is romanised Hindi. `LOANWORDS` (English words
Hindi has absorbed: bank, laptop, doctor) and `AMBIGUOUS` (Hindi words
spelled like English ones: "to", "the", "me") are rewritten once the
gate has opened, but never help open it. Without that split, "Please
send me the meeting notes" would read as Hindi on the strength of
"me" and "the".
"""

import re


# At least this share of a sentence's words, and at least this many
# words, must be unambiguous Hindi before anything is rewritten. Two
# words is what separates a Hindi sentence from an English one that
# happens to contain "sarkar".
MIN_HINDI_RATIO = 0.18
MIN_HINDI_WORDS = 2

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

_WORD = re.compile(r"[A-Za-z]+")


def _fold(word):
    """
    A spelling-insensitive key for a romanised Hindi word.

    Collapses the ways the same sound gets typed — paani/pani,
    theek/thik, jhooth/jhuth, zaroor/jaroor, woh/voh — so the lexicon
    needs one entry per word rather than one per typist.
    """
    word = word.lower()

    # Order matters. A run of three or more is someone leaning on the
    # key ("bahuuut") and is squeezed first; the doubles that survive are
    # the ones that carry a long vowel (paani, theek, jhooth) and are
    # mapped before the general collapse, which would otherwise turn
    # "theek" into "thek" and lose the match with "thik".
    word = re.sub(r"(.)\1{2,}", r"\1", word)
    word = word.replace("ee", "i").replace("oo", "u")
    word = word.replace("aa", "a").replace("ii", "i").replace("uu", "u")
    word = re.sub(r"(.)\1+", r"\1", word)            # accha -> acha
    word = word.replace("z", "j").replace("w", "v").replace("q", "k")

    return word


def _table(pairs):
    """Fold a written-out mapping into its lookup form."""
    return {_fold(roman): devanagari for roman, devanagari in pairs.items()}


# Native Hindi. These are the words that open the gate.
HINDI = _table({
    # to be, to do, to go - the grammar that makes a sentence Hindi
    "hai": "है", "hain": "हैं",
    "tha": "था", "thi": "थी",
    "hoga": "होगा", "hogi": "होगी",
    "honge": "होंगे", "hona": "होना",
    "ho": "हो", "hota": "होता",
    "hoti": "होती", "hote": "होते",
    "hone": "होने", "hoon": "हूँ",
    "hun": "हूँ", "hua": "हुआ",
    "hui": "हुई",
    "jata": "जाता", "jati": "जाती",
    "jate": "जाते", "jana": "जाना",
    "jayega": "जाएगा",
    "jayegi": "जाएगी",
    "jayenge": "जाएंगे",
    "gaya": "गया", "gayi": "गई",
    "gaye": "गए",
    "raha": "रहा", "rahi": "रही",
    "rahe": "रहे", "rahega": "रहेगा",
    "kiya": "किया", "kiye": "किए",
    "karta": "करता", "karti": "करती",
    "karte": "करते", "karna": "करना",
    "karo": "करो", "kare": "करे",
    "kar": "कर", "karke": "करके",
    "karega": "करेगा",
    "karenge": "करेंगे",
    "diya": "दिया", "dena": "देना",
    "de": "दे", "dete": "देते",
    "deti": "देती", "liya": "लिया",
    "lena": "लेना", "mila": "मिला",
    "mile": "मिले",
    "milega": "मिलेगा",
    "milegi": "मिलेगी",
    "pane": "पाने", "pana": "पाना",
    "daalna": "डालना",
    "dalna": "डालना", "daal": "डाल",
    "bhejo": "भेजो", "bhej": "भेज",
    "bhejna": "भेजना",
    "batao": "बताओ", "dekho": "देखो",
    "suno": "सुनो", "padho": "पढ़ो",
    "peene": "पीने", "pina": "पीना",
    "piye": "पिए", "khana": "खाना",
    "khane": "खाने", "aana": "आना",
    "badha": "बढ़ा",
    "badhakar": "बढ़ाकर",
    "ghata": "घटा", "bik": "बिक",
    "bech": "बेच", "kharid": "खरीद",

    # negation, question words, connectives
    "nahi": "नहीं", "nahin": "नहीं",
    "nhi": "नहीं", "mat": "मत",
    "kya": "क्या", "kyun": "क्यों",
    "kyu": "क्यों", "kaise": "कैसे",
    "kab": "कब", "kahan": "कहाँ",
    "kaun": "कौन", "kitna": "कितना",
    "aur": "और", "lekin": "लेकिन",
    "magar": "मगर", "agar": "अगर",
    "phir": "फिर", "bhi": "भी",
    "sirf": "सिर्फ", "warna": "वरना",

    # postpositions and determiners
    "ke": "के", "ki": "की", "ka": "का",
    "ko": "को", "se": "से",
    "mein": "में", "tak": "तक",
    "liye": "लिए", "bina": "बिना",
    "saath": "साथ", "baad": "बाद",
    "pehle": "पहले", "upar": "ऊपर",
    "wala": "वाला", "wale": "वाले",
    "wali": "वाली", "sab": "सब",
    "sabhi": "सभी", "har": "हर",
    "koi": "कोई", "kuch": "कुछ",
    "bahut": "बहुत", "bohot": "बहुत",
    "bohut": "बहुत",
    "zyada": "ज़्यादा",
    "thoda": "थोड़ा",

    # pronouns
    "yeh": "यह", "woh": "वह",
    "iska": "इसका", "unka": "उनका",
    "apna": "अपना", "apne": "अपने",
    "apni": "अपनी", "aap": "आप",
    "aapka": "आपका", "aapke": "आपके",
    "aapko": "आपको", "hum": "हम",
    "hamara": "हमारा",
    "hamein": "हमें", "tum": "तुम",
    "mera": "मेरा", "meri": "मेरी",

    # time
    "aaj": "आज", "kal": "कल",
    "parso": "परसों", "abhi": "अभी",
    "turant": "तुरंत",
    "jaldi": "जल्दी",
    "zaroor": "ज़रूर",
    "hamesha": "हमेशा", "samay": "समय",
    "din": "दिन", "raat": "रात",
    "shaam": "शाम", "subah": "सुबह",
    "mahina": "महीना",
    "mahine": "महीने", "saal": "साल",
    "hafta": "हफ्ता",
    "hafte": "हफ्ते",

    # qualities
    "naya": "नया",
    "purana": "पुराना",
    "bada": "बड़ा", "chota": "छोटा",
    "accha": "अच्छा", "bura": "बुरा",
    "sasta": "सस्ता",
    "mehnga": "महंगा", "asli": "असली",
    "nakli": "नकली",
    "khatam": "खत्म", "shuru": "शुरू",
    "garam": "गरम", "thanda": "ठंडा",
    "theek": "ठीक", "khali": "खाली",
    "muft": "मुफ्त", "kami": "कमी",
    "poora": "पूरा",

    # people
    "bachcha": "बच्चा",
    "bachche": "बच्चे",
    "bachchon": "बच्चों",
    "ladka": "लड़का",
    "ladki": "लड़की",
    "aadmi": "आदमी", "aurat": "औरत",
    "logon": "लोगों",
    "logo": "लोगों",
    "vyakti": "व्यक्ति",
    "grahak": "ग्राहक",
    "upbhokta": "उपभोक्ता",
    "janta": "जनता",
    "nagrik": "नागरिक",
    "bhai": "भाई", "behen": "बहन",
    "dost": "दोस्त",
    "chhatra": "छात्र",
    "kisan": "किसान", "neta": "नेता",
    "mantri": "मंत्री",
    "pradhanmantri": "प्रधानमंत्री",
    "sena": "सेना",

    # the recurring rumour vocabulary
    "sarkar": "सरकार",
    "sarkari": "सरकारी",
    "yojana": "योजना", "desh": "देश",
    "rajya": "राज्य",
    "chunav": "चुनाव",
    "seema": "सीमा", "paisa": "पैसा",
    "paise": "पैसे",
    "rupaye": "रुपये",
    "rupay": "रुपये", "khata": "खाता",
    "kist": "किस्त",
    "inaam": "इनाम", "keemat": "कीमत",
    "daam": "दाम", "dukan": "दुकान",
    "dukano": "दुकानों",
    "sandesh": "संदेश", "khabar": "खबर",
    "samachar": "समाचार", "sach": "सच",
    "jhooth": "झूठ", "afwah": "अफवाह",
    "tasveer": "तस्वीर",
    "paani": "पानी", "doodh": "दूध",
    "namak": "नमक", "pyaz": "प्याज़",
    "aloo": "आलू", "kela": "केला",
    "tel": "तेल", "bijli": "बिजली",
    "bimari": "बीमारी",
    "ilaj": "इलाज", "dawa": "दवा",
    "dawai": "दवाई", "sehat": "सेहत",
    "aspatal": "अस्पताल",
    "tika": "टीका",
    "gomutra": "गोमूत्र",
    "gaumutra": "गोमूत्र",
    "kapoor": "कपूर", "laung": "लौंग",
    "ajwain": "अजवाइन",
    "bandhyapan": "बंध्यापन",
    "ghar": "घर", "padhai": "पढ़ाई",
    "jeevan": "जीवन", "maut": "मौत",
    "jaan": "जान",
    "lapata": "लापता", "chori": "चोरी",
    "thagi": "ठगी", "dhokha": "धोखा",
    "kharab": "खराब", "adhar": "आधार",
    "aadhaar": "आधार", "aadhar": "आधार",
    # First and second person. The lexicon had none of these, which left
    # every Devanagari sentence about its own sender unreadable to the
    # check-worthiness table that is supposed to recognise chat
    # (DECISIONS.md O2/O7).
    "maine": "मैंने", "mujhe": "मुझे",
    "mujhko": "मुझको",
    "hamein": "हमें", "humein": "हमें",
    "hamara": "हमारा",
    "hamari": "हमारी",
    "tumhe": "तुम्हें",
    "tumhara": "तुम्हारा",
    "tumhari": "तुम्हारी",
    "tera": "तेरा", "teri": "तेरी",
    "aapko": "आपको", "aapka": "आपका",
    "aapki": "आपकी",
    "apna": "अपना", "apni": "अपनी",
    "hoon": "हूँ",
})

# English that Hindi has absorbed. Rewritten once the gate is open - a
# Devanagari sentence reads better to the model with them in script -
# but never counted as evidence that the sentence is Hindi.
LOANWORDS = _table({
    "bank": "बैंक", "note": "नोट",
    "card": "कार्ड", "link": "लिंक",
    "school": "स्कूल",
    "college": "कॉलेज",
    "doctor": "डॉक्टर",
    "police": "पुलिस",
    "station": "स्टेशन",
    "train": "ट्रेन", "vote": "वोट",
    "photo": "फोटो",
    "video": "वीडियो", "chip": "चिप",
    "tower": "टावर",
    "virus": "वायरस",
    "cancer": "कैंसर",
    "corona": "कोरोना",
    "vaccine": "वैक्सीन",
    "oxygen": "ऑक्सीजन",
    "petrol": "पेट्रोल",
    "gas": "गैस", "mobile": "मोबाइल",
    "laptop": "लैपटॉप",
    "internet": "इंटरनेट",
    "message": "मैसेज", "kilo": "किलो",
    "litre": "लीटर", "jio": "जियो",
})

# Hindi words that are also ordinary English words. Rewritten once the
# gate is open, never counted toward it.
AMBIGUOUS = _table({
    "to": "तो", "the": "थे", "me": "में",
    "do": "दो", "par": "पर", "ye": "ये",
    "wo": "वो", "na": "ना", "ya": "या",
    "aa": "आ", "band": "बंद", "log": "लोग",
    "kam": "कम", "man": "मन", "hi": "ही",
    # "main" is both the Hindi first person and an English adjective, so
    # it is rewritten once the gate is open and never helps open it.
    "main": "मैं", "mai": "मैं",
})

LEXICON = {}
LEXICON.update(LOANWORDS)
LEXICON.update(AMBIGUOUS)
LEXICON.update(HINDI)                                # Hindi wins a clash


def is_devanagari(text):
    return bool(DEVANAGARI.search(text or ""))


def hindi_ratio(text):
    """
    (share, count) of the Latin words that are unambiguously Hindi.

    Only `HINDI` counts, which is what keeps an English sentence
    containing "bank" and "the" from reading as Hinglish.
    """
    words = _WORD.findall(text or "")

    if not words:
        return 0.0, 0

    hits = sum(1 for word in words if _fold(word) in HINDI)

    return hits / len(words), hits


def looks_romanised(text):
    """True when `text` is Latin-script Hindi worth transliterating."""
    if not text or is_devanagari(text):
        return False

    ratio, hits = hindi_ratio(text)

    return hits >= MIN_HINDI_WORDS and ratio >= MIN_HINDI_RATIO


def transliterate(text):
    """
    Rewrite every known romanised word, leaving everything else alone.

    Applied unconditionally — `devanagari()` is the gated entry point.
    Numbers, punctuation, acronyms and unrecognised words survive
    verbatim, which is why an unknown Hindi word costs coverage but
    never correctness.
    """
    def replace(match):
        word = match.group(0)

        # ALL-CAPS is an acronym (SBI, UPI, WHO, UNESCO) and is left as
        # it is: the model knows those, and they are what a Hinglish
        # claim shares with an English fact-check.
        if word.isupper() and len(word) > 1:
            return word

        return LEXICON.get(_fold(word), word)

    return _WORD.sub(replace, text or "")


def devanagari(text):
    """
    A Devanagari rewriting of `text`, or None to leave it alone.

    None means one of: empty, already Devanagari, not enough Hindi to be
    sure, or nothing in it was recognised. Callers treat None as "embed
    the original and nothing else", so every None is a guarantee that
    behaviour is unchanged.
    """
    if not looks_romanised(text):
        return None

    rewritten = transliterate(text)

    return rewritten if is_devanagari(rewritten) else None


# Devanagari -> the romanisation the feature tables are written in. Built
# by inverting LEXICON once, at import. Where several spellings map to the
# same Devanagari word ("paani" and "pani" both give "पानी") the first
# one wins, so the result is stable across runs rather than dict-order
# dependent.
_ROMAN = {}

for _roman, _dev in LEXICON.items():
    _ROMAN.setdefault(_dev, _roman)

# The danda and double danda (U+0964/U+0965) are sentence punctuation,
# not letters. Leaving them inside the word class made "हूँ।" a single
# unknown token, so the word before a full stop never romanised.
_DEV_WORD = re.compile(r"[ऀ-ॣ०-ॿ]+")


def romanised(text):
    """
    A Latin-script rewriting of Devanagari `text`, or None to leave it alone.

    The mirror of `devanagari()`, and it exists for the same reason in
    reverse: stage 2's check-worthiness features are a table of Latin and
    romanised-Hindi patterns ("sarkar", "muft", "ilaaj", "hai"), so a
    Devanagari sentence matches none of them and scores the bare base
    weight however plainly factual it is (DECISIONS.md O7).

    None means empty or nothing recognised, and a None is a guarantee
    that the caller's behaviour is exactly what it was before. A word the
    lexicon does not know stays in Devanagari, where it matches nothing —
    which is precisely what it did anyway.
    """
    if not text or not is_devanagari(text):
        return None

    rewritten = _DEV_WORD.sub(
        lambda match: _ROMAN.get(match.group(0), match.group(0)), text
    )

    return rewritten if rewritten != text else None


def spellings(text):
    """
    Every spelling of `text` worth reading, the original always first.

    `variants()` answers the retrieval question ("what should I embed?")
    and only ever adds Devanagari. This answers the wider one — "what
    should a reader that knows one script be shown?" — and so also
    romanises Devanagari. Two entries at most, and exactly one for
    English.
    """
    text = (text or "").strip()

    if not text:
        return []

    other = devanagari(text) or romanised(text)

    return [text, other] if other else [text]


def variants(text):
    """
    The spellings of `text` worth embedding, original always first.

    One entry for English and for Devanagari; two for romanised Hindi.
    """
    text = (text or "").strip()

    if not text:
        return []

    hindi = devanagari(text)

    return [text, hindi] if hindi else [text]
