"""
The labelled evaluation set: loading, validating and slicing it.

The cases live in `data/eval_cases.jsonl`, one JSON object per line, the
same shape as `data/seed_factchecks.jsonl` uses — greppable, diffable in
review, and appendable without rewriting the file.

    {
      "id":       "hot_water_hinglish",
      "text":     "...garam paani peene se corona theek ho jata hai.",
      "language": "en" | "hi" | "hinglish",
      "category": "scam" | "health" | "govt" | "money" | "event"
                  | "misc" | "true" | "chat",
      "expect": {
        "verdict":      "false" | "misleading" | "true" | "disputed"
                        | "unverified",
        "check_worthy": true,               # stage 2 should find a claim
        "retrieves":    "seed_covid_hot_water" | null   # stage 3b target
      },
      "in_corpus": true,        # is the answer in the seed index at all?
      "note":      "why this case is in the set"
    }

`in_corpus` is the field that keeps the numbers honest. A case whose
answer sits in the demo index is testing *retrieval and stance*, not
knowledge; a case with `in_corpus: false` is testing whether the system
knows when to say nothing. Reporting the two together would let a rise in
one hide a fall in the other, so `backend.evaluation.metrics` always
splits them.

See DATA.md in this package for what the set has to contain and why.
"""

import json
import logging
import os


log = logging.getLogger(__name__)


DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval_cases.jsonl",
)

LANGUAGES = ("en", "hi", "hinglish")

VERDICTS = ("false", "misleading", "true", "disputed", "unverified")

# A verdict that publicly contradicts the message. Getting one of these
# wrong is not the same kind of error as missing a rumour, so the metrics
# count them separately.
ACCUSATIONS = ("false", "misleading")

REQUIRED = ("id", "text", "language", "category", "expect")


def path():
    return os.getenv("EVAL_CASES", DEFAULT_PATH)


class Case(dict):
    """A case, with the accessors the runner and metrics want."""

    @property
    def id(self):
        return self["id"]

    @property
    def text(self):
        return self["text"]

    @property
    def language(self):
        return self.get("language", "en")

    @property
    def category(self):
        return self.get("category", "misc")

    @property
    def expected_verdict(self):
        return self["expect"]["verdict"]

    @property
    def expects_check_worthy(self):
        return bool(self["expect"].get("check_worthy"))

    @property
    def expected_seed_id(self):
        """The seed entry stage 3b should find, or None if there isn't one."""
        return self["expect"].get("retrieves")

    @property
    def in_corpus(self):
        return bool(self.get("in_corpus"))

    @property
    def must_not_accuse(self):
        """
        True when calling this message false would be a wrong accusation.

        Every case that is not itself a rumour: real chat, a true claim,
        and — importantly — the out-of-corpus rumours, where the honest
        answer is "nothing found" and a confident `false` would be the
        system inventing one.
        """
        return self.expected_verdict not in ACCUSATIONS


def validate(case, index=None):
    """Raise ValueError describing the first thing wrong with `case`."""
    where = f"case {index}" if index is not None else "case"

    for field in REQUIRED:
        if field not in case:
            raise ValueError(f"{where}: missing {field!r}")

    where = f"case {case['id']!r}"

    if not str(case.get("text") or "").strip():
        raise ValueError(f"{where}: empty text")

    if case["language"] not in LANGUAGES:
        raise ValueError(
            f"{where}: language {case['language']!r} not in {LANGUAGES}"
        )

    expect = case["expect"]

    if expect.get("verdict") not in VERDICTS:
        raise ValueError(
            f"{where}: expected verdict {expect.get('verdict')!r} not in {VERDICTS}"
        )

    if not isinstance(expect.get("check_worthy"), bool):
        raise ValueError(f"{where}: expect.check_worthy must be true or false")

    seed_id = expect.get("retrieves")

    if seed_id is not None and not isinstance(seed_id, str):
        raise ValueError(f"{where}: expect.retrieves must be a seed id or null")

    # A case cannot name a seed entry and also claim not to be in the
    # corpus; that combination would silently break the honest/not-honest
    # split the whole report rests on.
    if seed_id and not case.get("in_corpus"):
        raise ValueError(f"{where}: names {seed_id!r} but is marked in_corpus=false")

    return case


def load(source=None):
    """
    Every case, validated, in file order.

    Unlike the seed corpus this does *not* skip a malformed line. A
    corpus is data and data has typos; an evaluation set is the thing the
    numbers are computed against, and quietly dropping a case would move
    every score in the report without saying so.
    """
    source = source or path()
    cases = []

    with open(source, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source} line {number}: {error}") from error

            cases.append(Case(validate(row, index=number)))

    ids = [case.id for case in cases]
    duplicates = {name for name in ids if ids.count(name) > 1}

    if duplicates:
        raise ValueError(f"{source}: duplicate case ids {sorted(duplicates)}")

    return cases


def select(cases, language=None, category=None, ids=None, in_corpus=None):
    """The subset matching every filter given; None means "don't filter"."""
    def keep(case):
        if language and case.language != language:
            return False

        if category and case.category != category:
            return False

        if ids and case.id not in ids:
            return False

        if in_corpus is not None and case.in_corpus != in_corpus:
            return False

        return True

    return [case for case in cases if keep(case)]
