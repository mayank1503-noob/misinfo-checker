# Data requirements for the evaluation set

`data/eval_cases.jsonl` — 41 labelled messages. This document says what
has to be in it, what each field means, and what the resulting numbers
are and are not evidence of.

## The shape of a case

One JSON object per line:

```json
{
  "id": "hot_water_hinglish",
  "text": "Bhai suno, WHO ne kaha hai ki garam paani peene se corona theek ho jata hai.",
  "language": "hinglish",
  "category": "health",
  "expect": {
    "verdict": "false",
    "check_worthy": true,
    "retrieves": "seed_covid_hot_water"
  },
  "in_corpus": true,
  "note": "Retrieved nothing before translit.py."
}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | unique, stable; it is what error lists print |
| `text` | yes | the message exactly as a user would send it |
| `language` | yes | `en`, `hi` or `hinglish` — the report splits on this |
| `category` | yes | `scam`, `health`, `govt`, `money`, `event`, `misc`, `true`, `chat` |
| `expect.verdict` | yes | one of the five labels in `backend/verdict.py` |
| `expect.check_worthy` | yes | should stage 2 find at least one claim worth checking |
| `expect.retrieves` | yes | the seed id stage 3b should return, or `null` |
| `in_corpus` | yes | can the demo index settle this at all |
| `note` | no | why the case is in the set; write one for anything non-obvious |

`dataset.validate()` enforces all of this, including the rule that a case
naming a seed id cannot also be `in_corpus: false` — that combination
would silently corrupt the split the whole report rests on.

## What the set has to contain

**1. Both directions of the asymmetry.** The set is useless if it only
contains rumours. Roughly 40% of it (17 of 41) is messages that must
*not* be accused: real chat, a true claim, and rumours the index has
never heard of. Without those, a system that answered "false" to
everything would score 100%.

**2. Every language, on the same claims.** The same rumour appears in
English, Devanagari Hindi and romanised Hinglish (hot water, free laptop,
cow urine, UPI cashback, note GPS). That is what makes the per-language
column a like-for-like comparison rather than three different tests, and
it is how the transliteration work in DECISIONS.md O1 is kept honest.

**3. Adversarial near-misses.** Three `benign_*` cases are true, sensible
statements worded like the rumour they sit next to — "boiling water
before drinking it reduces the risk of waterborne disease" against the
hot-water cure, "never share your UPI PIN" against the PIN scam. These
are where a retrieval change that loosens matching will show up first,
and calling one of them false is the worst thing the system can do.

**4. Known regressions, pinned as cases.** `chat_salt` exists because
transliteration pushed "Namak lekar aana ghar aate waqt" from 0.257 to
0.459 against the salt-shortage rumour when that sentence is handed
straight to the seed index. Run through the whole pipeline it does *not*
stray — stage 2 extracts a shorter claim and the routed query falls back
under the floor — so the case currently passes. That is the point: a case
is how a known problem stops being folklore in a decisions doc and starts
being something a run can tell you about, including telling you it is not
reproducing.

**5. Out-of-corpus claims that are plausible.** `oos_*` cases are the
kind of thing that really circulates and that a 32-entry index cannot
possibly settle. The expected answer is `unverified`, and getting them
right is a measure of restraint, not knowledge.

## What these numbers are not

**The corpus is mostly the answer key.** 24 of 41 cases have their answer
in `data/seed_factchecks.jsonl`, which is 32 *synthetic* demo records. On
those cases the harness measures whether retrieval, stance and the
verdict rules work — genuinely useful, and the thing that regresses when
someone changes a threshold. It does not measure whether the system knows
anything about the world, and an accuracy figure from them must never be
quoted as "the system is N% accurate at detecting misinformation".

**The set is small and hand-written.** 41 cases, written by the same
people who wrote the pipeline, which is the classic way to encode your
own blind spots. It catches regressions. It does not establish
performance.

**No images or video.** Stage 5 and the recycled-media path are not
covered here at all; `backend/tests/test_images.py` is what exercises
those. A case set for them needs fixture images and belongs in its own
file.

**`unverified` is doing heavy lifting.** 17 cases expect it. It is the
right answer for all of them, but it means a system that had simply
broken and returned `unverified` for everything would score around 40%.
Read `false_accusations` and the `in_corpus` / `out_of_corpus` split
together, never the single accuracy number.

## Running it on a small machine

Stage 2's default `auto` backend prefers the transformer extractor, and on
an 8 GB box that does not fit alongside the retrieval embedder — every case
fails with `DefaultCPUAllocator: not enough memory` and the report says
`STAGES THAT FAILED: claims`. Two knobs make the set runnable there:

```powershell
$env:CLAIM_BACKEND = "heuristic"    # stage 2 without the NER model
python -m backend.evaluation --mode retrieval          # ~110 s for 41 cases
python -m backend.evaluation --mode full --no-nli      # ~85 s, verdicts are a floor
```

Both narrow what is being measured, and both are recorded in the report
and in the JSON (`claims_backend`, `nli`) so a run cannot be quoted as
something it was not.

## Growing the set

Append; do not rewrite. Case ids are referenced from DECISIONS.md and
from error output, so keep them stable. When you add a case:

- give it a `note` if the reason it exists is not obvious from the text;
- prefer a real forward you have actually seen over an invented one;
- when you add a rumour, add the benign near-miss that goes with it —
  otherwise the set drifts towards rewarding an over-eager matcher;
- if the answer is not in `data/seed_factchecks.jsonl`, set
  `in_corpus: false` and expect `unverified`. Adding a seed entry to make
  a case pass is how an evaluation set quietly turns into a memorisation
  test.

`backend/evaluation/_build_cases.py` regenerates the file from a Python
list, which is easier to review in a diff than raw JSONL; it is kept for
that reason and is not imported by the harness.
