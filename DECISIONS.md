# Decisions

Choices made while building Stages 3a / 3b / 4 / 5 autonomously, with no one
available to ask. Each entry says what was ambiguous, what was chosen, and why.

---

## Setup

**D1 — `git init` in a directory that was not a repository.**
The brief says "commit after each milestone", but the project was not under
version control. Initialised a local repository and committed there. Nothing is
pushed, and no remote is configured.

**D2 — Ran with the system Python, not `venv/`.**
The checked-in `venv/` is broken: its `pyvenv.cfg` points at a Python install
under a different user's home directory (`C:\Users\vansh\...`). Recreating it
would have meant re-downloading torch, so the suite runs on the system
interpreter (Python 3.14, pydantic 2.12). `venv/` is left untouched.

---

## Architecture

**D3 — Stage 3a lives in `backend/graph/`; `backend/evidence/` is Stage 3b.**
The brief puts the evidence graph in `backend/graph/` and retrieval in
`backend/evidence/`. An earlier iteration of this project had put a pydantic
evidence graph in `backend/evidence/` (`schema.py`, `graph.py`). That module is
superseded: the graph is now a NetworkX `MultiDiGraph` in `backend/graph/`, with
the node and edge vocabulary the brief specifies, and `backend/evidence/` was
rebuilt for retrieval. `backend/evidence/graph.py` is kept as a thin
compatibility shim that re-exports `backend.graph.EvidenceGraph`, so any stale
import keeps working. `backend/tests/test_evidence_graph.py` was rewritten to
test the NetworkX store — it covers the same concern (the evidence graph)
against the API that now exists.

**D4 — Retrieval talks HTTP with `requests`, not vendor SDKs.**
Google Fact Check Tools, Tavily and SerpAPI are all plain JSON over HTTPS. Using
`requests` directly (rather than `tavily-python` / `google-api-python-client` /
`serpapi`) keeps the dependency list small, makes every call trivially cacheable
through one helper, and makes the tests mockable at a single seam
(`backend.common.http.get_json`).

**D5 — One disk cache and one fail-soft HTTP helper for every external call.**
`backend/common/cache.py` (JSON blobs under `cache/`, keyed by a sha1 of the
request) and `backend/common/http.py` (`get_json`, which logs and returns `None`
on any failure). Every retriever and the article fetcher go through them, so the
pipeline runs with no API keys, no network and no exceptions: a missing key
means that retriever yields nothing.

**D6 — `stance` returns `support`/`refute`; the graph stores `SUPPORTS`/`REFUTES`.**
The two vocabularies were already different (Stage 4 was written before the
graph). Rather than churn one of them, `backend/stance/graph_adapter.py` maps
between them in one place.

**D7 — A small rule-based verdict stage was added (`backend/verdict.py`).**
"Integrated" needs the API to return something other than the raw packet. The
verdict is deliberately simple and transparent — decisive fact-check hits first,
then weighted stance totals, then "unverified" — because the brief does not
specify a scoring model and a guessed one would be worse than an explainable
one. It writes `verdict` nodes into the graph, which the node vocabulary
already allowed for.

**D8 — Demo data is generated, marked, and never presented as real.**
`data/seed_factchecks.jsonl` (30 entries) and `data/seed_images/` are
hand-written stand-ins for a real index. Every record carries
`"demo": true` and a `demo_note`, the loader logs that it is demo data, and
`backend/graph` marks evidence from the seed index `demo=True` so a verdict
built on it can say so. They exist to make the pipeline runnable offline, not to
make claims about the real world.

**D9 — Image work reuses Stage 1's outputs and never recomputes them.**
`backend/images/` reads `embedding`, `description`, `ocr_text`, `exif_date`,
`ai_generated_score` and `frame_time` straight off the packet. The only models
it loads are CLIP (for caption/image consistency) and, for the index, the same
cached DINOv2 loader from `backend/analyzers/models.py`.

**D10 — FAISS is optional everywhere.**
Both the seed fact-check index and the image index import `faiss` lazily and
fall back to exact NumPy cosine over the same vectors when it is missing, so
`pip install faiss-cpu` is a speed-up, not a requirement. Tests cover the
fallback path.

**D11 — Every network-touching test stubs at the helper seam.**
No test makes a socket call. HTTP tests monkeypatch `backend.common.http.get_json`;
model tests monkeypatch the loader functions. The two opt-in integration tests
are gated behind `STANCE_TESTS_WITH_MODELS=1` and `CLAIM_TESTS_WITH_MODELS=1`.

**D12 — Models are warmed on the calling thread before any pool starts.**
Retrieval runs in two nested thread pools. Loading a sentence-transformer
inside one of those workers crashed the process outright on this machine —
first a Windows access violation, later a hard abort — because torch's own
weight materialisation is itself threaded. `collector.warm_up()` loads what
the retrievers need on the calling thread first; every later call is an
`lru_cache` hit, so it costs nothing but the ordering.

**D13 — The verdict is rules, not a model.**
Four rules in a fixed order: a decisive fact-check, then a recycled picture,
then weighted stance with both an absolute floor and a 2× dominance ratio,
then `unverified`. The brief did not specify a scoring model, and a guessed
one would be both unexplainable and unarguable. Every label carries the
evidence that produced it. `unverified` is a first-class answer: guessing
"false" on no evidence is how a checker loses the credibility it needs for
the cases where it is right.

**D14 — The image index floor is 0.95, set from measurement.**
The first demo placeholders were flat colour cards, and DINOv2 embedded them
so close together that one picture matched three archive entries at once.
Measured over the regenerated noise-textured cards: distinct images peak at
0.94 cosine, while the same image recompressed as JPEG at quality 70 scores
0.985. The floor sits in that gap, nearer the top, because a false match
tells someone a real photograph is recycled — the more expensive mistake.

**D15 — The test suite cannot reach the network, structurally.**
`backend/tests/conftest.py` blocks `get_json` / `post_json` / `get_text` for
every test, autouse, with no opt-out. A suite that *can* reach the internet
passes on a plane and fails in CI, or quietly spends someone's API quota.
The two tests that are about those helpers ask for the real ones through a
`real_http` fixture, and still reach no further than a stubbed `requests`.

---

## Agent layer

**D16 — The agents coordinate stages 1-5; they do not re-implement any of them.**
`backend/agents/tools.py` registers every stage function as a named tool
(`claims.extract`, `evidence.collect`, `stance.apply`, `verdict.decide`, ...)
and calls straight through to it. The only new code in the package that
touches evidence at all is `attach_candidates`, ten lines wrapping
`graph.add_evidence` for the targeted second retrieval round, which works
claim by claim where stage 3b's own collector works over a whole `ClaimSet`.
No stage was modified, and the linear `backend/pipeline.py` is untouched and
remains the default path.

**D17 — The agent layer is additive and opt-in, not a replacement.**
`run_agentic` returns the same dict `analyze` returns, plus `agents` and
`explanation`; `analyze(..., agentic=True)` and `?agentic=true` are the
switches. The bot, the API's `results` array and every existing test read the
same shape either way. A second response contract would have been a bigger
change to this project than the architecture itself.

**D18 — `verification` assembles; `backend/verdict.py` still decides.**
The obvious design is an agent that weighs the evidence and picks a label.
That agent would be the one part of the system whose answer could not be
explained, and D13 already settled that the verdict is rules. So the
verification agent runs stage 4, reads the graph totals and the decisive
hits, folds in the media findings, and calls `decide()` unchanged. What it
adds is the *abstention*: when the verdict comes back `unverified` it reports
`abstained`, which is what lets the orchestrator go back out for more
evidence before anything is written.

**D19 — Abstention is a status, not an error.**
`AgentResult.status` is one of `ok` / `abstained` / `skipped` / `failed`.
Four states rather than a boolean because "I was asked and found nothing",
"there was nothing here to ask about" and "I broke" are three different
facts, and a checker that reports them identically is a checker that cannot
say how confident it is. The explanation agent uses exactly this distinction:
"no source could be searched" and "the sources were searched and returned
nothing" are different sentences.

**D20 — One escalation round, never two.**
The first retrieval pass asks the precise sources (fact-check API, seed
index) and holds the web search in reserve; a `verification` abstention with
claims still open spends the reserve on just those claims with wider queries.
`max_rounds=2` and `max_steps=16` bound it. A loop that keeps retrieving
until it likes the answer eventually reports whatever it was looking for, and
the bill is unbounded long before that.

**D21 — Routing is a `Planner`; the default one is rules, not a model.**
`RulePlanner` decides from `AgentContext.observe()` — a flat, JSON-safe dict
of input type, claim counts and types, and which tools actually probe
available. `LLMPlanner` takes any `complete(prompt) -> str | dict` callable,
which is the entire provider contract: no SDK, no client object, no
tool-calling protocol. What a model returns is validated against the `Plan`
schema, a step naming an unknown agent is dropped, a route missing
verification or explanation is completed, and any failure falls back to the
rules with the fallback recorded in the trace. `revise()` (escalation) stays
with the rules even behind a model, per D20. The explanation writer is a
callable on the same terms.

**D22 — Tool probes are cached per run, and media probes only run when there
is media.**
`images.consistency.available()` imports `sentence_transformers`, and
therefore torch. Building the planner's observation asked it on every run,
including text-only forwards that can never use it: 86 seconds for a 13 ms
check on this machine. Probes now answer once per `ToolRegistry` (one per
run, so nothing is cached across runs and a test that switches a model off
still gets a fresh probe), and the media probes are skipped entirely when the
packet has no images — reported as `None`, meaning "not asked", rather than
`False`, which would have been a claim about the machine.

---

## Open issues

**O1 — Romanised Hinglish retrieval. RESOLVED, with one regression.**
The multilingual embedder handled Devanagari well — "गर्म पानी पीने से
कोरोना ठीक होता है" matched the English hot-water debunk at 0.48 — but
Latin-script Hindi did not: "garam paani peene se corona theek ho jata hai"
scored 0.32 against the same entry, below the 0.45 floor.

`backend/evidence/translit.py` now rewrites a romanised query into Devanagari
before it is embedded, and `seed_index.search` scores each corpus entry on
whichever spelling matches it better. The floor did not move.

Measured on the shipped 32-entry corpus with the real embedder, 22 queries:

| | before | after |
|---|---|---|
| Hinglish claims matched to the right entry | 6 / 10 | **10 / 10** |
| English | 5 / 5 | 5 / 5 (scores identical) |
| Devanagari Hindi | 4 / 4 | 4 / 4 (scores identical) |

English and Devanagari are untouched by construction, not by luck: the gate
returns None for both, one spelling is embedded, and the arithmetic is the
same one as before.

End to end on `backend/samples/hinglish_rumor.txt`, full pipeline, no keys:

    before   false: 1, unverified: 2   "...garam paani peene se corona
                                        theek ho jata hai"  -> unverified
    after    false: 2, unverified: 1   the same claim       -> false (0.58),
                                        on the hot-water debunk

**The cost, on a 16-sentence everyday-Hinglish set: one new false match.**
"Namak lekar aana ghar aate waqt" (bring salt on your way home) went 0.257 →
0.459 against the salt-shortage rumour, which is over the floor. Raising
Hinglish similarity raises it for chit-chat too, and salt-vs-salt is the
plausible end of that. Stage 4 finds no entailment and the verdict abstains,
so it costs a stance round rather than an accusation — but it is a real
regression and the evaluation harness should track it. (An unrelated
pre-existing false match, an English sentence about quarterly revenue at
0.498, is untouched by this change.)

Not done, and deliberately: the lexicon is ~330 words, so an unrecognised
Hindi word stays in Latin script. That failure leaves the word exactly as
the retriever saw it before, which is why coverage can grow later without
re-validating anything.

**O2 — Resolved: personal chat is no longer scored as check-worthy.**
Was: precision 81.2%, recall 78.8%, f1 80.0%, with 6 of the 8 chat messages
flagged. Two defects fed each other, and neither was in the weight table.

*The artifact.* `entities.py` read a lone capitalised word at the start of a
sentence as a proper noun. Every sentence starts with a capital, so this is the
weakest evidence there is, and it hits romanised Hindi hardest because its
everyday words are in no English vocabulary: "Kal", "Aaj", "Namak", "Amma" and
"Happy" each scored a `named_entity` worth +0.15. A guard for this already
existed but only fired on adjective-shaped words (`_ADJ_SHAPE`), so none of
these met it.

*The amplifier.* `checkworthy.py` read `personal and not (named_entity or
quantity)`, so the spurious name also switched off the −0.30 rule that exists
to catch chat. "Happy birthday bhai" scored exactly 0.45, the threshold.

The fixes are three, each a rule rather than a word list:

1. A sentence-initial single capitalised word must corroborate itself — an
   acronym, an org or place hint, a person title in front of it, the same word
   capitalised again later, **or a finite verb straight after it**. That last
   clause matters: "Modi announces ..." is a capitalised word doing a verb,
   which is what a name does and what "Kal shaam", "Namak lekar" and "Aaj
   office" — capital then noun — do not. Without it the guard ate real
   subjects, which the existing suite caught immediately.
2. A name no longer cancels the chat penalty; only a quantity does, because
   "they credited me Rs 5,000" is checkable however personally it is phrased.
3. A new `person_marked` signal, because romanised Hindi marks person on the
   verb: "aa raha hoon" (I am coming) and "kar paya" (I could) are first-person
   with no pronoun anywhere for `PERSONAL` to match.

The chat penalty is then gated on the sentence asserting nothing checkable. A
forward that opens "Bhai suno" and attributes a health claim to the WHO is a
rumour wearing a vocative, and that is the commonest shape a rumour arrives in;
an unguarded rule turned two real Hinglish rumours into misses before the gate
was added. Two vocabulary gaps the same cases exposed were filled: `sarkari` as
an inflection of `sarkar`, and `polio` in a health list that already named
dengue and malaria.

Measured, `--mode claims`, heuristic backend, all 41 cases:

| | before | after |
|---|---|---|
| precision | 81.2% | **96.3%** |
| recall | 78.8% | **78.8%** |
| f1 | 80.0% | **86.7%** |
| chat flagged check-worthy | 6 of 8 | **1 of 8** |
| Hinglish check-worthy f1 | 78.3% | **100%** |

**Recall did not move.** That is the number that mattered: precision bought by
dropping real claims would be worth nothing, and the intermediate states of this
change did exactly that before the gate and the subject-of-verb clause were
added.

The one remaining chat case is `chat_match_en`, "The cricket match was postponed
due to rain in Mumbai last night" — a verifiable statement about the world that
the dataset files under chat. It is not obviously an error, and anything that
suppressed it would also suppress `oos_metro_fare`, which has the same shape, so
it is left alone deliberately. `unesco_anthem_en` remains a miss for an
unrelated pre-existing reason (`OPINION` matches "the best"); it is pinned as a
strict xfail in `backend/tests/test_checkworthy_chat.py` so it stays visible.

Hindi (Devanagari) check-worthiness is unchanged at 50% f1 — that is O7, and it
is a segmentation problem, not this one.

---

**O6 — True health statements were labelled false. RESOLVED by a
relevance gate; it was the worst failure class the system had.**
Found by the harness on its first full run, which is what it was built
for. Two sensible, true messages are called **false**:

    "Boiling water before drinking it reduces the risk of
     waterborne disease."
        -> retrieved seed_covid_hot_water (the "hot water cures
           COVID" debunk), verdict: false

    "COVID-19 vaccines were tested in clinical trials before
     being approved."
        -> retrieved seed_vaccine_microchip and seed_5g_covid,
           verdict: false

The mechanism is clear from the per-stage numbers. These are the only two
of the 17 no-right-answer cases where retrieval strayed (11.8% stray
rate): the benign statement is close enough in embedding space to the
rumour it sits beside to clear the 0.45 floor. Retrieval returning them
is defensible — they *are* topically about hot water and about vaccines.
Turning that into a verdict of `false` is not.

**Important caveat: this run had the NLI model off** (`--no-nli`, which
was forced on this machine when O6 was measured; O3 has since lifted that
constraint, but the 41-case set has not been re-run), so stage 4 judged from
publisher ratings alone:
"a credible fact-checker rated the nearest matching claim false,
therefore false". Deciding that the benign statement is *not* the claim
the fact-check refutes is exactly the entailment model's job, so it may
well prevent both. That is a hypothesis, not a result — it has not been
measured here, and until it has, the honest statement is that the system
has a demonstrated path to calling a true health message false.

Whichever way that lands, the rule the verdict stage is missing is that
a publisher rating should not settle a claim the stance stage never
actually scored. `--gate` fails on this today, which is the intended
behaviour.

**Update — that hypothesis was tested with real NLI, and it is wrong.**
Once O3 made `--mode full` runnable, the fix this section predicted was
implemented and measured: in `classify.py`, when the entailment model had read
the passages and found none of them took a side, the publisher rating was
withheld instead of being allowed to settle the claim (`method=
"rating_withheld"`). It did not work, and it cost something.

Ten cases, `--mode full`, real NLI, one process:

| | baseline (`--no-nli`) | with the rule |
|---|---|---|
| wrong accusations | 2 | **2** |
| rumours missed | 0 | **1** (`polio_drops_en`) |

`benign_boiling_water` and `benign_vaccine_safe` both still came back `false`.
The premise was that the model would read the hot-water debunk and find it
neutral about "boiling water reduces waterborne disease". It does not — it
takes a side. The model is not abstaining on these pairs, so there is no
abstention for a rule to defer to, and the rating never had to be withheld to
produce the wrong answer. Meanwhile on `polio_drops_en`, where the fact-check
genuinely is about the claim, the model *was* neutral, the rating was withheld,
and a real rumour was downgraded to `unverified`.

So the change was reverted. Its one durable contribution is the measurement:
**the defect is not in the rating override, it is upstream in what the
entailment model does with a benign sentence that is topically adjacent to a
rumour.** Any real fix has to start there — a stronger relevance gate on which
claim a fact-check is *about*, keyed on something other than the same embedding
similarity that retrieved it — not in the verdict rules. O6 remains open and
`--gate` still fails on it, which is correct.

Recorded rather than dropped because the negative result is the useful part: it
rules out the obvious fix and says where not to look next.

**Resolved — the gate the last paragraph asked for: `backend/aboutness.py`.**
The question retrieval never answers is asked explicitly, between retrieval and
the stance model:

    claim -> retrieve -> [ is this evidence about this claim? ] -> NLI -> rating -> verdict
                               no -> neutral, rating withheld

It is asked of the evidence's **claim under review** — a fact-check's headline
claim, not its debunking prose, which is written about the rumour and shares the
rumour's vocabulary whichever claim you arrived from. And it is asked
**lexically**, which is the point: the failure is an embedding being
confidently wrong, so a second embedding of the same two sentences is the same
evidence twice. Word overlap is independent evidence — "microchip",
"coronavirus" and "lungs" are absent from the benign sentence, and no amount of
topical closeness puts them there.

Two directions are measured over stemmed content words: how much of the
*claim's* content the reviewed claim accounts for, and how much of the
*reviewed claim's* content the claim accounts for. Either one clearing 0.5
means they are the same claim told at different lengths, which is what a
paraphrase looks like from each side. Only when **neither** accounts for half
of the other is the evidence gated — one topic, two different assertions inside
it, which is the O6 shape exactly.

Measured, `--mode full`, real NLI, one process, the three benign cases plus five
rumours:

| | before | after |
|---|---|---|
| wrong accusations | **3** | **0** |
| rumours missed | 0 | **0** |
| verdict accuracy | 62.5% | **100%** |
| retrieval stray rate | 100% (3/3) | 100% (3/3) |

Three, not the two this section was written about: `benign_upi_safe` was also
coming back `false`. The stray rate is deliberately unchanged — retrieval is
*right* to return a topically adjacent fact-check, and fixing that number was
never the goal.

Over all 41 cases against all 32 seed entries, the gate says `not_about` for
every one of the 288 comparable claim/entry pairs with no right answer, and
`not_about` for none of the 24 pairs that have one.

**What it refuses to do is the load-bearing part.** The gate has three answers
and only one of them acts. Across languages a lexical comparison is
meaningless — true Hinglish matches score 0.21-0.37 against English
fact-checks, benign English mismatches score 0.25-0.35, so no threshold
separates them — and transliterating one side to meet the other measures the
lexicon's coverage as much as the claim's content (it mis-gated four of eight
Hinglish pairs when tried). Both cases return `unclear`, which leaves stage 4
behaving exactly as it did before. So the cost of a wrong answer here is one
rumour reported `unverified` instead of `false` — never a true statement called
false, which was the failure being fixed.

The gate is applied in two places and to rated evidence only, because a rating
is what can settle a claim with nothing having scored it:

  * `stance/classify.py` — the item is neutral with `method="not_about"`, its
    rating is withheld, and it reaches neither the embedder nor the entailment
    model, so the models are never asked a question whose answer would be
    discarded;
  * `evidence/normalize.py` — `decisive` is not set, which is what keeps
    `verdict.py`'s first rule (a decisive fact-check wins outright) from firing.
    Belt and braces on purpose: the benign claims score 0.33-0.44 on the
    one-directional `claim_overlap` that guards `decisive`, and 0.44 against a
    0.45 floor is not a safety margin.

A gated fact-check is still reported, and reported *without* its rating: "Alt
News has checked a similar-sounding claim (...), which is not this one."
Repeating "rated it false" beside a claim the rating was never applied to would
be the same accusation in prose.

**Known gap, recorded rather than papered over.** A Hindi or Hinglish claim
retrieved against an English fact-check is `unclear`, so O6 is unguarded in
those languages — the gate cannot see a mismatch it cannot read. Devanagari
against Devanagari is guarded (the seed corpus stores `claim_hi`; the candidate
it builds carries only the English `claim`, so putting the Hindi phrasing into
the candidate's text would extend the gate to Hindi claims and is the obvious
next step). Nothing here needs the gate to be clever in Hindi to be correct in
English.

---

**O7 — Every retrieval miss is a stage 2 miss.**
The harness answered a question the seed-index benchmark could not.
Retrieval finds 19 of the 24 expected entries (79.2%), and **all five
misses are cases where stage 2 extracted no claim at all** — `claims=0`,
so retrieval was never asked. On every claim that reached it, retrieval
hit its target.

Per language (retrieval mode, heuristic backend):

| | retrieval | check-worthy f1 |
|---|---|---|
| Hinglish | **100%** (15/15) | 78.3% |
| English | 81.8% | 88.2% |
| Devanagari Hindi | **40%** (2/5) | 50.0% |

So after O1, romanised Hinglish is now the *best* served language and
Devanagari Hindi is the worst — the opposite of where this started, and
not because retrieval got worse at Hindi (the seed-index benchmark still
shows 4/4 there) but because the heuristic claim extractor drops
Devanagari sentences before retrieval sees them. The next piece of
Hindi work belongs in stage 2, not stage 3b.

Measured with `CLAIM_BACKEND=heuristic`. The shipping default is `auto`,
which prefers the transformer backend; that configuration OOMs on this
machine alongside the retrieval embedder when this was measured (O3, now
resolved — but not re-run), so whether the transformer extractor handles
Devanagari better is still **unmeasured**.

**O3 — Resolved: the full pipeline now runs with every model loaded.**
The original finding stands as the symptom: the machine has 8 GB with ~400 MB
free, and running all models in one process failed with "the paging file is too
small" (OS error 1455). The pipeline handled it as designed — logged the
failure, returned `unverified`, no crash — but `--mode full` could not be
measured, so the harness ran behind `--no-nli` and its verdict numbers were a
floor.

The cause was not the number of models. It was how many forward passes were
*live at once*. Loading was already serialised (`collector.warm_up()` runs on
the calling thread before either pool starts), but inference was not: stage 3b
searches claims in a `ThreadPoolExecutor` and each claim fans out to a second
pool over the retrievers, so `seed_index.search` — and the sentence-transformer
forward pass inside it — could be executing on up to 8 worker threads at once,
each with its own activation buffers, each multiplied again by torch's intra-op
pool, while mDeBERTa's weights sat resident for stage 4. Peak RSS scaled with
the pool width instead of with the batch. Nothing leaked; the peak was simply
reached, which is why a single case could survive and a run of several could
not.

The fix is `backend/torch_runtime.py`: one process-wide reentrant lock and
`torch.inference_mode()`, entered by every forward pass in the codebase
(`rank.embed`, `nli.score_pairs`), plus `eval()` applied once at load. Concurrent
model execution becomes sequential model execution; the IO-bound part of
retrieval — the HTTP the pools exist for — stays parallel. Stage 4 still borrows
the claim stage's mDeBERTa through its existing `lru_cache`, so there is still
exactly one copy of those weights. **No stage's output changes**: serialising a
forward pass and dropping autograd gives identical logits.

Measured on this machine, real NLI on (`nli: true` in every record), one
process:

| run | cases | result |
|---|---|---|
| `--mode full --case upi_cashback_en` | 1 | exit 0, verdict `false` |
| `--mode full` × 5 (upi_cashback_en, upi_pin_refund_en, free_laptop_en, note_gps_en, hot_water_en) | 5 | exit 0, 354.9s, **no `stage_errors` on any case** |

First case 113s (model load), the rest 43–53s each — the flat per-case time
after the first is what says the fix holds rather than merely deferring the
peak.

**What this does not claim.** The full 41-case benchmark has *not* been re-run
with NLI on; the numbers elsewhere in this file that were measured under
`--no-nli` are still `--no-nli` numbers and are still a floor. In particular
**O6 was not exercised by this smoke set**: `hot_water_en` is in the five above
and comes back `false`, but that case's own label expects `false`, so it does
not touch the defect — the true health statements O6 is about were not in this
set. They were measured separately afterwards, under `--mode full` with real
NLI, first to confirm the defect (three wrong accusations) and then to confirm
the gate that fixed it; see O6. `--no-nli` also stays, because a deployment without the entailment
checkpoint is a real configuration, not just a workaround for this laptop.

**O4 — One test is sensitive to memory pressure.**
`test_enrich_adds_full_text_to_the_top_results` failed once, during a run
that overlapped a live model process, and has not reproduced since. The
cause is trafilatura failing to extract under memory pressure, which the
fetcher then correctly treats as "no article". Worth knowing before chasing
it as a logic bug.

**O5 — Pre-existing gaps left alone.**
`backend/analyzers/video.py` uses `tempfile.mktemp` and cleans up nothing;
`models.py` loads BLIP through the `image-text-to-text` task when that
checkpoint normally wants `image-to-text`; `venv/` is broken. All predate this work and sit in stage
1, which the brief said not to modify.

**Update — the BLIP half was a live defect, and is fixed.** The note above had
the diagnosis backwards. `image-to-text` is gone in transformers 5 (the installed
version is 5.15): `image-text-to-text` is the only captioning task left, so the
task name was already the right one. What was wrong was the call. That pipeline
is built for prompted vision-language models and raises on an image alone —

    >>> models.blip()(image)
    ValueError: You must provide text for this pipeline

— which means `describe()` raised for *every* image, and the image tests never
caught it because they supply descriptions directly. `describe()` now passes
`text=""`, the empty prompt BLIP takes for an unconditional caption. Measured
against the real checkpoint: a plain image now returns a caption ("a green
screen with a white background") where it previously raised. The empty prompt is
also why `generated_text` needs no stripping — with a non-empty prompt the
pipeline prefixes the caption with it.
