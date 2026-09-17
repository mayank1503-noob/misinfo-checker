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

## Open issues

**O1 — Romanised Hinglish does not retrieve.**
The multilingual embedder handles Devanagari well — "गर्म पानी पीने से कोरोना
ठीक होता है" matches the English hot-water debunk at 0.48 — but Latin-script
Hindi does not. "garam paani peene se corona theek ho jata hai" scores below
the 0.45 floor against the same entry, while a Hinglish sentence carrying
English proper nouns ("UNESCO ne Jana Gana Mana ko best anthem declare kiya")
matches at 0.85. Since romanised Hindi is most of what circulates on
WhatsApp, this is the biggest real gap. Fix: transliterate to Devanagari
(`indic-transliteration`) before embedding, and index a romanised form of
each seed entry. Not attempted here — it needs its own evaluation to avoid
lowering the floor and inviting false matches.

**O2 — Stage 2 scores personal chat as check-worthy.**
"Kal shaam ko ghar aa raha hoon" (I'm coming home tomorrow evening) passes
the check-worthiness threshold: it has a resolved time reference and a
subject. The verdict stage abstains rather than accusing, so nothing bad
reaches the user, but it costs a retrieval round per chat message. The fix
belongs in `backend/claims/checkworthy.py`, which the brief said not to
modify.

**O3 — The full pipeline with every model loaded needs ~2 GB free.**
Each stage was verified live and separately on this machine — seed-index
retrieval (0.82 on the SBI scam, 0.48 cross-lingual), the image stage with
real DINOv2 and CLIP (recycled photo detected, verdict `misleading`), and the
offline pipeline across all 246 tests. Running *all* models in one process
was not achievable here: the machine has 8 GB with ~400 MB free, and model
loads fail with "the paging file is too small" (OS error 1455). The pipeline
handled that exactly as designed — logged the failure, returned `unverified`
with a coherent summary, no crash — but it does mean the full-model path has
not been observed end to end in one process.

**O4 — One test is sensitive to memory pressure.**
`test_enrich_adds_full_text_to_the_top_results` failed once, during a run
that overlapped a live model process, and has not reproduced since. The
cause is trafilatura failing to extract under memory pressure, which the
fetcher then correctly treats as "no article". Worth knowing before chasing
it as a logic bug.

**O5 — Pre-existing gaps left alone.**
`backend/analyzers/video.py` uses `tempfile.mktemp` and cleans up nothing;
`models.py` loads BLIP through the `image-text-to-text` task when that
checkpoint normally wants `image-to-text`; `venv/` is broken;
`backend/evaluation/` is still empty. All predate this work and sit in stage
1, which the brief said not to modify.
