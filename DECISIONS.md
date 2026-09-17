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

---

## Open issues

_None recorded yet._
