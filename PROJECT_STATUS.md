# Misinfo Checker — Current Status

_Last updated: 2026-09-17_

## What this project is

A misinformation detection service for forwarded text, links, images and videos, with a
WhatsApp/Telegram-style "guardian bot" front door. FastAPI backend; the analysis runs as
a staged pipeline where each stage produces a typed object the next one consumes.

## Where it stands

**Every stage is built, tested and wired together.** `POST /check/text` returns a
verdict with the evidence behind it, not the raw packet.

```
User input (text / link / image / video)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1 — Ingest → MediaPacket            (IMPLEMENTED)      │
│   backend/analyzers/   text, link, image, video keyframes    │
└──────────────────────────────────────────────────────────────┘
        ▼  {input_type, text, source_date, images[]}
┌──────────────────────────────────────────────────────────────┐
│ Stage 2 — Claim extraction → ClaimSet     (IMPLEMENTED)      │
│   backend/claims/      regex / transformer / ollama backends │
└──────────────────────────────────────────────────────────────┘
        ▼  to_graph_seed()
┌──────────────────────────────────────────────────────────────┐
│ Stage 3a — Evidence graph                 (IMPLEMENTED)      │
│   backend/graph/       NetworkX MultiDiGraph: packet, claim, │
│   entity, image, evidence, source, date, verdict nodes;      │
│   15 edge types; deterministic ids; idempotent adds;         │
│   timeline, stance totals, decisive hits, pyvis export       │
└──────────────────────────────────────────────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3b — Evidence retrieval             (IMPLEMENTED)      │
│   backend/evidence/    Google Fact Check + Tavily + a local  │
│   FAISS seed index; claim-type routed queries, Hindi too;    │
│   rating normalisation, domain credibility, dedupe, cap      │
│   Retrieval only — every candidate leaves with stance=None   │
└──────────────────────────────────────────────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 5 — Image evidence                  (IMPLEMENTED)      │
│   backend/images/      keyframe dedupe, local DINOv2 index,  │
│   SerpAPI reverse search, CLIP caption consistency           │
│   Reuses stage 1's embeddings — nothing is recomputed        │
└──────────────────────────────────────────────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4 — Stance                          (IMPLEMENTED)      │
│   backend/stance/      passages → rank → NLI; a publisher's  │
│   rating overrides the model; writes SUPPORTS / REFUTES /    │
│   NEUTRAL edges back into the graph                          │
└──────────────────────────────────────────────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Verdict                                   (IMPLEMENTED)      │
│   backend/verdict.py   decisive fact-check > recycled image  │
│   > weighted stance > unverified. Rules, not a model.        │
└──────────────────────────────────────────────────────────────┘
        ▼
   API response / bot reply, with reasons and sources
```

Example — `POST /check/text` with a UPI cashback forward, no API keys set:

```json
{
  "verdict": {
    "label": "false",
    "confidence": 0.72,
    "summary": "This message contains a false claim. A fact-checker has already
                checked this claim. Demo Fact Check Archive rated it false on
                2024-03-11  [demo data]"
  },
  "results": [
    {
      "claim": "SBI is giving Rs 5,000 cashback to every customer today.",
      "label": "false",
      "confidence": 0.72,
      "reasons": ["Demo Fact Check Archive rated it false on 2024-03-11 ..."],
      "evidence_ids": ["ev_..."],
      "demo_only": true
    }
  ],
  "timeline": [...],
  "stages": {"claims": {...}, "evidence": {...}, "stance": {...}, "verdict": {...}}
}
```

## What runs with nothing installed and no keys

| | Without keys / models | With them |
|---|---|---|
| Claim extraction | regex backend | + transformer NER and zero-shot typing |
| Retrieval | local seed index (32 demo fact-checks) | + Google Fact Check, Tavily |
| Stance | everything neutral, `method="unavailable"` | passage ranking + NLI |
| Images | EXIF dates only | + DINOv2 index, CLIP captions, reverse search |
| Verdict | mostly `unverified` | the full rule set |

Nothing raises in either column. Every external call is cached under `cache/` and fails
soft.

## Tests

246 tests, about 5 seconds, entirely offline — `backend/tests/conftest.py` blocks the
HTTP helpers for every test and the models are stubbed.

| File | Covers |
|---|---|
| `test_claims.py`, `test_claims_transformer.py` | stages 1–2 (pre-existing) |
| `test_graph_store.py` | stage 3a: construction, images, evidence, stance, duplicates, dates, serialisation, pyvis |
| `test_evidence_retrieval.py` | stage 3b: queries, ratings, credibility, dedupe, all three retrievers, fetching, the collector, the cache |
| `test_stance.py`, `test_stance_graph_adapter.py` | stage 4: chunking, ranking, NLI, rating override, the graph adapter |
| `test_images.py` | stage 5: keyframes, local index, CLIP, reverse search, the collector |
| `test_pipeline.py` | the verdict rules, the pipeline, the API, the bot |
| `test_evidence_graph.py` | the compatibility shim for the moved module |

Two integration tests run against the real models behind `CLAIM_TESTS_WITH_MODELS=1`
and `STANCE_TESTS_WITH_MODELS=1`.

## Gaps that were closed

| Was | Now |
|---|---|
| `results` hard-coded to `[]` | a verdict per claim, with reasons and sources |
| No `/check/image` or `/check/video` | both, plus `/check/packet` and a richer `/health` |
| `guardian_bot` echoed the message | runs the pipeline and replies with a verdict |
| `contains_url` accepted bare `www.` that `/check/link` rejected | the bot normalises it to `https://` before fetching |
| The five `backend/samples/*.txt` were empty | filled with a scam, a Hinglish rumour, personal chat, a recycled-image claim and a true claim |
| `README.md` was empty | documents every stage |
| `requirements.txt` missing the image/video stack | networkx, pyvis, PyYAML, faiss-cpu, sentence-transformers, numpy, Pillow, torchvision added |

## Still open

- `venv/` is still broken: `pyvenv.cfg` points at a Python install under another user's
  home directory. Recreate it with `python -m venv venv`. Everything here was run on the
  system interpreter.
- `backend/analyzers/video.py` still uses `tempfile.mktemp` and never cleans up keyframe
  directories or downloaded videos.
- `backend/analyzers/models.py` loads BLIP through the `image-text-to-text` pipeline
  task; that checkpoint is normally loaded with `image-to-text`. Untested here — the
  image tests supply descriptions directly.
- `backend/evaluation/` is still empty. The samples now have content, so a harness that
  asserts expected labels per sample is the obvious next piece.
- Romanised Hinglish retrieval, and stage 2 scoring personal chat as check-worthy. See
  DECISIONS.md, open issues.
