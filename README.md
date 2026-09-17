# Misinfo Checker

A misinformation detection service for forwarded text, links, images and videos —
the kind of content that circulates on WhatsApp/Telegram in India (fake UPI cashback
offers, Hinglish rumours, recycled disaster footage, health cures).

The system is a staged pipeline. Each stage produces a typed object that the next
stage consumes, so stages can be built, tested and swapped independently.

```
 input ──► [1] Ingest ──► MediaPacket ──► [2] Claim Extraction ──► ClaimSet
                                                                       │
                                                  to_graph_seed()      ▼
                                         [3A] Evidence Graph ◄── nodes / edges
                                                  │
                                    Evidence      │
                       [3B] Retrieval ──────────► │
                                                  ▼
                                          [4] Verdict ──► API / bot reply

 [1] done   [2] done (regex + transformer backends)   [3A] done (data layer)
 [3B] retrieval not started   [4] not started
```

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the gap analysis.

---

## Stage 1 — Ingest → `MediaPacket`

`backend/analyzers/packet.py` normalises every input type into one dict:

```python
{
  "input_type": "text" | "link" | "image" | "video",
  "text":        "...",        # raw text / article body / caption + transcript
  "source_date": "2024-06-08" | None,
  "images": [                  # one per image, or one per video keyframe
    {
      "path": "...", "ocr_text": "...", "description": "...",
      "embedding": [768 floats], "exif_date": ..., "ai_generated_score": ...,
      "frame_time": 12.5 | None
    }
  ]
}
```

| Builder | What it does |
|---|---|
| `from_text(text)` | passes text through |
| `from_link(url)` | trafilatura body + htmldate publish date |
| `from_image(path, caption)` | EasyOCR, BLIP caption, DINOv2 embedding, EXIF date, optional AI-image detector |
| `from_video(path, caption)` | ffmpeg → Whisper transcript (merged into `text`), keyframes every 5 s → `from_image` on each |
| `from_video_url(url)` | yt-dlp download → `from_video` |

Model loading lives in `backend/analyzers/models.py` (all lazy, `lru_cache`d).

## Stage 2 — Claim Extraction → `ClaimSet`

`backend/claims/` turns a packet into separate, structured, check-worthy claims.

```python
from backend.claims import extract_claims

claims = extract_claims(packet)                       # backend="auto"
claims = extract_claims(packet, backend="heuristic")  # regex only, no models
claims = extract_claims(packet, backend="transformer")# require the HF models

claims.check_worthy()                    # only the ones worth fact-checking
claims.to_graph_seed()                   # {"nodes": [...], "edges": [...]}
claims.model_dump(mode="json")           # serialisable
claims.backend                           # which backend actually ran
```

Two backends share the same schema and pipeline; `auto` (the default) picks
`transformer` when `transformers` + `torch` import, else `heuristic`.

| Step | `heuristic` | `transformer` |
|---|---|---|
| Sentence split | regex | regex |
| Patterned entities (money, %, dates, URL, phone, UPI) | regex | regex |
| Named entities (PERSON / ORG / GPE) | capitalisation + keyword hints | **HF token-classification NER**, merged with regex; regex guesses kept only where the model found nothing |
| Check-worthy score | feature→weight table | **0.5 × heuristic + 0.5 × zero-shot NLI** P("a factual claim that can be verified") |
| Claim type | rule order | **zero-shot NLI over the 10 types**, used when top-2 margin ≥ 0.10; `chain_offer` and `health` from the rules are sticky (their regex cues are more precise) |
| Hard rejects (greeting, < 4 tokens) | yes | yes — the model is never called for these |

Models (lazy-loaded, `lru_cache`d, CPU is fine — ~1.2 GB download on first use):

| Env var | Default | Notes |
|---|---|---|
| `CLAIM_NER_MODEL` | `dslim/bert-base-NER` | English. Try `ai4bharat/IndicNER` for Hindi / Hinglish names |
| `CLAIM_ZSC_MODEL` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Multilingual NLI, covers Hindi |

Each claim's `signals` records both sides (`heuristic_confidence`, `model_worthy`,
`heuristic_type`, `model_type`, `model_type_margin`) so the two backends can be
compared on the same input, and `dropped` reasons include `model_not_claim` when the
model demoted a sentence the rules had accepted.

### Pipeline

1. **Segment** every text-bearing field — `text`, each image's `ocr_text`, each image's
   `description` — into sentences, keeping char offsets, image index and frame time
   (`segment.py`).
2. **Extract entities and time references** with regex: money (with lakh/crore
   normalisation), percentages, numbers, dates (absolute → ISO, relative → resolved
   against `today`, recurring), URLs, phone numbers, UPI IDs, hashtags, and
   capitalised proper-noun spans typed as ORG / GPE / PERSON / PROPER (`entities.py`).
3. **Score check-worthiness** from a transparent feature → weight table: named entities,
   quantities, time refs, attribution verbs, policy/health/event vocabulary and
   chain-offer vocabulary push a sentence up; questions, opinion markers, personal chat,
   greetings and bare imperatives push it down. Threshold 0.45 (`checkworthy.py`).
4. **Classify** each claim into one type: `chain_offer`, `health`, `policy`,
   `attribution`, `money`, `statistic`, `event`, `causal`, `prediction`, `generic`.
   Attribution claims also get `attributed_to` (the speaker).
5. **Dedupe** near-identical sentences across fields (OCR text frequently repeats the
   caption) with token Jaccard ≥ 0.8, keeping the `text` copy first.
6. Rejected sentences are kept in `ClaimSet.dropped` with a reason
   (`question`, `opinion`, `personal`, `greeting`, `too_short`, `low_signal`, …)
   so the evaluation harness can tune the weights later.

### Claim shape

```python
Claim(
  id="clm_…",                       # stable sha1 of packet + normalised text
  text="You have won Rs 5,000 cashback from SBI.",
  normalized_text="you have won rs 5 000 cashback from sbi",
  claim_type="chain_offer",
  confidence=0.9,
  check_worthy=True,
  entities=[Entity(label="MONEY", text="Rs 5,000", normalized="INR 5000", …),
            Entity(label="ORG",   text="SBI",      normalized="sbi", …)],
  time_refs=[TimeRef(kind="relative", text="today", normalized="2026-09-17", …)],
  numbers=["INR 5000"],
  attributed_to=None,
  source=SourceSpan(field="text" | "ocr" | "caption", image_index=…, frame_time=…, start=…, end=…),
  language="en" | "hi" | "hinglish",
  signals={…},                      # the feature flags that produced the score
  evidence_ids=[], verdict=None,    # reserved for stages 3–4
)
```

### Evidence-graph seed

`ClaimSet.to_graph_seed()` emits the skeleton the evidence graph will grow from:

- nodes: one `packet`, one per `claim`, one per unique `entity`
- edges: `packet -HAS_CLAIM-> claim`, `claim -MENTIONS-> entity`,
  `claim -SHARES_ENTITY-> claim` (via a common entity id)

Stage 3A grows `evidence` nodes and stance edges on top of this seed without touching
this module; ids are deterministic so the same input always yields the same graph.

### Why keep the heuristic layer under the model

The regex layer is not just a fallback: models are unreliable on UPI ids, `Rs 2 lakh
crore`, `15/08/2023`, phone numbers and forward-this-message patterns, so those come
from patterns in both backends. The model adds what regex cannot do — recognising
names without capitalisation cues, judging whether a sentence is a verifiable claim
rather than chit-chat, and typing claims in Hindi/Hinglish.

Known limits: the default NER model is English-only (swap `CLAIM_NER_MODEL` for Indic
text); zero-shot typing is ~0.3 s per sentence on CPU, so long articles are slow —
batching is the obvious next optimisation.

## Stage 3A — Evidence Graph

`backend/evidence/` takes the seed above and grows it into the structure stage 4 will
read a verdict off. It is the **data layer only** — nothing here touches the network.

```python
from backend.claims import extract_claims
from backend.evidence import Evidence, EvidenceGraph, EvidenceSource

claims = extract_claims(packet)
graph  = EvidenceGraph.from_claimset(claims)      # packet/claim/entity skeleton

graph.add_evidence(claim_id, Evidence(
    evidence_type="fact_check",
    text="No such SBI cashback scheme exists.",
    relation="refutes",                            # supports | refutes | uncertain
    relevance=0.9, credibility=0.9,
    source=EvidenceSource(url="https://www.altnews.in/…", published="2026-09-10"),
))

graph.get_claim_evidence(claim_id)                 # strongest first
graph.get_claim_evidence(claim_id, relation="refutes")
graph.to_dict()                                    # JSON-serialisable
```

The skeleton is consumed, never re-derived: `from_claimset` reads
`ClaimSet.to_graph_seed()` as-is, so stage 2 stays the single owner of claim and
entity ids and can be edited without this module noticing.

### Nodes and edges

| Node kind | Comes from |
|---|---|
| `packet` | stage 1 |
| `claim` | stage 2 |
| `entity` | stage 2 |
| `evidence` | stage 3 |

```
packet   -HAS_CLAIM->      claim
claim    -MENTIONS->       entity
claim    -SHARES_ENTITY->  claim      (via a common entity)
claim    -HAS_EVIDENCE->   evidence   (retrieved for this claim)
evidence -SUPPORTS->       claim
evidence -REFUTES->        claim
evidence -UNCERTAIN_FOR->  claim      (relevant, takes no side)
```

`HAS_EVIDENCE` records *that* something was retrieved; the stance edge records *how it
bears on the claim*, and re-stating a relation replaces the old edge rather than
leaving a stale `SUPPORTS` behind.

### Models (`backend/evidence/schema.py`)

- **`EvidenceSource`** — `url`, `title`, `publisher`, `domain` (derived from the URL),
  `published` (what the source claims), `retrieved` (when we fetched it), `rating`
  (the publisher's own verdict text). `published` and `retrieved` stay apart because
  only `published` matters for recycled-media checks.
- **`Evidence`** — `id`, `evidence_type` (`fact_check`, `news`, `rumor_index`,
  `reverse_image`, `date_check`, `official`, `other`), `text`, `source`, `relation`,
  `relevance` (does it match the claim?), `credibility` (is the source worth
  trusting?), `match` (why the retriever returned it), `meta`. `weight` is
  `relevance × credibility`, which is what `get_claim_evidence` ranks by. Relevance
  and credibility stay separate so stage 4 can tell "a spot-on match from a random
  blog" from "a loose match from PIB".
- **`EvidenceRelation`** — the `claim ↔ evidence` stance record kept alongside the
  edge it produced, with the scores at the time and an optional `note`.
- **`Node` / `Edge`** — typed access to `id` / `kind` / `type`, with everything else in
  `attrs`, flattened to the seed's plain-dict shape on the wire so a graph round-trips
  through JSON with no migration (`Edge.src`/`dst` serialise as `from`/`to`).

### Deterministic ids

`Evidence.id` is `sha1(evidence_type | url | text)`, or
`sha1(evidence_type | publisher|title | text)` when there is no URL — the **retrieval
date is deliberately excluded**, so re-running retrieval reuses the node instead of
forking the graph. The same fact-check attached to two claims is one node with two
`HAS_EVIDENCE` edges, and `graph.to_dict()` is byte-identical across runs for the same
input.

### API

| Method | What it does |
|---|---|
| `EvidenceGraph.from_claimset(claimset)` | skeleton from a `ClaimSet` (`from_claim_set` is an alias) |
| `.from_seed(seed)` / `.from_dict(data)` / `.to_dict()` | serialisation |
| `.add_evidence(claim_id, evidence, relation=None)` | evidence node + `HAS_EVIDENCE` + stance edge |
| `.add_relation(claim_id, evidence_id, relation, …)` | set/replace a stance |
| `.get_claim_evidence(claim_id, relation=None)` | `[Evidence]`, strongest first |
| `.get_claim_relations(claim_id)` / `.get_entities(claim_id)` / `.related_claims(claim_id)` | read-back helpers |
| `.claims` / `.entities` / `.evidence_nodes` | filtered views of `nodes` |
| `.neighbors(node_id, edge_type=None)` / `.summary()` / `.merge(other)` | traversal, counts, union |
| `.apply_to_claimset(claimset)` | writes `evidence_ids` back onto the claims |

Unknown claim ids raise `KeyError`; attaching evidence to a non-claim node raises
`ValueError`. `nodes` is the single source of truth — `claims` / `entities` /
`evidence_nodes` are filtered views, so they cannot drift.

## Stages 3B–4 — not built yet

Evidence *retrieval* (fact-check APIs, local rumour index, reverse-image via DINOv2
embeddings, date comparison for recycled media) that produces `Evidence` objects, then
scoring, verdict, and the `guardian_bot` → API wiring. The API still returns
`{"packet": …, "results": []}`.

---

## Layout

```
backend/
  analyzers/     Stage 1: packet.py, image.py, video.py, link.py, models.py
  claims/        Stage 2: schema.py, segment.py, entities.py, checkworthy.py,
                          transformer.py, ollama.py, extractor.py
  evidence/      Stage 3A: schema.py (Evidence, EvidenceSource, EvidenceRelation,
                          Node, Edge), graph.py (EvidenceGraph)
  api/main.py    FastAPI: /health, /check/text, /check/link
  bot/           guardian_bot.py, filters.py (message classification only)
  tests/         pytest suite
  samples/       (empty placeholders for the five eval cases)
  data/  evaluation/  frontend/   (empty)
```

## Running

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt          # API, link ingest, claim extraction (both backends), tests

pytest                                   # 54 tests, < 2 s (model calls are stubbed)
$env:CLAIM_TESTS_WITH_MODELS = "1"; pytest   # + 1 end-to-end test with the real models

uvicorn backend.api.main:app --reload    # POST /check/text {"text": "..."}
```

Image/video ingest additionally needs `Pillow easyocr openai-whisper transformers torch yt-dlp`
and `ffmpeg` on PATH. Set `AI_DETECTOR_MODEL=<hf model id>` in `.env` to enable
AI-generated-image scoring.

Try the extractor directly:

```powershell
python -c "from backend.claims import extract_claims; import json; print(json.dumps(extract_claims({'input_type':'text','text':'You have won Rs 5,000 cashback from SBI. Forward this to 10 people.','images':[]}).model_dump(mode='json'), indent=1))"
```
