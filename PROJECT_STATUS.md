# Misinfo Checker — Current Status

_Last updated: 2026-09-17_

## What this project is

A misinformation detection service for text, links, images, and videos (with a
WhatsApp/Telegram-style "guardian bot" front door). The backend is a FastAPI app;
the analysis pipeline lives in `backend/analyzers/`.

## What is happening right now

**Every request to the API returns the raw "media packet" instead of a verdict.**

Example — `POST /check/text` with `{"text": "Free UPI cashback, forward to 10 people"}`:

```json
{
  "packet": {
    "input_type": "text",
    "text": "Free UPI cashback, forward to 10 people",
    "source_date": null,
    "images": []
  },
  "results": []
}
```

`results` is always an empty list. No claim, no fact-check, no score, no verdict.

## Why: the pipeline stops after Stage 1

The system was designed as a multi-stage pipeline, but only the first stage
(building a normalized "packet" from whatever the user sent) is implemented.

```
User input (text / link / image / video)
        │
        ▼
┌────────────────────────────────────────────┐
│ Stage 1 — Ingest → packet   (IMPLEMENTED)  │
│   backend/analyzers/packet.py              │
│   - from_text   → text as-is               │
│   - from_link   → trafilatura + htmldate   │
│   - from_image  → OCR, BLIP caption,       │
│                   DINOv2 embedding, EXIF,  │
│                   AI-generated score       │
│   - from_video  → ffmpeg audio → Whisper,  │
│                   keyframes → image.analyze│
└────────────────────────────────────────────┘
        │  packet = {input_type, text, source_date, images}
        ▼
┌────────────────────────────────────────────┐
│ Stage 2 — Claim extraction  (IMPLEMENTED)  │
│   backend/claims/  -> ClaimSet             │
│   sentences -> entities/time -> score ->   │
│   type -> dedupe -> to_graph_seed()        │
│   NOT yet called from main.py              │
└────────────────────────────────────────────┘
        ▼
┌────────────────────────────────────────────┐
│ Stage 3A — Evidence graph   (IMPLEMENTED)  │
│   backend/evidence/  -> EvidenceGraph      │
│   to_graph_seed() -> packet/claim/entity   │
│   nodes + evidence nodes, HAS_EVIDENCE and │
│   SUPPORTS / REFUTES / UNCERTAIN_FOR edges │
│   data layer only; NOT called from main.py │
├────────────────────────────────────────────┤
│ Stage 3B — Evidence retrieval /            │
│            fact-check lookup   (MISSING)   │
│   nothing produces Evidence objects yet    │
└────────────────────────────────────────────┘
        ▼
┌────────────────────────────────────────────┐
│ Stage 4 — Scoring + verdict  (MISSING)     │
└────────────────────────────────────────────┘
        ▼
   API response  →  currently just {packet, results: []}
```

In `backend/api/main.py`, both endpoints literally do:

```python
packet = from_text(text)      # or from_link(url)
return {"packet": packet, "results": []}
```

`results` is hard-coded to `[]`. The claim extractor (`backend.claims.extract_claims`)
exists and is tested, but nothing in the API or bot calls it yet.

## What the packet contains (per input type)

| Field         | text  | link                          | image                   | video                                  |
|---------------|-------|-------------------------------|-------------------------|----------------------------------------|
| `input_type`  | text  | link                          | image                   | video                                  |
| `text`        | input | article body (trafilatura)    | caption                 | caption + Whisper transcript           |
| `source_date` | null  | published date (htmldate)     | null                    | null                                   |
| `images`      | []    | []                            | 1 analyzed image        | up to 6 keyframes, each analyzed       |

Each analyzed image (`image.analyze`) is:
`{path, ocr_text, description, embedding (768-d DINOv2), exif_date, ai_generated_score, frame_time}`.

## Other gaps found while reviewing

| Area | Issue |
|------|-------|
| `backend/api/main.py` | Only `/check/text` and `/check/link` exist. No `/check/image` or `/check/video` endpoints, even though `from_image`, `from_video`, `from_video_url` are implemented. |
| `backend/bot/guardian_bot.py` | `process_message` only classifies the message (`text` / `link` / `empty`) and echoes it back. It never calls the analyzers or the API. |
| `backend/bot/filters.py` | `contains_url` matches `"www."` but `/check/link` rejects anything not starting with `http(s)://`. |
| `backend/samples/*.txt` | All five sample files are **empty** (0 bytes). |
| `backend/data/`, `backend/evaluation/`, `frontend/` | Empty directories. |
| `.env` | Empty. `AI_DETECTOR_MODEL` is read from env but never set, so `ai_generated_score` is always `null`. |
| `requirements.txt` | Now includes `trafilatura`, `htmldate`, `pytest`. Still missing the image/video stack: `Pillow`, `easyocr`, `openai-whisper`, `transformers`, `torch`, `yt-dlp`. ffmpeg must also be on PATH. |
| `venv/` | Broken: `pyvenv.cfg` points at a Python install under another user's home directory. Recreate with `python -m venv venv`. |
| `backend/analyzers/video.py` | Uses `tempfile.mktemp` (deprecated/racy) and never cleans up keyframe dirs or downloaded videos. |
| `backend/analyzers/models.py` | `blip()` uses the `image-text-to-text` pipeline task with `blip-image-captioning-base`; that model is normally loaded with `image-to-text`. Worth verifying it loads. |

## What needs to happen to get a real output

1. ~~**Claim extraction**~~ — done: `backend/claims/` (see README). Heuristic baseline;
   a model-backed extractor can be swapped in behind the same `ClaimSet` schema.
2. ~~**Evidence graph**~~ — done: `backend/evidence/` (Stage 3A, see README).
   `EvidenceGraph.from_claimset()` + `add_evidence` / `add_relation` /
   `get_claim_evidence` / `to_dict`. No retrieval yet.
3. **Evidence / fact-check retrieval (3B)** — e.g. Google Fact Check Tools API, a local
   index of known scams/rumors (`backend/data/`), reverse-image lookup using the
   DINOv2 embeddings, date comparison (`source_date` / `exif_date`) for recycled content.
4. **Scoring + verdict** — combine signals (fact-check matches, AI-generated score,
   recycled-media flag, scam patterns like fake UPI) into a label
   (`true` / `false` / `misleading` / `unverified`) with a confidence and explanation.
5. **Wire it up** — replace `"results": []` in `main.py` with the verdict, add
   `/check/image` and `/check/video`, and have `guardian_bot.process_message`
   call the pipeline.
6. **Fill in samples** and build `backend/evaluation/` so the five sample cases
   (`fake_upi`, `hinglish_rumor`, `personal_chat`, `recycled_rumor`, `true_claim`)
   can be run as a regression check.

## How to run what exists

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pytest                                   # 54 tests: claim extraction + evidence graph
uvicorn backend.api.main:app --reload
# then POST to http://127.0.0.1:8000/check/text or /check/link
```

Image/video paths additionally need `Pillow easyocr openai-whisper transformers torch yt-dlp`
and ffmpeg on PATH.
