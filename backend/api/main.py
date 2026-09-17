"""
The HTTP front door.

Every endpoint does the same thing: build a packet for whatever was sent,
run `backend.pipeline.analyze` over it, and return the verdict with the
evidence behind it. `results` used to be hard-coded to `[]`; it now
carries one entry per check-worthy claim, each with its label, its
confidence and the sources that produced it.

The response is deliberately layered, because two very different callers
use it: a bot wants `verdict.summary` and nothing else, while a reviewer
wants the claims, the evidence and the graph. Nobody is forced to parse
what they do not need, and `?graph=true` is opt-in because a serialised
graph is large.

`?agentic=true` runs the same stages through `backend.agents` instead of
the fixed pipeline order, and adds an `agents` block (the plan, each
agent's structured result, the tool trace) plus `explanation`. Everything
else in the response is identical, which is the point: a caller opts into
the agent layer for the trace and the second retrieval round, not because
its own parsing has to change.

Failures inside the pipeline are not HTTP failures. A dead retriever or a
missing model produces `unverified` with an explanation, which is a
useful answer; only a genuinely unreadable request is a 4xx.
"""

import logging
import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..pipeline import analyze, analyze_image, analyze_link, analyze_text, analyze_video
from .shape import respond


log = logging.getLogger(__name__)


app = FastAPI(
    title="Misinformation Detection API",
    version="2.0.0",
    description=(
        "Checks forwarded text, links, images and videos for misinformation. "
        "Runs with no API keys: it falls back to a local evidence index and "
        "returns 'unverified' rather than guessing."
    ),
)


MAX_TEXT = 10000


class TextRequest(BaseModel):
    text: str


class LinkRequest(BaseModel):
    url: str


class MediaRequest(BaseModel):
    path: str
    caption: str = ""


def _run(function, *args, graph=False, agentic=False, **kwargs):
    """Call a pipeline entry point, turning only real failures into 5xx."""
    try:
        result = function(*args, graph_json=graph, agentic=agentic, **kwargs)
    except Exception as error:
        log.exception("pipeline failed")

        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {error}"
        ) from error

    return respond(result, include_graph=graph)


@app.get("/health")
def health():
    """Liveness, plus which optional retrievers and models are actually on."""
    from ..evidence.retrievers import RETRIEVERS
    from ..images import consistency, local_index
    from ..stance import nli, rank

    return {
        "status": "ok",
        "service": "misinformation-detection-api",
        "version": app.version,
        "retrievers": {name: module.available() for name, module in RETRIEVERS.items()},
        "models": {
            "embedder": rank.available(),
            "nli": nli.available(),
            "clip": consistency.available(),
        },
        "image_index": local_index.available(),
        "cache_dir": os.getenv("CACHE_DIR", "cache"),
    }


@app.post("/check/text")
def check_text(request: TextRequest, graph: bool = Query(False),
               agentic: bool = Query(False)):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    if len(text) > MAX_TEXT:
        raise HTTPException(
            status_code=413,
            detail=f"Text is too long. Maximum length is {MAX_TEXT:,} characters.",
        )

    return _run(analyze_text, text, graph=graph, agentic=agentic)


@app.post("/check/link")
def check_link(request: LinkRequest, graph: bool = Query(False),
               agentic: bool = Query(False)):
    url = request.url.strip()

    if not url:
        raise HTTPException(status_code=422, detail="URL cannot be empty")

    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422, detail="URL must start with http:// or https://"
        )

    return _run(analyze_link, url, graph=graph, agentic=agentic)


@app.post("/check/image")
def check_image(request: MediaRequest, graph: bool = Query(False),
                agentic: bool = Query(False)):
    """
    Check an image already on disk.

    A path rather than an upload: the analyzers work on files, and the
    bot that feeds this API has already saved what it received. Upload
    handling belongs in front of this, not inside it.
    """
    path = request.path.strip()

    if not path:
        raise HTTPException(status_code=422, detail="Path cannot be empty")

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No such file: {path}")

    return _run(analyze_image, path, request.caption, graph=graph, agentic=agentic)


@app.post("/check/video")
def check_video(request: MediaRequest, graph: bool = Query(False),
                agentic: bool = Query(False)):
    """Check a video: transcript plus deduplicated keyframes."""
    path = request.path.strip()

    if not path:
        raise HTTPException(status_code=422, detail="Path cannot be empty")

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No such file: {path}")

    return _run(analyze_video, path, request.caption, graph=graph, agentic=agentic)


@app.post("/check/packet")
def check_packet(packet: dict, graph: bool = Query(False),
                 agentic: bool = Query(False)):
    """
    Check an already-built packet.

    Useful when ingestion happened elsewhere (a worker that did the OCR,
    a cached packet) and only the checking needs doing.
    """
    if not isinstance(packet, dict) or not packet.get("input_type"):
        raise HTTPException(
            status_code=422, detail="A packet needs at least an input_type"
        )

    # Temp-path bookkeeping is ours, never the client's: a packet posted
    # with one would otherwise name directories for us to delete.
    from ..analyzers.packet import TEMP_PATHS

    packet.pop(TEMP_PATHS, None)

    return _run(analyze, packet, graph=graph, agentic=agentic)
