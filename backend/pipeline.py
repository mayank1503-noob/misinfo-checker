"""
The whole thing, end to end.

    from backend.pipeline import analyze, analyze_text

    result = analyze_text("SBI is giving Rs 5,000 cashback, forward to 10 people")
    result["verdict"]["label"]        # false / misleading / true / disputed / unverified
    result["verdict"]["summary"]      # one sentence a bot can send back

Stage by stage:

    1  ingest      backend.analyzers.packet   -> MediaPacket
    2  claims      backend.claims             -> ClaimSet
    3a graph       backend.graph              -> EvidenceGraph
    3b retrieval   backend.evidence           -> evidence nodes
    5  images      backend.images             -> duplicates, date mismatches
    4  stance      backend.stance             -> SUPPORTS / REFUTES / NEUTRAL
    6  verdict     backend.verdict            -> labels, written into the graph

Images run *before* stance on purpose: the CLIP check writes its own
stance, and the stance pass then has the image evidence in front of it
along with everything else, in one batched model call per claim rather
than two.

Pass `agentic=True` and the same stages are driven by `backend.agents`
instead of by this fixed order: an orchestrator plans which specialists
are worth running for *this* message, each one may abstain, and retrieval
gets a second, targeted round when a claim is left open. The result has
the same shape, with an extra `agents` block. This module remains the
default path, and it stays linear on purpose — it is the one that always
finishes, with no keys, no models and no planner.

Every stage is optional and every stage fails soft. With no API keys and
no models installed this still returns a well-formed result — claims from
the regex backend, whatever the local seed index knows, and `unverified`
for anything it cannot settle. That is the point: the pipeline degrades,
it does not break.
"""

import logging
import time

from .claims import extract_claims
from .evidence import collect_evidence
from .graph import EvidenceGraph
from .images import collect_image_evidence
from .stance import apply_stances
from .verdict import decide


log = logging.getLogger(__name__)


def analyze(packet, backend="auto", today=None, retrieve=True, images=True,
            stance=True, graph_json=False, max_text=None, agentic=False,
            **agent_options):
    """
    Run the pipeline over a packet from `backend.analyzers.packet`.

    Returns a dict with `packet`, `claims`, `verdict`, `graph` (a
    summary, or the full serialised graph with `graph_json=True`) and a
    `stages` block reporting what each stage did and how long it took.

    Toggles exist for the same reason the stages fail soft: a bot reply
    needs an answer in a second or two, and `retrieve=False` gives a
    claims-only reading instantly.

    With `agentic=True` the same stages are run by `backend.agents`,
    which plans the route, lets each specialist abstain, and escalates
    retrieval when a claim stays open. The result has the same shape plus
    an `agents` block, so nothing downstream has to know which path ran.
    `agent_options` is forwarded (`planner`, `agents`, `tools`, `writer`,
    `max_rounds`) and is ignored on the linear path.
    """
    if agentic:
        from .agents import run_agentic

        return run_agentic(
            packet, backend=backend, today=today, retrieve=retrieve,
            images=images, stance=stance, graph_json=graph_json,
            max_text=max_text, **agent_options,
        )

    started = time.perf_counter()
    stages = {}

    def timed(name, function, *args, **kwargs):
        """Run a stage, record its timing, and never let it raise."""
        stage_started = time.perf_counter()

        try:
            result = function(*args, **kwargs)
            error = None
        except Exception as failure:                 # fail soft, never raise
            log.warning("stage %s failed (%s): %s", name, type(failure).__name__, failure)
            result, error = None, f"{type(failure).__name__}: {failure}"

        stages[name] = {
            "ms": round((time.perf_counter() - stage_started) * 1000, 1),
            "error": error,
        }

        return result

    packet = dict(packet or {})

    if max_text and packet.get("text"):
        packet["text"] = packet["text"][:max_text]

    claimset = timed(
        "claims", extract_claims, packet, today=today, backend=backend
    )

    if claimset is None:
        # Without claims there is nothing to check; return the packet and
        # say so rather than inventing a verdict.
        return {
            "packet": _public_packet(packet),
            "claims": [],
            "verdict": {
                "label": "unverified",
                "confidence": 0.0,
                "summary": "The message could not be read as any checkable claim.",
                "counts": {},
                "claims": [],
            },
            "graph": None,
            "stages": stages,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }

    stages["claims"]["claims"] = len(claimset.claims)
    stages["claims"]["check_worthy"] = len(claimset.check_worthy())
    stages["claims"]["backend"] = claimset.backend

    graph = timed("graph", EvidenceGraph.from_claimset, claimset, packet)

    if graph is None:
        graph = EvidenceGraph(packet_id=claimset.packet_id,
                              input_type=claimset.input_type,
                              source_date=claimset.source_date)

    if retrieve:
        report = timed("evidence", collect_evidence, claimset, graph)

        if report is not None:
            stages["evidence"].update(report.as_dict())

    if images and (packet.get("images") or []):
        report = timed("images", collect_image_evidence, claimset, graph, packet)

        if report is not None:
            stages["images"].update(report.as_dict())

    if stance:
        report = timed("stance", apply_stances, graph)

        if report is not None:
            stages["stance"].update(report.as_dict())

    result = timed("verdict", decide, graph) or {
        "label": "unverified",
        "confidence": 0.0,
        "summary": "The verdict stage failed; nothing can be concluded.",
        "counts": {},
        "claims": [],
    }

    return {
        "packet": _public_packet(packet),
        "claims": claimset.model_dump(mode="json")["claims"],
        "dropped": claimset.dropped,
        "verdict": result,
        "graph": graph.to_dict() if graph_json else graph.summary(2000),
        "open_claims": graph.open_claims(),
        "timeline": graph.timeline(),
        "stages": stages,
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _public_packet(packet):
    """
    The packet without the embeddings.

    A 768-float vector per image is noise in an API response and a
    hundred kilobytes in a bot reply; the graph drops them for the same
    reason. The temp-path bookkeeping goes the same way.
    """
    from .analyzers.packet import TEMP_PATHS

    public = dict(packet)

    public.pop(TEMP_PATHS, None)

    public["images"] = [
        {key: value for key, value in (image or {}).items() if key != "embedding"}
        for image in (packet.get("images") or [])
    ]

    return public


def analyze_text(text, **kwargs):
    """Check a plain text message."""
    from .analyzers.packet import from_text

    return analyze(from_text(text), **kwargs)


def analyze_link(url, **kwargs):
    """Check a link: the article is fetched and its body is what gets checked."""
    from .analyzers.packet import from_link

    return analyze(from_link(url), **kwargs)


def analyze_image(path, caption="", **kwargs):
    """Check an image: OCR, caption, embedding, EXIF, then the pipeline."""
    from .analyzers.packet import from_image

    return analyze(from_image(path, caption), **kwargs)


def analyze_video(path, caption="", **kwargs):
    """
    Check a video: transcript plus keyframes, then the pipeline.

    The keyframes are temp files the packet owns; they are read right
    through stage 5, so they are only removed once `analyze` returns.
    The video at `path` is the caller's and is left alone.
    """
    from .analyzers.packet import cleanup, from_video

    packet = from_video(path, caption)

    try:
        return analyze(packet, **kwargs)
    finally:
        cleanup(packet)


def analyze_video_url(url, caption="", **kwargs):
    """Check a video by URL: it is downloaded, checked, then deleted."""
    from .analyzers.packet import cleanup, from_video_url

    packet = from_video_url(url, caption)

    try:
        return analyze(packet, **kwargs)
    finally:
        cleanup(packet)
