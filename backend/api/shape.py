"""
The wire format, in one place.

Both front doors — `main.py` (FastAPI, what the bot talks to) and
`flask_app.py` (Flask, what the browser talks to) — return the same
JSON for the same pipeline result. That only stays true if there is
one function that builds it, so this is it.

The shape is deliberately layered, because two very different callers
use it: a bot wants `verdict.summary` and nothing else, while a reviewer
wants the claims, the evidence and the graph. Nobody is forced to parse
what they do not need.
"""


def respond(result, include_graph=False):
    """
    Shape one pipeline result for the wire.

    `results` is the per-claim list, one entry per check-worthy claim
    with its label, its confidence and the evidence behind it. `agents`
    and `explanation` appear only on an agentic run, so the default
    response is byte-for-byte what it has always been.
    """
    verdict = result["verdict"]

    return {
        "verdict": {
            "label": verdict["label"],
            "confidence": verdict["confidence"],
            "summary": verdict["summary"],
            "counts": verdict.get("counts", {}),
        },
        "results": [
            {
                "claim": claim.get("claim"),
                "claim_id": claim.get("claim_id"),
                "label": claim["label"],
                "confidence": claim["confidence"],
                "explanation": claim["explanation"],
                "reasons": claim.get("reasons", []),
                "evidence_ids": claim.get("evidence_ids", []),
                "demo_only": claim.get("demo_only", False),
            }
            for claim in verdict.get("claims", [])
        ],
        "packet": result["packet"],
        "claims": result["claims"],
        "timeline": result.get("timeline", []),
        "open_claims": result.get("open_claims", []),
        "stages": result["stages"],
        "ms": result["ms"],
        **({"graph": result["graph"]} if include_graph else {}),
        **({"agents": result["agents"]} if result.get("agents") else {}),
        **({"explanation": result["explanation"]} if result.get("explanation") else {}),
    }
