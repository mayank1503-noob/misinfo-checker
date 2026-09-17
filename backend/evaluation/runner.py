"""
Running the labelled set through the pipeline and recording what happened.

The harness calls `backend.pipeline.analyze` — the same entry point the
API and the bot use — rather than reaching into the stages. An evaluation
that exercises a private path measures a system nobody ships.

## Modes

The full pipeline loads a multilingual embedder, an NLI model and (with
images on) two CLIP towers, and on a modest laptop that is minutes per
case, not milliseconds. One slow honest mode would just mean the harness
never gets run, so there are three, and each one only claims the numbers
it actually earned:

| mode | stages | measures | cost |
|---|---|---|---|
| `claims` | 1-2 | check-worthiness | seconds |
| `retrieval` | 1-3b | + did the right entry come back | ~a minute |
| `full` | everything | + the verdict itself | minutes per case |

`retrieval` is the default because it is the one that catches what
actually breaks — a claim that is never extracted, or evidence that is
never found, makes the verdict moot — and it is fast enough to run while
you work.

**Outside `full`, `verdict` in each record is None**, not "unverified".
With stage 4 switched off every claim would come back unverified, and
recording that as a prediction would manufacture a meaningless 40%
accuracy against the cases that expect it. A metric that was never
measured is absent, not zero.

## The agent layer

`agentic=True` runs the same cases through `backend.agents` instead of the
fixed pipeline order, and records what the agents *did* alongside what the
pipeline concluded: the route the planner chose, which agents abstained,
whether retrieval escalated, and how many tool calls it took. The point of
evaluating the layer is not a second accuracy figure — the labels come from
the same rules either way — it is whether the coordination earns its cost:
does it change any label, does it change the *safety* number, and what does
it charge in seconds and calls to do it. `run_comparison` runs both and
`metrics.compare` scores the difference, which is the only honest way to
report an architecture change.

## Determinism

`today` is pinned (`EVAL_TODAY`), because stage 2 resolves "kal" and
"tomorrow" against it and an unpinned clock makes yesterday's report
incomparable with today's. Images are off: this set is text.
"""

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from datetime import date

from .dataset import load, select


log = logging.getLogger(__name__)


# Stage 2 resolves relative dates against this. Pinned so two runs a week
# apart are comparable; override with EVAL_TODAY=YYYY-MM-DD.
DEFAULT_TODAY = "2026-09-17"

MODES = ("claims", "retrieval", "full")

STAGES = {
    #           retrieve  stance
    "claims":   (False,   False),
    "retrieval": (True,   False),
    "full":     (True,    True),
}


@contextmanager
def nli_off():
    """
    Run as if the entailment model were not installed.

    Not a convenience, and no longer a workaround either. It used to be
    both: loading mDeBERTa alongside the embedder exhausted memory on the
    machine this was built on, so `--mode full` could not run and the
    verdict path went unmeasured. That is fixed (DECISIONS.md O3) —
    forward passes are now serialised through `backend/torch_runtime.py`
    and full mode runs with real entailment — so this is left for the
    reason it is worth having: a deployment that has the retrieval corpus
    but no entailment checkpoint is a real configuration, and this is how
    you measure it. With the NLI probe forced to False, stage 4 takes the
    path it already has for a missing model: a publisher's own rating
    decides the stance and everything without one is neutral.

    The embedder is deliberately left alone. The seed index needs it to
    retrieve anything, so switching it off as well would not measure a
    degraded verdict — it would measure an empty one.

    This is a real configuration: it is what a deployment with the
    retrieval corpus but no entailment model does. It is *not* the full
    system — it cannot judge evidence whose publisher gave no rating — so
    what it produces is a floor, and the report says so on its own line.

    Nothing is patched permanently; the probes are restored on the way
    out, which matters because a test that runs after this must see the
    real ones.
    """
    from ..images import consistency
    from ..stance import nli

    saved = [
        (nli, "available", nli.available),
        (consistency, "available", consistency.available),
    ]

    for module, name, _original in saved:
        setattr(module, name, lambda: False)

    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def today():
    return os.getenv("EVAL_TODAY", DEFAULT_TODAY)


def _seed_ids(graph):
    """Every seed-index entry that ended up attached to a claim."""
    found = []

    for node in (graph or {}).get("nodes", []):
        if node.get("kind") != "evidence":
            continue

        if node.get("source_type") != "seed_index":
            continue

        seed_id = (node.get("meta") or {}).get("seed_id")

        if seed_id and seed_id not in found:
            found.append(seed_id)

    return found


def _agent_fields(result):
    """
    What the agent layer did on one case, flattened for the metrics.

    Deliberately small: statuses, the route, the escalation and the
    counts. The full trace is in the pipeline result and belongs in a
    `--json` dump, not in a per-case record that a report iterates.
    """
    report = result.get("agents") or {}
    results = report.get("results") or {}
    evidence = (results.get("evidence") or {}).get("data") or {}
    explanation = result.get("explanation") or {}

    return {
        "planner": report.get("planner"),
        "route": [step["agent"] for step in report.get("plan", {}).get("steps", [])],
        "dispatched": report.get("dispatched"),
        "rounds": report.get("rounds"),
        "skipped": [entry["agent"] for entry in report.get("skipped") or []],
        "statuses": {name: row["status"] for name, row in results.items()},
        "abstained": report.get("abstained") or [],
        "failed": report.get("failed") or [],
        "escalated": bool(evidence.get("escalated")),
        "retrievers": evidence.get("retrievers") or [],
        "kept": evidence.get("kept"),
        "decisive": evidence.get("decisive"),
        "tool_calls": len(report.get("trace") or []),
        "explanation": explanation.get("explanation"),
    }


def _stage_errors(stages):
    return {
        name: block["error"]
        for name, block in (stages or {}).items()
        if block.get("error")
    }


def run_case(case, mode="retrieval", analyze=None, agentic=False, nli=True):
    """
    One case through the pipeline; returns a flat record for the metrics.

    A stage that fails soft is recorded in `stage_errors` rather than
    being allowed to look like a prediction — a crashed retriever and a
    retriever that found nothing are the same empty list, and the report
    has to be able to tell them apart.
    """
    if mode not in STAGES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

    if analyze is None:                              # imported late: loads torch
        from ..pipeline import analyze as analyze

    retrieve, stance = STAGES[mode]
    started = time.perf_counter()

    with nli_off() if not nli else nullcontext():
        result = analyze(
            {"input_type": "text", "text": case.text},
            today=date.fromisoformat(today()),
            retrieve=retrieve,
            stance=stance,
            images=False,
            graph_json=True,
            agentic=agentic,
        )

    stages = result.get("stages") or {}
    claims = stages.get("claims") or {}

    record = {
        "id": case.id,
        "language": case.language,
        "category": case.category,
        "in_corpus": case.in_corpus,
        "mode": mode,
        "text": case.text,

        "expected_verdict": case.expected_verdict,
        "expected_check_worthy": case.expects_check_worthy,
        "expected_seed_id": case.expected_seed_id,

        "check_worthy": bool(claims.get("check_worthy")),
        "n_claims": claims.get("claims"),
        "claims_backend": claims.get("backend"),

        "retrieved_seed_ids": _seed_ids(result.get("graph")) if retrieve else None,

        # Only a run that actually judged may report a verdict. See the
        # module docstring.
        "verdict": (result.get("verdict") or {}).get("label") if stance else None,
        "confidence": (result.get("verdict") or {}).get("confidence") if stance else None,

        "stage_errors": _stage_errors(stages),
        "ms": round((time.perf_counter() - started) * 1000, 1),

        # `None` rather than `{}` when the agents did not run, for the
        # same reason `verdict` is None outside `full`: a metric that was
        # never measured is absent, not empty.
        "agentic": agentic,
        "agents": _agent_fields(result) if agentic else None,

        # False means the entailment model was switched off for this
        # run, so stage 4 judged from publisher ratings alone. The report
        # says so: a number measured without it is a floor, not the system.
        "nli": nli,
    }

    return record


def run(cases=None, mode="retrieval", progress=None, analyze=None,
        agentic=False, nli=True, **filters):
    """
    Every case (or the filtered subset), as a list of records.

    `progress(index, total, record)` is called after each case — the CLI
    uses it to print a line, because a full run is long enough that
    silence looks like a hang.
    """
    cases = cases if cases is not None else load()

    if filters:
        cases = select(cases, **filters)

    records = []

    for index, case in enumerate(cases, start=1):
        try:
            record = run_case(
                case, mode=mode, analyze=analyze, agentic=agentic, nli=nli
            )
        except Exception as error:                   # one bad case ends one case
            log.warning(
                "case %s raised (%s): %s", case.id, type(error).__name__, error
            )
            record = {
                "id": case.id,
                "language": case.language,
                "category": case.category,
                "in_corpus": case.in_corpus,
                "mode": mode,
                "expected_verdict": case.expected_verdict,
                "expected_check_worthy": case.expects_check_worthy,
                "expected_seed_id": case.expected_seed_id,
                "check_worthy": None,
                "retrieved_seed_ids": None,
                "verdict": None,
                "stage_errors": {"runner": f"{type(error).__name__}: {error}"},
                "ms": 0.0,
                "agentic": agentic,
                "agents": None,
                "nli": nli,
            }

        records.append(record)

        if progress:
            progress(index, len(cases), record)

    return records


def run_comparison(cases=None, mode="full", progress=None, analyze=None,
                   nli=True, **filters):
    """
    Every case both ways, plus the scored difference.

    Returns `{"linear": [...], "agentic": [...], "comparison": {...}}`.
    The linear run goes first so the models are warm for both and the
    second run is not charged for a load the first one paid for -
    otherwise the latency comparison would say the agent layer is slower
    by exactly the cost of importing torch.

    This is the function that answers the only interesting question about
    an architecture change: did it move a label, did it move the safety
    number, and what did it cost.
    """
    from . import metrics

    cases = cases if cases is not None else load()

    if filters:
        cases = select(cases, **filters)

    def tagged(tag):
        if progress is None:
            return None

        def report(index, total, record):
            progress(index, total, {**record, "id": f"{tag}:{record['id']}"})

        return report

    linear = run(
        cases, mode=mode, progress=tagged("linear"), analyze=analyze, nli=nli
    )
    agentic = run(
        cases, mode=mode, progress=tagged("agents"), analyze=analyze,
        agentic=True, nli=nli,
    )

    return {
        "linear": linear,
        "agentic": agentic,
        "comparison": metrics.compare(linear, agentic),
    }


def warm_up():
    """
    Load the shared embedder before the first case.

    The seed index does this for its own thread pool; doing it here as
    well keeps the first case's timing from including a model load and
    makes the per-case numbers in the report comparable.
    """
    try:
        from ..evidence.retrievers import seed_index

        return seed_index.warm_up()
    except Exception as error:                       # fail soft, never raise
        log.warning("warm-up failed (%s): %s", type(error).__name__, error)
        return False
