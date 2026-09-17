"""
Run the whole pipeline over a message and print what it concluded.

    python scripts/smoke_pipeline.py "SBI is giving Rs 5,000 cashback"
    python scripts/smoke_pipeline.py --all-samples
    python scripts/smoke_pipeline.py --sample fake_upi --html graph.html

This is the demo script: ingest, claims, graph, retrieval, images, stance
and verdict, with a per-stage timing breakdown so it is obvious where the
seconds went and which stage found nothing.

It works with no API keys — the local seed index answers offline — and it
says which retrievers were skipped for want of one, so an empty result is
never mysterious. Add `--html` to write the graph out as an interactive
page, which is the thing worth putting on a screen.
"""

import argparse
import glob
import json
import logging
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline import analyze_text                       # noqa: E402


SAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "samples"
)

DEFAULT_TEXT = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your reward."
)

BADGE = {
    "false": "[FALSE]",
    "misleading": "[MISLEADING]",
    "true": "[TRUE]",
    "disputed": "[DISPUTED]",
    "unverified": "[UNVERIFIED]",
}


def show_environment():
    from backend.evidence.retrievers import RETRIEVERS
    from backend.images import consistency, local_index
    from backend.stance import nli, rank

    print("=" * 72)
    print("environment")
    print("=" * 72)

    for name, module in RETRIEVERS.items():
        print(f"  retriever {name:<12} {'ready' if module.available() else 'SKIPPED (no key/model)'}")

    print(f"  embedder            {'ready' if rank.available() else 'SKIPPED'}")
    print(f"  nli                 {'ready' if nli.available() else 'SKIPPED'}")
    print(f"  clip                {'ready' if consistency.available() else 'SKIPPED'}")
    print(f"  image index         {'ready' if local_index.available() else 'SKIPPED (run build_image_index.py)'}")


def report(label, text, result, verbose=False):
    verdict = result["verdict"]

    print()
    print("=" * 72)
    print(f"{label}: {BADGE.get(verdict['label'], '[?]')} "
          f"(confidence {verdict['confidence']})")
    print("=" * 72)
    print(f"  {text[:200]}")
    print()
    print(f"  {verdict['summary']}")

    for claim in verdict.get("claims", []):
        print()
        print(f"  [{claim['label']}] {claim['claim'][:110]}")

        for reason in claim.get("reasons", []):
            print(f"      - {reason[:110]}")

    if result.get("timeline"):
        print()
        print("  timeline:")

        for event in result["timeline"]:
            print(f"      {event['date']}  {event['label'][:80]}")

    print()
    print("  stages:")

    for name, stage in result["stages"].items():
        extra = {
            key: value for key, value in stage.items()
            if key not in ("ms", "error") and value not in (None, 0, [], {})
        }
        line = f"      {name:<10} {stage['ms']:>8.1f} ms"

        if stage.get("error"):
            line += f"  ERROR: {stage['error']}"
        elif extra:
            line += f"  {json.dumps(extra, ensure_ascii=False)[:120]}"

        print(line)

    print(f"      {'total':<10} {result['ms']:>8.1f} ms")

    if verbose:
        print()
        print(result["graph"] if isinstance(result["graph"], str) else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--sample", help="a file from backend/samples, without .txt")
    parser.add_argument("--all-samples", action="store_true",
                        help="run every file in backend/samples")
    parser.add_argument("--backend", default="heuristic",
                        help="claim backend: heuristic (default), transformer, ollama, auto")
    parser.add_argument("--html", help="write the graph to this HTML file")
    parser.add_argument("--json", dest="json_path", help="write the graph JSON here")
    parser.add_argument("--no-retrieve", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    show_environment()

    jobs = []

    if args.all_samples:
        for path in sorted(glob.glob(os.path.join(SAMPLES, "*.txt"))):
            with open(path, encoding="utf-8") as handle:
                body = handle.read().strip()

            if body:
                jobs.append((os.path.basename(path), body))
    elif args.sample:
        path = os.path.join(SAMPLES, f"{args.sample}.txt")

        with open(path, encoding="utf-8") as handle:
            jobs.append((args.sample, handle.read().strip()))
    else:
        jobs.append(("message", args.text or DEFAULT_TEXT))

    want_graph = bool(args.html or args.json_path)

    for label, text in jobs:
        result = analyze_text(
            text,
            backend=args.backend,
            retrieve=not args.no_retrieve,
            graph_json=want_graph,
        )

        report(label, text, result, verbose=args.verbose)

        if want_graph:
            from backend.graph import EvidenceGraph

            graph = EvidenceGraph.from_dict(result["graph"])

            if args.json_path:
                graph.to_json(path=args.json_path)
                print(f"\n  graph written to {args.json_path}")

            if args.html:
                written = graph.export_html(args.html)
                print(f"  graph view written to {written}" if written
                      else "  pyvis is not installed; no HTML written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
