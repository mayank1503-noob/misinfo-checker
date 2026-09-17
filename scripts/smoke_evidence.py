"""
Live smoke test for Stage 3b retrieval — the one thing the test suite
cannot check, because the test suite is not allowed to touch the network.

    python scripts/smoke_evidence.py
    python scripts/smoke_evidence.py "SBI is giving Rs 5,000 cashback today"
    python scripts/smoke_evidence.py --sample fake_upi --json out.json

It runs the real retrievers against whatever keys are in the environment,
prints what each one found, and says plainly which ones were skipped for
want of a key. Everything it fetches lands in `cache/`, so a second run
costs nothing and works offline — which is the point: run this once
before a demo and the demo no longer needs the network.

Keys (all optional):
    GOOGLE_FACTCHECK_KEY   Google Fact Check Tools
    TAVILY_API_KEY         Tavily web search
"""

import argparse
import json
import logging
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.claims import extract_claims                      # noqa: E402
from backend.evidence import collect_evidence                  # noqa: E402
from backend.evidence.queries import build_queries             # noqa: E402
from backend.evidence.retrievers import RETRIEVERS             # noqa: E402
from backend.graph import EvidenceGraph                        # noqa: E402


DEFAULT_TEXT = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your reward. "
    "RBI has confirmed the scheme."
)

SAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "samples"
)


def load_sample(name):
    path = os.path.join(SAMPLES, f"{name}.txt")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError as error:
        print(f"could not read {path}: {error}")
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=None, help="the message to check")
    parser.add_argument("--sample", help="a file from backend/samples, without .txt")
    parser.add_argument("--json", dest="json_path", help="write the graph to this path")
    parser.add_argument("--html", dest="html_path", help="write a pyvis view here")
    parser.add_argument("--backend", default="heuristic",
                        help="claim extraction backend (default: heuristic, no models)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    text = args.text or (load_sample(args.sample) if args.sample else None) or DEFAULT_TEXT

    print("=" * 70)
    print("retrievers")
    print("=" * 70)

    for name, module in RETRIEVERS.items():
        state = "ready" if module.available() else "SKIPPED (no key or no model)"
        print(f"  {name:<12} {state}")

    print()
    print("=" * 70)
    print("message")
    print("=" * 70)
    print(text[:500])

    packet = {"input_type": "text", "text": text, "source_date": None, "images": []}
    claimset = extract_claims(packet, backend=args.backend)

    print()
    print(f"{len(claimset.claims)} claims, {len(claimset.check_worthy())} check-worthy "
          f"(backend: {claimset.backend})")

    graph = EvidenceGraph.from_claimset(claimset, packet)

    for claim in claimset.check_worthy():
        print()
        print("-" * 70)
        print(f"[{claim.claim_type}] {claim.text}")

        for query in build_queries(claim):
            scope = f" site:{','.join(query.sites[:3])}..." if query.sites else ""
            print(f"    query ({query.kind}, {query.language}): {query.text[:90]}{scope}")

    report = collect_evidence(claimset, graph)

    print()
    print("=" * 70)
    print("results")
    print("=" * 70)
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))

    for claim in claimset.check_worthy():
        evidence = graph.evidence_for(claim.id)

        if not evidence:
            continue

        print()
        print(f"{claim.text[:70]}")

        for item in evidence:
            flags = []

            if item.get("decisive"):
                flags.append("DECISIVE")

            if item.get("demo"):
                flags.append("demo data")

            marks = f"  [{', '.join(flags)}]" if flags else ""

            print(f"  - ({item.get('source_type')}, w={item.get('source_weight')}) "
                  f"{item.get('publisher') or item.get('domain')}"
                  f" [{item.get('rating') or 'no rating'}]{marks}")
            print(f"    {(item.get('title') or '')[:90]}")

            if item.get("url"):
                print(f"    {item['url']}")

    if args.json_path:
        graph.to_json(path=args.json_path)
        print(f"\ngraph written to {args.json_path}")

    if args.html_path:
        written = graph.export_html(args.html_path)
        print(f"graph view written to {written}" if written
              else "pyvis is not installed; no HTML written")

    print()
    print("Stances are all None here: deciding what the evidence means is "
          "stage 4 (backend/stance).")


if __name__ == "__main__":
    main()
