"""
Run the agent layer over a message and print what each agent decided.

    python scripts/smoke_agents.py "SBI is giving Rs 5,000 cashback"
    python scripts/smoke_agents.py --sample recycled_rumor
    python scripts/smoke_agents.py --all-samples --trace
    python scripts/smoke_agents.py --compare            # agents vs. the pipeline

This is the demo script for `backend/agents`: the plan, every agent's
status and notes, the tools each one actually called, the handoffs, and
the verdict at the end. `--trace` prints the tool calls in order with
their timings, which is the view that shows where a second retrieval
round went.

Like the rest of the project it works with no API keys. Offline, the run
you will see is the honest one: one usable retrieval path, an evidence
agent that abstains because it was asked and found nothing, and
`unverified` rather than a guess. `--compare` runs the linear pipeline on
the same text and prints both labels side by side — they are meant to
agree, because the agents call the same rules.
"""

import argparse
import glob
import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents import run_agentic                          # noqa: E402
from backend.analyzers.packet import from_text                  # noqa: E402
from backend.pipeline import analyze_text                       # noqa: E402


SAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "samples"
)

DEFAULT_TEXT = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your reward."
)

BADGE = {
    "ok": "[ok]",
    "abstained": "[abstained]",
    "skipped": "[skipped]",
    "failed": "[FAILED]",
}


def rule(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_plan(report):
    rule(f"plan (planner: {report['planner']})")

    for position, step in enumerate(report["steps"], start=1):
        marker = f"r{step['round']}" if step["round"] else "  "

        print(f"  {position:>2}. {marker} {step['agent']:<13} {step['reason']}")

    for note in report["plan"]["notes"]:
        print(f"      note: {note}")

    for entry in report["skipped"]:
        print(f"      skipped {entry['agent']}: {entry['why']}")


def show_agents(report):
    rule("agents")

    for name, result in report["results"].items():
        badge = BADGE.get(result["status"], result["status"])

        print(f"  {name:<13} {badge:<12} confidence {result['confidence']:.2f}"
              f"  {result['ms']:.0f} ms")

        for note in result["notes"]:
            print(f"      {note}")

        tools = [call["tool"] for call in result["tool_calls"]]

        if tools:
            print(f"      tools: {', '.join(tools)}")

    if report["handoffs"]:
        hops = " -> ".join(
            dict.fromkeys(
                [report["handoffs"][0]["from"]]
                + [hop["to"] for hop in report["handoffs"]]
            )
        )
        print(f"\n  handoffs: {hops}")

    print(f"  rounds: {report['rounds']}, agents dispatched: {report['dispatched']}")

    for note in report["notes"]:
        print(f"  note: {note}")


def show_trace(report):
    rule("tool calls, in order")

    for call in report["trace"]:
        status = "ok " if call["ok"] else "ERR"
        summary = json.dumps(call["summary"]) if call["summary"] else ""

        print(f"  {status} {call['ms']:>7.1f} ms  {call['agent'] or '-':<13} "
              f"{call['tool']:<22} {summary}")

        if call["error"]:
            print(f"      {call['error']}")


def show_verdict(result):
    verdict = result["verdict"]

    rule(f"verdict: {verdict['label'].upper()}  ({verdict['confidence']:.2f})")

    print(f"  summary:     {verdict['summary']}")

    if verdict.get("explanation"):
        print(f"  explanation: {verdict['explanation']}")

    for bullet in result["explanation"].get("bullets", []):
        print(f"    - {bullet}")

    print()

    for stage, detail in result["stages"].items():
        error = f"  ERROR {detail['error']}" if detail.get("error") else ""

        print(f"  stage {stage:<10} {detail['ms']:>8.1f} ms  "
              f"({detail['calls']} call(s) by {detail['agent'] or '-'}){error}")

    print(f"\n  total {result['ms']:.0f} ms")


def report_on(label, text, args):
    rule(f"{label}: {text[:60]}{'...' if len(text) > 60 else ''}")

    result = run_agentic(
        from_text(text),
        backend=args.backend,
        retrieve=not args.no_retrieve,
        max_rounds=args.max_rounds,
    )
    report = result["agents"]

    show_plan(report)
    show_agents(report)

    if args.trace:
        show_trace(report)

    show_verdict(result)

    if args.compare:
        linear = analyze_text(text, backend=args.backend,
                              retrieve=not args.no_retrieve)

        rule("agents vs. the linear pipeline")
        print(f"  agentic: {result['verdict']['label']:<12} "
              f"{result['ms']:>8.0f} ms")
        print(f"  linear:  {linear['verdict']['label']:<12} "
              f"{linear['ms']:>8.0f} ms")
        print("  the labels agree" if result["verdict"]["label"]
              == linear["verdict"]["label"] else "  THE LABELS DISAGREE")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--sample", help="a name from backend/samples")
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--backend", default="heuristic",
                        help="claim backend: heuristic (default), transformer, ollama, auto")
    parser.add_argument("--no-retrieve", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--trace", action="store_true",
                        help="print every tool call in order")
    parser.add_argument("--compare", action="store_true",
                        help="also run the linear pipeline and compare labels")
    args = parser.parse_args()

    if args.all_samples:
        for path in sorted(glob.glob(os.path.join(SAMPLES, "*.txt"))):
            with open(path, encoding="utf-8") as handle:
                report_on(os.path.basename(path), handle.read().strip(), args)

        return 0

    if args.sample:
        path = os.path.join(SAMPLES, f"{args.sample}.txt")

        if not os.path.isfile(path):
            print(f"no such sample: {path}", file=sys.stderr)

            return 1

        with open(path, encoding="utf-8") as handle:
            report_on(args.sample, handle.read().strip(), args)

        return 0

    report_on("message", args.text or DEFAULT_TEXT, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
