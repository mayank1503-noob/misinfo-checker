"""
The command line:

    python -m backend.evaluation                      # fast, the default
    python -m backend.evaluation --mode full          # the whole pipeline
    python -m backend.evaluation --mode claims        # stage 2 only

    python -m backend.evaluation --language hinglish  # one slice
    python -m backend.evaluation --case chat_salt --case hot_water_hinglish
    python -m backend.evaluation --json runs/today.json

    python -m backend.evaluation --mode full --gate   # exit 1 on a wrong
                                                      # accusation

    python -m backend.evaluation --agentic            # the agent layer drives
    python -m backend.evaluation --mode full --no-nli   # verdicts from ratings
                                                        # alone, on a small box
    python -m backend.evaluation --mode full --compare  # both, side by side

`--agentic` runs the cases through `backend.agents` and adds an AGENT
LAYER section: the routes the planner chose, which agents abstained,
how often retrieval escalated and what that yielded. `--compare` runs
both paths over the same cases and scores the difference, which is the
question an architecture change actually has to answer.

Runs entirely offline against the demo corpora, like everything else in
this project: no keys, no network.
"""

import argparse
import json
import sys
import time

from . import dataset, metrics, report, runner


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m backend.evaluation",
        description="Score the pipeline against data/eval_cases.jsonl.",
    )
    parser.add_argument(
        "--mode", choices=runner.MODES, default="retrieval",
        help="how much of the pipeline to run (default: retrieval). "
             "'full' loads the NLI model and takes roughly a minute a case.",
    )
    parser.add_argument("--language", choices=dataset.LANGUAGES,
                        help="only cases in this language")
    parser.add_argument("--category", help="only cases in this category")
    parser.add_argument("--case", action="append", dest="ids", metavar="ID",
                        help="only this case; repeatable")
    parser.add_argument("--in-corpus", dest="in_corpus", action="store_true",
                        default=None,
                        help="only cases whose answer is in the demo index")
    parser.add_argument("--out-of-corpus", dest="in_corpus", action="store_false",
                        help="only cases whose answer is not")
    parser.add_argument("--cases", dest="path", help="a different eval set")
    parser.add_argument("--json", dest="json_path", metavar="PATH",
                        help="also write the full result as JSON")
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if the run is not shippable")
    parser.add_argument("--no-nli", dest="nli", action="store_false",
                        help="run as if the entailment model were not installed: "
                             "stage 4 judges from publisher ratings alone. The "
                             "retrieval embedder stays on. This is the only way "
                             "to measure the verdict path on a machine that "
                             "cannot hold both checkpoints; the report says so, "
                             "and the numbers are a floor.")
    parser.add_argument("--agentic", action="store_true",
                        help="drive the stages with backend.agents")
    parser.add_argument("--compare", action="store_true",
                        help="run both the linear pipeline and the agent layer, "
                             "and score the difference")
    parser.add_argument("--quiet", action="store_true",
                        help="no per-case progress")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    cases = dataset.load(args.path)
    cases = dataset.select(
        cases,
        language=args.language,
        category=args.category,
        ids=set(args.ids) if args.ids else None,
        in_corpus=args.in_corpus,
    )

    if not cases:
        print("no cases matched those filters", file=sys.stderr)
        return 2

    if args.mode == "full" and args.nli and not args.quiet:
        print(
            f"running {len(cases)} cases through the full pipeline; "
            "this loads the NLI model and takes about a minute a case",
            file=sys.stderr,
        )

    def progress(index, total, record):
        if args.quiet:
            return

        got = record.get("verdict") or (
            "worthy" if record.get("check_worthy") else "not worthy"
        )
        print(
            f"  [{index:>3}/{total}] {record['id']:<26} {record['ms']:>8.0f}ms  {got}",
            file=sys.stderr,
        )

    started = time.perf_counter()

    # Only the retrieving modes need the embedder; loading it for a
    # stage-2 run would add a minute to a run that takes seconds.
    if args.mode != "claims":
        runner.warm_up()

    comparison = None

    if args.compare:
        # Both paths over the same cases. The linear run goes first so
        # the agentic one is not charged for the model load.
        both = runner.run_comparison(
            cases, mode=args.mode, progress=progress, nli=args.nli
        )
        records = both["linear"]
        agentic_records = both["agentic"]
        comparison = both["comparison"]
    else:
        records = runner.run(
            cases, mode=args.mode, progress=progress, agentic=args.agentic,
            nli=args.nli,
        )
        agentic_records = None

    elapsed = time.perf_counter() - started

    summary = metrics.summarize(records)

    if not args.quiet:
        print("", file=sys.stderr)

    print(report.render(summary, records, mode=args.mode, elapsed=elapsed))

    if comparison is not None:
        print()
        print(report.render_comparison(comparison, elapsed=elapsed))

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mode": args.mode,
                    "today": runner.today(),
                    "agentic": args.agentic or args.compare,
                    "nli": args.nli,
                    "elapsed_s": round(elapsed, 1),
                    "summary": summary,
                    "records": records,
                    **({"comparison": comparison} if comparison else {}),
                    **({"agentic_records": agentic_records} if agentic_records else {}),
                },
                handle, indent=2, ensure_ascii=False,
            )

        print(f"\nwrote {args.json_path}")

    if args.gate:
        gated = summary

        if agentic_records is not None:
            #  On a comparison the agent run is the one being proposed,
            #  so it is the one the gate has to hold.
            gated = metrics.summarize(agentic_records)

        ok, reasons = metrics.gate(gated)

        if not ok:
            print("\nGATE FAILED", file=sys.stderr)

            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)

            return 1

        print("\ngate passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
