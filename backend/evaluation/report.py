"""
Rendering a summary as something a person reads in the terminal.

Kept apart from `metrics.py` so the arithmetic can be tested without
matching strings, and so a caller that wants JSON never builds any of
this. The layout puts the asymmetric errors at the top, because the
number that decides whether a build is shippable should not be something
you scroll to find.
"""

from .dataset import ACCUSATIONS


BAR = "=" * 72
RULE = "-" * 72


def _pct(value):
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _count(value):
    return "-" if value is None else str(value)


def _line(label, value, width=34):
    return f"  {label:<{width}} {value}"


def agent_section(agents):
    """
    The agent layer, when it drove the run.

    Reads as "what did the coordination do and what did it charge",
    because the labels are already scored above and scoring them twice
    would suggest the layer moved them when it did not.
    """
    out = ["AGENT LAYER", RULE]

    out.append(_line("cases the agents drove", agents["n"]))
    out.append(_line(
        "planner", ", ".join(
            f"{name} x{count}" for name, count in sorted(agents["planners"].items())
        )
    ))

    out.append(_line("routes chosen", ""))

    for route, count in agents["routes"].items():
        out.append(f"      {count:>3}x  {route}")

    out.append(_line("abstentions by agent", ", ".join(
        f"{name} {count}" for name, count in sorted(agents["abstentions"].items())
    ) or "none"))

    if agents["failures"]:
        out.append(_line("AGENTS THAT FAILED", ", ".join(
            f"{name} x{count}" for name, count in sorted(agents["failures"].items())
        )))

        for case_id in agents["failed_ids"][:8]:
            out.append(f"      ! {case_id}")

    out.append(_line(
        "escalated to a second round",
        f"{_pct(agents['escalation_rate'])}   {agents['escalated']}/{agents['n']}",
    ))
    out.append(_line(
        "  of those, reached a label",
        f"{_pct(agents['escalation_yield'])}   "
        f"{agents['escalation_settled']}/{agents['escalated']}",
    ))
    out.append(_line("retrievers asked", ", ".join(
        f"{name} {count}" for name, count in sorted(agents["retrievers"].items())
    ) or "none"))
    out.append(_line("tool calls per case (mean)", _count(agents["tool_calls_mean"])))
    out.append(_line("ms per case (mean)", _count(agents["ms_mean"])))
    out.append(_line(
        "cases with an explanation", f"{agents['explained']}/{agents['n']}"
    ))
    out.append("")

    return out


def render_comparison(comparison, elapsed=None):
    """
    The linear pipeline against the agent layer.

    Agreement first, then the two numbers that decide whether the layer
    ships: did safety move, and what does it cost per case. A
    disagreement is only a result once it is classified as an improvement
    or a regression, so they are printed apart.
    """
    out = [BAR, "  LINEAR PIPELINE vs. AGENT LAYER", BAR, ""]

    out.append(_line("cases", comparison["n"]))
    out.append(_line("cases both runs judged", comparison["judged"]))

    if elapsed is not None:
        out.append(_line("elapsed (both runs)", f"{elapsed:.1f}s"))

    out.append("")

    if not comparison["judged"]:
        out.append("  neither run produced a verdict in this mode - "
                   "run with --mode full")
        out.append("")
        out.append(BAR)

        return "\n".join(out)

    out.append("LABELS")
    out.append(RULE)
    out.append(_line(
        "agreement",
        f"{_pct(comparison['agreement'])}   "
        f"{comparison['agreed']}/{comparison['judged']}",
    ))
    out.append(_line("agentic is right, linear is not",
                     len(comparison["improvements"])))

    for entry in comparison["improvements"][:8]:
        out.append(f"      + {entry['id']:<24} {entry['linear']} -> {entry['agentic']}"
                   f"   (expected {entry['expected']})")

    out.append(_line("linear is right, agentic is not",
                     len(comparison["regressions"])))

    for entry in comparison["regressions"][:8]:
        out.append(f"      ! {entry['id']:<24} {entry['linear']} -> {entry['agentic']}"
                   f"   (expected {entry['expected']})")

    if comparison["neither"]:
        out.append(_line("differ, both wrong", len(comparison["neither"])))

        for entry in comparison["neither"][:8]:
            out.append(f"      - {entry['id']:<24} {entry['linear']} -> "
                       f"{entry['agentic']}   (expected {entry['expected']})")

    out.append("")
    out.append("SAFETY AND COST")
    out.append(RULE)
    out.append(f"  {'':<34} {'linear':>10}  {'agents':>10}")
    out.append(f"  {'wrong accusations (must be 0)':<34} "
               f"{_count(comparison['safety']['linear']):>10}  "
               f"{_count(comparison['safety']['agentic']):>10}")
    out.append(f"  {'accuracy':<34} "
               f"{_pct(comparison['accuracy']['linear']):>10}  "
               f"{_pct(comparison['accuracy']['agentic']):>10}")
    out.append(f"  {'ms per case (mean)':<34} "
               f"{_count(comparison['ms']['linear']):>10}  "
               f"{_count(comparison['ms']['agentic']):>10}")
    out.append("")

    if comparison.get("agents"):
        out.extend(agent_section(comparison["agents"]))

    out.append(BAR)

    return "\n".join(out)


def render(summary, records, mode="retrieval", elapsed=None):
    """The whole report as one string."""
    out = []
    overall = summary["overall"]
    verdict = overall["verdict"]
    worthy = overall["check_worthy"]
    retrieval = overall["retrieval"]

    out.append(BAR)
    out.append(f"  MISINFO CHECKER - EVALUATION ({mode} mode)")
    out.append(BAR)
    out.append("")
    out.append(_line("cases", overall["n"]))

    if elapsed is not None:
        out.append(_line("elapsed", f"{elapsed:.1f}s"))

    if records and all(record.get("nli") is False for record in records):
        out.append(_line(
            "NLI MODEL", "off - stage 4 judged from publisher ratings alone"
        ))
        out.append(_line("", "(so these verdict numbers are a floor)"))

    errors = sorted({
        name for record in records for name in (record.get("stage_errors") or {})
    })

    if errors:
        out.append(_line("STAGES THAT FAILED", ", ".join(errors)))

    out.append("")

    # --- the two that are not symmetric --------------------------------
    if verdict.get("n"):
        out.append("SAFETY")
        out.append(RULE)
        out.append(_line(
            "wrong accusations (must be 0)", _count(verdict["false_accusations"])
        ))

        for case_id in verdict["false_accusation_ids"]:
            out.append(f"      ! {case_id}")

        out.append(_line("rumours missed", _count(verdict["missed_rumours"])))

        for case_id in verdict["missed_rumour_ids"][:8]:
            out.append(f"      - {case_id}")

        out.append("")
        out.append("VERDICT")
        out.append(RULE)
        out.append(_line("accuracy (all cases)", _pct(verdict["accuracy"])))
        out.append(_line(
            "  answer is in the demo index",
            _pct(summary["in_corpus"]["verdict"].get("accuracy")) +
            f"   n={summary['in_corpus']['n']}",
        ))
        out.append(_line(
            "  answer is not (says nothing?)",
            _pct(summary["out_of_corpus"]["verdict"].get("accuracy")) +
            f"   n={summary['out_of_corpus']['n']}",
        ))
        out.append(_line("abstained when an answer existed",
                         _count(verdict["abstentions"])))
        out.append("")
    else:
        out.append("VERDICT")
        out.append(RULE)
        out.append("  not measured in this mode - run with --mode full")
        out.append("")

    # --- stage 2 --------------------------------------------------------
    out.append("STAGE 2  check-worthiness")
    out.append(RULE)
    out.append(_line("precision", _pct(worthy.get("precision"))))
    out.append(_line("recall", _pct(worthy.get("recall"))))
    out.append(_line("f1", _pct(worthy.get("f1"))))

    if worthy.get("false_positive_ids"):
        out.append(_line("chat scored as check-worthy",
                         len(worthy["false_positive_ids"])))

        for case_id in worthy["false_positive_ids"][:8]:
            out.append(f"      - {case_id}")

    out.append("")

    # --- stage 3b -------------------------------------------------------
    out.append("STAGE 3b  retrieval")
    out.append(RULE)
    out.append(_line(
        "expected entry found",
        f"{_pct(retrieval['hit_rate'])}   {retrieval['hits']}/{retrieval['n_targeted']}",
    ))

    for case_id in retrieval["miss_ids"][:8]:
        out.append(f"      - {case_id}")

    out.append(_line(
        "stray hits (no right answer)",
        f"{_pct(retrieval['stray_rate'])}   "
        f"{len(retrieval['stray_ids'])}/{retrieval['n_untargeted']}",
    ))

    for case_id in retrieval["stray_ids"][:8]:
        out.append(f"      ! {case_id}")

    out.append("")

    # --- the agent layer, when it ran -----------------------------------
    if overall.get("agents"):
        out.extend(agent_section(overall["agents"]))

    # --- by language ----------------------------------------------------
    out.append("BY LANGUAGE")
    out.append(RULE)
    out.append(f"  {'':<10} {'n':>3}  {'retrieval':>10}  {'worthy f1':>10}  {'verdict':>10}")

    for language, block in summary["by_language"].items():
        out.append(
            f"  {str(language):<10} {block['n']:>3}  "
            f"{_pct(block['retrieval']['hit_rate']):>10}  "
            f"{_pct(block['check_worthy'].get('f1')):>10}  "
            f"{_pct(block['verdict'].get('accuracy')):>10}"
        )

    out.append("")

    # --- what went wrong ------------------------------------------------
    if summary["errors"]:
        out.append("MISLABELLED CASES")
        out.append(RULE)

        for error in summary["errors"]:
            marker = "!" if error["got"] in ACCUSATIONS else " "
            out.append(
                f"  {marker} {error['id']:<26} "
                f"expected {error['expected']:<11} got {error['got']}"
            )

        out.append("")

    out.append(BAR)

    return "\n".join(out)
