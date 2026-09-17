"""
Turning per-case results into the numbers that mean something.

Plain accuracy is close to useless for this system, for two reasons.

**The errors are not symmetric.** Missing a rumour leaves the user where
they started. Telling someone their true message is false is the system
doing harm on its own initiative, and it is also the failure that
destroys trust in the tool. So `false_accusations` is reported on its own
line and is the number to read first. It should be zero, and a release
that raises it has regressed no matter what happened to accuracy.

**Most of the corpus is the answer key.** 24 of the 41 cases have their
answer sitting in `data/seed_factchecks.jsonl`. Scoring those measures
retrieval and stance — real things, worth tracking — but it says nothing
about claims the index has never seen. The out-of-corpus cases measure
the opposite and much rarer skill: knowing when to say nothing. A single
blended accuracy would let a gain in one hide a loss in the other, so
every report splits by `in_corpus`.

Everything here is pure: it takes the result records the runner produced
and returns dicts. Nothing in this module runs the pipeline, which is
what lets the tests check the arithmetic against handwritten records.
"""

from collections import Counter, OrderedDict

from .dataset import ACCUSATIONS, VERDICTS


def _rate(part, whole):
    return round(part / whole, 3) if whole else None


def _prf(true_positive, false_positive, false_negative):
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)

    if not precision or not recall:
        f1 = 0.0 if (precision is not None and recall is not None) else None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 3)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def verdict_scores(records):
    """
    Label accuracy, the confusion matrix, and the asymmetric error counts.

    `false_accusations` and `missed_rumours` are the two that matter:
    the first is harm done, the second is harm not prevented.
    """
    scored = [r for r in records if r.get("verdict") is not None]

    if not scored:
        return {"n": 0}

    correct = [r for r in scored if r["verdict"] == r["expected_verdict"]]

    false_accusations = [
        r for r in scored
        if r["expected_verdict"] not in ACCUSATIONS and r["verdict"] in ACCUSATIONS
    ]
    missed = [
        r for r in scored
        if r["expected_verdict"] in ACCUSATIONS and r["verdict"] not in ACCUSATIONS
    ]
    abstained = [
        r for r in scored
        if r["verdict"] == "unverified" and r["expected_verdict"] != "unverified"
    ]

    confusion = Counter(
        (r["expected_verdict"], r["verdict"]) for r in scored
    )

    return {
        "n": len(scored),
        "accuracy": _rate(len(correct), len(scored)),
        "false_accusations": len(false_accusations),
        "false_accusation_ids": [r["id"] for r in false_accusations],
        "missed_rumours": len(missed),
        "missed_rumour_ids": [r["id"] for r in missed],
        "abstentions": len(abstained),
        "abstention_ids": [r["id"] for r in abstained],
        "confusion": {
            f"{expected}->{got}": count
            for (expected, got), count in sorted(confusion.items())
        },
    }


def check_worthiness_scores(records):
    """
    Stage 2 as a binary classifier: did this message contain a claim?

    Recall is what protects the user (a rumour nobody looked at cannot be
    caught); precision is what protects the budget, since every
    false-positive message costs a retrieval round. DECISIONS.md O2 is
    exactly a precision complaint, so this is the number that would show
    it being fixed.
    """
    scored = [r for r in records if r.get("check_worthy") is not None]

    if not scored:
        return {"n": 0}

    tp = sum(1 for r in scored if r["check_worthy"] and r["expected_check_worthy"])
    fp = sum(1 for r in scored if r["check_worthy"] and not r["expected_check_worthy"])
    fn = sum(1 for r in scored if not r["check_worthy"] and r["expected_check_worthy"])
    tn = sum(
        1 for r in scored if not r["check_worthy"] and not r["expected_check_worthy"]
    )

    scores = _prf(tp, fp, fn)
    scores.update({
        "n": len(scored),
        "tn": tn,
        "accuracy": _rate(tp + tn, len(scored)),
        "false_positive_ids": [
            r["id"] for r in scored
            if r["check_worthy"] and not r["expected_check_worthy"]
        ],
    })

    return scores


def retrieval_scores(records):
    """
    Stage 3b: did the expected seed entry come back, and what else did.

    Only cases where retrieval actually ran are counted at all. `hit_rate`
    is over those that name a target; `stray_rate` is over those that name
    none — a case with no right answer that still
    attracted seed evidence is the retriever reaching, which is how a
    lowered floor or an over-eager query would show up here first.
    """
    # `retrieved_seed_ids is None` means retrieval never ran (claims
    # mode). Scoring those as misses would report a 0% hit rate for a
    # stage that was switched off - the same trap the verdict avoids.
    targeted = [
        r for r in records
        if r.get("expected_seed_id") and r.get("retrieved_seed_ids") is not None
    ]
    untargeted = [
        r for r in records
        if not r.get("expected_seed_id") and r.get("retrieved_seed_ids") is not None
    ]

    hits = [r for r in targeted if r["expected_seed_id"] in (r.get("retrieved_seed_ids") or [])]
    strays = [r for r in untargeted if r.get("retrieved_seed_ids")]

    return {
        "n_targeted": len(targeted),
        "hit_rate": _rate(len(hits), len(targeted)),
        "hits": len(hits),
        "miss_ids": [r["id"] for r in targeted if r not in hits],
        "n_untargeted": len(untargeted),
        "stray_rate": _rate(len(strays), len(untargeted)),
        "stray_ids": [r["id"] for r in strays],
    }


def by(records, key):
    """Split `records` into an ordered {value: [records]} on one field."""
    groups = OrderedDict()

    for record in records:
        groups.setdefault(record.get(key), []).append(record)

    return groups


def summarize(records):
    """
    The whole picture: overall, split by corpus provenance, and by language.

    The language split is not decoration. Romanised Hinglish is the one
    the retrieval work moved and the one the false-positive risk landed
    in (DECISIONS.md O1), so it needs to be visible on every run rather
    than averaged into a single figure.
    """
    def block(subset):
        scores = {
            "n": len(subset),
            "verdict": verdict_scores(subset),
            "check_worthy": check_worthiness_scores(subset),
            "retrieval": retrieval_scores(subset),
        }

        agents = agent_scores(subset)

        #  Absent, not empty, on a run the agents did not drive, so an
        #  existing report sees exactly what it saw before.
        if agents.get("n"):
            scores["agents"] = agents

        return scores

    in_corpus = [r for r in records if r.get("in_corpus")]
    out_of_corpus = [r for r in records if not r.get("in_corpus")]

    return {
        "overall": block(records),
        "in_corpus": block(in_corpus),
        "out_of_corpus": block(out_of_corpus),
        "by_language": {
            language: block(subset)
            for language, subset in by(records, "language").items()
        },
        "by_category": {
            category: block(subset)
            for category, subset in by(records, "category").items()
        },
        "errors": [
            {
                "id": r["id"],
                "language": r.get("language"),
                "category": r.get("category"),
                "expected": r.get("expected_verdict"),
                "got": r.get("verdict"),
                "in_corpus": r.get("in_corpus"),
                "retrieved": r.get("retrieved_seed_ids"),
            }
            for r in records
            if r.get("verdict") is not None
            and r["verdict"] != r["expected_verdict"]
        ],
    }


def agent_scores(records):
    """
    What the agent layer did, and what it cost.

    Not an accuracy figure: the labels come from the same rules whether
    the agents drove the stages or the linear pipeline did, so scoring
    them again here would just double-count `verdict_scores`. What is
    worth measuring is the coordination itself:

      * **routes** - which agents the planner actually reached for. A
        planner that always produces the same route is not deciding
        anything.
      * **abstentions, per agent** - the outcome the layer exists to make
        possible. An evidence agent that never abstains is not being
        honest about a 32-entry offline corpus.
      * **failures** - should be zero. An agent that raised is the same
        class of problem as a crashed stage, and `gate` treats it that way.
      * **escalation, and what it yielded** - the second retrieval round
        is the layer's one real expense, so the share of cases that
        escalated and the share of those that then reached a label is the
        number that says whether it pays for itself.
      * **cost** - tool calls and milliseconds per case.
    """
    scored = [r for r in records if r.get("agents")]

    if not scored:
        return {"n": 0}

    agents = [r["agents"] for r in scored]

    routes = Counter(" -> ".join(a["route"]) for a in agents)
    planners = Counter(a["planner"] for a in agents)

    abstained = Counter(name for a in agents for name in a["abstained"])
    failed = Counter(name for a in agents for name in a["failed"])

    escalated = [r for r in scored if r["agents"]["escalated"]]
    settled = [r for r in escalated if r.get("verdict") not in (None, "unverified")]

    calls = [a["tool_calls"] for a in agents if a["tool_calls"] is not None]
    times = [r["ms"] for r in scored if r.get("ms") is not None]

    return {
        "n": len(scored),
        "planners": dict(planners),
        "routes": dict(routes.most_common()),
        "abstentions": dict(abstained),
        "failures": dict(failed),
        "failed_ids": [r["id"] for r in scored if r["agents"]["failed"]],
        "escalated": len(escalated),
        "escalation_rate": _rate(len(escalated), len(scored)),
        "escalation_settled": len(settled),
        "escalation_yield": _rate(len(settled), len(escalated)),
        "retrievers": dict(Counter(name for a in agents for name in a["retrievers"])),
        "tool_calls_mean": round(sum(calls) / len(calls), 1) if calls else None,
        "ms_mean": round(sum(times) / len(times), 1) if times else None,
        "explained": sum(1 for a in agents if (a.get("explanation") or "").strip()),
    }


def compare(linear, agentic):
    """
    The linear pipeline against the agent layer, case by case.

    Only cases both runs actually judged are compared. A mode where
    neither produced a verdict compares nothing and says so, rather than
    reporting perfect agreement - which would be true and useless.

    Every disagreement is classified, because "the label moved" is not a
    result: moving *towards* the expected label is an improvement and away
    from it is a regression, and a change that trades one for the other at
    parity has earned nothing. `safety` is reported on its own for the
    same reason it is everywhere else in this package - a wrong accusation
    is not one error among many.
    """
    left = {r["id"]: r for r in linear}
    right = {r["id"]: r for r in agentic}
    shared = [case_id for case_id in left if case_id in right]

    judged = [
        case_id for case_id in shared
        if left[case_id].get("verdict") is not None
        and right[case_id].get("verdict") is not None
    ]

    agreed = [
        case_id for case_id in judged
        if left[case_id]["verdict"] == right[case_id]["verdict"]
    ]

    improvements = []
    regressions = []
    neither = []

    for case_id in judged:
        if case_id in agreed:
            continue

        expected = left[case_id].get("expected_verdict")
        entry = {
            "id": case_id,
            "expected": expected,
            "linear": left[case_id]["verdict"],
            "agentic": right[case_id]["verdict"],
        }

        if right[case_id]["verdict"] == expected:
            improvements.append(entry)
        elif left[case_id]["verdict"] == expected:
            regressions.append(entry)
        else:
            neither.append(entry)

    def mean(records, field):
        values = [r[field] for r in records if r.get(field) is not None]

        return round(sum(values) / len(values), 1) if values else None

    linear_summary = summarize([left[case_id] for case_id in shared])
    agentic_summary = summarize([right[case_id] for case_id in shared])

    return {
        "n": len(shared),
        "judged": len(judged),
        "agreed": len(agreed),
        "agreement": _rate(len(agreed), len(judged)),
        "improvements": improvements,
        "regressions": regressions,
        "neither": neither,
        "safety": {
            "linear": linear_summary["overall"]["verdict"].get("false_accusations"),
            "agentic": agentic_summary["overall"]["verdict"].get("false_accusations"),
        },
        "accuracy": {
            "linear": linear_summary["overall"]["verdict"].get("accuracy"),
            "agentic": agentic_summary["overall"]["verdict"].get("accuracy"),
        },
        "ms": {
            "linear": mean([left[c] for c in shared], "ms"),
            "agentic": mean([right[c] for c in shared], "ms"),
        },
        "agents": agentic_summary["overall"].get("agents"),
    }


def gate(summary, max_false_accusations=0):
    """
    (ok, reasons) — the pass/fail a CI step or a pre-demo check can use.

    Deliberately narrow. It fails on wrong accusations, because that is
    the error the system must not make, and on a stage crash, because a
    green run over a pipeline that silently failed is worse than a red
    one. It does *not* fail on accuracy: the right threshold for a
    32-entry demo index is not knowable, and a gate nobody trusts gets
    switched off.
    """
    reasons = []

    accusations = summary["overall"]["verdict"].get("false_accusations") or 0

    if accusations > max_false_accusations:
        ids = summary["overall"]["verdict"]["false_accusation_ids"]
        reasons.append(
            f"{accusations} wrong accusation(s), limit {max_false_accusations}: "
            f"{', '.join(ids)}"
        )

    # An agent that raised is the same class of problem as a crashed
    # stage: the run can look green while a specialist never ran. Only
    # looked at when the agents drove the run at all.
    agents = summary["overall"].get("agents") or {}

    if agents.get("failures"):
        failures = ", ".join(
            f"{name} x{count}" for name, count in sorted(agents["failures"].items())
        )
        reasons.append(
            f"agent failure(s): {failures} "
            f"(cases: {', '.join(agents.get('failed_ids') or [])})"
        )

    return (not reasons), reasons


__all__ = [
    "VERDICTS",
    "agent_scores",
    "compare",
    "verdict_scores",
    "check_worthiness_scores",
    "retrieval_scores",
    "summarize",
    "gate",
    "by",
]
