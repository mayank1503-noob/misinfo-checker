"""
The evaluation harness (backend/evaluation/).

Two kinds of test here. The metrics ones check arithmetic against
handwritten records, with no pipeline anywhere near them — a harness
whose scoring is wrong is worse than no harness, because it is believed.
The runner ones drive `run_case` with a stubbed `analyze`, so the whole
file stays in milliseconds and needs no model.

The dataset tests assert the *composition* of the shipped set, not just
that it parses. A set that drifts into being all rumours would still load
fine and would still report a lovely accuracy while measuring nothing.
"""

import json

import pytest

from backend.evaluation import dataset, metrics, report, runner


# --- the dataset ------------------------------------------------------


def test_the_shipped_set_loads_and_validates():
    cases = dataset.load()

    assert len(cases) >= 40
    assert len({case.id for case in cases}) == len(cases)


def test_the_set_contains_both_directions():
    """
    A set of nothing but rumours would score 100% for a system that
    always shouts "false". At least a third of it has to be messages
    that must not be accused.
    """
    cases = dataset.load()
    innocent = [case for case in cases if case.must_not_accuse]

    assert len(innocent) / len(cases) >= 0.3


def test_every_language_is_represented():
    cases = dataset.load()
    languages = {case.language for case in cases}

    assert languages == set(dataset.LANGUAGES)


def test_the_same_rumour_appears_in_all_three_languages():
    """The per-language column is only a comparison if the claims match."""
    cases = dataset.load()

    by_seed = {}

    for case in cases:
        if case.expected_seed_id:
            by_seed.setdefault(case.expected_seed_id, set()).add(case.language)

    shared = [seed for seed, langs in by_seed.items() if len(langs) == 3]

    assert len(shared) >= 3, f"only {shared} appear in all three languages"


def test_out_of_corpus_cases_expect_silence():
    """
    Nothing in a 32-entry demo index can settle these, so any expectation
    other than `unverified` would be asking the system to guess.
    """
    for case in dataset.load():
        if not case.in_corpus:
            assert case.expected_verdict == "unverified", case.id


def test_benign_near_misses_are_present():
    """The cases a loosened retrieval threshold breaks first."""
    ids = {case.id for case in dataset.load()}

    assert {"benign_boiling_water", "benign_upi_safe"} <= ids


def test_the_known_hinglish_regression_is_pinned():
    """DECISIONS.md O1 says this sentence reached 0.459. It has a case."""
    ids = {case.id for case in dataset.load()}

    assert "chat_salt" in ids


def test_chat_is_never_expected_to_be_accused():
    for case in dataset.load():
        if case.category == "chat":
            assert case.expected_verdict not in dataset.ACCUSATIONS
            assert not case.expects_check_worthy


@pytest.mark.parametrize("broken,message", [
    ({"id": "x", "text": "t", "language": "en", "category": "chat"}, "expect"),
    ({"id": "x", "text": "", "language": "en", "category": "chat",
      "expect": {"verdict": "false", "check_worthy": True}}, "empty text"),
    ({"id": "x", "text": "t", "language": "klingon", "category": "chat",
      "expect": {"verdict": "false", "check_worthy": True}}, "language"),
    ({"id": "x", "text": "t", "language": "en", "category": "chat",
      "expect": {"verdict": "probably", "check_worthy": True}}, "verdict"),
    ({"id": "x", "text": "t", "language": "en", "category": "chat",
      "expect": {"verdict": "false", "check_worthy": "yes"}}, "check_worthy"),
])
def test_validate_rejects_a_malformed_case(broken, message):
    with pytest.raises(ValueError, match=message):
        dataset.validate(broken)


def test_a_case_cannot_name_a_seed_entry_and_deny_being_in_the_corpus():
    """The contradiction that would silently corrupt the report's split."""
    with pytest.raises(ValueError, match="in_corpus"):
        dataset.validate({
            "id": "x", "text": "t", "language": "en", "category": "health",
            "expect": {"verdict": "false", "check_worthy": True,
                       "retrieves": "seed_covid_hot_water"},
            "in_corpus": False,
        })


def test_a_malformed_line_is_not_skipped(tmp_path):
    """
    The seed corpus skips bad lines; an eval set must not. Dropping a
    case moves every number in the report without saying so.
    """
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a"} not json\n', encoding="utf-8")

    with pytest.raises(ValueError):
        dataset.load(str(path))


def test_duplicate_ids_are_rejected(tmp_path):
    row = {
        "id": "same", "text": "t", "language": "en", "category": "chat",
        "expect": {"verdict": "unverified", "check_worthy": False},
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate"):
        dataset.load(str(path))


def test_select_filters_and_combines():
    cases = dataset.load()

    hinglish = dataset.select(cases, language="hinglish")
    chat = dataset.select(cases, category="chat")
    both = dataset.select(cases, language="hinglish", category="chat")

    assert hinglish and chat and both
    assert len(both) <= min(len(hinglish), len(chat))
    assert all(c.language == "hinglish" and c.category == "chat" for c in both)


# --- the metrics ------------------------------------------------------


def record(id, expected, got, **kwargs):
    base = {
        "id": id,
        "language": kwargs.get("language", "en"),
        "category": kwargs.get("category", "health"),
        "in_corpus": kwargs.get("in_corpus", True),
        "expected_verdict": expected,
        "verdict": got,
        "expected_check_worthy": kwargs.get("expected_check_worthy", True),
        "check_worthy": kwargs.get("check_worthy", True),
        "expected_seed_id": kwargs.get("expected_seed_id"),
        "retrieved_seed_ids": kwargs.get("retrieved_seed_ids"),
        "stage_errors": {},
    }

    return base


def test_verdict_accuracy_and_confusion():
    records = [
        record("a", "false", "false"),
        record("b", "false", "unverified"),
        record("c", "unverified", "unverified"),
        record("d", "true", "true"),
    ]

    scores = metrics.verdict_scores(records)

    assert scores["n"] == 4
    assert scores["accuracy"] == 0.75
    assert scores["confusion"]["false->unverified"] == 1


def test_calling_a_true_message_false_is_a_wrong_accusation():
    records = [
        record("innocent", "true", "false"),
        record("chat", "unverified", "misleading"),
        record("rumour", "false", "false"),
    ]

    scores = metrics.verdict_scores(records)

    assert scores["false_accusations"] == 2
    assert set(scores["false_accusation_ids"]) == {"innocent", "chat"}


def test_missing_a_rumour_is_counted_apart_from_accusing_wrongly():
    records = [
        record("missed", "false", "unverified"),
        record("wrong", "unverified", "false"),
    ]

    scores = metrics.verdict_scores(records)

    assert scores["missed_rumours"] == 1
    assert scores["missed_rumour_ids"] == ["missed"]
    assert scores["false_accusations"] == 1


def test_a_mode_that_did_not_judge_reports_nothing_rather_than_zero():
    """
    The trap this guards: with stage 4 off every claim is `unverified`,
    and scoring that would invent an accuracy out of a stage that never
    ran.
    """
    records = [record("a", "false", None), record("b", "unverified", None)]

    assert metrics.verdict_scores(records) == {"n": 0}


def test_check_worthiness_precision_and_recall():
    records = [
        record("r1", "false", "false", expected_check_worthy=True, check_worthy=True),
        record("r2", "false", "false", expected_check_worthy=True, check_worthy=False),
        record("c1", "unverified", "unverified",
               expected_check_worthy=False, check_worthy=True),
        record("c2", "unverified", "unverified",
               expected_check_worthy=False, check_worthy=False),
    ]

    scores = metrics.check_worthiness_scores(records)

    assert scores["tp"] == 1 and scores["fp"] == 1 and scores["fn"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["false_positive_ids"] == ["c1"]


def test_retrieval_hit_rate_and_strays():
    records = [
        record("hit", "false", "false",
               expected_seed_id="seed_a", retrieved_seed_ids=["seed_a"]),
        record("miss", "false", "false",
               expected_seed_id="seed_b", retrieved_seed_ids=["seed_z"]),
        record("quiet", "unverified", "unverified",
               expected_seed_id=None, retrieved_seed_ids=[]),
        record("stray", "unverified", "unverified",
               expected_seed_id=None, retrieved_seed_ids=["seed_salt"]),
    ]

    scores = metrics.retrieval_scores(records)

    assert scores["hit_rate"] == 0.5
    assert scores["miss_ids"] == ["miss"]
    assert scores["stray_rate"] == 0.5
    assert scores["stray_ids"] == ["stray"]


def test_retrieval_is_not_scored_in_a_mode_that_never_retrieved():
    """
    claims mode leaves `retrieved_seed_ids` as None. Counting those as
    misses would print a 0% hit rate for a stage that was switched off.
    """
    records = [
        record("a", "false", None,
               expected_seed_id="seed_a", retrieved_seed_ids=None),
        record("b", "unverified", None,
               expected_seed_id=None, retrieved_seed_ids=None),
    ]

    scores = metrics.retrieval_scores(records)

    assert scores["n_targeted"] == 0
    assert scores["hit_rate"] is None
    assert scores["n_untargeted"] == 0


def test_the_report_splits_in_corpus_from_out_of_corpus():
    records = [
        record("known", "false", "false", in_corpus=True),
        record("unknown", "unverified", "false", in_corpus=False),
    ]

    summary = metrics.summarize(records)

    assert summary["in_corpus"]["verdict"]["accuracy"] == 1.0
    assert summary["out_of_corpus"]["verdict"]["accuracy"] == 0.0
    assert summary["overall"]["verdict"]["accuracy"] == 0.5


def test_the_report_splits_by_language():
    records = [
        record("a", "false", "false", language="hinglish"),
        record("b", "false", "unverified", language="hinglish"),
        record("c", "false", "false", language="en"),
    ]

    summary = metrics.summarize(records)

    assert summary["by_language"]["hinglish"]["verdict"]["accuracy"] == 0.5
    assert summary["by_language"]["en"]["verdict"]["accuracy"] == 1.0


def test_the_gate_fails_only_on_a_wrong_accusation():
    missed = metrics.summarize([record("m", "false", "unverified")])
    accused = metrics.summarize([record("a", "true", "false")])

    assert metrics.gate(missed)[0] is True          # poor recall still ships
    ok, reasons = metrics.gate(accused)

    assert ok is False
    assert "a" in reasons[0]


# --- the runner -------------------------------------------------------


def fake_analyze(verdict="false", check_worthy=1, seed_ids=("seed_a",)):
    """A stand-in for backend.pipeline.analyze that loads no models."""
    def analyze(packet, **kwargs):
        return {
            "verdict": {"label": verdict, "confidence": 0.7},
            "stages": {"claims": {"claims": 2, "check_worthy": check_worthy,
                                  "backend": "regex", "error": None}},
            "graph": {
                "nodes": [
                    {"kind": "evidence", "source_type": "seed_index",
                     "meta": {"seed_id": seed_id}}
                    for seed_id in seed_ids
                ] + [{"kind": "claim"}],
            },
        }

    return analyze


def a_case(**kwargs):
    row = {
        "id": "t1", "text": "some text", "language": "en", "category": "health",
        "expect": {"verdict": "false", "check_worthy": True,
                   "retrieves": "seed_a"},
        "in_corpus": True,
    }
    row.update(kwargs)

    return dataset.Case(row)


def test_run_case_records_the_verdict_in_full_mode():
    got = runner.run_case(a_case(), mode="full", analyze=fake_analyze())

    assert got["verdict"] == "false"
    assert got["check_worthy"] is True
    assert got["retrieved_seed_ids"] == ["seed_a"]


def test_run_case_withholds_the_verdict_outside_full_mode():
    got = runner.run_case(a_case(), mode="retrieval", analyze=fake_analyze())

    assert got["verdict"] is None
    assert got["retrieved_seed_ids"] == ["seed_a"]


def test_claims_mode_reports_no_retrieval_at_all():
    got = runner.run_case(a_case(), mode="claims", analyze=fake_analyze())

    assert got["retrieved_seed_ids"] is None
    assert got["check_worthy"] is True


def test_run_case_passes_a_real_date_to_the_pipeline():
    """
    Stage 2 does date arithmetic against `today`, so a string raises.
    The stubbed analyze in these tests ignores its kwargs, which is
    exactly how that bug survived until a real run hit it.
    """
    from datetime import date

    seen = {}

    def analyze(packet, **kwargs):
        seen.update(kwargs)

        return {"verdict": {"label": "false"},
                "stages": {"claims": {"check_worthy": 1}},
                "graph": {"nodes": []}}

    runner.run_case(a_case(), mode="full", analyze=analyze)

    assert isinstance(seen["today"], date)


def test_run_case_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown mode"):
        runner.run_case(a_case(), mode="turbo", analyze=fake_analyze())


def test_only_seed_index_evidence_counts_as_a_retrieval_hit():
    """A web result is not proof the local index found anything."""
    def analyze(packet, **kwargs):
        return {
            "verdict": {"label": "false"},
            "stages": {"claims": {"check_worthy": 1}},
            "graph": {"nodes": [
                {"kind": "evidence", "source_type": "web",
                 "meta": {"seed_id": "seed_a"}},
            ]},
        }

    got = runner.run_case(a_case(), mode="retrieval", analyze=analyze)

    assert got["retrieved_seed_ids"] == []


def test_a_case_that_raises_becomes_a_record_rather_than_ending_the_run():
    def explode(packet, **kwargs):
        raise RuntimeError("model gone")

    records = runner.run([a_case(), a_case(id="t2")], mode="full", analyze=explode)

    assert len(records) == 2
    assert records[0]["verdict"] is None
    assert "model gone" in records[0]["stage_errors"]["runner"]


def test_a_stage_that_failed_soft_is_recorded():
    def analyze(packet, **kwargs):
        return {
            "verdict": {"label": "unverified"},
            "stages": {
                "claims": {"check_worthy": 0},
                "evidence": {"error": "ConnectionError: no route to host"},
            },
            "graph": {"nodes": []},
        }

    got = runner.run_case(a_case(), mode="retrieval", analyze=analyze)

    assert "evidence" in got["stage_errors"]


def test_run_reports_progress_per_case():
    seen = []

    runner.run(
        [a_case(), a_case(id="t2")], mode="full", analyze=fake_analyze(),
        progress=lambda i, total, rec: seen.append((i, total, rec["id"])),
    )

    assert seen == [(1, 2, "t1"), (2, 2, "t2")]


# --- the report -------------------------------------------------------


def test_the_report_renders_in_every_mode():
    records = [
        record("a", "false", "false", expected_seed_id="s", retrieved_seed_ids=["s"]),
        record("b", "unverified", None, language="hinglish"),
    ]

    text = report.render(metrics.summarize(records), records, mode="retrieval")

    assert "STAGE 3b" in text
    assert "BY LANGUAGE" in text


def test_the_report_says_so_rather_than_printing_a_made_up_accuracy():
    records = [record("a", "false", None)]

    text = report.render(metrics.summarize(records), records, mode="retrieval")

    assert "--mode full" in text
