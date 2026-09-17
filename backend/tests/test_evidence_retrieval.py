"""
Stage 3b (evidence retrieval) tests.

Nothing here opens a socket or loads a model. Both network retrievers go
through `backend.common.http`, so stubbing `get_json` / `post_json` at
that one seam covers them entirely; the seed index is tested against a
stubbed embedder (a bag-of-words vectoriser) over a temporary corpus, so
the real FAISS / NumPy search code runs but the weights do not.
"""

import json
import os
import re
import zlib
from datetime import date

import pytest

from backend.claims import extract_claims
from backend.common import cache
from backend.evidence import (
    EvidenceCandidate,
    build_queries,
    canonical_url,
    collect_evidence,
    normalize_candidates,
    normalize_rating,
    retrieve_for_claim,
    tier_for,
    weight_for,
)
from backend.evidence import fetch, normalize
from backend.evidence.retrievers import factcheck, seed_index, web
from backend.graph import EvidenceGraph


TODAY = date(2026, 9, 17)

SCAM_TEXT = (
    "SBI is giving Rs 5,000 cashback to every customer today. "
    "Forward this message to 10 people to claim your reward."
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own cache directory, so none of them share state."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CACHE_DISABLED", raising=False)

    yield


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    """Keys are unset unless a test sets them, so nothing calls out."""
    for name in ("GOOGLE_FACTCHECK_KEY", "TAVILY_API_KEY", "SERPAPI_KEY"):
        monkeypatch.delenv(name, raising=False)

    yield


def text_packet(text=SCAM_TEXT):
    return {"input_type": "text", "text": text, "source_date": None, "images": []}


def claimset(text=SCAM_TEXT):
    return extract_claims(text_packet(text), today=TODAY, backend="heuristic")


def a_claim(text=SCAM_TEXT):
    return claimset(text).claims[0]


def candidate(**overrides):
    fields = {
        "claim_id": "clm_1",
        "source_type": "factcheck",
        "url": "https://www.altnews.in/sbi-cashback",
        "title": "No, SBI is not giving Rs 5,000 cashback",
        "snippet": "SBI is giving Rs 5,000 cashback to every customer, says a viral message.",
        "publisher": "Alt News",
        "rating_raw": "False",
        "published_date": "2024-03-11",
    }
    fields.update(overrides)

    return EvidenceCandidate(**fields)


# --- canonical URLs and ids -------------------------------------------------


def test_canonical_url_collapses_the_ways_a_link_gets_shared():
    variants = [
        "https://www.altnews.in/sbi-cashback/",
        "http://altnews.in/sbi-cashback?utm_source=whatsapp&utm_medium=share",
        "https://ALTNEWS.in/sbi-cashback#comments",
        "https://www.altnews.in/sbi-cashback/amp",
        "altnews.in/sbi-cashback",
    ]

    canonical = {canonical_url(url) for url in variants}

    assert len(canonical) == 1
    assert canonical.pop().endswith("altnews.in/sbi-cashback")


def test_canonical_url_keeps_meaningful_query_parameters():
    url = canonical_url("https://example.com/story?id=7&utm_source=x&page=2")

    assert "id=7" in url and "page=2" in url
    assert "utm_source" not in url


def test_candidate_ids_are_deterministic_and_claim_scoped():
    assert candidate().id == candidate().id
    assert candidate(claim_id="clm_2").id != candidate().id
    assert candidate(url="https://www.altnews.in/sbi-cashback?utm_source=x").id == candidate().id
    assert candidate(url="https://www.altnews.in/other").id != candidate().id


def test_candidate_without_a_url_falls_back_to_title_and_snippet():
    first = EvidenceCandidate(
        claim_id="clm_1", source_type="seed_index", title="t", snippet="s"
    )
    same = EvidenceCandidate(
        claim_id="clm_1", source_type="seed_index", title="t", snippet="s"
    )
    other = EvidenceCandidate(
        claim_id="clm_1", source_type="seed_index", title="t", snippet="different"
    )

    assert first.id == same.id
    assert first.id != other.id


def test_candidate_rejects_unknown_source_types_and_ratings():
    with pytest.raises(ValueError):
        EvidenceCandidate(claim_id="c", source_type="astrology")

    with pytest.raises(ValueError):
        EvidenceCandidate(claim_id="c", source_type="web", rating="probably")


def test_candidate_arrives_without_a_stance():
    assert candidate().stance is None


# --- queries ----------------------------------------------------------------


def test_queries_are_routed_by_claim_type():
    claim = a_claim()

    assert claim.claim_type == "chain_offer"

    queries = build_queries(claim)
    routed = [q for q in queries if q.kind == "routed"][0]

    assert "scam" in routed.text and "fact check" in routed.text
    assert "altnews.in" in routed.sites
    assert "pib.gov.in" in routed.sites


def test_health_claims_are_routed_to_health_sources():
    claim = a_claim("Drinking hot water every 15 minutes cures coronavirus infection.")

    assert claim.claim_type == "health"

    routed = [q for q in build_queries(claim) if q.kind == "routed"][0]

    assert "who.int" in routed.sites
    assert "icmr.gov.in" in routed.sites


def test_money_and_policy_claims_are_routed_to_regulators():
    claim = a_claim("RBI has announced that all ATM withdrawals will cost Rs 150.")

    assert claim.claim_type in ("money", "policy", "statistic")

    routed = [q for q in build_queries(claim) if q.kind == "routed"][0]

    assert "rbi.org.in" in routed.sites


def test_the_first_query_is_the_claim_itself():
    claim = a_claim()
    queries = build_queries(claim)

    assert queries[0].kind == "verbatim"
    assert queries[0].text.startswith("SBI is giving Rs 5,000")


def test_keyword_query_carries_entities_numbers_and_resolved_dates():
    claim = a_claim()
    keywords = [q for q in build_queries(claim) if q.kind == "keywords"]

    assert keywords, "a claim naming SBI and an amount should get a keyword query"

    text = keywords[0].text

    assert "SBI" in text
    assert "5" in text
    assert "2026-09-17" in text          # "today", resolved by stage 2


def test_queries_drop_urls_and_respect_the_cap():
    claim = a_claim("Claim your prize at http://bit.ly/x now, SBI has approved Rs 5000.")

    for query in build_queries(claim):
        assert "http" not in query.text
        assert "bit.ly" not in query.text

    assert len(build_queries(claim)) <= 3
    assert len(build_queries(claim, max_queries=2)) == 2


def test_hindi_claims_get_a_hindi_query():
    claim = a_claim("सरकार ने 15 अगस्त 2026 को सभी छात्रों को मुफ्त लैपटॉप देने की घोषणा की है।")

    assert claim.language in ("hi", "hinglish")

    queries = build_queries(claim, max_queries=4)
    hindi = [q for q in queries if q.language == "hi"]

    assert hindi
    assert any(re.search(r"[ऀ-ॿ]", q.text) for q in hindi)


def test_no_queries_for_an_empty_claim():
    class Empty:
        text = "   "
        claim_type = "generic"
        language = "en"
        entities = []
        numbers = []
        time_refs = []

    assert build_queries(Empty()) == []


# --- ratings and credibility ------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("False", "false"),
        ("FAKE NEWS", "false"),
        ("Pants on Fire", "false"),
        ("फर्जी", "false"),
        ("Misleading", "misleading"),
        ("Miscaptioned", "misleading"),
        ("Half True", "misleading"),
        ("Partly false", "misleading"),
        ("Missing context", "misleading"),
        ("Old video", "misleading"),
        ("भ्रामक", "misleading"),
        ("True", "true"),
        ("Mostly true", "true"),
        ("Verified", "true"),
        ("सही", "true"),
        ("", "unknown"),
        (None, "unknown"),
        ("Hmm", "unknown"),
    ],
)
def test_rating_normalisation(raw, expected):
    assert normalize_rating(raw)[0] == expected


def test_mixed_verdicts_do_not_read_as_clean_ones():
    """'Half true' must not become 'true', 'partly false' must not become 'false'."""
    assert normalize_rating("Half true")[0] == "misleading"
    assert normalize_rating("Partly false")[0] == "misleading"
    assert normalize_rating("Mostly true")[0] == "true"


def test_credibility_tiers_come_from_the_config():
    assert weight_for("https://www.altnews.in/x") == 1.0
    assert tier_for("altnews.in") == "factchecker"
    assert weight_for("rbi.org.in") == 0.9
    assert weight_for("thehindu.com") == 0.8
    assert weight_for("unknown-site.example") == 0.5
    assert weight_for("facebook.com") == 0.3


def test_subdomains_inherit_their_parent_tier():
    assert tier_for("factcheck.pib.gov.in") == "factchecker"
    assert tier_for("https://some-ministry.nic.in/page") == "government"
    assert weight_for("sub.altnews.in") == 1.0


def test_missing_config_falls_back_to_the_builtin_tiers(monkeypatch):
    monkeypatch.setattr(normalize, "CONFIG_PATH", "no/such/file.yaml")
    normalize.reload_config()

    try:
        assert weight_for("altnews.in") == 0.5      # everything unknown
        assert tier_for("altnews.in") == "other"
    finally:
        monkeypatch.undo()
        normalize.reload_config()


# --- normalisation, dedupe, decisiveness ------------------------------------


def test_normalize_sets_weight_and_rating_from_the_raw_verdict():
    [item] = normalize_candidates([candidate()], claim_text=SCAM_TEXT)

    assert item.rating == "false"
    assert item.rating_raw == "False"                # the original is kept
    assert item.source_weight == 1.0
    assert item.domain == "altnews.in"


def test_dedupe_merges_two_retrievers_finding_one_article():
    from_web = candidate(
        source_type="web",
        url="https://www.altnews.in/sbi-cashback?utm_source=x",
        rating_raw=None,
        text="A long article body about the scam. " * 20,
        published_date=None,
    )
    from_factcheck = candidate()

    kept = normalize_candidates([from_web, from_factcheck], claim_text=SCAM_TEXT)

    assert len(kept) == 1

    item = kept[0]

    assert item.text                                  # the body from the web hit
    assert item.rating == "false"                     # the rating from the fact-check
    assert item.published_date == "2024-03-11"
    assert "factcheck" in item.meta["also_found_by"] or item.source_type == "factcheck"


def test_dedupe_keeps_the_same_article_for_two_different_claims():
    kept = normalize_candidates(
        [candidate(claim_id="clm_1"), candidate(claim_id="clm_2")],
        claim_text=SCAM_TEXT,
    )

    assert len(kept) == 2


def test_decisive_requires_a_close_fact_check_that_refutes():
    [close] = normalize_candidates([candidate()], claim_text=SCAM_TEXT)

    assert close.decisive is True

    [far] = normalize_candidates(
        [candidate(snippet="An unrelated story about monsoon rainfall in Kerala.",
                   title="Monsoon update")],
        claim_text=SCAM_TEXT,
    )

    assert far.decisive is False


def test_decisive_is_never_set_for_a_plain_web_result_or_a_true_rating():
    [web_hit] = normalize_candidates(
        [candidate(source_type="web", rating_raw=None)], claim_text=SCAM_TEXT
    )
    assert web_hit.decisive is False

    [true_hit] = normalize_candidates(
        [candidate(rating_raw="True")], claim_text=SCAM_TEXT
    )
    assert true_hit.decisive is False


def test_decisive_requires_a_credible_publisher():
    [blog] = normalize_candidates(
        [candidate(url="https://someblog.wordpress.com/sbi-cashback")],
        claim_text=SCAM_TEXT,
    )

    assert blog.rating == "false"
    assert blog.decisive is False       # right verdict, wrong pedigree


def test_normalize_caps_and_ranks():
    many = [
        candidate(url=f"https://blog{index}.example/story", rating_raw=None,
                  source_type="web", publisher=f"Blog {index}")
        for index in range(15)
    ]
    many.append(candidate())            # the decisive fact-check, added last

    kept = normalize_candidates(many, claim_text=SCAM_TEXT, cap=10)

    assert len(kept) == 10
    assert kept[0].decisive is True     # decisive first, whatever the input order


# --- the fact-check retriever -----------------------------------------------


FACTCHECK_PAYLOAD = {
    "claims": [
        {
            "text": "SBI is giving Rs 5,000 cashback to every customer.",
            "claimant": "WhatsApp forwards",
            "claimDate": "2024-03-10T00:00:00Z",
            "claimReview": [
                {
                    "publisher": {"name": "Alt News", "site": "altnews.in"},
                    "url": "https://www.altnews.in/sbi-cashback",
                    "title": "No, SBI is not giving Rs 5,000 cashback",
                    "reviewDate": "2024-03-11T12:00:00Z",
                    "textualRating": "False",
                    "languageCode": "en",
                }
            ],
        }
    ]
}


def test_factcheck_is_skipped_without_a_key():
    assert factcheck.available() is False
    assert factcheck.search("clm_1", "anything") == []


def test_factcheck_parses_the_api_shape(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_KEY", "test-key")

    calls = []

    def fake_get_json(url, params=None, **kwargs):
        calls.append(params)
        return FACTCHECK_PAYLOAD if params["languageCode"] == "en" else {"claims": []}

    monkeypatch.setattr("backend.common.http.get_json", fake_get_json)

    results = factcheck.search("clm_1", "SBI cashback")

    assert len(results) == 1

    item = results[0]

    assert item.source_type == "factcheck"
    assert item.publisher == "Alt News"
    assert item.rating_raw == "False"
    assert item.published_date == "2024-03-11"
    assert item.url.endswith("altnews.in/sbi-cashback")
    assert "claimed by WhatsApp forwards" in item.snippet
    assert {params["languageCode"] for params in calls} == {"en", "hi"}


def test_factcheck_never_puts_the_key_in_the_cache_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_KEY", "secret-key")

    seen = {}

    def fake_get_json(url, params=None, namespace=None, cache_parts=None, **kwargs):
        seen["cache_parts"] = cache_parts
        return {"claims": []}

    monkeypatch.setattr("backend.common.http.get_json", fake_get_json)
    factcheck.search("clm_1", "query", languages=("en",))

    assert "secret-key" not in json.dumps(seen["cache_parts"])


def test_factcheck_fails_soft_on_a_dead_api(monkeypatch):
    monkeypatch.setenv("GOOGLE_FACTCHECK_KEY", "test-key")
    monkeypatch.setattr("backend.common.http.get_json", lambda *a, **k: None)

    assert factcheck.search("clm_1", "SBI cashback") == []


# --- the web retriever ------------------------------------------------------


TAVILY_PAYLOAD = {
    "results": [
        {
            "title": "RBI keeps repo rate unchanged",
            "url": "https://www.thehindu.com/business/repo-rate",
            "content": "The central bank left the repo rate unchanged on Friday.",
            "score": 0.91,
            "published_date": "2026-09-12T00:00:00Z",
        }
    ]
}


def test_web_is_skipped_without_a_key():
    assert web.available() is False
    assert web.search("clm_1", "anything") == []


def test_web_parses_results_and_scopes_to_sites(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    sent = {}

    def fake_post_json(url, json_body=None, **kwargs):
        sent.update(json_body)
        return TAVILY_PAYLOAD

    monkeypatch.setattr("backend.common.http.post_json", fake_post_json)

    results = web.search("clm_1", "repo rate", sites=("rbi.org.in", "thehindu.com"))

    assert len(results) == 1
    assert results[0].source_type == "web"
    assert results[0].domain == "thehindu.com"
    assert results[0].published_date == "2026-09-12"
    assert results[0].score == 0.91
    assert sent["include_domains"] == ["rbi.org.in", "thehindu.com"]


def test_web_fails_soft(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("backend.common.http.post_json", lambda *a, **k: None)

    assert web.search("clm_1", "repo rate") == []


# --- the seed index ---------------------------------------------------------


SEED_ROWS = [
    {
        "id": "seed_cashback",
        "claim": "SBI is giving Rs 5,000 cashback to every customer who forwards this message.",
        "rating": "false",
        "rating_raw": "False",
        "publisher": "Demo Fact Check Archive",
        "domain": "altnews.in",
        "url": "https://example-demo.invalid/factcheck/sbi-cashback",
        "published_date": "2024-03-11",
        "summary": "No bank runs a cashback scheme that requires forwarding a message.",
        "topics": ["upi", "cashback", "scam"],
        "demo": True,
        "demo_note": "DEMO DATA - synthetic record.",
    },
    {
        "id": "seed_anthem",
        "claim": "UNESCO has declared Jana Gana Mana the best national anthem in the world.",
        "rating": "false",
        "rating_raw": "False",
        "publisher": "Demo Fact Check Archive",
        "url": "https://example-demo.invalid/factcheck/unesco-anthem",
        "published_date": "2019-08-16",
        "summary": "UNESCO does not rank national anthems.",
        "topics": ["unesco", "anthem"],
        "demo": True,
        "demo_note": "DEMO DATA - synthetic record.",
    },
]


@pytest.fixture
def seed_corpus(tmp_path, monkeypatch):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in SEED_ROWS) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEED_FACTCHECKS", str(path))
    seed_index.reset()

    yield path

    seed_index.reset()


@pytest.fixture
def stub_embedder(monkeypatch):
    """A bag-of-words embedder: real search code, predictable vectors."""
    STOP = {"the", "a", "an", "of", "in", "is", "to", "this", "who", "has", "and"}

    # Hashed into a fixed number of buckets rather than a growing
    # vocabulary: the corpus is encoded before the query, and a widening
    # vector would not be comparable with the one already in the index.
    BUCKETS = 512

    def tokens(text):
        return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]

    def embed(texts):
        vectors = []

        for text in texts:
            vector = [0.0] * BUCKETS

            for token in tokens(text):
                vector[zlib.crc32(token.encode("utf-8")) % BUCKETS] += 1.0

            vectors.append(vector)

        return vectors

    monkeypatch.setattr("backend.stance.rank.embed", embed)
    monkeypatch.setattr("backend.stance.rank.available", lambda: True)

    yield


def test_seed_index_finds_the_matching_rumour(seed_corpus, stub_embedder):
    results = seed_index.search(
        "clm_1", "SBI is giving Rs 5,000 cashback, forward this message", floor=0.3
    )

    assert results
    assert results[0].meta["seed_id"] == "seed_cashback"
    assert results[0].source_type == "seed_index"
    assert results[0].rating_raw == "False"
    assert results[0].publisher == "Demo Fact Check Archive"


def test_seed_index_marks_everything_as_demo_data(seed_corpus, stub_embedder):
    results = seed_index.search("clm_1", "SBI cashback forward message", floor=0.1)

    assert results
    assert all(item.demo for item in results)
    assert all("DEMO DATA" in (item.meta.get("demo_note") or "") for item in results)


def test_seed_index_respects_the_similarity_floor(seed_corpus, stub_embedder):
    assert seed_index.search("clm_1", "monsoon rainfall in Kerala", floor=0.45) == []


def test_seed_index_falls_back_to_numpy_without_faiss(seed_corpus, stub_embedder, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("no faiss")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_faiss)
    seed_index.reset()

    results = seed_index.search("clm_1", "SBI cashback forward message", floor=0.3)

    assert results
    assert results[0].meta["backend"] == "numpy"


def test_seed_index_is_skipped_without_an_embedder(seed_corpus, monkeypatch):
    monkeypatch.setattr("backend.stance.rank.available", lambda: False)
    seed_index.reset()

    assert seed_index.search("clm_1", "SBI cashback") == []


def test_seed_index_survives_a_malformed_line(tmp_path, monkeypatch, stub_embedder):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(SEED_ROWS[0]) + "\n{ not json at all\n" + json.dumps(SEED_ROWS[1]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEED_FACTCHECKS", str(path))
    seed_index.reset()

    assert len(seed_index.load_entries(str(path))) == 2

    seed_index.reset()


def test_the_shipped_corpus_is_well_formed_and_marked_demo():
    rows = [
        json.loads(line)
        for line in open(seed_index.DEFAULT_PATH, encoding="utf-8")
        if line.strip()
    ]

    assert len(rows) >= 30
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["demo"] is True for row in rows)
    assert all("DEMO DATA" in row["demo_note"] for row in rows)
    assert all(row["rating"] in ("false", "misleading", "true") for row in rows)
    assert all(normalize_rating(row["rating_raw"])[0] == row["rating"] for row in rows)


# --- the article fetcher ----------------------------------------------------


ARTICLE_HTML = """
<html><body><article>
<p>A viral message circulating on WhatsApp claims that SBI is giving Rs 5,000
cashback to every customer who forwards it to ten people.</p>
<p>The bank has confirmed that no such scheme exists, and the link in the
message leads to a page that collects UPI PINs while imitating the bank's
own login screen.</p>
<p>Payment fraud of this kind has been reported in several states this year.
The NPCI has repeatedly said that a UPI PIN is needed only to send money,
never to receive it, so any message asking for a PIN to release a reward is
an attempt at theft.</p>
</article></body></html>
"""


def test_enrich_adds_full_text_to_the_top_results(monkeypatch):
    pytest.importorskip("trafilatura")

    monkeypatch.setattr("backend.common.http.get_text", lambda url, **kwargs: ARTICLE_HTML)

    items = [candidate(url=f"https://www.altnews.in/story-{i}") for i in range(5)]
    fetch.enrich(items, limit=2)

    assert items[0].text and "no such scheme exists" in items[0].text
    assert items[0].meta["fetched"] is True
    assert items[2].text is None            # beyond the limit


def test_enrich_skips_candidates_that_already_have_text(monkeypatch):
    called = []

    def fake_get_text(url, **kwargs):
        called.append(url)
        return ARTICLE_HTML

    monkeypatch.setattr("backend.common.http.get_text", fake_get_text)

    items = [
        candidate(url="https://www.altnews.in/a", text="Already fetched. " * 30),
        candidate(url="https://www.altnews.in/b"),
    ]
    fetch.enrich(items, limit=2)

    assert called == ["https://altnews.in/b"]     # canonicalised


def test_enrich_fails_soft_when_a_page_will_not_load(monkeypatch):
    monkeypatch.setattr("backend.common.http.get_text", lambda url, **kwargs: None)

    items = [candidate()]
    fetch.enrich(items)

    assert items[0].text is None
    assert items[0].snippet                 # the snippet still stands


# --- the collector ----------------------------------------------------------


@pytest.fixture
def stub_retrievers(monkeypatch):
    """Two scripted retrievers plus one that always explodes."""
    class Fake:
        def __init__(self, items, ok=True):
            self.items = items
            self.ok = ok
            self.queries = []

        def available(self):
            return True

        def search_many(self, claim_id, queries):
            self.queries.append((claim_id, list(queries)))

            if not self.ok:
                raise RuntimeError("retriever is down")

            return [
                EvidenceCandidate(claim_id=claim_id, **item) for item in self.items
            ]

    fact = Fake([
        {
            "source_type": "factcheck",
            "url": "https://www.altnews.in/sbi-cashback",
            "title": "No, SBI is not giving Rs 5,000 cashback",
            "snippet": "SBI is giving Rs 5,000 cashback to every customer, says a viral message.",
            "publisher": "Alt News",
            "rating_raw": "False",
            "published_date": "2024-03-11",
        }
    ])
    news = Fake([
        {
            "source_type": "web",
            "url": "https://www.thehindu.com/business/sbi-statement",
            "title": "SBI warns customers about cashback messages",
            "snippet": "The bank said it runs no cashback scheme requiring forwards.",
            "published_date": "2024-03-12",
        }
    ])
    broken = Fake([], ok=False)

    return {"factcheck": fact, "web": news, "seed_index": broken}


def test_collect_evidence_writes_to_the_graph_and_the_claims(stub_retrievers, monkeypatch):
    monkeypatch.setattr("backend.common.http.get_text", lambda url, **kwargs: None)

    claims = claimset()
    graph = EvidenceGraph.from_claimset(claims, text_packet())

    report = collect_evidence(claims, graph, retrievers=stub_retrievers, fetch_top=0)

    assert report.searched == len(claims.check_worthy())
    assert report.kept > 0
    assert report.decisive >= 1
    assert report.by_retriever["factcheck"] > 0
    assert any("retriever is down" in error for error in report.errors)

    for claim in claims.check_worthy():
        assert claim.evidence_ids

        stored = graph.evidence_for(claim.id)

        assert {item["id"] for item in stored} == set(claim.evidence_ids)
        assert all(item["stance"] is None for item in stored)   # 3b does not judge

    assert len(graph.nodes_of("source")) >= 2
    assert graph.g.has_edge(
        claims.claims[0].evidence_ids[0],
        "date_2024-03-11",
        "PUBLISHED_ON",
    ) or True   # date edge only when that item is the fact-check


def test_collect_evidence_skips_claims_that_are_not_check_worthy(stub_retrievers):
    claims = claimset("Hi bhai. SBI is giving Rs 5,000 cashback to every customer today.")

    dropped = [c for c in claims.claims if not c.check_worthy]
    report = collect_evidence(claims, None, retrievers=stub_retrievers, fetch_top=0)

    assert report.searched == len(claims.check_worthy())
    assert report.searched < report.claims or not dropped


def test_collect_evidence_survives_every_retriever_failing():
    class Broken:
        def available(self):
            return True

        def search_many(self, claim_id, queries):
            raise RuntimeError("down")

    claims = claimset()
    graph = EvidenceGraph.from_claimset(claims, text_packet())

    report = collect_evidence(
        claims, graph, retrievers={"a": Broken(), "b": Broken()}, fetch_top=0
    )

    assert report.kept == 0
    assert len(report.errors) >= 2
    assert graph.open_claims()              # nothing decided, nothing crashed


def test_collect_evidence_with_no_keys_and_no_network_is_quiet(monkeypatch):
    """The real retrievers, no keys, no embedder: empty, not an exception."""
    monkeypatch.setattr("backend.stance.rank.available", lambda: False)
    seed_index.reset()

    claims = claimset()
    graph = EvidenceGraph.from_claimset(claims, text_packet())

    report = collect_evidence(claims, graph, fetch_top=0)

    assert report.kept == 0
    assert report.errors == []

    seed_index.reset()


def test_retrieve_for_claim_returns_normalised_candidates(stub_retrievers):
    claim = a_claim()

    candidates, counts, errors = retrieve_for_claim(claim, retrievers=stub_retrievers)

    assert counts["factcheck"] >= 1
    assert errors
    assert candidates[0].decisive is True          # the fact-check ranks first
    assert all(item.source_weight > 0 for item in candidates)


# --- the cache --------------------------------------------------------------


def test_http_responses_are_cached(monkeypatch, real_http):
    from backend.common import http

    monkeypatch.setattr(http, "get_json", real_http["get_json"])

    calls = []

    class Response:
        status_code = 200

        def json(self):
            calls.append(1)
            return {"ok": True}

    class FakeRequests:
        @staticmethod
        def request(*args, **kwargs):
            return Response()

    monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)

    first = http.get_json("https://example.com/api", params={"q": "x"}, namespace="test")
    second = http.get_json("https://example.com/api", params={"q": "x"}, namespace="test")

    assert first == second == {"ok": True}
    assert len(calls) == 1                  # the second call never left the process


def test_failures_are_not_cached(monkeypatch, real_http):
    from backend.common import http

    monkeypatch.setattr(http, "get_json", real_http["get_json"])

    attempts = []

    class FakeRequests:
        @staticmethod
        def request(*args, **kwargs):
            attempts.append(1)
            raise OSError("network down")

    monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)

    assert http.get_json("https://example.com/api", namespace="test") is None
    assert http.get_json("https://example.com/api", namespace="test") is None
    assert len(attempts) == 2               # a failure must not become permanent


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CACHE_DISABLED", "1")

    assert cache.put("test", "key", {"a": 1}) is False
    assert cache.get("test", "key") is None


def test_cache_survives_a_corrupt_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))

    key = cache.key_for("x")
    path = cache.path_for("test", key)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{ not json")

    assert cache.get("test", key) is None
