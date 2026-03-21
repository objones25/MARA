"""Tests for mara.agent.nodes.corrective_retriever.

All LLM, Brave, Firecrawl, and DB calls are mocked — no real I/O.

Test strategy:
  - _generate_corrective_sub_queries: pure async helper — single failing claim,
    fenced LLM responses, parse-error fallback.
  - corrective_retriever node — DB-first path: no failing claims, DB sufficient,
    hash dedup, link_leaves_to_run called, leaf_db_enabled=False skips DB.
  - corrective_retriever node — scrape path: DB insufficient triggers Brave +
    Firecrawl, URL dedup, index continuity, upsert_leaves called,
    max_new_pages_per_round respected.
  - General: loop_count incremented, corrective_sub_queries accumulated,
    correct state keys returned.
"""

import json
import pytest

from mara.agent.nodes.corrective_retriever import (
    _generate_corrective_sub_queries,
    corrective_retriever,
)
from mara.agent.state import SubQuery
from mara.confidence.scorer import ScoredClaim
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored_claim(text: str, confidence: float = 0.30, n_leaves: int = 5, n_unique_urls: int = 0) -> ScoredClaim:
    """Build a ScoredClaim with sensible defaults for tests."""
    return ScoredClaim(
        text=text,
        source_indices=[0],
        confidence=confidence,
        corroborating=1,
        n_leaves=n_leaves,
        n_unique_urls=n_unique_urls,
    )


def _sub_queries_json(queries: list[tuple[str, str]]) -> str:
    return json.dumps([{"query": q, "domain": d} for q, d in queries])


def _mock_llm(mocker, content: str):
    """Patch make_llm in corrective_retriever to return a fake async LLM."""
    mock_msg = mocker.MagicMock()
    mock_msg.content = content
    mock_llm = mocker.AsyncMock()
    mock_llm.ainvoke = mocker.AsyncMock(return_value=mock_msg)
    mocker.patch(
        "mara.agent.nodes.corrective_retriever.make_llm",
        return_value=mock_llm,
    )
    return mock_llm


def _make_leaf_dict(url: str, text: str = "content", idx: int = 0) -> dict:
    """Build a MerkleLeaf-shaped dict for use as mock DB rows."""
    leaf_hash = hash_chunk(url, text, "2026-03-20T10:00:00Z", "sha256")
    return {
        "url": url,
        "text": text,
        "retrieved_at": "2026-03-20T10:00:00Z",
        "hash": leaf_hash,
        "index": idx,
        "sub_query": "test",
        "contextualized_text": text,
        "embedding": None,
        "embedding_model": None,
    }


def _mock_brave(mocker, urls: list[str]):
    """Patch brave_search to return SearchResults for the given URLs."""
    search_results = [
        {
            "url": u,
            "title": f"Title {i}",
            "description": "",
            "extra_snippets": [],
            "page_age": "",
            "result_type": "web",
        }
        for i, u in enumerate(urls)
    ]
    mocker.patch(
        "mara.agent.nodes.corrective_retriever.brave_search",
        return_value={"search_results": search_results},
    )


def _mock_firecrawl(mocker, chunks: list[dict]):
    """Patch firecrawl_scrape to return the given raw_chunks."""
    mocker.patch(
        "mara.agent.nodes.corrective_retriever.firecrawl_scrape",
        return_value={"raw_chunks": chunks},
    )


# ---------------------------------------------------------------------------
# _generate_corrective_sub_queries
# ---------------------------------------------------------------------------


class TestGenerateCorrectiveSubQueries:
    async def test_single_failing_claim_returns_sub_queries(self, mocker, make_mara_state):
        payload = _sub_queries_json([("automation job loss 2024", "statistical")])
        _mock_llm(mocker, payload)
        cfg = ResearchConfig(leaf_db_enabled=False)
        result = await _generate_corrective_sub_queries(
            [_scored_claim("Automation displaced workers")],
            "effects of automation",
            cfg,
            config={},
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["query"] == "automation job loss 2024"
        assert result[0]["domain"] == "statistical"

    async def test_handles_fenced_llm_response(self, mocker, make_mara_state):
        fenced = "```json\n" + _sub_queries_json([("query A", "empirical")]) + "\n```"
        _mock_llm(mocker, fenced)
        cfg = ResearchConfig(leaf_db_enabled=False)
        result = await _generate_corrective_sub_queries(
            [_scored_claim("Some claim")],
            "original query",
            cfg,
            config={},
        )
        assert result[0]["query"] == "query A"

    async def test_parse_error_falls_back_to_claim_text(self, mocker, make_mara_state):
        _mock_llm(mocker, "this is not json at all")
        cfg = ResearchConfig(leaf_db_enabled=False)
        claim = _scored_claim("My specific claim text that is long enough")
        result = await _generate_corrective_sub_queries(
            [claim], "original query", cfg, config={}
        )
        assert len(result) == 1
        assert result[0]["query"] == claim.text[:200]
        assert result[0]["domain"] == "general"

    async def test_multiple_claims_produce_sub_queries_for_each(self, mocker, make_mara_state):
        payload = _sub_queries_json([("q1", "d1")])
        _mock_llm(mocker, payload)
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("claim A"), _scored_claim("claim B")]
        result = await _generate_corrective_sub_queries(
            claims, "query", cfg, config={}
        )
        # 1 sub-query per claim × 2 claims = 2 total
        assert len(result) == 2

    async def test_up_to_two_sub_queries_per_claim(self, mocker, make_mara_state):
        payload = _sub_queries_json([("q1", "d1"), ("q2", "d2")])
        _mock_llm(mocker, payload)
        cfg = ResearchConfig(leaf_db_enabled=False)
        result = await _generate_corrective_sub_queries(
            [_scored_claim("claim")], "query", cfg, config={}
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# corrective_retriever — no failing claims
# ---------------------------------------------------------------------------


class TestNoFailingClaims:
    async def test_no_failing_returns_incremented_loop_count(self, mocker, make_mara_state):
        # All claims above low_confidence_threshold → no failing
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("approved", confidence=0.90)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=1)
        result = await corrective_retriever(state, config={})
        assert result["loop_count"] == 2

    async def test_no_failing_does_not_add_leaves(self, mocker, make_mara_state):
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("approved", confidence=0.90)]
        state = make_mara_state(scored_claims=claims, config=cfg)
        result = await corrective_retriever(state, config={})
        assert "merkle_leaves" not in result

    async def test_contested_claims_not_treated_as_failing(self, mocker, make_mara_state):
        # n_unique_urls >= n_leaves_contested_threshold → not a failing claim
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("contested", confidence=0.30, n_leaves=cfg.n_leaves_contested_threshold, n_unique_urls=cfg.n_leaves_contested_threshold)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        result = await corrective_retriever(state, config={})
        assert result["loop_count"] == 1
        assert "merkle_leaves" not in result


# ---------------------------------------------------------------------------
# corrective_retriever — DB-first path
# ---------------------------------------------------------------------------


class TestDbFirstPath:
    def _make_leaf_repo(self, mocker, db_rows: list[dict]):
        leaf_repo = mocker.MagicMock()
        leaf_repo.bm25_search = mocker.MagicMock(return_value=db_rows)
        leaf_repo.link_leaves_to_run = mocker.MagicMock()
        leaf_repo.upsert_leaves = mocker.MagicMock()
        return leaf_repo

    async def test_db_sufficient_appends_new_leaves(self, mocker, make_mara_state, make_merkle_leaf):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        # 3 * 1 failing = 3 new DB leaves satisfies sufficiency
        db_rows = [_make_leaf_dict(f"https://db.example.com/{i}") for i in range(3)]
        leaf_repo = self._make_leaf_repo(mocker, db_rows)
        cfg = ResearchConfig(leaf_db_enabled=True)
        claims = [_scored_claim("failing claim")]
        existing = [make_merkle_leaf(index=0)]
        state = make_mara_state(scored_claims=claims, merkle_leaves=existing, config=cfg)
        result = await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-1"}}
        )
        assert "merkle_leaves" in result
        assert len(result["merkle_leaves"]) == 1 + 3  # 1 original + 3 DB

    async def test_db_leaves_exclude_existing_hashes(self, mocker, make_mara_state, make_merkle_leaf):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        existing_leaf = make_merkle_leaf(index=0, url="https://example.com", text="sample text")
        # DB returns the same leaf (same hash) + 2 new ones
        same_hash_row = dict(existing_leaf)
        same_hash_row["embedding"] = None
        same_hash_row["embedding_model"] = None
        new_rows = [_make_leaf_dict(f"https://new.example.com/{i}") for i in range(2)]
        db_rows = [same_hash_row] + new_rows
        leaf_repo = self._make_leaf_repo(mocker, db_rows)
        cfg = ResearchConfig(leaf_db_enabled=True)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, merkle_leaves=[existing_leaf], config=cfg)
        result = await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-1"}}
        )
        # Only 2 genuinely new leaves added
        assert len(result["merkle_leaves"]) == 1 + 2

    async def test_link_leaves_to_run_called_with_new_db_leaves(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        db_rows = [_make_leaf_dict(f"https://db.example.com/{i}") for i in range(3)]
        leaf_repo = self._make_leaf_repo(mocker, db_rows)
        cfg = ResearchConfig(leaf_db_enabled=True)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-42"}}
        )
        leaf_repo.link_leaves_to_run.assert_called_once()
        call_run_id, call_leaves = leaf_repo.link_leaves_to_run.call_args.args
        assert call_run_id == "run-42"
        assert len(call_leaves) == 3

    async def test_leaf_db_disabled_skips_db_calls(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        # Even though we inject a leaf_repo, db_enabled=False should skip DB
        leaf_repo = mocker.MagicMock()
        _mock_brave(mocker, [])
        _mock_firecrawl(mocker, [])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-1"}}
        )
        leaf_repo.bm25_search.assert_not_called()

    async def test_db_new_leaves_have_correct_indices(self, mocker, make_mara_state, make_merkle_leaf):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        db_rows = [_make_leaf_dict(f"https://db.example.com/{i}") for i in range(3)]
        leaf_repo = self._make_leaf_repo(mocker, db_rows)
        cfg = ResearchConfig(leaf_db_enabled=True)
        claims = [_scored_claim("failing")]
        existing = [make_merkle_leaf(index=0), make_merkle_leaf(index=1, url="https://b.com")]
        state = make_mara_state(scored_claims=claims, merkle_leaves=existing, config=cfg)
        result = await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-1"}}
        )
        # New DB leaves should start at index 2 (len of existing)
        new_leaves = result["merkle_leaves"][2:]
        for i, leaf in enumerate(new_leaves):
            assert leaf["index"] == 2 + i


# ---------------------------------------------------------------------------
# corrective_retriever — scrape path
# ---------------------------------------------------------------------------


class TestScrapePath:
    def _make_leaf_repo(self, mocker, db_rows=None):
        leaf_repo = mocker.MagicMock()
        leaf_repo.bm25_search = mocker.MagicMock(return_value=db_rows or [])
        leaf_repo.link_leaves_to_run = mocker.MagicMock()
        leaf_repo.upsert_leaves = mocker.MagicMock()
        return leaf_repo

    def _make_chunk(self, url: str, text: str = "scraped content") -> dict:
        return {
            "url": url,
            "text": text,
            "retrieved_at": "2026-03-20T12:00:00Z",
            "sub_query": "corrective query",
        }

    async def test_db_insufficient_triggers_scrape(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        # DB returns 0 rows → insufficient for 1 failing claim (need >= 3)
        leaf_repo = self._make_leaf_repo(mocker, db_rows=[])
        brave_mock = mocker.patch(
            "mara.agent.nodes.corrective_retriever.brave_search",
            return_value={"search_results": [
                {"url": "https://new.com/1", "title": "t", "description": "", "extra_snippets": [], "page_age": "", "result_type": "web"}
            ]},
        )
        chunk = self._make_chunk("https://new.com/1")
        _mock_firecrawl(mocker, [chunk])
        cfg = ResearchConfig(leaf_db_enabled=True)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-1"}}
        )
        brave_mock.assert_called_once()

    async def test_scraped_leaves_appended_to_merkle_leaves(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, ["https://new.com/1"])
        chunk = self._make_chunk("https://new.com/1")
        _mock_firecrawl(mocker, [chunk])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        result = await corrective_retriever(state, config={})
        assert "merkle_leaves" in result
        assert len(result["merkle_leaves"]) >= 1

    async def test_url_dedup_skips_existing_urls(self, mocker, make_mara_state, make_merkle_leaf):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        existing_leaf = make_merkle_leaf(index=0, url="https://existing.com", text="existing")
        brave_results = [
            {"url": "https://existing.com", "title": "t", "description": "", "extra_snippets": [], "page_age": "", "result_type": "web"},
            {"url": "https://new.com/page", "title": "t", "description": "", "extra_snippets": [], "page_age": "", "result_type": "web"},
        ]
        mocker.patch(
            "mara.agent.nodes.corrective_retriever.brave_search",
            return_value={"search_results": brave_results},
        )
        chunk = self._make_chunk("https://new.com/page")
        _mock_firecrawl(mocker, [chunk])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, merkle_leaves=[existing_leaf], config=cfg)
        result = await corrective_retriever(state, config={})
        urls = [leaf["url"] for leaf in result["merkle_leaves"]]
        # existing.com should appear once only
        assert urls.count("https://existing.com") == 1

    async def test_new_leaf_indices_start_at_len_original(self, mocker, make_mara_state, make_merkle_leaf):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, ["https://new.com/1", "https://new.com/2"])
        chunks = [self._make_chunk("https://new.com/1"), self._make_chunk("https://new.com/2", "other content")]
        _mock_firecrawl(mocker, chunks)
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        existing = [make_merkle_leaf(index=0), make_merkle_leaf(index=1, url="https://b.com")]
        state = make_mara_state(scored_claims=claims, merkle_leaves=existing, config=cfg)
        result = await corrective_retriever(state, config={})
        new_leaves = result["merkle_leaves"][2:]
        for i, leaf in enumerate(new_leaves):
            assert leaf["index"] == 2 + i

    async def test_upsert_leaves_called_for_scraped_leaves(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, ["https://new.com/1"])
        chunk = self._make_chunk("https://new.com/1")
        _mock_firecrawl(mocker, [chunk])
        leaf_repo = self._make_leaf_repo(mocker)
        cfg = ResearchConfig(leaf_db_enabled=True)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        await corrective_retriever(
            state, config={"configurable": {"leaf_repo": leaf_repo, "run_id": "run-1"}}
        )
        leaf_repo.upsert_leaves.assert_called_once()

    async def test_max_new_pages_per_round_respected(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        # Brave returns 10 URLs but max_new_pages_per_round=2
        brave_results = [
            {"url": f"https://site.com/{i}", "title": "t", "description": "", "extra_snippets": [], "page_age": "", "result_type": "web"}
            for i in range(10)
        ]
        mocker.patch(
            "mara.agent.nodes.corrective_retriever.brave_search",
            return_value={"search_results": brave_results},
        )
        scrape_mock = mocker.patch(
            "mara.agent.nodes.corrective_retriever.firecrawl_scrape",
            return_value={"raw_chunks": []},
        )
        cfg = ResearchConfig(max_new_pages_per_round=2, leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        await corrective_retriever(state, config={})
        # firecrawl_scrape was called with search_results capped at 2
        call_state = scrape_mock.call_args.args[0]
        assert len(call_state["search_results"]) == 2

    async def test_leaf_db_disabled_still_scrapes(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        brave_mock = _mock_brave(mocker, ["https://new.com/1"])
        chunk = self._make_chunk("https://new.com/1")
        _mock_firecrawl(mocker, [chunk])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        result = await corrective_retriever(state, config={})
        assert len(result["merkle_leaves"]) >= 1


# ---------------------------------------------------------------------------
# corrective_retriever — general state management
# ---------------------------------------------------------------------------


class TestStateManagement:
    async def test_loop_count_incremented_on_every_call(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, [])
        _mock_firecrawl(mocker, [])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=1)
        result = await corrective_retriever(state, config={})
        assert result["loop_count"] == 2

    async def test_loop_count_incremented_when_no_failing(self, mocker, make_mara_state):
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("approved", confidence=0.90)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        result = await corrective_retriever(state, config={})
        assert result["loop_count"] == 1

    async def test_corrective_sub_queries_accumulated(self, mocker, make_mara_state):
        prior_sq = SubQuery(query="prior query", domain="prior")
        _mock_llm(mocker, _sub_queries_json([("new query", "empirical")]))
        _mock_brave(mocker, [])
        _mock_firecrawl(mocker, [])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(
            scored_claims=claims, config=cfg, corrective_sub_queries=[prior_sq]
        )
        result = await corrective_retriever(state, config={})
        assert "corrective_sub_queries" in result
        queries = [sq["query"] for sq in result["corrective_sub_queries"]]
        assert "prior query" in queries
        assert "new query" in queries

    async def test_returns_merkle_leaves_key(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, [])
        _mock_firecrawl(mocker, [])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        result = await corrective_retriever(state, config={})
        assert "merkle_leaves" in result

    async def test_returns_loop_count_key(self, mocker, make_mara_state):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, [])
        _mock_firecrawl(mocker, [])
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, config=cfg)
        result = await corrective_retriever(state, config={})
        assert "loop_count" in result

    async def test_original_leaves_preserved(self, mocker, make_mara_state, make_merkle_leaf):
        _mock_llm(mocker, _sub_queries_json([("q", "d")]))
        _mock_brave(mocker, ["https://new.com/1"])
        from tests.agent.nodes.test_corrective_retriever import TestScrapePath
        helper = TestScrapePath()
        _mock_firecrawl(mocker, [helper._make_chunk("https://new.com/1")])
        cfg = ResearchConfig(leaf_db_enabled=False)
        original = make_merkle_leaf(index=0, url="https://original.com", text="original")
        claims = [_scored_claim("failing")]
        state = make_mara_state(scored_claims=claims, merkle_leaves=[original], config=cfg)
        result = await corrective_retriever(state, config={})
        assert result["merkle_leaves"][0]["url"] == "https://original.com"
