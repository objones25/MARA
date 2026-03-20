"""Tests for mara.agent.nodes.search_worker.firecrawl_scrape.

All Firecrawl SDK calls are mocked — no real network access is made.
asyncio.to_thread is NOT mocked; the synchronous mock callable runs in a real
thread pool, which exercises the actual to_thread wrapping path.

Design notes mirrored from the module under test:
  - URL deduplication happens HERE (not in brave_search) so the same page is
    not scraped twice.  A URL appearing in both 'web' and 'news' brave results
    carries distinct metadata but refers to the same page — we only scrape once.
  - Chunking is deterministic: same source text + same config → same chunks.
  - Empty/whitespace-only chunks are dropped.
  - retrieved_at is a single timestamp shared across all chunks from one call.
"""

import re
import pytest

from mara.agent.nodes.search_worker.firecrawl_scrape import (
    firecrawl_scrape,
    _chunk_text,
)
from mara.agent.state import SearchResult, SubQuery
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(url: str, result_type: str = "web") -> SearchResult:
    return SearchResult(
        url=url,
        title="",
        description="",
        extra_snippets=[],
        page_age="",
        result_type=result_type,
    )


def _make_doc(url: str, markdown: str):
    """Build a minimal mock document object matching the Firecrawl SDK shape."""
    doc = type("Doc", (), {})()
    doc.markdown = markdown
    doc.metadata = type("Meta", (), {"source_url": url})()
    return doc


def _make_job(docs: list):
    job = type("Job", (), {})()
    job.data = docs
    return job


def _mock_firecrawl(mocker, docs: list):
    """Patch Firecrawl so batch_scrape returns a job with the given docs."""
    job = _make_job(docs)
    mock_fc = mocker.MagicMock()
    mock_fc.batch_scrape.return_value = job
    mocker.patch(
        "mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl",
        return_value=mock_fc,
    )
    return mock_fc


# ---------------------------------------------------------------------------
# _chunk_text — pure function tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_string_returns_empty_list(self):
        assert _chunk_text("", 100, 20) == []

    def test_single_chunk_when_text_shorter_than_chunk_size(self):
        text = "hello world"
        result = _chunk_text(text, 100, 0)
        assert result == ["hello world"]

    def test_single_chunk_when_text_equals_chunk_size(self):
        text = "a" * 100
        assert _chunk_text(text, 100, 0) == ["a" * 100]

    def test_multiple_chunks_produced(self):
        text = "a" * 250
        chunks = _chunk_text(text, 100, 0)
        # step = 100, so windows at 0, 100, 200 → 3 chunks
        assert len(chunks) == 3

    def test_overlap_shifts_windows_correctly(self):
        text = "abcdefghij"  # 10 chars
        # chunk_size=6, overlap=2 → step=4; windows: [0:6], [4:10]
        chunks = _chunk_text(text, 6, 2)
        assert chunks[0] == "abcdef"
        assert chunks[1] == "efghij"

    def test_last_chunk_shorter_than_chunk_size(self):
        text = "a" * 110
        chunks = _chunk_text(text, 100, 0)
        assert chunks[-1] == "a" * 10

    def test_whitespace_only_chunk_is_dropped(self):
        # Construct text where the second window is all spaces
        text = "hello" + " " * 95 + "world" * 2
        # With chunk_size=100, overlap=0: chunk 0 = "hello" + 95 spaces (whitespace-only? no, has "hello")
        # Build a simpler case: first chunk has content, second is pure whitespace
        text = "content" + " " * 200
        chunks = _chunk_text(text, 100, 0)
        # All chunks after the first are spaces — they should be dropped
        assert all(c.strip() for c in chunks)

    def test_all_whitespace_input_returns_empty_list(self):
        assert _chunk_text("   \n\t  ", 100, 20) == []

    def test_chunk_size_one_no_overlap(self):
        chunks = _chunk_text("abc", 1, 0)
        assert chunks == ["a", "b", "c"]

    def test_deterministic_same_input_same_output(self):
        text = "x" * 500
        assert _chunk_text(text, 100, 25) == _chunk_text(text, 100, 25)


# ---------------------------------------------------------------------------
# firecrawl_scrape — early-return / empty paths
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeEmptyPaths:
    async def test_empty_search_results_returns_empty_raw_chunks(self, mocker, make_search_worker_state):
        mock_fc = mocker.MagicMock()
        mocker.patch(
            "mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl",
            return_value=mock_fc,
        )
        result = await firecrawl_scrape(make_search_worker_state(search_results=[]), {})
        assert result == {"raw_chunks": []}

    async def test_empty_search_results_does_not_call_batch_scrape(self, mocker, make_search_worker_state):
        mock_fc = mocker.MagicMock()
        mocker.patch(
            "mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl",
            return_value=mock_fc,
        )
        await firecrawl_scrape(make_search_worker_state(search_results=[]), {})
        mock_fc.batch_scrape.assert_not_called()

    async def test_job_data_none_returns_empty_raw_chunks(self, mocker, make_search_worker_state):
        job = type("Job", (), {"data": None})()
        mock_fc = mocker.MagicMock()
        mock_fc.batch_scrape.return_value = job
        mocker.patch(
            "mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl",
            return_value=mock_fc,
        )
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert result["raw_chunks"] == []

    async def test_doc_with_empty_markdown_is_skipped(self, mocker, make_search_worker_state):
        _mock_firecrawl(mocker, [_make_doc("https://a.com", "")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert result["raw_chunks"] == []

    async def test_doc_with_none_markdown_is_skipped(self, mocker, make_search_worker_state):
        doc = _make_doc("https://a.com", "")
        doc.markdown = None
        _mock_firecrawl(mocker, [doc])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert result["raw_chunks"] == []

    async def test_doc_with_none_metadata_is_skipped(self, mocker, make_search_worker_state):
        doc = _make_doc("https://a.com", "some content here")
        doc.metadata = None
        _mock_firecrawl(mocker, [doc])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert result["raw_chunks"] == []

    async def test_doc_with_empty_source_url_is_skipped(self, mocker, make_search_worker_state):
        doc = _make_doc("", "some content here")
        _mock_firecrawl(mocker, [doc])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert result["raw_chunks"] == []


# ---------------------------------------------------------------------------
# firecrawl_scrape — URL deduplication
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeDeduplication:
    async def test_duplicate_urls_deduplicated_before_batch_scrape(self, mocker, make_search_worker_state):
        """Same URL appearing twice in search_results (e.g. web + news) must be
        passed to batch_scrape only once."""
        mock_fc = _mock_firecrawl(mocker, [])
        state = make_search_worker_state(search_results=[
            _make_search_result("https://shared.com", "web"),
            _make_search_result("https://shared.com", "news"),
        ])
        await firecrawl_scrape(state, {})
        urls_sent = mock_fc.batch_scrape.call_args.args[0]
        assert urls_sent.count("https://shared.com") == 1

    async def test_deduplication_preserves_first_occurrence_order(self, mocker, make_search_worker_state):
        """dict.fromkeys preserves insertion order; web results come first."""
        mock_fc = _mock_firecrawl(mocker, [])
        state = make_search_worker_state(search_results=[
            _make_search_result("https://first.com", "web"),
            _make_search_result("https://second.com", "news"),
            _make_search_result("https://first.com", "discussion"),
        ])
        await firecrawl_scrape(state, {})
        urls_sent = mock_fc.batch_scrape.call_args.args[0]
        assert urls_sent == ["https://first.com", "https://second.com"]

    async def test_unique_urls_all_passed_to_batch_scrape(self, mocker, make_search_worker_state):
        mock_fc = _mock_firecrawl(mocker, [])
        urls = [f"https://site{i}.com" for i in range(5)]
        state = make_search_worker_state(search_results=[_make_search_result(u) for u in urls])
        await firecrawl_scrape(state, {})
        urls_sent = mock_fc.batch_scrape.call_args.args[0]
        assert urls_sent == urls


# ---------------------------------------------------------------------------
# firecrawl_scrape — Firecrawl API call parameters
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeAPICall:
    async def test_batch_scrape_called_with_markdown_format(self, mocker, make_search_worker_state):
        mock_fc = _mock_firecrawl(mocker, [])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        await firecrawl_scrape(state, {})
        assert mock_fc.batch_scrape.call_args.kwargs["formats"] == ["markdown"]

    async def test_firecrawl_instantiated_with_api_key(self, mocker, make_search_worker_state):
        mock_cls = mocker.patch(
            "mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl"
        )
        mock_cls.return_value.batch_scrape.return_value = _make_job([])
        config = ResearchConfig(firecrawl_api_key="fc-test-key")
        state = make_search_worker_state(
            search_results=[_make_search_result("https://a.com")],
            config=config,
        )
        await firecrawl_scrape(state, {})
        mock_cls.assert_called_once_with(api_key="fc-test-key")


# ---------------------------------------------------------------------------
# firecrawl_scrape — SourceChunk population
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeChunkPopulation:
    async def test_returns_dict_with_raw_chunks_key(self, mocker, make_search_worker_state):
        _mock_firecrawl(mocker, [_make_doc("https://a.com", "hello world content")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert "raw_chunks" in result

    async def test_chunk_url_matches_doc_source_url(self, mocker, make_search_worker_state):
        _mock_firecrawl(mocker, [_make_doc("https://a.com", "some text content here")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        assert all(c["url"] == "https://a.com" for c in result["raw_chunks"])

    async def test_chunk_text_contains_source_content(self, mocker, make_search_worker_state):
        _mock_firecrawl(mocker, [_make_doc("https://a.com", "unique content phrase")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        combined = " ".join(c["text"] for c in result["raw_chunks"])
        assert "unique content phrase" in combined

    async def test_chunk_sub_query_matches_state_sub_query(self, mocker, make_search_worker_state):
        _mock_firecrawl(mocker, [_make_doc("https://a.com", "content")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        state["sub_query"] = SubQuery(query="my test query", domain="test")
        result = await firecrawl_scrape(state, {})
        assert all(c["sub_query"] == "my test query" for c in result["raw_chunks"])

    async def test_retrieved_at_is_iso8601_utc_format(self, mocker, make_search_worker_state):
        _mock_firecrawl(mocker, [_make_doc("https://a.com", "content")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        retrieved_at = result["raw_chunks"][0]["retrieved_at"]
        # Must match YYYY-MM-DDTHH:MM:SSZ
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", retrieved_at)

    async def test_all_chunks_from_same_call_share_retrieved_at(self, mocker, make_search_worker_state):
        """retrieved_at is set once per node call, not per chunk."""
        long_text = "word " * 300  # enough to produce multiple chunks
        _mock_firecrawl(mocker, [_make_doc("https://a.com", long_text)])
        config = ResearchConfig(chunk_size=100, chunk_overlap=0)
        state = make_search_worker_state(
            search_results=[_make_search_result("https://a.com")],
            config=config,
        )
        result = await firecrawl_scrape(state, {})
        timestamps = {c["retrieved_at"] for c in result["raw_chunks"]}
        assert len(timestamps) == 1

    async def test_chunking_respects_chunk_size_config(self, mocker, make_search_worker_state):
        text = "a" * 500
        _mock_firecrawl(mocker, [_make_doc("https://a.com", text)])
        config = ResearchConfig(chunk_size=100, chunk_overlap=0)
        state = make_search_worker_state(
            search_results=[_make_search_result("https://a.com")],
            config=config,
        )
        result = await firecrawl_scrape(state, {})
        for chunk in result["raw_chunks"]:
            assert len(chunk["text"]) <= 100

    async def test_chunking_respects_chunk_overlap_config(self, mocker, make_search_worker_state):
        text = "abcdefghij" * 10  # 100 chars
        _mock_firecrawl(mocker, [_make_doc("https://a.com", text)])
        config = ResearchConfig(chunk_size=10, chunk_overlap=2)
        state = make_search_worker_state(
            search_results=[_make_search_result("https://a.com")],
            config=config,
        )
        result = await firecrawl_scrape(state, {})
        # step = 8; windows: 0, 8, 16, ...; first two chunks should share 2-char overlap
        assert result["raw_chunks"][0]["text"][-2:] == result["raw_chunks"][1]["text"][:2]

    async def test_multiple_docs_all_produce_chunks(self, mocker, make_search_worker_state):
        docs = [
            _make_doc("https://a.com", "content from A"),
            _make_doc("https://b.com", "content from B"),
        ]
        _mock_firecrawl(mocker, docs)
        state = make_search_worker_state(search_results=[
            _make_search_result("https://a.com"),
            _make_search_result("https://b.com"),
        ])
        result = await firecrawl_scrape(state, {})
        urls_in_chunks = {c["url"] for c in result["raw_chunks"]}
        assert "https://a.com" in urls_in_chunks
        assert "https://b.com" in urls_in_chunks

    async def test_mixed_valid_and_invalid_docs(self, mocker, make_search_worker_state):
        """One valid doc + one with empty markdown → only valid doc produces chunks."""
        docs = [
            _make_doc("https://good.com", "valid content here"),
            _make_doc("https://bad.com", ""),
        ]
        _mock_firecrawl(mocker, docs)
        state = make_search_worker_state(search_results=[
            _make_search_result("https://good.com"),
            _make_search_result("https://bad.com"),
        ])
        result = await firecrawl_scrape(state, {})
        urls = {c["url"] for c in result["raw_chunks"]}
        assert "https://good.com" in urls
        assert "https://bad.com" not in urls


# ---------------------------------------------------------------------------
# firecrawl_scrape — DB cache integration
# ---------------------------------------------------------------------------


class TestFirecrawlScrapeCacheIntegration:
    """Verify freshness-cache behaviour when leaf_repo is injected."""

    def _make_cached_leaf(self, url: str, text: str = "cached content") -> dict:
        return {
            "hash": "abc123",
            "url": url,
            "text": text,
            "retrieved_at": "2026-03-19T10:00:00Z",
            "contextualized_text": text,
        }

    async def test_cache_hit_skips_firecrawl(self, mocker, make_search_worker_state):
        """When leaf_repo returns fresh leaves, batch_scrape must not be called."""
        mock_fc_cls = mocker.patch(
            "mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl"
        )
        repo = mocker.MagicMock()
        repo.get_fresh_leaves_for_url.return_value = [
            self._make_cached_leaf("https://a.com")
        ]
        config = {"configurable": {"leaf_repo": repo}}
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        await firecrawl_scrape(state, config)
        mock_fc_cls.assert_not_called()

    async def test_cache_hit_returns_cached_chunks(self, mocker, make_search_worker_state):
        mocker.patch("mara.agent.nodes.search_worker.firecrawl_scrape.Firecrawl")
        repo = mocker.MagicMock()
        repo.get_fresh_leaves_for_url.return_value = [
            self._make_cached_leaf("https://a.com", text="cached text here")
        ]
        config = {"configurable": {"leaf_repo": repo}}
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, config)
        assert len(result["raw_chunks"]) == 1
        assert result["raw_chunks"][0]["text"] == "cached text here"

    async def test_cache_miss_falls_through_to_firecrawl(self, mocker, make_search_worker_state):
        mock_fc = _mock_firecrawl(mocker, [_make_doc("https://a.com", "live content")])
        repo = mocker.MagicMock()
        repo.get_fresh_leaves_for_url.return_value = []  # cache miss
        config = {"configurable": {"leaf_repo": repo}}
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, config)
        mock_fc.batch_scrape.assert_called_once()
        assert any("live content" in c["text"] for c in result["raw_chunks"])

    async def test_partial_cache_scrapes_only_misses(self, mocker, make_search_worker_state):
        """URLs with a cache hit are skipped; only misses go to Firecrawl."""
        mock_fc = _mock_firecrawl(mocker, [_make_doc("https://b.com", "live B")])
        repo = mocker.MagicMock()

        def _cache(url, max_age):
            if url == "https://a.com":
                return [self._make_cached_leaf("https://a.com", "cached A")]
            return []

        repo.get_fresh_leaves_for_url.side_effect = _cache
        config = {"configurable": {"leaf_repo": repo}}
        state = make_search_worker_state(search_results=[
            _make_search_result("https://a.com"),
            _make_search_result("https://b.com"),
        ])
        result = await firecrawl_scrape(state, config)
        urls_scraped = mock_fc.batch_scrape.call_args.args[0]
        assert urls_scraped == ["https://b.com"]
        texts = {c["text"] for c in result["raw_chunks"]}
        assert "cached A" in texts
        assert any("live B" in t for t in texts)

    async def test_no_repo_bypasses_cache(self, mocker, make_search_worker_state):
        """When leaf_repo is absent, all URLs go straight to Firecrawl."""
        mock_fc = _mock_firecrawl(mocker, [_make_doc("https://a.com", "fresh content")])
        state = make_search_worker_state(search_results=[_make_search_result("https://a.com")])
        result = await firecrawl_scrape(state, {})
        mock_fc.batch_scrape.assert_called_once()
        assert result["raw_chunks"]
