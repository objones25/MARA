"""Tests for mara.agent.nodes.search_worker.semantic_scholar_search.

All HTTP calls are mocked — no real network access is made.
Tests cover the async semantic_scholar_search node which calls the
Semantic Scholar /snippet/search endpoint and returns SourceChunks directly
(no Firecrawl scraping required).
"""

import time

import pytest
import httpx

import mara.agent.nodes.search_worker.semantic_scholar_search as _s2_mod
from mara.agent.nodes.search_worker.semantic_scholar_search import (
    semantic_scholar_search,
    _S2_SNIPPET_URL,
    _S2_MIN_INTERVAL,
)
from mara.config import ResearchConfig


@pytest.fixture(autouse=True)
def reset_s2_rate_limiter():
    """Reset rate-limit state before every test so no test waits for the window."""
    _s2_mod._s2_last_call_time = float("-inf")
    yield
    _s2_mod._s2_last_call_time = float("-inf")


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_ONE_SNIPPET_RESPONSE = {
    "data": [
        {
            "snippet": {
                "text": "Large language models have shown significant productivity gains.",
                "section": "Introduction",
                "snippetKind": "body",
            },
            "paper": {
                "corpusId": "12345",
                "title": "LLM Productivity Study",
                "authors": ["Alice Smith"],
            },
            "score": 0.95,
        }
    ]
}

_TWO_SNIPPET_RESPONSE = {
    "data": [
        {
            "snippet": {"text": "First snippet text.", "section": "Abstract", "snippetKind": "abstract"},
            "paper": {"corpusId": "11111", "title": "Paper A"},
            "score": 0.9,
        },
        {
            "snippet": {"text": "Second snippet text.", "section": "Results", "snippetKind": "body"},
            "paper": {"corpusId": "22222", "title": "Paper B"},
            "score": 0.8,
        },
    ]
}

_EMPTY_RESPONSE = {"data": []}

_MISSING_TEXT_RESPONSE = {
    "data": [
        {
            "snippet": {"text": "", "section": "Introduction", "snippetKind": "body"},
            "paper": {"corpusId": "99999", "title": "Paper C"},
            "score": 0.5,
        }
    ]
}

_MISSING_CORPUS_ID_RESPONSE = {
    "data": [
        {
            "snippet": {"text": "Some text here.", "section": "Introduction", "snippetKind": "body"},
            "paper": {"title": "No ID Paper"},
            "score": 0.5,
        }
    ]
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s2_state(make_search_worker_state):
    def _factory(query="llm productivity", max_results=5, api_key=""):
        return make_search_worker_state(
            query=query,
            domain="cs",
            config=ResearchConfig(
                semantic_scholar_max_results=max_results,
                semantic_scholar_api_key=api_key,
                leaf_db_enabled=False,
            ),
        )

    return _factory


def _mock_http(mocker, response_data: dict, status_error: bool = False):
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = response_data
    if status_error:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=mocker.MagicMock(), response=mocker.MagicMock()
        )
    else:
        mock_resp.raise_for_status.return_value = None

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_client.get = mocker.AsyncMock(return_value=mock_resp)

    mocker.patch(
        "mara.agent.nodes.search_worker.semantic_scholar_search.httpx.AsyncClient",
        return_value=mock_client,
    )
    return mock_client


# ---------------------------------------------------------------------------
# semantic_scholar_search node
# ---------------------------------------------------------------------------


class TestSemanticScholarSearch:
    @pytest.mark.asyncio
    async def test_returns_raw_chunks_key(self, mocker, s2_state):
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert "raw_chunks" in result

    @pytest.mark.asyncio
    async def test_calls_snippet_search_endpoint(self, mocker, s2_state):
        mock_client = _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(), config={})
        mock_client.get.assert_called_once()
        assert mock_client.get.call_args.args[0] == _S2_SNIPPET_URL

    @pytest.mark.asyncio
    async def test_query_param_matches_sub_query(self, mocker, s2_state):
        mock_client = _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(query="economic effects of AI"), config={})
        params = mock_client.get.call_args.kwargs["params"]
        assert params["query"] == "economic effects of AI"

    @pytest.mark.asyncio
    async def test_limit_param_matches_config(self, mocker, s2_state):
        mock_client = _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(max_results=3), config={})
        params = mock_client.get.call_args.kwargs["params"]
        assert params["limit"] == 3

    @pytest.mark.asyncio
    async def test_one_snippet_returns_one_chunk(self, mocker, s2_state):
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert len(result["raw_chunks"]) == 1

    @pytest.mark.asyncio
    async def test_two_snippets_returns_two_chunks(self, mocker, s2_state):
        _mock_http(mocker, _TWO_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert len(result["raw_chunks"]) == 2

    @pytest.mark.asyncio
    async def test_chunk_text_matches_snippet_text(self, mocker, s2_state):
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result["raw_chunks"][0]["text"] == "Large language models have shown significant productivity gains."

    @pytest.mark.asyncio
    async def test_chunk_url_uses_corpus_id(self, mocker, s2_state):
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result["raw_chunks"][0]["url"] == "https://www.semanticscholar.org/paper/CorpusId:12345"

    @pytest.mark.asyncio
    async def test_chunk_sub_query_matches_query(self, mocker, s2_state):
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(query="developer productivity"), config={})
        assert result["raw_chunks"][0]["sub_query"] == "developer productivity"

    @pytest.mark.asyncio
    async def test_chunk_retrieved_at_is_iso_format(self, mocker, s2_state):
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        ts = result["raw_chunks"][0]["retrieved_at"]
        assert "T" in ts and ts.endswith("Z")

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self, mocker, s2_state):
        _mock_http(mocker, _EMPTY_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result["raw_chunks"] == []

    @pytest.mark.asyncio
    async def test_entry_with_empty_text_skipped(self, mocker, s2_state):
        _mock_http(mocker, _MISSING_TEXT_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result["raw_chunks"] == []

    @pytest.mark.asyncio
    async def test_entry_with_missing_corpus_id_skipped(self, mocker, s2_state):
        _mock_http(mocker, _MISSING_CORPUS_ID_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result["raw_chunks"] == []

    @pytest.mark.asyncio
    async def test_api_key_included_in_header_when_set(self, mocker, s2_state):
        mock_client = _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(api_key="test-key-123"), config={})
        headers = mock_client.get.call_args.kwargs["headers"]
        assert headers.get("x-api-key") == "test-key-123"

    @pytest.mark.asyncio
    async def test_api_key_absent_from_header_when_empty(self, mocker, s2_state):
        mock_client = _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(api_key=""), config={})
        headers = mock_client.get.call_args.kwargs["headers"]
        assert "x-api-key" not in headers

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_chunks(self, mocker, s2_state):
        _mock_http(mocker, {}, status_error=True)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_chunks(self, mocker, s2_state):
        mock_client = mocker.AsyncMock()
        mock_client.__aenter__.return_value.get = mocker.AsyncMock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.semantic_scholar_search.httpx.AsyncClient",
            return_value=mock_client,
        )
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_multiple_chunks_have_correct_corpus_id_urls(self, mocker, s2_state):
        _mock_http(mocker, _TWO_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        urls = [c["url"] for c in result["raw_chunks"]]
        assert "https://www.semanticscholar.org/paper/CorpusId:11111" in urls
        assert "https://www.semanticscholar.org/paper/CorpusId:22222" in urls

    @pytest.mark.asyncio
    async def test_all_chunks_share_same_retrieved_at(self, mocker, s2_state):
        _mock_http(mocker, _TWO_SNIPPET_RESPONSE)
        result = await semantic_scholar_search(s2_state(), config={})
        timestamps = [c["retrieved_at"] for c in result["raw_chunks"]]
        assert timestamps[0] == timestamps[1]


# ---------------------------------------------------------------------------
# Error differentiation — status-code-specific branches
# ---------------------------------------------------------------------------


def _mock_http_status_code(mocker, status_code: int):
    """Patch AsyncClient so raise_for_status() raises HTTPStatusError with a real status_code."""
    mock_response = mocker.MagicMock()
    mock_response.status_code = status_code
    mock_resp = mocker.MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=mocker.MagicMock(), response=mock_response
    )
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_client.get = mocker.AsyncMock(return_value=mock_resp)
    mocker.patch(
        "mara.agent.nodes.search_worker.semantic_scholar_search.httpx.AsyncClient",
        return_value=mock_client,
    )


def _mock_transport_error(mocker, exc: httpx.HTTPError):
    """Patch AsyncClient so client.get() raises a transport-level HTTPError."""
    mock_client = mocker.AsyncMock()
    mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mocker.AsyncMock(return_value=None)
    mock_client.get = mocker.AsyncMock(side_effect=exc)
    mocker.patch(
        "mara.agent.nodes.search_worker.semantic_scholar_search.httpx.AsyncClient",
        return_value=mock_client,
    )


class TestSemanticScholarStatusErrors:
    @pytest.mark.asyncio
    async def test_401_returns_empty_chunks(self, mocker, s2_state):
        _mock_http_status_code(mocker, 401)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_403_returns_empty_chunks(self, mocker, s2_state):
        _mock_http_status_code(mocker, 403)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_429_returns_empty_chunks(self, mocker, s2_state):
        _mock_http_status_code(mocker, 429)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_500_returns_empty_chunks(self, mocker, s2_state):
        _mock_http_status_code(mocker, 500)
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_transport_error_returns_empty_chunks(self, mocker, s2_state):
        _mock_transport_error(mocker, httpx.RemoteProtocolError("peer closed connection"))
        result = await semantic_scholar_search(s2_state(), config={})
        assert result == {"raw_chunks": []}

    @pytest.mark.asyncio
    async def test_auth_failure_logs_at_error_level(self, mocker, s2_state):
        mock_log = mocker.patch(
            "mara.agent.nodes.search_worker.semantic_scholar_search._log"
        )
        _mock_http_status_code(mocker, 403)
        await semantic_scholar_search(s2_state(), config={})
        mock_log.error.assert_called_once()
        mock_log.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_logs_at_warning_level(self, mocker, s2_state):
        mock_log = mocker.patch(
            "mara.agent.nodes.search_worker.semantic_scholar_search._log"
        )
        _mock_http_status_code(mocker, 429)
        await semantic_scholar_search(s2_state(), config={})
        mock_log.warning.assert_called_once()
        mock_log.error.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting — _acquire_s2_slot behaviour
# ---------------------------------------------------------------------------


class TestSemanticScholarRateLimit:
    @pytest.mark.asyncio
    async def test_no_sleep_on_first_call(self, mocker, s2_state):
        """First call with last_call_time=-inf must not sleep."""
        mock_sleep = mocker.patch("asyncio.sleep")
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(), config={})
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_sleep_when_interval_already_elapsed(self, mocker, s2_state):
        """Call made >1s after the previous one must not sleep."""
        _s2_mod._s2_last_call_time = time.monotonic() - (1.5)
        mock_sleep = mocker.patch("asyncio.sleep")
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(), config={})
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_sleeps_for_remaining_interval(self, mocker, s2_state):
        """Call made 0.3s after the previous one must sleep ~0.7s."""
        _s2_mod._s2_last_call_time = time.monotonic() - 0.3
        mock_sleep = mocker.patch("asyncio.sleep")
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        await semantic_scholar_search(s2_state(), config={})
        mock_sleep.assert_called_once()
        duration = mock_sleep.call_args.args[0]
        assert 0.5 < duration <= _S2_MIN_INTERVAL

    @pytest.mark.asyncio
    async def test_last_call_time_updated_after_slot_acquired(self, mocker, s2_state):
        """_s2_last_call_time must be set to a recent monotonic value after each call."""
        mocker.patch("asyncio.sleep")
        _mock_http(mocker, _ONE_SNIPPET_RESPONSE)
        before = time.monotonic()
        await semantic_scholar_search(s2_state(), config={})
        after = time.monotonic()
        assert before <= _s2_mod._s2_last_call_time <= after

    @pytest.mark.asyncio
    async def test_rate_limiter_fires_even_on_http_error(self, mocker, s2_state):
        """Slot must be acquired (and timestamp updated) even when the request fails."""
        mocker.patch("asyncio.sleep")
        _mock_http_status_code(mocker, 500)
        before = time.monotonic()
        await semantic_scholar_search(s2_state(), config={})
        after = time.monotonic()
        assert before <= _s2_mod._s2_last_call_time <= after
