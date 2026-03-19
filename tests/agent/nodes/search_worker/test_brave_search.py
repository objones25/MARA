"""Tests for mara.agent.nodes.search_worker.brave_search.

All HTTP calls are mocked — no real network access is made.
Tests cover the per-section extraction helpers (_web_results, _news_results,
_discussion_results, _faq_results) as pure functions, then the async
brave_search node as an integration of those helpers.

Design note: brave_search deliberately does NOT deduplicate results across
sections.  A URL appearing in both 'web' and 'news' carries distinct metadata
in each entry (different description, page_age, extra_snippets) and both
records are preserved.  URL-level deduplication before scraping is the
responsibility of firecrawl_scrape.
"""

import pytest
import httpx

from mara.agent.nodes.search_worker.brave_search import (
    brave_search,
    _discussion_results,
    _faq_results,
    _news_results,
    _web_results,
    _BRAVE_SEARCH_URL,
)
from mara.agent.state import SearchWorkerState, SubQuery
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "renewable energy trends",
    research_config: ResearchConfig | None = None,
) -> SearchWorkerState:
    return SearchWorkerState(
        sub_query=SubQuery(query=query, domain="energy"),
        research_config=research_config or ResearchConfig(),
        search_results=[],
        raw_chunks=[],
    )


def _mock_http(mocker, json_data: dict, status_error: bool = False):
    mock_resp = mocker.MagicMock()
    mock_resp.json.return_value = json_data
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
        "mara.agent.nodes.search_worker.brave_search.httpx.AsyncClient",
        return_value=mock_client,
    )
    return mock_client


# ---------------------------------------------------------------------------
# _web_results
# ---------------------------------------------------------------------------


class TestWebResults:
    def test_extracts_url_title_description(self):
        raw = [{"url": "https://a.com", "title": "T", "description": "D"}]
        results = _web_results(raw)
        assert results[0]["url"] == "https://a.com"
        assert results[0]["title"] == "T"
        assert results[0]["description"] == "D"

    def test_result_type_is_web(self):
        results = _web_results([{"url": "https://a.com"}])
        assert results[0]["result_type"] == "web"

    def test_extra_snippets_preserved(self):
        raw = [{"url": "https://a.com", "extra_snippets": ["s1", "s2"]}]
        assert _web_results(raw)[0]["extra_snippets"] == ["s1", "s2"]

    def test_missing_extra_snippets_defaults_to_empty_list(self):
        assert _web_results([{"url": "https://a.com"}])[0]["extra_snippets"] == []

    def test_none_extra_snippets_defaults_to_empty_list(self):
        results = _web_results([{"url": "https://a.com", "extra_snippets": None}])
        assert results[0]["extra_snippets"] == []

    def test_page_age_from_page_age_key(self):
        results = _web_results([{"url": "https://a.com", "page_age": "2026-01-01T00:00:00"}])
        assert results[0]["page_age"] == "2026-01-01T00:00:00"

    def test_page_age_falls_back_to_age_key(self):
        results = _web_results([{"url": "https://a.com", "age": "3 months ago"}])
        assert results[0]["page_age"] == "3 months ago"

    def test_missing_page_age_defaults_to_empty_string(self):
        assert _web_results([{"url": "https://a.com"}])[0]["page_age"] == ""

    def test_result_without_url_is_skipped(self):
        raw = [{"title": "No URL here"}, {"url": "https://a.com", "title": "Has URL"}]
        results = _web_results(raw)
        assert len(results) == 1
        assert results[0]["url"] == "https://a.com"

    def test_empty_input_returns_empty_list(self):
        assert _web_results([]) == []

    def test_missing_title_defaults_to_empty_string(self):
        assert _web_results([{"url": "https://a.com"}])[0]["title"] == ""

    def test_missing_description_defaults_to_empty_string(self):
        assert _web_results([{"url": "https://a.com"}])[0]["description"] == ""


# ---------------------------------------------------------------------------
# _news_results
# ---------------------------------------------------------------------------


class TestNewsResults:
    def test_result_type_is_news(self):
        assert _news_results([{"url": "https://news.com/story"}])[0]["result_type"] == "news"

    def test_age_key_used_for_page_age(self):
        """News results use 'age' not 'page_age'."""
        results = _news_results([{"url": "https://a.com", "age": "2 days ago"}])
        assert results[0]["page_age"] == "2 days ago"

    def test_page_age_key_also_accepted(self):
        results = _news_results([{"url": "https://a.com", "page_age": "2026-01-01T00:00:00"}])
        assert results[0]["page_age"] == "2026-01-01T00:00:00"

    def test_result_without_url_skipped(self):
        assert _news_results([{"title": "no url"}]) == []

    def test_empty_input_returns_empty_list(self):
        assert _news_results([]) == []


# ---------------------------------------------------------------------------
# _discussion_results
# ---------------------------------------------------------------------------


class TestDiscussionResults:
    def test_result_type_is_discussion(self):
        results = _discussion_results([{"url": "https://reddit.com/r/foo"}])
        assert results[0]["result_type"] == "discussion"

    def test_top_comment_used_when_description_empty(self):
        """data.top_comment fills description when description is absent."""
        raw = [{"url": "https://a.com", "description": "", "data": {"top_comment": "The answer is 42."}}]
        assert _discussion_results(raw)[0]["description"] == "The answer is 42."

    def test_description_takes_priority_over_top_comment(self):
        raw = [{"url": "https://a.com", "description": "Main desc", "data": {"top_comment": "comment"}}]
        assert _discussion_results(raw)[0]["description"] == "Main desc"

    def test_missing_data_field_does_not_raise(self):
        assert _discussion_results([{"url": "https://a.com"}])[0]["description"] == ""

    def test_none_data_field_does_not_raise(self):
        assert _discussion_results([{"url": "https://a.com", "data": None}])[0]["description"] == ""

    def test_result_without_url_skipped(self):
        assert _discussion_results([{"description": "no url"}]) == []

    def test_empty_input_returns_empty_list(self):
        assert _discussion_results([]) == []


# ---------------------------------------------------------------------------
# _faq_results
# ---------------------------------------------------------------------------


class TestFAQResults:
    def test_result_type_is_faq(self):
        results = _faq_results([{"url": "https://a.com", "question": "Q?", "answer": "A."}])
        assert results[0]["result_type"] == "faq"

    def test_question_and_answer_joined_in_description(self):
        results = _faq_results([{"url": "https://a.com", "question": "What?", "answer": "This."}])
        assert "What?" in results[0]["description"]
        assert "This." in results[0]["description"]

    def test_empty_question_and_answer_produces_empty_description(self):
        assert _faq_results([{"url": "https://a.com"}])[0]["description"] == ""

    def test_extra_snippets_always_empty(self):
        results = _faq_results([{"url": "https://a.com", "question": "Q?", "answer": "A."}])
        assert results[0]["extra_snippets"] == []

    def test_page_age_always_empty(self):
        results = _faq_results([{"url": "https://a.com", "question": "Q?", "answer": "A."}])
        assert results[0]["page_age"] == ""

    def test_result_without_url_skipped(self):
        assert _faq_results([{"question": "Q?", "answer": "A."}]) == []

    def test_empty_input_returns_empty_list(self):
        assert _faq_results([]) == []


# ---------------------------------------------------------------------------
# brave_search node — request parameters
# ---------------------------------------------------------------------------


class TestBraveSearchRequest:
    async def test_uses_correct_url(self, mocker):
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(), {})
        assert mock_client.get.call_args.args[0] == _BRAVE_SEARCH_URL

    async def test_query_sent_as_q_param(self, mocker):
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(query="ocean acidification"), {})
        assert mock_client.get.call_args.kwargs["params"]["q"] == "ocean acidification"

    async def test_extra_snippets_always_requested(self, mocker):
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(), {})
        assert mock_client.get.call_args.kwargs["params"]["extra_snippets"] == "true"

    async def test_count_uses_max_sources_directly(self, mocker):
        mock_client = _mock_http(mocker, {})
        config = ResearchConfig(max_sources=15)
        await brave_search(_make_state(research_config=config), {})
        assert mock_client.get.call_args.kwargs["params"]["count"] == 15

    async def test_count_capped_at_20_by_brave_api_limit(self, mocker):
        """Brave hard cap: no matter what max_sources says, count <= 20."""
        mock_client = _mock_http(mocker, {})
        config = ResearchConfig(max_sources=30)
        await brave_search(_make_state(research_config=config), {})
        assert mock_client.get.call_args.kwargs["params"]["count"] == 20

    async def test_api_key_in_subscription_token_header(self, mocker):
        mock_client = _mock_http(mocker, {})
        config = ResearchConfig(brave_api_key="my-key-abc")
        await brave_search(_make_state(research_config=config), {})
        assert mock_client.get.call_args.kwargs["headers"]["X-Subscription-Token"] == "my-key-abc"

    async def test_accept_json_header(self, mocker):
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(), {})
        assert mock_client.get.call_args.kwargs["headers"]["Accept"] == "application/json"

    async def test_accept_encoding_gzip_header(self, mocker):
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(), {})
        assert mock_client.get.call_args.kwargs["headers"]["Accept-Encoding"] == "gzip"

    async def test_freshness_param_sent_when_configured(self, mocker):
        mock_client = _mock_http(mocker, {})
        config = ResearchConfig(brave_freshness="pw")
        await brave_search(_make_state(research_config=config), {})
        assert mock_client.get.call_args.kwargs["params"]["freshness"] == "pw"

    async def test_freshness_param_absent_when_empty(self, mocker):
        """No freshness param when brave_freshness is '' (default)."""
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(research_config=ResearchConfig(brave_freshness="")), {})
        assert "freshness" not in mock_client.get.call_args.kwargs["params"]


# ---------------------------------------------------------------------------
# brave_search node — response handling
# ---------------------------------------------------------------------------


class TestBraveSearchResponse:
    async def test_returns_dict_with_search_results_key(self, mocker):
        _mock_http(mocker, {})
        result = await brave_search(_make_state(), {})
        assert "search_results" in result

    async def test_collects_web_results(self, mocker):
        _mock_http(mocker, {"web": {"results": [{"url": "https://web.com", "title": "Web"}]}})
        result = await brave_search(_make_state(), {})
        assert any(r["url"] == "https://web.com" for r in result["search_results"])

    async def test_collects_news_results(self, mocker):
        _mock_http(mocker, {"news": {"results": [{"url": "https://news.com/story", "title": "News"}]}})
        result = await brave_search(_make_state(), {})
        assert any(r["result_type"] == "news" for r in result["search_results"])

    async def test_collects_discussion_results(self, mocker):
        _mock_http(mocker, {"discussions": {"results": [{"url": "https://reddit.com/post"}]}})
        result = await brave_search(_make_state(), {})
        assert any(r["result_type"] == "discussion" for r in result["search_results"])

    async def test_collects_faq_results(self, mocker):
        _mock_http(mocker, {"faq": {"results": [{"url": "https://faq.com/q", "question": "Why?", "answer": "Because."}]}})
        result = await brave_search(_make_state(), {})
        assert any(r["result_type"] == "faq" for r in result["search_results"])

    async def test_same_url_in_two_sections_produces_two_results(self, mocker):
        """brave_search must NOT deduplicate.  Both entries carry distinct metadata
        (different result_type, description, extra_snippets) and are preserved.
        URL deduplication is firecrawl_scrape's responsibility."""
        shared_url = "https://shared.com/article"
        _mock_http(mocker, {
            "web": {"results": [{"url": shared_url, "title": "Web entry", "description": "Web desc"}]},
            "news": {"results": [{"url": shared_url, "title": "News entry", "description": "News desc"}]},
        })
        result = await brave_search(_make_state(), {})
        urls = [r["url"] for r in result["search_results"]]
        assert urls.count(shared_url) == 2

    async def test_all_four_sections_combined(self, mocker):
        _mock_http(mocker, {
            "web": {"results": [{"url": "https://web.com"}]},
            "news": {"results": [{"url": "https://news.com"}]},
            "discussions": {"results": [{"url": "https://disc.com"}]},
            "faq": {"results": [{"url": "https://faq.com", "question": "Q?", "answer": "A."}]},
        })
        result = await brave_search(_make_state(), {})
        types = {r["result_type"] for r in result["search_results"]}
        assert types == {"web", "news", "discussion", "faq"}

    async def test_empty_response_returns_empty_list(self, mocker):
        _mock_http(mocker, {})
        result = await brave_search(_make_state(), {})
        assert result["search_results"] == []

    async def test_raise_for_status_always_called(self, mocker):
        mock_client = _mock_http(mocker, {})
        await brave_search(_make_state(), {})
        mock_client.get.return_value.raise_for_status.assert_called_once()

    async def test_http_error_propagates(self, mocker):
        _mock_http(mocker, {}, status_error=True)
        with pytest.raises(httpx.HTTPStatusError):
            await brave_search(_make_state(), {})
