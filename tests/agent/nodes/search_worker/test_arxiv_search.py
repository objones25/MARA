"""Tests for mara.agent.nodes.search_worker.arxiv_search.

All HTTP calls are mocked — no real network access is made.
Tests cover the XML parser (_parse_entries) as a pure function, then the
async arxiv_search node.

Design: arxiv_search mirrors brave_search in its interface — it produces
SearchResult objects with result_type="arxiv" and versioned PDF URLs.
The actual SourceChunks are produced downstream by firecrawl_scrape.
"""

import pytest
import httpx

from mara.agent.nodes.search_worker.arxiv_search import (
    arxiv_search,
    _parse_entries,
    _ARXIV_API_URL,
)
from mara.agent.state import SearchWorkerState, SubQuery
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_ONE_PAPER_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.07041</id>
    <title>Advances in Neural Architecture Search</title>
    <summary>We present a new method for neural architecture search that significantly reduces search cost.</summary>
    <published>2023-01-17T00:00:00Z</published>
    <updated>2023-01-17T00:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2301.07041v1"/>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/2301.07041v1"/>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

_WITH_JOURNAL_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/hep-ex/0307015</id>
    <title>Multi-Electron Production</title>
    <summary>Multi-electron production is studied at high electron transverse momentum.</summary>
    <published>2003-07-07T13:46:39-04:00</published>
    <author><name>H1 Collaboration</name></author>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/hep-ex/0307015v1"/>
    <arxiv:primary_category term="hep-ex" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:journal_ref>Eur.Phys.J. C31 (2003) 17-29</arxiv:journal_ref>
  </entry>
</feed>"""

_ERROR_ENTRY_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id>
    <title>Error</title>
    <summary>incorrect id format for 1234</summary>
  </entry>
</feed>"""

_NO_PDF_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.99999</id>
    <title>No PDF Link</title>
    <summary>This entry has no PDF link.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Some Author</name></author>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/2301.99999v1"/>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

_NO_ABSTRACT_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.88888</id>
    <title>No Abstract</title>
    <summary></summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Some Author</name></author>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/2301.88888v1"/>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

_TWO_PAPERS_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001</id>
    <title>Paper One</title>
    <summary>Abstract one.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>Author A</name></author>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/2301.00001v1"/>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002</id>
    <title>Paper Two</title>
    <summary>Abstract two.</summary>
    <published>2023-01-02T00:00:00Z</published>
    <author><name>Author B</name></author>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/2301.00002v1"/>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""

_EMPTY_FEED_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "neural architecture search",
    max_results: int = 5,
) -> SearchWorkerState:
    return SearchWorkerState(
        sub_query=SubQuery(query=query, domain="cs"),
        research_config=ResearchConfig(arxiv_max_results=max_results),
        search_results=[],
        raw_chunks=[],
    )


def _mock_http(mocker, xml_text: str, status_error: bool = False):
    mock_resp = mocker.MagicMock()
    mock_resp.text = xml_text
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
        "mara.agent.nodes.search_worker.arxiv_search.httpx.AsyncClient",
        return_value=mock_client,
    )
    return mock_client


# ---------------------------------------------------------------------------
# _parse_entries
# ---------------------------------------------------------------------------


class TestParseEntries:
    def test_valid_entry_returns_one_paper(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert len(papers) == 1

    def test_url_is_versioned_pdf_url(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert papers[0]["url"] == "http://arxiv.org/pdf/2301.07041v1"

    def test_title_extracted(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert papers[0]["title"] == "Advances in Neural Architecture Search"

    def test_description_is_abstract(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert "neural architecture search" in papers[0]["description"]

    def test_authors_extracted_as_list(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert papers[0]["authors"] == ["Alice Smith", "Bob Jones"]

    def test_published_date_truncated_to_date(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert papers[0]["published"] == "2023-01-17"

    def test_primary_category_extracted(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert papers[0]["primary_category"] == "cs.LG"

    def test_journal_ref_extracted_when_present(self):
        papers = _parse_entries(_WITH_JOURNAL_XML)
        assert papers[0]["journal_ref"] == "Eur.Phys.J. C31 (2003) 17-29"

    def test_journal_ref_empty_when_absent(self):
        papers = _parse_entries(_ONE_PAPER_XML)
        assert papers[0]["journal_ref"] == ""

    def test_error_entry_skipped(self):
        papers = _parse_entries(_ERROR_ENTRY_XML)
        assert papers == []

    def test_entry_without_pdf_link_skipped(self):
        papers = _parse_entries(_NO_PDF_XML)
        assert papers == []

    def test_entry_with_empty_abstract_skipped(self):
        papers = _parse_entries(_NO_ABSTRACT_XML)
        assert papers == []

    def test_empty_feed_returns_empty_list(self):
        papers = _parse_entries(_EMPTY_FEED_XML)
        assert papers == []

    def test_multiple_entries_returned(self):
        papers = _parse_entries(_TWO_PAPERS_XML)
        assert len(papers) == 2

    def test_multiple_entries_order_preserved(self):
        papers = _parse_entries(_TWO_PAPERS_XML)
        assert papers[0]["title"] == "Paper One"
        assert papers[1]["title"] == "Paper Two"


# ---------------------------------------------------------------------------
# arxiv_search node
# ---------------------------------------------------------------------------


class TestArxivSearchNode:
    @pytest.mark.asyncio
    async def test_returns_search_results_key(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert "search_results" in result

    @pytest.mark.asyncio
    async def test_calls_arxiv_api(self, mocker):
        mock_client = _mock_http(mocker, _ONE_PAPER_XML)
        await arxiv_search(_make_state(), config={})
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args.args[0] == _ARXIV_API_URL

    @pytest.mark.asyncio
    async def test_search_query_uses_all_prefix(self, mocker):
        mock_client = _mock_http(mocker, _ONE_PAPER_XML)
        await arxiv_search(_make_state(query="neural search"), config={})
        params = mock_client.get.call_args.kwargs["params"]
        assert params["search_query"] == "all:neural search"

    @pytest.mark.asyncio
    async def test_max_results_from_config(self, mocker):
        mock_client = _mock_http(mocker, _ONE_PAPER_XML)
        await arxiv_search(_make_state(max_results=3), config={})
        params = mock_client.get.call_args.kwargs["params"]
        assert params["max_results"] == 3

    @pytest.mark.asyncio
    async def test_sort_by_relevance(self, mocker):
        mock_client = _mock_http(mocker, _ONE_PAPER_XML)
        await arxiv_search(_make_state(), config={})
        params = mock_client.get.call_args.kwargs["params"]
        assert params["sortBy"] == "relevance"

    @pytest.mark.asyncio
    async def test_result_type_is_arxiv(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert result["search_results"][0]["result_type"] == "arxiv"

    @pytest.mark.asyncio
    async def test_result_url_is_pdf_url(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert result["search_results"][0]["url"] == "http://arxiv.org/pdf/2301.07041v1"

    @pytest.mark.asyncio
    async def test_result_title_matches_paper(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert result["search_results"][0]["title"] == "Advances in Neural Architecture Search"

    @pytest.mark.asyncio
    async def test_result_description_is_abstract(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert "neural architecture search" in result["search_results"][0]["description"]

    @pytest.mark.asyncio
    async def test_result_page_age_is_published_date(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert result["search_results"][0]["page_age"] == "2023-01-17"

    @pytest.mark.asyncio
    async def test_result_extra_snippets_empty(self, mocker):
        _mock_http(mocker, _ONE_PAPER_XML)
        result = await arxiv_search(_make_state(), config={})
        assert result["search_results"][0]["extra_snippets"] == []

    @pytest.mark.asyncio
    async def test_two_papers_returns_two_results(self, mocker):
        _mock_http(mocker, _TWO_PAPERS_XML)
        result = await arxiv_search(_make_state(), config={})
        assert len(result["search_results"]) == 2

    @pytest.mark.asyncio
    async def test_empty_feed_returns_empty_list(self, mocker):
        _mock_http(mocker, _EMPTY_FEED_XML)
        result = await arxiv_search(_make_state(), config={})
        assert result["search_results"] == []

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_results(self, mocker):
        _mock_http(mocker, "", status_error=True)
        result = await arxiv_search(_make_state(), config={})
        assert result == {"search_results": []}

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_results(self, mocker):
        mock_client = mocker.AsyncMock()
        mock_client.__aenter__.return_value.get = mocker.AsyncMock(
            side_effect=httpx.ReadTimeout("timed out")
        )
        mocker.patch("mara.agent.nodes.search_worker.arxiv_search.httpx.AsyncClient", return_value=mock_client)
        result = await arxiv_search(_make_state(), config={})
        assert result == {"search_results": []}
