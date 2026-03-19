"""Tests for mara.agent.nodes.search_worker.graph.

Verifies the compiled subgraph's topology and end-to-end data flow by patching
the two node callables (brave_search and firecrawl_scrape) so that no network
calls are made.

Tests are deliberately lightweight: node-level logic is covered exhaustively in
test_brave_search.py and test_firecrawl_scrape.py.  This file focuses on:
  1. Graph topology — edge order, START/END wiring.
  2. End-to-end plumbing — state fields flow correctly through the pipeline.
  3. Node execution order — brave_search runs before firecrawl_scrape.
  4. Return shape — the compiled graph returns a dict with the expected keys.
"""

import pytest

from mara.agent.nodes.search_worker.graph import search_worker, _build_search_worker
from mara.agent.state import SearchWorkerState, SubQuery, SourceChunk
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_initial_state(query: str = "climate change") -> SearchWorkerState:
    return SearchWorkerState(
        sub_query=SubQuery(query=query, domain="environment"),
        research_config=ResearchConfig(),
        search_results=[],
        raw_chunks=[],
    )


def _make_source_chunk(url: str, text: str, sub_query: str) -> SourceChunk:
    return SourceChunk(
        url=url,
        text=text,
        retrieved_at="2026-03-19T10:00:00Z",
        sub_query=sub_query,
    )


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------


class TestSearchWorkerTopology:
    def test_compiled_graph_is_not_none(self):
        assert search_worker is not None

    def test_graph_has_brave_search_node(self):
        assert "brave_search" in search_worker.nodes

    def test_graph_has_firecrawl_scrape_node(self):
        assert "firecrawl_scrape" in search_worker.nodes

    def test_build_returns_new_compiled_graph(self):
        graph = _build_search_worker()
        assert graph is not None
        assert "brave_search" in graph.nodes
        assert "firecrawl_scrape" in graph.nodes


# ---------------------------------------------------------------------------
# End-to-end integration (nodes patched)
# ---------------------------------------------------------------------------


class TestSearchWorkerEndToEnd:
    async def test_graph_invokes_brave_search(self, mocker):
        call_log = []

        async def fake_brave(state, config):
            call_log.append("brave_search")
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            call_log.append("firecrawl_scrape")
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        await graph.ainvoke(_make_initial_state())
        assert "brave_search" in call_log

    async def test_graph_invokes_firecrawl_scrape(self, mocker):
        call_log = []

        async def fake_brave(state, config):
            call_log.append("brave_search")
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            call_log.append("firecrawl_scrape")
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        await graph.ainvoke(_make_initial_state())
        assert "firecrawl_scrape" in call_log

    async def test_brave_search_runs_before_firecrawl_scrape(self, mocker):
        call_log = []

        async def fake_brave(state, config):
            call_log.append("brave_search")
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            call_log.append("firecrawl_scrape")
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        await graph.ainvoke(_make_initial_state())
        assert call_log.index("brave_search") < call_log.index("firecrawl_scrape")

    async def test_output_contains_raw_chunks_key(self, mocker):
        async def fake_brave(state, config):
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        result = await graph.ainvoke(_make_initial_state())
        assert "raw_chunks" in result

    async def test_raw_chunks_from_firecrawl_present_in_output(self, mocker):
        expected_chunk = _make_source_chunk(
            "https://a.com", "chunk text", "climate change"
        )

        async def fake_brave(state, config):
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            return {"raw_chunks": [expected_chunk]}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        result = await graph.ainvoke(_make_initial_state())
        assert expected_chunk in result["raw_chunks"]

    async def test_search_results_from_brave_visible_to_firecrawl(self, mocker):
        """brave_search's search_results output must be in state when firecrawl runs."""
        received_state: dict = {}

        async def fake_brave(state, config):
            return {
                "search_results": [
                    {
                        "url": "https://relay.com",
                        "title": "T",
                        "description": "D",
                        "extra_snippets": [],
                        "page_age": "",
                        "result_type": "web",
                    }
                ]
            }

        async def fake_firecrawl(state, config):
            received_state["search_results"] = state["search_results"]
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        await graph.ainvoke(_make_initial_state())
        assert any(
            r["url"] == "https://relay.com"
            for r in received_state["search_results"]
        )

    async def test_sub_query_preserved_through_graph(self, mocker):
        received: dict = {}

        async def fake_brave(state, config):
            received["brave_query"] = state["sub_query"]["query"]
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            received["fc_query"] = state["sub_query"]["query"]
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        await graph.ainvoke(_make_initial_state(query="ocean acidification"))
        assert received["brave_query"] == "ocean acidification"
        assert received["fc_query"] == "ocean acidification"

    async def test_empty_pipeline_returns_empty_raw_chunks(self, mocker):
        async def fake_brave(state, config):
            return {"search_results": []}

        async def fake_firecrawl(state, config):
            return {"raw_chunks": []}

        mocker.patch(
            "mara.agent.nodes.search_worker.graph.brave_search", fake_brave
        )
        mocker.patch(
            "mara.agent.nodes.search_worker.graph.firecrawl_scrape", fake_firecrawl
        )
        graph = _build_search_worker()
        result = await graph.ainvoke(_make_initial_state())
        assert result["raw_chunks"] == []
