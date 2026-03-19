"""Tests for mara.agent.state — TypedDict definitions and MARAState reducers.

These tests verify that the state schema is correctly structured and that the
Annotated[list[SourceChunk], operator.add] reducer on raw_chunks correctly
merges parallel worker outputs as LangGraph requires.
"""

import operator

import pytest

from mara.agent.state import (
    MARAState,
    SearchResult,
    SearchWorkerState,
    SourceChunk,
    SubQuery,
)
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# TypedDict field access
# ---------------------------------------------------------------------------


class TestSubQuery:
    def test_fields_accessible(self):
        sq = SubQuery(query="climate change effects", domain="environmental")
        assert sq["query"] == "climate change effects"
        assert sq["domain"] == "environmental"

    def test_is_dict_subtype(self):
        sq = SubQuery(query="q", domain="d")
        assert isinstance(sq, dict)


class TestSearchResult:
    def test_fields_accessible(self):
        sr = SearchResult(url="https://example.com", title="Example", description="A page")
        assert sr["url"] == "https://example.com"
        assert sr["title"] == "Example"
        assert sr["description"] == "A page"


class TestSourceChunk:
    def test_fields_accessible(self):
        chunk = SourceChunk(
            url="https://example.com",
            text="Some text content",
            retrieved_at="2026-03-19T10:00:00Z",
            sub_query="test query",
        )
        assert chunk["url"] == "https://example.com"
        assert chunk["text"] == "Some text content"
        assert chunk["retrieved_at"] == "2026-03-19T10:00:00Z"
        assert chunk["sub_query"] == "test query"


# ---------------------------------------------------------------------------
# SearchWorkerState
# ---------------------------------------------------------------------------


class TestSearchWorkerState:
    def test_fields_accessible(self):
        state = SearchWorkerState(
            sub_query=SubQuery(query="q", domain="d"),
            research_config=ResearchConfig(),
            search_results=[],
            raw_chunks=[],
        )
        assert state["sub_query"]["query"] == "q"
        assert isinstance(state["research_config"], ResearchConfig)
        assert state["search_results"] == []
        assert state["raw_chunks"] == []


# ---------------------------------------------------------------------------
# MARAState raw_chunks reducer (operator.add)
# ---------------------------------------------------------------------------


class TestMARAStateReducer:
    """The Annotated[list[SourceChunk], operator.add] reducer must merge lists."""

    def test_raw_chunks_reducer_is_operator_add(self):
        """Inspect the annotation to confirm the reducer is operator.add."""
        hints = MARAState.__annotations__
        annotation = hints["raw_chunks"]
        # Annotated[list[SourceChunk], operator.add]
        # __metadata__[0] is the reducer
        metadata = getattr(annotation, "__metadata__", ())
        assert metadata[0] is operator.add

    def test_reducer_merges_two_lists(self):
        """Simulate what LangGraph does: reducer(old, new) for fan-in."""
        chunk_a = SourceChunk(
            url="https://a.com", text="chunk a", retrieved_at="2026-01-01T00:00:00Z", sub_query="q1"
        )
        chunk_b = SourceChunk(
            url="https://b.com", text="chunk b", retrieved_at="2026-01-01T00:00:00Z", sub_query="q2"
        )
        merged = operator.add([chunk_a], [chunk_b])
        assert len(merged) == 2
        assert merged[0]["url"] == "https://a.com"
        assert merged[1]["url"] == "https://b.com"

    def test_reducer_empty_plus_nonempty(self):
        chunk = SourceChunk(
            url="https://a.com", text="t", retrieved_at="2026-01-01T00:00:00Z", sub_query="q"
        )
        assert operator.add([], [chunk]) == [chunk]

    def test_reducer_nonempty_plus_empty(self):
        chunk = SourceChunk(
            url="https://a.com", text="t", retrieved_at="2026-01-01T00:00:00Z", sub_query="q"
        )
        assert operator.add([chunk], []) == [chunk]

    def test_reducer_empty_plus_empty(self):
        assert operator.add([], []) == []

    def test_reducer_three_workers_merged_in_sequence(self):
        """Simulate three parallel workers each returning two chunks."""
        make_chunk = lambda i: SourceChunk(
            url=f"https://src{i}.com",
            text=f"chunk {i}",
            retrieved_at="2026-01-01T00:00:00Z",
            sub_query="q",
        )
        w1 = [make_chunk(1), make_chunk(2)]
        w2 = [make_chunk(3), make_chunk(4)]
        w3 = [make_chunk(5), make_chunk(6)]
        combined = operator.add(operator.add(w1, w2), w3)
        assert len(combined) == 6
        urls = [c["url"] for c in combined]
        assert "https://src1.com" in urls
        assert "https://src6.com" in urls
