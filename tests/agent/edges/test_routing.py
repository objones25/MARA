"""Tests for mara.agent.edges.routing.

Tests cover:
  - dispatch_search_workers: returns one Send per sub_query, correct target
    node name, correct payload shape (sub_query, research_config, empty lists)
  - route_after_scoring: always returns "hitl_checkpoint" regardless of
    claim confidence distribution
"""

import pytest
from dataclasses import dataclass

from langgraph.types import Send

from mara.agent.edges.routing import dispatch_search_workers, route_after_scoring
from mara.agent.state import MARAState, SubQuery
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Minimal ScoredClaim stand-in
# ---------------------------------------------------------------------------


@dataclass
class _SC:
    text: str
    confidence: float
    source_indices: list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_query(q: str = "test", domain: str = "general") -> SubQuery:
    return SubQuery(query=q, domain=domain)


def _make_state(
    sub_queries: list[SubQuery] | None = None,
    scored_claims: list | None = None,
    loop_count: int = 0,
) -> MARAState:
    return MARAState(
        query="q",
        config=ResearchConfig(
            brave_api_key="x",
            firecrawl_api_key="x",
            anthropic_api_key="x",
        ),
        sub_queries=sub_queries or [],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=[],
        merkle_tree=None,
        retrieved_leaves=[],
        extracted_claims=[],
        scored_claims=scored_claims or [],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=loop_count,
    )


# ---------------------------------------------------------------------------
# dispatch_search_workers
# ---------------------------------------------------------------------------


class TestDispatchSearchWorkers:
    def test_returns_list(self):
        result = dispatch_search_workers(_make_state(sub_queries=[_sub_query()]))
        assert isinstance(result, list)

    def test_empty_sub_queries_returns_empty(self):
        result = dispatch_search_workers(_make_state(sub_queries=[]))
        assert result == []

    def test_one_send_per_sub_query(self):
        queries = [_sub_query("q1"), _sub_query("q2"), _sub_query("q3")]
        result = dispatch_search_workers(_make_state(sub_queries=queries))
        assert len(result) == 3

    def test_each_item_is_send(self):
        result = dispatch_search_workers(_make_state(sub_queries=[_sub_query()]))
        assert isinstance(result[0], Send)

    def test_send_targets_search_worker(self):
        result = dispatch_search_workers(_make_state(sub_queries=[_sub_query()]))
        assert result[0].node == "search_worker"

    def test_send_payload_contains_sub_query(self):
        q = _sub_query("robots in manufacturing", "economics")
        result = dispatch_search_workers(_make_state(sub_queries=[q]))
        assert result[0].arg["sub_query"] == q

    def test_send_payload_contains_research_config(self):
        state = _make_state(sub_queries=[_sub_query()])
        result = dispatch_search_workers(state)
        assert result[0].arg["research_config"] is state["config"]

    def test_send_payload_initialises_empty_search_results(self):
        result = dispatch_search_workers(_make_state(sub_queries=[_sub_query()]))
        assert result[0].arg["search_results"] == []

    def test_send_payload_initialises_empty_raw_chunks(self):
        result = dispatch_search_workers(_make_state(sub_queries=[_sub_query()]))
        assert result[0].arg["raw_chunks"] == []

    def test_sends_preserve_sub_query_order(self):
        queries = [_sub_query("q1"), _sub_query("q2"), _sub_query("q3")]
        result = dispatch_search_workers(_make_state(sub_queries=queries))
        for i, send in enumerate(result):
            assert send.arg["sub_query"]["query"] == f"q{i + 1}"

    def test_single_sub_query_produces_one_send(self):
        result = dispatch_search_workers(_make_state(sub_queries=[_sub_query("only")]))
        assert len(result) == 1
        assert result[0].arg["sub_query"]["query"] == "only"

    def test_all_sends_share_same_config_reference(self):
        queries = [_sub_query("q1"), _sub_query("q2")]
        state = _make_state(sub_queries=queries)
        result = dispatch_search_workers(state)
        assert result[0].arg["research_config"] is result[1].arg["research_config"]


# ---------------------------------------------------------------------------
# route_after_scoring
# ---------------------------------------------------------------------------


class TestRouteAfterScoring:
    def test_returns_string(self):
        assert isinstance(route_after_scoring(_make_state()), str)

    def test_returns_hitl_checkpoint(self):
        assert route_after_scoring(_make_state()) == "hitl_checkpoint"

    def test_empty_scored_claims_returns_hitl(self):
        assert route_after_scoring(_make_state(scored_claims=[])) == "hitl_checkpoint"

    def test_high_confidence_claims_returns_hitl(self):
        claims = [_SC("c", 0.95, []), _SC("d", 0.90, [])]
        assert route_after_scoring(_make_state(scored_claims=claims)) == "hitl_checkpoint"

    def test_low_confidence_claims_returns_hitl(self):
        claims = [_SC("c", 0.10, []), _SC("d", 0.20, [])]
        assert route_after_scoring(_make_state(scored_claims=claims)) == "hitl_checkpoint"

    def test_mixed_confidence_returns_hitl(self):
        claims = [_SC("high", 0.95, []), _SC("low", 0.20, [])]
        assert route_after_scoring(_make_state(scored_claims=claims)) == "hitl_checkpoint"

    def test_max_loops_reached_returns_hitl(self):
        claims = [_SC("c", 0.10, [])]
        state = _make_state(scored_claims=claims, loop_count=2)
        assert route_after_scoring(state) == "hitl_checkpoint"
