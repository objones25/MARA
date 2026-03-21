"""Tests for mara.agent.edges.routing.

Tests cover:
  - dispatch_search_workers: returns one Send per sub_query, correct target
    node name, correct payload shape (sub_query, research_config, empty lists)
  - route_after_scoring: routes to "corrective_retriever" when there are
    failing claims (low confidence, small n_leaves) and loop budget remains;
    routes to "hitl_checkpoint" otherwise (all approved, contested, loop cap).
"""

from dataclasses import dataclass

from langgraph.types import Send

from mara.agent.edges.routing import dispatch_search_workers, route_after_scoring
from mara.agent.state import SubQuery
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Minimal ScoredClaim stand-in
# ---------------------------------------------------------------------------


@dataclass
class _SC:
    text: str
    confidence: float
    source_indices: list
    corroborating: int = 0
    n_leaves: int = 0
    n_unique_urls: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_query(q: str = "test", domain: str = "general") -> SubQuery:
    return SubQuery(query=q, domain=domain)


# ---------------------------------------------------------------------------
# dispatch_search_workers
# ---------------------------------------------------------------------------


class TestDispatchSearchWorkers:
    def test_returns_list(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert isinstance(result, list)

    def test_empty_sub_queries_returns_empty(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[]))
        assert result == []

    def test_three_sends_per_sub_query(self, make_mara_state):
        # Each sub-query dispatches search_worker + arxiv_worker + semantic_scholar_worker.
        queries = [_sub_query("q1"), _sub_query("q2"), _sub_query("q3")]
        result = dispatch_search_workers(make_mara_state(sub_queries=queries))
        assert len(result) == 9  # 3 per query

    def test_single_sub_query_produces_three_sends(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query("only")]))
        assert len(result) == 3

    def test_each_item_is_send(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert all(isinstance(s, Send) for s in result)

    def test_first_send_targets_search_worker(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert result[0].node == "search_worker"

    def test_second_send_targets_arxiv_worker(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert result[1].node == "arxiv_worker"

    def test_third_send_targets_semantic_scholar_worker(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert result[2].node == "semantic_scholar_worker"

    def test_sends_ordered_per_sub_query(self, make_mara_state):
        # Sends are ordered: [sw_q1, ax_q1, ss_q1, sw_q2, ax_q2, ss_q2, ...]
        queries = [_sub_query("q1"), _sub_query("q2")]
        result = dispatch_search_workers(make_mara_state(sub_queries=queries))
        assert result[0].node == "search_worker"
        assert result[1].node == "arxiv_worker"
        assert result[2].node == "semantic_scholar_worker"
        assert result[3].node == "search_worker"
        assert result[4].node == "arxiv_worker"
        assert result[5].node == "semantic_scholar_worker"

    def test_send_payload_contains_sub_query(self, make_mara_state):
        q = _sub_query("robots in manufacturing", "economics")
        result = dispatch_search_workers(make_mara_state(sub_queries=[q]))
        assert result[0].arg["sub_query"] == q

    def test_arxiv_send_payload_contains_same_sub_query(self, make_mara_state):
        q = _sub_query("robots in manufacturing", "economics")
        result = dispatch_search_workers(make_mara_state(sub_queries=[q]))
        assert result[1].arg["sub_query"] == q

    def test_send_payload_contains_research_config(self, make_mara_state):
        state = make_mara_state(sub_queries=[_sub_query()])
        result = dispatch_search_workers(state)
        assert result[0].arg["research_config"] is state["config"]

    def test_send_payload_initialises_empty_search_results(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert result[0].arg["search_results"] == []

    def test_send_payload_initialises_empty_raw_chunks(self, make_mara_state):
        result = dispatch_search_workers(make_mara_state(sub_queries=[_sub_query()]))
        assert result[0].arg["raw_chunks"] == []

    def test_web_sends_preserve_sub_query_order(self, make_mara_state):
        # search_worker sends are at indices 0, 3, 6
        queries = [_sub_query("q1"), _sub_query("q2"), _sub_query("q3")]
        result = dispatch_search_workers(make_mara_state(sub_queries=queries))
        web_sends = [s for s in result if s.node == "search_worker"]
        for i, send in enumerate(web_sends):
            assert send.arg["sub_query"]["query"] == f"q{i + 1}"

    def test_arxiv_sends_preserve_sub_query_order(self, make_mara_state):
        # arxiv_worker sends are at indices 1, 4, 7
        queries = [_sub_query("q1"), _sub_query("q2"), _sub_query("q3")]
        result = dispatch_search_workers(make_mara_state(sub_queries=queries))
        arxiv_sends = [s for s in result if s.node == "arxiv_worker"]
        for i, send in enumerate(arxiv_sends):
            assert send.arg["sub_query"]["query"] == f"q{i + 1}"

    def test_semantic_scholar_sends_preserve_sub_query_order(self, make_mara_state):
        # semantic_scholar_worker sends are at indices 2, 5, 8
        queries = [_sub_query("q1"), _sub_query("q2"), _sub_query("q3")]
        result = dispatch_search_workers(make_mara_state(sub_queries=queries))
        ss_sends = [s for s in result if s.node == "semantic_scholar_worker"]
        for i, send in enumerate(ss_sends):
            assert send.arg["sub_query"]["query"] == f"q{i + 1}"

    def test_all_sends_share_same_config_reference(self, make_mara_state):
        queries = [_sub_query("q1"), _sub_query("q2")]
        state = make_mara_state(sub_queries=queries)
        result = dispatch_search_workers(state)
        configs = [s.arg["research_config"] for s in result]
        assert all(c is state["config"] for c in configs)


# ---------------------------------------------------------------------------
# route_after_scoring
# ---------------------------------------------------------------------------


class TestRouteAfterScoring:
    def test_returns_string(self, make_mara_state):
        assert isinstance(route_after_scoring(make_mara_state()), str)

    def test_empty_scored_claims_returns_hitl(self, make_mara_state):
        assert route_after_scoring(make_mara_state(scored_claims=[])) == "hitl_checkpoint"

    def test_high_confidence_claims_returns_hitl(self, make_mara_state):
        # All approved — no failing claims
        claims = [_SC("c", 0.95, [], n_leaves=5), _SC("d", 0.90, [], n_leaves=5)]
        assert route_after_scoring(make_mara_state(scored_claims=claims)) == "hitl_checkpoint"

    def test_failing_claims_within_loop_budget_returns_corrective(self, make_mara_state):
        # Low confidence, small n_leaves, loop_count < max (default max=2)
        claims = [_SC("c", 0.10, [], n_leaves=5)]
        state = make_mara_state(scored_claims=claims, loop_count=0)
        assert route_after_scoring(state) == "corrective_retriever"

    def test_failing_claims_at_loop_cap_returns_hitl(self, make_mara_state):
        # loop_count == max_corrective_rag_loops (2) → no budget left
        claims = [_SC("c", 0.10, [], n_leaves=5)]
        state = make_mara_state(scored_claims=claims, loop_count=2)
        assert route_after_scoring(state) == "hitl_checkpoint"

    def test_failing_claims_exceed_loop_cap_returns_hitl(self, make_mara_state):
        claims = [_SC("c", 0.10, [], n_leaves=5)]
        state = make_mara_state(scored_claims=claims, loop_count=99)
        assert route_after_scoring(state) == "hitl_checkpoint"

    def test_contested_claims_only_returns_hitl(self, make_mara_state):
        # corroborating >= n_leaves_contested_threshold (default=2) → contested, not failing
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_SC("c", 0.10, [], corroborating=cfg.n_leaves_contested_threshold)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        assert route_after_scoring(state) == "hitl_checkpoint"

    def test_mixed_failing_and_contested_returns_corrective(self, make_mara_state):
        # One contested (large corroborating) + one failing (small corroborating) → still has failing → corrective
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [
            _SC("contested", 0.10, [], corroborating=cfg.n_leaves_contested_threshold),
            _SC("failing", 0.10, [], corroborating=0),
        ]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        assert route_after_scoring(state) == "corrective_retriever"

    def test_mixed_high_and_failing_returns_corrective(self, make_mara_state):
        # One high-confidence + one failing → corrective fires
        claims = [_SC("high", 0.95, []), _SC("low", 0.20, [], corroborating=0)]
        state = make_mara_state(scored_claims=claims, loop_count=0)
        assert route_after_scoring(state) == "corrective_retriever"

    def test_custom_loop_budget_respected(self, make_mara_state):
        cfg = ResearchConfig(max_corrective_rag_loops=3, leaf_db_enabled=False)
        claims = [_SC("c", 0.10, [], corroborating=0)]
        # loop_count=2 < max=3 → corrective
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=2)
        assert route_after_scoring(state) == "corrective_retriever"
        # loop_count=3 == max=3 → hitl
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=3)
        assert route_after_scoring(state) == "hitl_checkpoint"

    def test_custom_n_leaves_contested_threshold(self, make_mara_state):
        cfg = ResearchConfig(n_leaves_contested_threshold=5, leaf_db_enabled=False)
        # corroborating=5 >= threshold=5 → contested, no failing
        claims = [_SC("c", 0.10, [], corroborating=5)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        assert route_after_scoring(state) == "hitl_checkpoint"
        # corroborating=4 < threshold=5 → failing
        claims = [_SC("c", 0.10, [], corroborating=4)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        assert route_after_scoring(state) == "corrective_retriever"

    def test_confidence_at_low_threshold_not_failing(self, make_mara_state):
        # confidence == low_confidence_threshold (0.55) → not failing (strict <)
        cfg = ResearchConfig(leaf_db_enabled=False)
        claims = [_SC("c", cfg.low_confidence_threshold, [], n_leaves=5)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        assert route_after_scoring(state) == "hitl_checkpoint"

    def test_loop_count_zero_budget_zero_returns_hitl(self, make_mara_state):
        # max_corrective_rag_loops=0 means no loops allowed
        cfg = ResearchConfig(max_corrective_rag_loops=0, leaf_db_enabled=False)
        claims = [_SC("c", 0.10, [], n_leaves=5)]
        state = make_mara_state(scored_claims=claims, config=cfg, loop_count=0)
        assert route_after_scoring(state) == "hitl_checkpoint"
