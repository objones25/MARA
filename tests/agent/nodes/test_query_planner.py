"""Tests for mara.agent.nodes.query_planner.

All LLM calls are mocked — no real API calls are made.

Test strategy:
  - _parse_sub_queries: pure function, exhaustive branch coverage including
    clean JSON, fenced JSON, non-list responses, missing 'domain' key, and
    truncation to n.
  - query_planner: async node integration — mock make_llm to inject a fake
    LLM, verify the node reads state correctly and returns {"sub_queries": ...}.
"""

import json
import pytest

from mara.agent.nodes.query_planner import (
    _parse_sub_queries,
    query_planner,
)
from mara.agent.state import MARAState, SubQuery
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    query: str = "What are the health effects of ultra-processed food?",
    config: ResearchConfig | None = None,
) -> MARAState:
    cfg = config or ResearchConfig(
        hf_token="test-token",
        brave_api_key="x",
        firecrawl_api_key="x",
        max_workers=3,
    )
    return MARAState(
        query=query,
        config=cfg,
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=[],
        merkle_tree=None,
        extracted_claims=[],
        scored_claims=[],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


def _make_llm_response(mocker, content: str):
    """Return a mock that looks like a ChatHuggingFace response message."""
    mock_msg = mocker.MagicMock()
    mock_msg.content = content
    return mock_msg


def _sub_queries_json(n: int = 3) -> str:
    items = [
        {"query": f"query {i}", "domain": f"domain{i}"}
        for i in range(n)
    ]
    return json.dumps(items)


# ---------------------------------------------------------------------------
# _parse_sub_queries — pure function tests
# ---------------------------------------------------------------------------


class TestParseSubQueries:
    def test_clean_json_array(self):
        raw = '[{"query": "renewable energy policy", "domain": "regulatory"}]'
        result = _parse_sub_queries(raw, 3)
        assert len(result) == 1
        assert result[0]["query"] == "renewable energy policy"
        assert result[0]["domain"] == "regulatory"

    def test_missing_domain_defaults_to_general(self):
        raw = '[{"query": "some query"}]'
        result = _parse_sub_queries(raw, 3)
        assert result[0]["domain"] == "general"

    def test_truncates_to_n(self):
        raw = _sub_queries_json(n=5)
        result = _parse_sub_queries(raw, 3)
        assert len(result) == 3

    def test_returns_all_when_fewer_than_n(self):
        raw = _sub_queries_json(n=2)
        result = _parse_sub_queries(raw, 5)
        assert len(result) == 2

    def test_strips_json_fences(self):
        raw = '```json\n[{"query": "fenced query", "domain": "technical"}]\n```'
        result = _parse_sub_queries(raw, 3)
        assert result[0]["query"] == "fenced query"

    def test_strips_plain_fences(self):
        raw = '```\n[{"query": "plain fence", "domain": "empirical"}]\n```'
        result = _parse_sub_queries(raw, 3)
        assert result[0]["query"] == "plain fence"

    def test_query_coerced_to_str(self):
        raw = '[{"query": 42, "domain": "general"}]'
        result = _parse_sub_queries(raw, 3)
        assert result[0]["query"] == "42"

    def test_domain_coerced_to_str(self):
        raw = '[{"query": "q", "domain": 99}]'
        result = _parse_sub_queries(raw, 3)
        assert result[0]["domain"] == "99"

    def test_empty_array(self):
        result = _parse_sub_queries("[]", 3)
        assert result == []

    def test_returns_list_of_sub_query_typeddict(self):
        raw = '[{"query": "climate models", "domain": "empirical"}]'
        result = _parse_sub_queries(raw, 3)
        assert isinstance(result, list)
        assert "query" in result[0]
        assert "domain" in result[0]

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_sub_queries("not json", 3)

    def test_raises_on_non_list_json(self):
        with pytest.raises(ValueError, match="Expected a JSON array"):
            _parse_sub_queries('{"query": "oops", "domain": "x"}', 3)

    def test_whitespace_around_json(self):
        raw = '  \n[{"query": "q", "domain": "d"}]\n  '
        result = _parse_sub_queries(raw, 3)
        assert result[0]["query"] == "q"

    def test_exact_n_items(self):
        raw = _sub_queries_json(n=3)
        result = _parse_sub_queries(raw, 3)
        assert len(result) == 3

    def test_multiple_items_preserve_order(self):
        items = [{"query": f"q{i}", "domain": "d"} for i in range(3)]
        raw = json.dumps(items)
        result = _parse_sub_queries(raw, 3)
        assert [r["query"] for r in result] == ["q0", "q1", "q2"]


# ---------------------------------------------------------------------------
# query_planner — async node integration
# ---------------------------------------------------------------------------


class TestQueryPlannerNode:
    def _mock_llm(self, mocker, response_content: str):
        """Patch _make_llm to return a fake async LLM."""
        mock_llm = mocker.AsyncMock()
        mock_llm.ainvoke = mocker.AsyncMock(
            return_value=_make_llm_response(mocker, response_content)
        )
        mocker.patch(
            "mara.agent.nodes.query_planner.make_llm",
            return_value=mock_llm,
        )
        return mock_llm

    async def test_returns_sub_queries_key(self, mocker):
        self._mock_llm(mocker, _sub_queries_json(3))
        result = await query_planner(_make_state(), config={})
        assert "sub_queries" in result

    async def test_sub_queries_count_matches_max_workers(self, mocker):
        self._mock_llm(mocker, _sub_queries_json(3))
        state = _make_state()
        result = await query_planner(state, config={})
        assert len(result["sub_queries"]) == state["config"].max_workers

    async def test_sub_queries_are_list_of_sub_query(self, mocker):
        self._mock_llm(mocker, _sub_queries_json(3))
        result = await query_planner(_make_state(), config={})
        for sq in result["sub_queries"]:
            assert "query" in sq
            assert "domain" in sq

    async def test_uses_config_model(self, mocker):
        mock_make_llm = mocker.patch("mara.agent.nodes.query_planner.make_llm")
        mock_llm = mocker.AsyncMock()
        mock_llm.ainvoke = mocker.AsyncMock(
            return_value=_make_llm_response(mocker, _sub_queries_json(3))
        )
        mock_make_llm.return_value = mock_llm

        cfg = ResearchConfig(
            hf_token="my-hf-token",
            brave_api_key="x",
            firecrawl_api_key="x",
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            max_workers=3,
        )
        await query_planner(_make_state(config=cfg), config={})
        mock_make_llm.assert_called_once_with("Qwen/Qwen3-30B-A3B-Instruct-2507", "my-hf-token", 1024, "featherless-ai")

    async def test_uses_config_api_key(self, mocker):
        mock_make_llm = mocker.patch("mara.agent.nodes.query_planner.make_llm")
        mock_llm = mocker.AsyncMock()
        mock_llm.ainvoke = mocker.AsyncMock(
            return_value=_make_llm_response(mocker, _sub_queries_json(3))
        )
        mock_make_llm.return_value = mock_llm

        cfg = ResearchConfig(
            hf_token="secret-token",
            brave_api_key="x",
            firecrawl_api_key="x",
            max_workers=3,
        )
        await query_planner(_make_state(config=cfg), config={})
        _, called_api_key, _, _ = mock_make_llm.call_args.args
        assert called_api_key == "secret-token"

    async def test_invokes_llm_with_system_and_user_messages(self, mocker):
        mock_llm = self._mock_llm(mocker, _sub_queries_json(3))
        await query_planner(_make_state(), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user"]

    async def test_system_message_contains_system_prompt(self, mocker):
        from mara.prompts.query_planner import SYSTEM_PROMPT
        mock_llm = self._mock_llm(mocker, _sub_queries_json(3))
        await query_planner(_make_state(), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert messages[0]["content"] == SYSTEM_PROMPT

    async def test_user_message_contains_research_question(self, mocker):
        mock_llm = self._mock_llm(mocker, _sub_queries_json(3))
        state = _make_state(query="effects of sleep deprivation on cognition")
        await query_planner(state, config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert "effects of sleep deprivation on cognition" in messages[1]["content"]

    async def test_user_message_contains_n(self, mocker):
        mock_llm = self._mock_llm(mocker, _sub_queries_json(3))
        cfg = ResearchConfig(
            hf_token="k",
            brave_api_key="x",
            firecrawl_api_key="x",
            max_workers=3,
        )
        await query_planner(_make_state(config=cfg), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert "3" in messages[1]["content"]

    async def test_handles_fenced_llm_response(self, mocker):
        fenced = '```json\n' + _sub_queries_json(3) + '\n```'
        self._mock_llm(mocker, fenced)
        result = await query_planner(_make_state(), config={})
        assert len(result["sub_queries"]) == 3

    async def test_truncates_to_max_workers_when_llm_over_produces(self, mocker):
        self._mock_llm(mocker, _sub_queries_json(5))
        cfg = ResearchConfig(
            hf_token="k",
            brave_api_key="x",
            firecrawl_api_key="x",
            max_workers=3,
        )
        result = await query_planner(_make_state(config=cfg), config={})
        assert len(result["sub_queries"]) == 3

    async def test_returns_all_when_llm_under_produces(self, mocker):
        self._mock_llm(mocker, _sub_queries_json(2))
        cfg = ResearchConfig(
            hf_token="k",
            brave_api_key="x",
            firecrawl_api_key="x",
            max_workers=3,
        )
        result = await query_planner(_make_state(config=cfg), config={})
        assert len(result["sub_queries"]) == 2

    async def test_sub_query_fields_populated_from_llm(self, mocker):
        payload = '[{"query": "processed food inflammation study", "domain": "clinical"}]'
        cfg = ResearchConfig(
            hf_token="k",
            brave_api_key="x",
            firecrawl_api_key="x",
            max_workers=1,
        )
        self._mock_llm(mocker, payload)
        result = await query_planner(_make_state(config=cfg), config={})
        sq = result["sub_queries"][0]
        assert sq["query"] == "processed food inflammation study"
        assert sq["domain"] == "clinical"
