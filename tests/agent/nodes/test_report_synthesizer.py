"""Tests for mara.agent.nodes.report_synthesizer.

All LLM calls are mocked. Tests cover:
  - _citation: format string for a single leaf
  - _format_claims: multi-claim formatting with confidence and citations
  - _make_llm: ChatAnthropic instantiation with max_tokens=8192
  - report_synthesizer: async node — empty claims path, populated path,
    human_approved_claims vs scored_claims fallback, config forwarding
"""

import pytest
from dataclasses import dataclass

from mara.agent.nodes.report_synthesizer import (
    _citation,
    _format_claims,
    _make_llm,
    report_synthesizer,
)
from mara.agent.state import MARAState, MerkleLeaf
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# Minimal ScoredClaim stand-in
# ---------------------------------------------------------------------------


@dataclass
class _SC:
    text: str
    confidence: float
    source_indices: list
    sa: float = 0.5
    csc: float = 0.5
    lsa: float = 0.5
    similarities: list = None

    def __post_init__(self):
        if self.similarities is None:
            self.similarities = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(index: int, url: str = "https://example.com", text: str = "t") -> MerkleLeaf:
    digest = hash_chunk(url, text, "2026-03-19T10:00:00Z", "sha256")
    return MerkleLeaf(
        url=url, text=text, retrieved_at="2026-03-19T10:00:00Z",
        hash=digest, index=index, sub_query="q", contextualized_text=text,
    )


def _make_state(
    scored: list | None = None,
    human_approved: list | None = None,
    leaves: list | None = None,
    query: str = "What is the impact of X?",
) -> MARAState:
    return MARAState(
        query=query,
        config=ResearchConfig(
            brave_api_key="x", firecrawl_api_key="x", anthropic_api_key="test-key",
        ),
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=[],
        merkle_tree=None,
        retrieved_leaves=leaves or [],
        extracted_claims=[],
        scored_claims=scored or [],
        human_approved_claims=human_approved or [],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


def _mock_llm(mocker, content: str = "The synthesised report."):
    mock_llm = mocker.AsyncMock()
    msg = mocker.MagicMock()
    msg.content = content
    mock_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("mara.agent.nodes.report_synthesizer._make_llm", return_value=mock_llm)
    return mock_llm


# ---------------------------------------------------------------------------
# _citation
# ---------------------------------------------------------------------------


class TestCitation:
    def test_format_ml_index_hash_prefix(self):
        leaf = _make_leaf(3, text="hello world")
        result = _citation(leaf)
        assert result.startswith("[ML:3:")
        assert result.endswith("]")

    def test_hash_prefix_is_six_chars(self):
        leaf = _make_leaf(0)
        tag = _citation(leaf)
        # format: [ML:0:xxxxxx]  — hash prefix between last : and ]
        prefix = tag.split(":")[-1][:-1]
        assert len(prefix) == 6

    def test_index_in_citation(self):
        leaf = _make_leaf(7)
        assert "[ML:7:" in _citation(leaf)


# ---------------------------------------------------------------------------
# _format_claims
# ---------------------------------------------------------------------------


class TestFormatClaims:
    def test_single_claim_with_citation(self):
        leaf = _make_leaf(0)
        claim = _SC("GDP rose 3%", 0.87, [0])
        result = _format_claims([claim], [leaf])
        assert "GDP rose 3%" in result
        assert "0.87" in result
        assert "[ML:0:" in result

    def test_claim_without_source_indices_has_no_citation(self):
        claim = _SC("orphan claim", 0.70, [])
        result = _format_claims([claim], [])
        assert "orphan claim" in result
        assert "[ML:" not in result

    def test_multiple_source_indices_produce_multiple_citations(self):
        leaves = [_make_leaf(0), _make_leaf(1)]
        claim = _SC("multi-source", 0.90, [0, 1])
        result = _format_claims([claim], leaves)
        assert result.count("[ML:") == 2

    def test_out_of_range_index_skipped(self):
        leaves = [_make_leaf(0)]
        claim = _SC("claim", 0.80, [0, 99])
        result = _format_claims([claim], leaves)
        assert result.count("[ML:") == 1

    def test_multiple_claims_each_on_own_line(self):
        leaves = [_make_leaf(0), _make_leaf(1)]
        claims = [_SC("claim A", 0.80, [0]), _SC("claim B", 0.90, [1])]
        result = _format_claims(claims, leaves)
        assert "claim A" in result
        assert "claim B" in result
        assert result.count("\n") >= 1

    def test_empty_claims_returns_empty_string(self):
        assert _format_claims([], []) == ""

    def test_confidence_formatted_to_two_decimals(self):
        leaf = _make_leaf(0)
        claim = _SC("c", 0.8765, [0])
        result = _format_claims([claim], [leaf])
        assert "0.88" in result


# ---------------------------------------------------------------------------
# _make_llm
# ---------------------------------------------------------------------------


class TestMakeLlm:
    def test_instantiates_chat_anthropic(self, mocker):
        mock_cls = mocker.patch("mara.agent.nodes.report_synthesizer.ChatAnthropic")
        _make_llm("claude-sonnet-4-6", "key")
        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-6", api_key="key", max_tokens=8192
        )

    def test_max_tokens_is_8192(self, mocker):
        mock_cls = mocker.patch("mara.agent.nodes.report_synthesizer.ChatAnthropic")
        _make_llm("m", "k")
        assert mock_cls.call_args.kwargs["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# report_synthesizer — async node
# ---------------------------------------------------------------------------


class TestReportSynthesizerNode:
    async def test_empty_claims_returns_empty_report(self, mocker):
        mocker.patch("mara.agent.nodes.report_synthesizer._make_llm")
        result = await report_synthesizer(_make_state(), config={})
        assert result == {"report_draft": ""}

    async def test_empty_claims_does_not_call_llm(self, mocker):
        mock_cls = mocker.patch("mara.agent.nodes.report_synthesizer._make_llm")
        await report_synthesizer(_make_state(), config={})
        mock_cls.assert_not_called()

    async def test_returns_report_draft_key(self, mocker):
        leaf = _make_leaf(0)
        _mock_llm(mocker, "Report text.")
        result = await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [0])], leaves=[leaf]), config={}
        )
        assert "report_draft" in result

    async def test_report_draft_is_llm_content(self, mocker):
        leaf = _make_leaf(0)
        _mock_llm(mocker, "The final research report.")
        result = await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [0])], leaves=[leaf]), config={}
        )
        assert result["report_draft"] == "The final research report."

    async def test_prefers_human_approved_over_scored(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = _mock_llm(mocker, "report")
        human = [_SC("human claim", 0.70, [0])]
        scored = [_SC("scored claim", 0.90, [0])]
        await report_synthesizer(
            _make_state(scored=scored, human_approved=human, leaves=[leaf]), config={}
        )
        user_msg = mock_llm.ainvoke.call_args.args[0][1]["content"]
        assert "human claim" in user_msg
        assert "scored claim" not in user_msg

    async def test_falls_back_to_scored_when_human_approved_empty(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = _mock_llm(mocker, "report")
        scored = [_SC("scored claim", 0.90, [0])]
        await report_synthesizer(
            _make_state(scored=scored, human_approved=[], leaves=[leaf]), config={}
        )
        user_msg = mock_llm.ainvoke.call_args.args[0][1]["content"]
        assert "scored claim" in user_msg

    async def test_query_included_in_user_message(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = _mock_llm(mocker, "report")
        await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [])], query="Effects of automation?"),
            config={},
        )
        user_msg = mock_llm.ainvoke.call_args.args[0][1]["content"]
        assert "Effects of automation?" in user_msg

    async def test_config_forwarded_to_ainvoke(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = _mock_llm(mocker, "report")
        sentinel = {"tags": ["test"]}
        await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [0])], leaves=[leaf]), config=sentinel
        )
        _, forwarded = mock_llm.ainvoke.call_args.args
        assert forwarded is sentinel

    async def test_llm_messages_have_system_and_user_roles(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = _mock_llm(mocker, "report")
        await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [0])], leaves=[leaf]), config={}
        )
        messages = mock_llm.ainvoke.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_leaf_citations_appear_in_user_message(self, mocker):
        leaf = _make_leaf(2)
        mock_llm = _mock_llm(mocker, "report")
        await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [2])], leaves=[leaf, leaf, leaf]),
            config={},
        )
        user_msg = mock_llm.ainvoke.call_args.args[0][1]["content"]
        assert "[ML:2:" in user_msg

    async def test_report_draft_is_stripped(self, mocker):
        leaf = _make_leaf(0)
        _mock_llm(mocker, "  Report with whitespace.  \n")
        result = await report_synthesizer(
            _make_state(scored=[_SC("c", 0.9, [])]), config={}
        )
        assert result["report_draft"] == "Report with whitespace."
