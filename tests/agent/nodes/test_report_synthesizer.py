"""Tests for mara.agent.nodes.report_synthesizer.

All LLM calls are mocked. Tests cover:
  - _citation: format string for a single leaf
  - _format_claims: multi-claim formatting with confidence and citations
  - report_synthesizer: async node — empty claims path, populated path,
    human_approved_claims vs scored_claims fallback, config forwarding
"""

import pytest
from dataclasses import dataclass

from mara.agent.nodes.report_synthesizer import (
    _citation,
    _format_claims,
    report_synthesizer,
)


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


def _mock_llm(mocker, content: str = "The synthesised report."):
    mock_llm = mocker.AsyncMock()
    msg = mocker.MagicMock()
    msg.content = content
    mock_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("mara.agent.nodes.report_synthesizer.make_llm", return_value=mock_llm)
    return mock_llm


# ---------------------------------------------------------------------------
# _citation
# ---------------------------------------------------------------------------


class TestCitation:
    def test_format_ml_index_hash_prefix(self, make_merkle_leaf):
        leaf = make_merkle_leaf(index=3, text="hello world")
        result = _citation(leaf)
        assert result.startswith("[ML:3:")
        assert result.endswith("]")

    def test_hash_prefix_is_six_chars(self, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        tag = _citation(leaf)
        # format: [ML:0:xxxxxx]  — hash prefix between last : and ]
        prefix = tag.split(":")[-1][:-1]
        assert len(prefix) == 6

    def test_index_in_citation(self, make_merkle_leaf):
        leaf = make_merkle_leaf(index=7)
        assert "[ML:7:" in _citation(leaf)


# ---------------------------------------------------------------------------
# _format_claims
# ---------------------------------------------------------------------------


class TestFormatClaims:
    def test_single_claim_with_citation(self, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
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

    def test_multiple_source_indices_produce_multiple_citations(self, make_merkle_leaf):
        leaves = [make_merkle_leaf(index=0), make_merkle_leaf(index=1, url="https://b.com")]
        claim = _SC("multi-source", 0.90, [0, 1])
        result = _format_claims([claim], leaves)
        assert result.count("[ML:") == 2

    def test_out_of_range_index_skipped(self, make_merkle_leaf):
        leaves = [make_merkle_leaf(index=0)]
        claim = _SC("claim", 0.80, [0, 99])
        result = _format_claims([claim], leaves)
        assert result.count("[ML:") == 1

    def test_multiple_claims_each_on_own_line(self, make_merkle_leaf):
        leaves = [make_merkle_leaf(index=0), make_merkle_leaf(index=1, url="https://b.com")]
        claims = [_SC("claim A", 0.80, [0]), _SC("claim B", 0.90, [1])]
        result = _format_claims(claims, leaves)
        assert "claim A" in result
        assert "claim B" in result
        assert result.count("\n") >= 1

    def test_empty_claims_returns_empty_string(self):
        assert _format_claims([], []) == ""

    def test_confidence_formatted_to_two_decimals(self, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        claim = _SC("c", 0.8765, [0])
        result = _format_claims([claim], [leaf])
        assert "0.88" in result


# ---------------------------------------------------------------------------
# report_synthesizer — async node
# ---------------------------------------------------------------------------


class TestReportSynthesizerNode:
    async def test_empty_claims_returns_empty_report(self, mocker, make_mara_state):
        mocker.patch("mara.agent.nodes.report_synthesizer.make_llm")
        result = await report_synthesizer(make_mara_state(), config={})
        assert result == {"report_draft": ""}

    async def test_empty_claims_does_not_call_llm(self, mocker, make_mara_state):
        mock_cls = mocker.patch("mara.agent.nodes.report_synthesizer.make_llm")
        await report_synthesizer(make_mara_state(), config={})
        mock_cls.assert_not_called()

    async def test_returns_report_draft_key(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        _mock_llm(mocker, "Report text.")
        result = await report_synthesizer(
            make_mara_state(scored_claims=[_SC("c", 0.9, [0])], retrieved_leaves=[leaf]), config={}
        )
        assert "report_draft" in result

    async def test_report_draft_is_llm_content(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        _mock_llm(mocker, "The final research report.")
        result = await report_synthesizer(
            make_mara_state(scored_claims=[_SC("c", 0.9, [0])], retrieved_leaves=[leaf]), config={}
        )
        assert result["report_draft"] == "The final research report."

    async def test_prefers_human_approved_over_scored(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        mock_llm_obj = _mock_llm(mocker, "report")
        human = [_SC("human claim", 0.70, [0])]
        scored = [_SC("scored claim", 0.90, [0])]
        await report_synthesizer(
            make_mara_state(scored_claims=scored, human_approved_claims=human, retrieved_leaves=[leaf]),
            config={},
        )
        user_msg = mock_llm_obj.ainvoke.call_args.args[0][1]["content"]
        assert "human claim" in user_msg
        assert "scored claim" not in user_msg

    async def test_falls_back_to_scored_when_human_approved_is_none(self, mocker, make_mara_state, make_merkle_leaf):
        """None sentinel means HITL never ran — fall back to scored_claims."""
        leaf = make_merkle_leaf(index=0)
        mock_llm_obj = _mock_llm(mocker, "report")
        scored = [_SC("scored claim", 0.90, [0])]
        await report_synthesizer(
            make_mara_state(scored_claims=scored, human_approved_claims=None, retrieved_leaves=[leaf]),
            config={},
        )
        user_msg = mock_llm_obj.ainvoke.call_args.args[0][1]["content"]
        assert "scored claim" in user_msg

    async def test_empty_human_approved_not_overridden_by_scored(self, mocker, make_mara_state, make_merkle_leaf):
        """Empty list means HITL ran and approved nothing — do not fall back."""
        mocker.patch("mara.agent.nodes.report_synthesizer.make_llm")
        scored = [_SC("scored claim", 0.90, [0])]
        result = await report_synthesizer(
            make_mara_state(scored_claims=scored, human_approved_claims=[], retrieved_leaves=[]),
            config={},
        )
        assert result == {"report_draft": ""}

    async def test_query_included_in_user_message(self, mocker, make_mara_state):
        mock_llm_obj = _mock_llm(mocker, "report")
        await report_synthesizer(
            make_mara_state(scored_claims=[_SC("c", 0.9, [])], query="Effects of automation?"),
            config={},
        )
        user_msg = mock_llm_obj.ainvoke.call_args.args[0][1]["content"]
        assert "Effects of automation?" in user_msg

    async def test_config_forwarded_to_ainvoke(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        mock_llm_obj = _mock_llm(mocker, "report")
        sentinel = {"tags": ["test"]}
        await report_synthesizer(
            make_mara_state(scored_claims=[_SC("c", 0.9, [0])], retrieved_leaves=[leaf]),
            config=sentinel,
        )
        _, forwarded = mock_llm_obj.ainvoke.call_args.args
        assert forwarded is sentinel

    async def test_llm_messages_have_system_and_user_roles(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(index=0)
        mock_llm_obj = _mock_llm(mocker, "report")
        await report_synthesizer(
            make_mara_state(scored_claims=[_SC("c", 0.9, [0])], retrieved_leaves=[leaf]), config={}
        )
        messages = mock_llm_obj.ainvoke.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_leaf_citations_appear_in_user_message(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(index=2)
        mock_llm_obj = _mock_llm(mocker, "report")
        await report_synthesizer(
            make_mara_state(
                scored_claims=[_SC("c", 0.9, [2])],
                retrieved_leaves=[leaf, leaf, leaf],
            ),
            config={},
        )
        user_msg = mock_llm_obj.ainvoke.call_args.args[0][1]["content"]
        assert "[ML:2:" in user_msg

    async def test_report_draft_is_stripped(self, mocker, make_mara_state):
        _mock_llm(mocker, "  Report with whitespace.  \n")
        result = await report_synthesizer(
            make_mara_state(scored_claims=[_SC("c", 0.9, [])]), config={}
        )
        assert result["report_draft"] == "Report with whitespace."
