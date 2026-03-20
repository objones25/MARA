"""Tests for mara.agent.nodes.confidence_scorer.

All LLM calls are mocked. Tests cover:
  - _call_lsa: verdict normalisation, unknown-verdict fallback, whitespace handling
  - confidence_scorer: end-to-end with mocked make_llm and asyncio.to_thread,
    empty claims list, source index resolution from retrieved_leaves, return shape
"""

import asyncio
import pytest

from mara.agent.nodes.confidence_scorer import (
    _call_lsa,
    confidence_scorer,
)
from mara.agent.state import Claim, MARAState, MerkleLeaf
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(url: str, text: str, index: int) -> MerkleLeaf:
    digest = hash_chunk(url=url, text=text, retrieved_at="2026-03-19T10:00:00Z", algorithm="sha256")
    return MerkleLeaf(
        url=url,
        text=text,
        retrieved_at="2026-03-19T10:00:00Z",
        hash=digest,
        index=index,
        sub_query="test query",
        contextualized_text=text,
    )


def _make_state(
    claims: list[Claim] | None = None,
    leaves: list[MerkleLeaf] | None = None,
) -> MARAState:
    return MARAState(
        query="q",
        config=ResearchConfig(
            brave_api_key="x",
            firecrawl_api_key="x",
            hf_token="test-token",
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            lsa_model="Qwen/Qwen3-32B",
        ),
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=[],
        merkle_tree=None,
        retrieved_leaves=leaves or [],
        extracted_claims=claims or [],
        scored_claims=[],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


def _mock_lsa_response(mocker, verdict: str):
    """Return a mock ChatHuggingFace response with the given verdict as content."""
    mock_resp = mocker.MagicMock()
    mock_resp.content = verdict
    return mock_resp


# ---------------------------------------------------------------------------
# _call_lsa
# ---------------------------------------------------------------------------


class TestCallLsa:
    def _mock_llm(self, mocker, verdict: str):
        mock_llm = mocker.MagicMock()
        mock_llm.invoke.return_value = _mock_lsa_response(mocker, verdict)
        return mock_llm

    def test_returns_supported(self, mocker):
        llm = self._mock_llm(mocker, "supported")
        assert _call_lsa(llm, "claim", ["source"]) == "supported"

    def test_returns_partially_supported(self, mocker):
        llm = self._mock_llm(mocker, "partially_supported")
        assert _call_lsa(llm, "claim", ["source"]) == "partially_supported"

    def test_returns_unsupported(self, mocker):
        llm = self._mock_llm(mocker, "unsupported")
        assert _call_lsa(llm, "claim", ["source"]) == "unsupported"

    def test_unknown_verdict_defaults_to_unsupported(self, mocker):
        llm = self._mock_llm(mocker, "i am confused")
        assert _call_lsa(llm, "claim", ["source"]) == "unsupported"

    def test_strips_whitespace_from_verdict(self, mocker):
        llm = self._mock_llm(mocker, "  supported  \n")
        assert _call_lsa(llm, "claim", []) == "supported"

    def test_case_insensitive_normalisation(self, mocker):
        llm = self._mock_llm(mocker, "SUPPORTED")
        assert _call_lsa(llm, "claim", []) == "supported"

    def test_invokes_llm_with_system_and_user_messages(self, mocker):
        llm = self._mock_llm(mocker, "supported")
        _call_lsa(llm, "my claim", ["passage one"])
        messages = llm.invoke.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_contains_claim(self, mocker):
        llm = self._mock_llm(mocker, "supported")
        _call_lsa(llm, "the sky is blue", ["some source"])
        messages = llm.invoke.call_args.args[0]
        assert "the sky is blue" in messages[1]["content"]

    def test_user_message_contains_source_text(self, mocker):
        llm = self._mock_llm(mocker, "unsupported")
        _call_lsa(llm, "claim", ["unique passage xyz"])
        messages = llm.invoke.call_args.args[0]
        assert "unique passage xyz" in messages[1]["content"]

    def test_empty_sources_handled(self, mocker):
        llm = self._mock_llm(mocker, "unsupported")
        result = _call_lsa(llm, "claim", [])
        assert result == "unsupported"


# ---------------------------------------------------------------------------
# confidence_scorer — async node
# ---------------------------------------------------------------------------


class TestConfidenceScorerNode:
    def _mock_llm_factory(self, mocker, verdict: str = "supported"):
        """Patch _make_llm to return a fake LLM that always returns verdict."""
        mock_llm = mocker.MagicMock()
        mock_llm.invoke.return_value = _mock_lsa_response(mocker, verdict)
        mocker.patch(
            "mara.agent.nodes.confidence_scorer.make_llm",
            return_value=mock_llm,
        )
        return mock_llm

    async def test_empty_claims_returns_empty_scored(self, mocker):
        self._mock_llm_factory(mocker)
        result = await confidence_scorer(_make_state(claims=[], leaves=[]), config={})
        assert result == {"scored_claims": []}

    async def test_returns_scored_claims_key(self, mocker):
        self._mock_llm_factory(mocker)
        leaf = _make_leaf("https://a.com", "text", 0)
        claim = Claim(text="some claim", source_indices=[0])
        result = await confidence_scorer(_make_state([claim], [leaf]), config={})
        assert "scored_claims" in result

    async def test_one_claim_produces_one_scored_claim(self, mocker):
        self._mock_llm_factory(mocker)
        leaf = _make_leaf("https://a.com", "supporting text", 0)
        claim = Claim(text="test claim", source_indices=[0])
        result = await confidence_scorer(_make_state([claim], [leaf]), config={})
        assert len(result["scored_claims"]) == 1

    async def test_multiple_claims_produce_multiple_scored_claims(self, mocker):
        self._mock_llm_factory(mocker)
        leaves = [_make_leaf(f"https://a.com/{i}", f"text {i}", i) for i in range(3)]
        claims = [Claim(text=f"claim {i}", source_indices=[i]) for i in range(3)]
        result = await confidence_scorer(_make_state(claims, leaves), config={})
        assert len(result["scored_claims"]) == 3

    async def test_scored_claim_has_text(self, mocker):
        self._mock_llm_factory(mocker)
        leaf = _make_leaf("https://a.com", "text", 0)
        claim = Claim(text="verifiable claim", source_indices=[0])
        result = await confidence_scorer(_make_state([claim], [leaf]), config={})
        assert result["scored_claims"][0].text == "verifiable claim"

    async def test_scored_claim_has_confidence_in_range(self, mocker):
        self._mock_llm_factory(mocker, verdict="supported")
        leaf = _make_leaf("https://a.com", "supporting evidence text", 0)
        claim = Claim(text="a well supported claim", source_indices=[0])
        result = await confidence_scorer(_make_state([claim], [leaf]), config={})
        score = result["scored_claims"][0].confidence
        assert 0.0 <= score <= 1.0

    async def test_source_texts_resolved_from_leaf_indices(self, mocker):
        mock_llm = self._mock_llm_factory(mocker, "supported")
        leaves = [
            _make_leaf("https://a.com/0", "text zero", 0),
            _make_leaf("https://a.com/1", "text one", 1),
            _make_leaf("https://a.com/2", "text two", 2),
        ]
        # Claim references indices 0 and 2 only
        claim = Claim(text="my claim", source_indices=[0, 2])
        await confidence_scorer(_make_state([claim], leaves), config={})
        # Check that the LLM received source texts from indices 0 and 2
        user_msg = mock_llm.invoke.call_args.args[0][1]["content"]
        assert "text zero" in user_msg
        assert "text two" in user_msg
        assert "text one" not in user_msg

    async def test_uses_lsa_model_and_hf_token(self, mocker):
        mock_make_llm = mocker.patch("mara.agent.nodes.confidence_scorer.make_llm")
        mock_llm = mocker.MagicMock()
        mock_llm.invoke.return_value = _mock_lsa_response(mocker, "supported")
        mock_make_llm.return_value = mock_llm

        cfg = ResearchConfig(
            hf_token="my-hf-token",
            brave_api_key="x",
            firecrawl_api_key="x",
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            lsa_model="Qwen/Qwen3-32B",
        )
        state = MARAState(
            query="q",
            config=cfg,
            sub_queries=[],
            search_results=[],
            raw_chunks=[],
            merkle_leaves=[],
            merkle_tree=None,
            retrieved_leaves=[],
            extracted_claims=[],
            scored_claims=[],
            human_approved_claims=[],
            report_draft="",
            certified_report=None,
            messages=[],
            loop_count=0,
        )
        await confidence_scorer(state, config={})
        mock_make_llm.assert_called_once_with("Qwen/Qwen3-32B", "my-hf-token", 32, "featherless-ai")

    async def test_claim_with_no_sources_still_scores(self, mocker):
        self._mock_llm_factory(mocker, "unsupported")
        claim = Claim(text="orphan claim", source_indices=[])
        result = await confidence_scorer(_make_state([claim], []), config={})
        assert len(result["scored_claims"]) == 1
        assert 0.0 <= result["scored_claims"][0].confidence <= 1.0
