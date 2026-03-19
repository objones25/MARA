"""Tests for mara.agent.nodes.hitl_checkpoint.

interrupt() is mocked so no LangGraph runtime is needed.

Tests cover:
  - All claims above threshold → no interrupt, all auto-approved
  - All claims below threshold → interrupt called with full review payload
  - Mixed → correct split between auto_approved and needs_review
  - Human approves a subset → only those indices merged with auto_approved
  - Human approves none (empty approved_indices) → only auto_approved returned
  - Out-of-range indices in approved_indices are silently ignored
  - Interrupt payload contains correct serialisable structure
"""

import pytest
from dataclasses import dataclass

from mara.agent.nodes.hitl_checkpoint import hitl_checkpoint
from mara.agent.state import MARAState
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Minimal ScoredClaim stand-in (mirrors mara.confidence.scorer.ScoredClaim)
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


def _make_state(
    scored_claims: list | None = None,
    high_threshold: float = 0.80,
) -> MARAState:
    return MARAState(
        query="q",
        config=ResearchConfig(
            brave_api_key="x",
            firecrawl_api_key="x",
            anthropic_api_key="x",
            high_confidence_threshold=high_threshold,
            low_confidence_threshold=0.55,
        ),
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=[],
        merkle_tree=None,
        extracted_claims=[],
        scored_claims=scored_claims or [],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


# ---------------------------------------------------------------------------
# All high-confidence — no interrupt
# ---------------------------------------------------------------------------


class TestAllAutoApproved:
    def test_no_interrupt_called(self, mocker):
        mock_interrupt = mocker.patch("mara.agent.nodes.hitl_checkpoint.interrupt")
        claims = [_SC("claim A", 0.90, [0]), _SC("claim B", 0.85, [1])]
        hitl_checkpoint(_make_state(claims), config={})
        mock_interrupt.assert_not_called()

    def test_all_claims_returned(self, mocker):
        mocker.patch("mara.agent.nodes.hitl_checkpoint.interrupt")
        claims = [_SC("claim A", 0.90, [0]), _SC("claim B", 0.85, [1])]
        result = hitl_checkpoint(_make_state(claims), config={})
        assert len(result["human_approved_claims"]) == 2

    def test_returns_human_approved_claims_key(self, mocker):
        mocker.patch("mara.agent.nodes.hitl_checkpoint.interrupt")
        result = hitl_checkpoint(_make_state([_SC("c", 0.95, [])]), config={})
        assert "human_approved_claims" in result

    def test_empty_scored_claims_returns_empty(self, mocker):
        mocker.patch("mara.agent.nodes.hitl_checkpoint.interrupt")
        result = hitl_checkpoint(_make_state([]), config={})
        assert result["human_approved_claims"] == []

    def test_threshold_boundary_auto_approves(self, mocker):
        mocker.patch("mara.agent.nodes.hitl_checkpoint.interrupt")
        # confidence == threshold → auto-approved (>= comparison)
        claim = _SC("boundary", 0.80, [])
        result = hitl_checkpoint(_make_state([claim], high_threshold=0.80), config={})
        assert len(result["human_approved_claims"]) == 1


# ---------------------------------------------------------------------------
# Claims below threshold — interrupt called
# ---------------------------------------------------------------------------


class TestInterruptCalled:
    def test_interrupt_called_for_low_confidence(self, mocker):
        mock_interrupt = mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("low claim", 0.50, [0])]
        hitl_checkpoint(_make_state(claims), config={})
        mock_interrupt.assert_called_once()

    def test_interrupt_payload_contains_needs_review(self, mocker):
        mock_interrupt = mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("low claim", 0.50, [0])]
        hitl_checkpoint(_make_state(claims), config={})
        payload = mock_interrupt.call_args.args[0]
        assert "needs_review" in payload

    def test_interrupt_payload_contains_auto_approved_count(self, mocker):
        mock_interrupt = mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("high", 0.90, []), _SC("low", 0.50, [])]
        hitl_checkpoint(_make_state(claims), config={})
        payload = mock_interrupt.call_args.args[0]
        assert payload["auto_approved_count"] == 1

    def test_needs_review_items_are_serialisable_dicts(self, mocker):
        mock_interrupt = mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("low claim", 0.60, [2])]
        hitl_checkpoint(_make_state(claims), config={})
        payload = mock_interrupt.call_args.args[0]
        item = payload["needs_review"][0]
        assert isinstance(item, dict)
        assert "text" in item
        assert "confidence" in item
        assert "source_indices" in item
        assert "index" in item

    def test_needs_review_item_index_is_position_in_list(self, mocker):
        mock_interrupt = mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("low A", 0.50, []), _SC("low B", 0.55, [])]
        hitl_checkpoint(_make_state(claims, high_threshold=0.80), config={})
        payload = mock_interrupt.call_args.args[0]
        assert payload["needs_review"][0]["index"] == 0
        assert payload["needs_review"][1]["index"] == 1


# ---------------------------------------------------------------------------
# Mixed — split between auto_approved and needs_review
# ---------------------------------------------------------------------------


class TestMixedClaims:
    def test_auto_approved_not_in_interrupt_payload(self, mocker):
        mock_interrupt = mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("high", 0.90, []), _SC("low", 0.50, [])]
        hitl_checkpoint(_make_state(claims), config={})
        payload = mock_interrupt.call_args.args[0]
        assert len(payload["needs_review"]) == 1
        assert payload["needs_review"][0]["text"] == "low"

    def test_human_approves_all_reviewed(self, mocker):
        mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": [0, 1]},
        )
        claims = [_SC("high", 0.90, []), _SC("low A", 0.50, []), _SC("low B", 0.60, [])]
        result = hitl_checkpoint(_make_state(claims), config={})
        # 1 auto-approved + 2 human-approved
        assert len(result["human_approved_claims"]) == 3

    def test_human_approves_subset(self, mocker):
        mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": [0]},  # approves only first reviewed
        )
        claims = [_SC("high", 0.90, []), _SC("low A", 0.50, []), _SC("low B", 0.60, [])]
        result = hitl_checkpoint(_make_state(claims), config={})
        # 1 auto-approved + 1 human-approved
        assert len(result["human_approved_claims"]) == 2

    def test_human_approves_none(self, mocker):
        mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": []},
        )
        claims = [_SC("high", 0.90, []), _SC("low", 0.50, [])]
        result = hitl_checkpoint(_make_state(claims), config={})
        # only auto_approved remains
        assert len(result["human_approved_claims"]) == 1
        assert result["human_approved_claims"][0].text == "high"

    def test_out_of_range_indices_ignored(self, mocker):
        mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={"approved_indices": [0, 99]},  # 99 is out of range
        )
        claims = [_SC("low", 0.50, [])]
        result = hitl_checkpoint(_make_state(claims), config={})
        # index 0 valid → 1 claim; index 99 silently ignored
        assert len(result["human_approved_claims"]) == 1

    def test_missing_approved_indices_key_treated_as_empty(self, mocker):
        mocker.patch(
            "mara.agent.nodes.hitl_checkpoint.interrupt",
            return_value={},  # no approved_indices key
        )
        claims = [_SC("high", 0.90, []), _SC("low", 0.50, [])]
        result = hitl_checkpoint(_make_state(claims), config={})
        assert len(result["human_approved_claims"]) == 1
