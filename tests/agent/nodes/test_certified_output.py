"""Tests for mara.agent.nodes.certified_output.

certified_output is a pure synchronous node — no I/O, no mocking needed.
Tests verify CertifiedReport field population, fallback from
human_approved_claims to scored_claims, and empty-tree handling.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone

from mara.agent.nodes.certified_output import certified_output
from mara.agent.state import CertifiedReport, MARAState, MerkleLeaf
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree


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


def _make_leaf(index: int) -> MerkleLeaf:
    url = f"https://example.com/{index}"
    text = f"text {index}"
    digest = hash_chunk(url, text, "2026-03-19T10:00:00Z", "sha256")
    return MerkleLeaf(
        url=url, text=text, retrieved_at="2026-03-19T10:00:00Z",
        hash=digest, index=index, sub_query="q",
    )


def _make_state(
    query: str = "test query",
    report_draft: str = "Report text.",
    leaves: list | None = None,
    tree=None,
    scored: list | None = None,
    human_approved: list | None = None,
) -> MARAState:
    return MARAState(
        query=query,
        config=ResearchConfig(
            brave_api_key="x", firecrawl_api_key="x", anthropic_api_key="x",
        ),
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=leaves or [],
        merkle_tree=tree,
        extracted_claims=[],
        scored_claims=scored or [],
        human_approved_claims=human_approved or [],
        report_draft=report_draft,
        certified_report=None,
        messages=[],
        loop_count=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCertifiedOutputFields:
    def test_returns_certified_report_key(self):
        result = certified_output(_make_state(), config={})
        assert "certified_report" in result

    def test_returns_certified_report_instance(self):
        result = certified_output(_make_state(), config={})
        assert isinstance(result["certified_report"], CertifiedReport)

    def test_query_copied_from_state(self):
        result = certified_output(_make_state(query="What is X?"), config={})
        assert result["certified_report"].query == "What is X?"

    def test_report_text_copied_from_report_draft(self):
        result = certified_output(_make_state(report_draft="The research report."), config={})
        assert result["certified_report"].report_text == "The research report."

    def test_leaves_copied_from_state(self):
        leaves = [_make_leaf(0), _make_leaf(1)]
        result = certified_output(_make_state(leaves=leaves), config={})
        assert len(result["certified_report"].leaves) == 2

    def test_leaves_are_independent_copy(self):
        leaves = [_make_leaf(0)]
        result = certified_output(_make_state(leaves=leaves), config={})
        # Mutating state leaves should not affect the report
        assert result["certified_report"].leaves is not leaves

    def test_generated_at_is_iso8601(self):
        result = certified_output(_make_state(), config={})
        ts = result["certified_report"].generated_at
        # Should be parseable as ISO-8601
        datetime.fromisoformat(ts)

    def test_generated_at_is_recent(self):
        before = datetime.now(timezone.utc)
        result = certified_output(_make_state(), config={})
        after = datetime.now(timezone.utc)
        ts = datetime.fromisoformat(result["certified_report"].generated_at)
        assert before <= ts <= after


class TestMerkleRoot:
    def test_merkle_root_from_tree(self):
        leaves = [_make_leaf(0)]
        tree = build_merkle_tree([leaves[0]["hash"]], "sha256")
        result = certified_output(_make_state(leaves=leaves, tree=tree), config={})
        assert result["certified_report"].merkle_root == tree.root

    def test_empty_tree_root_is_empty_string(self):
        result = certified_output(_make_state(tree=None), config={})
        assert result["certified_report"].merkle_root == ""


class TestClaimsHandling:
    def test_prefers_human_approved_over_scored(self):
        human = [_SC("human claim", 0.70, [])]
        scored = [_SC("scored claim", 0.90, [])]
        result = certified_output(
            _make_state(scored=scored, human_approved=human), config={}
        )
        texts = [c.text for c in result["certified_report"].scored_claims]
        assert "human claim" in texts
        assert "scored claim" not in texts

    def test_falls_back_to_scored_when_human_approved_empty(self):
        scored = [_SC("scored claim", 0.90, [])]
        result = certified_output(_make_state(scored=scored), config={})
        texts = [c.text for c in result["certified_report"].scored_claims]
        assert "scored claim" in texts

    def test_empty_claims_produces_empty_list(self):
        result = certified_output(_make_state(), config={})
        assert result["certified_report"].scored_claims == []

    def test_scored_claims_are_independent_copy(self):
        scored = [_SC("c", 0.9, [])]
        result = certified_output(_make_state(scored=scored), config={})
        assert result["certified_report"].scored_claims is not scored
