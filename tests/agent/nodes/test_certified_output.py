"""Tests for mara.agent.nodes.certified_output.

certified_output is a pure synchronous node — no I/O, no mocking needed.
Tests verify CertifiedReport field population, fallback from
human_approved_claims to scored_claims, and empty-tree handling.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from mara.agent.nodes.certified_output import certified_output
from mara.agent.state import CertifiedReport
from mara.merkle.tree import build_merkle_tree


# ---------------------------------------------------------------------------
# Minimal ScoredClaim stand-in
# ---------------------------------------------------------------------------


@dataclass
class _SC:
    text: str
    confidence: float
    source_indices: list
    similarities: list = None

    def __post_init__(self):
        if self.similarities is None:
            self.similarities = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCertifiedOutputFields:
    def test_returns_certified_report_key(self, make_mara_state):
        result = certified_output(make_mara_state(), config={})
        assert "certified_report" in result

    def test_returns_certified_report_instance(self, make_mara_state):
        result = certified_output(make_mara_state(), config={})
        assert isinstance(result["certified_report"], CertifiedReport)

    def test_query_copied_from_state(self, make_mara_state):
        result = certified_output(make_mara_state(query="What is X?"), config={})
        assert result["certified_report"].query == "What is X?"

    def test_report_text_copied_from_report_draft(self, make_mara_state):
        result = certified_output(make_mara_state(report_draft="The research report."), config={})
        assert result["certified_report"].report_text == "The research report."

    def test_leaves_copied_from_state(self, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(index=0), make_merkle_leaf(index=1)]
        result = certified_output(make_mara_state(retrieved_leaves=leaves), config={})
        assert len(result["certified_report"].leaves) == 2

    def test_leaves_are_independent_copy(self, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(index=0)]
        result = certified_output(make_mara_state(retrieved_leaves=leaves), config={})
        assert result["certified_report"].leaves is not leaves

    def test_generated_at_is_iso8601(self, make_mara_state):
        result = certified_output(make_mara_state(), config={})
        ts = result["certified_report"].generated_at
        datetime.fromisoformat(ts)

    def test_generated_at_is_recent(self, make_mara_state):
        before = datetime.now(timezone.utc)
        result = certified_output(make_mara_state(), config={})
        after = datetime.now(timezone.utc)
        ts = datetime.fromisoformat(result["certified_report"].generated_at)
        assert before <= ts <= after


class TestMerkleRoot:
    def test_merkle_root_from_retrieved_leaves(self, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(index=0), make_merkle_leaf(index=1, url="https://b.com")]
        expected_root = build_merkle_tree([l["hash"] for l in leaves], "sha256").root
        result = certified_output(make_mara_state(retrieved_leaves=leaves), config={})
        assert result["certified_report"].merkle_root == expected_root

    def test_merkle_root_independent_of_full_corpus_tree(self, make_mara_state, make_merkle_leaf):
        retrieved = [make_merkle_leaf(index=0)]
        all_leaves = [
            make_merkle_leaf(index=0),
            make_merkle_leaf(index=1, url="https://b.com"),
            make_merkle_leaf(index=2, url="https://c.com"),
        ]
        corpus_tree = build_merkle_tree([l["hash"] for l in all_leaves], "sha256")
        result = certified_output(
            make_mara_state(merkle_leaves=all_leaves, retrieved_leaves=retrieved, merkle_tree=corpus_tree),
            config={},
        )
        expected_root = build_merkle_tree([retrieved[0]["hash"]], "sha256").root
        assert result["certified_report"].merkle_root == expected_root
        assert result["certified_report"].merkle_root != corpus_tree.root

    def test_empty_retrieved_leaves_root_is_empty_string(self, make_mara_state):
        result = certified_output(make_mara_state(retrieved_leaves=[]), config={})
        assert result["certified_report"].merkle_root == ""

    def test_hash_algorithm_stored_in_report(self, make_mara_state):
        result = certified_output(make_mara_state(), config={})
        assert result["certified_report"].hash_algorithm == "sha256"


class TestClaimsHandling:
    def test_prefers_human_approved_over_scored(self, make_mara_state):
        human = [_SC("human claim", 0.70, [])]
        scored = [_SC("scored claim", 0.90, [])]
        result = certified_output(
            make_mara_state(scored_claims=scored, human_approved_claims=human), config={}
        )
        texts = [c.text for c in result["certified_report"].scored_claims]
        assert "human claim" in texts
        assert "scored claim" not in texts

    def test_falls_back_to_scored_when_human_approved_empty(self, make_mara_state):
        scored = [_SC("scored claim", 0.90, [])]
        result = certified_output(make_mara_state(scored_claims=scored), config={})
        texts = [c.text for c in result["certified_report"].scored_claims]
        assert "scored claim" in texts

    def test_empty_claims_produces_empty_list(self, make_mara_state):
        result = certified_output(make_mara_state(), config={})
        assert result["certified_report"].scored_claims == []

    def test_scored_claims_are_independent_copy(self, make_mara_state):
        scored = [_SC("c", 0.9, [])]
        result = certified_output(make_mara_state(scored_claims=scored), config={})
        assert result["certified_report"].scored_claims is not scored


class TestCertifiedOutputDbIntegration:
    """Verify that certified_output calls leaf_repo.complete_run when injected."""

    def test_complete_run_called_when_repo_and_run_id_injected(self, mocker, make_mara_state):
        repo = mocker.MagicMock()
        config = {"configurable": {"leaf_repo": repo, "run_id": "run-42"}}
        certified_output(make_mara_state(), config=config)
        repo.complete_run.assert_called_once()

    def test_complete_run_called_with_run_id(self, mocker, make_mara_state):
        repo = mocker.MagicMock()
        config = {"configurable": {"leaf_repo": repo, "run_id": "run-abc"}}
        certified_output(make_mara_state(), config=config)
        call_args = repo.complete_run.call_args.args
        assert call_args[0] == "run-abc"

    def test_complete_run_not_called_when_repo_missing(self, mocker, make_mara_state):
        repo = mocker.MagicMock()
        config = {"configurable": {"run_id": "run-1"}}
        certified_output(make_mara_state(), config=config)
        repo.complete_run.assert_not_called()

    def test_complete_run_not_called_when_run_id_missing(self, mocker, make_mara_state):
        repo = mocker.MagicMock()
        config = {"configurable": {"leaf_repo": repo}}
        certified_output(make_mara_state(), config=config)
        repo.complete_run.assert_not_called()
