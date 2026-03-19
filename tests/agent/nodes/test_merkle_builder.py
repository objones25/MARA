"""Tests for mara.agent.nodes.merkle_builder.

merkle_builder is a pure synchronous node. Tests verify it correctly
delegates to build_merkle_tree with the right leaf hashes and algorithm,
handles empty input, and populates MARAState.merkle_tree correctly.
"""

import pytest

from mara.agent.nodes.merkle_builder import merkle_builder
from mara.agent.state import MARAState, MerkleLeaf
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import MerkleTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(
    url: str = "https://example.com",
    text: str = "sample text",
    retrieved_at: str = "2026-03-19T10:00:00Z",
    sub_query: str = "test query",
    index: int = 0,
    algorithm: str = "sha256",
) -> MerkleLeaf:
    digest = hash_chunk(url=url, text=text, retrieved_at=retrieved_at, algorithm=algorithm)
    return MerkleLeaf(
        url=url,
        text=text,
        retrieved_at=retrieved_at,
        hash=digest,
        index=index,
        sub_query=sub_query,
    )


def _make_state(
    leaves: list[MerkleLeaf] | None = None,
    algorithm: str = "sha256",
) -> MARAState:
    return MARAState(
        query="q",
        config=ResearchConfig(
            brave_api_key="x",
            firecrawl_api_key="x",
            anthropic_api_key="x",
            hash_algorithm=algorithm,
        ),
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=leaves or [],
        merkle_tree=None,
        extracted_claims=[],
        scored_claims=[],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMerkleBuilderEmptyInput:
    def test_empty_leaves_returns_empty_tree(self):
        result = merkle_builder(_make_state([]), config={})
        tree = result["merkle_tree"]
        assert tree.root == ""
        assert tree.leaves == []

    def test_returns_dict_with_merkle_tree_key(self):
        result = merkle_builder(_make_state([]), config={})
        assert "merkle_tree" in result

    def test_returns_merkle_tree_instance(self):
        result = merkle_builder(_make_state([]), config={})
        assert isinstance(result["merkle_tree"], MerkleTree)


class TestMerkleBuilderSingleLeaf:
    def test_root_equals_leaf_hash_for_single_leaf(self):
        leaf = _make_leaf()
        result = merkle_builder(_make_state([leaf]), config={})
        assert result["merkle_tree"].root == leaf["hash"]

    def test_tree_leaves_contains_leaf_hash(self):
        leaf = _make_leaf()
        result = merkle_builder(_make_state([leaf]), config={})
        assert result["merkle_tree"].leaves == [leaf["hash"]]


class TestMerkleBuilderMultipleLeaves:
    def test_tree_has_correct_leaf_count(self):
        leaves = [_make_leaf(url=f"https://example.com/{i}", index=i) for i in range(4)]
        result = merkle_builder(_make_state(leaves), config={})
        assert len(result["merkle_tree"].leaves) == 4

    def test_leaf_hash_order_preserved(self):
        leaves = [_make_leaf(url=f"https://example.com/{i}", text=f"t{i}", index=i) for i in range(3)]
        result = merkle_builder(_make_state(leaves), config={})
        assert result["merkle_tree"].leaves == [leaf["hash"] for leaf in leaves]

    def test_root_is_nonempty_string(self):
        leaves = [_make_leaf(url=f"https://example.com/{i}", index=i) for i in range(2)]
        result = merkle_builder(_make_state(leaves), config={})
        assert isinstance(result["merkle_tree"].root, str)
        assert len(result["merkle_tree"].root) > 0

    def test_algorithm_stored_on_tree(self):
        leaves = [_make_leaf(algorithm="sha256")]
        result = merkle_builder(_make_state(leaves, algorithm="sha256"), config={})
        assert result["merkle_tree"].algorithm == "sha256"

    def test_algorithm_forwarded_to_tree(self):
        leaf_256 = _make_leaf(algorithm="sha256")
        leaf_512 = _make_leaf(algorithm="sha512")
        r256 = merkle_builder(_make_state([leaf_256], algorithm="sha256"), config={})
        r512 = merkle_builder(_make_state([leaf_512], algorithm="sha512"), config={})
        assert r256["merkle_tree"].algorithm == "sha256"
        assert r512["merkle_tree"].algorithm == "sha512"

    def test_deterministic_root_for_same_leaves(self):
        leaves = [_make_leaf(url=f"https://example.com/{i}", index=i) for i in range(3)]
        r1 = merkle_builder(_make_state(leaves), config={})
        r2 = merkle_builder(_make_state(leaves), config={})
        assert r1["merkle_tree"].root == r2["merkle_tree"].root

    def test_different_leaves_produce_different_roots(self):
        leaves_a = [_make_leaf(text="text A")]
        leaves_b = [_make_leaf(text="text B")]
        r_a = merkle_builder(_make_state(leaves_a), config={})
        r_b = merkle_builder(_make_state(leaves_b), config={})
        assert r_a["merkle_tree"].root != r_b["merkle_tree"].root
