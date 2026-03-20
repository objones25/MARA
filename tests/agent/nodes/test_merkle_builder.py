"""Tests for mara.agent.nodes.merkle_builder.

merkle_builder is a pure synchronous node. Tests verify it correctly
delegates to build_merkle_tree with the right leaf hashes and algorithm,
handles empty input, and populates MARAState.merkle_tree correctly.
"""

import pytest

from mara.agent.nodes.merkle_builder import merkle_builder
from mara.merkle.tree import MerkleTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder_state(make_mara_state):
    from mara.config import ResearchConfig

    def _factory(leaves=None, algorithm="sha256"):
        return make_mara_state(
            merkle_leaves=leaves or [],
            config=ResearchConfig(hash_algorithm=algorithm, leaf_db_enabled=False),
        )

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMerkleBuilderEmptyInput:
    def test_empty_leaves_returns_empty_tree(self, builder_state):
        result = merkle_builder(builder_state([]), config={})
        tree = result["merkle_tree"]
        assert tree.root == ""
        assert tree.leaves == []

    def test_returns_dict_with_merkle_tree_key(self, builder_state):
        result = merkle_builder(builder_state([]), config={})
        assert "merkle_tree" in result

    def test_returns_merkle_tree_instance(self, builder_state):
        result = merkle_builder(builder_state([]), config={})
        assert isinstance(result["merkle_tree"], MerkleTree)


class TestMerkleBuilderSingleLeaf:
    def test_root_equals_leaf_hash_for_single_leaf(self, builder_state, make_merkle_leaf):
        leaf = make_merkle_leaf()
        result = merkle_builder(builder_state([leaf]), config={})
        assert result["merkle_tree"].root == leaf["hash"]

    def test_tree_leaves_contains_leaf_hash(self, builder_state, make_merkle_leaf):
        leaf = make_merkle_leaf()
        result = merkle_builder(builder_state([leaf]), config={})
        assert result["merkle_tree"].leaves == [leaf["hash"]]


class TestMerkleBuilderMultipleLeaves:
    def test_tree_has_correct_leaf_count(self, builder_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://example.com/{i}", index=i) for i in range(4)]
        result = merkle_builder(builder_state(leaves), config={})
        assert len(result["merkle_tree"].leaves) == 4

    def test_leaf_hash_order_preserved(self, builder_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://example.com/{i}", text=f"t{i}", index=i) for i in range(3)]
        result = merkle_builder(builder_state(leaves), config={})
        assert result["merkle_tree"].leaves == [leaf["hash"] for leaf in leaves]

    def test_root_is_nonempty_string(self, builder_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://example.com/{i}", index=i) for i in range(2)]
        result = merkle_builder(builder_state(leaves), config={})
        assert isinstance(result["merkle_tree"].root, str)
        assert len(result["merkle_tree"].root) > 0

    def test_algorithm_stored_on_tree(self, builder_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(algorithm="sha256")]
        result = merkle_builder(builder_state(leaves, algorithm="sha256"), config={})
        assert result["merkle_tree"].algorithm == "sha256"

    def test_algorithm_forwarded_to_tree(self, builder_state, make_merkle_leaf):
        leaf_256 = make_merkle_leaf(algorithm="sha256")
        leaf_512 = make_merkle_leaf(algorithm="sha512")
        r256 = merkle_builder(builder_state([leaf_256], algorithm="sha256"), config={})
        r512 = merkle_builder(builder_state([leaf_512], algorithm="sha512"), config={})
        assert r256["merkle_tree"].algorithm == "sha256"
        assert r512["merkle_tree"].algorithm == "sha512"

    def test_deterministic_root_for_same_leaves(self, builder_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://example.com/{i}", index=i) for i in range(3)]
        r1 = merkle_builder(builder_state(leaves), config={})
        r2 = merkle_builder(builder_state(leaves), config={})
        assert r1["merkle_tree"].root == r2["merkle_tree"].root

    def test_different_leaves_produce_different_roots(self, builder_state, make_merkle_leaf):
        leaves_a = [make_merkle_leaf(text="text A")]
        leaves_b = [make_merkle_leaf(text="text B")]
        r_a = merkle_builder(builder_state(leaves_a), config={})
        r_b = merkle_builder(builder_state(leaves_b), config={})
        assert r_a["merkle_tree"].root != r_b["merkle_tree"].root
