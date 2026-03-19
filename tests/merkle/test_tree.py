"""Tests for mara.merkle.tree — MerkleTree construction and combine_hashes.

These tests verify structural correctness, odd-leaf padding, algorithm
propagation, and exact root hashes against manually computed references.
"""

import hashlib

import pytest

from mara.merkle.tree import MerkleTree, build_merkle_tree, combine_hashes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    """SHA-256 of a UTF-8 string — replicates combine_hashes logic."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _combine(left: str, right: str) -> str:
    return _sha256(left + right)


def _make_leaves(n: int) -> list[str]:
    """Generate n deterministic fake leaf hashes."""
    return [hashlib.sha256(f"leaf-{i}".encode()).hexdigest() for i in range(n)]


# ---------------------------------------------------------------------------
# combine_hashes
# ---------------------------------------------------------------------------

class TestCombineHashes:
    def test_returns_hex_string(self):
        h1, h2 = _make_leaves(2)
        result = combine_hashes(h1, h2, "sha256")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_matches_manual_computation(self):
        h1, h2 = _make_leaves(2)
        expected = _sha256(h1 + h2)
        assert combine_hashes(h1, h2, "sha256") == expected

    def test_not_commutative(self):
        """combine_hashes(a, b) != combine_hashes(b, a) — order matters."""
        h1, h2 = _make_leaves(2)
        assert combine_hashes(h1, h2, "sha256") != combine_hashes(h2, h1, "sha256")

    def test_sha512_produces_128_char_digest(self):
        h1, h2 = _make_leaves(2)
        result = combine_hashes(h1, h2, "sha512")
        assert len(result) == 128

    def test_different_algorithms_produce_different_results(self):
        h1, h2 = _make_leaves(2)
        r256 = combine_hashes(h1, h2, "sha256")
        r512 = combine_hashes(h1, h2, "sha512")
        assert r256 != r512

    def test_invalid_algorithm_raises(self):
        h1, h2 = _make_leaves(2)
        with pytest.raises(ValueError):
            combine_hashes(h1, h2, "not_real")


# ---------------------------------------------------------------------------
# build_merkle_tree — empty / edge cases
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeEmpty:
    def test_empty_input_returns_empty_tree(self):
        tree = build_merkle_tree([], "sha256")
        assert tree.leaves == []
        assert tree.levels == []
        assert tree.root == ""

    def test_empty_tree_stores_algorithm(self):
        tree = build_merkle_tree([], "sha256")
        assert tree.algorithm == "sha256"

    def test_empty_hash_raises(self):
        with pytest.raises(ValueError, match="empty"):
            build_merkle_tree([""], "sha256")

    def test_empty_hash_among_valid_raises(self):
        leaves = _make_leaves(2) + [""]
        with pytest.raises(ValueError, match="empty"):
            build_merkle_tree(leaves, "sha256")


# ---------------------------------------------------------------------------
# build_merkle_tree — single leaf
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeSingleLeaf:
    def test_root_equals_leaf(self):
        leaves = _make_leaves(1)
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.root == leaves[0]

    def test_levels_contains_one_level(self):
        leaves = _make_leaves(1)
        tree = build_merkle_tree(leaves, "sha256")
        assert len(tree.levels) == 1
        assert tree.levels[0] == leaves

    def test_leaves_preserved(self):
        leaves = _make_leaves(1)
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.leaves == leaves


# ---------------------------------------------------------------------------
# build_merkle_tree — two leaves (smallest non-trivial tree)
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeTwoLeaves:
    def test_root_is_combine_of_both_leaves(self):
        leaves = _make_leaves(2)
        tree = build_merkle_tree(leaves, "sha256")
        expected_root = _combine(leaves[0], leaves[1])
        assert tree.root == expected_root

    def test_levels_structure(self):
        leaves = _make_leaves(2)
        tree = build_merkle_tree(leaves, "sha256")
        assert len(tree.levels) == 2
        assert tree.levels[0] == leaves
        assert tree.levels[1] == [tree.root]

    def test_leaves_preserved(self):
        leaves = _make_leaves(2)
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.leaves == leaves


# ---------------------------------------------------------------------------
# build_merkle_tree — three leaves (odd: last leaf duplicated)
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeThreeLeaves:
    def test_root_computed_with_last_leaf_duplicated(self):
        h1, h2, h3 = _make_leaves(3)
        tree = build_merkle_tree([h1, h2, h3], "sha256")
        p12 = _combine(h1, h2)
        p33 = _combine(h3, h3)   # h3 is duplicated to pad to even
        expected_root = _combine(p12, p33)
        assert tree.root == expected_root

    def test_levels_structure(self):
        leaves = _make_leaves(3)
        tree = build_merkle_tree(leaves, "sha256")
        # levels[0] = original 3 leaves (unpadded)
        # levels[1] = 2 parent nodes
        # levels[2] = root
        assert len(tree.levels) == 3
        assert len(tree.levels[0]) == 3
        assert len(tree.levels[1]) == 2
        assert len(tree.levels[2]) == 1

    def test_original_leaves_not_mutated(self):
        """The stored leaves must be the originals, not the padded version."""
        leaves = _make_leaves(3)
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.leaves == leaves
        assert len(tree.leaves) == 3


# ---------------------------------------------------------------------------
# build_merkle_tree — four leaves (power of two, no padding needed)
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeFourLeaves:
    def test_root_computed_correctly(self):
        h1, h2, h3, h4 = _make_leaves(4)
        tree = build_merkle_tree([h1, h2, h3, h4], "sha256")
        p12 = _combine(h1, h2)
        p34 = _combine(h3, h4)
        expected_root = _combine(p12, p34)
        assert tree.root == expected_root

    def test_levels_structure(self):
        leaves = _make_leaves(4)
        tree = build_merkle_tree(leaves, "sha256")
        assert len(tree.levels) == 3
        assert len(tree.levels[0]) == 4
        assert len(tree.levels[1]) == 2
        assert len(tree.levels[2]) == 1


# ---------------------------------------------------------------------------
# build_merkle_tree — five leaves (odd, non-leaf level also becomes odd)
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeFiveLeaves:
    def test_root_computed_correctly(self):
        h = _make_leaves(5)
        tree = build_merkle_tree(h, "sha256")
        # Level 1: pad [h0..h4] to [h0..h4, h4]
        p01 = _combine(h[0], h[1])
        p23 = _combine(h[2], h[3])
        p44 = _combine(h[4], h[4])
        # Level 2: [p01, p23, p44] is odd, pad to [p01, p23, p44, p44]
        p0123 = _combine(p01, p23)
        p4444 = _combine(p44, p44)
        expected_root = _combine(p0123, p4444)
        assert tree.root == expected_root

    def test_levels_count(self):
        leaves = _make_leaves(5)
        tree = build_merkle_tree(leaves, "sha256")
        # levels: [5 leaves] → [3 nodes] → [2 nodes] → [1 root]
        assert len(tree.levels) == 4

    def test_leaves_preserved(self):
        leaves = _make_leaves(5)
        tree = build_merkle_tree(leaves, "sha256")
        assert tree.leaves == leaves
        assert len(tree.leaves) == 5


# ---------------------------------------------------------------------------
# build_merkle_tree — eight leaves (perfect binary tree)
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeEightLeaves:
    def test_levels_count(self):
        leaves = _make_leaves(8)
        tree = build_merkle_tree(leaves, "sha256")
        assert len(tree.levels) == 4  # 8 → 4 → 2 → 1

    def test_root_computed_correctly(self):
        h = _make_leaves(8)
        tree = build_merkle_tree(h, "sha256")
        p = [_combine(h[i], h[i + 1]) for i in range(0, 8, 2)]  # 4 nodes
        q = [_combine(p[i], p[i + 1]) for i in range(0, 4, 2)]  # 2 nodes
        expected_root = _combine(q[0], q[1])
        assert tree.root == expected_root


# ---------------------------------------------------------------------------
# build_merkle_tree — algorithm propagation
# ---------------------------------------------------------------------------

class TestBuildMerkleTreeAlgorithm:
    def test_algorithm_stored_on_tree(self):
        tree = build_merkle_tree(_make_leaves(2), "sha256")
        assert tree.algorithm == "sha256"

    def test_different_algorithms_produce_different_roots(self):
        leaves = _make_leaves(4)
        tree256 = build_merkle_tree(leaves, "sha256")
        tree512 = build_merkle_tree(leaves, "sha512")
        assert tree256.root != tree512.root

    def test_sha512_root_has_correct_length(self):
        leaves = _make_leaves(2)
        tree = build_merkle_tree(leaves, "sha512")
        assert len(tree.root) == 128

    def test_root_changes_if_any_leaf_changes(self):
        leaves = _make_leaves(4)
        tree1 = build_merkle_tree(leaves, "sha256")
        modified = leaves[:2] + [hashlib.sha256(b"different").hexdigest()] + leaves[3:]
        tree2 = build_merkle_tree(modified, "sha256")
        assert tree1.root != tree2.root


# ---------------------------------------------------------------------------
# MerkleTree dataclass defaults
# ---------------------------------------------------------------------------

class TestMerkleTreeDataclass:
    def test_default_empty_tree(self):
        tree = MerkleTree()
        assert tree.leaves == []
        assert tree.levels == []
        assert tree.root == ""
        assert tree.algorithm == "sha256"

    def test_leaves_not_shared_between_instances(self):
        """Default factory must create independent lists."""
        t1 = MerkleTree()
        t2 = MerkleTree()
        t1.leaves.append("x")
        assert t2.leaves == []
