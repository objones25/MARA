"""Tests for mara.merkle.proof — generate_merkle_proof and verify_merkle_proof.

Proof correctness is the core of MARA's integrity guarantee. These tests cover:
- Every leaf in trees of every interesting size (1, 2, 3, 4, 5, 8 leaves)
- Exact proof path structure for manually verifiable cases
- Tamper detection: altered leaf, altered proof step, wrong root
- All error conditions
"""

import hashlib

import pytest

from mara.merkle.hasher import hash_chunk
from mara.merkle.proof import ProofStep, generate_merkle_proof, verify_merkle_proof
from mara.merkle.tree import build_merkle_tree, combine_hashes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leaves(n: int) -> list[str]:
    return [hashlib.sha256(f"leaf-{i}".encode()).hexdigest() for i in range(n)]


def _build(n: int, algorithm: str = "sha256"):
    return build_merkle_tree(_leaves(n), algorithm)


def _combine(left: str, right: str) -> str:
    return combine_hashes(left, right, "sha256")


# ---------------------------------------------------------------------------
# generate_merkle_proof — error conditions
# ---------------------------------------------------------------------------

class TestGenerateMerkleProofErrors:
    def test_empty_tree_raises_value_error(self):
        tree = build_merkle_tree([], "sha256")
        with pytest.raises(ValueError, match="empty"):
            generate_merkle_proof(tree, 0)

    def test_negative_index_raises_index_error(self):
        tree = _build(3)
        with pytest.raises(IndexError):
            generate_merkle_proof(tree, -1)

    def test_index_equal_to_len_raises_index_error(self):
        tree = _build(3)
        with pytest.raises(IndexError):
            generate_merkle_proof(tree, 3)

    def test_index_beyond_len_raises_index_error(self):
        tree = _build(3)
        with pytest.raises(IndexError):
            generate_merkle_proof(tree, 100)


# ---------------------------------------------------------------------------
# Single-leaf tree: proof is empty, root == leaf
# ---------------------------------------------------------------------------

class TestProofSingleLeaf:
    def test_proof_is_empty(self):
        tree = _build(1)
        proof = generate_merkle_proof(tree, 0)
        assert proof == []

    def test_verification_succeeds_with_empty_proof(self):
        leaves = _leaves(1)
        tree = build_merkle_tree(leaves, "sha256")
        proof = generate_merkle_proof(tree, 0)
        assert verify_merkle_proof(leaves[0], proof, tree.root, "sha256") is True

    def test_wrong_root_fails(self):
        leaves = _leaves(1)
        tree = build_merkle_tree(leaves, "sha256")
        proof = generate_merkle_proof(tree, 0)
        assert verify_merkle_proof(leaves[0], proof, "bad" * 16, "sha256") is False


# ---------------------------------------------------------------------------
# Two-leaf tree: smallest non-trivial proof
# ---------------------------------------------------------------------------

class TestProofTwoLeaves:
    def setup_method(self):
        self.leaves = _leaves(2)
        self.tree = build_merkle_tree(self.leaves, "sha256")

    def test_leaf0_proof_structure(self):
        proof = generate_merkle_proof(self.tree, 0)
        assert len(proof) == 1
        assert proof[0].sibling_hash == self.leaves[1]
        assert proof[0].position == "right"

    def test_leaf1_proof_structure(self):
        proof = generate_merkle_proof(self.tree, 1)
        assert len(proof) == 1
        assert proof[0].sibling_hash == self.leaves[0]
        assert proof[0].position == "left"

    def test_all_leaves_verify(self):
        for i in range(2):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(self.leaves[i], proof, self.tree.root, "sha256") is True

    def test_tampered_leaf_fails(self):
        proof = generate_merkle_proof(self.tree, 0)
        tampered = hashlib.sha256(b"tampered").hexdigest()
        assert verify_merkle_proof(tampered, proof, self.tree.root, "sha256") is False

    def test_tampered_proof_sibling_fails(self):
        proof = generate_merkle_proof(self.tree, 0)
        bad_proof = [ProofStep(sibling_hash="ab" * 32, position=proof[0].position)]
        assert verify_merkle_proof(self.leaves[0], bad_proof, self.tree.root, "sha256") is False

    def test_tampered_proof_position_fails(self):
        """Swapping left/right on a proof step must break verification."""
        proof = generate_merkle_proof(self.tree, 0)
        flipped = [ProofStep(sibling_hash=proof[0].sibling_hash, position="left")]
        assert verify_merkle_proof(self.leaves[0], flipped, self.tree.root, "sha256") is False


# ---------------------------------------------------------------------------
# Three-leaf tree: odd leaf count triggers duplication
# ---------------------------------------------------------------------------

class TestProofThreeLeaves:
    def setup_method(self):
        self.h = _leaves(3)
        self.tree = build_merkle_tree(self.h, "sha256")
        h = self.h
        self.p12 = _combine(h[0], h[1])
        self.p33 = _combine(h[2], h[2])  # h[2] duplicated
        self.expected_root = _combine(self.p12, self.p33)

    def test_root_matches_manual_computation(self):
        assert self.tree.root == self.expected_root

    def test_leaf0_proof_structure(self):
        proof = generate_merkle_proof(self.tree, 0)
        assert len(proof) == 2
        assert proof[0] == ProofStep(sibling_hash=self.h[1], position="right")
        assert proof[1] == ProofStep(sibling_hash=self.p33, position="right")

    def test_leaf1_proof_structure(self):
        proof = generate_merkle_proof(self.tree, 1)
        assert len(proof) == 2
        assert proof[0] == ProofStep(sibling_hash=self.h[0], position="left")
        assert proof[1] == ProofStep(sibling_hash=self.p33, position="right")

    def test_leaf2_proof_structure(self):
        """Leaf 2 is the duplicated leaf — its sibling at level 0 is itself."""
        proof = generate_merkle_proof(self.tree, 2)
        assert len(proof) == 2
        assert proof[0] == ProofStep(sibling_hash=self.h[2], position="right")
        assert proof[1] == ProofStep(sibling_hash=self.p12, position="left")

    def test_all_leaves_verify(self):
        for i in range(3):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(self.h[i], proof, self.tree.root, "sha256") is True

    def test_tampered_leaf_fails_for_all_positions(self):
        tampered = hashlib.sha256(b"tampered").hexdigest()
        for i in range(3):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(tampered, proof, self.tree.root, "sha256") is False


# ---------------------------------------------------------------------------
# Four-leaf tree: perfect binary tree, two-level proof
# ---------------------------------------------------------------------------

class TestProofFourLeaves:
    def setup_method(self):
        self.h = _leaves(4)
        self.tree = build_merkle_tree(self.h, "sha256")

    def test_all_leaves_verify(self):
        for i in range(4):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(self.h[i], proof, self.tree.root, "sha256") is True

    def test_proof_length_is_two(self):
        for i in range(4):
            proof = generate_merkle_proof(self.tree, i)
            assert len(proof) == 2

    def test_cross_leaf_proof_fails(self):
        """Proof generated for leaf i must not verify for leaf j."""
        proof_0 = generate_merkle_proof(self.tree, 0)
        assert verify_merkle_proof(self.h[1], proof_0, self.tree.root, "sha256") is False


# ---------------------------------------------------------------------------
# Five-leaf tree: non-leaf level is also odd (deeper duplication)
# ---------------------------------------------------------------------------

class TestProofFiveLeaves:
    def setup_method(self):
        self.h = _leaves(5)
        self.tree = build_merkle_tree(self.h, "sha256")

    def test_all_leaves_verify(self):
        for i in range(5):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(self.h[i], proof, self.tree.root, "sha256") is True

    def test_tampered_leaf_fails_for_all_positions(self):
        tampered = hashlib.sha256(b"tampered").hexdigest()
        for i in range(5):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(tampered, proof, self.tree.root, "sha256") is False


# ---------------------------------------------------------------------------
# Eight-leaf tree: perfect binary tree
# ---------------------------------------------------------------------------

class TestProofEightLeaves:
    def setup_method(self):
        self.h = _leaves(8)
        self.tree = build_merkle_tree(self.h, "sha256")

    def test_all_leaves_verify(self):
        for i in range(8):
            proof = generate_merkle_proof(self.tree, i)
            assert verify_merkle_proof(self.h[i], proof, self.tree.root, "sha256") is True

    def test_proof_length_is_three(self):
        for i in range(8):
            assert len(generate_merkle_proof(self.tree, i)) == 3


# ---------------------------------------------------------------------------
# verify_merkle_proof — wrong root / wrong algorithm
# ---------------------------------------------------------------------------

class TestVerifyMerkleProof:
    def test_wrong_root_returns_false(self):
        leaves = _leaves(4)
        tree = build_merkle_tree(leaves, "sha256")
        proof = generate_merkle_proof(tree, 0)
        assert verify_merkle_proof(leaves[0], proof, "0" * 64, "sha256") is False

    def test_wrong_algorithm_returns_false(self):
        """Verifying with a different algorithm than the tree was built with must fail."""
        leaves = _leaves(4)
        tree = build_merkle_tree(leaves, "sha256")
        proof = generate_merkle_proof(tree, 0)
        # sha512 root would be a different length entirely — never equal
        assert verify_merkle_proof(leaves[0], proof, tree.root, "sha512") is False

    def test_empty_proof_for_multi_leaf_tree_fails(self):
        """An empty proof should only be valid for a single-leaf tree."""
        leaves = _leaves(4)
        tree = build_merkle_tree(leaves, "sha256")
        assert verify_merkle_proof(leaves[0], [], tree.root, "sha256") is False

    def test_extra_proof_step_fails(self):
        """Adding a spurious step to a valid proof must break verification."""
        leaves = _leaves(2)
        tree = build_merkle_tree(leaves, "sha256")
        proof = generate_merkle_proof(tree, 0)
        extra_step = ProofStep(sibling_hash="aa" * 32, position="right")
        assert verify_merkle_proof(leaves[0], proof + [extra_step], tree.root, "sha256") is False


# ---------------------------------------------------------------------------
# End-to-end: hash_chunk → build_merkle_tree → proof → verify
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_real_chunks_full_round_trip(self):
        """Simulate the full MARA pipeline from source chunks to proof verification."""
        sources = [
            ("https://source-a.com", "Climate change accelerates ice loss.", "2026-01-01T00:00:00Z"),
            ("https://source-b.com", "Arctic temperatures rise faster than global average.", "2026-01-01T00:00:01Z"),
            ("https://source-c.com", "Sea level projections updated by IPCC.", "2026-01-01T00:00:02Z"),
        ]
        leaf_hashes = [hash_chunk(url, text, ts, "sha256") for url, text, ts in sources]
        tree = build_merkle_tree(leaf_hashes, "sha256")

        for i, (url, text, ts) in enumerate(sources):
            proof = generate_merkle_proof(tree, i)
            recomputed_hash = hash_chunk(url, text, ts, "sha256")
            assert verify_merkle_proof(recomputed_hash, proof, tree.root, "sha256") is True

    def test_altered_source_text_fails_verification(self):
        """A source that has changed since retrieval must fail the hash check."""
        url = "https://source.com"
        original_text = "Original content."
        altered_text = "Altered content."
        retrieved_at = "2026-01-01T00:00:00Z"

        original_hash = hash_chunk(url, original_text, retrieved_at, "sha256")
        tree = build_merkle_tree([original_hash], "sha256")
        proof = generate_merkle_proof(tree, 0)

        altered_hash = hash_chunk(url, altered_text, retrieved_at, "sha256")
        assert verify_merkle_proof(altered_hash, proof, tree.root, "sha256") is False
