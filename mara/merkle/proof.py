"""Merkle proof generation and verification.

A Merkle proof for leaf i is the list of sibling hashes along the path from
the leaf to the root. A verifier recomputes the root from the leaf hash and
the proof path, then checks it against the known root.
"""

from dataclasses import dataclass

from mara.merkle.tree import MerkleTree, combine_hashes


@dataclass
class ProofStep:
    """One step in a Merkle proof path.

    Attributes:
        sibling_hash: Hash of the sibling node at this level.
        position:     'left' if the sibling is to the left of the current node,
                      'right' if it is to the right.
    """

    sibling_hash: str
    position: str  # 'left' | 'right'


def generate_merkle_proof(tree: MerkleTree, leaf_index: int) -> list[ProofStep]:
    """Generate a Merkle proof for the leaf at leaf_index.

    Args:
        tree:       A fully constructed MerkleTree.
        leaf_index: Zero-based index into tree.leaves.

    Returns:
        Ordered list of ProofStep objects from leaf level up to (but not
        including) the root level.

    Raises:
        IndexError: If leaf_index is out of range.
        ValueError: If the tree is empty.
    """
    if not tree.leaves:
        raise ValueError("Cannot generate a proof from an empty tree")

    n = len(tree.leaves)
    if leaf_index < 0 or leaf_index >= n:
        raise IndexError(f"leaf_index {leaf_index} out of range [0, {n})")

    proof: list[ProofStep] = []
    current_index = leaf_index

    for level in tree.levels[:-1]:  # skip root level
        # Pad odd-length levels the same way build_merkle_tree does
        padded = level if len(level) % 2 == 0 else level + [level[-1]]

        if current_index % 2 == 0:
            # current node is a left child; sibling is to the right
            sibling_index = current_index + 1
            proof.append(ProofStep(sibling_hash=padded[sibling_index], position="right"))
        else:
            # current node is a right child; sibling is to the left
            sibling_index = current_index - 1
            proof.append(ProofStep(sibling_hash=padded[sibling_index], position="left"))

        current_index //= 2

    return proof


def verify_merkle_proof(
    leaf_hash: str,
    proof: list[ProofStep],
    expected_root: str,
    algorithm: str,
) -> bool:
    """Verify that leaf_hash is committed to expected_root via proof.

    Args:
        leaf_hash:     Hex digest of the leaf being verified.
        proof:         Proof path as returned by generate_merkle_proof.
        expected_root: The root hash embedded in the CertifiedReport.
        algorithm:     hashlib algorithm name; must match the one used when
                       the tree was built (MerkleTree.algorithm).

    Returns:
        True if the recomputed root matches expected_root, False otherwise.
    """
    current = leaf_hash
    for step in proof:
        if step.position == "right":
            current = combine_hashes(current, step.sibling_hash, algorithm)
        else:
            current = combine_hashes(step.sibling_hash, current, algorithm)
    return current == expected_root
