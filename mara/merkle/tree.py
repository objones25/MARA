"""Binary Merkle tree construction from a list of leaf hashes.

Each internal node is hash_algorithm(left_child_hex + right_child_hex) encoded
as UTF-8. If the number of leaves is odd, the last leaf is duplicated to
balance the tree.

The algorithm used for internal nodes must match the algorithm used to produce
the leaf hashes (ResearchConfig.hash_algorithm).
"""

import hashlib
from dataclasses import dataclass, field


def combine_hashes(left: str, right: str, algorithm: str) -> str:
    """Hash two hex-digest strings into a parent node hex digest.

    Args:
        left:      Left child hex digest.
        right:     Right child hex digest.
        algorithm: hashlib algorithm name (e.g. 'sha256').

    Returns:
        Hex digest of hash(left_bytes + right_bytes).
    """
    data = (left + right).encode("utf-8")
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


@dataclass
class MerkleTree:
    """A balanced binary Merkle tree built from leaf hashes.

    Attributes:
        leaves:    The original leaf hashes (in insertion order).
        levels:    All tree levels; index 0 = leaf level, last index = [root].
        root:      The root hash; empty string if the tree has no leaves.
        algorithm: The hash algorithm used for all internal nodes.
    """

    leaves: list[str] = field(default_factory=list)
    levels: list[list[str]] = field(default_factory=list)
    root: str = ""
    algorithm: str = "sha256"


def build_merkle_tree(leaf_hashes: list[str], algorithm: str) -> MerkleTree:
    """Construct a MerkleTree from a list of hex-digest leaf hashes.

    Args:
        leaf_hashes: Ordered list of hex digests (one per source chunk).
        algorithm:   hashlib algorithm name for internal node hashing.
                     Must match the algorithm used to produce leaf_hashes
                     (i.e. ResearchConfig.hash_algorithm).

    Returns:
        A MerkleTree with all levels populated and root set.

    Raises:
        ValueError: If any leaf hash is an empty string.
    """
    if not leaf_hashes:
        return MerkleTree(algorithm=algorithm)

    if any(h == "" for h in leaf_hashes):
        raise ValueError("Leaf hashes must not be empty strings")

    current_level = list(leaf_hashes)
    levels: list[list[str]] = [current_level]

    while len(current_level) > 1:
        # Duplicate last leaf if odd count to keep the tree balanced
        if len(current_level) % 2 == 1:
            current_level = current_level + [current_level[-1]]

        next_level = [
            combine_hashes(current_level[i], current_level[i + 1], algorithm)
            for i in range(0, len(current_level), 2)
        ]
        levels.append(next_level)
        current_level = next_level

    return MerkleTree(
        leaves=list(leaf_hashes),
        levels=levels,
        root=current_level[0],
        algorithm=algorithm,
    )
