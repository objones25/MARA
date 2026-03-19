"""Merkle Builder node — constructs the MerkleTree from hashed leaves.

Runs immediately after source_hasher.  Extracts the ordered list of hex
digests from MARAState.merkle_leaves and passes them to build_merkle_tree().
The resulting MerkleTree is stored on MARAState.merkle_tree for use by the
Report Synthesizer (which embeds the root hash in the certified report) and
the mara verify CLI command.

Why is tree construction separate from hashing?
  See source_hasher.py — they are distinct operations (map vs. fold) and
  keeping them in separate nodes makes the graph trace explicit.

Why does an empty raw_chunks list produce an empty tree (not an error)?
  A research session with no scraped pages is a valid (if degenerate) outcome
  — e.g. all Brave results returned 404 or Firecrawl rate-limited every URL.
  The node returns a MerkleTree with root="" so downstream nodes can check
  ``state["merkle_tree"].root == ""`` and handle the empty case gracefully.
"""

from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState
from mara.logging import get_logger
from mara.merkle.tree import MerkleTree, build_merkle_tree

_log = get_logger(__name__)


def merkle_builder(state: MARAState, config: RunnableConfig) -> dict:
    """Build the MerkleTree from MARAState.merkle_leaves.

    Args:
        state:  MARAState with ``merkle_leaves`` and ``config`` populated.
        config: LangGraph RunnableConfig (unused directly).

    Returns:
        ``{"merkle_tree": MerkleTree}``
    """
    algorithm = state["config"].hash_algorithm
    leaf_hashes = [leaf["hash"] for leaf in state["merkle_leaves"]]
    tree: MerkleTree = build_merkle_tree(leaf_hashes, algorithm)

    if tree.root:
        _log.info(
            "Built Merkle tree: %d leaves, root %s…", len(leaf_hashes), tree.root[:12]
        )
    else:
        _log.warning("Built empty Merkle tree — no leaves provided")

    return {"merkle_tree": tree}
