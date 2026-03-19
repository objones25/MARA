"""Source Hasher node — hashes each SourceChunk into a MerkleLeaf.

Runs immediately after the search worker fan-in, once all raw_chunks have
been collected from parallel workers.  Produces the ordered list of
MerkleLeaf TypedDicts that the Merkle Builder node will use to construct
the tree.

Why is this a separate node from the Merkle Builder?
  Separation of concerns: this node performs per-chunk hashing (a map
  operation over raw_chunks), while the Merkle Builder performs tree
  construction (a fold over leaf hashes).  Keeping them separate makes
  each node unit-testable in isolation and makes the data flow explicit
  in the LangGraph execution trace.

Why preserve sub_query on the leaf?
  Retaining the originating sub-query on each MerkleLeaf enables downstream
  attribution: the Confidence Scorer and Report Synthesizer can trace a leaf
  back to the specific research angle that surfaced it.

Why is the index field set here?
  The leaf index is its position in the ordered merkle_leaves list, which
  must match its position in MerkleTree.leaves for generate_merkle_proof()
  to work correctly.  Setting it once at hashing time avoids any ambiguity
  about insertion order.
"""

from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState, MerkleLeaf
from mara.merkle.hasher import hash_chunk


def source_hasher(state: MARAState, config: RunnableConfig) -> dict:
    """Hash each raw SourceChunk and return the ordered MerkleLeaf list.

    Args:
        state:  MARAState with ``raw_chunks`` and ``config`` populated.
        config: LangGraph RunnableConfig (unused directly; present for the
                standard node signature).

    Returns:
        ``{"merkle_leaves": list[MerkleLeaf]}``
    """
    algorithm = state["config"].hash_algorithm
    leaves: list[MerkleLeaf] = []

    for i, chunk in enumerate(state["raw_chunks"]):
        digest = hash_chunk(
            url=chunk["url"],
            text=chunk["text"],
            retrieved_at=chunk["retrieved_at"],
            algorithm=algorithm,
        )
        leaves.append(
            MerkleLeaf(
                url=chunk["url"],
                text=chunk["text"],
                retrieved_at=chunk["retrieved_at"],
                hash=digest,
                index=i,
                sub_query=chunk["sub_query"],
            )
        )

    return {"merkle_leaves": leaves}
