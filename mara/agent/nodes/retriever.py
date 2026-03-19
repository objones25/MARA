"""Semantic Retriever node — selects the most relevant MerkleLeaves for claim extraction.

Scores all merkle_leaves against the research query and sub-queries using
cosine similarity over SentenceTransformer embeddings, then selects the
top max_claim_sources leaves.  This prevents context-window overflow in the
Claim Extractor and focuses claim extraction on the highest-signal evidence.

Why score against sub-queries rather than just the main query?
  Each sub-query targets a specific research angle.  A leaf may be highly
  relevant to one sub-query but not the top-level question.  Taking the max
  similarity across all sub-queries captures aspect-specific relevance.

Why use contextualized_text rather than text for embedding?
  contextualized_text will carry LLM-generated context (Anthropic's
  Contextual Retrieval technique) once that step is implemented.  Using
  it here means the retriever automatically improves when that step is
  added, with no changes to this node.  Until then, contextualized_text
  equals text, so behaviour is identical to embedding the raw text.

Why asyncio.to_thread?
  embed() runs a SentenceTransformer model synchronously.  Wrapping in
  asyncio.to_thread keeps the LangGraph event loop responsive during the
  blocking inference pass.
"""

import asyncio

import numpy as np
from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState, MerkleLeaf
from mara.confidence.embeddings import embed
from mara.logging import get_logger

_log = get_logger(__name__)


async def retriever(state: MARAState, config: RunnableConfig) -> dict:
    """Select the top-K most relevant MerkleLeaves for claim extraction.

    Args:
        state:  MARAState with ``merkle_leaves``, ``sub_queries``, ``query``,
                and ``config`` populated.
        config: LangGraph RunnableConfig (unused directly).

    Returns:
        ``{"retrieved_leaves": list[MerkleLeaf]}``
    """
    leaves = state["merkle_leaves"]
    research_config = state["config"]
    k = research_config.max_claim_sources

    if not leaves:
        _log.warning("No Merkle leaves to retrieve from — returning empty set")
        return {"retrieved_leaves": []}

    if len(leaves) <= k:
        _log.info(
            "Leaf count (%d) ≤ max_claim_sources (%d) — skipping retrieval, using all",
            len(leaves),
            k,
        )
        return {"retrieved_leaves": list(leaves)}

    model_name = research_config.embedding_model
    query_texts = [state["query"]] + [sq["query"] for sq in state["sub_queries"]]
    leaf_texts = [leaf["contextualized_text"] for leaf in leaves]

    _log.info(
        "Retrieving top %d leaves from %d using %d query text(s)",
        k,
        len(leaves),
        len(query_texts),
    )

    query_embs, leaf_embs = await asyncio.gather(
        asyncio.to_thread(embed, query_texts, model_name),
        asyncio.to_thread(embed, leaf_texts, model_name),
    )

    # scores[i] = max cosine similarity between leaf i and any query text.
    # L2-normalised embeddings → dot product == cosine similarity.
    scores = (leaf_embs @ query_embs.T).max(axis=1)

    top_indices = np.argsort(scores)[::-1][:k]
    retrieved: list[MerkleLeaf] = [leaves[int(i)] for i in top_indices]

    _log.info(
        "Retrieved %d leaf/leaves (top score %.3f, bottom score %.3f)",
        len(retrieved),
        float(scores[top_indices[0]]),
        float(scores[top_indices[-1]]),
    )
    return {"retrieved_leaves": retrieved}
