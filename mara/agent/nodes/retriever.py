"""Semantic + BM25 hybrid Retriever node.

Selects the most relevant MerkleLeaves for claim extraction using a
two-stage strategy:

  1. Embedding cache
     Leaf embeddings are loaded from the SQLite DB when available.  Only
     leaves with no stored embedding — or whose cached blob has a dimension
     mismatch (model changed) — are passed through the SentenceTransformer
     model.  New embeddings are written back immediately so subsequent runs
     on the same corpus skip inference entirely.

  2. Hybrid Reciprocal Rank Fusion (RRF)
     BM25 (FTS5, keyword match) and semantic (cosine similarity) rankings
     are fused using RRF (k=60).  BM25 captures exact-match / named-entity
     evidence; semantic captures paraphrases and synonym matches.  RRF
     combines both without requiring any tuning.

     RRF_score(d) = 1/(k + r_sem(d) + 1) + 1/(k + r_bm25(d) + 1)

     Leaves absent from BM25 results receive a penalty rank of len(leaves),
     keeping their BM25 contribution small but non-zero.

Pure-semantic fallback
----------------------
When ``leaf_repo`` is not injected (``leaf_db_enabled=False`` or the node
is running outside the CLI), the retriever falls back to the original
cosine-similarity-only ranking.  All existing behaviour is preserved.

Thread safety
-------------
``leaf_repo.get_embeddings_for_hashes`` and ``leaf_repo.update_embeddings``
are synchronous SQLite calls and execute on the event loop thread.  They
are fast (< 5 ms at this scale) and do not require an executor.
``embed()`` and ``bm25_search()`` are CPU-bound / potentially slower and
are wrapped in ``asyncio.to_thread``.
"""

import asyncio

import numpy as np
from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState, MerkleLeaf
from mara.confidence.embeddings import embed
from mara.logging import get_logger

_log = get_logger(__name__)

# Standard RRF constant from Cormack, Clarke & Buettcher (2009).
# Higher values dampen the influence of rank-1 results; 60 is the
# conventional default that works well across diverse corpora.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _load_or_compute_leaf_embeddings(
    leaves: list[MerkleLeaf],
    model_name: str,
    target_dim: int,
    leaf_repo,
    hf_token: str = "",
) -> np.ndarray:
    """Return a ``(len(leaves), target_dim)`` float32 embedding matrix.

    When *leaf_repo* is available:

    - Cached blobs with the correct dimension are used as-is.
    - Blobs with a dimension mismatch (model changed between runs) are
      re-embedded and the stale blob is overwritten.
    - Corrupt blobs (byte-length not divisible by 4) are treated as cache
      misses and re-embedded.
    - Leaves with no stored blob are embedded and stored.

    When *leaf_repo* is None (DB disabled), all leaves are embedded fresh.

    Note: the background sweep ``WHERE embedding_model != :current_model``
    can proactively update stale blobs across the whole corpus.  Inline
    re-embedding here only handles leaves encountered in the current run.
    """
    leaf_texts = [leaf["contextualized_text"] for leaf in leaves]

    if leaf_repo is None:
        return await asyncio.to_thread(embed, leaf_texts, model_name, hf_token)

    hashes = [leaf["hash"] for leaf in leaves]
    cached_blobs = leaf_repo.get_embeddings_for_hashes(hashes)

    valid: dict[str, np.ndarray] = {}
    to_embed_indices: list[int] = []
    to_embed_texts: list[str] = []

    for i, leaf in enumerate(leaves):
        blob = cached_blobs.get(leaf["hash"])
        if blob is not None:
            try:
                arr = np.frombuffer(blob, dtype=np.float32).copy()
            except ValueError:
                _log.warning(
                    "Corrupt embedding blob for leaf %s — re-embedding",
                    leaf["hash"][:12],
                )
                arr = None

            if arr is not None and arr.shape[0] == target_dim:
                valid[leaf["hash"]] = arr
                continue

            if arr is not None:
                _log.warning(
                    "Cached embedding for leaf %s has dim %d, expected %d"
                    " — re-embedding (model changed?)",
                    leaf["hash"][:12],
                    arr.shape[0],
                    target_dim,
                )

        to_embed_indices.append(i)
        to_embed_texts.append(leaf_texts[i])

    if to_embed_texts:
        new_embs: np.ndarray = await asyncio.to_thread(embed, to_embed_texts, model_name, hf_token)
        new_blobs: dict[str, bytes] = {}
        for j, idx in enumerate(to_embed_indices):
            h = leaves[idx]["hash"]
            valid[h] = new_embs[j]
            new_blobs[h] = new_embs[j].astype(np.float32).tobytes()
        leaf_repo.update_embeddings(new_blobs, model_name)
        _log.debug(
            "Embedded %d uncached/stale leaf/leaves; %d served from cache",
            len(to_embed_texts),
            len(leaves) - len(to_embed_texts),
        )
    else:
        _log.debug("All %d leaf/leaves served from embedding cache", len(leaves))

    return np.stack([valid[leaf["hash"]] for leaf in leaves])


def _rrf_scores(
    leaves: list[MerkleLeaf],
    semantic_scores: np.ndarray,
    semantic_order: np.ndarray,
    bm25_hash_rank: dict[str, int],
) -> np.ndarray:
    """Compute RRF score for each leaf.

    Uses 1-indexed ranks: ``1/(k + rank + 1)`` where rank is 0-based.
    Leaves absent from *bm25_hash_rank* receive penalty rank ``len(leaves)``.

    Returns a ``(len(leaves),)`` float64 array; higher is better.
    """
    n = len(leaves)
    penalty = n

    # Map each leaf's position-in-leaves to its semantic rank.
    sem_rank = np.empty(n, dtype=np.float64)
    for rank, idx in enumerate(semantic_order):
        sem_rank[int(idx)] = rank

    bm25_rank = np.array(
        [bm25_hash_rank.get(leaf["hash"], penalty) for leaf in leaves],
        dtype=np.float64,
    )

    return (
        1.0 / (_RRF_K + sem_rank + 1)
        + 1.0 / (_RRF_K + bm25_rank + 1)
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def retriever(state: MARAState, config: RunnableConfig) -> dict:
    """Select the top-K most relevant MerkleLeaves for claim extraction.

    Args:
        state:  MARAState with ``merkle_leaves``, ``sub_queries``, ``query``,
                and ``config`` populated.
        config: LangGraph RunnableConfig; ``config["configurable"]`` may
                contain ``leaf_repo`` and ``run_id`` injected by the CLI.

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

    configurable = config.get("configurable", {}) if config else {}
    leaf_repo = configurable.get("leaf_repo")
    run_id = configurable.get("run_id")

    model_name = research_config.embedding_model
    query_texts = [state["query"]] + [sq["query"] for sq in state["sub_queries"]]

    _log.info(
        "Retrieving top %d leaves from %d using %d query text(s) [%s]",
        k,
        len(leaves),
        len(query_texts),
        "hybrid" if leaf_repo else "semantic",
    )

    # Query embeddings are always computed fresh.  Computing them first
    # establishes target_dim for the embedding cache dimension check.
    hf_token = research_config.hf_token
    query_embs = await asyncio.to_thread(embed, query_texts, model_name, hf_token)
    target_dim = query_embs.shape[1]

    # Leaf embeddings: load from cache where possible, compute the rest.
    leaf_embs = await _load_or_compute_leaf_embeddings(
        leaves, model_name, target_dim, leaf_repo, hf_token
    )

    # Semantic scores: max cosine similarity across all query texts.
    # L2-normalised embeddings → dot product == cosine similarity.
    semantic_scores = (leaf_embs @ query_embs.T).max(axis=1)
    semantic_order = np.argsort(semantic_scores)[::-1]  # descending

    # Pure-semantic fallback — DB disabled or leaf_repo not injected.
    if leaf_repo is None:
        retrieved = [leaves[int(i)] for i in semantic_order[:k]]
        _log.info(
            "Pure semantic — top score %.3f, bottom %.3f",
            float(semantic_scores[semantic_order[0]]),
            float(semantic_scores[semantic_order[k - 1]]),
        )
        return {"retrieved_leaves": retrieved}

    # --- Hybrid BM25 + semantic via Reciprocal Rank Fusion ---

    # Use the main research query for BM25.  FTS5 porter-ascii tokeniser
    # applies stemming, so sub-query keywords surface via the root query.
    bm25_results = await asyncio.to_thread(
        leaf_repo.bm25_search,
        state["query"],
        run_id,
        len(leaves),
    )
    bm25_hash_rank: dict[str, int] = {
        r["hash"]: i for i, r in enumerate(bm25_results)
    }

    rrf = _rrf_scores(leaves, semantic_scores, semantic_order, bm25_hash_rank)
    top_indices = np.argsort(rrf)[::-1][:k]
    retrieved = [leaves[int(i)] for i in top_indices]

    _log.info(
        "Hybrid RRF — %d BM25 hit(s), top RRF %.4f",
        len(bm25_results),
        float(rrf[top_indices[0]]) if len(top_indices) else 0.0,
    )
    return {"retrieved_leaves": retrieved}
