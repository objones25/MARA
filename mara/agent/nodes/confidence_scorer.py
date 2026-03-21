"""Confidence Scorer node — scores each extracted claim against all merkle leaves.

For each Claim in MARAState.extracted_claims this node calls score_claim()
with ALL merkle_leaves as the candidate pool (not just retrieved_leaves).
The confidence score is the Beta-Binomial posterior mean: (1 + k) / (2 + k),
where k is the number of *unique source URLs* whose best chunk exceeds the
similarity threshold.

Why merkle_leaves instead of retrieved_leaves?
  retrieved_leaves is a top-K subset selected for claim extraction quality.
  Scoring against only those K leaves means a claim extracted from leaf 3 is
  never tested against leaf 47, even if leaf 47 is a stronger corroboration.
  Scoring against the full corpus gives every claim the best possible evidence
  test.

Why source-deduplicated k?
  A single article chunked into 15 pieces can push k to 15, but that is one
  source's opinion repeated — not independent corroboration. Deduplication
  enforces the independence assumption of the Beta-Binomial model: k counts
  unique URLs, not chunks.

Why two embed() batches instead of one per claim?
  The previous implementation called embed([claim_text] + all_leaf_texts)
  inside score_claim on every loop iteration, resulting in O(n_claims) batches
  each containing all leaf texts. The node now embeds all leaves once (or
  reads them from RunContext.leaf_embeddings if the retriever cached them) and
  all claims in a single second batch. Embedding calls drop from n_claims
  batches to at most 2.

Why asyncio.to_thread?
  embed() runs SentenceTransformer inference — a blocking CPU operation.
  Wrapping in asyncio.to_thread keeps the LangGraph event loop responsive.
  score_claim() itself is now pure numpy and needs no threading.
"""

import asyncio

import numpy as np
from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState
from mara.confidence.embeddings import embed
from mara.confidence.scorer import ScoredClaim, score_claim  # noqa: F401 — ScoredClaim re-exported for callers
from mara.logging import get_logger

_log = get_logger(__name__)


async def confidence_scorer(state: MARAState, config: RunnableConfig) -> dict:
    """Score each extracted claim and return the list of ScoredClaims.

    Args:
        state:  MARAState with ``extracted_claims``, ``merkle_leaves``, and
                ``config`` populated.
        config: LangGraph RunnableConfig; ``config["configurable"]`` may
                contain a ``run_context`` with cached leaf embeddings.

    Returns:
        ``{"scored_claims": list[ScoredClaim]}``
    """
    research_config = state["config"]
    leaves = state["merkle_leaves"]
    claims = state["extracted_claims"]

    if not claims:
        return {"scored_claims": []}

    # Empty leaf pool — return prior mean for every claim.
    if not leaves:
        _log.warning("No merkle leaves — all claims receive Beta(1,1) prior confidence (0.5)")
        return {
            "scored_claims": [
                ScoredClaim(
                    text=c["text"],
                    source_indices=c["source_indices"],
                    confidence=0.5,
                    corroborating=0,
                    n_leaves=0,
                    n_unique_urls=0,
                )
                for c in claims
            ]
        }

    leaf_urls = [leaf["url"] for leaf in leaves]
    leaf_texts = [leaf["contextualized_text"] for leaf in leaves]

    # --- Leaf embeddings: free from RunContext if retriever cached them ---
    configurable = config.get("configurable", {}) if config else {}
    run_context = configurable.get("run_context")

    cached_hashes = getattr(run_context, "leaf_embedding_hashes", None) if run_context else None
    current_hashes = [leaf["hash"] for leaf in leaves]

    if (
        run_context is not None
        and run_context.leaf_embeddings is not None
        and cached_hashes == current_hashes
    ):
        leaf_embs: np.ndarray = run_context.leaf_embeddings
        _log.debug("Leaf embeddings served from RunContext (%d leaves)", len(leaves))
    else:
        leaf_embs = await asyncio.to_thread(
            embed, leaf_texts, research_config.embedding_model, research_config.hf_token
        )
        _log.debug("Leaf embeddings computed fresh (%d leaves)", len(leaves))

    # --- Claim embeddings: one batch for all claims ---
    claim_texts = [c["text"] for c in claims]
    claim_embs: np.ndarray = await asyncio.to_thread(
        embed, claim_texts, research_config.embedding_model, research_config.hf_token
    )

    _log.info(
        "Scoring %d claims against %d leaves (%d unique URLs)",
        len(claims),
        len(leaves),
        len(set(leaf_urls)),
    )

    scored = [
        score_claim(
            claims[i]["text"],
            claim_embs[i],
            leaf_embs,
            leaf_urls,
            claims[i]["source_indices"],
            research_config,
        )
        for i in range(len(claims))
    ]

    if scored:
        all_sims = [s for c in scored for s in c.similarities]
        if all_sims:
            all_sims_sorted = sorted(all_sims)
            n = len(all_sims_sorted)
            p25 = all_sims_sorted[n // 4]
            p50 = all_sims_sorted[n // 2]
            p75 = all_sims_sorted[(3 * n) // 4]
            _log.info(
                "Similarity distribution — min: %.3f  p25: %.3f  p50: %.3f  p75: %.3f  max: %.3f  (threshold: %.2f)",
                all_sims_sorted[0],
                p25,
                p50,
                p75,
                all_sims_sorted[-1],
                research_config.similarity_support_threshold,
            )

    _log.info("Scored %d claims", len(scored))
    return {"scored_claims": scored}
