"""Confidence Scorer node — scores each extracted claim against all retrieved leaves.

For each Claim in MARAState.extracted_claims this node calls score_claim()
with ALL retrieved leaves as the candidate pool. The confidence score is the
Beta-Binomial posterior mean: (1 + k) / (2 + n), where k is the number of
leaves that semantically corroborate the claim and n is the total leaf count.

Why all leaves instead of just the cited sources?
  Each claim is extracted from one leaf and cites that leaf. Scoring against
  only the cited source gives n=1 in the Beta-Binomial — essentially useless.
  Scoring against all 50 retrieved leaves gives a meaningful signal: a claim
  corroborated by 30/50 leaves is genuinely well-supported; one corroborated
  by 1/50 is not.

Why asyncio.to_thread?
  score_claim() calls embed(), which runs SentenceTransformer inference — a
  blocking CPU operation. Wrapping in asyncio.to_thread keeps the LangGraph
  event loop responsive.
"""

import asyncio

from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState
from mara.confidence.scorer import ScoredClaim, score_claim  # noqa: F401 — ScoredClaim re-exported for callers
from mara.logging import get_logger

_log = get_logger(__name__)


async def confidence_scorer(state: MARAState, config: RunnableConfig) -> dict:
    """Score each extracted claim and return the list of ScoredClaims.

    Args:
        state:  MARAState with ``extracted_claims``, ``retrieved_leaves``, and
                ``config`` populated.
        config: LangGraph RunnableConfig (unused directly).

    Returns:
        ``{"scored_claims": list[ScoredClaim]}``
    """
    research_config = state["config"]
    leaves = state["retrieved_leaves"]
    claims = state["extracted_claims"]

    if not claims:
        return {"scored_claims": []}

    _log.info("Scoring %d claims against %d leaves", len(claims), len(leaves))

    all_leaf_texts = [leaf["text"] for leaf in leaves]

    scored = []
    for claim in claims:
        result = await asyncio.to_thread(
            score_claim,
            claim["text"],
            all_leaf_texts,
            claim["source_indices"],
            research_config,
        )
        scored.append(result)

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
