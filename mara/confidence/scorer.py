"""Confidence scorer: scores a claim via Beta-Binomial Source Agreement.

score_claim takes pre-computed embedding vectors and applies the
Beta-Binomial posterior mean (1 + k) / (2 + k) as the confidence score,
where k is the number of *unique source URLs* that corroborate the claim.

Why source-deduplicated k?
  A single article split into 20 chunks inflates the naive chunk count by 20×,
  violating the independence assumption of the Beta-Binomial model. Counting
  the number of distinct URLs that have at least one chunk above the similarity
  threshold restores the intended semantics: k = number of independent sources
  that agree with the claim.

Why pre-computed embeddings?
  score_claim is called once per claim in a tight loop. Embedding all leaves
  once at the node level (rather than inside each score_claim call) reduces
  SentenceTransformer inference from O(n_claims × n_leaves) to O(n_leaves +
  n_claims) — two batches instead of n_claims batches.

No LangGraph imports. Pure and testable without a live LLM or graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mara.confidence.signals import compute_sa
from mara.config import ResearchConfig


@dataclass
class ScoredClaim:
    """A factual claim with its Beta-Binomial confidence score.

    Attributes:
        text:           The atomic claim text.
        source_indices: Indices of the leaves the claim extractor cited.
        confidence:     Beta-Binomial posterior mean (1 + k) / (2 + k).
        corroborating:  k — unique source URLs whose best chunk exceeds the
                        similarity threshold.
        n_leaves:       Total leaf chunks evaluated (chunk count, not URL count).
        n_unique_urls:  Number of distinct source URLs in the scoring pool.
        similarities:   Raw per-chunk cosine similarities (for diagnostics).
        contested:      True when SA is low but n_unique_urls >=
                        n_leaves_contested_threshold, meaning sources exist but
                        disagree rather than being absent.
    """

    text: str
    source_indices: list[int]
    confidence: float
    corroborating: int
    n_leaves: int
    n_unique_urls: int = 0
    similarities: list[float] = field(default_factory=list)
    contested: bool = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two pre-normalized unit vectors.

    embed() uses normalize_embeddings=True so np.dot equals cosine similarity.
    """
    return float(np.dot(a, b))


def score_claim(
    claim_text: str,
    claim_embedding: np.ndarray,
    leaf_embeddings: np.ndarray,
    leaf_urls: list[str],
    source_indices: list[int],
    config: ResearchConfig,
) -> ScoredClaim:
    """Score a single claim against pre-computed leaf embeddings.

    All embedding computation must be done by the caller before invoking this
    function.  score_claim itself is pure numpy — no I/O, no model inference.

    Source-deduplication: for each unique URL, only the maximum chunk
    similarity is retained.  k counts unique URLs above the threshold, not
    individual chunks.

    Args:
        claim_text:       Text of the claim (stored on ScoredClaim for display).
        claim_embedding:  Unit vector of shape (dim,).
        leaf_embeddings:  Unit matrix of shape (n_leaves, dim).
        leaf_urls:        URL for each row of leaf_embeddings (same order).
        source_indices:   Leaf indices attributed to the claim by the extractor.
        config:           ResearchConfig providing similarity_support_threshold.

    Returns:
        A ScoredClaim with confidence in the open interval (0, 1).
    """
    threshold = config.similarity_support_threshold

    if leaf_embeddings.shape[0] == 0:
        return ScoredClaim(
            text=claim_text,
            source_indices=source_indices,
            confidence=compute_sa([], threshold),  # Beta(1,1) prior = 0.5
            corroborating=0,
            n_leaves=0,
            n_unique_urls=0,
            similarities=[],
        )

    similarities = (leaf_embeddings @ claim_embedding).tolist()
    n = len(similarities)

    # Source-deduplicated k: one vote per unique URL (max chunk similarity).
    url_max_sim: dict[str, float] = {}
    for url, sim in zip(leaf_urls, similarities):
        if url not in url_max_sim or sim > url_max_sim[url]:
            url_max_sim[url] = sim
    per_source_sims = list(url_max_sim.values())
    k = sum(1 for s in per_source_sims if s > threshold)
    confidence = compute_sa(per_source_sims, threshold)

    return ScoredClaim(
        text=claim_text,
        source_indices=source_indices,
        confidence=confidence,
        corroborating=k,
        n_leaves=n,
        n_unique_urls=len(url_max_sim),
        similarities=similarities,
    )
