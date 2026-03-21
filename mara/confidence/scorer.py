"""Confidence scorer: scores a claim via Beta-Binomial Source Agreement.

score_claim embeds the claim against ALL retrieved leaves and applies the
Beta-Binomial posterior mean (1 + k) / (2 + k) as the confidence score,
where k is the number of *unique source URLs* that corroborate the claim.

Why source-deduplicated k?
  A single article split into 20 chunks inflates the naive chunk count by 20×,
  violating the independence assumption of the Beta-Binomial model. Counting
  the number of distinct URLs that have at least one chunk above the similarity
  threshold restores the intended semantics: k = number of independent sources
  that agree with the claim.

No LangGraph imports. Pure and testable without a live LLM or graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mara.confidence.embeddings import embed
from mara.confidence.signals import compute_sa
from mara.config import ResearchConfig


@dataclass
class ScoredClaim:
    """A factual claim with its Beta-Binomial confidence score.

    Attributes:
        text:           The atomic claim text.
        source_indices: Indices of the leaves the claim extractor cited.
        confidence:     Beta-Binomial posterior mean over all retrieved leaves.
        corroborating:  k — leaves whose similarity exceeds the threshold.
        n_leaves:       N — total leaves evaluated.
        similarities:   Raw cosine similarities (for diagnostics / HITL display).
        contested:      True when SA is low but n >= n_leaves_contested_threshold,
                        meaning sources exist but disagree rather than being absent.
    """

    text: str
    source_indices: list[int]
    confidence: float
    corroborating: int
    n_leaves: int
    similarities: list[float] = field(default_factory=list)
    contested: bool = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two pre-normalized unit vectors.

    embed() uses normalize_embeddings=True so np.dot equals cosine similarity.
    """
    return float(np.dot(a, b))


def score_claim(
    claim_text: str,
    all_leaf_texts: list[str],
    source_indices: list[int],
    config: ResearchConfig,
    leaf_urls: list[str] | None = None,
) -> ScoredClaim:
    """Score a single claim against all retrieved leaves.

    Embeds the claim and all leaf texts, computes cosine similarities, then
    applies the Beta-Binomial posterior mean as the confidence score.

    When ``leaf_urls`` is provided, k counts unique source URLs (one vote per
    URL, using the maximum similarity across all chunks from that URL).  This
    enforces the independence assumption of the Beta-Binomial model.  When
    ``leaf_urls`` is None, k counts individual chunks (legacy behaviour).

    Args:
        claim_text:     The claim to score.
        all_leaf_texts: ALL retrieved leaf texts (not just cited sources).
        source_indices: Leaf indices the claim extractor attributed the claim to.
        config:         ResearchConfig driving embedding model and threshold.
        leaf_urls:      URL for each leaf in ``all_leaf_texts`` (same order).
                        When provided, enables source-level deduplication of k.

    Returns:
        A ScoredClaim with confidence in the open interval (0, 1).
    """
    threshold = config.similarity_support_threshold

    if not all_leaf_texts:
        return ScoredClaim(
            text=claim_text,
            source_indices=source_indices,
            confidence=compute_sa([], threshold),  # Beta(1,1) prior = 0.5
            corroborating=0,
            n_leaves=0,
            similarities=[],
        )

    all_embeddings = embed([claim_text] + all_leaf_texts, config.embedding_model, config.hf_token)
    claim_embedding = all_embeddings[0]
    leaf_embeddings = all_embeddings[1:]

    similarities = [cosine_similarity(claim_embedding, le) for le in leaf_embeddings]
    n = len(similarities)

    if leaf_urls is not None:
        # Source-deduplicated k: one vote per unique URL (max chunk similarity).
        url_max_sim: dict[str, float] = {}
        for url, sim in zip(leaf_urls, similarities):
            if url not in url_max_sim or sim > url_max_sim[url]:
                url_max_sim[url] = sim
        per_source_sims = list(url_max_sim.values())
        k = sum(1 for s in per_source_sims if s > threshold)
        confidence = compute_sa(per_source_sims, threshold)
    else:
        k = sum(1 for s in similarities if s > threshold)
        confidence = compute_sa(similarities, threshold)

    return ScoredClaim(
        text=claim_text,
        source_indices=source_indices,
        confidence=confidence,
        corroborating=k,
        n_leaves=n,
        similarities=similarities,
    )
