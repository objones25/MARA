"""Confidence scorer: scores a claim via Beta-Binomial Source Agreement.

score_claim embeds the claim against ALL retrieved leaves and applies the
Beta-Binomial posterior mean as the confidence score.

Why all leaves, not just the cited sources?
  Each claim is attributed to one leaf by the extractor (n=1 in the
  Beta-Binomial — informationally useless). Scoring against all retrieved
  leaves gives a real signal: a claim corroborated by 30/50 leaves is
  genuinely well-supported; one corroborated by 1/50 is not.

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
    """

    text: str
    source_indices: list[int]
    confidence: float
    corroborating: int
    n_leaves: int
    similarities: list[float] = field(default_factory=list)


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
) -> ScoredClaim:
    """Score a single claim against all retrieved leaves.

    Embeds the claim and all leaf texts, computes cosine similarities, then
    applies the Beta-Binomial posterior mean as the confidence score.

    Args:
        claim_text:     The claim to score.
        all_leaf_texts: ALL retrieved leaf texts (not just cited sources).
        source_indices: Leaf indices the claim extractor attributed the claim to.
        config:         ResearchConfig driving embedding model and threshold.

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
