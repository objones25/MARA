"""Confidence scorer: orchestrates SA, CSC, and LSA into a ScoredClaim.

This module is the top-level entry point for the confidence package. It:
  1. Embeds the claim and source chunks.
  2. Computes cosine similarities.
  3. Delegates to signals.py for SA, CSC, and LSA score computation.
  4. Returns a ScoredClaim dataclass with all signal values and the composite.

No LangGraph imports. The LSA LLM call is injected as a callable so this
module stays pure and testable without a live LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mara.confidence.embeddings import embed
from mara.confidence.signals import (
    LSAVerdict,
    compute_composite,
    compute_csc,
    compute_sa,
    lsa_verdict_to_score,
)
from mara.config import ResearchConfig


@dataclass
class ScoredClaim:
    """A factual claim with all confidence signals and the composite score.

    Attributes:
        text:           The atomic claim text.
        source_indices: Indices into the MerkleLeaf list used for scoring.
        sa:             Source Agreement Rate (Beta-Binomial posterior mean).
        csc:            Cross-Source Consistency (1 − CV among supporters).
        lsa:            LLM Self-Assessment score (0.0 / 0.5 / 1.0).
        confidence:     Composite weighted score.
        similarities:   Raw cosine similarities (for diagnostics / HITL display).
    """

    text: str
    source_indices: list[int]
    sa: float
    csc: float
    lsa: float
    confidence: float
    similarities: list[float] = field(default_factory=list)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two pre-normalized unit vectors.

    Because embed() uses normalize_embeddings=True the vectors already have
    unit norm, so np.dot is equivalent to cosine similarity. np.dot on
    float32 ndarrays is orders of magnitude faster than a Python-loop sum
    for the 384-dim all-MiniLM-L6-v2 vectors.
    """
    return float(np.dot(a, b))


def score_claim(
    claim_text: str,
    source_texts: list[str],
    source_indices: list[int],
    lsa_callable: Callable[[str, list[str]], LSAVerdict],
    config: ResearchConfig,
) -> ScoredClaim:
    """Score a single atomic claim against a list of source texts.

    All scoring parameters (embedding model, support threshold, confidence
    weights) are read from config so there is a single source of truth.

    Args:
        claim_text:     The claim to score.
        source_texts:   List of source chunk texts to score against.
        source_indices: Corresponding Merkle leaf indices for each source text.
        lsa_callable:   Function(claim_text, source_texts) → LSAVerdict.
                        Injected to keep this module testable without a live LLM.
        config:         ResearchConfig instance driving all scoring parameters.

    Returns:
        A fully populated ScoredClaim.

    Raises:
        ValueError: If source_texts and source_indices have different lengths.
    """
    if len(source_texts) != len(source_indices):
        raise ValueError(
            f"source_texts length ({len(source_texts)}) must equal "
            f"source_indices length ({len(source_indices)})"
        )

    w = config.confidence_weights
    threshold = config.similarity_support_threshold

    if not source_texts:
        # Delegate to signals so the degenerate-case logic lives in one place.
        # compute_sa([]) → Beta(1,1) prior mean = 0.5
        # compute_csc([]) → "not enough data" sentinel = 0.5
        sa = compute_sa([], threshold)
        csc = compute_csc([], threshold)
        lsa_verdict = lsa_callable(claim_text, [])
        lsa = lsa_verdict_to_score(lsa_verdict)
        composite = compute_composite(sa=sa, csc=csc, lsa=lsa, alpha=w.alpha, beta=w.beta, gamma=w.gamma)
        return ScoredClaim(
            text=claim_text,
            source_indices=[],
            sa=sa,
            csc=csc,
            lsa=lsa,
            confidence=composite,
            similarities=[],
        )

    # Embed claim + all sources in one batch for efficiency
    all_embeddings = embed([claim_text] + source_texts, config.embedding_model)
    claim_embedding = all_embeddings[0]
    source_embeddings = all_embeddings[1:]

    similarities = [cosine_similarity(claim_embedding, s) for s in source_embeddings]

    sa = compute_sa(similarities, threshold)
    csc = compute_csc(similarities, threshold)

    lsa_verdict = lsa_callable(claim_text, source_texts)
    lsa = lsa_verdict_to_score(lsa_verdict)

    composite = compute_composite(sa=sa, csc=csc, lsa=lsa, alpha=w.alpha, beta=w.beta, gamma=w.gamma)

    return ScoredClaim(
        text=claim_text,
        source_indices=source_indices,
        sa=sa,
        csc=csc,
        lsa=lsa,
        confidence=composite,
        similarities=similarities,
    )
