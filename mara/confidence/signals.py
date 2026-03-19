"""Individual confidence signal computation functions.

All functions are pure: they take data in, return a float out, and have no
side effects. No LangGraph imports. Testable without running the full graph.

All thresholds and weights are required parameters — callers must pass them
from ResearchConfig. No defaults here; ResearchConfig is the single source
of truth for all tunable values.

Signals:
  SA  — Source Agreement Rate (Beta-Binomial posterior mean)
  CSC — Cross-Source Consistency (1 − coefficient of variation)
  LSA — LLM Self-Assessment (mapped from categorical to float)
"""

from __future__ import annotations

import math
from typing import Literal


# ---------------------------------------------------------------------------
# SA — Source Agreement Rate
# ---------------------------------------------------------------------------

def compute_sa(similarities: list[float], support_threshold: float) -> float:
    """Compute the Laplace-smoothed Beta-Binomial posterior mean.

    Models each source as a Bernoulli trial: does its cosine similarity to the
    claim exceed the support threshold? Prior: Beta(1, 1) — uniform.

    Posterior mean = (1 + k) / (2 + n)

    where k = number of supporting sources, n = total sources.

    Returns (1 + 0) / (2 + 0) = 0.5 when similarities is empty, which is the
    prior mean of Beta(1, 1) — maximum uncertainty with no observations.

    Args:
        similarities:      Cosine similarities between claim and each source chunk.
        support_threshold: ResearchConfig.similarity_support_threshold.

    Returns:
        SA score in (0, 1).
    """
    if not similarities:
        return (1 + 0) / (2 + 0)  # Beta(1,1) prior mean with zero observations

    n = len(similarities)
    k = sum(1 for s in similarities if s > support_threshold)
    return (1 + k) / (2 + n)


# ---------------------------------------------------------------------------
# CSC — Cross-Source Consistency
# ---------------------------------------------------------------------------

def compute_csc(similarities: list[float], support_threshold: float) -> float:
    """Compute 1 − coefficient of variation among supporting source similarities.

    CV = std(similarities) / mean(similarities). High CV means the supporting
    sources vary widely in how strongly they endorse the claim.

    Returns 0.5 (neutral / maximum-uncertainty sentinel) when fewer than 2
    sources support the claim — insufficient data for a meaningful CV estimate.

    Args:
        similarities:      Cosine similarities for *all* retrieved sources.
        support_threshold: ResearchConfig.similarity_support_threshold.

    Returns:
        CSC in [0, 1].
    """
    supporting = [s for s in similarities if s > support_threshold]

    if len(supporting) < 2:
        return 0.5  # not enough data; neutral sentinel

    mean = sum(supporting) / len(supporting)

    if mean == 0.0:
        return 0.5  # degenerate: all supporting similarities are exactly zero

    variance = sum((s - mean) ** 2 for s in supporting) / len(supporting)
    std = math.sqrt(variance)
    cv = std / mean
    return max(0.0, min(1.0, 1.0 - cv))


# ---------------------------------------------------------------------------
# LSA — LLM Self-Assessment
# ---------------------------------------------------------------------------

LSAVerdict = Literal["supported", "partially_supported", "unsupported"]

_LSA_SCORE_MAP: dict[LSAVerdict, float] = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
}


def lsa_verdict_to_score(verdict: LSAVerdict) -> float:
    """Map a categorical LSA verdict to a float score.

    Args:
        verdict: One of 'supported', 'partially_supported', 'unsupported'.

    Returns:
        1.0, 0.5, or 0.0 respectively.

    Raises:
        ValueError: If the verdict is not a recognised category.
    """
    if verdict not in _LSA_SCORE_MAP:
        raise ValueError(
            f"Unknown LSA verdict '{verdict}'. "
            f"Expected one of {list(_LSA_SCORE_MAP)}"
        )
    return _LSA_SCORE_MAP[verdict]


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite(
    sa: float,
    csc: float,
    lsa: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Compute the weighted composite confidence score.

    confidence = alpha*SA + beta*CSC + gamma*LSA

    Weights must be provided by the caller from ResearchConfig.confidence_weights.
    No defaults here — ResearchConfig is the single source of truth.

    Args:
        sa:    Source Agreement Rate.
        csc:   Cross-Source Consistency.
        lsa:   LLM Self-Assessment score.
        alpha: ResearchConfig.confidence_weights.alpha
        beta:  ResearchConfig.confidence_weights.beta
        gamma: ResearchConfig.confidence_weights.gamma

    Returns:
        Composite score clamped to [0, 1].
    """
    return max(0.0, min(1.0, alpha * sa + beta * csc + gamma * lsa))
