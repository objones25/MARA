"""Confidence signal: Beta-Binomial Source Agreement Rate.

The single confidence metric is SA (Source Agreement Rate): the
Beta-Binomial posterior mean updated only on corroborating leaves.

    SA = (1 + k) / (2 + k)

where k = leaves whose cosine similarity to the claim exceeds the support
threshold.

Non-corroborating leaves are not treated as negative evidence — a leaf
that doesn't mention a claim is simply irrelevant, not contradictory.
Only positive corroboration moves the score.

The Beta(1, 1) prior (uniform / maximum uncertainty) means:
  - 0 corroborating leaves → SA = 0.5  (prior mean, maximum uncertainty)
  - 1 corroborating leaf   → SA = 0.67
  - 3 corroborating leaves → SA = 0.80 (clears high_confidence_threshold)
  - k → ∞                 → SA → 1.0  (never exactly reached)

Laplace smoothing guarantees SA is always in the open interval (0, 1).
"""

from __future__ import annotations


def compute_sa(similarities: list[float], support_threshold: float) -> float:
    """Compute the Beta-Binomial posterior mean (Source Agreement Rate).

    Updates a Beta(1, 1) prior with k successes only — non-corroborating
    leaves are excluded from the denominator because their absence of
    support is not evidence of contradiction.

    Posterior mean = (1 + k) / (2 + k)

    where k = number of corroborating leaves (similarity > support_threshold).

    Args:
        similarities:      Cosine similarities between the claim and each leaf.
        support_threshold: Minimum similarity (exclusive) for a leaf to count
                           as corroborating.

    Returns:
        SA score in the open interval (0, 1).
    """
    k = sum(1 for s in similarities if s > support_threshold)
    return (1 + k) / (2 + k)
