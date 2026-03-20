"""Confidence signal: Beta-Binomial Source Agreement Rate.

The single confidence metric is SA (Source Agreement Rate): the
Beta-Binomial posterior mean over how many retrieved leaves semantically
corroborate a claim.

    SA = (1 + k) / (2 + n)

where k = leaves whose cosine similarity to the claim exceeds the support
threshold, and n = total leaves evaluated.

The Beta(1, 1) prior (uniform / maximum uncertainty) means:
  - 0 leaves evaluated  → SA = 0.5 (prior mean, maximum uncertainty)
  - 0 of n supporting   → SA approaches 0 but never reaches it
  - n of n supporting   → SA approaches 1 but never reaches it

Laplace smoothing guarantees SA is always in the open interval (0, 1).
"""

from __future__ import annotations


def compute_sa(similarities: list[float], support_threshold: float) -> float:
    """Compute the Beta-Binomial posterior mean (Source Agreement Rate).

    Models each leaf as a Bernoulli trial: does its cosine similarity to the
    claim exceed the support threshold?

    Posterior mean = (1 + k) / (2 + n)

    where k = number of corroborating leaves, n = total leaves.

    Args:
        similarities:      Cosine similarities between the claim and each leaf.
        support_threshold: Minimum similarity (exclusive) for a leaf to count
                           as corroborating.

    Returns:
        SA score in the open interval (0, 1).
    """
    n = len(similarities)
    k = sum(1 for s in similarities if s > support_threshold)
    return (1 + k) / (2 + n)
