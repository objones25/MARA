"""Tests for mara.confidence.signals — Beta-Binomial Source Agreement Rate.

compute_sa is a pure function; no mocking needed.
Numerical values are verified against the closed-form formula (1+k)/(2+k),
where k = corroborating leaves only (non-corroborating leaves are excluded
from the denominator — their silence is not evidence of contradiction).
"""

import pytest

from mara.confidence.signals import compute_sa


class TestComputeSA:
    """SA = (1 + k) / (2 + k), where k = corroborating leaves only."""

    def test_empty_returns_prior_mean(self):
        """Beta(1,1) prior mean with zero observations = (1+0)/(2+0) = 0.5."""
        assert compute_sa([], support_threshold=0.72) == pytest.approx(0.5)

    def test_all_supporting(self):
        """k=3: (1+3)/(2+3) = 0.8."""
        sims = [0.8, 0.9, 0.95]  # all > 0.72
        expected = (1 + 3) / (2 + 3)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_none_supporting(self):
        """k=0: non-corroborating leaves excluded → SA = (1+0)/(2+0) = 0.5 (prior mean)."""
        sims = [0.1, 0.2, 0.3]  # all < 0.72 → k=0
        expected = (1 + 0) / (2 + 0)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_partial_support(self):
        sims = [0.8, 0.5, 0.9, 0.1]  # 2 above threshold → k=2
        expected = (1 + 2) / (2 + 2)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_single_supporting_source(self):
        sims = [0.9]
        expected = (1 + 1) / (2 + 1)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_single_non_supporting_source(self):
        """Non-corroborating leaf excluded → same as no data: (1+0)/(2+0) = 0.5."""
        sims = [0.1]
        expected = (1 + 0) / (2 + 0)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_threshold_boundary_exclusive(self):
        """Similarity exactly equal to threshold does NOT count as supporting."""
        sims = [0.72]
        expected = (1 + 0) / (2 + 0)  # not > 0.72 → k=0
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_just_above_threshold_counts(self):
        sims = [0.7201]
        expected = (1 + 1) / (2 + 1)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_result_always_in_open_interval(self):
        """Laplace smoothing guarantees SA is never exactly 0 or 1."""
        assert 0.0 < compute_sa([], 0.72) < 1.0
        assert 0.0 < compute_sa([0.9] * 100, 0.72) < 1.0
        assert 0.0 < compute_sa([0.1] * 100, 0.72) < 1.0

    def test_large_k_approaches_one(self):
        """With many corroborating sources, SA approaches 1 asymptotically."""
        sims = [0.9] * 1000
        result = compute_sa(sims, 0.72)
        expected = (1 + 1000) / (2 + 1000)
        assert result == pytest.approx(expected)
        assert result > 0.99

    def test_custom_threshold(self):
        sims = [0.5, 0.6, 0.7, 0.8]
        # with threshold=0.65: 0.7 and 0.8 are above → k=2
        expected = (1 + 2) / (2 + 2)
        assert compute_sa(sims, 0.65) == pytest.approx(expected)

    def test_more_corroborating_sources_yield_higher_score(self):
        """Adding corroborating sources strictly increases SA."""
        low = compute_sa([0.8], 0.72)          # k=1
        high = compute_sa([0.8, 0.9], 0.72)    # k=2
        assert high > low

    def test_k_count_matches_threshold(self):
        """k is the count of similarities strictly above the threshold."""
        sims = [0.73, 0.72, 0.71, 0.80]
        # 0.73 and 0.80 are > 0.72; 0.72 and 0.71 are not (exclusive)
        expected = (1 + 2) / (2 + 2)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)
