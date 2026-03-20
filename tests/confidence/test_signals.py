"""Tests for mara.confidence.signals — Beta-Binomial Source Agreement Rate.

compute_sa is a pure function; no mocking needed.
Numerical values are verified against the closed-form formula (1+k)/(2+n).
"""

import pytest

from mara.confidence.signals import compute_sa


class TestComputeSA:
    """SA = (1 + k) / (2 + n), where k = supporters, n = total."""

    def test_empty_returns_prior_mean(self):
        """Beta(1,1) prior mean with zero observations = (1+0)/(2+0) = 0.5."""
        assert compute_sa([], support_threshold=0.72) == pytest.approx(0.5)

    def test_all_supporting(self):
        """k=n: (1+n)/(2+n) approaches 1 but never reaches it."""
        sims = [0.8, 0.9, 0.95]  # all > 0.72
        expected = (1 + 3) / (2 + 3)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_none_supporting(self):
        """k=0: (1+0)/(2+n) — Laplace smoothing keeps result above 0."""
        sims = [0.1, 0.2, 0.3]  # all < 0.72
        expected = (1 + 0) / (2 + 3)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_partial_support(self):
        sims = [0.8, 0.5, 0.9, 0.1]  # 2 above threshold
        expected = (1 + 2) / (2 + 4)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_single_supporting_source(self):
        sims = [0.9]
        expected = (1 + 1) / (2 + 1)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_single_non_supporting_source(self):
        sims = [0.1]
        expected = (1 + 0) / (2 + 1)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)

    def test_threshold_boundary_exclusive(self):
        """Similarity exactly equal to threshold does NOT count as supporting."""
        sims = [0.72]
        expected = (1 + 0) / (2 + 1)  # not > 0.72
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

    def test_large_sample_approaches_empirical_rate(self):
        """With large n, posterior mean ≈ k/n."""
        n = 1000
        sims = [0.9] * 700 + [0.1] * 300  # 70% supporting
        result = compute_sa(sims, 0.72)
        assert result == pytest.approx(701 / 1002)

    def test_custom_threshold(self):
        sims = [0.5, 0.6, 0.7, 0.8]
        # with threshold=0.65: 0.7 and 0.8 are above → k=2
        expected = (1 + 2) / (2 + 4)
        assert compute_sa(sims, 0.65) == pytest.approx(expected)

    def test_more_corroborating_sources_yield_higher_score(self):
        """Adding corroborating sources strictly increases SA."""
        low = compute_sa([0.8], 0.72)          # k=1, n=1
        high = compute_sa([0.8, 0.9], 0.72)    # k=2, n=2
        assert high > low

    def test_k_count_matches_threshold(self):
        """k is the count of similarities strictly above the threshold."""
        sims = [0.73, 0.72, 0.71, 0.80]
        # 0.73 and 0.80 are > 0.72; 0.72 is not (exclusive)
        expected = (1 + 2) / (2 + 4)
        assert compute_sa(sims, 0.72) == pytest.approx(expected)
