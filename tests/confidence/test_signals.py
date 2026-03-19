"""Tests for mara.confidence.signals — SA, CSC, LSA, and composite scoring.

All functions are pure Python — no mocking needed. Numerical values are
verified against the closed-form formulas from the README.
"""

import math

import pytest

from mara.confidence.signals import (
    LSAVerdict,
    _LSA_SCORE_MAP,
    compute_composite,
    compute_csc,
    compute_sa,
    lsa_verdict_to_score,
)


# ---------------------------------------------------------------------------
# compute_sa — Beta-Binomial posterior mean
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# compute_csc — Cross-Source Consistency
# ---------------------------------------------------------------------------

class TestComputeCSC:
    def test_empty_returns_neutral_sentinel(self):
        assert compute_csc([], support_threshold=0.72) == pytest.approx(0.5)

    def test_single_supporter_returns_neutral_sentinel(self):
        assert compute_csc([0.9], support_threshold=0.72) == pytest.approx(0.5)

    def test_zero_supporters_returns_neutral_sentinel(self):
        assert compute_csc([0.1, 0.2], support_threshold=0.72) == pytest.approx(0.5)

    def test_identical_similarities_produce_csc_of_one(self):
        """Zero variance → CV=0 → CSC=1.0."""
        sims = [0.9, 0.9, 0.9, 0.9]
        assert compute_csc(sims, 0.72) == pytest.approx(1.0)

    def test_high_variance_produces_low_csc(self):
        """Widely spread similarities → high CV → CSC closer to 0."""
        sims = [0.73, 0.99]  # very different, both above 0.72
        mean = (0.73 + 0.99) / 2
        std = math.sqrt(((0.73 - mean) ** 2 + (0.99 - mean) ** 2) / 2)
        cv = std / mean
        expected = max(0.0, min(1.0, 1.0 - cv))
        assert compute_csc(sims, 0.72) == pytest.approx(expected)

    def test_result_clamped_to_zero_when_cv_exceeds_one(self):
        """If CV > 1 the raw formula goes negative; clamp must hold at 0."""
        # Construct sims where std/mean > 1 among supporters
        sims = [0.73, 0.73, 9.0]  # 9.0 is far above threshold — extreme outlier
        result = compute_csc(sims, 0.72)
        assert result >= 0.0

    def test_result_never_exceeds_one(self):
        sims = [0.9, 0.91, 0.89, 0.9]
        assert compute_csc(sims, 0.72) <= 1.0

    def test_manual_computation_two_supporters(self):
        """Verify CSC formula with two supporters against manual calculation."""
        sims = [0.80, 0.90]  # both > 0.72
        supporting = [0.80, 0.90]
        mean = sum(supporting) / 2
        std = math.sqrt(sum((s - mean) ** 2 for s in supporting) / 2)
        cv = std / mean
        expected = max(0.0, min(1.0, 1.0 - cv))
        assert compute_csc(sims, 0.72) == pytest.approx(expected)

    def test_zero_mean_supporters_returns_neutral_sentinel(self):
        """Degenerate guard: all supporting similarities are exactly 0.0.
        This cannot occur in practice (0.0 < threshold means sim > threshold
        would never select 0.0), but the guard must still return 0.5."""
        # Patch: manually exercise the branch by using threshold=-1.0 so that
        # 0.0 similarities pass the > threshold check, creating a zero-mean set.
        result = compute_csc([0.0, 0.0], support_threshold=-1.0)
        assert result == pytest.approx(0.5)

    def test_non_supporters_excluded_from_calculation(self):
        """Low-similarity sources below threshold must not affect CSC."""
        sims_with_noise = [0.9, 0.9, 0.1, 0.05, 0.0]
        sims_without_noise = [0.9, 0.9]
        assert compute_csc(sims_with_noise, 0.72) == pytest.approx(
            compute_csc(sims_without_noise, 0.72)
        )

    def test_custom_threshold_changes_supporter_set(self):
        sims = [0.6, 0.7, 0.8, 0.9]
        # threshold=0.75: only 0.8 and 0.9 are supporters → 2 items
        csc_high_thresh = compute_csc(sims, 0.75)
        # threshold=0.55: all four are supporters
        csc_low_thresh = compute_csc(sims, 0.55)
        assert csc_high_thresh != csc_low_thresh


# ---------------------------------------------------------------------------
# lsa_verdict_to_score
# ---------------------------------------------------------------------------

class TestLSAVerdictToScore:
    def test_supported_maps_to_one(self):
        assert lsa_verdict_to_score("supported") == pytest.approx(1.0)

    def test_partially_supported_maps_to_half(self):
        assert lsa_verdict_to_score("partially_supported") == pytest.approx(0.5)

    def test_unsupported_maps_to_zero(self):
        assert lsa_verdict_to_score("unsupported") == pytest.approx(0.0)

    def test_all_verdicts_covered_by_map(self):
        """Every key in _LSA_SCORE_MAP must be handled without error."""
        for verdict in _LSA_SCORE_MAP:
            score = lsa_verdict_to_score(verdict)
            assert isinstance(score, float)

    def test_unknown_verdict_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LSA verdict"):
            lsa_verdict_to_score("definitely_supported")  # type: ignore[arg-type]

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            lsa_verdict_to_score("")  # type: ignore[arg-type]

    def test_scores_are_ordered(self):
        assert (
            lsa_verdict_to_score("unsupported")
            < lsa_verdict_to_score("partially_supported")
            < lsa_verdict_to_score("supported")
        )


# ---------------------------------------------------------------------------
# compute_composite
# ---------------------------------------------------------------------------

class TestComputeComposite:
    def test_weighted_sum(self):
        result = compute_composite(sa=0.6, csc=0.8, lsa=1.0, alpha=0.4, beta=0.2, gamma=0.4)
        expected = 0.4 * 0.6 + 0.2 * 0.8 + 0.4 * 1.0
        assert result == pytest.approx(expected)

    def test_all_zeros_returns_zero(self):
        assert compute_composite(0.0, 0.0, 0.0, 0.4, 0.2, 0.4) == pytest.approx(0.0)

    def test_all_ones_returns_one(self):
        assert compute_composite(1.0, 1.0, 1.0, 0.4, 0.2, 0.4) == pytest.approx(1.0)

    def test_result_clamped_at_zero(self):
        """Negative inputs must be clamped to 0.0, not produce negative output."""
        result = compute_composite(sa=-0.5, csc=-0.5, lsa=-0.5, alpha=0.4, beta=0.2, gamma=0.4)
        assert result == pytest.approx(0.0)

    def test_result_clamped_at_one(self):
        """Values > 1 (e.g. from floating-point error) must clamp to 1.0."""
        result = compute_composite(sa=1.1, csc=1.1, lsa=1.1, alpha=0.4, beta=0.2, gamma=0.4)
        assert result == pytest.approx(1.0)

    def test_alpha_only_weights(self):
        """With beta=gamma=0, result equals alpha*SA."""
        result = compute_composite(sa=0.7, csc=0.5, lsa=0.3, alpha=1.0, beta=0.0, gamma=0.0)
        assert result == pytest.approx(0.7)

    def test_gamma_only_weights(self):
        result = compute_composite(sa=0.5, csc=0.5, lsa=0.8, alpha=0.0, beta=0.0, gamma=1.0)
        assert result == pytest.approx(0.8)

    def test_equal_weights(self):
        result = compute_composite(
            sa=0.6, csc=0.9, lsa=0.3,
            alpha=1 / 3, beta=1 / 3, gamma=1 / 3,
        )
        expected = (0.6 + 0.9 + 0.3) / 3
        assert result == pytest.approx(expected)

    def test_default_weights_from_readme(self):
        """Verify the README default weights (α=0.4, β=0.2, γ=0.4) produce expected output."""
        result = compute_composite(sa=0.8, csc=0.6, lsa=0.9, alpha=0.4, beta=0.2, gamma=0.4)
        expected = 0.4 * 0.8 + 0.2 * 0.6 + 0.4 * 0.9
        assert result == pytest.approx(expected)
