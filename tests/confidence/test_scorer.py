"""Tests for mara.confidence.scorer — score_claim and ScoredClaim.

The embedding model IS loaded (real model) since it's fast and required
for integration confidence. All scoring is deterministic given fixed texts.
"""

import pytest

from mara.confidence.scorer import ScoredClaim, cosine_similarity, score_claim
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig()


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_unit_vectors_return_one(self):
        import numpy as np
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        import numpy as np
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_unit_vectors_return_negative_one(self):
        import numpy as np
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_returns_python_float(self):
        import numpy as np
        a = np.array([1.0, 0.0], dtype=np.float32)
        result = cosine_similarity(a, a)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# score_claim — empty leaves (degenerate case)
# ---------------------------------------------------------------------------

class TestScoreClaimEmptyLeaves:
    def test_returns_scored_claim(self, config):
        result = score_claim("A claim.", [], [], config)
        assert isinstance(result, ScoredClaim)

    def test_confidence_is_prior_mean_with_no_leaves(self, config):
        """Beta(1,1) prior mean with 0 leaves = (1+0)/(2+0) = 0.5."""
        result = score_claim("A claim.", [], [], config)
        assert result.confidence == pytest.approx(0.5)

    def test_corroborating_is_zero(self, config):
        result = score_claim("A claim.", [], [], config)
        assert result.corroborating == 0

    def test_n_leaves_is_zero(self, config):
        result = score_claim("A claim.", [], [], config)
        assert result.n_leaves == 0

    def test_similarities_is_empty(self, config):
        result = score_claim("A claim.", [], [], config)
        assert result.similarities == []

    def test_source_indices_preserved(self, config):
        result = score_claim("A claim.", [], [], config)
        assert result.source_indices == []


# ---------------------------------------------------------------------------
# score_claim — with real leaf texts
# ---------------------------------------------------------------------------

class TestScoreClaimWithLeaves:
    def test_returns_scored_claim(self, config):
        result = score_claim(
            claim_text="The sky is blue.",
            all_leaf_texts=["The sky appears blue due to Rayleigh scattering."],
            source_indices=[0],
            config=config,
        )
        assert isinstance(result, ScoredClaim)

    def test_similarities_length_matches_leaves(self, config):
        leaves = ["Source one text.", "Source two text.", "Source three text."]
        result = score_claim("A claim.", leaves, [0, 1, 2], config)
        assert len(result.similarities) == 3

    def test_n_leaves_matches_leaf_count(self, config):
        leaves = ["text one", "text two", "text three"]
        result = score_claim("A claim.", leaves, [0, 1, 2], config)
        assert result.n_leaves == 3

    def test_corroborating_is_count_above_threshold(self, config):
        """corroborating must equal the number of sims > threshold."""
        leaves = ["text one", "text two"]
        result = score_claim("A claim.", leaves, [0, 1], config)
        threshold = config.similarity_support_threshold
        expected_k = sum(1 for s in result.similarities if s > threshold)
        assert result.corroborating == expected_k

    def test_confidence_equals_beta_binomial(self, config):
        """confidence must equal (1 + k) / (2 + k) exactly."""
        leaves = ["text one", "text two", "text three"]
        result = score_claim("A claim.", leaves, [0], config)
        expected = (1 + result.corroborating) / (2 + result.corroborating)
        assert result.confidence == pytest.approx(expected)

    def test_source_indices_preserved(self, config):
        result = score_claim(
            "A claim.",
            ["source a", "source b"],
            [5, 12],
            config,
        )
        assert result.source_indices == [5, 12]

    def test_claim_text_preserved(self, config):
        claim = "The Earth orbits the Sun."
        result = score_claim(claim, ["Astronomy fact."], [0], config)
        assert result.text == claim

    def test_confidence_within_unit_interval(self, config):
        result = score_claim(
            "Climate change is driven by CO2 emissions.",
            ["CO2 and other greenhouse gases cause warming.", "Fossil fuels emit CO2."],
            [0, 1],
            config,
        )
        assert 0.0 < result.confidence < 1.0

    def test_more_corroborating_leaves_yield_higher_confidence(self, config):
        """A claim matching many leaves scores higher than one matching few."""
        claim = "The sky is blue due to Rayleigh scattering of sunlight."
        many_leaves = [claim] * 10  # identical to claim — maximum corroboration
        few_leaves = ["unrelated text about cooking recipes"] * 10
        high = score_claim(claim, many_leaves, [0], config)
        low = score_claim(claim, few_leaves, [0], config)
        assert high.confidence > low.confidence

    def test_custom_threshold_changes_corroborating_count(self, config):
        """Lowering the threshold admits more corroborating leaves."""
        claim = "Neural networks learn from data."
        leaves = [
            "Deep learning models train on examples.",
            "unrelated text about medieval history",
        ]
        low_threshold_config = ResearchConfig(similarity_support_threshold=0.1)
        high_threshold_config = ResearchConfig(similarity_support_threshold=0.99)
        low = score_claim(claim, leaves, [0], low_threshold_config)
        high = score_claim(claim, leaves, [0], high_threshold_config)
        assert low.corroborating >= high.corroborating


# ---------------------------------------------------------------------------
# ScoredClaim dataclass
# ---------------------------------------------------------------------------

class TestScoredClaim:
    def test_all_fields_accessible(self):
        claim = ScoredClaim(
            text="test",
            source_indices=[0, 1],
            confidence=0.6,
            corroborating=3,
            n_leaves=10,
            similarities=[0.8, 0.9],
        )
        assert claim.text == "test"
        assert claim.source_indices == [0, 1]
        assert claim.confidence == 0.6
        assert claim.corroborating == 3
        assert claim.n_leaves == 10
        assert claim.similarities == [0.8, 0.9]

    def test_similarities_defaults_to_empty_list(self):
        claim = ScoredClaim(
            text="test", source_indices=[], confidence=0.5,
            corroborating=0, n_leaves=0,
        )
        assert claim.similarities == []

    def test_similarities_default_not_shared_between_instances(self):
        c1 = ScoredClaim(text="a", source_indices=[], confidence=0.5, corroborating=0, n_leaves=0)
        c2 = ScoredClaim(text="b", source_indices=[], confidence=0.5, corroborating=0, n_leaves=0)
        c1.similarities.append(0.9)
        assert c2.similarities == []
