"""Tests for mara.confidence.scorer — score_claim and ScoredClaim.

The LSA callable is always injected as a mock so these tests run without
a live LLM. The embedding model IS loaded (real model) since it's fast
and required for integration confidence.
"""

import pytest

from mara.confidence.scorer import ScoredClaim, cosine_similarity, score_claim
from mara.confidence.signals import compute_composite, compute_csc, compute_sa
from mara.config import ConfidenceWeights, ResearchConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> ResearchConfig:
    return ResearchConfig()


def _lsa_supported(_claim: str, _sources: list[str]) -> str:
    return "supported"


def _lsa_partially(_claim: str, _sources: list[str]) -> str:
    return "partially_supported"


def _lsa_unsupported(_claim: str, _sources: list[str]) -> str:
    return "unsupported"


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
# score_claim — input validation
# ---------------------------------------------------------------------------

class TestScoreClaimValidation:
    def test_mismatched_lengths_raise_value_error(self, config):
        with pytest.raises(ValueError, match="length"):
            score_claim(
                claim_text="A claim.",
                source_texts=["source one", "source two"],
                source_indices=[0],          # length mismatch
                lsa_callable=_lsa_supported,
                config=config,
            )

    def test_empty_texts_empty_indices_is_valid(self, config):
        result = score_claim(
            claim_text="A claim.",
            source_texts=[],
            source_indices=[],
            lsa_callable=_lsa_supported,
            config=config,
        )
        assert isinstance(result, ScoredClaim)


# ---------------------------------------------------------------------------
# score_claim — empty sources (degenerate case)
# ---------------------------------------------------------------------------

class TestScoreClaimEmptySources:
    def test_returns_scored_claim(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        assert isinstance(result, ScoredClaim)

    def test_sa_is_prior_mean(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        expected_sa = compute_sa([], config.similarity_support_threshold)
        assert result.sa == pytest.approx(expected_sa)

    def test_csc_is_neutral_sentinel(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        expected_csc = compute_csc([], config.similarity_support_threshold)
        assert result.csc == pytest.approx(expected_csc)

    def test_lsa_supported_maps_correctly(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        assert result.lsa == pytest.approx(1.0)

    def test_lsa_unsupported_maps_correctly(self, config):
        result = score_claim("A claim.", [], [], _lsa_unsupported, config)
        assert result.lsa == pytest.approx(0.0)

    def test_similarities_is_empty_list(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        assert result.similarities == []

    def test_source_indices_is_empty_list(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        assert result.source_indices == []

    def test_confidence_uses_config_weights(self, config):
        result = score_claim("A claim.", [], [], _lsa_supported, config)
        w = config.confidence_weights
        expected = compute_composite(
            sa=result.sa, csc=result.csc, lsa=result.lsa,
            alpha=w.alpha, beta=w.beta, gamma=w.gamma,
        )
        assert result.confidence == pytest.approx(expected)


# ---------------------------------------------------------------------------
# score_claim — with real sources
# ---------------------------------------------------------------------------

class TestScoreClaimWithSources:
    def test_returns_scored_claim(self, config):
        result = score_claim(
            claim_text="The sky is blue.",
            source_texts=["The sky appears blue due to Rayleigh scattering."],
            source_indices=[0],
            lsa_callable=_lsa_supported,
            config=config,
        )
        assert isinstance(result, ScoredClaim)

    def test_similarities_length_matches_sources(self, config):
        sources = ["Source one text.", "Source two text.", "Source three text."]
        result = score_claim("A claim.", sources, [0, 1, 2], _lsa_supported, config)
        assert len(result.similarities) == 3

    def test_source_indices_preserved(self, config):
        result = score_claim(
            "A claim.",
            ["source a", "source b"],
            [5, 12],
            _lsa_supported,
            config,
        )
        assert result.source_indices == [5, 12]

    def test_claim_text_preserved(self, config):
        claim = "The Earth orbits the Sun."
        result = score_claim(claim, ["Astronomy fact."], [0], _lsa_supported, config)
        assert result.text == claim

    def test_confidence_within_unit_interval(self, config):
        result = score_claim(
            "Climate change is driven by CO2 emissions.",
            ["CO2 and other greenhouse gases cause warming.", "Fossil fuels emit CO2."],
            [0, 1],
            _lsa_supported,
            config,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_lsa_callable_receives_correct_arguments(self, config):
        received = {}

        def capturing_lsa(claim: str, sources: list[str]) -> str:
            received["claim"] = claim
            received["sources"] = sources
            return "supported"

        claim = "Test claim."
        sources = ["Source A.", "Source B."]
        score_claim(claim, sources, [0, 1], capturing_lsa, config)
        assert received["claim"] == claim
        assert received["sources"] == sources

    def test_confidence_uses_config_weights_not_hardcoded(self, config):
        """Changing confidence weights in config changes the composite score."""
        result_default = score_claim(
            "A claim about science.",
            ["Scientific fact about the world."],
            [0],
            _lsa_supported,
            config,
        )
        custom_config = ResearchConfig(
            confidence_weights=ConfidenceWeights(alpha=0.1, beta=0.1, gamma=0.8)
        )
        result_custom = score_claim(
            "A claim about science.",
            ["Scientific fact about the world."],
            [0],
            _lsa_supported,
            custom_config,
        )
        # Same SA/CSC but different weights → different composite
        assert result_default.confidence != pytest.approx(result_custom.confidence)

    def test_supported_claim_scores_higher_than_unsupported(self, config):
        """With all else equal, 'supported' LSA verdict must yield higher confidence."""
        kwargs = dict(
            claim_text="The sky is blue.",
            source_texts=["The sky is blue due to Rayleigh scattering."],
            source_indices=[0],
            config=config,
        )
        high = score_claim(**kwargs, lsa_callable=_lsa_supported)
        low = score_claim(**kwargs, lsa_callable=_lsa_unsupported)
        assert high.confidence > low.confidence

    def test_sa_computed_from_config_threshold(self, config):
        """SA must be computed using config.similarity_support_threshold."""
        result = score_claim(
            "A claim.",
            ["supporting text that is highly related"],
            [0],
            _lsa_supported,
            config,
        )
        # SA must be in the valid Beta-Binomial range for n=1
        # (1+0)/(2+1) = 0.333 ≤ SA ≤ (1+1)/(2+1) = 0.667
        assert (1 + 0) / (2 + 1) <= result.sa <= (1 + 1) / (2 + 1)


# ---------------------------------------------------------------------------
# ScoredClaim dataclass
# ---------------------------------------------------------------------------

class TestScoredClaim:
    def test_all_fields_accessible(self):
        claim = ScoredClaim(
            text="test",
            source_indices=[0, 1],
            sa=0.6,
            csc=0.7,
            lsa=1.0,
            confidence=0.8,
            similarities=[0.8, 0.9],
        )
        assert claim.text == "test"
        assert claim.source_indices == [0, 1]
        assert claim.sa == 0.6
        assert claim.csc == 0.7
        assert claim.lsa == 1.0
        assert claim.confidence == 0.8
        assert claim.similarities == [0.8, 0.9]

    def test_similarities_defaults_to_empty_list(self):
        claim = ScoredClaim(
            text="test", source_indices=[], sa=0.5, csc=0.5, lsa=0.5, confidence=0.5
        )
        assert claim.similarities == []

    def test_similarities_default_not_shared_between_instances(self):
        c1 = ScoredClaim(text="a", source_indices=[], sa=0.5, csc=0.5, lsa=0.5, confidence=0.5)
        c2 = ScoredClaim(text="b", source_indices=[], sa=0.5, csc=0.5, lsa=0.5, confidence=0.5)
        c1.similarities.append(0.9)
        assert c2.similarities == []
