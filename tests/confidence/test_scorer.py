"""Tests for mara.confidence.scorer — score_claim and ScoredClaim.

The embedding model IS loaded (real model) since it's fast and required
for integration confidence. All scoring is deterministic given fixed texts.

score_claim now accepts pre-computed embeddings; tests call embed() directly
to obtain them, making the embedding step explicit in each test.
"""

import numpy as np
import pytest

from mara.confidence.embeddings import embed
from mara.confidence.scorer import ScoredClaim, cosine_similarity, score_claim
from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _embs(texts: list[str], config: ResearchConfig) -> np.ndarray:
    """Embed a list of texts using the config's embedding model."""
    return embed(texts, config.embedding_model, config.hf_token)


def _score(
    claim: str,
    leaves: list[str],
    urls: list[str] | None = None,
    source_indices: list[int] | None = None,
    config: ResearchConfig | None = None,
) -> ScoredClaim:
    """Convenience wrapper: embed claim + leaves, then call score_claim."""
    if config is None:
        config = ResearchConfig(leaf_db_enabled=False)
    if source_indices is None:
        source_indices = []
    if urls is None:
        urls = [f"https://source-{i}.com" for i in range(len(leaves))]

    if not leaves:
        claim_emb = _embs([claim], config)[0]
        leaf_embs = np.empty((0, claim_emb.shape[0]), dtype=np.float32)
        return score_claim(claim, claim_emb, leaf_embs, urls, source_indices, config)

    all_embs = _embs([claim] + leaves, config)
    return score_claim(claim, all_embs[0], all_embs[1:], urls, source_indices, config)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_unit_vectors_return_one(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_unit_vectors_return_negative_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_returns_python_float(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        result = cosine_similarity(a, a)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# score_claim — empty leaves (degenerate case)
# ---------------------------------------------------------------------------


class TestScoreClaimEmptyLeaves:
    def test_returns_scored_claim(self):
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], config=config)
        assert isinstance(result, ScoredClaim)

    def test_confidence_is_prior_mean_with_no_leaves(self):
        """Beta(1,1) prior mean with 0 leaves = (1+0)/(2+0) = 0.5."""
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], config=config)
        assert result.confidence == pytest.approx(0.5)

    def test_corroborating_is_zero(self):
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], config=config)
        assert result.corroborating == 0

    def test_n_leaves_is_zero(self):
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], config=config)
        assert result.n_leaves == 0

    def test_n_unique_urls_is_zero(self):
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], config=config)
        assert result.n_unique_urls == 0

    def test_similarities_is_empty(self):
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], config=config)
        assert result.similarities == []

    def test_source_indices_preserved(self):
        config = ResearchConfig(leaf_db_enabled=False)
        result = _score("A claim.", [], source_indices=[], config=config)
        assert result.source_indices == []


# ---------------------------------------------------------------------------
# score_claim — with real leaf texts
# ---------------------------------------------------------------------------


class TestScoreClaimWithLeaves:
    def test_returns_scored_claim(self):
        result = _score(
            "The sky is blue.",
            ["The sky appears blue due to Rayleigh scattering."],
        )
        assert isinstance(result, ScoredClaim)

    def test_similarities_length_matches_leaves(self):
        leaves = ["Source one text.", "Source two text.", "Source three text."]
        result = _score("A claim.", leaves, source_indices=[0, 1, 2])
        assert len(result.similarities) == 3

    def test_n_leaves_matches_leaf_count(self):
        leaves = ["text one", "text two", "text three"]
        result = _score("A claim.", leaves, source_indices=[0, 1, 2])
        assert result.n_leaves == 3

    def test_n_unique_urls_matches_unique_url_count(self):
        leaves = ["text one", "text two", "text three"]
        urls = ["https://a.com", "https://b.com", "https://c.com"]
        result = _score("A claim.", leaves, urls=urls)
        assert result.n_unique_urls == 3

    def test_corroborating_is_unique_url_count_above_threshold(self):
        """corroborating counts unique URLs (not chunks) above threshold."""
        config = ResearchConfig(leaf_db_enabled=False)
        leaves = ["text one", "text two"]
        urls = ["https://a.com", "https://b.com"]
        result = _score("A claim.", leaves, urls=urls, config=config)
        threshold = config.similarity_support_threshold
        # corroborating = number of unique URLs with max_sim > threshold
        url_max = {}
        for url, sim in zip(urls, result.similarities):
            if url not in url_max or sim > url_max[url]:
                url_max[url] = sim
        expected_k = sum(1 for s in url_max.values() if s > threshold)
        assert result.corroborating == expected_k

    def test_confidence_equals_beta_binomial(self):
        """confidence must equal (1 + k) / (2 + k) where k = corroborating."""
        leaves = ["text one", "text two", "text three"]
        result = _score("A claim.", leaves)
        expected = (1 + result.corroborating) / (2 + result.corroborating)
        assert result.confidence == pytest.approx(expected)

    def test_source_indices_preserved(self):
        result = _score(
            "A claim.",
            ["source a", "source b"],
            source_indices=[5, 12],
        )
        assert result.source_indices == [5, 12]

    def test_claim_text_preserved(self):
        claim = "The Earth orbits the Sun."
        result = _score(claim, ["Astronomy fact."])
        assert result.text == claim

    def test_confidence_within_unit_interval(self):
        result = _score(
            "Climate change is driven by CO2 emissions.",
            ["CO2 and other greenhouse gases cause warming.", "Fossil fuels emit CO2."],
        )
        assert 0.0 < result.confidence < 1.0

    def test_more_corroborating_leaves_yield_higher_confidence(self):
        """A claim matching many unique sources scores higher than one matching few."""
        claim = "The sky is blue due to Rayleigh scattering of sunlight."
        many_leaves = [claim] * 10
        many_urls = [f"https://source-{i}.com" for i in range(10)]
        few_leaves = ["unrelated text about cooking recipes"] * 10
        few_urls = [f"https://food-{i}.com" for i in range(10)]
        high = _score(claim, many_leaves, urls=many_urls)
        low = _score(claim, few_leaves, urls=few_urls)
        assert high.confidence > low.confidence

    def test_custom_threshold_changes_corroborating_count(self):
        """Lowering the threshold admits more corroborating sources."""
        claim = "Neural networks learn from data."
        leaves = [
            "Deep learning models train on examples.",
            "unrelated text about medieval history",
        ]
        urls = ["https://a.com", "https://b.com"]
        low_threshold_config = ResearchConfig(
            similarity_support_threshold=0.1, leaf_db_enabled=False
        )
        high_threshold_config = ResearchConfig(
            similarity_support_threshold=0.99, leaf_db_enabled=False
        )
        low = _score(claim, leaves, urls=urls, config=low_threshold_config)
        high = _score(claim, leaves, urls=urls, config=high_threshold_config)
        assert low.corroborating >= high.corroborating


# ---------------------------------------------------------------------------
# score_claim — source-deduplicated k (multiple chunks per URL)
# ---------------------------------------------------------------------------


class TestScoreClaimSourceDedup:
    def test_multiple_chunks_same_url_counted_once(self):
        """3 identical chunks from the same URL → corroborating ≤ 1 (one source vote)."""
        config = ResearchConfig(similarity_support_threshold=0.5, leaf_db_enabled=False)
        claim = "The sky is blue due to Rayleigh scattering."
        leaves = [claim] * 3  # identical to claim → high similarity
        urls = ["https://example.com/article"] * 3
        result = _score(claim, leaves, urls=urls, config=config)
        assert result.corroborating <= 1

    def test_unique_urls_each_counted(self):
        """Same text, 3 distinct URLs → corroborating equals chunk-level count."""
        config = ResearchConfig(similarity_support_threshold=0.5, leaf_db_enabled=False)
        claim = "The sky is blue due to Rayleigh scattering."
        leaves = [claim] * 3
        urls = ["https://a.com", "https://b.com", "https://c.com"]
        result = _score(claim, leaves, urls=urls, config=config)
        # All unique → corroborating counts each separately
        assert result.corroborating == result.n_unique_urls

    def test_source_dedup_reduces_corroborating_vs_naive_chunk_count(self):
        """4 chunks from 2 URLs: dedup corroborating ≤ naive chunk count."""
        config = ResearchConfig(similarity_support_threshold=0.5, leaf_db_enabled=False)
        claim = "The sky is blue due to Rayleigh scattering."
        leaves = [claim] * 4
        urls_dedup = ["https://a.com", "https://a.com", "https://b.com", "https://b.com"]
        urls_unique = [f"https://source-{i}.com" for i in range(4)]
        result_dedup = _score(claim, leaves, urls=urls_dedup, config=config)
        result_unique = _score(claim, leaves, urls=urls_unique, config=config)
        assert result_dedup.corroborating <= result_unique.corroborating

    def test_n_leaves_is_chunk_count_not_url_count(self):
        """n_leaves always reflects total chunk count, regardless of dedup."""
        config = ResearchConfig(leaf_db_enabled=False)
        leaves = ["Some text."] * 6
        urls = ["https://a.com"] * 6
        result = _score("A claim.", leaves, urls=urls, config=config)
        assert result.n_leaves == 6

    def test_n_unique_urls_reflects_distinct_url_count(self):
        """n_unique_urls counts distinct URLs in the scoring pool."""
        config = ResearchConfig(leaf_db_enabled=False)
        leaves = ["Some text."] * 6
        urls = ["https://a.com"] * 3 + ["https://b.com"] * 3
        result = _score("A claim.", leaves, urls=urls, config=config)
        assert result.n_unique_urls == 2


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
            n_unique_urls=5,
            similarities=[0.8, 0.9],
        )
        assert claim.text == "test"
        assert claim.source_indices == [0, 1]
        assert claim.confidence == 0.6
        assert claim.corroborating == 3
        assert claim.n_leaves == 10
        assert claim.n_unique_urls == 5
        assert claim.similarities == [0.8, 0.9]

    def test_n_unique_urls_defaults_to_zero(self):
        claim = ScoredClaim(
            text="test", source_indices=[], confidence=0.5,
            corroborating=0, n_leaves=0,
        )
        assert claim.n_unique_urls == 0

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
