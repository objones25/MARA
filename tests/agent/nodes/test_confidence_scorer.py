"""Tests for mara.agent.nodes.confidence_scorer.

All embedding calls are mocked — no real model inference is made.
Tests cover: empty claims, return shape, correct leaf texts passed to
score_claim, confidence values, and per-claim corroborating/n_leaves fields.
"""

import pytest
import numpy as np

from mara.agent.nodes.confidence_scorer import confidence_scorer
from mara.agent.state import Claim
from mara.confidence.scorer import ScoredClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_embed(mocker, n_texts: int, dim: int = 4):
    """Patch embed() to return random unit vectors of shape (n_texts, dim)."""
    vecs = np.random.rand(n_texts, dim).astype(np.float32)
    # L2-normalize each row
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    mock = mocker.patch(
        "mara.confidence.scorer.embed",
        return_value=vecs,
    )
    return mock


# ---------------------------------------------------------------------------
# confidence_scorer node
# ---------------------------------------------------------------------------


class TestConfidenceScorerNode:
    async def test_empty_claims_returns_empty_scored(self, mocker, make_mara_state):
        result = await confidence_scorer(make_mara_state(extracted_claims=[], retrieved_leaves=[]), config={})
        assert result == {"scored_claims": []}

    async def test_returns_scored_claims_key(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="some claim", source_indices=[0])
        # 1 claim + 1 leaf = 2 texts → 2 embeddings
        _mock_embed(mocker, n_texts=2)
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[leaf]), config={})
        assert "scored_claims" in result

    async def test_one_claim_produces_one_scored_claim(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="supporting text", index=0)
        claim = Claim(text="test claim", source_indices=[0])
        _mock_embed(mocker, n_texts=2)
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[leaf]), config={})
        assert len(result["scored_claims"]) == 1

    async def test_multiple_claims_produce_multiple_scored_claims(self, mocker, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://a.com/{i}", text=f"text {i}", index=i) for i in range(3)]
        claims = [Claim(text=f"claim {i}", source_indices=[i]) for i in range(3)]
        # Each call: 1 claim + 3 leaves = 4 embeddings
        mock_embed = mocker.patch("mara.confidence.scorer.embed")
        mock_embed.side_effect = [
            np.eye(4, dtype=np.float32)[:4],  # call 1
            np.eye(4, dtype=np.float32)[:4],  # call 2
            np.eye(4, dtype=np.float32)[:4],  # call 3
        ]
        result = await confidence_scorer(make_mara_state(extracted_claims=claims, retrieved_leaves=leaves), config={})
        assert len(result["scored_claims"]) == 3

    async def test_scored_claim_has_text(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="verifiable claim", source_indices=[0])
        _mock_embed(mocker, n_texts=2)
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[leaf]), config={})
        assert result["scored_claims"][0].text == "verifiable claim"

    async def test_scored_claim_confidence_in_open_interval(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="a claim", source_indices=[0])
        _mock_embed(mocker, n_texts=2)
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[leaf]), config={})
        score = result["scored_claims"][0].confidence
        assert 0.0 < score < 1.0

    async def test_scored_claim_has_corroborating_field(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="a claim", source_indices=[0])
        _mock_embed(mocker, n_texts=2)
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[leaf]), config={})
        sc = result["scored_claims"][0]
        assert hasattr(sc, "corroborating")
        assert isinstance(sc.corroborating, int)

    async def test_scored_claim_n_leaves_matches_retrieved(self, mocker, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://a.com/{i}", text=f"text {i}", index=i) for i in range(5)]
        claim = Claim(text="a claim", source_indices=[0])
        _mock_embed(mocker, n_texts=6)  # 1 claim + 5 leaves
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=leaves), config={})
        assert result["scored_claims"][0].n_leaves == 5

    async def test_all_leaf_texts_passed_to_embed(self, mocker, make_mara_state, make_merkle_leaf):
        leaves = [
            make_merkle_leaf(url="https://a.com/0", text="text zero", index=0),
            make_merkle_leaf(url="https://a.com/1", text="text one", index=1),
        ]
        claim = Claim(text="my claim", source_indices=[0])
        mock_embed = mocker.patch(
            "mara.confidence.scorer.embed",
            return_value=np.eye(4, 4, dtype=np.float32)[:3],
        )
        await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=leaves), config={})
        call_texts = mock_embed.call_args.args[0]
        # First text is the claim; remaining are all leaf texts
        assert call_texts[0] == "my claim"
        assert "text zero" in call_texts
        assert "text one" in call_texts

    async def test_claim_with_no_sources_still_scores(self, mocker, make_mara_state, make_merkle_leaf):
        """A claim with source_indices=[] scores against all leaves anyway."""
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="orphan claim", source_indices=[])
        _mock_embed(mocker, n_texts=2)
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[leaf]), config={})
        assert len(result["scored_claims"]) == 1
        assert 0.0 < result["scored_claims"][0].confidence < 1.0

    async def test_no_leaves_yields_prior_mean(self, mocker, make_mara_state):
        """With no leaves, confidence = Beta(1,1) prior mean = 0.5."""
        claim = Claim(text="a claim", source_indices=[])
        result = await confidence_scorer(make_mara_state(extracted_claims=[claim], retrieved_leaves=[]), config={})
        assert result["scored_claims"][0].confidence == pytest.approx(0.5)
