"""Tests for mara.agent.nodes.confidence_scorer.

All embedding calls are mocked — no real model inference is made.
Tests cover: empty claims, empty leaves, return shape, correct leaf texts
passed to embed, confidence values, and per-claim corroborating/n_leaves fields.

The node now uses merkle_leaves (not retrieved_leaves) and makes two embed()
calls per run: one for all leaves, one for all claims.
"""

import pytest
import numpy as np

from mara.agent.nodes.confidence_scorer import confidence_scorer
from mara.agent.state import Claim
from mara.confidence.scorer import ScoredClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf_embs(n: int, dim: int = 4) -> np.ndarray:
    """Return n random unit vectors of shape (n, dim)."""
    if n == 0:
        return np.empty((0, dim), dtype=np.float32)
    vecs = np.random.rand(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _make_claim_embs(n: int, dim: int = 4) -> np.ndarray:
    """Return n random unit vectors of shape (n, dim)."""
    if n == 0:
        return np.empty((0, dim), dtype=np.float32)
    vecs = np.random.rand(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def _mock_embed(mocker, n_leaves: int, n_claims: int, dim: int = 4):
    """Mock embed() to return leaf embs on first call, claim embs on second."""
    leaf_vecs = _make_leaf_embs(n_leaves, dim)
    claim_vecs = _make_claim_embs(n_claims, dim)
    mock = mocker.patch(
        "mara.agent.nodes.confidence_scorer.embed",
        side_effect=[leaf_vecs, claim_vecs],
    )
    return mock


# ---------------------------------------------------------------------------
# confidence_scorer node
# ---------------------------------------------------------------------------


class TestConfidenceScorerNode:
    async def test_empty_claims_returns_empty_scored(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[], merkle_leaves=[leaf]), config={}
        )
        assert result == {"scored_claims": []}

    async def test_empty_leaves_returns_prior_mean(self, mocker, make_mara_state):
        """With no leaves, confidence = Beta(1,1) prior mean = 0.5."""
        claim = Claim(text="a claim", source_indices=[])
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[]), config={}
        )
        assert result["scored_claims"][0].confidence == pytest.approx(0.5)

    async def test_returns_scored_claims_key(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="some claim", source_indices=[0])
        _mock_embed(mocker, n_leaves=1, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        assert "scored_claims" in result

    async def test_one_claim_produces_one_scored_claim(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="supporting text", index=0)
        claim = Claim(text="test claim", source_indices=[0])
        _mock_embed(mocker, n_leaves=1, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        assert len(result["scored_claims"]) == 1

    async def test_multiple_claims_produce_multiple_scored_claims(self, mocker, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://a.com/{i}", text=f"text {i}", index=i) for i in range(3)]
        claims = [Claim(text=f"claim {i}", source_indices=[i]) for i in range(3)]
        _mock_embed(mocker, n_leaves=3, n_claims=3)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=claims, merkle_leaves=leaves), config={}
        )
        assert len(result["scored_claims"]) == 3

    async def test_scored_claim_has_text(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="verifiable claim", source_indices=[0])
        _mock_embed(mocker, n_leaves=1, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        assert result["scored_claims"][0].text == "verifiable claim"

    async def test_scored_claim_confidence_in_open_interval(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="a claim", source_indices=[0])
        _mock_embed(mocker, n_leaves=1, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        score = result["scored_claims"][0].confidence
        assert 0.0 < score < 1.0

    async def test_scored_claim_has_corroborating_field(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="a claim", source_indices=[0])
        _mock_embed(mocker, n_leaves=1, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        sc = result["scored_claims"][0]
        assert hasattr(sc, "corroborating")
        assert isinstance(sc.corroborating, int)

    async def test_scored_claim_n_leaves_matches_merkle_leaves(self, mocker, make_mara_state, make_merkle_leaf):
        leaves = [make_merkle_leaf(url=f"https://a.com/{i}", text=f"text {i}", index=i) for i in range(5)]
        claim = Claim(text="a claim", source_indices=[0])
        _mock_embed(mocker, n_leaves=5, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=leaves), config={}
        )
        assert result["scored_claims"][0].n_leaves == 5

    async def test_all_leaf_texts_passed_to_embed_first_call(self, mocker, make_mara_state, make_merkle_leaf):
        leaves = [
            make_merkle_leaf(url="https://a.com/0", text="text zero", index=0),
            make_merkle_leaf(url="https://a.com/1", text="text one", index=1),
        ]
        claim = Claim(text="my claim", source_indices=[0])
        mock_embed = mocker.patch(
            "mara.agent.nodes.confidence_scorer.embed",
            side_effect=[
                _make_leaf_embs(2),   # first call: leaf embeddings
                _make_claim_embs(1),  # second call: claim embeddings
            ],
        )
        await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=leaves), config={}
        )
        # First call is for leaf texts
        first_call_texts = mock_embed.call_args_list[0].args[0]
        assert "text zero" in first_call_texts
        assert "text one" in first_call_texts

    async def test_claim_text_passed_to_embed_second_call(self, mocker, make_mara_state, make_merkle_leaf):
        leaf = make_merkle_leaf(url="https://a.com", text="leaf text", index=0)
        claim = Claim(text="my specific claim", source_indices=[0])
        mock_embed = mocker.patch(
            "mara.agent.nodes.confidence_scorer.embed",
            side_effect=[
                _make_leaf_embs(1),
                _make_claim_embs(1),
            ],
        )
        await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        # Second call is for claim texts
        second_call_texts = mock_embed.call_args_list[1].args[0]
        assert "my specific claim" in second_call_texts

    async def test_claim_with_no_sources_still_scores(self, mocker, make_mara_state, make_merkle_leaf):
        """A claim with source_indices=[] scores against all leaves anyway."""
        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="orphan claim", source_indices=[])
        _mock_embed(mocker, n_leaves=1, n_claims=1)
        result = await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]), config={}
        )
        assert len(result["scored_claims"]) == 1
        assert 0.0 < result["scored_claims"][0].confidence < 1.0

    async def test_runcontext_leaf_embeddings_used_when_hashes_match(
        self, mocker, make_mara_state, make_merkle_leaf
    ):
        """When RunContext provides cached leaf embeddings with matching hashes, embed() is
        called only once (for claims), not twice."""
        from mara.agent.run_context import RunContext

        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="a claim", source_indices=[0])

        run_context = RunContext()
        run_context.leaf_embeddings = _make_leaf_embs(1)
        run_context.leaf_embedding_hashes = [leaf["hash"]]

        mock_embed = mocker.patch(
            "mara.agent.nodes.confidence_scorer.embed",
            return_value=_make_claim_embs(1),
        )
        await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]),
            config={"configurable": {"run_context": run_context}},
        )
        # embed() called only once — for claims (leaves served from RunContext)
        assert mock_embed.call_count == 1

    async def test_runcontext_ignored_when_hashes_mismatch(
        self, mocker, make_mara_state, make_merkle_leaf
    ):
        """When RunContext hashes don't match current leaves, embed() is called twice."""
        from mara.agent.run_context import RunContext

        leaf = make_merkle_leaf(url="https://a.com", text="text", index=0)
        claim = Claim(text="a claim", source_indices=[0])

        run_context = RunContext()
        run_context.leaf_embeddings = _make_leaf_embs(1)
        run_context.leaf_embedding_hashes = ["stale_hash_that_does_not_match"]

        mock_embed = mocker.patch(
            "mara.agent.nodes.confidence_scorer.embed",
            side_effect=[_make_leaf_embs(1), _make_claim_embs(1)],
        )
        await confidence_scorer(
            make_mara_state(extracted_claims=[claim], merkle_leaves=[leaf]),
            config={"configurable": {"run_context": run_context}},
        )
        assert mock_embed.call_count == 2
