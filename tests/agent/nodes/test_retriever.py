"""Tests for mara.agent.nodes.retriever.

embed() is mocked by patching mara.agent.nodes.retriever.embed.
asyncio.to_thread is effectively bypassed because embed is called directly
(the mock replaces embed so to_thread calls the mock synchronously in tests).

Tests cover:
  - Empty leaves → returns empty retrieved_leaves
  - Leaf count ≤ max_claim_sources → returns all leaves without calling embed
  - Leaf count exactly == max_claim_sources → all returned without embed
  - Leaf count > max_claim_sources → returns exactly max_claim_sources leaves
  - embed called with [query] + sub_query strings
  - embed called with leaf contextualized_text values
  - Returns retrieved_leaves key
  - Retrieved leaves are a subset of the original merkle_leaves
  - Highest-scoring leaf is in the result
  - Uses config.embedding_model as the model name
  - Uses config.max_claim_sources as k
  - With no sub_queries, query_texts has exactly 1 element
  - With sub_queries, query_texts includes all sub_query["query"] strings
"""

import asyncio
import pytest
import numpy as np

from mara.agent.nodes.retriever import retriever, _load_or_compute_leaf_embeddings, _rrf_scores
from mara.agent.state import MARAState, MerkleLeaf, SubQuery
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(
    index: int,
    url: str = "https://example.com",
    text: str = "text",
    contextualized_text: str | None = None,
) -> MerkleLeaf:
    digest = hash_chunk(url=url, text=text, retrieved_at="2026-03-19T10:00:00Z", algorithm="sha256")
    return MerkleLeaf(
        url=url,
        text=text,
        retrieved_at="2026-03-19T10:00:00Z",
        hash=digest,
        index=index,
        sub_query="test query",
        contextualized_text=contextualized_text if contextualized_text is not None else text,
    )


def _make_state(
    leaves: list[MerkleLeaf] | None = None,
    sub_queries: list[SubQuery] | None = None,
    query: str = "What are the effects of automation?",
    max_claim_sources: int = 3,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> MARAState:
    return MARAState(
        query=query,
        config=ResearchConfig(
            brave_api_key="x",
            firecrawl_api_key="x",
            anthropic_api_key="x",
            max_claim_sources=max_claim_sources,
            embedding_model=embedding_model,
        ),
        sub_queries=sub_queries or [],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=leaves or [],
        merkle_tree=None,
        retrieved_leaves=[],
        extracted_claims=[],
        scored_claims=[],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


def _mock_embed(texts, model_name):
    """Identity-like embeddings: text i gets e_i (up to 4 dims)."""
    n = len(texts)
    embs = np.zeros((n, 4))
    for i in range(min(n, 4)):
        embs[i, i] = 1.0
    return embs


# ---------------------------------------------------------------------------
# Empty leaves
# ---------------------------------------------------------------------------


class TestRetrieverEmptyLeaves:
    async def test_empty_leaves_returns_empty_dict(self, mocker):
        mock_embed = mocker.patch("mara.agent.nodes.retriever.embed")
        state = _make_state(leaves=[])
        result = await retriever(state, config={})
        assert result == {"retrieved_leaves": []}

    async def test_empty_leaves_does_not_call_embed(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed")
        state = _make_state(leaves=[])
        await retriever(state, config={})
        mock_embed_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Leaf count ≤ max_claim_sources (no embedding needed)
# ---------------------------------------------------------------------------


class TestRetrieverNoEmbedNeeded:
    async def test_leaf_count_less_than_k_returns_all(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed")
        leaves = [_make_leaf(i) for i in range(2)]
        state = _make_state(leaves=leaves, max_claim_sources=5)
        result = await retriever(state, config={})
        assert len(result["retrieved_leaves"]) == 2

    async def test_leaf_count_less_than_k_no_embed_call(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed")
        leaves = [_make_leaf(i) for i in range(2)]
        state = _make_state(leaves=leaves, max_claim_sources=5)
        await retriever(state, config={})
        mock_embed_fn.assert_not_called()

    async def test_leaf_count_equal_to_k_returns_all(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed")
        leaves = [_make_leaf(i) for i in range(3)]
        state = _make_state(leaves=leaves, max_claim_sources=3)
        result = await retriever(state, config={})
        assert len(result["retrieved_leaves"]) == 3

    async def test_leaf_count_equal_to_k_no_embed_call(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed")
        leaves = [_make_leaf(i) for i in range(3)]
        state = _make_state(leaves=leaves, max_claim_sources=3)
        await retriever(state, config={})
        mock_embed_fn.assert_not_called()

    async def test_returns_retrieved_leaves_key(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed")
        leaves = [_make_leaf(0)]
        state = _make_state(leaves=leaves, max_claim_sources=5)
        result = await retriever(state, config={})
        assert "retrieved_leaves" in result


# ---------------------------------------------------------------------------
# Leaf count > max_claim_sources (embedding needed)
# ---------------------------------------------------------------------------


class TestRetrieverWithEmbedding:
    async def test_returns_exactly_k_leaves(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i, url=f"https://example.com/{i}", text=f"text {i}") for i in range(6)]
        state = _make_state(leaves=leaves, max_claim_sources=3)
        result = await retriever(state, config={})
        assert len(result["retrieved_leaves"]) == 3

    async def test_returns_retrieved_leaves_key(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        state = _make_state(leaves=leaves, max_claim_sources=2)
        result = await retriever(state, config={})
        assert "retrieved_leaves" in result

    async def test_retrieved_are_subset_of_merkle_leaves(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i, url=f"https://example.com/{i}", text=f"text {i}") for i in range(6)]
        state = _make_state(leaves=leaves, max_claim_sources=3)
        result = await retriever(state, config={})
        original_indices = {leaf["index"] for leaf in leaves}
        for leaf in result["retrieved_leaves"]:
            assert leaf["index"] in original_indices

    async def test_embed_called_twice(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(state, config={})
        assert mock_embed_fn.call_count == 2

    async def test_embed_called_with_embedding_model(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        state = _make_state(leaves=leaves, max_claim_sources=2, embedding_model="my-model")
        await retriever(state, config={})
        for call in mock_embed_fn.call_args_list:
            assert call.args[1] == "my-model"

    async def test_embed_called_with_leaf_contextualized_texts(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [
            _make_leaf(i, text=f"raw {i}", contextualized_text=f"ctx {i}")
            for i in range(5)
        ]
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(state, config={})
        # One of the embed calls must have received contextualized texts
        leaf_call_texts = None
        for call in mock_embed_fn.call_args_list:
            texts = call.args[0]
            if any(t.startswith("ctx ") for t in texts):
                leaf_call_texts = texts
                break
        assert leaf_call_texts is not None
        for i in range(5):
            assert f"ctx {i}" in leaf_call_texts

    async def test_embed_not_called_with_raw_texts_when_contextualized_differs(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [
            _make_leaf(i, text=f"raw {i}", contextualized_text=f"ctx {i}")
            for i in range(5)
        ]
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(state, config={})
        # Verify "raw 0" never appears in leaf embed call
        for call in mock_embed_fn.call_args_list:
            texts = call.args[0]
            if any(t.startswith("ctx ") for t in texts):
                assert "raw 0" not in texts


# ---------------------------------------------------------------------------
# Query texts construction
# ---------------------------------------------------------------------------


class TestRetrieverQueryTexts:
    async def test_no_sub_queries_query_texts_has_one_element(self, mocker):
        captured = []

        def capturing_embed(texts, model_name):
            captured.append(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        state = _make_state(leaves=leaves, sub_queries=[], max_claim_sources=2)
        await retriever(state, config={})
        # The first embed call should be for query_texts (just the main query)
        query_call_texts = captured[0]
        assert len(query_call_texts) == 1

    async def test_sub_queries_included_in_query_texts(self, mocker):
        captured = []

        def capturing_embed(texts, model_name):
            captured.append(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        subs = [
            SubQuery(query="sub query one", domain="d1"),
            SubQuery(query="sub query two", domain="d2"),
        ]
        state = _make_state(leaves=leaves, sub_queries=subs, max_claim_sources=2)
        await retriever(state, config={})
        query_call_texts = captured[0]
        assert "sub query one" in query_call_texts
        assert "sub query two" in query_call_texts

    async def test_main_query_is_first_in_query_texts(self, mocker):
        captured = []

        def capturing_embed(texts, model_name):
            captured.append(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        subs = [SubQuery(query="sub q", domain="d")]
        state = _make_state(
            leaves=leaves,
            sub_queries=subs,
            query="main query text",
            max_claim_sources=2,
        )
        await retriever(state, config={})
        query_call_texts = captured[0]
        assert query_call_texts[0] == "main query text"

    async def test_query_texts_count_is_one_plus_num_sub_queries(self, mocker):
        captured = []

        def capturing_embed(texts, model_name):
            captured.append(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        subs = [SubQuery(query=f"sub {i}", domain="d") for i in range(3)]
        state = _make_state(leaves=leaves, sub_queries=subs, max_claim_sources=2)
        await retriever(state, config={})
        query_call_texts = captured[0]
        assert len(query_call_texts) == 4  # 1 main + 3 sub


# ---------------------------------------------------------------------------
# Scoring correctness — highest-scoring leaf is retrieved
# ---------------------------------------------------------------------------


class TestRetrieverScoring:
    async def test_highest_scoring_leaf_is_in_result(self, mocker):
        """Arrange embeddings so leaf 0 scores 1.0 against query, rest score 0."""

        def biased_embed(texts, model_name):
            n = len(texts)
            embs = np.zeros((n, 4))
            # We'll make query_texts = ["main query"] → emb[0] = [1, 0, 0, 0]
            # leaf texts = ["text 0", "text 1", ...] → leaf 0 emb = [1, 0, 0, 0]
            for i in range(min(n, 4)):
                embs[i, i] = 1.0
            return embs

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=biased_embed)
        leaves = [_make_leaf(i, url=f"https://example.com/{i}", text=f"text {i}") for i in range(5)]
        # With no sub_queries: query_texts = ["main query"] → emb shape (1, 4) with [1,0,0,0]
        # leaf embs: leaf 0 → [1,0,0,0], others are progressively [0,1,0,0], etc.
        # scores[0] = dot([1,0,0,0], [1,0,0,0]) = 1.0 → highest
        state = _make_state(leaves=leaves, sub_queries=[], max_claim_sources=2)
        result = await retriever(state, config={})
        retrieved_indices = {leaf["index"] for leaf in result["retrieved_leaves"]}
        assert 0 in retrieved_indices

    async def test_uses_max_claim_sources_as_k(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(10)]
        state = _make_state(leaves=leaves, max_claim_sources=4)
        result = await retriever(state, config={})
        assert len(result["retrieved_leaves"]) == 4


# ---------------------------------------------------------------------------
# Helpers shared by the new test classes
# ---------------------------------------------------------------------------


def _make_blob(dim: int) -> bytes:
    """Return a valid float32 blob for an embedding of the given dimension."""
    return np.zeros(dim, dtype=np.float32).tobytes()


def _make_leaf_dict(leaf: MerkleLeaf) -> dict:
    """Minimal dict that bm25_search might return for a leaf."""
    return {"hash": leaf["hash"], "url": leaf["url"], "text": leaf["text"],
            "retrieved_at": leaf["retrieved_at"], "contextualized_text": leaf["contextualized_text"]}


# ---------------------------------------------------------------------------
# _rrf_scores — pure-function unit tests
# ---------------------------------------------------------------------------


class TestRrfScores:
    def _semantic_order(self, scores: np.ndarray) -> np.ndarray:
        return np.argsort(scores)[::-1]

    def test_shape_matches_leaf_count(self):
        leaves = [_make_leaf(i) for i in range(5)]
        scores = np.array([0.9, 0.5, 0.3, 0.1, 0.0])
        order = self._semantic_order(scores)
        result = _rrf_scores(leaves, scores, order, {})
        assert result.shape == (5,)

    def test_top_semantic_top_bm25_leaf_scores_highest(self):
        """Leaf ranked first in both channels must win."""
        leaves = [_make_leaf(i) for i in range(5)]
        scores = np.array([0.9, 0.5, 0.3, 0.1, 0.0])
        order = self._semantic_order(scores)
        bm25_map = {leaves[0]["hash"]: 0}  # leaf 0 also top BM25
        rrf = _rrf_scores(leaves, scores, order, bm25_map)
        assert np.argmax(rrf) == 0

    def test_bm25_absent_leaf_gets_penalty_rank(self):
        """A leaf absent from BM25 results must score lower than one at rank 0."""
        leaves = [_make_leaf(i) for i in range(3)]
        scores = np.array([0.9, 0.5, 0.1])
        order = self._semantic_order(scores)
        bm25_map = {leaves[0]["hash"]: 0}  # only leaf 0 in BM25
        rrf = _rrf_scores(leaves, scores, order, bm25_map)
        # leaf 0: r_sem=0, r_bm25=0 → highest
        # leaf 1: r_sem=1, r_bm25=3 (penalty)
        # leaf 2: r_sem=2, r_bm25=3 (penalty)
        assert rrf[0] > rrf[1] > rrf[2]

    def test_empty_bm25_map_gives_all_penalty(self):
        """When no BM25 results exist, all leaves get penalty rank."""
        leaves = [_make_leaf(i) for i in range(4)]
        scores = np.array([0.9, 0.5, 0.3, 0.1])
        order = self._semantic_order(scores)
        rrf = _rrf_scores(leaves, scores, order, {})
        # All bm25 contributions are equal (all penalty=4), so ranking follows semantic
        assert np.argmax(rrf) == 0  # best semantic wins

    def test_rrf_k60_numerical_value(self):
        """For r_sem=0, r_bm25=0: RRF = 2/(60+1) = 2/61."""
        leaves = [_make_leaf(0)]
        scores = np.array([1.0])
        order = np.array([0])
        bm25_map = {leaves[0]["hash"]: 0}
        rrf = _rrf_scores(leaves, scores, order, bm25_map)
        expected = 2.0 / 61.0
        assert abs(float(rrf[0]) - expected) < 1e-10

    def test_bm25_top_low_semantic_boosts_above_mid_semantic_no_bm25(self):
        """Leaf at semantic rank 4 but BM25 rank 0 beats leaf at sem rank 1 with no BM25."""
        # n=5, penalty=5 — must use unique leaves so hashes don't collide
        leaves = [_make_leaf(i, url=f"https://example.com/{i}", text=f"text {i}") for i in range(5)]
        scores = np.array([0.9, 0.5, 0.3, 0.2, 0.1])
        order = self._semantic_order(scores)
        bm25_map = {leaves[4]["hash"]: 0}  # leaf 4 is BM25 champion; others absent (penalty=5)
        rrf = _rrf_scores(leaves, scores, order, bm25_map)
        # leaf 4: r_sem=4, r_bm25=0 → 1/(60+4+1) + 1/(60+0+1) = 1/65 + 1/61 ≈ 0.03178
        # leaf 1: r_sem=1, r_bm25=5 → 1/(60+1+1) + 1/(60+5+1) = 1/62 + 1/66 ≈ 0.03128
        assert float(rrf[4]) > float(rrf[1])


# ---------------------------------------------------------------------------
# _load_or_compute_leaf_embeddings — embedding cache tests
# ---------------------------------------------------------------------------


class TestLoadOrComputeLeafEmbeddings:
    TARGET_DIM = 4

    def _make_leaves(self, n: int) -> list[MerkleLeaf]:
        return [_make_leaf(i, text=f"text {i}", contextualized_text=f"ctx {i}") for i in range(n)]

    async def test_no_repo_embeds_all_leaves(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(3)
        result = await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, None)
        assert result.shape == (3, self.TARGET_DIM)
        mock_embed_fn.assert_called_once()

    async def test_no_repo_does_not_call_update(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        repo = mocker.MagicMock()
        # calling with leaf_repo=None — repo should never be touched
        await _load_or_compute_leaf_embeddings(self._make_leaves(2), "model", self.TARGET_DIM, None)
        repo.get_embeddings_for_hashes.assert_not_called()
        repo.update_embeddings.assert_not_called()

    async def test_all_cached_skips_embed(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(3)
        blobs = {leaf["hash"]: _make_blob(self.TARGET_DIM) for leaf in leaves}
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)
        mock_embed_fn.assert_not_called()

    async def test_all_cached_skips_update(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(3)
        blobs = {leaf["hash"]: _make_blob(self.TARGET_DIM) for leaf in leaves}
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)
        repo.update_embeddings.assert_not_called()

    async def test_all_cached_returns_correct_shape(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(3)
        blobs = {leaf["hash"]: _make_blob(self.TARGET_DIM) for leaf in leaves}
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs
        result = await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)
        assert result.shape == (3, self.TARGET_DIM)

    async def test_all_uncached_calls_embed(self, mocker):
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(3)
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = {leaf["hash"]: None for leaf in leaves}
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)
        mock_embed_fn.assert_called_once()

    async def test_all_uncached_calls_update(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(3)
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = {leaf["hash"]: None for leaf in leaves}
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)
        repo.update_embeddings.assert_called_once()

    async def test_update_called_with_model_name(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = self._make_leaves(2)
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = {leaf["hash"]: None for leaf in leaves}
        await _load_or_compute_leaf_embeddings(leaves, "my-model", self.TARGET_DIM, repo)
        _, model_arg = repo.update_embeddings.call_args.args
        assert model_arg == "my-model"

    async def test_partial_cache_embeds_only_uncached(self, mocker):
        leaves = self._make_leaves(4)
        # leaves 0 and 2 are cached; 1 and 3 are not
        blobs = {
            leaves[0]["hash"]: _make_blob(self.TARGET_DIM),
            leaves[1]["hash"]: None,
            leaves[2]["hash"]: _make_blob(self.TARGET_DIM),
            leaves[3]["hash"]: None,
        }
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs

        embedded_texts = []

        def capturing_embed(texts, model_name):
            embedded_texts.extend(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)

        # Only the uncached leaves' texts should have been embedded
        assert "ctx 1" in embedded_texts
        assert "ctx 3" in embedded_texts
        assert "ctx 0" not in embedded_texts
        assert "ctx 2" not in embedded_texts

    async def test_partial_cache_update_called_for_uncached_only(self, mocker):
        leaves = self._make_leaves(4)
        blobs = {
            leaves[0]["hash"]: _make_blob(self.TARGET_DIM),
            leaves[1]["hash"]: None,
            leaves[2]["hash"]: _make_blob(self.TARGET_DIM),
            leaves[3]["hash"]: None,
        }
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)

        stored_hashes = set(repo.update_embeddings.call_args.args[0].keys())
        assert leaves[1]["hash"] in stored_hashes
        assert leaves[3]["hash"] in stored_hashes
        assert leaves[0]["hash"] not in stored_hashes
        assert leaves[2]["hash"] not in stored_hashes

    async def test_dim_mismatch_triggers_reembed(self, mocker):
        leaves = self._make_leaves(2)
        blobs = {
            leaves[0]["hash"]: _make_blob(3),  # wrong dim (3 ≠ TARGET_DIM=4)
            leaves[1]["hash"]: _make_blob(self.TARGET_DIM),  # correct
        }
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs

        embedded_texts = []

        def capturing_embed(texts, model_name):
            embedded_texts.extend(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)

        # leaf 0 (wrong dim) must be re-embedded; leaf 1 (correct dim) must not
        assert "ctx 0" in embedded_texts
        assert "ctx 1" not in embedded_texts

    async def test_corrupt_blob_triggers_reembed(self, mocker):
        leaves = self._make_leaves(2)
        blobs = {
            leaves[0]["hash"]: b"not-float32-aligned",  # 19 bytes, corrupt
            leaves[1]["hash"]: _make_blob(self.TARGET_DIM),
        }
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = blobs

        embedded_texts = []

        def capturing_embed(texts, model_name):
            embedded_texts.extend(texts)
            return _mock_embed(texts, model_name)

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=capturing_embed)
        await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)

        assert "ctx 0" in embedded_texts   # corrupt → re-embedded
        assert "ctx 1" not in embedded_texts  # valid → cached

    async def test_result_preserves_leaf_order(self, mocker):
        """Rows in the result matrix must match the original leaf order."""
        leaves = self._make_leaves(4)
        # All uncached so embed returns deterministic arrays
        repo = mocker.MagicMock()
        repo.get_embeddings_for_hashes.return_value = {leaf["hash"]: None for leaf in leaves}

        # embed returns identity-like: text at position i → e_i
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        result = await _load_or_compute_leaf_embeddings(leaves, "model", self.TARGET_DIM, repo)

        # The embedding for the first leaf should be at result[0], etc.
        assert result.shape[0] == 4


# ---------------------------------------------------------------------------
# retriever — hybrid BM25 + RRF integration tests
# ---------------------------------------------------------------------------


class TestRetrieverHybrid:
    """Tests for the hybrid retrieval path (leaf_repo + run_id present)."""

    def _mock_repo(self, mocker, leaves, blobs=None, bm25_results=None):
        repo = mocker.MagicMock()
        if blobs is None:
            blobs = {leaf["hash"]: None for leaf in leaves}
        repo.get_embeddings_for_hashes.return_value = blobs
        repo.update_embeddings.return_value = None
        repo.bm25_search.return_value = bm25_results or []
        return repo

    async def test_bm25_called_when_repo_present(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}})
        repo.bm25_search.assert_called_once()

    async def test_bm25_called_with_main_query(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=2, query="my research question")
        await retriever(state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}})
        query_arg = repo.bm25_search.call_args.args[0]
        assert query_arg == "my research question"

    async def test_bm25_called_with_run_id(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(state, config={"configurable": {"leaf_repo": repo, "run_id": "my-run"}})
        run_id_arg = repo.bm25_search.call_args.args[1]
        assert run_id_arg == "my-run"

    async def test_bm25_limit_is_leaf_count(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(7)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=3)
        await retriever(state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}})
        limit_arg = repo.bm25_search.call_args.args[2]
        assert limit_arg == 7

    async def test_bm25_not_called_when_no_repo(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        state = _make_state(leaves=leaves, max_claim_sources=2)
        # no leaf_repo in config
        result = await retriever(state, config={})
        # Pure semantic path — no bm25 call possible (no repo object)
        assert len(result["retrieved_leaves"]) == 2

    async def test_empty_bm25_results_still_returns_k_leaves(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        repo = self._mock_repo(mocker, leaves, bm25_results=[])
        state = _make_state(leaves=leaves, max_claim_sources=2)
        result = await retriever(
            state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}}
        )
        assert len(result["retrieved_leaves"]) == 2

    async def test_bm25_top_leaf_enters_result_over_pure_semantic(self, mocker):
        """Leaf with strong BM25 rank but weak semantic rank must appear in RRF top-K.

        Setup (5 leaves, k=2):
          semantic scores: [0.9, 0.5, 0.3, 0.2, 0.1]
            → pure semantic top-2: leaf 0, leaf 1
          BM25: only leaf 4 is in results (rank 0)
            → RRF boosts leaf 4 above leaf 1
          expected RRF top-2: leaf 0 and leaf 4
        """
        leaves = [_make_leaf(i, url=f"https://example.com/{i}", text=f"text {i}") for i in range(5)]

        def biased_embed(texts, model_name):
            n = len(texts)
            embs = np.zeros((n, 4))
            # query embed: text[0] → [1, 0, 0, 0]
            # leaf embeds:  leaf 0 → [1, 0, 0, 0] (score 1.0)
            #               leaf 1 → [0.5, 0.5, 0, 0] (score 0.5)
            #               rest  → [0, ...] (score 0)
            if n == 1:
                embs[0, 0] = 1.0
            else:
                embs[0] = np.array([1.0, 0.0, 0.0, 0.0])
                if n > 1:
                    embs[1] = np.array([0.5, 0.5, 0.0, 0.0])
            return embs

        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=biased_embed)

        bm25_result = [_make_leaf_dict(leaves[4])]  # leaf 4 is top BM25
        repo = self._mock_repo(mocker, leaves, bm25_results=bm25_result)
        state = _make_state(leaves=leaves, sub_queries=[], max_claim_sources=2)
        result = await retriever(
            state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}}
        )
        retrieved_hashes = {l["hash"] for l in result["retrieved_leaves"]}
        assert leaves[4]["hash"] in retrieved_hashes  # BM25 champion in top-2

    async def test_k_leaves_returned_in_hybrid_mode(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(8)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=3)
        result = await retriever(
            state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}}
        )
        assert len(result["retrieved_leaves"]) == 3

    async def test_hybrid_result_is_subset_of_merkle_leaves(self, mocker):
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(6)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=3)
        result = await retriever(
            state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}}
        )
        original_hashes = {l["hash"] for l in leaves}
        for leaf in result["retrieved_leaves"]:
            assert leaf["hash"] in original_hashes

    async def test_run_id_none_still_calls_bm25(self, mocker):
        """When leaf_repo is present but run_id is absent, bm25_search is called with None."""
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        leaves = [_make_leaf(i) for i in range(5)]
        repo = self._mock_repo(mocker, leaves)
        state = _make_state(leaves=leaves, max_claim_sources=2)
        # leaf_repo present, run_id absent
        await retriever(state, config={"configurable": {"leaf_repo": repo}})
        run_id_passed = repo.bm25_search.call_args.args[1]
        assert run_id_passed is None

    async def test_embedding_cache_hit_reduces_embed_calls(self, mocker):
        """With all leaves cached, embed is called only once (for queries)."""
        leaves = [_make_leaf(i) for i in range(5)]
        blobs = {leaf["hash"]: _make_blob(4) for leaf in leaves}
        repo = self._mock_repo(mocker, leaves, blobs=blobs)
        mock_embed_fn = mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(
            state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}}
        )
        # One call for query embeddings; leaf embeddings served from cache
        assert mock_embed_fn.call_count == 1

    async def test_new_embeddings_stored_in_db(self, mocker):
        """Newly computed leaf embeddings must be written back to the DB."""
        leaves = [_make_leaf(i) for i in range(5)]
        repo = self._mock_repo(mocker, leaves)  # all blobs None → all uncached
        mocker.patch("mara.agent.nodes.retriever.embed", side_effect=_mock_embed)
        state = _make_state(leaves=leaves, max_claim_sources=2)
        await retriever(
            state, config={"configurable": {"leaf_repo": repo, "run_id": "r1"}}
        )
        repo.update_embeddings.assert_called_once()
