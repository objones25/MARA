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

from mara.agent.nodes.retriever import retriever
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
