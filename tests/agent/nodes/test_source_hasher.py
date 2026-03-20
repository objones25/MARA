"""Tests for mara.agent.nodes.source_hasher.

source_hasher is a pure synchronous node — no I/O, no mocking needed.
Tests cover field population, correct hash values, index assignment,
empty input, and algorithm forwarding.
"""

import pytest

from mara.agent.nodes.source_hasher import source_hasher
from mara.agent.state import SourceChunk
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    url: str = "https://example.com",
    text: str = "sample text",
    retrieved_at: str = "2026-03-19T10:00:00Z",
    sub_query: str = "test query",
) -> SourceChunk:
    return SourceChunk(
        url=url,
        text=text,
        retrieved_at=retrieved_at,
        sub_query=sub_query,
    )


@pytest.fixture
def hasher_state(make_mara_state):
    from mara.config import ResearchConfig

    def _factory(chunks=None, algorithm="sha256"):
        return make_mara_state(
            raw_chunks=chunks or [],
            config=ResearchConfig(hash_algorithm=algorithm, leaf_db_enabled=False),
        )

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSourceHasherEmptyInput:
    def test_empty_chunks_returns_empty_leaves(self, hasher_state):
        result = source_hasher(hasher_state([]), config={})
        assert result == {"merkle_leaves": []}

    def test_returns_dict_with_merkle_leaves_key(self, hasher_state):
        result = source_hasher(hasher_state([]), config={})
        assert "merkle_leaves" in result


class TestSourceHasherFieldPopulation:
    def test_url_copied_from_chunk(self, hasher_state):
        chunk = _make_chunk(url="https://example.com/article")
        result = source_hasher(hasher_state([chunk]), config={})
        assert result["merkle_leaves"][0]["url"] == "https://example.com/article"

    def test_text_copied_from_chunk(self, hasher_state):
        chunk = _make_chunk(text="important research finding")
        result = source_hasher(hasher_state([chunk]), config={})
        assert result["merkle_leaves"][0]["text"] == "important research finding"

    def test_retrieved_at_copied_from_chunk(self, hasher_state):
        chunk = _make_chunk(retrieved_at="2026-01-15T08:30:00Z")
        result = source_hasher(hasher_state([chunk]), config={})
        assert result["merkle_leaves"][0]["retrieved_at"] == "2026-01-15T08:30:00Z"

    def test_sub_query_copied_from_chunk(self, hasher_state):
        chunk = _make_chunk(sub_query="renewable energy policy impacts")
        result = source_hasher(hasher_state([chunk]), config={})
        assert result["merkle_leaves"][0]["sub_query"] == "renewable energy policy impacts"

    def test_index_set_to_zero_for_first_chunk(self, hasher_state):
        result = source_hasher(hasher_state([_make_chunk()]), config={})
        assert result["merkle_leaves"][0]["index"] == 0

    def test_index_increments_per_chunk(self, hasher_state):
        chunks = [_make_chunk(url=f"https://example.com/{i}") for i in range(5)]
        result = source_hasher(hasher_state(chunks), config={})
        indices = [leaf["index"] for leaf in result["merkle_leaves"]]
        assert indices == [0, 1, 2, 3, 4]


class TestSourceHasherHashValues:
    def test_hash_matches_hash_chunk(self, hasher_state):
        chunk = _make_chunk(
            url="https://example.com",
            text="some text",
            retrieved_at="2026-03-19T10:00:00Z",
        )
        result = source_hasher(hasher_state([chunk]), config={})
        expected = hash_chunk(
            url="https://example.com",
            text="some text",
            retrieved_at="2026-03-19T10:00:00Z",
            algorithm="sha256",
        )
        assert result["merkle_leaves"][0]["hash"] == expected

    def test_hash_is_nonempty_string(self, hasher_state):
        result = source_hasher(hasher_state([_make_chunk()]), config={})
        assert isinstance(result["merkle_leaves"][0]["hash"], str)
        assert len(result["merkle_leaves"][0]["hash"]) > 0

    def test_different_texts_produce_different_hashes(self, hasher_state):
        chunk_a = _make_chunk(text="text A")
        chunk_b = _make_chunk(text="text B")
        result = source_hasher(hasher_state([chunk_a, chunk_b]), config={})
        assert result["merkle_leaves"][0]["hash"] != result["merkle_leaves"][1]["hash"]

    def test_same_input_produces_same_hash(self, hasher_state):
        chunk = _make_chunk()
        r1 = source_hasher(hasher_state([chunk]), config={})
        r2 = source_hasher(hasher_state([chunk]), config={})
        assert r1["merkle_leaves"][0]["hash"] == r2["merkle_leaves"][0]["hash"]

    def test_hash_algorithm_forwarded_to_hash_chunk(self, hasher_state):
        chunk = _make_chunk()
        result_sha256 = source_hasher(hasher_state([chunk], algorithm="sha256"), config={})
        result_sha512 = source_hasher(hasher_state([chunk], algorithm="sha512"), config={})
        # sha512 digests are longer than sha256
        assert len(result_sha512["merkle_leaves"][0]["hash"]) > len(
            result_sha256["merkle_leaves"][0]["hash"]
        )


class TestSourceHasherMultipleChunks:
    def test_count_matches_input_chunks(self, hasher_state):
        chunks = [_make_chunk(url=f"https://example.com/{i}") for i in range(7)]
        result = source_hasher(hasher_state(chunks), config={})
        assert len(result["merkle_leaves"]) == 7

    def test_order_preserved(self, hasher_state):
        chunks = [_make_chunk(url=f"https://example.com/{i}", text=f"text {i}") for i in range(3)]
        result = source_hasher(hasher_state(chunks), config={})
        for i, leaf in enumerate(result["merkle_leaves"]):
            assert leaf["url"] == f"https://example.com/{i}"
            assert leaf["text"] == f"text {i}"


class TestContextualizedText:
    def test_contextualized_text_equals_text(self, hasher_state):
        chunk = _make_chunk(text="sample text for contextualization")
        result = source_hasher(hasher_state([chunk]), config={})
        leaf = result["merkle_leaves"][0]
        assert leaf["contextualized_text"] == leaf["text"]

    def test_contextualized_text_present_on_all_leaves(self, hasher_state):
        chunks = [_make_chunk(url=f"https://example.com/{i}", text=f"text {i}") for i in range(4)]
        result = source_hasher(hasher_state(chunks), config={})
        for leaf in result["merkle_leaves"]:
            assert "contextualized_text" in leaf


class TestSourceHasherDbIntegration:
    """Verify that source_hasher calls leaf_repo when repo + run_id are injected."""

    def test_upsert_and_link_called_when_repo_injected(self, mocker, hasher_state):
        repo = mocker.MagicMock()
        repo.upsert_leaves.return_value = 1
        chunk = _make_chunk()
        config = {"configurable": {"leaf_repo": repo, "run_id": "run-1"}}
        source_hasher(hasher_state([chunk]), config=config)
        repo.upsert_leaves.assert_called_once()
        repo.link_leaves_to_run.assert_called_once()

    def test_link_called_with_run_id(self, mocker, hasher_state):
        repo = mocker.MagicMock()
        repo.upsert_leaves.return_value = 1
        chunk = _make_chunk()
        config = {"configurable": {"leaf_repo": repo, "run_id": "my-run-id"}}
        source_hasher(hasher_state([chunk]), config=config)
        run_id_passed = repo.link_leaves_to_run.call_args.args[0]
        assert run_id_passed == "my-run-id"

    def test_no_db_calls_when_repo_missing(self, mocker, hasher_state):
        repo = mocker.MagicMock()
        chunk = _make_chunk()
        # No leaf_repo in configurable
        config = {"configurable": {"run_id": "run-1"}}
        source_hasher(hasher_state([chunk]), config=config)
        repo.upsert_leaves.assert_not_called()

    def test_no_db_calls_when_run_id_missing(self, mocker, hasher_state):
        repo = mocker.MagicMock()
        chunk = _make_chunk()
        config = {"configurable": {"leaf_repo": repo}}
        source_hasher(hasher_state([chunk]), config=config)
        repo.upsert_leaves.assert_not_called()
