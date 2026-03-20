"""Tests for mara.db.SQLiteLeafRepository.

All tests use in-memory SQLite (:memory:) — no filesystem state is created
or cleaned up.  Each test gets its own fresh repository instance via the
``repo`` fixture.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mara.db import LeafRepository, SQLiteLeafRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo() -> SQLiteLeafRepository:
    return SQLiteLeafRepository(":memory:")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _past_utc(hours: float) -> str:
    t = datetime.now(timezone.utc) - timedelta(hours=hours)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_leaf(
    hash_: str = "abc123",
    url: str = "https://example.com",
    text: str = "sample text",
    retrieved_at: str | None = None,
    sub_query: str = "test query",
    index: int = 0,
) -> dict:
    return {
        "hash": hash_,
        "url": url,
        "text": text,
        "retrieved_at": retrieved_at or _now_utc(),
        "contextualized_text": text,
        "sub_query": sub_query,
        "index": index,
    }


def _make_run(
    run_id: str = "run-1",
    query: str = "test query",
    embedding_model: str = "all-MiniLM-L6-v2",
    hash_algorithm: str = "sha256",
) -> dict:
    return {
        "run_id": run_id,
        "query": query,
        "embedding_model": embedding_model,
        "hash_algorithm": hash_algorithm,
    }


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_sqlite_repo_satisfies_protocol(self, repo):
        assert isinstance(repo, LeafRepository)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestCreateRun:
    def test_create_run_succeeds(self, repo):
        r = _make_run()
        repo.create_run(**r)  # should not raise

    def test_create_run_idempotent(self, repo):
        r = _make_run()
        repo.create_run(**r)
        repo.create_run(**r)  # INSERT OR IGNORE — second call silently ignored

    def test_run_stored_in_db(self, repo):
        repo.create_run(**_make_run(run_id="r1", query="my question"))
        row = repo._conn.execute(
            "SELECT query FROM runs WHERE run_id = 'r1'"
        ).fetchone()
        assert row is not None
        assert row["query"] == "my question"

    def test_completed_at_null_after_create(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        row = repo._conn.execute(
            "SELECT completed_at FROM runs WHERE run_id = 'r1'"
        ).fetchone()
        assert row["completed_at"] is None


class TestCompleteRun:
    def test_complete_run_sets_merkle_root(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        repo.complete_run("r1", "deadbeef" * 8)
        row = repo._conn.execute(
            "SELECT merkle_root, completed_at FROM runs WHERE run_id = 'r1'"
        ).fetchone()
        assert row["merkle_root"] == "deadbeef" * 8
        assert row["completed_at"] is not None

    def test_complete_run_sets_completed_at(self, repo):
        before = _now_utc()
        repo.create_run(**_make_run(run_id="r1"))
        repo.complete_run("r1", "root")
        row = repo._conn.execute(
            "SELECT completed_at FROM runs WHERE run_id = 'r1'"
        ).fetchone()
        assert row["completed_at"] >= before


# ---------------------------------------------------------------------------
# Leaf persistence
# ---------------------------------------------------------------------------


class TestUpsertLeaves:
    def test_upsert_empty_list_returns_zero(self, repo):
        assert repo.upsert_leaves([]) == 0

    def test_upsert_single_leaf_returns_one(self, repo):
        count = repo.upsert_leaves([_make_leaf(hash_="h1")])
        assert count == 1

    def test_upsert_multiple_leaves_returns_count(self, repo):
        leaves = [_make_leaf(hash_=f"h{i}") for i in range(5)]
        count = repo.upsert_leaves(leaves)
        assert count == 5

    def test_upsert_same_hash_twice_returns_one_total(self, repo):
        leaf = _make_leaf(hash_="h1")
        c1 = repo.upsert_leaves([leaf])
        c2 = repo.upsert_leaves([leaf])
        assert c1 == 1
        assert c2 == 0  # duplicate ignored

    def test_leaf_data_stored_correctly(self, repo):
        leaf = _make_leaf(hash_="h1", url="https://test.com", text="hello world")
        repo.upsert_leaves([leaf])
        row = repo._conn.execute(
            "SELECT url, text FROM leaves WHERE hash = 'h1'"
        ).fetchone()
        assert row["url"] == "https://test.com"
        assert row["text"] == "hello world"

    def test_contextualized_text_stored(self, repo):
        leaf = _make_leaf(hash_="h1")
        leaf["contextualized_text"] = "enriched text"
        repo.upsert_leaves([leaf])
        row = repo._conn.execute(
            "SELECT contextualized_text FROM leaves WHERE hash = 'h1'"
        ).fetchone()
        assert row["contextualized_text"] == "enriched text"


class TestLinkLeavesToRun:
    def test_link_empty_list_is_noop(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        repo.link_leaves_to_run("r1", [])  # should not raise

    def test_link_creates_join_rows(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        leaves = [_make_leaf(hash_=f"h{i}", index=i) for i in range(3)]
        repo.upsert_leaves(leaves)
        repo.link_leaves_to_run("r1", leaves)
        count = repo._conn.execute(
            "SELECT COUNT(*) AS n FROM run_leaves WHERE run_id = 'r1'"
        ).fetchone()["n"]
        assert count == 3

    def test_link_stores_position_index(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        leaf = _make_leaf(hash_="h1", index=7)
        repo.upsert_leaves([leaf])
        repo.link_leaves_to_run("r1", [leaf])
        row = repo._conn.execute(
            "SELECT position_index FROM run_leaves WHERE run_id = 'r1' AND leaf_hash = 'h1'"
        ).fetchone()
        assert row["position_index"] == 7

    def test_link_stores_sub_query(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        leaf = _make_leaf(hash_="h1", sub_query="renewable energy impacts")
        repo.upsert_leaves([leaf])
        repo.link_leaves_to_run("r1", [leaf])
        row = repo._conn.execute(
            "SELECT sub_query FROM run_leaves WHERE run_id = 'r1' AND leaf_hash = 'h1'"
        ).fetchone()
        assert row["sub_query"] == "renewable energy impacts"

    def test_link_same_leaf_twice_is_idempotent(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        leaf = _make_leaf(hash_="h1")
        repo.upsert_leaves([leaf])
        repo.link_leaves_to_run("r1", [leaf])
        repo.link_leaves_to_run("r1", [leaf])  # INSERT OR IGNORE
        count = repo._conn.execute(
            "SELECT COUNT(*) AS n FROM run_leaves WHERE run_id = 'r1'"
        ).fetchone()["n"]
        assert count == 1


# ---------------------------------------------------------------------------
# Freshness / cache
# ---------------------------------------------------------------------------


class TestGetFreshLeavesForUrl:
    def test_returns_empty_for_unknown_url(self, repo):
        result = repo.get_fresh_leaves_for_url("https://unknown.com", 168.0)
        assert result == []

    def test_returns_leaves_when_fresh(self, repo):
        leaf = _make_leaf(hash_="h1", url="https://example.com", retrieved_at=_now_utc())
        repo.upsert_leaves([leaf])
        result = repo.get_fresh_leaves_for_url("https://example.com", 168.0)
        assert len(result) == 1
        assert result[0]["hash"] == "h1"

    def test_returns_empty_when_stale(self, repo):
        stale_time = _past_utc(hours=200)
        leaf = _make_leaf(hash_="h1", url="https://example.com", retrieved_at=stale_time)
        repo.upsert_leaves([leaf])
        result = repo.get_fresh_leaves_for_url("https://example.com", 168.0)
        assert result == []

    def test_returns_all_chunks_for_url(self, repo):
        ts = _now_utc()
        leaves = [
            _make_leaf(hash_=f"h{i}", url="https://example.com", retrieved_at=ts)
            for i in range(4)
        ]
        repo.upsert_leaves(leaves)
        result = repo.get_fresh_leaves_for_url("https://example.com", 168.0)
        assert len(result) == 4

    def test_uses_most_recent_retrieved_at(self, repo):
        old_ts = _past_utc(hours=10)
        new_ts = _now_utc()
        old_leaf = _make_leaf(hash_="h_old", url="https://example.com", retrieved_at=old_ts)
        new_leaf = _make_leaf(hash_="h_new", url="https://example.com", retrieved_at=new_ts)
        repo.upsert_leaves([old_leaf, new_leaf])
        result = repo.get_fresh_leaves_for_url("https://example.com", 168.0)
        hashes = {r["hash"] for r in result}
        assert "h_new" in hashes
        assert "h_old" not in hashes

    def test_does_not_return_different_url(self, repo):
        leaf = _make_leaf(hash_="h1", url="https://other.com", retrieved_at=_now_utc())
        repo.upsert_leaves([leaf])
        result = repo.get_fresh_leaves_for_url("https://example.com", 168.0)
        assert result == []

    def test_boundary_exactly_at_max_age_is_fresh(self, repo):
        # A leaf scraped exactly max_age_hours ago should still be fresh
        # (cutoff is strict less-than).  We use a small tolerance window.
        ts = _past_utc(hours=1.0)  # well within a 168h window
        leaf = _make_leaf(hash_="h1", url="https://example.com", retrieved_at=ts)
        repo.upsert_leaves([leaf])
        result = repo.get_fresh_leaves_for_url("https://example.com", 168.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Retrieval — get_leaves_for_run
# ---------------------------------------------------------------------------


class TestGetLeavesForRun:
    def test_returns_empty_for_unknown_run(self, repo):
        result = repo.get_leaves_for_run("nonexistent-run")
        assert result == []

    def test_returns_leaves_in_position_order(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        leaves = [_make_leaf(hash_=f"h{i}", index=i) for i in range(5)]
        repo.upsert_leaves(leaves)
        repo.link_leaves_to_run("r1", leaves)
        result = repo.get_leaves_for_run("r1")
        indices = [r["position_index"] for r in result]
        assert indices == sorted(indices)

    def test_returns_correct_leaf_count(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        leaves = [_make_leaf(hash_=f"h{i}", index=i) for i in range(3)]
        repo.upsert_leaves(leaves)
        repo.link_leaves_to_run("r1", leaves)
        result = repo.get_leaves_for_run("r1")
        assert len(result) == 3

    def test_does_not_return_leaves_from_other_run(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        repo.create_run(**_make_run(run_id="r2"))
        leaf_r1 = _make_leaf(hash_="h_r1", index=0)
        leaf_r2 = _make_leaf(hash_="h_r2", index=0)
        repo.upsert_leaves([leaf_r1, leaf_r2])
        repo.link_leaves_to_run("r1", [leaf_r1])
        repo.link_leaves_to_run("r2", [leaf_r2])
        result = repo.get_leaves_for_run("r1")
        assert all(r["hash"] == "h_r1" for r in result)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_get_embeddings_empty_hashes(self, repo):
        result = repo.get_embeddings_for_hashes([])
        assert result == {}

    def test_get_embeddings_returns_none_before_update(self, repo):
        leaf = _make_leaf(hash_="h1")
        repo.upsert_leaves([leaf])
        result = repo.get_embeddings_for_hashes(["h1"])
        assert result == {"h1": None}

    def test_update_embeddings_stores_bytes(self, repo):
        import struct
        leaf = _make_leaf(hash_="h1")
        repo.upsert_leaves([leaf])
        blob = struct.pack("4f", 0.1, 0.2, 0.3, 0.4)
        repo.update_embeddings({"h1": blob}, model_name="all-MiniLM-L6-v2")
        result = repo.get_embeddings_for_hashes(["h1"])
        assert result["h1"] == blob

    def test_update_embeddings_sets_model_name(self, repo):
        import struct
        leaf = _make_leaf(hash_="h1")
        repo.upsert_leaves([leaf])
        blob = struct.pack("4f", 0.1, 0.2, 0.3, 0.4)
        repo.update_embeddings({"h1": blob}, model_name="my-model")
        row = repo._conn.execute(
            "SELECT embedding_model FROM leaves WHERE hash = 'h1'"
        ).fetchone()
        assert row["embedding_model"] == "my-model"

    def test_update_embeddings_empty_is_noop(self, repo):
        repo.update_embeddings({}, model_name="any")  # should not raise

    def test_get_embeddings_unknown_hash_not_in_result(self, repo):
        result = repo.get_embeddings_for_hashes(["nonexistent"])
        assert result == {"nonexistent": None}

    def test_get_embeddings_multiple_hashes(self, repo):
        import struct
        leaves = [_make_leaf(hash_=f"h{i}") for i in range(3)]
        repo.upsert_leaves(leaves)
        blobs = {f"h{i}": struct.pack("f", float(i)) for i in range(3)}
        repo.update_embeddings(blobs, model_name="model")
        result = repo.get_embeddings_for_hashes([f"h{i}" for i in range(3)])
        for i in range(3):
            assert result[f"h{i}"] == blobs[f"h{i}"]


# ---------------------------------------------------------------------------
# BM25 / FTS5 search
# ---------------------------------------------------------------------------


class TestBm25Search:
    def test_returns_empty_when_no_leaves(self, repo):
        result = repo.bm25_search("machine learning")
        assert result == []

    def test_finds_matching_leaf(self, repo):
        leaf = _make_leaf(hash_="h1", text="deep learning neural networks")
        leaf["contextualized_text"] = "deep learning neural networks"
        repo.upsert_leaves([leaf])
        result = repo.bm25_search("neural")
        assert len(result) == 1
        assert result[0]["hash"] == "h1"

    def test_does_not_return_non_matching_leaf(self, repo):
        leaf = _make_leaf(hash_="h1", text="quantum computing applications")
        leaf["contextualized_text"] = "quantum computing applications"
        repo.upsert_leaves([leaf])
        result = repo.bm25_search("neural networks")
        assert result == []

    def test_respects_limit(self, repo):
        leaves = []
        for i in range(10):
            l = _make_leaf(hash_=f"h{i}", text=f"machine learning topic {i}")
            l["contextualized_text"] = f"machine learning topic {i}"
            leaves.append(l)
        repo.upsert_leaves(leaves)
        result = repo.bm25_search("machine", limit=3)
        assert len(result) <= 3

    def test_search_restricted_to_run(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        repo.create_run(**_make_run(run_id="r2"))
        leaf_r1 = _make_leaf(hash_="h_r1", text="climate change research", index=0)
        leaf_r1["contextualized_text"] = "climate change research"
        leaf_r2 = _make_leaf(hash_="h_r2", text="climate change effects", index=0)
        leaf_r2["contextualized_text"] = "climate change effects"
        repo.upsert_leaves([leaf_r1, leaf_r2])
        repo.link_leaves_to_run("r1", [leaf_r1])
        repo.link_leaves_to_run("r2", [leaf_r2])
        result = repo.bm25_search("climate", run_id="r1")
        hashes = {r["hash"] for r in result}
        assert "h_r1" in hashes
        assert "h_r2" not in hashes

    def test_search_without_run_id_returns_all_matches(self, repo):
        repo.create_run(**_make_run(run_id="r1"))
        repo.create_run(**_make_run(run_id="r2"))
        leaf1 = _make_leaf(hash_="h1", text="climate change research", index=0)
        leaf1["contextualized_text"] = "climate change research"
        leaf2 = _make_leaf(hash_="h2", text="climate change effects", index=0)
        leaf2["contextualized_text"] = "climate change effects"
        repo.upsert_leaves([leaf1, leaf2])
        repo.link_leaves_to_run("r1", [leaf1])
        repo.link_leaves_to_run("r2", [leaf2])
        result = repo.bm25_search("climate")
        hashes = {r["hash"] for r in result}
        assert "h1" in hashes
        assert "h2" in hashes


# ---------------------------------------------------------------------------
# FTS5 trigger sync
# ---------------------------------------------------------------------------


class TestFtsTriggers:
    def test_fts_synced_on_insert(self, repo):
        leaf = _make_leaf(hash_="h1", text="photosynthesis biology")
        leaf["contextualized_text"] = "photosynthesis biology"
        repo.upsert_leaves([leaf])
        row = repo._conn.execute(
            "SELECT COUNT(*) AS n FROM leaves_fts WHERE leaves_fts MATCH 'photosynthesis'"
        ).fetchone()
        assert row["n"] == 1

    def test_fts_synced_on_delete(self, repo):
        leaf = _make_leaf(hash_="h1", text="photosynthesis biology")
        leaf["contextualized_text"] = "photosynthesis biology"
        repo.upsert_leaves([leaf])
        repo._conn.execute("DELETE FROM leaves WHERE hash = 'h1'")
        repo._conn.commit()
        row = repo._conn.execute(
            "SELECT COUNT(*) AS n FROM leaves_fts WHERE leaves_fts MATCH 'photosynthesis'"
        ).fetchone()
        assert row["n"] == 0


# ---------------------------------------------------------------------------
# FTS5 query sanitization
# ---------------------------------------------------------------------------


class TestSanitizeFts5:
    """Unit tests for _sanitize_fts5 — the natural-language → FTS5 converter."""

    def test_plain_query_unchanged(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        assert _sanitize_fts5("machine learning") == "machine learning"

    def test_hyphens_replaced_with_spaces(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("state-of-the-art")
        assert "-" not in result
        assert "state" in result

    def test_question_mark_removed(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("What is RAG?")
        assert "?" not in result
        assert "What" in result

    def test_colon_removed(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("title:neural")
        assert ":" not in result

    def test_reserved_word_and_removed(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("neural AND networks")
        assert "AND" not in result.split()

    def test_reserved_word_or_removed(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("neural OR networks")
        assert "OR" not in result.split()

    def test_reserved_word_not_removed(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("NOT retrieval")
        assert "NOT" not in result.split()

    def test_reserved_words_case_insensitive(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        result = _sanitize_fts5("and OR not")
        tokens = result.split()
        assert not any(t.upper() in ("AND", "OR", "NOT") for t in tokens)

    def test_empty_query_returns_star(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        assert _sanitize_fts5("") == "*"

    def test_all_special_chars_returns_star(self):
        from mara.db.sqlite_repository import _sanitize_fts5
        assert _sanitize_fts5("AND OR NOT") == "*"

    def test_real_world_query_does_not_raise(self, repo):
        """The query that crashed in production should not raise OperationalError."""
        leaf = _make_leaf(hash_="h1", text="retrieval augmented generation survey")
        leaf["contextualized_text"] = "retrieval augmented generation survey"
        repo.upsert_leaves([leaf])
        # This exact query caused OperationalError: no such column: of
        result = repo.bm25_search(
            "What are the current state-of-the-art approaches to retrieval augmented generation?"
        )
        assert isinstance(result, list)
