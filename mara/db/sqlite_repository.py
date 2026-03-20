"""SQLite implementation of LeafRepository.

Connection settings applied at open time:
  - WAL mode for concurrent readers
  - 5-second busy timeout (handles write contention gracefully)
  - Foreign key enforcement
  - NORMAL synchronous mode (safe + fast with WAL)

Embeddings are stored as raw ``BLOB`` (float32 ``ndarray.tobytes()``) and
loaded back in Python for cosine similarity.  This maps cleanly to
``vector(N)`` + pgvector HNSW index in a future Postgres backend.
"""

import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mara.logging import get_logger

_log = get_logger(__name__)

# FTS5 reserved words and special characters that must not appear verbatim in
# a MATCH expression when the input is raw natural-language text.
#
# FTS5 interprets:
#   -term          → NOT term (hyphens as unary NOT operator)
#   column:term    → column filter (colon as column specifier)
#   AND / OR / NOT → boolean operators (case-insensitive)
#   ( )            → grouping
#   *              → prefix match
#   ^              → initial token boost
#   "..."          → phrase query (safe when balanced, fragile otherwise)
#   ?              → glob single-char wildcard in some builds
#
# Stripping non-alphanumeric characters and the three reserved words produces
# a safe token list that the porter-ascii tokeniser handles correctly.
_FTS5_RESERVED = frozenset(("AND", "OR", "NOT"))


def _sanitize_fts5(query: str) -> str:
    """Return a safe FTS5 MATCH expression from a natural-language query.

    Replaces every character that is not alphanumeric or whitespace with a
    space, then removes tokens that are FTS5 boolean operators or shorter than
    4 characters (stopwords, articles, prepositions).  Joins surviving tokens
    with ``OR`` so any matching token scores a result — avoiding the implicit
    AND semantics that FTS5 applies by default and that returns zero results
    for long natural-language research questions.  Falls back to ``"*"``
    (match everything) if the result is empty.

    Examples
    --------
    >>> _sanitize_fts5("state-of-the-art benchmarks")
    'state OR benchmarks'
    >>> _sanitize_fts5("What are current nuclear fusion approaches?")
    'What OR current OR nuclear OR fusion OR approaches'
    """
    cleaned = re.sub(r"[^\w\s]", " ", query)
    tokens = [t for t in cleaned.split() if len(t) >= 4 and t.upper() not in _FTS5_RESERVED]
    return " OR ".join(tokens) if tokens else "*"

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_connection(db_path: str) -> sqlite3.Connection:
    """Open (or create) the database and apply PRAGMAs + schema."""
    resolved = str(Path(db_path).expanduser())
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    for pragma in _PRAGMAS:
        conn.execute(pragma)

    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.commit()

    return conn


class SQLiteLeafRepository:
    """Concrete SQLite-backed LeafRepository.

    Thread safety: ``check_same_thread=False`` is set so the same connection
    object can be passed between the main thread and LangGraph's async
    executor threads.  All writes are protected by SQLite's WAL-mode
    serialisation — no additional locking is needed at this scale.

    Args:
        db_path: Filesystem path to the SQLite database file.  Use
            ``":memory:"`` in tests for an isolated in-memory database.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = _open_connection(db_path)
        _log.debug("SQLiteLeafRepository opened: %s", db_path)

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        query: str,
        embedding_model: str,
        hash_algorithm: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO runs
                (run_id, query, embedding_model, hash_algorithm, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, query, embedding_model, hash_algorithm, _now_utc()),
        )
        self._conn.commit()

    def complete_run(self, run_id: str, merkle_root: str) -> None:
        self._conn.execute(
            "UPDATE runs SET merkle_root = ?, completed_at = ? WHERE run_id = ?",
            (merkle_root, _now_utc(), run_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Leaf persistence
    # ------------------------------------------------------------------

    def upsert_leaves(self, leaves: list[dict]) -> int:
        """INSERT OR IGNORE each leaf.  Returns the count of new rows."""
        if not leaves:
            return 0
        inserted = 0
        for leaf in leaves:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO leaves
                    (hash, url, text, retrieved_at, contextualized_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    leaf["hash"],
                    leaf["url"],
                    leaf["text"],
                    leaf["retrieved_at"],
                    leaf["contextualized_text"],
                ),
            )
            inserted += cursor.rowcount
        self._conn.commit()
        return inserted

    def link_leaves_to_run(self, run_id: str, leaves: list[dict]) -> None:
        if not leaves:
            return
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO run_leaves
                (run_id, leaf_hash, position_index, sub_query)
            VALUES (?, ?, ?, ?)
            """,
            [
                (run_id, leaf["hash"], leaf["index"], leaf["sub_query"])
                for leaf in leaves
            ],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Freshness / cache
    # ------------------------------------------------------------------

    def get_fresh_leaves_for_url(
        self, url: str, max_age_hours: float
    ) -> list[dict]:
        """Return cached leaves for *url* if scraped within *max_age_hours*.

        Freshness is based on the ``retrieved_at`` of the most-recent leaf
        row for this URL.  If that timestamp is within the window, ALL leaves
        for that URL sharing the same ``retrieved_at`` are returned.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Find the most recent retrieved_at for this URL
        row = self._conn.execute(
            "SELECT MAX(retrieved_at) AS latest FROM leaves WHERE url = ?",
            (url,),
        ).fetchone()

        if row is None or row["latest"] is None or row["latest"] < cutoff_str:
            return []

        latest = row["latest"]
        rows = self._conn.execute(
            """
            SELECT hash, url, text, retrieved_at, contextualized_text,
                   embedding, embedding_model
            FROM leaves
            WHERE url = ? AND retrieved_at = ?
            ORDER BY rowid
            """,
            (url, latest),
        ).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_leaves_for_run(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT l.hash, l.url, l.text, l.retrieved_at, l.contextualized_text,
                   l.embedding, l.embedding_model,
                   rl.position_index, rl.sub_query
            FROM leaves l
            JOIN run_leaves rl ON rl.leaf_hash = l.hash
            WHERE rl.run_id = ?
            ORDER BY rl.position_index
            """,
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_embeddings_for_hashes(
        self, hashes: list[str]
    ) -> dict[str, bytes | None]:
        if not hashes:
            return {}
        placeholders = ",".join("?" * len(hashes))
        rows = self._conn.execute(
            f"SELECT hash, embedding FROM leaves WHERE hash IN ({placeholders})",
            hashes,
        ).fetchall()
        result: dict[str, bytes | None] = {h: None for h in hashes}
        for row in rows:
            result[row["hash"]] = row["embedding"]  # may still be None
        return result

    def update_embeddings(
        self, embeddings: dict[str, bytes], model_name: str
    ) -> None:
        if not embeddings:
            return
        self._conn.executemany(
            "UPDATE leaves SET embedding = ?, embedding_model = ? WHERE hash = ?",
            [(blob, model_name, h) for h, blob in embeddings.items()],
        )
        self._conn.commit()

    def bm25_search(
        self,
        query_text: str,
        run_id: str | None = None,
        limit: int = 150,
    ) -> list[dict]:
        """FTS5 BM25 search over ``contextualized_text``.

        When *run_id* is provided the search is restricted to leaves linked
        to that run via the ``run_leaves`` join table.
        """
        fts_query = _sanitize_fts5(query_text)
        if run_id is not None:
            rows = self._conn.execute(
                """
                SELECT l.hash, l.url, l.text, l.retrieved_at,
                       l.contextualized_text, l.embedding, l.embedding_model
                FROM leaves_fts
                JOIN leaves l ON l.hash = leaves_fts.hash
                JOIN run_leaves rl ON rl.leaf_hash = l.hash
                WHERE leaves_fts MATCH ?
                  AND rl.run_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, run_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT l.hash, l.url, l.text, l.retrieved_at,
                       l.contextualized_text, l.embedding, l.embedding_model
                FROM leaves_fts
                JOIN leaves l ON l.hash = leaves_fts.hash
                WHERE leaves_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
