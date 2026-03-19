"""LeafRepository Protocol — storage-backend-agnostic interface.

All methods accept and return plain Python types (dicts, lists, str, bytes).
No SQLite- or Postgres-specific types leak through the interface.  Callers
work exclusively with MerkleLeaf-shaped dicts and primitives.

A Postgres implementation needs only to provide the same method signatures;
no imports from the SQLite file are required (structural subtyping via Protocol).
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LeafRepository(Protocol):
    """Repository interface for leaf and run persistence."""

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
        """Insert a new run row with started_at = now().  Idempotent."""
        ...

    def complete_run(self, run_id: str, merkle_root: str) -> None:
        """Set completed_at = now() and store the final merkle_root."""
        ...

    # ------------------------------------------------------------------
    # Leaf persistence
    # ------------------------------------------------------------------

    def upsert_leaves(self, leaves: list[dict]) -> int:
        """INSERT OR IGNORE each leaf by hash.

        Args:
            leaves: List of MerkleLeaf-shaped dicts.  Required keys:
                ``hash``, ``url``, ``text``, ``retrieved_at``,
                ``contextualized_text``.

        Returns:
            Number of rows actually inserted (existing rows skipped).
        """
        ...

    def link_leaves_to_run(self, run_id: str, leaves: list[dict]) -> None:
        """Populate run_leaves join table for this run.

        Args:
            run_id: UUID4 matching an existing runs row.
            leaves: List of MerkleLeaf-shaped dicts.  Required keys:
                ``hash``, ``index``, ``sub_query``.
        """
        ...

    # ------------------------------------------------------------------
    # Freshness / cache
    # ------------------------------------------------------------------

    def get_fresh_leaves_for_url(
        self, url: str, max_age_hours: float
    ) -> list[dict]:
        """Return cached leaves for *url* if they are within *max_age_hours*.

        Returns an empty list when no fresh leaves exist, meaning the caller
        should proceed with a live scrape.

        Args:
            url: Exact URL to look up.
            max_age_hours: Maximum age of the most recent scrape in hours.

        Returns:
            List of MerkleLeaf-shaped dicts, or ``[]`` if stale / missing.
        """
        ...

    # ------------------------------------------------------------------
    # Retrieval (Phase 2 hooks — exposed now so the interface is stable)
    # ------------------------------------------------------------------

    def get_leaves_for_run(self, run_id: str) -> list[dict]:
        """Return all leaves associated with *run_id*, ordered by position_index."""
        ...

    def get_embeddings_for_hashes(
        self, hashes: list[str]
    ) -> dict[str, bytes | None]:
        """Return ``{hash: embedding_bytes}`` for each hash in *hashes*.

        Missing or not-yet-embedded hashes map to ``None``.
        """
        ...

    def update_embeddings(
        self, embeddings: dict[str, bytes], model_name: str
    ) -> None:
        """Store pre-computed float32 embeddings and record *model_name*.

        Args:
            embeddings: ``{leaf_hash: ndarray.tobytes()}``
            model_name: Name of the embedding model (stored on the leaf row).
        """
        ...

    def bm25_search(
        self,
        query_text: str,
        run_id: str | None = None,
        limit: int = 150,
    ) -> list[dict]:
        """Full-text BM25 search over ``contextualized_text``.

        Args:
            query_text: Free-text query string.
            run_id:     When provided, restrict results to leaves in this run.
            limit:      Maximum number of results to return.

        Returns:
            List of MerkleLeaf-shaped dicts, best-match first.
        """
        ...
