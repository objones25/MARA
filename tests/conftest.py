"""Shared pytest fixture factories for MARA tests.

Centralising state construction here means a MARAState or MerkleLeaf field
change only requires updating this file — individual test modules no longer
duplicate the full constructor.

Usage
-----
Each fixture returns a *callable* (factory pattern) so tests can parameterise
construction without writing their own builder functions:

    def test_something(make_mara_state, make_merkle_leaf):
        leaf  = make_merkle_leaf(index=2, text="specific content")
        state = make_mara_state(retrieved_leaves=[leaf])
"""

import pytest

from mara.agent.state import MARAState, MerkleLeaf
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# MerkleLeaf factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_merkle_leaf():
    """Return a factory that builds MerkleLeaf dicts with a real SHA-256 hash."""

    def _factory(
        index: int = 0,
        url: str = "https://example.com",
        text: str = "sample text",
        retrieved_at: str = "2026-03-19T10:00:00Z",
        sub_query: str = "test query",
        contextualized_text: str | None = None,
        algorithm: str = "sha256",
    ) -> MerkleLeaf:
        digest = hash_chunk(url=url, text=text, retrieved_at=retrieved_at, algorithm=algorithm)
        return MerkleLeaf(
            url=url,
            text=text,
            retrieved_at=retrieved_at,
            hash=digest,
            index=index,
            sub_query=sub_query,
            contextualized_text=contextualized_text if contextualized_text is not None else text,
        )

    return _factory


# ---------------------------------------------------------------------------
# MARAState factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_mara_state():
    """Return a factory that builds a fully-populated MARAState.

    All fields have sensible defaults.  Pass only what your test needs to
    override — the rest comes from the defaults here.

    The default ResearchConfig has ``leaf_db_enabled=False`` so tests never
    touch the filesystem database unless they explicitly inject their own config.
    """

    def _factory(
        query: str = "What are the effects of automation?",
        run_date: str = "2026-03-20",
        config: ResearchConfig | None = None,
        sub_queries=None,
        search_results=None,
        raw_chunks=None,
        merkle_leaves=None,
        merkle_tree=None,
        retrieved_leaves=None,
        extracted_claims=None,
        scored_claims=None,
        human_approved_claims=None,
        report_draft: str = "",
        certified_report=None,
        messages=None,
        loop_count: int = 0,
    ) -> MARAState:
        return MARAState(
            query=query,
            run_date=run_date,
            config=config or ResearchConfig(leaf_db_enabled=False),
            sub_queries=sub_queries or [],
            search_results=search_results or [],
            raw_chunks=raw_chunks or [],
            merkle_leaves=merkle_leaves or [],
            merkle_tree=merkle_tree,
            retrieved_leaves=retrieved_leaves or [],
            extracted_claims=extracted_claims or [],
            scored_claims=scored_claims or [],
            human_approved_claims=human_approved_claims or [],
            report_draft=report_draft,
            certified_report=certified_report,
            messages=messages or [],
            loop_count=loop_count,
        )

    return _factory
