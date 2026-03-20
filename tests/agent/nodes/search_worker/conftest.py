"""Shared fixture factory for SearchWorkerState tests."""

import pytest

from mara.agent.state import SearchWorkerState, SubQuery
from mara.config import ResearchConfig


@pytest.fixture
def make_search_worker_state():
    """Return a factory that builds a SearchWorkerState.

    The default ResearchConfig has ``leaf_db_enabled=False`` so tests never
    touch the filesystem database unless they explicitly inject their own config.
    """

    def _factory(
        query: str = "test query",
        domain: str = "general",
        config: ResearchConfig | None = None,
        search_results=None,
        raw_chunks=None,
    ) -> SearchWorkerState:
        return SearchWorkerState(
            sub_query=SubQuery(query=query, domain=domain),
            research_config=config or ResearchConfig(),
            search_results=search_results or [],
            raw_chunks=raw_chunks or [],
        )

    return _factory
