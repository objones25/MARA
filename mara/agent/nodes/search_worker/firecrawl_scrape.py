"""Firecrawl scrape node for the search worker subgraph.

For each URL returned by ``brave_search``, this node fetches the full page
markdown via Firecrawl's batch scrape API, then splits the content into
fixed-size character chunks.

These chunks — not Brave snippets — become the SourceChunks fed to the Source
Hasher.  Using the raw scraped text is a hard requirement of the Merkle
integrity protocol: we must be able to reproduce the exact bytes that were
hashed, which requires the text to come from our own scrape rather than a
search-provider-controlled snippet.

Chunking is deterministic: same source text + same config always produces the
same list of chunks.  This guarantees that re-running the same research session
on an unchanged page produces identical leaf hashes.
"""

import asyncio
from datetime import datetime, timezone

from firecrawl import Firecrawl
from langchain_core.runnables import RunnableConfig

from mara.agent.state import SourceChunk, SearchWorkerState
from mara.logging import get_logger

_log = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into fixed-size character chunks with overlap.

    The ``chunk_overlap_less_than_chunk_size`` validator on ResearchConfig
    guarantees ``overlap < chunk_size``, so ``step`` is always positive.
    Empty or whitespace-only chunks are dropped.
    """
    if not text:
        return []
    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


async def firecrawl_scrape(state: SearchWorkerState, config: RunnableConfig) -> dict:
    """Batch-scrape all URLs from ``search_results`` and chunk their content.

    Uses the synchronous Firecrawl SDK wrapped in ``asyncio.to_thread`` because
    the SDK's async batch API is not yet stable.

    Each page's markdown is split into fixed-size chunks.  The ``retrieved_at``
    timestamp is shared across all chunks from a single node invocation so that
    the Merkle leaf for chunk i records when the scrape job ran, not when each
    individual page was processed.

    Returns:
        ``{"raw_chunks": list[SourceChunk]}``
    """
    research_config = state["research_config"]
    search_results = state["search_results"]

    if not search_results:
        _log.debug("No search results — skipping scrape")
        return {"raw_chunks": []}

    # Deduplicate URLs before scraping — the same URL may appear in multiple
    # Brave response sections (web + news + discussions), and we must not scrape
    # it twice.  dict.fromkeys preserves insertion order (web results first).
    # NOTE: cross-worker deduplication (same URL retrieved by two different
    # sub-query workers) is a future concern; a shared in-memory cache with a
    # threading.Lock would handle it, but is deferred until benchmarked.
    urls = list(dict.fromkeys(r["url"] for r in search_results))
    _log.debug("Scraping %d unique URL(s) from %d search results", len(urls), len(search_results))
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Freshness cache: skip scraping URLs already in the DB.
    # This check runs BEFORE the Firecrawl API call to avoid wasting credits.
    # ------------------------------------------------------------------
    configurable = config.get("configurable", {}) if config else {}
    leaf_repo = configurable.get("leaf_repo")
    sub_query_text = state["sub_query"]["query"]

    raw_chunks: list[SourceChunk] = []
    urls_to_scrape: list[str] = []

    if leaf_repo is not None and research_config.leaf_db_enabled:
        max_age = research_config.leaf_cache_max_age_hours
        for url in urls:
            cached = leaf_repo.get_fresh_leaves_for_url(url, max_age)
            if cached:
                _log.debug("Cache hit for %s (%d chunk(s))", url, len(cached))
                for leaf in cached:
                    raw_chunks.append(
                        SourceChunk(
                            url=leaf["url"],
                            text=leaf["text"],
                            retrieved_at=leaf["retrieved_at"],
                            sub_query=sub_query_text,
                        )
                    )
            else:
                urls_to_scrape.append(url)
    else:
        urls_to_scrape = urls

    if not urls_to_scrape:
        _log.debug("All %d URL(s) served from cache", len(urls))
        return {"raw_chunks": raw_chunks}

    _log.debug("Scraping %d URL(s) via Firecrawl (%d served from cache)", len(urls_to_scrape), len(urls) - len(urls_to_scrape))

    fc = Firecrawl(api_key=research_config.firecrawl_api_key)
    job = await asyncio.to_thread(fc.batch_scrape, urls_to_scrape, formats=["markdown"])

    for doc in job.data or []:
        markdown: str = doc.markdown or ""
        url: str = doc.metadata.source_url if doc.metadata else ""
        if not url or not markdown:
            continue
        for chunk in _chunk_text(
            markdown, research_config.chunk_size, research_config.chunk_overlap
        ):
            raw_chunks.append(
                SourceChunk(
                    url=url,
                    text=chunk,
                    retrieved_at=retrieved_at,
                    sub_query=sub_query_text,
                )
            )

    _log.debug("Produced %d chunk(s) from:\n%s", len(raw_chunks), "\n".join(f"  {u}" for u in urls))
    return {"raw_chunks": raw_chunks}
