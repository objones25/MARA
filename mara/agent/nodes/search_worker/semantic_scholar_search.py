"""Semantic Scholar snippet search node.

Calls the Semantic Scholar /snippet/search endpoint and returns SourceChunk
objects directly — no Firecrawl scraping required.

Why snippets instead of full PDFs?
  The /snippet/search endpoint returns ~500-word text excerpts drawn from a
  paper's title, abstract, and body text.  These are already chunked to a
  size suitable for embedding and retrieval, and the API returns only the
  snippets most relevant to the query.  Fetching full PDFs via Firecrawl
  would cost credits and take longer without proportional benefit for
  coverage — the snippets already represent the most salient content.

Why SourceChunks directly instead of SearchResults?
  The standard search_worker pattern (search → SearchResults → firecrawl_scrape
  → SourceChunks) exists because Brave and ArXiv return URLs that must be
  scraped for full text.  Semantic Scholar snippets ARE the text — there is
  nothing to scrape.  Returning SourceChunks directly removes the unnecessary
  indirection and avoids a Firecrawl round-trip entirely.

Canonical URL:
  Each SourceChunk uses https://www.semanticscholar.org/paper/CorpusId:{id}
  as its URL.  This is a stable, resolvable identifier — the same corpus ID
  always refers to the same paper.  Multiple snippets from the same paper
  share the same URL, which the per-URL chunk cap in the retriever manages
  appropriately.

API key:
  Without a key the endpoint is subject to shared rate limits.  Set the
  S2_API_KEY environment variable (read into ResearchConfig.semantic_scholar_api_key)
  to pass it in the x-api-key header for higher throughput.

Rate limits:
  S2 allows 1 RPS cumulative across all endpoints per API key.  MARA fires
  one S2 request per sub-query concurrently (up to max_workers at a time),
  so without throttling multiple requests would land simultaneously.
  _acquire_s2_slot() serialises all callers behind an asyncio.Lock and
  sleeps for the remainder of the 1-second window opened by the previous
  call before recording a new timestamp and releasing the lock.
"""

import asyncio
import time
from datetime import datetime, timezone

import httpx
from langchain_core.runnables import RunnableConfig

from mara.agent.state import SearchWorkerState, SourceChunk
from mara.logging import get_logger

_log = get_logger(__name__)

_S2_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
_S2_FIELDS = "snippet.text,snippet.section,snippet.snippetKind"

# Rate limiting — S2 enforces 1 RPS cumulative across all endpoints per key.
_S2_MIN_INTERVAL: float = 1.0  # seconds
_S2_RATE_LOCK = asyncio.Lock()
_s2_last_call_time: float = float("-inf")  # monotonic; -inf → no sleep on first call


async def _acquire_s2_slot() -> None:
    """Block until it is safe to fire the next S2 request (≤ 1 RPS).

    Serialises concurrent callers behind _S2_RATE_LOCK.  Each caller sleeps
    for the remainder of the 1-second window opened by the previous call,
    then records its own timestamp before releasing the lock.  The HTTP
    request is made *after* the lock is released so the network round-trip
    does not extend the inter-request gap beyond the minimum.
    """
    global _s2_last_call_time
    async with _S2_RATE_LOCK:
        now = time.monotonic()
        wait = _S2_MIN_INTERVAL - (now - _s2_last_call_time)
        if wait > 0:
            await asyncio.sleep(wait)
        _s2_last_call_time = time.monotonic()


async def semantic_scholar_search(state: SearchWorkerState, config: RunnableConfig) -> dict:
    """Fetch relevant text snippets from Semantic Scholar.

    Calls /snippet/search and converts each result directly to a SourceChunk.
    No Firecrawl scraping is required — the snippet text IS the leaf content.

    Args:
        state:  SearchWorkerState with ``sub_query`` and ``research_config``.
        config: RunnableConfig (unused; present for LangGraph compatibility).

    Returns:
        ``{"raw_chunks": list[SourceChunk]}``
    """
    research_config = state["research_config"]
    query = state["sub_query"]["query"]
    limit = research_config.semantic_scholar_max_results

    _log.debug("Semantic Scholar search: %r (limit=%d)", query, limit)

    params = {
        "query": query,
        "limit": limit,
        "fields": _S2_FIELDS,
    }
    headers: dict[str, str] = {"Accept": "application/json"}
    if research_config.semantic_scholar_api_key:
        headers["x-api-key"] = research_config.semantic_scholar_api_key

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    await _acquire_s2_slot()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                _S2_SNIPPET_URL,
                params=params,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            _log.error(
                "Semantic Scholar authentication failure (%d) for %r — check S2_API_KEY",
                status,
                query,
            )
        elif status == 429:
            _log.warning(
                "Semantic Scholar rate limit exceeded (%d) for %r — returning no chunks",
                status,
                query,
            )
        else:
            _log.warning(
                "Semantic Scholar HTTP %d for %r — returning no chunks", status, query
            )
        return {"raw_chunks": []}
    except httpx.TimeoutException as exc:
        _log.warning("Semantic Scholar request timed out for %r: %s — returning no chunks", query, exc)
        return {"raw_chunks": []}
    except httpx.HTTPError as exc:
        _log.warning("Semantic Scholar request failed for %r: %s — returning no chunks", query, exc)
        return {"raw_chunks": []}

    raw_chunks: list[SourceChunk] = []
    for match in data.get("data", []):
        snippet = match.get("snippet", {})
        paper = match.get("paper", {})

        text = snippet.get("text", "").strip()
        corpus_id = str(paper.get("corpusId", "")).strip()

        if not text or not corpus_id:
            continue

        raw_chunks.append(
            SourceChunk(
                url=f"https://www.semanticscholar.org/paper/CorpusId:{corpus_id}",
                text=text,
                retrieved_at=retrieved_at,
                sub_query=query,
            )
        )

    _log.debug("Semantic Scholar returned %d snippet(s) for %r", len(raw_chunks), query)
    return {"raw_chunks": raw_chunks}
