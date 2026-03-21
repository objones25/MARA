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
  This node fires once per sub-query concurrently with search_worker and
  arxiv_worker.  Set an API key for production use to avoid rate-limit errors
  under concurrent load.
"""

from datetime import datetime, timezone

import httpx
from langchain_core.runnables import RunnableConfig

from mara.agent.state import SearchWorkerState, SourceChunk
from mara.logging import get_logger

_log = get_logger(__name__)

_S2_SNIPPET_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
_S2_FIELDS = "snippet.text,snippet.section,snippet.snippetKind"


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
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.HTTPError) as exc:
        _log.warning(
            "Semantic Scholar request failed for %r: %s — returning no chunks", query, exc
        )
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
