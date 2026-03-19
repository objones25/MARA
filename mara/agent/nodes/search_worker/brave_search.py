"""Brave Search node for the search worker subgraph.

Calls the Brave Web Search API (GET /v1/web/search) for the sub-query and
returns a deduplicated list of SearchResult objects drawn from all relevant
response sections: web results, news, discussions, and FAQ.

Why GET rather than POST?
  The Brave docs support both.  GET is sufficient here because sub-query
  strings stay well within the 400-character / 50-word URL param limit.  POST
  adds nothing for this use case.

Why collect multiple sections (web + news + discussions + faq)?
  Each section surfaces a different information type.  News results give
  recency; discussions give practitioner perspectives; FAQs give structured
  Q&A that can be highly relevant for factual research claims.  Collecting all
  of them maximises the variety of pages sent to Firecrawl for full-text
  scraping, which in turn maximises Merkle leaf diversity.

Why NOT divide count by max_workers?
  Each parallel worker processes a DIFFERENT sub-query.  Reducing the result
  count per worker would discard valid, non-overlapping search results.
  max_sources controls how many results are requested per sub-query; the Brave
  API hard-caps this at 20 per request.

Extra snippets:
  Sending extra_snippets=true asks Brave to return up to 5 additional page
  excerpts per result.  These are stored on SearchResult but NOT hashed — they
  are Brave-controlled and not reproducible from the raw source page.

Freshness:
  Controlled by ResearchConfig.brave_freshness (default: "" = no filter).
  Valid values: "pd" (24 h), "pw" (7 d), "pm" (31 d), "py" (1 y), or a custom
  "YYYY-MM-DDtoYYYY-MM-DD" range.
"""

import httpx
from langchain_core.runnables import RunnableConfig

from mara.agent.state import SearchResult, SearchWorkerState
from mara.logging import get_logger

_log = get_logger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


# ---------------------------------------------------------------------------
# Per-section extraction helpers
# ---------------------------------------------------------------------------


def _web_results(results: list[dict]) -> list[SearchResult]:
    return [
        SearchResult(
            url=r["url"],
            title=r.get("title", ""),
            description=r.get("description", ""),
            extra_snippets=r.get("extra_snippets") or [],
            page_age=r.get("page_age") or r.get("age") or "",
            result_type="web",
        )
        for r in results
        if r.get("url")
    ]


def _news_results(results: list[dict]) -> list[SearchResult]:
    return [
        SearchResult(
            url=r["url"],
            title=r.get("title", ""),
            description=r.get("description", ""),
            extra_snippets=r.get("extra_snippets") or [],
            page_age=r.get("age") or r.get("page_age") or "",
            result_type="news",
        )
        for r in results
        if r.get("url")
    ]


def _discussion_results(results: list[dict]) -> list[SearchResult]:
    """Discussions include a ``data.top_comment`` field with the leading reply."""
    out = []
    for r in results:
        if not r.get("url"):
            continue
        description = r.get("description", "")
        top_comment = (r.get("data") or {}).get("top_comment", "")
        if top_comment and not description:
            description = top_comment
        out.append(
            SearchResult(
                url=r["url"],
                title=r.get("title", ""),
                description=description,
                extra_snippets=r.get("extra_snippets") or [],
                page_age=r.get("page_age") or r.get("age") or "",
                result_type="discussion",
            )
        )
    return out


def _faq_results(results: list[dict]) -> list[SearchResult]:
    """FAQ items combine question + answer into the description field."""
    out = []
    for r in results:
        if not r.get("url"):
            continue
        question = r.get("question", "")
        answer = r.get("answer", "")
        description = f"Q: {question}  A: {answer}" if question or answer else ""
        out.append(
            SearchResult(
                url=r["url"],
                title=r.get("title", ""),
                description=description,
                extra_snippets=[],
                page_age="",
                result_type="faq",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def brave_search(state: SearchWorkerState, config: RunnableConfig) -> dict:
    """Fetch search results from the Brave Web Search API.

    Requests up to ``min(20, max_sources)`` results (20 is the Brave per-request
    cap).  Collects ALL results from web, news, discussions, and FAQ sections
    without deduplication — the same URL can legitimately appear in multiple
    sections with different metadata (e.g. a news article also showing up as a
    web result carries distinct descriptions, extra_snippets, and page_age in
    each entry).  Deduplication at the URL level happens in firecrawl_scrape,
    where it actually matters: to avoid scraping the same page twice.

    Returns:
        ``{"search_results": list[SearchResult]}``
    """
    research_config = state["research_config"]
    count = min(20, research_config.max_sources)
    query = state["sub_query"]["query"]

    _log.debug("Brave search: %r (count=%d)", query, count)

    params: dict[str, str | int] = {
        "q": state["sub_query"]["query"],
        "count": count,
        "extra_snippets": "true",
    }
    if research_config.brave_freshness:
        params["freshness"] = research_config.brave_freshness

    async with httpx.AsyncClient() as client:
        response = await client.get(
            _BRAVE_SEARCH_URL,
            params=params,
            headers={
                "X-Subscription-Token": research_config.brave_api_key,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

    results: list[SearchResult] = []
    results.extend(_web_results(data.get("web", {}).get("results", [])))
    results.extend(_news_results(data.get("news", {}).get("results", [])))
    results.extend(_discussion_results(data.get("discussions", {}).get("results", [])))
    results.extend(_faq_results(data.get("faq", {}).get("results", [])))

    _log.debug("Brave returned %d results for %r", len(results), query)
    return {"search_results": results}
