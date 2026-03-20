"""ArXiv search node for the ArXiv worker subgraph.

Calls the ArXiv API (export.arxiv.org/api/query) and returns SearchResult
objects whose ``url`` is the paper's versioned PDF URL.  The PDF URL is
then scraped by ``firecrawl_scrape`` to produce full-text SourceChunks,
exactly mirroring the brave_search → firecrawl_scrape pattern for web
sources.

Why PDF URLs and not abstract URLs?
  The abstract page contains only the title, authors, and abstract — a
  summary of the paper.  Research claims require evidence from the full
  text: methods, results, and discussion sections.  The PDF URL gives
  Firecrawl the complete paper converted to markdown.

Why versioned PDF URLs?
  ArXiv assigns a version number to every submission.  The versioned URL
  (e.g. https://arxiv.org/pdf/2301.07041v2) is immutable — it always
  returns exactly the same PDF.  This satisfies the Merkle integrity
  requirement: hash(url, text, retrieved_at) can be independently verified
  by any reader who fetches the same versioned URL.

Why is the abstract stored in SearchResult.description but not hashed?
  Same reasoning as Brave's description/extra_snippets: it is a
  provider-supplied summary, not the authoritative source bytes.  It is
  useful for HITL context and pre-screening but cannot serve as a source
  of record for claim integrity.  The hashed content comes from Firecrawl's
  scrape of the full PDF.

Rate limits:
  The ArXiv API recommends a 3-second delay between consecutive requests
  from the same client.  This node fires once per sub-query concurrently
  alongside the web search workers — not as a sequential bulk downloader —
  so the guideline does not apply here.  Do not call this node in a tight
  loop.

Search query construction:
  Uses the ``all:`` prefix to search across title, abstract, and full
  text.  Results are sorted by relevance (Apache Lucene relevance, the
  ArXiv API default).  Field-specific prefix queries (ti:, au:, abs:) and
  boolean operators (AND, ANDNOT) can be passed through by the query
  planner if the sub-query is crafted for ArXiv specifically.
"""

import xml.etree.ElementTree as ET

import httpx
from langchain_core.runnables import RunnableConfig

from mara.agent.state import SearchResult, SearchWorkerState
from mara.logging import get_logger

_log = get_logger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


def _t(ns: str, name: str) -> str:
    """Clark-notation tag ``{namespace}localname``."""
    return f"{{{ns}}}{name}"


def _parse_entries(xml_text: str) -> list[dict]:
    """Parse an Atom feed and return a list of paper dicts.

    Each dict contains:
      url            — versioned PDF URL (from <link title="pdf">)
      title          — paper title
      description    — abstract text (for observability only; not hashed)
      authors        — list of author name strings
      published      — submission date string (YYYY-MM-DD)
      primary_category — primary arXiv classification (e.g. "cs.AI")
      journal_ref    — journal reference string, empty if absent

    Entries are skipped when:
      - The <id> contains "/api/errors#" (API error entry)
      - No versioned PDF link is present
      - The abstract (<summary>) is empty
    """
    root = ET.fromstring(xml_text)
    papers: list[dict] = []

    for entry in root.findall(_t(_ATOM_NS, "entry")):
        entry_id = entry.findtext(_t(_ATOM_NS, "id"), "") or ""
        if "/api/errors#" in entry_id:
            _log.warning("ArXiv API error entry: %s", entry_id)
            continue

        title = (entry.findtext(_t(_ATOM_NS, "title"), "") or "").strip()
        abstract = (entry.findtext(_t(_ATOM_NS, "summary"), "") or "").strip()
        published = (entry.findtext(_t(_ATOM_NS, "published"), "") or "")

        authors = [
            (a.findtext(_t(_ATOM_NS, "name"), "") or "").strip()
            for a in entry.findall(_t(_ATOM_NS, "author"))
        ]

        primary_cat_el = entry.find(_t(_ARXIV_NS, "primary_category"))
        primary_category = (
            primary_cat_el.get("term", "") if primary_cat_el is not None else ""
        )

        journal_ref = (
            entry.findtext(_t(_ARXIV_NS, "journal_ref"), "") or ""
        ).strip()

        # Versioned PDF URL — <link title="pdf" rel="related" ...>
        pdf_url = ""
        for link in entry.findall(_t(_ATOM_NS, "link")):
            if link.get("title") == "pdf" and link.get("rel") == "related":
                pdf_url = link.get("href", "")
                break

        if not pdf_url or not abstract:
            continue

        papers.append(
            {
                "url": pdf_url,
                "title": title,
                "description": abstract,
                "authors": authors,
                "published": published[:10] if published else "",
                "primary_category": primary_category,
                "journal_ref": journal_ref,
            }
        )

    return papers


async def arxiv_search(state: SearchWorkerState, config: RunnableConfig) -> dict:
    """Fetch papers from the ArXiv API for the sub-query.

    Returns SearchResult objects whose ``url`` is the versioned PDF URL.
    The ``description`` field contains the abstract for HITL observability
    but is never committed to the Merkle tree.

    The PDF URLs are subsequently scraped by ``firecrawl_scrape`` in the
    arxiv_worker subgraph, producing the actual SourceChunks.

    Args:
        state:  SearchWorkerState with ``sub_query`` and ``research_config``.
        config: RunnableConfig (unused; present for LangGraph compatibility).

    Returns:
        ``{"search_results": list[SearchResult]}``
    """
    research_config = state["research_config"]
    query = state["sub_query"]["query"]
    max_results = research_config.arxiv_max_results

    _log.debug("ArXiv search: %r (max_results=%d)", query, max_results)

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            _ARXIV_API_URL,
            params=params,
            headers={"Accept": "application/xml"},
            timeout=30.0,
        )
        response.raise_for_status()
        xml_text = response.text

    papers = _parse_entries(xml_text)

    results: list[SearchResult] = [
        SearchResult(
            url=p["url"],
            title=p["title"],
            description=p["description"],
            extra_snippets=[],
            page_age=p["published"],
            result_type="arxiv",
        )
        for p in papers
    ]

    _log.debug("ArXiv returned %d paper(s) for %r", len(results), query)
    return {"search_results": results}
