"""Agent state definitions.

MARAState is the top-level graph state. SearchWorkerState is the state for the
search worker subgraph, which is compiled separately and invoked via Send().

All user-defined types here are TypedDict so they remain JSON-serializable by
LangGraph's checkpointing layer.
"""

import operator
from typing import Any, Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# Search worker types
# ---------------------------------------------------------------------------


class SubQuery(TypedDict):
    """A focused sub-query produced by the Query Planner.

    ``query`` is the search string dispatched to Brave.
    ``domain`` is a hint (e.g. "economics", "empirical") used by the planner to
    balance coverage; it is logged but not consumed by the search worker.
    """

    query: str
    domain: str


class SearchResult(TypedDict):
    """A single result returned by the Brave Search API.

    Brave results are available to all downstream nodes via MARAState.search_results
    but are NEVER committed to the Merkle tree.  Two reasons:

    1. Provider-controlled, not reproducible.  Brave's snippets and titles are
       Brave's own extractions of the page.  Brave re-crawls and updates its
       extraction logic continuously; the same URL queried tomorrow may return
       different extra_snippets or a different description.  A hash of Brave's
       output would produce false verification failures that indicate nothing
       about tampering.

    2. Not the authoritative source text.  A Brave snippet is a third-party
       excerpt, not the raw page.  Hashing it would only prove Brave said a
       page contained certain text — not that the page actually says it.
       Merkle leaves must bind claims to the exact bytes MARA scraped itself.

    Brave data informs the agent (pre-screening, recency signals, HITL context,
    source attribution metadata) but cannot serve as a source of record for
    claim integrity.  Only SourceChunks — raw text scraped by firecrawl_scrape
    and chunked deterministically — are hashed.

    ``extra_snippets``: up to 5 additional page excerpts from Brave when
    ``extra_snippets=true`` is sent.  Useful for pre-screening before scraping.
    ``page_age``: publication/last-modified hint from Brave (string, not parsed).
    ``result_type``: which Brave response section this result came from
    (``"web"`` | ``"news"`` | ``"discussion"`` | ``"faq"``).
    """

    url: str
    title: str
    description: str
    extra_snippets: list[str]
    page_age: str
    result_type: str  # "web" | "news" | "discussion" | "faq"


class SourceChunk(TypedDict):
    """A fixed-size text chunk extracted from a scraped page.

    This is the atomic unit that gets passed to the Source Hasher to become a
    MerkleLeaf.  The (url, text, retrieved_at) triple is what gets hashed by
    ``mara.merkle.hasher.canonical_serialise``.
    """

    url: str
    text: str
    retrieved_at: str  # ISO-8601, e.g. "2026-03-19T10:30:45Z"
    sub_query: str     # originating sub-query text (for observability)


# ---------------------------------------------------------------------------
# Search worker subgraph state
# ---------------------------------------------------------------------------


class SearchWorkerState(TypedDict):
    """Private state for the search_worker compiled subgraph.

    Each parallel worker receives ``sub_query`` and ``research_config`` via
    ``Send()``.  The worker populates ``search_results`` (Brave) then
    ``raw_chunks`` (Firecrawl) before returning.

    ``raw_chunks`` is the only field the parent graph reads back; the
    ``Annotated[list[SourceChunk], operator.add]`` reducer on ``MARAState``
    merges all workers' chunks into a single list.
    """

    sub_query: SubQuery
    research_config: ResearchConfig  # passed explicitly via Send payload
    search_results: list[SearchResult]
    raw_chunks: list[SourceChunk]


# ---------------------------------------------------------------------------
# Top-level graph state
# ---------------------------------------------------------------------------


class MARAState(TypedDict):
    """State for the top-level MARA StateGraph.

    Fields annotated with a reducer (e.g. ``operator.add``) are safe to update
    from parallel nodes.  All other fields are written by exactly one node.

    Types from not-yet-implemented modules (Claim, CertifiedReport, MerkleLeaf,
    MerkleTree) are annotated as ``Any`` here and will be tightened when those
    modules are built.
    """

    # ---- Input ----
    query: str
    config: ResearchConfig

    # ---- Planner output ----
    sub_queries: list[SubQuery]

    # ---- Search worker fan-in (parallel reduce) ----
    search_results: Annotated[list[SearchResult], operator.add]
    raw_chunks: Annotated[list[SourceChunk], operator.add]

    # ---- Source Hasher / Merkle output ----
    merkle_leaves: list[Any]   # list[MerkleLeaf]
    merkle_tree: Any           # MerkleTree

    # ---- Confidence Scorer output ----
    extracted_claims: list[Any]    # list[Claim]
    scored_claims: list[Any]       # list[ScoredClaim]

    # ---- HITL Checkpoint output ----
    human_approved_claims: list[Any]  # list[ScoredClaim]

    # ---- Report Synthesizer output ----
    report_draft: str

    # ---- Certified Output ----
    certified_report: Any  # CertifiedReport | None

    # ---- Internal ----
    messages: Annotated[list, add_messages]
    loop_count: int  # corrective RAG iteration counter
