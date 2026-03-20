"""Routing functions for the MARA StateGraph conditional edges."""

from langgraph.types import Send

from mara.agent.state import MARAState
from mara.logging import get_logger

_log = get_logger(__name__)


def dispatch_search_workers(state: MARAState) -> list[Send]:
    """Fan-out one search_worker and one arxiv_worker per sub_query.

    Called on the conditional edge from query_planner.  Each Send carries
    the sub_query and a copy of ResearchConfig so the subgraph is fully
    self-contained — it needs no access to the parent MARAState.

    For each sub-query two workers run concurrently:
      - search_worker: Brave Search → Firecrawl (web pages)
      - arxiv_worker:  ArXiv API   → Firecrawl (academic PDFs)

    Both produce raw_chunks merged into MARAState via operator.add.

    Args:
        state: MARAState with sub_queries populated by query_planner.

    Returns:
        Two Send objects per sub_query (search_worker + arxiv_worker).
    """
    sub_queries = state["sub_queries"]
    research_config = state["config"]
    sends: list[Send] = []
    for q in sub_queries:
        payload = {
            "sub_query": q,
            "research_config": research_config,
            "search_results": [],
            "raw_chunks": [],
        }
        sends.append(Send("search_worker", payload))
        sends.append(Send("arxiv_worker", payload))
    _log.info(
        "Dispatching %d worker(s): %d web + %d arxiv",
        len(sends),
        len(sub_queries),
        len(sub_queries),
    )
    return sends


def route_after_scoring(state: MARAState) -> str:
    """Route after confidence scoring.

    All scored claims — whether high, medium, or low confidence — are routed
    to hitl_checkpoint, which handles the split internally:
      - Claims at or above high_confidence_threshold are auto-approved.
      - Claims below the threshold are presented to the human reviewer.

    A corrective RAG loop (re-dispatching search workers for claims below
    low_confidence_threshold) is planned for a future iteration.  Until that
    node is implemented, this function always returns "hitl_checkpoint".

    Args:
        state: MARAState after confidence_scorer has run.

    Returns:
        "hitl_checkpoint"
    """
    _log.debug(
        "Routing %d scored claim(s) → hitl_checkpoint",
        len(state["scored_claims"]),
    )
    return "hitl_checkpoint"
