"""Routing functions for the MARA StateGraph conditional edges."""

from langgraph.types import Send

from mara.agent.state import MARAState
from mara.logging import get_logger

_log = get_logger(__name__)


def dispatch_search_workers(state: MARAState) -> list[Send]:
    """Fan-out one search_worker subgraph invocation per sub_query.

    Called on the conditional edge from query_planner.  Each Send carries
    the sub_query and a copy of ResearchConfig so the subgraph is fully
    self-contained — it needs no access to the parent MARAState.

    Args:
        state: MARAState with sub_queries populated by query_planner.

    Returns:
        One Send object per sub_query, targeting the "search_worker" node.
    """
    sub_queries = state["sub_queries"]
    research_config = state["config"]
    _log.info("Dispatching %d search worker(s)", len(sub_queries))
    return [
        Send(
            "search_worker",
            {
                "sub_query": q,
                "research_config": research_config,
                "search_results": [],
                "raw_chunks": [],
            },
        )
        for q in sub_queries
    ]


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
