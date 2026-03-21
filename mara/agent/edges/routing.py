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
        sends.append(Send("semantic_scholar_worker", payload))
    _log.info(
        "Dispatching %d worker(s): %d web + %d arxiv + %d semantic_scholar",
        len(sends),
        len(sub_queries),
        len(sub_queries),
        len(sub_queries),
    )
    return sends


def route_after_scoring(state: MARAState) -> str:
    """Route after confidence scoring.

    Claims with low SA and insufficient corroboration
    (corroborating < n_leaves_contested_threshold) are routed to
    corrective_retriever to acquire more data, as long as the loop cap has
    not been reached.

    Since scoring now runs against the full merkle_leaves corpus, n_unique_urls
    is always the total corpus size and cannot distinguish "few sources checked"
    from "many sources checked but none corroborate".  corroborating (k) is the
    correct signal: k=0 with 69 sources means no evidence was found, not that
    evidence is genuinely contested.

    Contested claims (low SA but k >= n_leaves_contested_threshold — enough
    independent sources were checked yet still failed to corroborate) and
    approved claims are sent directly to hitl_checkpoint.

    Args:
        state: MARAState after confidence_scorer has run.

    Returns:
        "corrective_retriever" if there are failing claims and loop budget remains,
        "hitl_checkpoint" otherwise.
    """
    cfg = state["config"]
    failing = [
        c for c in state["scored_claims"]
        if c.confidence < cfg.low_confidence_threshold
        and c.corroborating < cfg.n_leaves_contested_threshold
    ]
    if failing and state["loop_count"] < cfg.max_corrective_rag_loops:
        _log.info(
            "%d failing claim(s), loop %d/%d → corrective_retriever",
            len(failing),
            state["loop_count"],
            cfg.max_corrective_rag_loops,
        )
        return "corrective_retriever"
    _log.debug(
        "Routing %d scored claim(s) → hitl_checkpoint",
        len(state["scored_claims"]),
    )
    return "hitl_checkpoint"
