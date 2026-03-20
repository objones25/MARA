"""Compiled search worker subgraphs.

Two subgraphs are exported, each dispatched in parallel per sub-query:

    search_worker:  brave_search  →  firecrawl_scrape
    arxiv_worker:   arxiv_search  →  firecrawl_scrape

Both subgraphs use the same SearchWorkerState and the same firecrawl_scrape
node implementation.  search_worker retrieves web pages via Brave Search;
arxiv_worker retrieves full academic papers via the ArXiv API, passing
versioned PDF URLs to firecrawl_scrape for full-text extraction.

Usage (from the parent graph's fan-out edge):

    from mara.agent.nodes.search_worker.graph import arxiv_worker, search_worker

    def dispatch_workers(state: MARAState):
        sends = []
        for q in state["sub_queries"]:
            payload = {
                "sub_query": q,
                "research_config": state["config"],
                "search_results": [],
                "raw_chunks": [],
            }
            sends.append(Send("search_worker", payload))
            sends.append(Send("arxiv_worker", payload))
        return sends

    graph.add_node("search_worker", search_worker)
    graph.add_node("arxiv_worker", arxiv_worker)
    graph.add_conditional_edges(
        "query_planner", dispatch_workers, ["search_worker", "arxiv_worker"]
    )
"""

from langgraph.graph import StateGraph, START, END

from mara.agent.state import SearchWorkerState
from .arxiv_search import arxiv_search
from .brave_search import brave_search
from .firecrawl_scrape import firecrawl_scrape


def _build_search_worker():
    builder: StateGraph = StateGraph(SearchWorkerState)
    builder.add_node("brave_search", brave_search)
    builder.add_node("firecrawl_scrape", firecrawl_scrape)
    builder.add_edge(START, "brave_search")
    builder.add_edge("brave_search", "firecrawl_scrape")
    builder.add_edge("firecrawl_scrape", END)
    return builder.compile()


def _build_arxiv_worker():
    builder: StateGraph = StateGraph(SearchWorkerState)
    builder.add_node("arxiv_search", arxiv_search)
    builder.add_node("firecrawl_scrape", firecrawl_scrape)
    builder.add_edge(START, "arxiv_search")
    builder.add_edge("arxiv_search", "firecrawl_scrape")
    builder.add_edge("firecrawl_scrape", END)
    return builder.compile()


search_worker = _build_search_worker()
arxiv_worker = _build_arxiv_worker()
