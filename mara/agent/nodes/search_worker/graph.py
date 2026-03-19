"""Compiled search worker subgraph.

The search worker is a two-node pipeline:

    brave_search  →  firecrawl_scrape

It is invoked in parallel for each SubQuery via LangGraph's Send() API.  The
parent graph uses an ``Annotated[list[SourceChunk], operator.add]`` reducer on
``raw_chunks`` so that each worker's output is merged (not overwritten) into the
shared state.

Usage (from the parent graph's fan-out edge):

    from mara.agent.nodes.search_worker.graph import search_worker

    def dispatch_workers(state: MARAState):
        return [
            Send("search_worker", {
                "sub_query": q,
                "research_config": state["config"],
                "search_results": [],
                "raw_chunks": [],
            })
            for q in state["sub_queries"]
        ]

    graph.add_node("search_worker", search_worker)
    graph.add_conditional_edges("query_planner", dispatch_workers, ["search_worker"])
"""

from langgraph.graph import StateGraph, START, END

from mara.agent.state import SearchWorkerState
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


search_worker = _build_search_worker()
