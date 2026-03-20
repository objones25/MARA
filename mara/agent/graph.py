"""Main MARA StateGraph — wires all pipeline nodes and edges.

Node sequence
-------------
    query_planner
        → [search_worker × N]   (fan-out via Send API — Brave + Firecrawl)
        → [arxiv_worker  × N]   (fan-out via Send API — ArXiv API + Firecrawl)
        → source_hasher
        → merkle_builder
        → claim_extractor
        → confidence_scorer
        → hitl_checkpoint       (auto-approves high-confidence; interrupts for low)
        → report_synthesizer
        → certified_output
        → END

HITL interrupt / resume
-----------------------
The hitl_checkpoint node uses LangGraph's interrupt() to pause execution when
low-confidence claims need human review.  A checkpointer is required for this
to work correctly — without one, interrupt() raises at runtime.

Pass a checkpointer to build_graph() for interactive use:

    from langgraph.checkpoint.memory import MemorySaver
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "run-1"}}

    # First call — runs until the interrupt
    graph.invoke({"query": "...", "config": research_config}, config)

    # Resume after human review
    from langgraph.types import Command
    graph.invoke(Command(resume={"approved_indices": [0, 2]}), config)
"""

from langgraph.graph import END, START, StateGraph

from mara.agent.edges.routing import dispatch_search_workers, route_after_scoring
from mara.agent.nodes.certified_output import certified_output
from mara.agent.nodes.claim_extractor import claim_extractor
from mara.agent.nodes.confidence_scorer import confidence_scorer
from mara.agent.nodes.hitl_checkpoint import hitl_checkpoint
from mara.agent.nodes.merkle_builder import merkle_builder
from mara.agent.nodes.query_planner import query_planner
from mara.agent.nodes.report_synthesizer import report_synthesizer
from mara.agent.nodes.retriever import retriever
from mara.agent.nodes.search_worker.graph import arxiv_worker, search_worker
from mara.agent.nodes.source_hasher import source_hasher
from mara.agent.state import MARAState
from mara.logging import get_logger

_log = get_logger(__name__)


def build_graph(checkpointer=None, config_schemas=None):
    """Compile and return the MARA StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver or
            PostgresSaver).  Required for HITL interrupt/resume to work
            correctly across process boundaries.
        config_schemas: Ignored — present for forward-compatibility with
            LangGraph's configurable schema API.

    Returns:
        A compiled CompiledStateGraph ready for ``invoke`` / ``ainvoke``.

    Leaf DB injection
    -----------------
    The ``leaf_repo`` (``SQLiteLeafRepository``) and ``run_id`` (UUID4) are
    injected via ``RunnableConfig["configurable"]`` by the CLI before each
    run.  Nodes that need them pull them out of ``config["configurable"]``
    directly — they are never placed in typed state (non-serialisable objects
    cannot go into LangGraph state).
    """
    builder: StateGraph = StateGraph(MARAState)

    # --- Nodes -----------------------------------------------------------
    builder.add_node("query_planner", query_planner)
    builder.add_node("search_worker", search_worker)
    builder.add_node("arxiv_worker", arxiv_worker)
    builder.add_node("source_hasher", source_hasher)
    builder.add_node("merkle_builder", merkle_builder)
    builder.add_node("retriever", retriever)
    builder.add_node("claim_extractor", claim_extractor)
    builder.add_node("confidence_scorer", confidence_scorer)
    builder.add_node("hitl_checkpoint", hitl_checkpoint)
    builder.add_node("report_synthesizer", report_synthesizer)
    builder.add_node("certified_output", certified_output)

    # --- Edges -----------------------------------------------------------
    builder.add_edge(START, "query_planner")

    # Fan-out: search_worker + arxiv_worker per sub_query
    builder.add_conditional_edges(
        "query_planner", dispatch_search_workers, ["search_worker", "arxiv_worker"]
    )

    # Fan-in: both worker types converge on source_hasher
    builder.add_edge("search_worker", "source_hasher")
    builder.add_edge("arxiv_worker", "source_hasher")
    builder.add_edge("source_hasher", "merkle_builder")
    builder.add_edge("merkle_builder", "retriever")
    builder.add_edge("retriever", "claim_extractor")
    builder.add_edge("claim_extractor", "confidence_scorer")

    # Confidence routing (currently always → hitl_checkpoint)
    builder.add_conditional_edges("confidence_scorer", route_after_scoring)

    # Terminal pipeline
    builder.add_edge("hitl_checkpoint", "report_synthesizer")
    builder.add_edge("report_synthesizer", "certified_output")
    builder.add_edge("certified_output", END)

    _log.info("MARA graph compiled")
    return builder.compile(checkpointer=checkpointer)
