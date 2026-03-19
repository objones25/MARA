"""Main MARA StateGraph — wires all pipeline nodes and edges.

Node sequence
-------------
    query_planner
        → [search_worker × N]   (fan-out via Send API)
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
from mara.agent.nodes.search_worker.graph import search_worker
from mara.agent.nodes.source_hasher import source_hasher
from mara.agent.state import MARAState
from mara.logging import get_logger

_log = get_logger(__name__)


def build_graph(checkpointer=None):
    """Compile and return the MARA StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver or
            PostgresSaver).  Required for HITL interrupt/resume to work
            correctly across process boundaries.

    Returns:
        A compiled CompiledStateGraph ready for ``invoke`` / ``ainvoke``.
    """
    builder: StateGraph = StateGraph(MARAState)

    # --- Nodes -----------------------------------------------------------
    builder.add_node("query_planner", query_planner)
    builder.add_node("search_worker", search_worker)
    builder.add_node("source_hasher", source_hasher)
    builder.add_node("merkle_builder", merkle_builder)
    builder.add_node("claim_extractor", claim_extractor)
    builder.add_node("confidence_scorer", confidence_scorer)
    builder.add_node("hitl_checkpoint", hitl_checkpoint)
    builder.add_node("report_synthesizer", report_synthesizer)
    builder.add_node("certified_output", certified_output)

    # --- Edges -----------------------------------------------------------
    builder.add_edge(START, "query_planner")

    # Fan-out: one search_worker per sub_query
    builder.add_conditional_edges(
        "query_planner", dispatch_search_workers, ["search_worker"]
    )

    # Fan-in → sequential pipeline
    builder.add_edge("search_worker", "source_hasher")
    builder.add_edge("source_hasher", "merkle_builder")
    builder.add_edge("merkle_builder", "claim_extractor")
    builder.add_edge("claim_extractor", "confidence_scorer")

    # Confidence routing (currently always → hitl_checkpoint)
    builder.add_conditional_edges("confidence_scorer", route_after_scoring)

    # Terminal pipeline
    builder.add_edge("hitl_checkpoint", "report_synthesizer")
    builder.add_edge("report_synthesizer", "certified_output")
    builder.add_edge("certified_output", END)

    _log.info("MARA graph compiled")
    return builder.compile(checkpointer=checkpointer)
