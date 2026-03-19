"""Tests for mara.agent.graph.build_graph.

Tests verify the graph compiles without error and has the expected node
topology.  No end-to-end invocation is performed — that would require real
API keys and a running checkpointer.
"""

import pytest

from mara.agent.graph import build_graph

EXPECTED_NODES = {
    "query_planner",
    "search_worker",
    "source_hasher",
    "merkle_builder",
    "retriever",
    "claim_extractor",
    "confidence_scorer",
    "hitl_checkpoint",
    "report_synthesizer",
    "certified_output",
}


class TestBuildGraph:
    def test_build_graph_does_not_raise(self):
        graph = build_graph()
        assert graph is not None

    def test_build_graph_accepts_none_checkpointer(self):
        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_returns_compiled_graph_with_invoke(self):
        graph = build_graph()
        assert hasattr(graph, "invoke")

    def test_returns_compiled_graph_with_ainvoke(self):
        graph = build_graph()
        assert hasattr(graph, "ainvoke")

    def test_all_expected_nodes_present(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        assert EXPECTED_NODES.issubset(node_names)

    def test_no_unexpected_non_internal_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        # LangGraph adds __start__ and __end__ internally
        extra = node_names - EXPECTED_NODES - {"__start__", "__end__"}
        assert extra == set()

    def test_query_planner_node_present(self):
        graph = build_graph()
        assert "query_planner" in graph.nodes

    def test_search_worker_node_present(self):
        graph = build_graph()
        assert "search_worker" in graph.nodes

    def test_hitl_checkpoint_node_present(self):
        graph = build_graph()
        assert "hitl_checkpoint" in graph.nodes

    def test_certified_output_node_present(self):
        graph = build_graph()
        assert "certified_output" in graph.nodes
