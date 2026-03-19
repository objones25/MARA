"""Tests for mara.logging.

get_logger is a thin wrapper around logging.getLogger. Tests verify the
name is passed through correctly and the stdlib cache is preserved.
"""

import logging

from mara.logging import get_logger


class TestGetLogger:
    def test_returns_logger_instance(self):
        assert isinstance(get_logger("mara.test"), logging.Logger)

    def test_name_passed_through(self):
        logger = get_logger("mara.agent.nodes.query_planner")
        assert logger.name == "mara.agent.nodes.query_planner"

    def test_same_name_returns_same_instance(self):
        l1 = get_logger("mara.test.cache")
        l2 = get_logger("mara.test.cache")
        assert l1 is l2

    def test_different_names_return_different_instances(self):
        l1 = get_logger("mara.a")
        l2 = get_logger("mara.b")
        assert l1 is not l2

    def test_logger_is_in_mara_hierarchy(self):
        logger = get_logger("mara.agent.nodes.source_hasher")
        # Python logger hierarchy: "mara.agent.nodes.source_hasher" is a child of "mara"
        assert logger.name.startswith("mara.")
