"""Centralized logger factory for MARA.

All modules obtain a logger via ``get_logger(__name__)``, which places it in
the ``mara.*`` hierarchy.  This means the entire application can be silenced,
filtered, or directed to a file handler by configuring the root ``mara``
logger once at application startup:

    import logging
    logging.getLogger("mara").setLevel(logging.DEBUG)

LangSmith handles tracing for all LLM calls automatically.  This logger
covers the non-LLM parts of the pipeline — node entry/exit timing, business
logic events ("generated N sub-queries", "hashed N chunks"), and degraded
but non-fatal conditions ("LLM under-produced sub-queries").

Usage (in any mara module):

    from mara.logging import get_logger
    _log = get_logger(__name__)
    _log.info("Processing %d chunks", n)
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a standard library Logger for the given name.

    Callers should pass ``__name__`` so the logger sits in the correct
    position of the ``mara.*`` hierarchy.

    Args:
        name: Fully qualified module name (e.g. ``mara.agent.nodes.query_planner``).

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
