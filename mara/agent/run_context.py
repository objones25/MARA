"""RunContext — mutable state container shared across nodes within one graph run.

Passed via ``config["configurable"]["run_context"]``.  Nodes can read and
write fields during the run to share computed artefacts without encoding
them in MARAState (which is serialized to the checkpointer on every step).

Current use: the retriever caches leaf embeddings so the confidence scorer
can skip re-embedding the entire leaf corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunContext:
    """Mutable container injected into LangGraph configurable for a single run.

    Attributes:
        leaf_embeddings:       Float32 matrix of shape (n_leaves, dim) computed
                               by the retriever.  None until the retriever runs.
        leaf_embedding_hashes: Leaf hash list parallel to leaf_embeddings rows,
                               used by the scorer to verify alignment before use.
    """

    leaf_embeddings: np.ndarray | None = None
    leaf_embedding_hashes: list[str] = field(default_factory=list)
