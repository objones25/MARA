"""Certified Output node — assembles the final CertifiedReport.

This is the terminal node of the MARA pipeline.  It reads all the outputs
accumulated across the pipeline and packages them into a single
``CertifiedReport`` dataclass that is:

  - Self-contained: the report text, Merkle root, all leaves, and all scored
    claims are bundled together so a verifier needs only this object.
  - Verifiable: any reader can recompute leaf hashes from (url, text,
    retrieved_at), rebuild the Merkle tree, and confirm the root matches.
  - Auditable: all scored_claims are included with their signal breakdowns
    (SA, CSC, LSA, composite confidence) for transparency.

No LLM call is made here — this is a pure assembly step.
"""

from langchain_core.runnables import RunnableConfig

from mara.agent.state import CertifiedReport, MARAState
from mara.logging import get_logger
from mara.merkle.tree import build_merkle_tree

_log = get_logger(__name__)


def certified_output(state: MARAState, config: RunnableConfig) -> dict:
    """Assemble and return the CertifiedReport.

    Args:
        state:  Fully populated MARAState after report_synthesizer has run.
        config: LangGraph RunnableConfig (unused directly).

    Returns:
        ``{"certified_report": CertifiedReport}``
    """
    algorithm = state["config"].hash_algorithm
    retrieved = list(state["retrieved_leaves"])

    if retrieved:
        leaf_hashes = [leaf["hash"] for leaf in retrieved]
        merkle_root = build_merkle_tree(leaf_hashes, algorithm).root
    else:
        merkle_root = ""

    approved_claims = state["human_approved_claims"] or state["scored_claims"]

    report = CertifiedReport(
        query=state["query"],
        report_text=state["report_draft"],
        merkle_root=merkle_root,
        leaves=retrieved,
        scored_claims=list(approved_claims),
        hash_algorithm=algorithm,
    )

    _log.info(
        "Certified report assembled — root %s…, %d retrieved leaf/leaves, %d claim(s)",
        merkle_root[:12] if merkle_root else "(empty)",
        len(report.leaves),
        len(report.scored_claims),
    )
    return {"certified_report": report}
