"""HITL Checkpoint node — pauses for human review of low-confidence claims.

Claims whose composite confidence score is at or above
``ResearchConfig.high_confidence_threshold`` are auto-approved and pass
through without interruption.  Claims below the threshold are surfaced to a
human reviewer via LangGraph's ``interrupt()`` mechanism.

How interrupt() works:
  Calling ``interrupt(payload)`` raises a ``GraphInterrupt`` exception that
  LangGraph catches, persists the graph state to the checkpointer, and
  returns control to the caller.  The payload is available on the result
  under the ``__interrupt__`` key.  Execution resumes when the caller
  invokes the graph again with ``Command(resume=value)``, at which point
  ``interrupt()`` returns ``value`` inside this node.

Resume payload contract:
  The caller must resume with a dict:
    ``{"approved_indices": list[int]}``
  where each integer is a zero-based index into the ``needs_review`` list
  that was included in the interrupt payload.  Claims not listed are
  considered rejected.

Why indices instead of full claim objects?
  The interrupt payload is serialised by the checkpointer.  Passing
  lightweight indices keeps the resume payload small and avoids
  deserialisation of dataclass instances across process boundaries.
"""

import statistics
from dataclasses import replace

from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState
from mara.logging import get_logger

_log = get_logger(__name__)


def hitl_checkpoint(state: MARAState, config: RunnableConfig) -> dict:
    """Auto-approve high-confidence claims; interrupt for the rest.

    Contested claims (low confidence but large n_leaves — sources exist but
    disagree) are flagged with ``contested=True`` before entering the review
    flow so that human reviewers have full context.

    Args:
        state:  MARAState with ``scored_claims`` and ``config`` populated.
        config: LangGraph RunnableConfig (unused directly).

    Returns:
        ``{"human_approved_claims": list[ScoredClaim]}``
    """
    cfg = state["config"]
    high_threshold = cfg.high_confidence_threshold

    # Mark contested claims before any HITL logic — covers first pass,
    # post-loop-cap, and corrective rounds that exhausted the budget.
    scored = []
    for claim in state["scored_claims"]:
        if (
            claim.confidence < cfg.low_confidence_threshold
            and claim.n_unique_urls >= cfg.n_leaves_contested_threshold
        ):
            claim = replace(claim, contested=True)
        scored.append(claim)

    confidences = [c.confidence for c in scored]
    if confidences:
        _log.info(
            "Confidence stats — mean: %.3f  median: %.3f  std dev: %.3f  min: %.3f  max: %.3f",
            statistics.mean(confidences),
            statistics.median(confidences),
            statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            min(confidences),
            max(confidences),
        )

    auto_approved = [c for c in scored if c.confidence >= high_threshold]
    needs_review = [c for c in scored if c.confidence < high_threshold]

    if not needs_review:
        _log.info(
            "All %d claim(s) auto-approved (confidence >= %.2f)",
            len(auto_approved),
            high_threshold,
        )
        return {"human_approved_claims": auto_approved}

    _log.info(
        "%d claim(s) auto-approved, %d claim(s) sent for human review",
        len(auto_approved),
        len(needs_review),
    )

    # Serialisable payload — no dataclass instances
    review_payload = [
        {
            "index": i,
            "text": c.text,
            "confidence": c.confidence,
            "corroborating": c.corroborating,
            "n_leaves": c.n_leaves,
            "n_unique_urls": c.n_unique_urls,
            "source_indices": c.source_indices,
            "contested": c.contested,
        }
        for i, c in enumerate(needs_review)
    ]

    decision = interrupt(
        {
            "needs_review": review_payload,
            "auto_approved_count": len(auto_approved),
        }
    )

    approved_indices: list[int] = decision.get("approved_indices", [])
    human_approved = [
        needs_review[i] for i in approved_indices if i < len(needs_review)
    ]

    _log.info(
        "Human approved %d of %d reviewed claim(s)",
        len(human_approved),
        len(needs_review),
    )
    return {"human_approved_claims": auto_approved + human_approved}
