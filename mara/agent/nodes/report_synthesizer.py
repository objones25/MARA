"""Report Synthesizer node — writes the research report from approved claims.

Reads the human-approved (or auto-approved) ScoredClaims and the Merkle
leaves, formats each claim with its inline citations, and asks the LLM to
synthesise a coherent research report.

Citation format (from README): [ML:index:hash_prefix]
  ML         — Merkle Leaf
  index      — leaf's zero-based position in MARAState.merkle_leaves
  hash_prefix — first 6 characters of the leaf's SHA-256 hex digest

Why format citations before the LLM call?
  Providing pre-built citation strings prevents the model from inventing
  citation indices.  The model is instructed to use the exact [ML:...] tags
  shown in the formatted claims — its only creative freedom is prose style
  and structure.
"""

from langchain_core.runnables import RunnableConfig

from mara.agent.llm import make_llm, strip_think
from mara.agent.state import MARAState, MerkleLeaf
from mara.logging import get_logger
from mara.prompts.report_synthesizer import build_system_prompt, build_user_message

_log = get_logger(__name__)


def _citation(leaf: MerkleLeaf) -> str:
    """Build the inline citation string for a single leaf."""
    return f"[ML:{leaf['index']}:{leaf['hash'][:6]}]"


def _format_claims(claims: list, leaves: list[MerkleLeaf]) -> str:
    """Format approved ScoredClaims as numbered lines with citation tags.

    Each line: ``- <claim text> (confidence: 0.87) [ML:0:a4f2c1] [ML:2:8e3d90]``

    Args:
        claims: list[ScoredClaim] dataclass instances.
        leaves: Ordered list of MerkleLeaf TypedDicts from MARAState.

    Returns:
        Multi-line string of formatted claims ready for the prompt.
    """
    leaf_by_index = {leaf["index"]: leaf for leaf in leaves}
    lines = []
    for claim in claims:
        citations = " ".join(
            _citation(leaf_by_index[i]) for i in claim.source_indices if i in leaf_by_index
        )
        line = f"- {claim.text} (confidence: {claim.confidence:.2f})"
        if citations:
            line = f"{line} {citations}"
        lines.append(line)
    return "\n".join(lines)


async def report_synthesizer(state: MARAState, config: RunnableConfig) -> dict:
    """Synthesise the research report from approved claims.

    Uses ``human_approved_claims`` if the HITL node ran; falls back to
    ``scored_claims`` for runs where all claims were auto-approved and the
    HITL node wrote nothing.

    Returns:
        ``{"report_draft": str}``
    """
    claims = state["human_approved_claims"] or state["scored_claims"]
    leaves = state["retrieved_leaves"]
    query = state["query"]

    _log.info("Synthesising report from %d approved claim(s)", len(claims))

    if not claims:
        _log.warning("No approved claims — report will be empty")
        return {"report_draft": ""}

    research_config = state["config"]
    llm = make_llm(research_config.model, research_config.hf_token, 8192, research_config.hf_provider)

    formatted = _format_claims(claims, leaves)
    messages = [
        {"role": "system", "content": build_system_prompt(state["run_date"])},
        {"role": "user", "content": build_user_message(query, formatted)},
    ]

    response = await llm.ainvoke(messages, config)
    report_text = strip_think(response.content)

    _log.info("Report synthesised (%d characters)", len(report_text))
    return {"report_draft": report_text}
