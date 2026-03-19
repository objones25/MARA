"""LLM Self-Assessment (LSA) prompt templates.

The LSA step asks Claude to judge whether a factual claim is supported by
the provided source passages.  The verdict feeds into the composite confidence
score alongside the embedding-based SA and CSC signals.

This module is excluded from coverage measurement (see pyproject.toml).
A formal prompt evaluation framework is planned for a future milestone.
"""

SYSTEM_PROMPT = """\
You are an evidence evaluator for MARA, a Merkle-Assured Research Agent.

Given a factual claim and a set of source passages, determine whether the
claim is supported by the evidence.

Respond with EXACTLY ONE of the following strings and nothing else:
  supported
  partially_supported
  unsupported

Definitions:
  supported            — The claim is clearly and directly confirmed by at
                         least one of the provided passages.
  partially_supported  — The claim is plausible given the passages but not
                         directly confirmed; relevant context is present but
                         incomplete or indirect.
  unsupported          — The passages do not support the claim. Either the
                         relevant information is absent, or the passages
                         contradict the claim.
"""


def build_user_message(claim_text: str, source_texts: list[str]) -> str:
    """Build the user turn for an LSA evaluation request.

    Args:
        claim_text:   The atomic factual claim to evaluate.
        source_texts: Relevant source passages (MerkleLeaf texts).

    Returns:
        A user-turn string ready to be passed to the LLM.
    """
    if source_texts:
        sources = "\n\n".join(
            f"[Source {i + 1}]\n{text}" for i, text in enumerate(source_texts)
        )
    else:
        sources = "(no sources provided)"

    return f"Claim: {claim_text}\n\nSources:\n{sources}"
