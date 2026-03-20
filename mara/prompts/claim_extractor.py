"""Claim Extractor prompt templates.

The claim extractor asks Claude to read a set of numbered source passages and
return a list of atomic factual claims, each tagged with the passage indices
that support it.

Kept in mara/prompts/ so it can be versioned, evaluated, and A/B tested
independently of the orchestration code.  A formal prompt evaluation
framework is planned for a future milestone.
"""

_BASE = """\
You are a claim extractor for MARA, a Merkle-Assured Research Agent.

Your task: read a set of numbered source passages and extract every distinct,
atomic factual claim they contain.

Rules:
- Each claim must be a SINGLE factual assertion — no compound statements.
  Split "X found Y and also Z" into two separate claims.
- Record ONLY the passage index numbers that directly support each claim.
  Do not include a passage index unless that passage contains clear evidence.
- Omit vague, meta, or trivial statements (e.g. "The article discusses...").
- Return ONLY a valid JSON array. No preamble, no explanation, no markdown.

Output format (strict JSON array):
[
  {"text": "<atomic claim>", "source_indices": [<int>, ...]},
  ...
]"""


def build_system_prompt(run_date: str) -> str:
    """Return the system prompt with today's date injected.

    Args:
        run_date: YYYY-MM-DD string for the pipeline start date (UTC).
    """
    return f"Today's date is {run_date}.\n\n{_BASE}"


def build_user_message(passages: list[tuple[int, str, str]]) -> str:
    """Build the user turn for claim extraction.

    Args:
        passages: List of (index, url, text) tuples — one per MerkleLeaf.
                  ``index`` is the leaf's global position in merkle_leaves.

    Returns:
        A user-turn string ready to be passed to the LLM.
    """
    lines = ["Source passages:\n"]
    for index, url, text in passages:
        lines.append(f"[{index}] {url}")
        lines.append(text)
        lines.append("")  # blank line between passages
    lines.append("Extract all atomic factual claims from the passages above.")
    return "\n".join(lines)
