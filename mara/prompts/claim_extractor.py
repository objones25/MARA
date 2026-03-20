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

Your task: read a set of numbered source passages and extract the most
important distinct, atomic factual claims they contain.

Rules:
- Each claim must be a SINGLE factual assertion — no compound statements.
  Split "X found Y and also Z" into two separate claims.
- Record ONLY the passage index numbers that directly support each claim.
  Do not include a passage index unless that passage contains clear evidence.
- Omit vague, meta, or trivial statements (e.g. "The article discusses...").
- Prioritise claims with the broadest cross-source support and highest factual
  significance. If there are more claims than the limit, keep the most important.
- Return ONLY a valid JSON array. No preamble, no explanation, no markdown fences.

Output format — strict JSON array, nothing else:
[
  {{"text": "<atomic claim>", "source_indices": [<int>, ...]}},
  ...
]

Example:
[
  {{"text": "Global average temperatures rose by 1.1°C above pre-industrial levels by 2023.", "source_indices": [0, 2]}},
  {{"text": "Arctic sea ice extent reached a record low in September 2023.", "source_indices": [1]}}
]

HARD LIMIT: output AT MOST {max_claims} items. Close the array and stop immediately once you reach {max_claims} items."""


def build_system_prompt(run_date: str, max_claims: int) -> str:
    """Return the system prompt with today's date and claim cap injected.

    Args:
        run_date:   YYYY-MM-DD string for the pipeline start date (UTC).
        max_claims: Maximum number of claims the LLM should extract.
    """
    return f"Today's date is {run_date}.\n\n{_BASE.format(max_claims=max_claims)}"


def build_user_message(passages: list[tuple[int, str, str]], max_claims: int) -> str:
    """Build the user turn for claim extraction.

    Args:
        passages:   List of (index, url, text) tuples — one per MerkleLeaf.
                    ``index`` is the leaf's global position in merkle_leaves.
        max_claims: Maximum number of claims to extract (repeated at end of
                    turn so the constraint appears after the data payload).

    Returns:
        A user-turn string ready to be passed to the LLM.
    """
    lines = ["Source passages:\n"]
    for index, url, text in passages:
        lines.append(f"[{index}] {url}")
        lines.append(text)
        lines.append("")  # blank line between passages
    lines.append(
        f"Extract the most important atomic factual claims from the passages above.\n"
        f"Output ONLY a raw JSON array. Stop after {max_claims} items. No markdown."
    )
    return "\n".join(lines)
