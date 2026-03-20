"""Report Synthesizer prompt templates.

The report synthesizer asks Claude to write a coherent research report from
a set of confidence-scored, human-approved claims, using inline Merkle
citations throughout.

Kept in mara/prompts/ so it can be versioned and evaluated independently.
"""

_BASE = """\
You are the Report Synthesizer for MARA, a Merkle-Assured Research Agent.

Your task: write a well-structured, flowing research report that answers
the research question using ONLY the approved claims provided.

Rules:
- Place an inline citation immediately after every factual statement.
  Citation format: [ML:index:hash_prefix]  (e.g. [ML:3:a4f2c1])
- Each citation must match one of the [ML:...] tags shown in the claims.
- Do not fabricate facts or introduce information not in the approved claims.
- Write in clear, formal prose — no bullet lists in the final report.
- Structure: brief introduction, thematic body paragraphs, concise conclusion.
- State the research date (provided below) in the introduction so readers
  know when this research was conducted."""


def build_system_prompt(run_date: str) -> str:
    """Return the system prompt with today's date injected.

    Args:
        run_date: YYYY-MM-DD string for the pipeline start date (UTC).
    """
    return f"Today's date is {run_date}.\n\n{_BASE}"


def build_user_message(query: str, formatted_claims: str) -> str:
    """Build the user turn for report synthesis.

    Args:
        query:            The original research question.
        formatted_claims: Pre-formatted claim lines, each with citation tags.

    Returns:
        A user-turn string ready to be passed to the LLM.
    """
    return (
        f"Research question: {query}\n\n"
        f"Approved claims:\n{formatted_claims}\n\n"
        "Write the research report now."
    )
