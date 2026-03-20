"""Query Planner prompt templates.

SYSTEM_PROMPT instructs Claude to decompose a research question into focused
sub-queries. build_user_message constructs the per-request user turn.

These strings are intentionally kept separate from node logic so they can be
evaluated, versioned, and A/B tested independently of the orchestration code.
A formal prompt evaluation framework (LangSmith evaluators, prompt versioning)
is planned for a future milestone.
"""

_BASE = """\
You are the Query Planner for MARA, a Merkle-Assured Research Agent.

Your task: decompose a broad research question into focused sub-queries that
together provide comprehensive, non-overlapping coverage of the topic.

Rules:
- Each sub-query must target a DISTINCT angle (e.g. empirical, regulatory,
  historical, economic, technical, clinical, comparative, etc.).
- Write sub-queries as effective web-search strings, NOT as questions.
- Avoid semantic overlap — each sub-query should surface different pages.
- Return ONLY a valid JSON array. No preamble, no explanation, no markdown.

Output format (strict JSON array):
[
  {"query": "<search string>", "domain": "<short domain label>"},
  ...
]"""


def build_system_prompt(run_date: str) -> str:
    """Return the system prompt with today's date injected.

    Args:
        run_date: YYYY-MM-DD string for the pipeline start date (UTC).
    """
    return f"Today's date is {run_date}.\n\n{_BASE}"


def build_user_message(research_question: str, n: int) -> str:
    """Build the user turn for the query planner prompt.

    Args:
        research_question: The raw research question from the user.
        n:                 How many sub-queries to produce (= max_workers).

    Returns:
        A user-turn string ready to be passed to the LLM.
    """
    return (
        f"Research question: {research_question}\n\n"
        f"Output ONLY a raw JSON array with exactly {n} objects. No markdown, no explanation."
    )
