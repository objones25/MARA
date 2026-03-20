"""Prompts for the corrective_retriever node.

These prompts instruct the LLM to generate 1-2 targeted web search queries
for a specific claim that failed confidence scoring.  The goal is to find
additional corroborating (or refuting) sources so the next scoring round has
richer evidence.
"""


def build_system_prompt() -> str:
    """Return the system prompt for corrective sub-query generation.

    Instructs the LLM to produce a compact JSON array of 1-2 targeted search
    queries for a specific failing claim.  Same format as query_planner so
    _parse_sub_queries can be reused directly.
    """
    return (
        "You are a research assistant that generates targeted web search queries "
        "to find additional evidence for a specific factual claim.\n\n"
        "Given a claim that lacks sufficient corroborating sources, generate 1-2 "
        "focused search queries that are likely to find relevant evidence.\n\n"
        "Respond with ONLY a JSON array.  Each element must have exactly two keys:\n"
        '  "query": the search string (be specific and use key terms from the claim)\n'
        '  "domain": the information domain (e.g. "empirical", "statistical", '
        '"academic", "news")\n\n'
        "Example:\n"
        '[{"query": "automation job displacement manufacturing 2020-2024 statistics", '
        '"domain": "statistical"}]\n\n'
        "Do NOT wrap the JSON in markdown fences.  Do NOT add explanations."
    )


def build_user_prompt(claim_text: str, original_query: str) -> str:
    """Return the user prompt for a single failing claim.

    Args:
        claim_text:     The text of the claim that failed confidence scoring.
        original_query: The original research question driving the pipeline.

    Returns:
        A formatted user prompt string.
    """
    return (
        f"Research question: {original_query}\n\n"
        f"Failing claim (insufficient corroborating sources):\n{claim_text}\n\n"
        "Generate 1-2 targeted search queries to find additional evidence for "
        "this specific claim."
    )
