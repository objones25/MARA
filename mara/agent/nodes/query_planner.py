"""Query Planner node for the top-level MARA graph.

Calls the LLM to decompose the user's research question into ``max_workers``
focused sub-queries. Each sub-query is dispatched to a parallel search worker
via LangGraph's Send() API by the fan-out edge that follows this node.

Why max_workers sub-queries?
  One sub-query per worker maximises parallel throughput while keeping the
  total number of Brave API calls bounded by ResearchConfig.max_workers.
  Producing more sub-queries than workers would queue extras sequentially,
  negating the benefit of parallelism.

Why parse JSON manually instead of using structured output?
  SubQuery is a TypedDict (JSON-serializable, LangGraph checkpointable).
  Introducing a parallel Pydantic model just for the LLM boundary adds
  unnecessary indirection. A prompt that returns a bare JSON array is equally
  reliable and keeps the data model in one place.

Why strip markdown fences?
  Models occasionally wrap JSON in ```json ... ``` even when instructed not
  to. _parse_sub_queries strips fences defensively without relaxing the
  prompt constraint.
"""

import json
import re

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableConfig

from mara.agent.state import MARAState, SubQuery
from mara.logging import get_logger
from mara.prompts.query_planner import SYSTEM_PROMPT, build_user_message

_log = get_logger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _make_llm(model: str, hf_token: str) -> ChatHuggingFace:
    """Instantiate the ChatHuggingFace client via HuggingFace Inference Providers."""
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=1024,
    )
    return ChatHuggingFace(llm=endpoint)


def _strip_think(text: str) -> str:
    """Strip Qwen3 thinking tokens from LLM output."""
    return _THINK_RE.sub("", text).strip()


def _parse_sub_queries(content: str, n: int) -> list[SubQuery]:
    """Parse the LLM response into a list of SubQuery TypedDicts.

    Handles optional ```json ... ``` fences. Returns up to ``n`` items;
    if the model returns fewer, all items are included without error.

    Args:
        content: Raw string content from the LLM response.
        n:       Maximum number of sub-queries to return.

    Returns:
        list[SubQuery] — may be shorter than n if the model under-produced.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON after fence
            stripping.
        ValueError: If the parsed JSON is not a list.
    """
    text = _strip_think(content)

    # Strip optional markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        text = "\n".join(lines[1:-1]).strip()

    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from the LLM, got {type(data).__name__}")

    sub_queries: list[SubQuery] = []
    for item in data[:n]:
        sub_queries.append(
            SubQuery(
                query=str(item["query"]),
                domain=str(item.get("domain", "general")),
            )
        )
    return sub_queries


async def query_planner(state: MARAState, config: RunnableConfig) -> dict:
    """Decompose the research question into sub-queries for parallel workers.

    Reads ``state["query"]`` and ``state["config"]``.  Calls the LLM with the
    query planner prompt and parses the JSON array response into SubQuery
    TypedDicts.

    Returns:
        ``{"sub_queries": list[SubQuery]}``
    """
    research_config = state["config"]
    n = research_config.max_workers

    _log.info("Planning query into %d sub-queries: %r", n, state["query"])

    llm = _make_llm(research_config.model, research_config.hf_token)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(state["query"], n)},
    ]

    response = await llm.ainvoke(messages, config)
    sub_queries = _parse_sub_queries(response.content, n)

    if len(sub_queries) < n:
        _log.warning(
            "LLM produced %d sub-queries, expected %d", len(sub_queries), n
        )
    else:
        _log.info("Generated %d sub-queries", len(sub_queries))

    return {"sub_queries": sub_queries}
