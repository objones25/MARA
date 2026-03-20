"""Claim Extractor node — extracts atomic factual claims from Merkle leaves.

Reads MARAState.retrieved_leaves and calls the LLM to extract every distinct,
atomic factual claim from the source passages.  Each claim is tagged with
the global leaf indices that support it, so the Confidence Scorer can look
up the exact source texts by index.

Why use retrieved_leaves rather than raw_chunks?
  MerkleLeaf carries the ``index`` field — the leaf's position in
  MARAState.merkle_leaves, which is exactly what Claim.source_indices
  references.  Using the already-indexed leaves avoids recomputing offsets
  and keeps source attribution consistent across the pipeline.

Why extract claims before confidence scoring?
  Confidence scoring operates at the claim level, not the document level.
  A compound source sentence like "Study X found Y, and also concluded Z"
  may have Y well supported while Z is not.  Splitting them lets the scorer
  surface mixed evidence instead of averaging it away.

Why a single LLM call over all leaves?
  Cross-leaf claim extraction — detecting that leaf 3 and leaf 7 support
  the same claim — requires the model to see both passages at once.
  Per-leaf extraction would fragment claims that are supported by multiple
  sources and under-populate source_indices.  For very large leaf sets,
  batching can be added later; for the current retrieval scale
  (max_workers * max_sources leaves) a single call is sufficient.
"""

import json

from langchain_core.runnables import RunnableConfig

from mara.agent.llm import make_llm, strip_think
from mara.agent.state import Claim, MARAState, MerkleLeaf
from mara.logging import get_logger
from mara.prompts.claim_extractor import SYSTEM_PROMPT, build_user_message

_log = get_logger(__name__)


def _format_leaves(leaves: list[MerkleLeaf]) -> list[tuple[int, str, str]]:
    """Convert MerkleLeaf list into (index, url, text) tuples for the prompt.

    Args:
        leaves: Ordered list of MerkleLeaf TypedDicts from MARAState.

    Returns:
        List of (index, url, text) tuples ready for build_user_message().
    """
    return [(leaf["index"], leaf["url"], leaf["text"]) for leaf in leaves]


def _parse_claims(content: str) -> list[Claim]:
    """Parse the LLM response into a list of Claim TypedDicts.

    Handles optional ```json ... ``` fences.  Items with empty ``text`` are
    dropped — they represent extraction artefacts rather than real claims.

    Args:
        content: Raw string content from the LLM response.

    Returns:
        list[Claim] — may be empty if the model found no claims.

    Raises:
        json.JSONDecodeError: If the content is not valid JSON after fence
            stripping.
        ValueError: If the parsed JSON is not a list.
    """
    text = strip_think(content)

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()

    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from the LLM, got {type(data).__name__}")

    claims: list[Claim] = []
    for item in data:
        claim_text = str(item.get("text", "")).strip()
        if not claim_text:
            continue
        raw_indices = item.get("source_indices") or []
        source_indices = [int(i) for i in raw_indices]
        claims.append(Claim(text=claim_text, source_indices=source_indices))

    return claims


async def claim_extractor(state: MARAState, config: RunnableConfig) -> dict:
    """Extract atomic factual claims from all Merkle leaves.

    Reads ``state["retrieved_leaves"]`` and ``state["config"]``.  Returns an
    empty list immediately if there are no leaves.

    Returns:
        ``{"extracted_claims": list[Claim]}``
    """
    leaves = state["retrieved_leaves"]

    if not leaves:
        _log.warning("No Merkle leaves — returning empty claim list")
        return {"extracted_claims": []}

    _log.info("Extracting claims from %d leaf/leaves", len(leaves))

    research_config = state["config"]
    llm = make_llm(research_config.model, research_config.hf_token, 4096, research_config.hf_provider)

    passages = _format_leaves(leaves)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(passages)},
    ]

    response = await llm.ainvoke(messages, config)
    claims = _parse_claims(response.content)

    _log.info("Extracted %d claim(s) from %d leaf/leaves", len(claims), len(leaves))
    return {"extracted_claims": claims}
