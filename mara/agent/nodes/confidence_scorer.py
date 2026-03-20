"""Confidence Scorer node — scores each extracted claim against Merkle leaves.

For each Claim in MARAState.extracted_claims, this node:
  1. Retrieves the source texts for the claim's source_indices from merkle_leaves.
  2. Calls score_claim() which computes SA, CSC, and LSA signals and returns a
     ScoredClaim dataclass with a composite confidence score.
  3. The LSA step (LLM Self-Assessment) is handled by a synchronous callable
     that calls the LLM with the lsa_scorer prompt.

Why run score_claim via asyncio.to_thread?
  score_claim() contains two synchronous blocking calls: embed() (loads a
  SentenceTransformer model and runs inference) and the LSA callable (calls
  the LLM API synchronously).  Wrapping in asyncio.to_thread allows the
  LangGraph event loop to remain responsive while the blocking I/O runs in
  the default thread pool executor.

Why a synchronous LSA callable (not async)?
  score_claim() in mara/confidence/scorer.py accepts a plain Callable, not a
  Coroutine.  Keeping it sync preserves the scorer's independence from the
  async stack and keeps it testable without an event loop.  The thread-pool
  wrapper at the node level provides the async boundary.
"""

import asyncio

from langchain_core.runnables import RunnableConfig

from mara.agent.llm import ChatHuggingFace, make_llm, strip_think
from mara.agent.state import MARAState
from mara.confidence.scorer import score_claim
from mara.confidence.signals import LSAVerdict
from mara.logging import get_logger
from mara.prompts.lsa_scorer import SYSTEM_PROMPT, build_user_message

_log = get_logger(__name__)


def _call_lsa(
    llm: ChatHuggingFace,
    claim_text: str,
    source_texts: list[str],
    config: RunnableConfig | None = None,
) -> LSAVerdict:
    """Call the LLM synchronously and return an LSAVerdict.

    The response is stripped and normalised to one of the three accepted
    verdicts.  Unrecognised responses default to "unsupported" so that a
    malformed LLM reply does not crash the scoring pipeline — the claim
    will simply receive a low LSA score.

    Args:
        llm:          Synchronous ChatHuggingFace instance.
        claim_text:   The factual claim being assessed.
        source_texts: Source passage texts to assess against.

    Returns:
        LSAVerdict: "supported", "partially_supported", or "unsupported".
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(claim_text, source_texts)},
    ]
    response = llm.invoke(messages, config)
    verdict = strip_think(response.content).lower()
    if verdict in ("supported", "partially_supported", "unsupported"):
        return verdict  # type: ignore[return-value]
    return "unsupported"


async def confidence_scorer(state: MARAState, config: RunnableConfig) -> dict:
    """Score each extracted claim and return the list of ScoredClaims.

    Args:
        state:  MARAState with ``extracted_claims``, ``merkle_leaves``, and
                ``config`` populated.
        config: LangGraph RunnableConfig (unused directly).

    Returns:
        ``{"scored_claims": list[ScoredClaim]}``
    """
    research_config = state["config"]
    leaves = state["retrieved_leaves"]
    leaf_by_index = {leaf["index"]: leaf for leaf in leaves}
    claims = state["extracted_claims"]

    _log.info("Scoring %d claims against %d leaves", len(claims), len(leaves))

    llm = make_llm(research_config.lsa_model, research_config.hf_token, 32, research_config.hf_provider)

    def lsa_callable(claim_text: str, source_texts: list[str]) -> LSAVerdict:
        return _call_lsa(llm, claim_text, source_texts, config)

    scored = []
    for claim in claims:
        source_texts = [
            leaf_by_index[i]["text"]
            for i in claim["source_indices"]
            if i in leaf_by_index
        ]
        result = await asyncio.to_thread(
            score_claim,
            claim["text"],
            source_texts,
            claim["source_indices"],
            lsa_callable,
            research_config,
        )
        scored.append(result)

    _log.info("Scored %d claims", len(scored))
    return {"scored_claims": scored}
