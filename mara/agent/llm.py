"""Shared LLM factory and output utilities for MARA agent nodes.

All nodes that call HuggingFace Inference Providers import from here so that
provider configuration, model loading, and output post-processing live in one
place.  Changing the backend (provider, model family, retry policy) only
requires editing this module.

``ChatHuggingFace`` is re-exported so callers that need it for type
annotations (e.g. confidence_scorer) don't import langchain_huggingface
directly.
"""

import re

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

__all__ = ["ChatHuggingFace", "make_llm", "strip_think"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def make_llm(model: str, hf_token: str, max_new_tokens: int) -> ChatHuggingFace:
    """Instantiate a ChatHuggingFace client via HuggingFace Inference Providers.

    Args:
        model:          HuggingFace model repo ID.
        hf_token:       HuggingFace Hub API token.
        max_new_tokens: Token budget for the completion.

    Returns:
        A ``ChatHuggingFace`` instance ready for ``.invoke()`` or ``.ainvoke()``.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=max_new_tokens,
        provider="auto",
    )
    return ChatHuggingFace(llm=endpoint)


def strip_think(text: str) -> str:
    """Strip Qwen3 <think>...</think> thinking tokens from LLM output."""
    return _THINK_RE.sub("", text).strip()
