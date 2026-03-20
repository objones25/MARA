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


def make_llm(
    model: str,
    hf_token: str,
    max_new_tokens: int,
    provider: str = "featherless-ai",
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 1.5,
) -> ChatHuggingFace:
    """Instantiate a ChatHuggingFace client via HuggingFace Inference Providers.

    Args:
        model:            HuggingFace model repo ID.
        hf_token:         HuggingFace Hub API token.
        max_new_tokens:   Token budget for the completion.
        provider:         HF inference provider name. Defaults to "featherless-ai".
                          Use "auto" to delegate to the HF conversational router, but
                          note the router only covers a limited model catalog.
        temperature:      Sampling temperature (Qwen3 default: 0.7).
        top_p:            Nucleus sampling probability (Qwen3 default: 0.8).
        top_k:            Top-k sampling (Qwen3 default: 20).
        presence_penalty: Penalises token repetition (Qwen3 default: 1.5).

    Returns:
        A ``ChatHuggingFace`` instance ready for ``.invoke()`` or ``.ainvoke()``.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        max_new_tokens=max_new_tokens,
        provider=provider,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        model_kwargs={"presence_penalty": presence_penalty},
    )
    return ChatHuggingFace(llm=endpoint)


def strip_think(text: str) -> str:
    """Strip Qwen3 <think>...</think> thinking tokens from LLM output."""
    return _THINK_RE.sub("", text).strip()
