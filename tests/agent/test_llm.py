"""Tests for mara.agent.llm — shared LLM factory and strip_think utility.

All external library calls are mocked — no real API calls are made.
"""

import pytest

from mara.agent.llm import make_llm, strip_think


# ---------------------------------------------------------------------------
# make_llm
# ---------------------------------------------------------------------------


class TestMakeLlm:
    def test_returns_chat_hugging_face_instance(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mock_chat_cls = mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("Qwen/Qwen3-30B-A3B-Instruct-2507", "hf-token", 1024)
        mock_endpoint_cls.assert_called_once_with(
            repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
            task="text-generation",
            huggingfacehub_api_token="hf-token",
            max_new_tokens=1024,
            provider="featherless-ai",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            model_kwargs={"presence_penalty": 1.5},
        )
        mock_chat_cls.assert_called_once_with(llm=mock_endpoint_cls.return_value)

    def test_passes_model(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("some/model", "k", 512)
        assert mock_endpoint_cls.call_args.kwargs["repo_id"] == "some/model"

    def test_passes_token(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("m", "secret-token", 512)
        assert mock_endpoint_cls.call_args.kwargs["huggingfacehub_api_token"] == "secret-token"

    def test_passes_max_new_tokens(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("m", "k", 4096)
        assert mock_endpoint_cls.call_args.kwargs["max_new_tokens"] == 4096

    def test_passes_provider_featherless_ai_by_default(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("m", "k", 512)
        assert mock_endpoint_cls.call_args.kwargs["provider"] == "featherless-ai"

    def test_passes_custom_provider(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("m", "k", 512, "groq")
        assert mock_endpoint_cls.call_args.kwargs["provider"] == "groq"

    def test_task_is_text_generation(self, mocker):
        mock_endpoint_cls = mocker.patch("mara.agent.llm.HuggingFaceEndpoint")
        mocker.patch("mara.agent.llm.ChatHuggingFace")
        make_llm("m", "k", 512)
        assert mock_endpoint_cls.call_args.kwargs["task"] == "text-generation"


# ---------------------------------------------------------------------------
# strip_think
# ---------------------------------------------------------------------------


class TestStripThink:
    def test_removes_think_block(self):
        assert strip_think("<think>reasoning here</think>answer") == "answer"

    def test_multiline_think_block(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nfinal answer"
        assert strip_think(text) == "final answer"

    def test_no_think_block_unchanged(self):
        assert strip_think("plain output") == "plain output"

    def test_strips_surrounding_whitespace(self):
        assert strip_think("  result  ") == "result"

    def test_strips_think_and_whitespace(self):
        assert strip_think("<think>...</think>  answer  ") == "answer"

    def test_empty_string(self):
        assert strip_think("") == ""
