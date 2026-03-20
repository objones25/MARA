"""Tests for mara.agent.nodes.claim_extractor.

All LLM calls are mocked. Tests cover:
  - _format_leaves: pure function, index/url/text extraction
  - _parse_claims: pure function, all branches (clean JSON, fences, empty text
    filtering, source_indices coercion, non-list error, invalid JSON error)
  - claim_extractor: async node — empty leaves path, populated path, state
    reading, config forwarding, return shape
"""

import json
import pytest

from mara.agent.nodes.claim_extractor import (
    _format_leaves,
    _parse_claims,
    claim_extractor,
)
from mara.agent.state import Claim, MARAState, MerkleLeaf
from mara.config import ResearchConfig
from mara.merkle.hasher import hash_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(index: int, url: str = "https://example.com", text: str = "text") -> MerkleLeaf:
    digest = hash_chunk(url=url, text=text, retrieved_at="2026-03-19T10:00:00Z", algorithm="sha256")
    return MerkleLeaf(
        url=url,
        text=text,
        retrieved_at="2026-03-19T10:00:00Z",
        hash=digest,
        index=index,
        sub_query="test query",
        contextualized_text=text,
    )


def _make_state(leaves: list[MerkleLeaf] | None = None) -> MARAState:
    return MARAState(
        query="What are the economic effects of automation?",
        run_date="2026-03-20",
        config=ResearchConfig(
            hf_token="test-token",
            brave_api_key="x",
            firecrawl_api_key="x",
        ),
        sub_queries=[],
        search_results=[],
        raw_chunks=[],
        merkle_leaves=[],
        merkle_tree=None,
        retrieved_leaves=leaves or [],
        extracted_claims=[],
        scored_claims=[],
        human_approved_claims=[],
        report_draft="",
        certified_report=None,
        messages=[],
        loop_count=0,
    )


def _claims_json(claims: list[dict] | None = None) -> str:
    if claims is None:
        claims = [
            {"text": "GDP grew by 3% in 2023", "source_indices": [0]},
            {"text": "Unemployment fell to 4.2%", "source_indices": [0, 1]},
        ]
    return json.dumps(claims)


def _mock_llm_response(mocker, content: str):
    msg = mocker.MagicMock()
    msg.content = content
    return msg


# ---------------------------------------------------------------------------
# _format_leaves
# ---------------------------------------------------------------------------


class TestFormatLeaves:
    def test_empty_leaves_returns_empty_list(self):
        assert _format_leaves([]) == []

    def test_extracts_index_url_text(self):
        leaf = _make_leaf(index=3, url="https://a.com", text="some text")
        result = _format_leaves([leaf])
        assert result == [(3, "https://a.com", "some text")]

    def test_preserves_order(self):
        leaves = [_make_leaf(index=i, url=f"https://a.com/{i}", text=f"t{i}") for i in range(3)]
        result = _format_leaves(leaves)
        assert [r[0] for r in result] == [0, 1, 2]

    def test_uses_leaf_index_not_list_position(self):
        # leaf has index=7 but is at list position 0
        leaf = _make_leaf(index=7, url="https://a.com", text="text")
        result = _format_leaves([leaf])
        assert result[0][0] == 7

    def test_multiple_leaves(self):
        leaves = [
            _make_leaf(index=0, url="https://a.com", text="alpha"),
            _make_leaf(index=1, url="https://b.com", text="beta"),
        ]
        result = _format_leaves(leaves)
        assert len(result) == 2
        assert result[1] == (1, "https://b.com", "beta")


# ---------------------------------------------------------------------------
# _parse_claims
# ---------------------------------------------------------------------------


class TestParseClaims:
    def test_clean_json_array(self):
        raw = '[{"text": "GDP rose by 3%", "source_indices": [0]}]'
        result = _parse_claims(raw)
        assert len(result) == 1
        assert result[0]["text"] == "GDP rose by 3%"
        assert result[0]["source_indices"] == [0]

    def test_multiple_claims(self):
        raw = _claims_json()
        result = _parse_claims(raw)
        assert len(result) == 2

    def test_empty_array_returns_empty_list(self):
        assert _parse_claims("[]") == []

    def test_strips_json_fences(self):
        raw = '```json\n[{"text": "fenced claim", "source_indices": [1]}]\n```'
        result = _parse_claims(raw)
        assert result[0]["text"] == "fenced claim"

    def test_strips_plain_fences(self):
        raw = '```\n[{"text": "plain claim", "source_indices": []}]\n```'
        result = _parse_claims(raw)
        assert result[0]["text"] == "plain claim"

    def test_drops_items_with_empty_text(self):
        raw = '[{"text": "", "source_indices": [0]}, {"text": "real claim", "source_indices": [0]}]'
        result = _parse_claims(raw)
        assert len(result) == 1
        assert result[0]["text"] == "real claim"

    def test_drops_items_with_whitespace_only_text(self):
        raw = '[{"text": "   ", "source_indices": [0]}, {"text": "valid", "source_indices": []}]'
        result = _parse_claims(raw)
        assert len(result) == 1

    def test_source_indices_coerced_to_int(self):
        raw = '[{"text": "claim", "source_indices": [0.0, 2.0]}]'
        result = _parse_claims(raw)
        assert result[0]["source_indices"] == [0, 2]

    def test_missing_source_indices_defaults_to_empty(self):
        raw = '[{"text": "orphan claim"}]'
        result = _parse_claims(raw)
        assert result[0]["source_indices"] == []

    def test_null_source_indices_defaults_to_empty(self):
        raw = '[{"text": "claim", "source_indices": null}]'
        result = _parse_claims(raw)
        assert result[0]["source_indices"] == []

    def test_text_coerced_to_str(self):
        raw = '[{"text": 42, "source_indices": [0]}]'
        result = _parse_claims(raw)
        assert result[0]["text"] == "42"

    def test_multiple_source_indices(self):
        raw = '[{"text": "multi-source claim", "source_indices": [0, 3, 7]}]'
        result = _parse_claims(raw)
        assert result[0]["source_indices"] == [0, 3, 7]

    def test_whitespace_around_json(self):
        raw = '  \n[{"text": "spaced", "source_indices": [0]}]\n  '
        result = _parse_claims(raw)
        assert result[0]["text"] == "spaced"

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_claims("not json at all")

    def test_raises_on_non_list_json(self):
        with pytest.raises(ValueError, match="Expected a JSON array"):
            _parse_claims('{"text": "oops", "source_indices": [0]}')

    def test_preserves_claim_order(self):
        claims = [{"text": f"claim {i}", "source_indices": [i]} for i in range(4)]
        result = _parse_claims(json.dumps(claims))
        assert [c["text"] for c in result] == [f"claim {i}" for i in range(4)]


# ---------------------------------------------------------------------------
# claim_extractor — async node
# ---------------------------------------------------------------------------


class TestClaimExtractorNode:
    def _mock_llm(self, mocker, content: str):
        mock_llm = mocker.AsyncMock()
        mock_llm.ainvoke = mocker.AsyncMock(
            return_value=_mock_llm_response(mocker, content)
        )
        mocker.patch(
            "mara.agent.nodes.claim_extractor.make_llm",
            return_value=mock_llm,
        )
        return mock_llm

    async def test_empty_leaves_returns_empty_claims(self, mocker):
        result = await claim_extractor(_make_state([]), config={})
        assert result == {"extracted_claims": []}

    async def test_empty_leaves_does_not_call_llm(self, mocker):
        mock_cls = mocker.patch("mara.agent.nodes.claim_extractor.make_llm")
        await claim_extractor(_make_state([]), config={})
        mock_cls.assert_not_called()

    async def test_returns_extracted_claims_key(self, mocker):
        leaf = _make_leaf(0)
        self._mock_llm(mocker, _claims_json())
        result = await claim_extractor(_make_state([leaf]), config={})
        assert "extracted_claims" in result

    async def test_claims_are_list_of_claim(self, mocker):
        leaf = _make_leaf(0)
        self._mock_llm(mocker, _claims_json())
        result = await claim_extractor(_make_state([leaf]), config={})
        for claim in result["extracted_claims"]:
            assert "text" in claim
            assert "source_indices" in claim

    async def test_claim_count_matches_llm_response(self, mocker):
        leaf = _make_leaf(0)
        payload = json.dumps([{"text": f"claim {i}", "source_indices": [0]} for i in range(5)])
        self._mock_llm(mocker, payload)
        result = await claim_extractor(_make_state([leaf]), config={})
        assert len(result["extracted_claims"]) == 5

    async def test_llm_invoked_with_system_and_user_messages(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = self._mock_llm(mocker, _claims_json())
        await claim_extractor(_make_state([leaf]), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    async def test_system_message_contains_system_prompt(self, mocker):
        from mara.prompts.claim_extractor import build_system_prompt
        leaf = _make_leaf(0)
        mock_llm = self._mock_llm(mocker, _claims_json())
        await claim_extractor(_make_state([leaf]), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert messages[0]["content"] == build_system_prompt("2026-03-20")

    async def test_user_message_contains_leaf_text(self, mocker):
        leaf = _make_leaf(0, text="unique leaf content xyz")
        mock_llm = self._mock_llm(mocker, _claims_json())
        await claim_extractor(_make_state([leaf]), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert "unique leaf content xyz" in messages[1]["content"]

    async def test_user_message_contains_leaf_url(self, mocker):
        leaf = _make_leaf(0, url="https://distinctive-url.com")
        mock_llm = self._mock_llm(mocker, _claims_json())
        await claim_extractor(_make_state([leaf]), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert "https://distinctive-url.com" in messages[1]["content"]

    async def test_user_message_contains_leaf_index(self, mocker):
        leaf = _make_leaf(index=5)
        mock_llm = self._mock_llm(mocker, _claims_json())
        await claim_extractor(_make_state([leaf]), config={})
        messages = mock_llm.ainvoke.call_args.args[0]
        assert "[5]" in messages[1]["content"]

    async def test_config_forwarded_to_ainvoke(self, mocker):
        leaf = _make_leaf(0)
        mock_llm = self._mock_llm(mocker, _claims_json())
        sentinel_config = {"run_name": "test", "tags": ["x"]}
        await claim_extractor(_make_state([leaf]), config=sentinel_config)
        _, forwarded_config = mock_llm.ainvoke.call_args.args
        assert forwarded_config is sentinel_config

    async def test_uses_config_model_and_api_key(self, mocker):
        mock_make_llm = mocker.patch("mara.agent.nodes.claim_extractor.make_llm")
        mock_llm = mocker.AsyncMock()
        mock_llm.ainvoke = mocker.AsyncMock(
            return_value=_mock_llm_response(mocker, _claims_json())
        )
        mock_make_llm.return_value = mock_llm

        cfg = ResearchConfig(
            hf_token="my-token",
            brave_api_key="x",
            firecrawl_api_key="x",
            model="Qwen/Qwen3-32B",
        )
        state = _make_state([_make_leaf(0)])
        state["config"] = cfg
        await claim_extractor(state, config={})
        mock_make_llm.assert_called_once_with("Qwen/Qwen3-32B", "my-token", 4096, "featherless-ai")

    async def test_handles_fenced_llm_response(self, mocker):
        leaf = _make_leaf(0)
        fenced = "```json\n" + _claims_json() + "\n```"
        self._mock_llm(mocker, fenced)
        result = await claim_extractor(_make_state([leaf]), config={})
        assert len(result["extracted_claims"]) == 2

    async def test_handles_empty_claim_list_from_llm(self, mocker):
        leaf = _make_leaf(0)
        self._mock_llm(mocker, "[]")
        result = await claim_extractor(_make_state([leaf]), config={})
        assert result["extracted_claims"] == []

    async def test_multiple_leaves_all_passed_to_llm(self, mocker):
        leaves = [_make_leaf(i, url=f"https://a.com/{i}", text=f"text {i}") for i in range(3)]
        mock_llm = self._mock_llm(mocker, _claims_json())
        await claim_extractor(_make_state(leaves), config={})
        user_msg = mock_llm.ainvoke.call_args.args[0][1]["content"]
        for i in range(3):
            assert f"text {i}" in user_msg
