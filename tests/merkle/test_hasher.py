"""Tests for mara.merkle.hasher — canonical_serialise and hash_chunk.

The central invariant: given the same (url, text, retrieved_at) tuple,
canonical_serialise must produce byte-identical output on any Python version,
platform, or locale. hash_chunk must produce an identical hex digest.
"""

import hashlib
import json

import pytest

from mara.merkle.hasher import canonical_serialise, hash_chunk


# ---------------------------------------------------------------------------
# canonical_serialise
# ---------------------------------------------------------------------------

class TestCanonicalSerialise:
    def test_returns_bytes(self):
        result = canonical_serialise("https://a.com", "text", "2026-01-01T00:00:00Z")
        assert isinstance(result, bytes)

    def test_is_valid_utf8(self):
        result = canonical_serialise("https://a.com", "text", "2026-01-01T00:00:00Z")
        # Should decode without error
        decoded = result.decode("utf-8")
        assert isinstance(decoded, str)

    def test_is_valid_json(self):
        result = canonical_serialise("https://a.com", "text", "2026-01-01T00:00:00Z")
        parsed = json.loads(result)
        assert parsed["url"] == "https://a.com"
        assert parsed["text"] == "text"
        assert parsed["retrieved_at"] == "2026-01-01T00:00:00Z"

    def test_keys_are_sorted(self):
        """sort_keys=True: JSON keys must appear in alphabetical order."""
        result = canonical_serialise("https://a.com", "text", "2026-01-01T00:00:00Z")
        decoded = result.decode("utf-8")
        # Keys in alphabetical order: retrieved_at, text, url
        r_pos = decoded.index('"retrieved_at"')
        t_pos = decoded.index('"text"')
        u_pos = decoded.index('"url"')
        assert r_pos < t_pos < u_pos

    def test_no_whitespace_between_tokens(self):
        """separators=(',', ':'): no spaces after colons or commas."""
        result = canonical_serialise("https://a.com", "text", "2026-01-01T00:00:00Z")
        decoded = result.decode("utf-8")
        assert ": " not in decoded
        assert ", " not in decoded

    def test_exact_byte_output(self):
        """Verify exact byte output against an independently computed reference."""
        url = "https://example.com/article"
        text = "The quick brown fox"
        retrieved_at = "2026-03-19T12:00:00Z"
        result = canonical_serialise(url, text, retrieved_at)
        # Build the reference independently using the same contract
        reference = json.dumps(
            {"retrieved_at": retrieved_at, "text": text, "url": url},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        assert result == reference

    def test_determinism_same_inputs(self):
        """Same inputs always produce the same bytes."""
        args = ("https://a.com", "hello world", "2026-01-01T00:00:00Z")
        assert canonical_serialise(*args) == canonical_serialise(*args)

    def test_different_url_produces_different_bytes(self):
        args1 = ("https://a.com", "text", "2026-01-01T00:00:00Z")
        args2 = ("https://b.com", "text", "2026-01-01T00:00:00Z")
        assert canonical_serialise(*args1) != canonical_serialise(*args2)

    def test_different_text_produces_different_bytes(self):
        args1 = ("https://a.com", "hello", "2026-01-01T00:00:00Z")
        args2 = ("https://a.com", "world", "2026-01-01T00:00:00Z")
        assert canonical_serialise(*args1) != canonical_serialise(*args2)

    def test_different_timestamp_produces_different_bytes(self):
        args1 = ("https://a.com", "text", "2026-01-01T00:00:00Z")
        args2 = ("https://a.com", "text", "2026-01-02T00:00:00Z")
        assert canonical_serialise(*args1) != canonical_serialise(*args2)

    def test_unicode_text_is_ascii_escaped(self):
        """ensure_ascii=True: non-ASCII characters must be escaped, not raw bytes."""
        result = canonical_serialise("https://a.com", "caf\u00e9", "2026-01-01T00:00:00Z")
        decoded = result.decode("utf-8")
        # The é should be escaped as \u00e9, not appear as raw bytes
        assert "\\u00e9" in decoded
        assert "\u00e9" not in decoded

    def test_empty_text_is_valid(self):
        result = canonical_serialise("https://a.com", "", "2026-01-01T00:00:00Z")
        parsed = json.loads(result)
        assert parsed["text"] == ""

    def test_text_with_special_json_characters(self):
        """Double quotes and backslashes in text must be properly escaped."""
        result = canonical_serialise('https://a.com', 'say "hello" \\ there', "2026-01-01T00:00:00Z")
        # Should round-trip cleanly
        parsed = json.loads(result)
        assert parsed["text"] == 'say "hello" \\ there'


# ---------------------------------------------------------------------------
# hash_chunk
# ---------------------------------------------------------------------------

class TestHashChunk:
    def test_returns_hex_string(self):
        result = hash_chunk("https://a.com", "text", "2026-01-01T00:00:00Z", "sha256")
        assert isinstance(result, str)
        # SHA-256 hex digest is always 64 characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_matches_manual_sha256(self):
        """hash_chunk output must match an independently computed SHA-256."""
        url = "https://example.com"
        text = "some content"
        retrieved_at = "2026-01-01T00:00:00Z"
        data = canonical_serialise(url, text, retrieved_at)
        expected = hashlib.sha256(data).hexdigest()
        assert hash_chunk(url, text, retrieved_at, "sha256") == expected

    def test_determinism(self):
        args = ("https://a.com", "hello", "2026-01-01T00:00:00Z", "sha256")
        assert hash_chunk(*args) == hash_chunk(*args)

    def test_different_inputs_different_hashes(self):
        h1 = hash_chunk("https://a.com", "text", "2026-01-01T00:00:00Z", "sha256")
        h2 = hash_chunk("https://b.com", "text", "2026-01-01T00:00:00Z", "sha256")
        assert h1 != h2

    def test_sha256_algorithm(self):
        result = hash_chunk("https://a.com", "text", "2026-01-01T00:00:00Z", "sha256")
        assert len(result) == 64  # SHA-256 → 32 bytes → 64 hex chars

    def test_sha512_algorithm(self):
        result = hash_chunk("https://a.com", "text", "2026-01-01T00:00:00Z", "sha512")
        assert len(result) == 128  # SHA-512 → 64 bytes → 128 hex chars

    def test_same_content_different_algorithms_different_hashes(self):
        args = ("https://a.com", "text", "2026-01-01T00:00:00Z")
        h256 = hash_chunk(*args, "sha256")
        h512 = hash_chunk(*args, "sha512")
        assert h256 != h512

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError):
            hash_chunk("https://a.com", "text", "2026-01-01T00:00:00Z", "not_a_real_algo")

    def test_avalanche_effect(self):
        """One-character change in text must produce a completely different hash."""
        h1 = hash_chunk("https://a.com", "hello", "2026-01-01T00:00:00Z", "sha256")
        h2 = hash_chunk("https://a.com", "hellp", "2026-01-01T00:00:00Z", "sha256")
        # No common prefix expected with high probability
        assert h1 != h2
        # SHA-256 avalanche: expect < 2 identical leading characters by chance
        assert h1[:8] != h2[:8]
