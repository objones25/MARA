"""Deterministic serialisation and hashing for Merkle leaves.

The canonical_serialise function must produce byte-identical output across
Python versions, platforms, and locales. Any reader with the same (url, text,
retrieved_at) tuple can recompute the hash independently using this function.
"""

import hashlib
import json


def canonical_serialise(url: str, text: str, retrieved_at: str) -> bytes:
    """Produce a deterministic UTF-8 byte string from source chunk fields.

    Key guarantees:
    - sort_keys=True: key order is stable regardless of insertion order
    - separators=(',', ':'): no whitespace variation
    - ensure_ascii=True: byte-identical output regardless of locale
    """
    payload = {"retrieved_at": retrieved_at, "text": text, "url": url}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def hash_chunk(url: str, text: str, retrieved_at: str, algorithm: str) -> str:
    """Return the hex digest of a source chunk.

    Args:
        url:          Source URL.
        text:         Chunk text.
        retrieved_at: ISO-8601 timestamp string.
        algorithm:    hashlib algorithm name (ResearchConfig.hash_algorithm).

    Returns:
        Hex digest string.
    """
    data = canonical_serialise(url, text, retrieved_at)
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()
