"""Tests for mara.verifier.

Tests cover:
  - verify_report: all-pass path, single leaf mismatch, root mismatch,
    empty report, multiple leaves, hash_algorithm forwarded to hasher
  - VerificationResult.passed property
  - VerificationResult.failed_leaves property
  - LeafVerification fields
"""

import pytest

from mara.agent.state import CertifiedReport
from mara.confidence.scorer import ScoredClaim
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree
from mara.verifier import LeafVerification, VerificationResult, verify_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALGO = "sha256"
_URL = "https://example.com/page"
_TEXT = "Some source text."
_RETRIEVED_AT = "2026-03-19T10:00:00+00:00"


def _good_hash(url=_URL, text=_TEXT, retrieved_at=_RETRIEVED_AT):
    return hash_chunk(url, text, retrieved_at, _ALGO)


def _make_leaf(
    index: int = 0,
    url: str = _URL,
    text: str = _TEXT,
    retrieved_at: str = _RETRIEVED_AT,
    corrupt_hash: bool = False,
) -> dict:
    h = _good_hash(url, text, retrieved_at)
    if corrupt_hash:
        h = "0" * len(h)
    return {
        "url": url,
        "text": text,
        "retrieved_at": retrieved_at,
        "hash": h,
        "index": index,
        "sub_query": "test query",
        "contextualized_text": text,
    }


def _make_report(leaves: list[dict], corrupt_root: bool = False) -> CertifiedReport:
    hashes = [leaf["hash"] for leaf in leaves]
    if leaves:
        root = build_merkle_tree(hashes, _ALGO).root
    else:
        root = ""
    if corrupt_root:
        root = "0" * len(root) if root else "0" * 64
    return CertifiedReport(
        query="What is X?",
        report_text="The answer is X.",
        merkle_root=root,
        leaves=leaves,
        scored_claims=[],
        hash_algorithm=_ALGO,
    )


# ---------------------------------------------------------------------------
# verify_report — passing cases
# ---------------------------------------------------------------------------


class TestVerifyReportPass:
    def test_returns_verification_result(self):
        result = verify_report(_make_report([_make_leaf()]))
        assert isinstance(result, VerificationResult)

    def test_single_good_leaf_passes(self):
        result = verify_report(_make_report([_make_leaf()]))
        assert result.passed is True

    def test_multiple_good_leaves_pass(self):
        leaves = [_make_leaf(i, url=f"https://example.com/{i}") for i in range(5)]
        result = verify_report(_make_report(leaves))
        assert result.passed is True

    def test_root_match_is_true_on_pass(self):
        result = verify_report(_make_report([_make_leaf()]))
        assert result.root_match is True

    def test_all_leaf_results_match_on_pass(self):
        leaves = [_make_leaf(i, url=f"https://example.com/{i}") for i in range(3)]
        result = verify_report(_make_report(leaves))
        assert all(lr.match for lr in result.leaf_results)

    def test_leaf_result_count_equals_leaf_count(self):
        leaves = [_make_leaf(i, url=f"https://example.com/{i}") for i in range(4)]
        result = verify_report(_make_report(leaves))
        assert len(result.leaf_results) == 4

    def test_leaf_result_index_preserved(self):
        leaves = [_make_leaf(7)]
        result = verify_report(_make_report(leaves))
        assert result.leaf_results[0].index == 7

    def test_leaf_result_url_preserved(self):
        leaves = [_make_leaf(url="https://specific.org/doc")]
        result = verify_report(_make_report(leaves))
        assert result.leaf_results[0].url == "https://specific.org/doc"

    def test_expected_and_computed_hashes_equal_on_pass(self):
        leaf = _make_leaf()
        result = verify_report(_make_report([leaf]))
        lr = result.leaf_results[0]
        assert lr.expected_hash == lr.computed_hash

    def test_report_query_in_result(self):
        report = _make_report([_make_leaf()])
        report = CertifiedReport(
            query="Unique query?",
            report_text=report.report_text,
            merkle_root=report.merkle_root,
            leaves=report.leaves,
            scored_claims=[],
            hash_algorithm=_ALGO,
        )
        result = verify_report(report)
        assert result.report_query == "Unique query?"

    def test_merkle_root_expected_in_result(self):
        report = _make_report([_make_leaf()])
        result = verify_report(report)
        assert result.merkle_root_expected == report.merkle_root

    def test_merkle_root_computed_equals_expected_on_pass(self):
        result = verify_report(_make_report([_make_leaf()]))
        assert result.merkle_root_computed == result.merkle_root_expected


# ---------------------------------------------------------------------------
# verify_report — failing cases
# ---------------------------------------------------------------------------


class TestVerifyReportFail:
    def test_corrupted_leaf_hash_fails(self):
        leaves = [_make_leaf(corrupt_hash=True)]
        result = verify_report(_make_report(leaves))
        assert result.passed is False

    def test_corrupted_leaf_detected_in_leaf_results(self):
        leaves = [_make_leaf(corrupt_hash=True)]
        result = verify_report(_make_report(leaves))
        assert result.leaf_results[0].match is False

    def test_corrupted_leaf_expected_differs_from_computed(self):
        leaf = _make_leaf(corrupt_hash=True)
        result = verify_report(_make_report([leaf]))
        lr = result.leaf_results[0]
        assert lr.expected_hash != lr.computed_hash

    def test_one_corrupted_leaf_among_many(self):
        leaves = [
            _make_leaf(0, url="https://example.com/0"),
            _make_leaf(1, url="https://example.com/1", corrupt_hash=True),
            _make_leaf(2, url="https://example.com/2"),
        ]
        result = verify_report(_make_report(leaves))
        assert result.passed is False
        assert result.leaf_results[0].match is True
        assert result.leaf_results[1].match is False
        assert result.leaf_results[2].match is True

    def test_corrupted_root_fails(self):
        result = verify_report(_make_report([_make_leaf()], corrupt_root=True))
        assert result.passed is False

    def test_corrupted_root_sets_root_match_false(self):
        result = verify_report(_make_report([_make_leaf()], corrupt_root=True))
        assert result.root_match is False

    def test_corrupted_root_leaf_hashes_still_match(self):
        # A tampered root does not affect individual leaf verification.
        result = verify_report(_make_report([_make_leaf()], corrupt_root=True))
        assert result.leaf_results[0].match is True

    def test_tampered_text_fails(self):
        leaf = _make_leaf()
        leaf["text"] = "Tampered content."
        # hash still reflects original text, so recomputed hash won't match
        result = verify_report(_make_report([leaf]))
        assert result.passed is False

    def test_tampered_url_fails(self):
        leaf = _make_leaf()
        leaf["url"] = "https://tampered.com"
        result = verify_report(_make_report([leaf]))
        assert result.passed is False


# ---------------------------------------------------------------------------
# Empty report
# ---------------------------------------------------------------------------


class TestVerifyReportEmpty:
    def test_empty_leaves_passes(self):
        result = verify_report(_make_report([]))
        assert result.passed is True

    def test_empty_leaves_root_is_empty_string(self):
        result = verify_report(_make_report([]))
        assert result.merkle_root_computed == ""
        assert result.merkle_root_expected == ""

    def test_empty_leaf_results(self):
        result = verify_report(_make_report([]))
        assert result.leaf_results == []


# ---------------------------------------------------------------------------
# VerificationResult properties
# ---------------------------------------------------------------------------


class TestVerificationResultProperties:
    def test_failed_leaves_empty_on_pass(self):
        result = verify_report(_make_report([_make_leaf()]))
        assert result.failed_leaves == []

    def test_failed_leaves_contains_corrupted(self):
        leaves = [
            _make_leaf(0, url="https://example.com/0"),
            _make_leaf(1, url="https://example.com/1", corrupt_hash=True),
        ]
        result = verify_report(_make_report(leaves))
        assert len(result.failed_leaves) == 1
        assert result.failed_leaves[0].index == 1

    def test_passed_false_when_root_mismatch_only(self):
        result = verify_report(_make_report([_make_leaf()], corrupt_root=True))
        assert result.passed is False

    def test_passed_false_when_leaf_mismatch_only(self):
        result = verify_report(_make_report([_make_leaf(corrupt_hash=True)]))
        # root will also differ because hash differs, but the point is passed is False
        assert result.passed is False


# ---------------------------------------------------------------------------
# Hash algorithm forwarding
# ---------------------------------------------------------------------------


class TestHashAlgorithmForwarding:
    def test_sha256_is_default(self):
        result = verify_report(_make_report([_make_leaf()]))
        assert result.passed is True

    def test_computed_hash_uses_stored_algorithm(self):
        # Build a report with sha256 hashes — verify_report should use sha256 from report
        leaf = _make_leaf()
        report = _make_report([leaf])
        result = verify_report(report)
        expected = hash_chunk(leaf["url"], leaf["text"], leaf["retrieved_at"], "sha256")
        assert result.leaf_results[0].computed_hash == expected
