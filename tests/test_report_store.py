"""Tests for mara.report_store.

Tests cover:
  - report_to_dict: produces a JSON-serializable dict with expected keys
  - report_from_dict: reconstructs ScoredClaim dataclasses correctly
  - round-trip: report_to_dict → report_from_dict reproduces the original
  - _report_filename: deterministic, human-readable, handles special chars
  - save_report: creates the file, content is valid JSON, returns correct path
  - load_report: reconstructs report equal to the original
  - save + load round-trip: equality of all fields
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mara.agent.state import CertifiedReport
from mara.confidence.scorer import ScoredClaim
from mara.report_store import (
    _report_filename,
    load_report,
    report_from_dict,
    report_to_dict,
    save_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf(index: int = 0, url: str = "https://example.com") -> dict:
    return {
        "url": url,
        "text": f"chunk text {index}",
        "retrieved_at": "2026-03-19T10:00:00+00:00",
        "hash": "a" * 64,
        "index": index,
        "sub_query": "test query",
        "contextualized_text": f"chunk text {index}",
    }


def _make_claim(text: str = "Test claim.", confidence: float = 0.85) -> ScoredClaim:
    return ScoredClaim(
        text=text,
        source_indices=[0, 1],
        confidence=confidence,
        corroborating=2,
        n_leaves=10,
        similarities=[0.85, 0.80],
    )


def _make_report(**kwargs) -> CertifiedReport:
    defaults = dict(
        query="What is X?",
        report_text="The research says X.",
        merkle_root="a" * 64,
        leaves=[_make_leaf(0), _make_leaf(1)],
        scored_claims=[_make_claim()],
        hash_algorithm="sha256",
        generated_at=datetime(2026, 3, 19, 15, 8, 32, tzinfo=timezone.utc).isoformat(),
    )
    defaults.update(kwargs)
    return CertifiedReport(**defaults)


# ---------------------------------------------------------------------------
# report_to_dict
# ---------------------------------------------------------------------------


class TestReportToDict:
    def test_returns_dict(self):
        assert isinstance(report_to_dict(_make_report()), dict)

    def test_has_query_key(self):
        assert report_to_dict(_make_report(query="Q?"))["query"] == "Q?"

    def test_has_report_text_key(self):
        d = report_to_dict(_make_report(report_text="Text."))
        assert d["report_text"] == "Text."

    def test_has_merkle_root_key(self):
        d = report_to_dict(_make_report(merkle_root="b" * 64))
        assert d["merkle_root"] == "b" * 64

    def test_has_generated_at_key(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
        d = report_to_dict(_make_report(generated_at=ts))
        assert d["generated_at"] == ts

    def test_leaves_is_list(self):
        assert isinstance(report_to_dict(_make_report())["leaves"], list)

    def test_leaves_count(self):
        report = _make_report(leaves=[_make_leaf(0), _make_leaf(1), _make_leaf(2)])
        assert len(report_to_dict(report)["leaves"]) == 3

    def test_leaf_fields_present(self):
        d = report_to_dict(_make_report())
        leaf = d["leaves"][0]
        for key in ("url", "text", "retrieved_at", "hash", "index", "sub_query", "contextualized_text"):
            assert key in leaf

    def test_scored_claims_is_list(self):
        assert isinstance(report_to_dict(_make_report())["scored_claims"], list)

    def test_scored_claim_fields_present(self):
        d = report_to_dict(_make_report())
        claim = d["scored_claims"][0]
        for key in ("text", "source_indices", "confidence", "corroborating", "n_leaves", "similarities"):
            assert key in claim

    def test_is_json_serializable(self):
        d = report_to_dict(_make_report())
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_empty_leaves_serialized(self):
        d = report_to_dict(_make_report(leaves=[]))
        assert d["leaves"] == []

    def test_empty_claims_serialized(self):
        d = report_to_dict(_make_report(scored_claims=[]))
        assert d["scored_claims"] == []


# ---------------------------------------------------------------------------
# report_from_dict
# ---------------------------------------------------------------------------


class TestReportFromDict:
    def test_returns_certified_report(self):
        d = report_to_dict(_make_report())
        assert isinstance(report_from_dict(d), CertifiedReport)

    def test_query_preserved(self):
        d = report_to_dict(_make_report(query="My question?"))
        assert report_from_dict(d).query == "My question?"

    def test_report_text_preserved(self):
        d = report_to_dict(_make_report(report_text="My report."))
        assert report_from_dict(d).report_text == "My report."

    def test_merkle_root_preserved(self):
        root = "c" * 64
        d = report_to_dict(_make_report(merkle_root=root))
        assert report_from_dict(d).merkle_root == root

    def test_generated_at_preserved(self):
        ts = datetime(2026, 6, 1, tzinfo=timezone.utc).isoformat()
        d = report_to_dict(_make_report(generated_at=ts))
        assert report_from_dict(d).generated_at == ts

    def test_scored_claims_are_scored_claim_instances(self):
        d = report_to_dict(_make_report(scored_claims=[_make_claim()]))
        report = report_from_dict(d)
        assert isinstance(report.scored_claims[0], ScoredClaim)

    def test_claim_confidence_preserved(self):
        d = report_to_dict(_make_report(scored_claims=[_make_claim(confidence=0.72)]))
        assert report_from_dict(d).scored_claims[0].confidence == pytest.approx(0.72)

    def test_claim_text_preserved(self):
        d = report_to_dict(_make_report(scored_claims=[_make_claim(text="Specific claim.")]))
        assert report_from_dict(d).scored_claims[0].text == "Specific claim."

    def test_leaf_count_preserved(self):
        leaves = [_make_leaf(i) for i in range(5)]
        d = report_to_dict(_make_report(leaves=leaves))
        assert len(report_from_dict(d).leaves) == 5

    def test_leaf_url_preserved(self):
        d = report_to_dict(_make_report(leaves=[_make_leaf(url="https://test.org")]))
        assert report_from_dict(d).leaves[0]["url"] == "https://test.org"

    def test_similarities_preserved(self):
        claim = _make_claim()
        d = report_to_dict(_make_report(scored_claims=[claim]))
        reconstructed = report_from_dict(d).scored_claims[0]
        assert reconstructed.similarities == pytest.approx([0.85, 0.80])


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_query_survives_round_trip(self):
        report = _make_report(query="Does X cause Y?")
        assert report_from_dict(report_to_dict(report)).query == report.query

    def test_report_text_survives_round_trip(self):
        report = _make_report(report_text="Evidence suggests Y.")
        assert report_from_dict(report_to_dict(report)).report_text == report.report_text

    def test_merkle_root_survives_round_trip(self):
        report = _make_report()
        assert report_from_dict(report_to_dict(report)).merkle_root == report.merkle_root

    def test_claim_count_survives_round_trip(self):
        report = _make_report(scored_claims=[_make_claim(), _make_claim("Second.")])
        assert len(report_from_dict(report_to_dict(report)).scored_claims) == 2

    def test_leaf_count_survives_round_trip(self):
        report = _make_report(leaves=[_make_leaf(i) for i in range(4)])
        assert len(report_from_dict(report_to_dict(report)).leaves) == 4


# ---------------------------------------------------------------------------
# _report_filename
# ---------------------------------------------------------------------------


class TestReportFilename:
    def test_returns_string(self):
        assert isinstance(_report_filename(_make_report()), str)

    def test_ends_with_json(self):
        assert _report_filename(_make_report()).endswith(".json")

    def test_contains_date(self):
        assert "2026-03-19" in _report_filename(_make_report())

    def test_contains_query_slug(self):
        report = _make_report(query="What is X?")
        assert "what_is_x" in _report_filename(report)

    def test_contains_short_root(self):
        report = _make_report(merkle_root="deadbeef" + "0" * 56)
        assert "deadbeef" in _report_filename(report)

    def test_special_chars_in_query_replaced(self):
        report = _make_report(query="What is X? And Y!")
        filename = _report_filename(report)
        assert "?" not in filename
        assert "!" not in filename

    def test_query_slug_truncated_to_40_chars(self):
        long_query = "a" * 100
        report = _make_report(query=long_query)
        slug_part = _report_filename(report).split("_", 2)[2]
        assert len(slug_part) <= 60  # 40 char slug + underscore + root8 + .json

    def test_no_root_fallback(self):
        report = _make_report(merkle_root="")
        assert "no_root" in _report_filename(report)

    def test_deterministic(self):
        report = _make_report()
        assert _report_filename(report) == _report_filename(report)


# ---------------------------------------------------------------------------
# save_report / load_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_returns_path(self, tmp_path):
        result = save_report(_make_report(), tmp_path)
        assert isinstance(result, Path)

    def test_file_exists_after_save(self, tmp_path):
        path = save_report(_make_report(), tmp_path)
        assert path.exists()

    def test_file_is_valid_json(self, tmp_path):
        path = save_report(_make_report(), tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_saved_file_in_specified_dir(self, tmp_path):
        path = save_report(_make_report(), tmp_path)
        assert path.parent == tmp_path

    def test_creates_report_dir_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        save_report(_make_report(), nested)
        assert nested.exists()

    def test_saved_content_has_query(self, tmp_path):
        path = save_report(_make_report(query="The question."), tmp_path)
        data = json.loads(path.read_text())
        assert data["query"] == "The question."

    def test_saved_content_has_merkle_root(self, tmp_path):
        path = save_report(_make_report(merkle_root="f" * 64), tmp_path)
        data = json.loads(path.read_text())
        assert data["merkle_root"] == "f" * 64

    def test_filename_matches_convention(self, tmp_path):
        report = _make_report()
        path = save_report(report, tmp_path)
        assert path.name == _report_filename(report)


class TestLoadReport:
    def test_returns_certified_report(self, tmp_path):
        path = save_report(_make_report(), tmp_path)
        assert isinstance(load_report(path), CertifiedReport)

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_report(tmp_path / "nonexistent.json")

    def test_query_survives_save_load(self, tmp_path):
        report = _make_report(query="Persistent question?")
        path = save_report(report, tmp_path)
        assert load_report(path).query == "Persistent question?"

    def test_report_text_survives_save_load(self, tmp_path):
        report = _make_report(report_text="Persistent text.")
        path = save_report(report, tmp_path)
        assert load_report(path).report_text == "Persistent text."

    def test_merkle_root_survives_save_load(self, tmp_path):
        report = _make_report(merkle_root="e" * 64)
        path = save_report(report, tmp_path)
        assert load_report(path).merkle_root == "e" * 64

    def test_claim_count_survives_save_load(self, tmp_path):
        report = _make_report(scored_claims=[_make_claim(), _make_claim("B.")])
        path = save_report(report, tmp_path)
        assert len(load_report(path).scored_claims) == 2

    def test_leaf_count_survives_save_load(self, tmp_path):
        report = _make_report(leaves=[_make_leaf(i) for i in range(7)])
        path = save_report(report, tmp_path)
        assert len(load_report(path).leaves) == 7

    def test_claim_confidence_survives_save_load(self, tmp_path):
        report = _make_report(scored_claims=[_make_claim(confidence=0.91)])
        path = save_report(report, tmp_path)
        assert load_report(path).scored_claims[0].confidence == pytest.approx(0.91)

    def test_scored_claims_are_instances_after_load(self, tmp_path):
        path = save_report(_make_report(), tmp_path)
        report = load_report(path)
        assert isinstance(report.scored_claims[0], ScoredClaim)

    def test_similarities_survive_save_load(self, tmp_path):
        claim = _make_claim()
        path = save_report(_make_report(scored_claims=[claim]), tmp_path)
        loaded = load_report(path).scored_claims[0]
        assert loaded.similarities == pytest.approx([0.85, 0.80])

    def test_generated_at_survives_save_load(self, tmp_path):
        ts = datetime(2026, 3, 19, 15, 8, 32, tzinfo=timezone.utc).isoformat()
        report = _make_report(generated_at=ts)
        path = save_report(report, tmp_path)
        assert load_report(path).generated_at == ts
