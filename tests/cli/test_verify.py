"""Tests for the ``mara verify`` CLI command.

All verification logic is delegated to mara.verifier.  These tests cover:
  - Exit code 0 on a passing report
  - Exit code 1 on a failing report
  - Exit code 1 when the file does not exist
  - Output contains key verification phrases
  - _display_verification: leaf status, root status, pass/fail line
"""

import pytest
from typer.testing import CliRunner

from mara.agent.state import CertifiedReport
from mara.cli.run import _display_verification, app
from mara.verifier import LeafVerification, VerificationResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_passing_result(n_leaves: int = 2) -> VerificationResult:
    leaves = [
        LeafVerification(
            index=i,
            url=f"https://example.com/{i}",
            expected_hash="a" * 64,
            computed_hash="a" * 64,
            match=True,
        )
        for i in range(n_leaves)
    ]
    root = "b" * 64
    return VerificationResult(
        report_query="What is X?",
        merkle_root_expected=root,
        merkle_root_computed=root,
        root_match=True,
        leaf_results=leaves,
    )


def _make_failing_result() -> VerificationResult:
    bad_leaf = LeafVerification(
        index=0,
        url="https://example.com/0",
        expected_hash="a" * 64,
        computed_hash="b" * 64,
        match=False,
    )
    return VerificationResult(
        report_query="What is X?",
        merkle_root_expected="c" * 64,
        merkle_root_computed="d" * 64,
        root_match=False,
        leaf_results=[bad_leaf],
    )


# ---------------------------------------------------------------------------
# mara verify — file not found
# ---------------------------------------------------------------------------


class TestVerifyCommandFileNotFound:
    def test_exits_1_when_file_missing(self, tmp_path):
        result = runner.invoke(app, ["verify", str(tmp_path / "missing.json")])
        assert result.exit_code == 1

    def test_error_message_for_missing_file(self, tmp_path):
        result = runner.invoke(app, ["verify", str(tmp_path / "missing.json")])
        assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()


# ---------------------------------------------------------------------------
# mara verify — via mocked verify_report
# ---------------------------------------------------------------------------


class TestVerifyCommandDispatch:
    def test_exits_0_on_passing_report(self, mocker, tmp_path):
        report_path = tmp_path / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        mocker.patch("mara.cli.run.load_report", return_value=mocker.MagicMock())
        mocker.patch("mara.cli.run.verify_report", return_value=_make_passing_result())
        result = runner.invoke(app, ["verify", str(report_path)])
        assert result.exit_code == 0

    def test_exits_1_on_failing_report(self, mocker, tmp_path):
        report_path = tmp_path / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        mocker.patch("mara.cli.run.load_report", return_value=mocker.MagicMock())
        mocker.patch("mara.cli.run.verify_report", return_value=_make_failing_result())
        result = runner.invoke(app, ["verify", str(report_path)])
        assert result.exit_code == 1

    def test_load_report_called_with_path(self, mocker, tmp_path):
        report_path = tmp_path / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        mock_load = mocker.patch("mara.cli.run.load_report", return_value=mocker.MagicMock())
        mocker.patch("mara.cli.run.verify_report", return_value=_make_passing_result())
        runner.invoke(app, ["verify", str(report_path)])
        assert mock_load.call_count == 1

    def test_verify_report_called_with_loaded_report(self, mocker, tmp_path):
        report_path = tmp_path / "report.json"
        report_path.write_text("{}", encoding="utf-8")
        mock_report = mocker.MagicMock()
        mocker.patch("mara.cli.run.load_report", return_value=mock_report)
        mock_verify = mocker.patch("mara.cli.run.verify_report", return_value=_make_passing_result())
        runner.invoke(app, ["verify", str(report_path)])
        mock_verify.assert_called_once_with(mock_report)


# ---------------------------------------------------------------------------
# _display_verification output
# ---------------------------------------------------------------------------


class TestDisplayVerification:
    def _capture(self, result: VerificationResult, mocker) -> str:
        lines = []
        mocker.patch(
            "mara.cli.run.typer.echo",
            side_effect=lambda s="", **kw: lines.append(str(s)),
        )
        _display_verification(result)
        return "\n".join(lines)

    def test_pass_appears_on_passing_result(self, mocker):
        output = self._capture(_make_passing_result(), mocker)
        assert "PASS" in output

    def test_fail_appears_on_failing_result(self, mocker):
        output = self._capture(_make_failing_result(), mocker)
        assert "FAIL" in output

    def test_query_appears_in_output(self, mocker):
        output = self._capture(_make_passing_result(), mocker)
        assert "What is X?" in output

    def test_leaf_ok_appears_for_passing_leaves(self, mocker):
        output = self._capture(_make_passing_result(n_leaves=1), mocker)
        assert "OK" in output

    def test_leaf_fail_appears_for_failing_leaves(self, mocker):
        output = self._capture(_make_failing_result(), mocker)
        assert "FAIL" in output

    def test_leaf_url_appears(self, mocker):
        output = self._capture(_make_passing_result(n_leaves=1), mocker)
        assert "https://example.com/0" in output

    def test_merkle_root_ok_appears_on_pass(self, mocker):
        output = self._capture(_make_passing_result(), mocker)
        assert "OK" in output

    def test_merkle_root_mismatch_appears_on_fail(self, mocker):
        output = self._capture(_make_failing_result(), mocker)
        assert "MISMATCH" in output

    def test_leaf_count_appears(self, mocker):
        output = self._capture(_make_passing_result(n_leaves=3), mocker)
        assert "3" in output

    def test_header_appears(self, mocker):
        output = self._capture(_make_passing_result(), mocker)
        assert "VERIFICATION" in output

    def test_failed_hash_values_shown(self, mocker):
        result = _make_failing_result()
        output = self._capture(result, mocker)
        assert result.leaf_results[0].expected_hash in output
        assert result.leaf_results[0].computed_hash in output
