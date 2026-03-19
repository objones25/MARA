"""Tests for mara.cli.run.

All LLM calls and graph execution are mocked.  Tests cover:
  - _setup_logging: verbose and non-verbose branches
  - _review_claims: empty input, valid indices, invalid tokens, source display
  - _display_report: output contains key fields
  - info command: config values and node list appear in output
  - run command: dispatches asyncio.run, passes query and thread_id, calls _setup_logging
  - _run (async): no-interrupt path, HITL interrupt + resume path, no-report error path
"""

import asyncio
import logging
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from mara.agent.state import CertifiedReport
from mara.cli.run import (
    _display_report,
    _review_claims,
    _run,
    _setup_logging,
    app,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(
    query: str = "What is X?",
    report_text: str = "The research says X.",
    merkle_root: str = "abc123",
    leaves: int = 3,
    claims: int = 2,
) -> CertifiedReport:
    return CertifiedReport(
        query=query,
        report_text=report_text,
        merkle_root=merkle_root,
        leaves=list(range(leaves)),
        scored_claims=list(range(claims)),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_interrupt_value(
    auto_approved_count: int = 1,
    needs_review: list | None = None,
) -> dict:
    if needs_review is None:
        needs_review = [
            {
                "index": 0,
                "text": "Claim needing review",
                "confidence": 0.60,
                "source_indices": [1, 2],
            }
        ]
    return {"needs_review": needs_review, "auto_approved_count": auto_approved_count}


# ---------------------------------------------------------------------------
# _setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_verbose_sets_debug_level(self):
        _setup_logging(verbose=True)
        assert logging.getLogger("mara").level == logging.DEBUG

    def test_non_verbose_sets_info_level(self):
        _setup_logging(verbose=False)
        assert logging.getLogger("mara").level == logging.INFO

    def test_adds_stream_handler(self):
        before = len(logging.getLogger("mara").handlers)
        _setup_logging(verbose=False)
        assert len(logging.getLogger("mara").handlers) > before


# ---------------------------------------------------------------------------
# _review_claims
# ---------------------------------------------------------------------------


class TestReviewClaims:
    def test_empty_input_returns_empty_list(self, mocker):
        mocker.patch("mara.cli.run.typer.prompt", return_value="")
        assert _review_claims(_make_interrupt_value()) == []

    def test_whitespace_input_returns_empty_list(self, mocker):
        mocker.patch("mara.cli.run.typer.prompt", return_value="   ")
        assert _review_claims(_make_interrupt_value()) == []

    def test_single_valid_index(self, mocker):
        mocker.patch("mara.cli.run.typer.prompt", return_value="0")
        assert _review_claims(_make_interrupt_value()) == [0]

    def test_multiple_valid_indices(self, mocker):
        mocker.patch("mara.cli.run.typer.prompt", return_value="0, 1, 2")
        assert _review_claims(_make_interrupt_value()) == [0, 1, 2]

    def test_invalid_tokens_are_skipped(self, mocker):
        mocker.patch("mara.cli.run.typer.prompt", return_value="0, abc, 2")
        assert _review_claims(_make_interrupt_value()) == [0, 2]

    def test_all_invalid_tokens_returns_empty(self, mocker):
        mocker.patch("mara.cli.run.typer.prompt", return_value="foo, bar")
        assert _review_claims(_make_interrupt_value()) == []

    def test_auto_approved_count_shown(self, mocker):
        mock_echo = mocker.patch("mara.cli.run.typer.echo")
        mocker.patch("mara.cli.run.typer.prompt", return_value="")
        _review_claims(_make_interrupt_value(auto_approved_count=3))
        output = " ".join(str(c) for c in mock_echo.call_args_list)
        assert "3" in output

    def test_claim_text_shown(self, mocker):
        mock_echo = mocker.patch("mara.cli.run.typer.echo")
        mocker.patch("mara.cli.run.typer.prompt", return_value="")
        _review_claims(_make_interrupt_value())
        output = " ".join(str(c) for c in mock_echo.call_args_list)
        assert "Claim needing review" in output

    def test_source_indices_shown_when_present(self, mocker):
        mock_echo = mocker.patch("mara.cli.run.typer.echo")
        mocker.patch("mara.cli.run.typer.prompt", return_value="")
        _review_claims(_make_interrupt_value())
        output = " ".join(str(c) for c in mock_echo.call_args_list)
        assert "sources:" in output

    def test_no_source_indices_omits_sources_line(self, mocker):
        mock_echo = mocker.patch("mara.cli.run.typer.echo")
        mocker.patch("mara.cli.run.typer.prompt", return_value="")
        value = _make_interrupt_value(
            needs_review=[{"index": 0, "text": "x", "confidence": 0.5, "source_indices": []}]
        )
        _review_claims(value)
        output = " ".join(str(c) for c in mock_echo.call_args_list)
        assert "sources:" not in output


# ---------------------------------------------------------------------------
# _display_report
# ---------------------------------------------------------------------------


class TestDisplayReport:
    def _capture(self, report: CertifiedReport, mocker) -> str:
        lines = []
        mocker.patch(
            "mara.cli.run.typer.echo",
            side_effect=lambda s, **kw: lines.append(str(s)),
        )
        _display_report(report)
        return "\n".join(lines)

    def test_report_text_appears(self, mocker):
        assert "This is the report." in self._capture(
            _make_report(report_text="This is the report."), mocker
        )

    def test_query_appears(self, mocker):
        assert "What is Y?" in self._capture(_make_report(query="What is Y?"), mocker)

    def test_merkle_root_appears(self, mocker):
        assert "deadbeef" in self._capture(_make_report(merkle_root="deadbeef"), mocker)

    def test_leaf_count_appears(self, mocker):
        assert "5" in self._capture(_make_report(leaves=5), mocker)

    def test_claim_count_appears(self, mocker):
        assert "4" in self._capture(_make_report(claims=4), mocker)

    def test_generated_at_appears(self, mocker):
        report = _make_report()
        output = self._capture(report, mocker)
        assert report.generated_at in output


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_exit_code_zero(self):
        assert runner.invoke(app, ["info"]).exit_code == 0

    def test_shows_model(self):
        assert "claude" in runner.invoke(app, ["info"]).output

    def test_shows_max_workers(self):
        assert "Max workers" in runner.invoke(app, ["info"]).output

    def test_shows_high_conf_threshold(self):
        assert "High conf threshold" in runner.invoke(app, ["info"]).output

    def test_shows_low_conf_threshold(self):
        assert "Low conf threshold" in runner.invoke(app, ["info"]).output

    def test_shows_graph_nodes_header(self):
        assert "Graph nodes" in runner.invoke(app, ["info"]).output

    def test_shows_query_planner_node(self):
        assert "query_planner" in runner.invoke(app, ["info"]).output

    def test_shows_certified_output_node(self):
        assert "certified_output" in runner.invoke(app, ["info"]).output

    def test_internal_langgraph_nodes_hidden(self):
        assert "__start__" not in runner.invoke(app, ["info"]).output


# ---------------------------------------------------------------------------
# run command (sync shell)
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_run_invokes_asyncio_run(self, mocker):
        mock_asyncio_run = mocker.patch("mara.cli.run.asyncio.run")
        runner.invoke(app, ["run", "test query"])
        mock_asyncio_run.assert_called_once()

    def test_run_passes_query(self, mocker):
        captured = []

        async def fake_run(query, thread_id):
            captured.append(query)

        mocker.patch("mara.cli.run._run", side_effect=fake_run)
        mocker.patch(
            "mara.cli.run.asyncio.run",
            side_effect=lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        runner.invoke(app, ["run", "my research question"])
        assert captured == ["my research question"]

    def test_run_passes_thread_id(self, mocker):
        captured = []

        async def fake_run(query, thread_id):
            captured.append(thread_id)

        mocker.patch("mara.cli.run._run", side_effect=fake_run)
        mocker.patch(
            "mara.cli.run.asyncio.run",
            side_effect=lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        runner.invoke(app, ["run", "q", "--thread-id", "my-thread"])
        assert captured == ["my-thread"]

    def test_run_calls_setup_logging(self, mocker):
        mock_setup = mocker.patch("mara.cli.run._setup_logging")
        mocker.patch("mara.cli.run.asyncio.run")
        runner.invoke(app, ["run", "q"])
        mock_setup.assert_called_once()

    def test_verbose_flag_forwarded_to_setup_logging(self, mocker):
        mock_setup = mocker.patch("mara.cli.run._setup_logging")
        mocker.patch("mara.cli.run.asyncio.run")
        runner.invoke(app, ["run", "q", "--verbose"])
        mock_setup.assert_called_once_with(True)

    def test_default_thread_id_is_mara_1(self, mocker):
        captured = []

        async def fake_run(query, thread_id):
            captured.append(thread_id)

        mocker.patch("mara.cli.run._run", side_effect=fake_run)
        mocker.patch(
            "mara.cli.run.asyncio.run",
            side_effect=lambda coro: asyncio.get_event_loop().run_until_complete(coro),
        )
        runner.invoke(app, ["run", "q"])
        assert captured == ["mara-1"]


# ---------------------------------------------------------------------------
# _run (async) — no interrupt
# ---------------------------------------------------------------------------


class TestRunAsync:
    def _mock_graph(self, mocker, result: dict):
        mock_graph = mocker.MagicMock()
        mock_graph.ainvoke = mocker.AsyncMock(return_value=result)
        mocker.patch("mara.cli.run.build_graph", return_value=mock_graph)
        mocker.patch("mara.cli.run.MemorySaver")
        return mock_graph

    async def test_displays_report_on_success(self, mocker):
        report = _make_report()
        self._mock_graph(mocker, {"certified_report": report})
        mock_display = mocker.patch("mara.cli.run._display_report")
        await _run("test query", "t-1")
        mock_display.assert_called_once_with(report)

    async def test_no_report_raises_exit_code_1(self, mocker):
        self._mock_graph(mocker, {"certified_report": None})
        import typer as _typer
        with pytest.raises(_typer.Exit) as exc_info:
            await _run("test query", "t-1")
        assert exc_info.value.exit_code == 1

    async def test_query_in_initial_state(self, mocker):
        report = _make_report()
        mock_graph = self._mock_graph(mocker, {"certified_report": report})
        mocker.patch("mara.cli.run._display_report")
        await _run("my question", "t-1")
        initial_state = mock_graph.ainvoke.call_args.args[0]
        assert initial_state["query"] == "my question"

    async def test_thread_id_in_run_config(self, mocker):
        report = _make_report()
        mock_graph = self._mock_graph(mocker, {"certified_report": report})
        mocker.patch("mara.cli.run._display_report")
        await _run("q", "my-thread-99")
        run_config = mock_graph.ainvoke.call_args.args[1]
        assert run_config["configurable"]["thread_id"] == "my-thread-99"

    async def test_initial_state_has_zero_loop_count(self, mocker):
        report = _make_report()
        mock_graph = self._mock_graph(mocker, {"certified_report": report})
        mocker.patch("mara.cli.run._display_report")
        await _run("q", "t-1")
        initial_state = mock_graph.ainvoke.call_args.args[0]
        assert initial_state["loop_count"] == 0
        assert initial_state["retrieved_leaves"] == []


# ---------------------------------------------------------------------------
# _run (async) — HITL interrupt + resume
# ---------------------------------------------------------------------------


class TestRunAsyncHitl:
    async def test_interrupt_triggers_review(self, mocker):
        from unittest.mock import MagicMock

        interrupt = MagicMock()
        interrupt.value = _make_interrupt_value()

        mock_graph = mocker.MagicMock()
        mock_graph.ainvoke = mocker.AsyncMock(
            side_effect=[
                {"__interrupt__": [interrupt]},
                {"certified_report": _make_report()},
            ]
        )
        mocker.patch("mara.cli.run.build_graph", return_value=mock_graph)
        mocker.patch("mara.cli.run.MemorySaver")
        mock_review = mocker.patch("mara.cli.run._review_claims", return_value=[0])
        mocker.patch("mara.cli.run._display_report")

        await _run("q", "t-1")
        mock_review.assert_called_once()

    async def test_resume_carries_approved_indices(self, mocker):
        from unittest.mock import MagicMock

        from langgraph.types import Command

        interrupt = MagicMock()
        interrupt.value = _make_interrupt_value()

        mock_graph = mocker.MagicMock()
        mock_graph.ainvoke = mocker.AsyncMock(
            side_effect=[
                {"__interrupt__": [interrupt]},
                {"certified_report": _make_report()},
            ]
        )
        mocker.patch("mara.cli.run.build_graph", return_value=mock_graph)
        mocker.patch("mara.cli.run.MemorySaver")
        mocker.patch("mara.cli.run._review_claims", return_value=[0, 1])
        mocker.patch("mara.cli.run._display_report")

        await _run("q", "t-1")

        resume_arg = mock_graph.ainvoke.call_args_list[1].args[0]
        assert isinstance(resume_arg, Command)
        assert resume_arg.resume == {"approved_indices": [0, 1]}

    async def test_multiple_interrupts_all_handled(self, mocker):
        from unittest.mock import MagicMock

        interrupt = MagicMock()
        interrupt.value = _make_interrupt_value()

        mock_graph = mocker.MagicMock()
        mock_graph.ainvoke = mocker.AsyncMock(
            side_effect=[
                {"__interrupt__": [interrupt]},
                {"__interrupt__": [interrupt]},
                {"certified_report": _make_report()},
            ]
        )
        mocker.patch("mara.cli.run.build_graph", return_value=mock_graph)
        mocker.patch("mara.cli.run.MemorySaver")
        mocker.patch("mara.cli.run._review_claims", return_value=[])
        mocker.patch("mara.cli.run._display_report")

        await _run("q", "t-1")
        assert mock_graph.ainvoke.call_count == 3
