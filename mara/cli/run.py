"""MARA command-line interface.

Commands
--------
  mara run QUERY   -- Run the full research pipeline for QUERY.
  mara info        -- Print configuration and graph structure.

HITL resume
-----------
When the pipeline pauses for human review, the terminal displays each
claim that fell below the high-confidence threshold.  Enter a
comma-separated list of indices to approve, or press Enter to skip all.
The graph then resumes with only the approved claims included.

Usage examples
--------------
    mara run "What are the long-term effects of remote work on productivity?"
    mara run "effects of remote work" --verbose
    mara info
"""

import asyncio
import logging
import uuid
from pathlib import Path

import typer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from mara.agent.graph import build_graph
from mara.agent.run_context import RunContext
from mara.agent.state import CertifiedReport
from mara.config import ResearchConfig
from mara.logging import get_logger
from mara.report_store import DEFAULT_REPORT_DIR, load_report, save_report
from mara.verifier import VerificationResult, verify_report

_log = get_logger(__name__)

app = typer.Typer(
    name="mara",
    help="Merkle-Anchored Research Agent — cryptographically verifiable research.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    """Configure the mara.* logger hierarchy."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger("mara")
    root.setLevel(level)
    root.addHandler(handler)


def _review_claims(interrupt_value: dict) -> list[int]:
    """Display claims needing human review and collect approval indices.

    Args:
        interrupt_value: Payload from hitl_checkpoint's interrupt() call.
            Keys: ``needs_review`` (list of dicts), ``auto_approved_count``.

    Returns:
        List of indices (into ``needs_review``) that the human approved.
    """
    needs_review = interrupt_value["needs_review"]
    auto_count = interrupt_value.get("auto_approved_count", 0)

    typer.echo(f"\n{'=' * 60}")
    typer.echo("HITL CHECKPOINT — Human Review Required")
    typer.echo(f"{'=' * 60}")
    typer.echo(f"{auto_count} claim(s) auto-approved (confidence ≥ threshold).\n")
    typer.echo(f"{len(needs_review)} claim(s) need your review:\n")

    for item in needs_review:
        typer.echo(f"  [{item['index']}] confidence={item['confidence']:.2f}")
        typer.echo(f"      {item['text']}")
        if item["source_indices"]:
            typer.echo(f"      sources: {item['source_indices']}")
        typer.echo()

    raw = typer.prompt(
        "Enter comma-separated indices to approve (or press Enter to approve none)",
        default="",
    )

    if not raw.strip():
        return []

    approved: list[int] = []
    for part in raw.split(","):
        try:
            approved.append(int(part.strip()))
        except ValueError:
            pass
    return approved


def _display_report(report: CertifiedReport) -> None:
    """Render the CertifiedReport to the terminal."""
    typer.echo(f"\n{'=' * 60}")
    typer.echo("CERTIFIED RESEARCH REPORT")
    typer.echo(f"{'=' * 60}\n")
    typer.echo(f"Query: {report.query}\n")
    typer.echo(report.report_text)
    typer.echo(f"\n{'─' * 60}")
    typer.echo(f"Merkle root:  {report.merkle_root}")
    typer.echo(f"Leaves:       {len(report.leaves)}")
    typer.echo(f"Claims:       {len(report.scored_claims)}")
    typer.echo(f"Generated at: {report.generated_at}")


async def _run(query: str, thread_id: str, output_dir: Path | None = None) -> None:
    """Core async pipeline runner.  Called by the ``run`` CLI command."""
    config = ResearchConfig()
    checkpointer = MemorySaver(serde=JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("mara.config", "ResearchConfig"),
            ("mara.merkle.tree", "MerkleTree"),
            ("mara.confidence.scorer", "ScoredClaim"),
        ]
    ))
    graph = build_graph(checkpointer=checkpointer)

    # Build the RunnableConfig, injecting leaf_repo + run_id when enabled.
    run_id = str(uuid.uuid4())
    configurable: dict = {"thread_id": thread_id}

    leaf_repo = None
    if config.leaf_db_enabled:
        from mara.db import SQLiteLeafRepository
        leaf_repo = SQLiteLeafRepository(config.leaf_db_path)
        leaf_repo.create_run(
            run_id=run_id,
            query=query,
            embedding_model=config.embedding_model,
            hash_algorithm=config.hash_algorithm,
        )
        configurable["leaf_repo"] = leaf_repo
        configurable["run_id"] = run_id
        _log.info("Leaf DB enabled — run_id=%s, db=%s", run_id, config.leaf_db_path)

    run_context = RunContext()
    configurable["run_context"] = run_context

    run_config = {"configurable": configurable}

    from datetime import datetime, timezone
    initial_state = {
        "query": query,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "config": config,
        "sub_queries": [],
        "search_results": [],
        "raw_chunks": [],
        "merkle_leaves": [],
        "merkle_tree": None,
        "retrieved_leaves": [],
        "extracted_claims": [],
        "scored_claims": [],
        "human_approved_claims": [],
        "report_draft": "",
        "certified_report": None,
        "messages": [],
        "loop_count": 0,
    }

    typer.echo(f"Running MARA for: {query!r}\n")
    result = await graph.ainvoke(initial_state, run_config)

    # Resume through any HITL interrupts
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        approved_indices = _review_claims(interrupt_value)
        result = await graph.ainvoke(
            Command(resume={"approved_indices": approved_indices}),
            run_config,
        )

    report = result.get("certified_report")
    if report is None:
        typer.echo("Pipeline produced no report.", err=True)
        if leaf_repo is not None:
            leaf_repo.close()
        raise typer.Exit(code=1)

    _display_report(report)

    if output_dir is not None:
        saved_path = save_report(report, output_dir)
        typer.echo(f"\nReport saved: {saved_path}")

    if leaf_repo is not None:
        leaf_repo.close()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    query: str = typer.Argument(..., help="Research question to investigate."),
    thread_id: str = typer.Option(
        "mara-1", "--thread-id", "-t", help="Checkpointer thread ID."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    output_dir: Path = typer.Option(
        DEFAULT_REPORT_DIR,
        "--output-dir",
        "-o",
        help="Directory to save the CertifiedReport JSON.",
    ),
) -> None:
    """Run the MARA research pipeline for QUERY."""
    _setup_logging(verbose)
    asyncio.run(_run(query, thread_id, output_dir))


@app.command()
def info() -> None:
    """Print MARA configuration and compiled graph node list."""
    config = ResearchConfig()
    typer.echo("MARA Configuration")
    typer.echo(f"  Model:                {config.model}")
    typer.echo(f"  Embedding model:      {config.embedding_model}")
    typer.echo(f"  HF provider:          {config.hf_provider}")
    typer.echo(f"  Max workers:          {config.max_workers}")
    typer.echo(f"  Max sources:          {config.max_sources}")
    typer.echo(f"  High conf threshold:  {config.high_confidence_threshold}")
    typer.echo(f"  Low conf threshold:   {config.low_confidence_threshold}")
    typer.echo(f"  Max corrective loops: {config.max_corrective_rag_loops}")
    typer.echo(f"  Hash algorithm:       {config.hash_algorithm}")
    typer.echo(f"  Checkpointer:         {config.checkpointer}")
    typer.echo(f"  Leaf DB enabled:      {config.leaf_db_enabled}")
    if config.leaf_db_enabled:
        typer.echo(f"  Leaf DB path:         {config.leaf_db_path}")
        typer.echo(f"  Leaf cache max age:   {config.leaf_cache_max_age_hours}h")
    typer.echo(f"  ArXiv max results:    {config.arxiv_max_results} papers/sub-query")

    typer.echo("\nGraph nodes:")
    graph = build_graph()
    for name in sorted(graph.nodes.keys()):
        if not name.startswith("__"):
            typer.echo(f"  {name}")


@app.command()
def verify(
    report_path: Path = typer.Argument(..., help="Path to a saved CertifiedReport JSON."),
) -> None:
    """Verify the cryptographic integrity of a saved CertifiedReport."""
    if not report_path.exists():
        typer.echo(f"Report not found: {report_path}", err=True)
        raise typer.Exit(code=1)

    report = load_report(report_path)
    result = verify_report(report)
    _display_verification(result)

    if not result.passed:
        raise typer.Exit(code=1)


def _display_verification(result: VerificationResult) -> None:
    """Render a VerificationResult to the terminal."""
    typer.echo(f"\n{'=' * 60}")
    typer.echo("MARA INTEGRITY VERIFICATION")
    typer.echo(f"{'=' * 60}\n")
    typer.echo(f"Query: {result.report_query}\n")

    for lr in result.leaf_results:
        mark = "OK" if lr.match else "FAIL"
        typer.echo(f"  [{mark}] leaf {lr.index}: {lr.url}")
        if not lr.match:
            typer.echo(f"         expected: {lr.expected_hash}")
            typer.echo(f"         computed: {lr.computed_hash}")

    typer.echo()
    total = len(result.leaf_results)
    passed_count = sum(1 for r in result.leaf_results if r.match)
    typer.echo(f"Leaves: {passed_count}/{total} verified")

    typer.echo()
    if result.root_match:
        typer.echo(f"Merkle root:  {result.merkle_root_expected[:16]}…  OK")
    else:
        typer.echo("Merkle root:  MISMATCH")
        typer.echo(f"  expected: {result.merkle_root_expected}")
        typer.echo(f"  computed: {result.merkle_root_computed}")

    typer.echo()
    typer.echo(f"Integrity: {'PASS' if result.passed else 'FAIL'}")


if __name__ == "__main__":  # pragma: no cover
    app()
