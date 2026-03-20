"""Corrective RAG retriever node.

When confidence scoring identifies claims with insufficient evidence (low SA
score AND few leaves), this node:

  1. Generates claim-level corrective sub-queries via the LLM.
  2. Searches the leaf DB (BM25, cross-run) for existing evidence first.
  3. Falls back to live Brave + Firecrawl scraping when the DB is insufficient.
  4. Appends new leaves directly to ``merkle_leaves``, then increments
     ``loop_count`` and returns.

The graph back-edge routes this node → ``merkle_builder``, so the full
scoring pipeline (merkle_builder → retriever → claim_extractor →
confidence_scorer) re-runs on the expanded leaf pool.  ``source_hasher`` is
intentionally skipped: corrective leaves are inserted directly with hashes
already computed here.

Contested claims (low SA but n_leaves ≥ n_leaves_contested_threshold) are
NOT processed here — they route directly to ``hitl_checkpoint``.  This node
only handles the *insufficient data* case.

When ``leaf_db_enabled=False`` all DB calls are skipped; LLM query generation
and live scraping still fire.
"""

from __future__ import annotations

import json
import logging

from langchain_core.runnables import RunnableConfig

from mara.agent.llm import make_llm
from mara.agent.nodes.query_planner import _parse_sub_queries
from mara.agent.nodes.search_worker.brave_search import brave_search
from mara.agent.nodes.search_worker.firecrawl_scrape import firecrawl_scrape
from mara.agent.state import MARAState, MerkleLeaf, SearchWorkerState, SubQuery
from mara.merkle.hasher import hash_chunk
from mara.prompts.corrective_retriever import build_system_prompt, build_user_prompt

_log = logging.getLogger(__name__)


async def _generate_corrective_sub_queries(
    failing_claims: list,
    query: str,
    cfg,
    config: RunnableConfig,
) -> list[SubQuery]:
    """Generate 1-2 targeted sub-queries per failing claim via the LLM.

    Args:
        failing_claims: list[ScoredClaim] — claims below the confidence threshold
                        with insufficient leaf coverage.
        query:          The original research question.
        cfg:            ResearchConfig.
        config:         LangGraph RunnableConfig passed through to the LLM.

    Returns:
        Flat list of SubQuery dicts (1-2 per claim, deduplicated by insertion order).
    """
    llm = make_llm(cfg.model, cfg.hf_token, cfg.corrective_retriever_max_tokens, cfg.hf_provider, cfg.temperature, cfg.top_p, cfg.top_k, cfg.presence_penalty)
    corrective_sub_queries: list[SubQuery] = []

    for claim in failing_claims:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(claim.text, query)},
        ]
        response = await llm.ainvoke(messages, config)
        try:
            sub_queries = _parse_sub_queries(response.content, 2)
        except (json.JSONDecodeError, ValueError, KeyError):
            _log.warning(
                "Failed to parse corrective sub-queries for claim %r — falling back to claim text",
                claim.text[:80],
            )
            sub_queries = [SubQuery(query=claim.text[:200], domain="general")]
        corrective_sub_queries.extend(sub_queries)

    return corrective_sub_queries


async def corrective_retriever(state: MARAState, config: RunnableConfig) -> dict:
    """Acquire additional evidence for low-confidence, data-poor claims.

    Args:
        state:  MARAState after ``confidence_scorer`` has run.
        config: LangGraph RunnableConfig — provides ``leaf_repo`` and ``run_id``
                via ``config["configurable"]``.

    Returns:
        dict with updated keys:
          - ``merkle_leaves``:          original + new DB + new scraped leaves
          - ``corrective_sub_queries``: accumulated across all corrective rounds
          - ``loop_count``:             incremented by 1
    """
    cfg = state["config"]

    # Step 1 — Identify failing claims (insufficient data, not contested)
    failing = [
        c for c in state["scored_claims"]
        if c.confidence < cfg.low_confidence_threshold
        and c.n_leaves < cfg.n_leaves_contested_threshold
    ]

    if not failing:
        _log.debug("No failing claims — incrementing loop_count and returning early")
        return {"loop_count": state["loop_count"] + 1}

    _log.info("Corrective retrieval for %d failing claim(s)", len(failing))

    # Step 2 — Generate corrective sub-queries
    corrective_sub_queries = await _generate_corrective_sub_queries(
        failing, state["query"], cfg, config
    )

    # Step 3 — DB-first retrieval
    configurable = config.get("configurable", {}) if config else {}
    leaf_repo = configurable.get("leaf_repo")
    run_id = configurable.get("run_id")

    existing_hashes: set[str] = {leaf["hash"] for leaf in state["merkle_leaves"]}
    new_db_leaves: list[MerkleLeaf] = []

    if leaf_repo is not None and cfg.leaf_db_enabled:
        for claim in failing:
            db_results = leaf_repo.bm25_search(claim.text, run_id=None, limit=20)
            for row in db_results:
                if row["hash"] not in existing_hashes:
                    idx = len(state["merkle_leaves"]) + len(new_db_leaves)
                    leaf = MerkleLeaf(
                        url=row["url"],
                        text=row["text"],
                        retrieved_at=row["retrieved_at"],
                        hash=row["hash"],
                        index=idx,
                        sub_query=claim.text[:200],
                        contextualized_text=row.get("contextualized_text") or row["text"],
                    )
                    new_db_leaves.append(leaf)
                    existing_hashes.add(row["hash"])

        if new_db_leaves and run_id:
            leaf_repo.link_leaves_to_run(run_id, new_db_leaves)

    # Step 4 — Scrape new pages if DB is insufficient
    db_sufficient = len(new_db_leaves) >= 3 * len(failing)
    new_scraped_leaves: list[MerkleLeaf] = []

    if not db_sufficient:
        existing_urls: set[str] = {leaf["url"] for leaf in state["merkle_leaves"]}
        existing_urls.update(leaf["url"] for leaf in new_db_leaves)

        for sub_query in corrective_sub_queries:
            # Brave search
            worker_state = SearchWorkerState(
                sub_query=sub_query,
                research_config=cfg,
                search_results=[],
                raw_chunks=[],
            )
            search_result = await brave_search(worker_state, config)
            search_results = search_result["search_results"]

            # Dedup and cap URLs
            new_search_results = [
                r for r in search_results if r["url"] not in existing_urls
            ]
            new_search_results = new_search_results[: cfg.max_new_pages_per_round]

            if not new_search_results:
                continue

            existing_urls.update(r["url"] for r in new_search_results)

            # Scrape
            scrape_state = SearchWorkerState(
                sub_query=sub_query,
                research_config=cfg,
                search_results=new_search_results,
                raw_chunks=[],
            )
            scrape_result = await firecrawl_scrape(scrape_state, config)
            raw_chunks = scrape_result["raw_chunks"]

            for chunk in raw_chunks:
                leaf_hash = hash_chunk(
                    chunk["url"], chunk["text"], chunk["retrieved_at"], cfg.hash_algorithm
                )
                if leaf_hash not in existing_hashes:
                    idx = len(state["merkle_leaves"]) + len(new_db_leaves) + len(new_scraped_leaves)
                    leaf = MerkleLeaf(
                        url=chunk["url"],
                        text=chunk["text"],
                        retrieved_at=chunk["retrieved_at"],
                        hash=leaf_hash,
                        index=idx,
                        sub_query=sub_query["query"],
                        contextualized_text=chunk["text"],
                    )
                    new_scraped_leaves.append(leaf)
                    existing_hashes.add(leaf_hash)

        if new_scraped_leaves and leaf_repo is not None and cfg.leaf_db_enabled:
            leaf_repo.upsert_leaves(new_scraped_leaves)
            if run_id:
                leaf_repo.link_leaves_to_run(run_id, new_scraped_leaves)

    all_new_leaves = new_db_leaves + new_scraped_leaves
    _log.info(
        "Corrective retrieval added %d leaf(ves): %d from DB, %d scraped",
        len(all_new_leaves),
        len(new_db_leaves),
        len(new_scraped_leaves),
    )

    # Step 5 — Return
    return {
        "merkle_leaves": state["merkle_leaves"] + all_new_leaves,
        "corrective_sub_queries": state.get("corrective_sub_queries", []) + corrective_sub_queries,
        "loop_count": state["loop_count"] + 1,
    }
