# MARA — Merkle-Anchored Research Agent

> A cryptographically verifiable, hallucination-resistant research agent built on LangGraph, grounded by Merkle trees, and evaluated through LangSmith.

---

## Table of Contents

- [MARA — Merkle-Anchored Research Agent](#mara--merkle-anchored-research-agent)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [The Problem](#the-problem)
  - [How MARA Is Different](#how-mara-is-different)
  - [Architecture](#architecture)
  - [Core Components](#core-components)
    - [1. Query Planner](#1-query-planner)
    - [2. Parallel Search Workers](#2-parallel-search-workers)
    - [3. Source Hasher](#3-source-hasher)
    - [4. Merkle Integrity Layer](#4-merkle-integrity-layer)
      - [How the Merkle tree is built](#how-the-merkle-tree-is-built)
      - [Why a Merkle tree instead of a flat list of hashes?](#why-a-merkle-tree-instead-of-a-flat-list-of-hashes)
    - [5. Hybrid Retriever](#5-hybrid-retriever)
    - [6. Confidence Scorer](#6-confidence-scorer)
      - [The scoring model](#the-scoring-model)
      - [Routing based on confidence](#routing-based-on-confidence)
    - [7. Corrective Retriever](#7-corrective-retriever)
    - [8. HITL Checkpoint](#8-hitl-checkpoint)
    - [9. Report Synthesizer](#9-report-synthesizer)
    - [10. Certified Output](#10-certified-output)
  - [The Merkle Integrity Protocol](#the-merkle-integrity-protocol)
  - [Leaf Database](#leaf-database)
    - [Freshness cache](#freshness-cache)
    - [Cross-session corpus growth](#cross-session-corpus-growth)
    - [Embedding persistence](#embedding-persistence)
    - [Schema overview](#schema-overview)
    - [Disabling the database](#disabling-the-database)
  - [Retrieval Strategy](#retrieval-strategy)
    - [Current approach](#current-approach)
    - [Why Brave search results are not hashed](#why-brave-search-results-are-not-hashed)
    - [The chunking constraint](#the-chunking-constraint)
    - [Hybrid retrieval over the scraped corpus](#hybrid-retrieval-over-the-scraped-corpus)
    - [Planned: local corpus retrieval](#planned-local-corpus-retrieval)
    - [Retrieval experimentation](#retrieval-experimentation)
  - [Statistical Confidence Scoring](#statistical-confidence-scoring)
    - [Model framing](#model-framing)
    - [Claim extraction](#claim-extraction)
  - [LangGraph Implementation Details](#langgraph-implementation-details)
    - [State schema](#state-schema)
    - [Checkpointing and durability](#checkpointing-and-durability)
    - [Subgraph isolation](#subgraph-isolation)
  - [LangSmith Observability](#langsmith-observability)
    - [What LangSmith captures](#what-langsmith-captures)
    - [Nested LLM traces via config forwarding](#nested-llm-traces-via-config-forwarding)
    - [Tags and metadata at invocation time](#tags-and-metadata-at-invocation-time)
    - [Python logger (`mara.logging`)](#python-logger-maralogging)
  - [Citation Format](#citation-format)
  - [Verification](#verification)
  - [Configuration Reference](#configuration-reference)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [Design Decisions \& Tradeoffs](#design-decisions--tradeoffs)
  - [Testing](#testing)
    - [Tools](#tools)
    - [`pyproject.toml` configuration](#pyprojecttoml-configuration)
    - [Running tests](#running-tests)
    - [Coverage target: 98%](#coverage-target-98)
    - [Testing philosophy](#testing-philosophy)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Install](#install)
    - [Configure](#configure)
    - [Run](#run)
    - [Run tests](#run-tests)

---

## Overview

**MARA** (Merkle-Anchored Research Agent) is a multi-agent research system that does something no current research tool does: it makes citations _cryptographically verifiable_. Every source chunk the agent retrieves is hashed using SHA-256 and committed to a Merkle tree. Every factual claim in the final report is tagged with the Merkle leaf it derives from. The Merkle root hash is embedded in the report itself.

This means that any reader, at any time, can recompute the hash of a cited source document and verify that it matches the leaf recorded at the time of research. A hallucinated citation — one pointing to a source that doesn't say what the agent claims — will always produce a hash mismatch.

On top of the cryptographic layer, MARA adds a **statistical confidence scorer** that estimates the probability each claim is grounded in the retrieved evidence, and a **human-in-the-loop checkpoint** that pauses execution for human review when confidence falls below a configurable threshold.

The agent is orchestrated as a `StateGraph` in LangGraph, traced end-to-end in LangSmith, and designed to be run as a local prototype or deployed on the LangGraph Platform.

---

## The Problem

Large language models hallucinate. In research contexts, this manifests in two distinct failure modes:

**Source-absent hallucination:** The model fabricates a plausible-sounding answer because it genuinely lacks the relevant knowledge — inventing a study, a statistic, or a quote that doesn't exist. This is what most people mean when they talk about LLM hallucination.

**Source-inconsistent hallucination:** The model _does_ encounter relevant source material but still synthesises it incorrectly — misquoting a number, reversing a causal claim, or overgeneralising from a single study. This is subtler and arguably more dangerous, because the source exists and looks legitimate. A RAG system that retrieves real documents doesn't prevent this: the model may still misread or ignore the retrieved content.

Current mitigations — RAG, chain-of-thought, temperature reduction — reduce the frequency of both failure modes but offer no formal guarantees against either, and provide no way for a downstream reader to independently verify that a citation says what the report claims it says.

---

## How MARA Is Different

| Property                  | Standard RAG Agent       | MARA                                                        |
| ------------------------- | ------------------------ | ----------------------------------------------------------- |
| Citations                 | URLs in footnotes        | SHA-256 Merkle leaf + URL + timestamp                       |
| Verification              | Trust the LLM            | Recompute hash against source                               |
| Hallucination detection   | Post-hoc / probabilistic | Structural (hash mismatch) + statistical (confidence score) |
| Source-absent halluc.     | None                     | ✅ HITL checkpoint surfaces low-confidence claims           |
| Source-inconsist. halluc. | None                     | ✅ HITL checkpoint + Beta-Binomial multi-source agreement   |
| Audit trail               | LLM output only          | Full Merkle tree + LangSmith trace                          |
| Reproducibility           | None                     | Same inputs → same root hash (deterministic)                |

---

## Architecture

The agent is implemented as a directed `StateGraph` in LangGraph. Control flows through twelve nodes, some of which fan out into parallel subgraphs using the `Send` API. The Merkle integrity layer is a data structure — not a separate node — that is built incrementally as sources are hashed and committed into the agent state.

```
┌────────────────────────────────────────────────────────────────┐
│                        MARA StateGraph                         │
│                                                                │
│  [Query Planner]                                               │
│       │                                                        │
│       ▼                                                        │
│  [search_worker ×N] ──┐  parallel fan-out via Send()          │
│  (brave → firecrawl)  │  one search_worker + one arxiv_worker  │
│  [arxiv_worker  ×N] ──┤  + one s2_worker dispatched per       │
│  [s2_worker     ×N] ──┘  sub-query (3 × max_workers total)   │
│       │  (arxiv: arxiv_search → firecrawl)                    │
│       │  (s2: semantic_scholar_search → raw_chunks directly)  │
│       ▼  (fan-in via operator.add on raw_chunks)              │
│  [Source Hasher]  →  [Merkle Builder] ◄────────────────┐      │
│                             │                           │      │
│                             ▼                           │      │
│                     [Retriever]                         │      │
│                      BM25 + semantic RRF → top-K leaves │      │
│                             │                           │      │
│                             ▼                           │      │
│                    [Claim Extractor]                    │      │
│                             │                           │      │
│                             ▼                           │      │
│                   [Confidence Scorer]                   │      │
│                             │                           │      │
│              ┌──────────────┴──────────────┐            │      │
│              │ route_after_scoring         │            │      │
│              ▼                            ▼            │      │
│  [Corrective Retriever]         [HITL Checkpoint]      │      │
│   (low conf + few leaves)        auto-approve high;    │      │
│   LLM sub-queries + scrape       interrupt() for low   │      │
│              │                            │            │      │
│              └────────────────────────────┘            │      │
│              (back-edge: corrective_retriever ──────────┘      │
│               skips source_hasher; re-enters merkle_builder)   │
│                                           │                    │
│                                           ▼                    │
│                                [Report Synthesizer]            │
│                                           │                    │
│                                [Certified Output] ◄─ root      │
└────────────────────────────────────────────────────────────────┘
```

LangSmith traces every node invocation, every LLM call, and every routing decision across the entire graph.

---

## Core Components

### 1. Query Planner

**Role:** Receives the user's natural language research question and decomposes it into a set of focused sub-queries that can be dispatched to the search workers independently.

**Why this matters:** A top-level question like _"What are the long-term economic effects of universal basic income?"_ is too broad for any single search. Breaking it into sub-queries — covering empirical pilots, theoretical models, criticisms, and regional case studies — gives the search workers tighter retrieval targets, which in turn improves the signal-to-noise ratio of retrieved chunks and makes the confidence scorer's job easier.

**Implementation note:** The planner instructs the LLM to return a bare JSON array, where each element has a `query` string and a `domain` hint. The response is parsed by `_parse_sub_queries`, which strips optional markdown fences and coerces field types. `SubQuery` is kept as a plain TypedDict (JSON-serialisable, LangGraph-checkpointable) — no parallel Pydantic model is needed at the LLM boundary.

**LangGraph pattern used:** A single node with a conditional edge that fans out to parallel search workers via the `Send` API.

---

### 2. Parallel Search Workers

**Role:** Execute retrieval for each sub-query simultaneously. For each sub-query, three parallel worker subgraphs are dispatched: a **web worker** (`search_worker`), an **ArXiv worker** (`arxiv_worker`), and a **Semantic Scholar worker** (`semantic_scholar_worker`). All three fan their chunks back into `MARAState.raw_chunks` via `operator.add`.

**`search_worker` subgraph** (`brave_search → firecrawl_scrape`):

1. **`brave_search`:** Calls the Brave Search API with the sub-query and collects results from all response sections: web results, news, discussions, and FAQ. Each result carries rich metadata — title, description, `extra_snippets` (up to 5 additional page excerpts), `page_age`, and `result_type`. This full result set fans back into `MARAState.search_results` via an `operator.add` reducer, making Brave's metadata available to every downstream node. Note that these results are **not hashed** — see the [Why Brave search results are not hashed](#why-brave-search-results-are-not-hashed) section for the full explanation.
2. **`firecrawl_scrape`:** Deduplicates the URLs from `search_results` and batch-scrapes each unique URL via Firecrawl, handling JavaScript rendering and anti-bot measures. The full page markdown is split into deterministic fixed-size chunks. These chunks — not Brave's snippets — are what get passed to the Source Hasher and committed to the Merkle tree. Includes a freshness cache check: URLs already in the leaf DB within their TTL are served from cache without re-scraping (see [Freshness cache](#freshness-cache)).

**`arxiv_worker` subgraph** (`arxiv_search → firecrawl_scrape`):

1. **`arxiv_search`:** Queries the ArXiv API (`export.arxiv.org/api/query`) with the sub-query and returns up to `arxiv_max_results` papers (default: 5) ranked by relevance. Uses the versioned PDF URL (e.g. `/pdf/2405.01234v2`) as the canonical source URL — versioned PDFs are immutable and satisfy the Merkle reproducibility requirement. The paper abstract is stored in `SearchResult.description` for HITL observability but is **never hashed** (only the full PDF text is committed to the tree).
2. **`firecrawl_scrape`:** Reuses the same scrape node as the web worker. Firecrawl fetches the full PDF text and splits it into chunks identical to how web pages are handled. This means ArXiv papers receive exactly the same Merkle treatment as any other source — full text committed, hash verifiable. Versioned ArXiv PDFs use `float('inf')` TTL — see [Freshness cache](#freshness-cache).

**`semantic_scholar_worker` subgraph** (`semantic_scholar_search`):

1. **`semantic_scholar_search`:** Calls the Semantic Scholar `/snippet/search` endpoint and converts each result directly to a `SourceChunk`. No Firecrawl scraping step is needed — the API returns ~500-word text excerpts drawn from paper titles, abstracts, and body text, already chunked to a size suitable for embedding and retrieval. Returns up to `semantic_scholar_max_results` (default: 5) snippets per sub-query. Each chunk uses `https://www.semanticscholar.org/paper/CorpusId:{id}` as its canonical URL — a stable, resolvable identifier.

   Set `S2_API_KEY` in your `.env` to avoid shared rate limits. Without a key the endpoint is accessible but subject to lower throughput. The node enforces ≤ 1 RPS across all concurrent sub-query workers via an `asyncio.Lock`-based rate limiter (`_acquire_s2_slot()`), honouring the Semantic Scholar per-key limit. HTTP errors are handled gracefully: 401/403 are logged at ERROR with a key check reminder; 429 and other transient failures are logged at WARNING and return empty chunks rather than crashing the pipeline.

A citation graph traversal strategy — crawling reference links discovered during scraping — is planned but not yet implemented. See `agent/nodes/search_worker/graph.py`.

**Why run them in parallel?** LLM-based agents spend most of their wall-clock time waiting on I/O — search APIs, scrape requests, LLM inference. Running multiple sub-queries simultaneously collapses this latency dramatically.

**LangGraph pattern used — the `Send` API:** LangGraph's `Send` API enables fan-out by dispatching one `Send` object per sub-query per worker type. Each sub-query produces three `Send` objects — one for each worker type — giving `3 × max_workers` concurrent workers per run.

```python
# agent/edges/routing.py
def dispatch_search_workers(state: MARAState) -> list[Send]:
    sends = []
    for q in state["sub_queries"]:
        payload = {"sub_query": q, "research_config": state["config"],
                   "search_results": [], "raw_chunks": []}
        sends.append(Send("search_worker", payload))
        sends.append(Send("arxiv_worker", payload))
        sends.append(Send("semantic_scholar_worker", payload))
    return sends

builder.add_conditional_edges("query_planner", dispatch_search_workers,
                              ["search_worker", "arxiv_worker", "semantic_scholar_worker"])
```

**Subgraph checkpointing:** Each worker subgraph is compiled with `checkpointer=True`, inheriting the parent checkpointer. If a scrape request times out mid-retrieval, LangGraph resumes the worker from `firecrawl_scrape` rather than re-running `brave_search` from scratch.

---

### 3. Source Hasher

**Role:** Receives the raw retrieved chunks from all workers and computes a canonical hash for each one, building up the set of Merkle leaves.

**What gets hashed:** For each source chunk, the hash input is a deterministic serialisation of:

```python
import json, hashlib

def canonical_serialise(url: str, text: str, retrieved_at: str) -> bytes:
    """Deterministic JSON serialisation using stdlib only.
    sort_keys=True ensures key order is stable across Python versions.
    separators=(',', ':') strips all whitespace.
    ensure_ascii=True guarantees byte-identical output regardless of locale.
    """
    payload = {"retrieved_at": retrieved_at, "text": text, "url": url}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

leaf_hash = hashlib.sha256(canonical_serialise(url, text, retrieved_at)).hexdigest()
```

This function lives in `merkle/hasher.py`. The determinism guarantee means that if you retrieve the same chunk twice — in two different sessions, on two different machines — you will always get the same hash, assuming the source page content hasn't changed.

**Hash algorithm:** SHA-256, producing a 64-character hex digest. This is stored as a `MerkleLeaf`:

```python
class MerkleLeaf(TypedDict):
    url: str               # source URL
    text: str              # raw chunk text (what was hashed)
    retrieved_at: str      # ISO-8601 timestamp
    hash: str              # SHA-256 hex digest of canonical_serialise(url, text, retrieved_at)
    index: int             # zero-based position in merkle_leaves
    sub_query: str         # originating sub-query (observability/attribution)
    contextualized_text: str  # embedding text; equals text until Contextual Retrieval is added
```

`contextualized_text` is a forward-compatibility stub: when Contextual Retrieval is implemented (prepending an LLM-generated document-level summary to each chunk before embedding), only `source_hasher.py` and the embedding call need to change — the `retriever` node automatically improves without modification.

**Why not hash the entire page?** Research agents retrieve _chunks_, not full documents. Hashing the full page would make the leaf hash sensitive to parts of the page the agent never read, and would break the 1:1 relationship between a leaf and the specific text a claim was derived from.

---

### 4. Merkle Integrity Layer

**Role:** This is not a LangGraph node — it is a data structure managed within the agent state. As the Source Hasher produces leaves, the Merkle builder computes the tree bottom-up, producing an intermediate hash for every parent node and ultimately a single root hash.

#### How the Merkle tree is built

A Merkle tree is a binary hash tree where each leaf is the hash of a data item, and each internal node is the hash of its two children concatenated:

```
parent_hash = SHA-256(left_child_hash + right_child_hash)
```

If the number of leaves is odd, the last leaf is duplicated to make the tree balanced.

```
         ROOT
        /    \
      H(1,2)  H(3,3)   ← H(3,3) because leaf 3 was duplicated
      /  \      |
   H(1)  H(2)  H(3)    ← SHA-256 of each source chunk
    │     │     │
  src1  src2  src3
```

The root hash is a cryptographic commitment to the entire set of retrieved sources. If any leaf is changed — even a single character in a source chunk — the root hash changes completely (avalanche effect).

#### Why a Merkle tree instead of a flat list of hashes?

A flat list of hashes would require checking every hash to verify a single claim. A Merkle tree provides **efficient membership proofs**: to prove that claim X came from source chunk Y, you only need to provide the sibling hashes along the path from the leaf to the root (a **Merkle proof**). For a tree with 1,024 leaves, this is 10 hash operations rather than 1,024. At scale — a deep research session with hundreds of sources — this matters.

A Merkle proof for leaf `i` consists of the sibling hash at each level of the tree from leaf to root. The verifier recomputes the root from the leaf hash + the proof path and checks that it matches the published root. Implementation: `merkle/proof.py → generate_merkle_proof(tree, leaf_index)`.

---

### 5. Hybrid Retriever

**Role:** Selects the top-K most relevant Merkle leaves from the full scraped corpus before passing them to the Claim Extractor. This is the critical step that prevents context-window overflow: a typical research session scrapes 2,000–5,000 chunks (~1.4M tokens), far exceeding the LLM's context limit. The retriever reduces this to `max_claim_sources` (default: 50) leaves, ~15K tokens.

**Why retrieval is necessary:** The full scraped corpus can easily exceed any LLM's context window. Without a retrieval step, the claim extractor would receive all scraped content and either overflow or require a model with a prohibitively large context window. The retriever selects only the most evidentially relevant chunks, making claim extraction tractable regardless of corpus size.

**How it works — two-stage strategy:**

**Stage 1 — Embedding cache.** Leaf embeddings are loaded from the SQLite DB when available. Only leaves with no stored embedding — or whose cached blob has a dimension mismatch (model changed between runs) — are passed through the SentenceTransformer model. New embeddings are written back immediately so subsequent runs on the same corpus skip inference entirely.

**Stage 2 — Hybrid Reciprocal Rank Fusion (RRF).** BM25 (FTS5, keyword match) and semantic (cosine similarity) rankings are fused using RRF (k=60). BM25 captures exact-match / named-entity evidence; semantic captures paraphrases and synonym matches. RRF combines both without requiring weight tuning:

```
RRF_score(d) = 1/(60 + r_sem(d) + 1) + 1/(60 + r_bm25(d) + 1)
```

Leaves absent from BM25 results receive a penalty rank of `len(leaves)`, keeping their BM25 contribution small but non-zero.

```python
# agent/nodes/retriever.py (simplified)
async def retriever(state: MARAState, config: RunnableConfig) -> dict:
    leaves = state["merkle_leaves"]
    k = state["config"].max_claim_sources
    query_texts = [state["query"]] + [sq["query"] for sq in state["sub_queries"]]

    # Query embeddings establish target_dim for cache dimension validation.
    query_embs = await asyncio.to_thread(embed, query_texts, model_name)
    target_dim = query_embs.shape[1]

    # Leaf embeddings: served from SQLite cache where possible.
    leaf_embs = await _load_or_compute_leaf_embeddings(
        leaves, model_name, target_dim, leaf_repo
    )
    semantic_scores = (leaf_embs @ query_embs.T).max(axis=1)
    semantic_order = np.argsort(semantic_scores)[::-1]

    # Hybrid: BM25 via FTS5, fused with semantic via RRF.
    bm25_results = await asyncio.to_thread(leaf_repo.bm25_search, state["query"], run_id, len(leaves))
    bm25_hash_rank = {r["hash"]: i for i, r in enumerate(bm25_results)}
    rrf = _rrf_scores(leaves, semantic_scores, semantic_order, bm25_hash_rank)
    top_indices = np.argsort(rrf)[::-1][:k]
    return {"retrieved_leaves": [leaves[int(i)] for i in top_indices]}
```

**Pure-semantic fallback:** When `leaf_db_enabled=False` (or `leaf_repo` is not injected), the retriever falls back to cosine-similarity-only ranking. All existing behaviour is preserved without requiring a database.

**Integrity separation:** The Merkle tree commits to _all_ scraped leaves — retrieval does not affect which sources are recorded. The retriever only determines which subset of those committed sources is used for claim extraction. This keeps the two layers orthogonal: the Merkle root proves what was scraped; `retrieved_leaves` records what was used for reasoning.

**Embedding model:** Uses `confidence.embeddings.embed()` with the configured `embedding_model` (default: `all-MiniLM-L6-v2`). The retriever shares the same model as the Confidence Scorer, so there is only one model load at startup.

**LangGraph pattern used:** Standard sequential node. CPU-bound work (`embed()`, `bm25_search()`) runs via `asyncio.to_thread`; synchronous SQLite cache reads execute on the event loop thread (fast, < 5 ms).

---

### 6. Confidence Scorer

**Role:** For each factual claim extracted from the retrieved evidence, computes a confidence score estimating the probability the claim is genuinely grounded in the retrieved sources.

#### The scoring model

The confidence score for claim `c` is the **Beta-Binomial posterior mean** updated only on corroborating leaves:

```
confidence(c) = SA(c) = (1 + k) / (2 + k)
```

where `k` is the number of *unique source URLs* in the full `merkle_leaves` corpus with at least one chunk whose cosine similarity to the claim exceeds `similarity_support_threshold` (default τ = 0.60).

Source deduplication enforces the independence assumption of the Beta-Binomial model. A single article split into 15 chunks would otherwise increment `k` by 15 — one source's opinion repeated, not independent corroboration. Counting unique URLs means a market research article chunked 20 times still casts exactly one vote.

Non-corroborating sources — those with no chunk above the threshold — are excluded from the denominator entirely. Their silence is not evidence of contradiction; they are simply irrelevant to that claim. Only positive corroboration moves the score.

This is the Laplace-smoothed estimator from a `Beta(1, 1)` prior (uniform / maximum uncertainty):

- 0 corroborating sources → SA = 0.5 (prior mean; maximum uncertainty)
- 1 corroborating source  → SA = 0.67
- 3 corroborating sources → SA = 0.80 (clears `high_confidence_threshold`)
- k → ∞                  → SA → 1.0 (never exactly reached)

The score is always in the open interval (0, 1). Similarities are computed between `SentenceTransformer("all-MiniLM-L6-v2")` embeddings of the claim and every leaf's `contextualized_text` across the full `merkle_leaves` pool — not just the `retrieved_leaves` subset passed to the claim extractor. Scoring against the full corpus gives every claim the best possible evidence test; a leaf not selected for extraction may still be the strongest corroboration for a given claim.

#### Routing based on confidence

After scoring, `route_after_scoring` distinguishes two low-confidence cases:

| Condition | Interpretation | Route |
|---|---|---|
| confidence ≥ `high_confidence_threshold` | Well-supported | auto-approve at HITL |
| confidence < `low_confidence_threshold` AND `n_unique_urls < n_leaves_contested_threshold` | **Insufficient data** — not enough sources | `corrective_retriever` (if loops remain) |
| confidence < `low_confidence_threshold` AND `n_unique_urls ≥ n_leaves_contested_threshold` | **Contested** — sources exist but disagree | `hitl_checkpoint` (flagged `contested=True`) |

```
confidence ≥ 0.80                          →  auto-approved at HITL
confidence < 0.55, few leaves, loops left  →  corrective_retriever → re-score
confidence < 0.55, many leaves             →  hitl_checkpoint (contested)
loop cap reached                           →  hitl_checkpoint
```

---

### 7. Corrective Retriever

**Role:** When claims have low confidence due to _insufficient data_ (few corroborating leaves, not genuine source disagreement), this node acquires targeted new evidence and re-runs the full scoring pipeline on the expanded leaf pool.

**Step 1 — Identify failing claims:** Claims where `confidence < low_confidence_threshold` AND `n_unique_urls < n_leaves_contested_threshold`. Contested claims (large `n_unique_urls`, low confidence) are routed directly to HITL — they have enough data, the sources just disagree.

**Step 2 — Generate corrective sub-queries:** The LLM generates 1–2 targeted search queries per failing claim, focused on finding specific supporting or contradicting evidence.

**Step 3 — DB-first retrieval:** Before scraping, the node queries the leaf database with BM25 full-text search across _all previous runs_ (not just the current one). Cross-run retrieval amortises scraping cost: a claim that appeared in a previous session may already have ample evidence in the corpus.

**Step 4 — Live scraping (if DB insufficient):** If the DB yields fewer than `3 × len(failing)` new leaves, Brave + Firecrawl fire for each corrective sub-query. New pages are capped at `max_new_pages_per_round` per sub-query to limit credit burn. New leaves are upserted to the DB and linked to the current run.

**Step 5 — Re-enter at `merkle_builder`:** The node appends new leaves directly to `merkle_leaves` and returns, routing to `merkle_builder` via a back-edge. The full pipeline (`merkle_builder → retriever → claim_extractor → confidence_scorer`) re-runs on the expanded corpus. `source_hasher` is skipped — corrective leaves are pre-hashed here.

The loop repeats up to `max_corrective_rag_loops` times (default: 2). After the cap, all remaining low-confidence claims go to HITL regardless.

---

### 8. HITL Checkpoint

**Role:** When claims have been through all corrective rounds (or were contested from the start), the graph pauses using LangGraph's `interrupt()` mechanism and surfaces them for human review.

**Contested flagging:** Before presenting claims for review, any claim with `confidence < low_confidence_threshold` AND `n_unique_urls ≥ n_leaves_contested_threshold` is flagged `contested=True`. This signals to the reviewer that sources were found but disagree — not that evidence was simply absent.

**Confidence stats:** At entry, the node logs mean, median, std dev, min, and max confidence across all scored claims, giving an at-a-glance picture of the overall evidence quality before the review begins.

The checkpoint presents each claim below `high_confidence_threshold` with its confidence score and source indices. The human enters a comma-separated list of indices to approve; the rest are dropped.

```python
# agent/nodes/hitl_checkpoint.py (simplified)
def hitl_checkpoint(state: MARAState) -> dict:
    high_threshold = state["config"].high_confidence_threshold
    auto_approved = [c for c in state["scored_claims"]
                     if c.confidence >= high_threshold]
    needs_review  = [c for c in state["scored_claims"]
                     if c.confidence <  high_threshold]

    if not needs_review:
        return {"human_approved_claims": auto_approved}

    decision = interrupt({
        "needs_review": [{"index": i, "text": c.text, "confidence": c.confidence,
                           "corroborating": c.corroborating, "n_leaves": c.n_leaves,
                           "n_unique_urls": c.n_unique_urls,
                           "source_indices": c.source_indices, "contested": c.contested}
                         for i, c in enumerate(needs_review)],
        "auto_approved_count": len(auto_approved),
    })
    # Resume: Command(resume={"approved_indices": [0, 2]})
    human_approved = [needs_review[i] for i in decision["approved_indices"]
                      if i < len(needs_review)]
    return {"human_approved_claims": auto_approved + human_approved}
```

This is implemented using LangGraph's `interrupt()` + `Command(resume=...)` pattern, which persists the full graph state to the configured checkpointer so the graph can be resumed hours or days later — even after the server restarts. The durable execution guarantee comes from LangGraph's checkpointing infrastructure.

---

### 9. Report Synthesizer

**Role:** Receives the approved claims along with the `retrieved_leaves` set and synthesises a coherent prose report with inline citations. Uses `human_approved_claims` when the HITL node ran; falls back to `scored_claims` when `human_approved_claims` is `None` (HITL never executed — all claims were within the corrective loop and none reached the checkpoint).

Each citation in the report is not a simple URL — it is a structured reference that includes the Merkle leaf index:

```
[1:H(src1)] → Merkle leaf index 1, hash H(src1)
```

The synthesizer is prompted to reference claims by their leaf index rather than by URL alone. This means the final report contains explicit pointers into the Merkle tree, making every sentence that makes a factual claim independently verifiable.

**Key constraint:** The synthesizer is instructed _not_ to introduce any new factual claims that are not present in the approved, scored claim set. It may only rephrase, combine, and structure claims that have already passed the confidence filter. This is enforced by including the scored claim set in the prompt context and explicitly prohibiting the addition of "background knowledge."

---

### 10. Certified Output

**Role:** The final node. It assembles the research report, the complete Merkle tree (all leaf hashes, all intermediate hashes, the root hash), the Merkle proofs for each cited leaf, and the full LangSmith run URL into a single `CertifiedReport` object.

```python
@dataclass
class CertifiedReport:
    query: str                       # the original research question
    report_text: str                 # prose with inline [ML:index:hash] citations
    merkle_root: str                 # 64-char SHA-256 hex digest of the full leaf set
    leaves: list[MerkleLeaf]         # retrieved leaves used as evidence (not all scraped)
    scored_claims: list[ScoredClaim] # human-approved claims with confidence scores
    generated_at: str                # ISO-8601 timestamp (auto-set at construction)
```

`leaves` contains the `retrieved_leaves` subset — the sources actually used for claim extraction and synthesis. The full scraped corpus (all `merkle_leaves`) is committed to the Merkle tree and available in state, but the report records only the evidence that was actively used for reasoning.

The `merkle_root` is the definitive identifier for this research session. Two reports with the same Merkle root were produced from exactly the same set of source chunks. Two reports with different Merkle roots, even if they look identical in prose, were produced from different source sets.

---

## The Merkle Integrity Protocol

The full verification lifecycle for a MARA report:

**At report generation time (inside the agent):**

1. Each source chunk is hashed → produces a leaf hash.
2. Leaf hashes are assembled into a balanced Merkle tree.
3. A Merkle proof is generated for each leaf cited in the report.
4. The Merkle root is embedded in the report header.
5. All of the above is serialised into the `CertifiedReport`.

**At verification time (any reader, any time):**

1. The reader selects a claim in the report and reads its leaf index and leaf hash.
2. The reader retrieves the cited source URL and the recorded chunk text from the `CertifiedReport.leaves` array.
3. The reader independently recomputes `SHA-256(canonical_serialise(url, text, retrieved_at))` using the stdlib serialiser in `merkle/hasher.py`.
4. If the recomputed hash matches the leaf hash: the chunk hasn't been altered.
5. The reader uses the Merkle proof path to recompute the root hash from the leaf hash.
6. If the recomputed root matches `CertifiedReport.merkle_root`: the entire source set is intact.

**What verification proves and what it doesn't:**

| Verification succeeds →                                      | Verification fails →                                              |
| ------------------------------------------------------------ | ----------------------------------------------------------------- |
| The source chunk existed at the URL at the recorded time     | The source chunk was altered (by the agent or by the source page) |
| The agent did not fabricate the text of the citation         | The leaf hash or report was tampered with post-generation         |
| The report's root hash was produced from these exact sources | A different source set was used than claimed                      |

What it does **not** prove: that the source itself is accurate, or that the agent's interpretation of the source is correct. Those are addressed by the confidence scorer and the HITL checkpoint respectively.

Also note: step 3 above references `canonical_serialise(url, text, retrieved_at)` — the stdlib implementation in `merkle/hasher.py` using `json.dumps(sort_keys=True, separators=(',', ':'))`. The verifier must use the same serialisation to reproduce the correct hash.

---

## Leaf Database

MARA persists every scraped leaf to a local SQLite database (`~/.mara/leaves.db` by default). This gives the pipeline three concrete benefits across sessions: a freshness cache that avoids re-scraping unchanged pages, a growing corpus that improves retrieval coverage over time, and an embedding cache that skips SentenceTransformer inference for leaves whose vector representations are already stored.

The database is implemented as a repository pattern (`mara/db/`). The `LeafRepository` Protocol defines the interface; `SQLiteLeafRepository` is the concrete implementation. A Postgres implementation would need only a new file — no callers change.

### Freshness cache

Before calling the Firecrawl API, `firecrawl_scrape` checks the database for each URL using a **source-type-aware TTL**. The TTL is computed by `url_ttl_hours(url, default)` in `agent/nodes/search_worker/url_ttl.py`, which classifies URLs into four tiers:

| Tier | TTL | Applies to |
|---|---|---|
| Immutable | `float('inf')` | Versioned ArXiv PDFs (`/abs/XXXXvN`, `/pdf/XXXXvN`), DOI-resolved URLs (`doi.org/10.…`) |
| Long-lived | 8 760 h (1 year) | Direct `.pdf` links; major publisher domains: `nature.com`, `science.org`, `cell.com`, `ncbi.nlm.nih.gov`, `pubmed.ncbi`, `nejm.org`, `thelancet.com`, `jamanetwork.com` |
| Semi-stable | 720 h (30 days) | Wikipedia, `semanticscholar.org`, `researchgate.net` |
| Default | 336 h (14 days) | Everything else (controlled by `leaf_cache_max_age_hours`) |

The `float('inf')` sentinel is understood by `SQLiteLeafRepository.get_fresh_leaves_for_url`: when TTL is infinite the cutoff comparison is skipped entirely and any cached leaves for the URL are returned unconditionally. Versioned ArXiv PDFs and DOI pages are frozen by design — they never need to be re-scraped.

```python
# firecrawl_scrape.py (simplified)
for url in urls:
    ttl = url_ttl_hours(url, research_config.leaf_cache_max_age_hours)
    fresh = leaf_repo.get_fresh_leaves_for_url(url, ttl)
    if fresh:
        # Use cached chunks, skip Firecrawl entirely for this URL
        cached_chunks.extend(fresh)
    else:
        urls_to_scrape.append(url)
```

### Cross-session corpus growth

Every run's leaves are linked to the run via a `run_leaves` join table. This many-to-many design means a leaf can appear across multiple runs if the page is unchanged within the freshness window. Over time the corpus grows organically: each new query that touches different pages adds leaves to the pool. The retriever's BM25 and semantic rankings are scoped to the current run's leaf set, so older leaves from unrelated runs don't pollute retrieval.

### Embedding persistence

After the retriever computes embeddings for a set of leaves, it writes them back to the database as packed `float32` blobs. On subsequent runs, cached blobs are deserialized with `np.frombuffer` and used directly — no SentenceTransformer inference needed. If the configured `embedding_model` changes, a dimension mismatch triggers transparent re-embedding and overwrites the stale blob.

### Schema overview

```
runs(run_id, query, merkle_root, embedding_model, hash_algorithm, started_at, completed_at)
leaves(hash, url, text, retrieved_at, contextualized_text, embedding BLOB, embedding_model, parent_hash)
run_leaves(run_id, leaf_hash, position_index, sub_query)   ← many-to-many join

leaves_fts   ← FTS5 virtual table (porter-ascii tokeniser, auto-sync triggers)
```

FTS5 provides BM25 ranking natively; three auto-sync triggers (INSERT / UPDATE / DELETE on `leaves`) keep the virtual table in sync without manual maintenance. The `parent_hash` column is nullable and reserved for future parent-child chunking without requiring a schema migration.

The run lifecycle spans the pipeline: `create_run()` at startup → `upsert_leaves()` + `link_leaves_to_run()` after hashing → `complete_run(run_id, merkle_root)` after certified output.

### Disabling the database

Set `leaf_db_enabled=False` (or `MARA_LEAF_DB_ENABLED=false` in `.env`) to disable all DB reads and writes. When disabled:

- `firecrawl_scrape` skips the freshness cache check entirely.
- `source_hasher` skips `upsert_leaves` / `link_leaves_to_run`.
- `certified_output` skips `complete_run`.
- The retriever falls back to pure cosine-similarity ranking.
- No `SQLiteLeafRepository` is created; no file is opened.

This is the default in CI and unit tests, and requires no code changes — only the environment variable.

---

## Retrieval Strategy

### Current approach

MARA runs three parallel search pipelines per sub-query:

1. **Web pipeline** (`brave_search → firecrawl_scrape`): Brave Search returns ranked URLs from its independent crawl index; Firecrawl fetches and chunks the full page text. Brave's snippets are not hashed — see [Why Brave search results are not hashed](#why-brave-search-results-are-not-hashed).
2. **ArXiv pipeline** (`arxiv_search → firecrawl_scrape`): The ArXiv API returns versioned PDF URLs; Firecrawl fetches and chunks the full PDF text. Versioned ArXiv PDFs have `float('inf')` TTL and are never re-scraped after the first fetch.
3. **Semantic Scholar pipeline** (`semantic_scholar_search`): The Semantic Scholar `/snippet/search` endpoint returns ~500-word body-text excerpts directly — no scraping required. Snippets are converted to `SourceChunk` objects in place and committed to the Merkle tree like any other source.

Separating search and scraping into distinct nodes means their failure modes, retry logic, and LangSmith traces are independent. Swapping out any provider does not touch the other nodes.

### Why Brave search results are not hashed

Brave results — titles, descriptions, `extra_snippets`, `page_age`, `result_type` — flow into `MARAState.search_results` and are available to every downstream node, but they are **never committed to the Merkle tree**. This is a hard requirement of the integrity protocol, not an oversight.

The Merkle integrity guarantee is: _"at any future time, you can recompute `SHA-256(canonical_serialise(url, text, retrieved_at))` and verify the hash matches."_ That guarantee requires the hashed bytes to be **reproducible from first principles** — meaning you fetch the source URL yourself and chunk the raw text yourself. Brave's metadata fails this test on two counts:

1. **Provider-controlled, not reproducible.** Brave's snippets and titles are Brave's own extractions of the page. Brave re-crawls, updates its extraction logic, and changes its ranking signals continuously. The same URL queried tomorrow returns different `extra_snippets` and potentially a different `description`. A hash of Brave's output would produce false verification failures — not because anything was tampered with, but because the provider changed its summary.

2. **Not the authoritative source text.** A Brave snippet is a third party's excerpt of a page, not the page itself. Hashing the snippet would only prove that Brave said a page contained certain text at query time — not that the page actually said it. The entire point of Merkle-backed citations is to bind claims to the raw source bytes that the agent read, so that a reader can independently verify the source says what the report claims. Only the full scraped text, retrieved and chunked by MARA itself, satisfies this.

Brave's data is genuinely useful for everything **except** integrity commitments: pre-screening URLs before scraping, surfacing recency signals (`page_age`) in citations, providing context to the Confidence Scorer and Report Synthesizer, and giving the HITL reviewer richer source context. It is intentionally preserved in state for these purposes. The constraint is narrowly scoped: Brave metadata informs the agent but cannot serve as a source of record for claim attribution.

### The chunking constraint

For Merkle integrity to hold, chunking must be **deterministic and reproducible**: the same source page must always be split into the same chunks. This rules out chunking strategies that depend on model tokenisation windows, probabilistic sentence boundary detection, or any state not encoded in the source text itself.

The current approach uses fixed-size character chunking with overlap, implemented in `agent/nodes/search_worker/firecrawl_scrape.py`. The chunk size and overlap are configurable parameters. This is the simplest chunking strategy that satisfies the determinism requirement.

### Hybrid retrieval over the scraped corpus

After all source pages are scraped and hashed, the `retriever` node selects the top `max_claim_sources` (default: 50) leaves using **Reciprocal Rank Fusion (RRF)** over two signals: BM25 keyword ranking (via SQLite FTS5) and dense semantic ranking (cosine similarity over SentenceTransformer embeddings). This prevents context-window overflow in the Claim Extractor: a typical session produces 2,000–5,000 chunks (~1.4M tokens), far exceeding any LLM's practical context limit.

BM25 excels at precise factual queries — exact entity names, statistics, quoted phrases — where a keyword hit is strong evidence of relevance. Semantic retrieval captures paraphrases, synonyms, and contextually related content that keyword matching misses. RRF (k=60, from Cormack, Clarke & Buettcher 2009) fuses the two rankings without requiring weight tuning: each document's score is the sum of its reciprocal ranks in both lists, so a document ranked highly by both signals rises significantly above documents ranked highly by only one.

Both signals score leaves against the main query and all sub-queries, with the max taken across query texts. This captures aspect-specific relevance: a leaf highly relevant to one sub-query should not be penalised by its distance from the others.

**Integrity–retrieval separation:** The Merkle tree is built from _all_ scraped leaves before retrieval runs. Retrieval only determines which subset of committed leaves is used for reasoning — the two layers are orthogonal. The Merkle root proves what was scraped; `retrieved_leaves` records what was used for claim extraction.

### Planned: local corpus retrieval

For research over a pre-indexed private corpus (internal documents, downloaded papers, licensed databases), a vector store retrieval step can be added as a third retrieval strategy in the search worker subgraph. The same determinism constraint applies: chunks stored in the vector index must be produced by the same chunking function used at hash time.

### Retrieval experimentation

The retrieval pipeline is one of the primary areas for experimentation as the project develops.

**Implemented:**

- **Hybrid BM25 + semantic RRF:** Dense (embedding) retrieval and sparse (FTS5/BM25) retrieval fused via Reciprocal Rank Fusion. Outperforms either alone, especially for precise factual queries where exact term match matters.
- **Embedding cache:** Leaf embeddings persisted as `float32` blobs in SQLite; loaded on subsequent runs to skip SentenceTransformer inference. Includes dim-mismatch detection for model changes.

**Worth evaluating next:**

- **Contextual retrieval:** Before embedding each chunk, prepend a short LLM-generated summary of where the chunk sits within the broader document. Improves retrieval precision for long documents where individual chunks lose their referential context. The `contextualized_text` field on `MerkleLeaf` is a forward-compatibility stub designed for this.
- **HyDE (Hypothetical Document Embeddings):** Instead of embedding the query directly, generate a hypothetical answer to the query and embed that. The hypothesis embedding is often closer in vector space to real answer documents than the raw question embedding is.

Any retrieval improvement must still produce deterministic chunks that can be hashed and committed to the Merkle tree. Retrieval quality improvements affect which chunks are found; the hash commitment records exactly which chunks were used, whatever the strategy.

---

## Statistical Confidence Scoring

### Model framing

The confidence score is the Beta-Binomial posterior mean (SA), covered in full in the Confidence Scorer node section above. SA is updated with `k` successes only — non-corroborating leaves are excluded from the denominator because their silence is not evidence of contradiction. The posterior mean `(1 + k) / (2 + k)` is equivalent to starting from a `Beta(1, 1)` prior (uniform, maximum uncertainty) and observing `k` corroborations with no denominator penalty for irrelevant leaves. This is a principled, well-established Bayesian estimator that requires no training data and produces appropriate uncertainty for small sample sizes.

### Claim extraction

Before scoring, an LLM extraction step converts raw retrieved text into a structured set of atomic claims. Each claim is:

- A single factual assertion (not a compound statement)
- Tagged with the source chunk indices it was extracted from
- Stored with its original sentence span for attribution

Atomic claim extraction is important because compound statements (e.g., _"Study X found Y, and also concluded Z"_) can have mixed support — Y might be well-supported but Z might not. Treating them as one claim would average away a real signal.

**Resilience against connection errors:** LLM inference providers occasionally drop large requests mid-response (`RemoteProtocolError`). The claim extractor catches `httpx.HTTPError` and retries with a halved leaf window, repeating until the call succeeds or the window would fall below `claim_extractor_min_leaves` (default: 10). This prevents a single flaky inference call from crashing the entire pipeline.

---

## LangGraph Implementation Details

### State schema

The graph is typed using Python `TypedDict` to ensure every node receives and returns well-formed state:

```python
import operator
from typing import Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class MARAState(TypedDict):
    # ---- Input ----
    query: str
    run_date: str          # YYYY-MM-DD UTC date, set at pipeline start
    config: ResearchConfig

    # ---- Planner output ----
    sub_queries: list[SubQuery]

    # ---- Search worker fan-in (parallel reduce) ----
    search_results: Annotated[list[SearchResult], operator.add]  # all Brave results
    raw_chunks: Annotated[list[SourceChunk], operator.add]       # all scraped chunks

    # ---- Source Hasher / Merkle output ----
    merkle_leaves: list[MerkleLeaf]    # all scraped + hashed leaves
    merkle_tree: MerkleTree | None

    # ---- Retriever output ----
    retrieved_leaves: list[MerkleLeaf] # top-K leaves selected by cosine similarity

    # ---- Claim extraction / scoring ----
    extracted_claims: list[Claim]
    scored_claims: list[Any]           # list[ScoredClaim] dataclass

    # ---- HITL output ----
    human_approved_claims: list[Any] | None  # None until HITL runs; [] if HITL ran but approved nothing

    # ---- Synthesizer output ----
    report_draft: str

    # ---- Final output ----
    certified_report: CertifiedReport | None

    # ---- Corrective RAG ----
    corrective_sub_queries: list[SubQuery]   # LLM-generated queries per failing claim

    # ---- Internal ----
    messages: Annotated[list, add_messages]
    loop_count: int  # corrective RAG iteration counter
```

The `Annotated[list[SourceChunk], operator.add]` type on `raw_chunks` is key: it tells LangGraph to _merge_ the lists returned by parallel search workers rather than overwriting the state with each worker's result. This is the standard LangGraph pattern for fan-in after a `Send`-based fan-out.

### Checkpointing and durability

The graph is compiled with a `PostgresSaver` checkpointer in production and a `MemorySaver` for local development. Every node completion writes a checkpoint — if the process dies mid-run, the graph resumes from the last successful node. No sources need to be re-fetched. See `agent/graph.py` for the checkpointer setup. The `checkpointer` config parameter controls which backend is used (`memory` or `postgres`).

### Subgraph isolation

Each search worker runs as a compiled subgraph with `checkpointer=True`, inheriting the parent checkpointer. This enables granular replay: you can re-run just the retrieval phase of a failed run without re-running the planner, or fork from after retrieval to test a different confidence threshold. See `agent/nodes/search_worker/graph.py`.

---

## LangSmith Observability

Every node in the MARA graph is automatically traced by LangSmith when the `LANGCHAIN_TRACING_V2=true` environment variable is set. No additional instrumentation is required.

### What LangSmith captures

- **Per-node traces:** Input state, output state, latency, token counts, and any errors, for every node in the graph.
- **LLM call traces:** Every individual LLM call within a node, including the exact prompt, model parameters, and response.
- **Tool call traces:** Every search API call, Firecrawl scrape, and hash computation.
- **Routing decisions:** Which branch of a conditional edge was taken, and what state triggered it.
- **Human-in-the-loop events:** The exact payload sent to the human reviewer and the decision returned.

### Nested LLM traces via config forwarding

LangGraph injects metadata into the `config: RunnableConfig` it passes to every node at runtime:

```python
config["metadata"]["langgraph_node"]      # e.g. "query_planner"
config["metadata"]["langgraph_step"]      # execution step number
config["metadata"]["langgraph_triggers"]  # what triggered this node
config["metadata"]["langgraph_checkpoint_ns"]
```

Every node that makes an LLM call forwards this `config` to `ainvoke` / `invoke`:

```python
# query_planner — LangSmith nests this span under "query_planner" step
response = await llm.ainvoke(messages, config)

# confidence_scorer — embed() call is nested under "confidence_scorer" step
response = llm.invoke(messages, config)
```

This produces a clean tree in LangSmith: graph run → node → LLM call, with step numbers and node names surfaced on every span. Tags and metadata passed at graph-invocation time propagate automatically to all child spans.

### Tags and metadata at invocation time

Pass `tags` and `metadata` in the top-level `config` dict when invoking the graph. They are inherited by every node and every LLM call inside the run:

```python
graph.invoke(
    {"query": "...", "config": research_config},
    config={
        "tags": ["production", "v0.1.0"],
        "metadata": {
            "user_id": "user-123",
            "session_id": "sess-456",
            "environment": "production",
        },
        "run_name": "mara-research-run",
    },
)
```

### Python logger (`mara.logging`)

LangSmith traces all LLM calls. For the non-LLM parts of the pipeline — chunking, hashing, tree construction — MARA uses a standard Python logger hierarchy rooted at `mara`:

```python
# In any mara module:
from mara.logging import get_logger
_log = get_logger(__name__)  # → e.g. "mara.agent.nodes.source_hasher"

_log.info("Hashing %d chunks with %s", n, algorithm)
```

Configure the root logger at application startup to control verbosity for the entire pipeline:

```python
import logging
logging.getLogger("mara").setLevel(logging.DEBUG)
```

The `mara.*` hierarchy is separate from LangSmith — it covers business-logic events like node entry/exit counts, degraded-but-non-fatal conditions (e.g. LLM under-produced sub-queries), and non-LLM node completions that don't appear in LangSmith traces at all.

---

## Citation Format

Citations in the MARA report use a structured inline format:

```
The global rate of deforestation has accelerated since 2015 [ML:3:a4f2c1] [ML:7:8e3d90].
```

Where `ML:3:a4f2c1` means:

- `ML` — Merkle Leaf
- `3` — leaf index 3 in the tree
- `a4f2c1` — the first 6 characters of the leaf's SHA-256 hash (a short fingerprint for display; the full hash is in the `CertifiedReport`)

The full hash is always available in the report's `leaves` array for complete verification. The short fingerprint in the inline citation is human-readable and provides a quick sanity check without cluttering the prose.

---

## Verification

A standalone CLI tool is included for independent verification of any `CertifiedReport` without running the full agent:

```bash
mara verify report.json

# Output:
# ✓ Merkle root matches: a4f2c1d8e9...
# ✓ All 12 leaves verified
# ✓ 31/31 citations have valid Merkle proofs
# ⚠ Source at leaf 7 (https://example.com/paper) returned HTTP 404 — page may have changed
# ✓ Overall integrity: PASS
```

The verifier:

1. Recomputes all leaf hashes from the stored `(url, text, retrieved_at)` tuples.
2. Rebuilds the Merkle tree from the recomputed leaf hashes.
3. Checks the recomputed root against `certified_report.merkle_root`.
4. For each cited leaf, verifies its Merkle proof path.
5. Optionally re-fetches the live source URLs and checks whether the content has changed since retrieval (an HTTP 200 with changed content is logged as a warning, not a failure — the hash records what the page said _at retrieval time_).

---

## Configuration Reference

`ResearchConfig` is a `pydantic-settings` `BaseSettings` subclass defined in `config.py`. All parameters can be overridden via environment variables (prefix `MARA_`, nested delimiter `__`) or a `.env` file in the project root. API keys use standard names without the prefix.

```bash
# .env
BRAVE_API_KEY=...
FIRECRAWL_API_KEY=...
HF_TOKEN=...                    # HuggingFace Hub token (model downloads + inference)
S2_API_KEY=...                  # Semantic Scholar API key — optional but recommended

MARA_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
MARA_HF_PROVIDER=featherless-ai
MARA_HIGH_CONFIDENCE_THRESHOLD=0.80
```

| Parameter                      | Default                            | Description                                                                                          |
| ------------------------------ | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `model`                        | `Qwen/Qwen3-30B-A3B-Instruct-2507` | LLM for query planning, claim extraction, and report synthesis                                       |
| `hf_provider`                  | `featherless-ai`                   | HuggingFace Inference Provider (`featherless-ai`, `groq`, `novita`, `auto`)                          |
| `embedding_model`              | `all-MiniLM-L6-v2`                 | `sentence-transformers` model for retrieval and confidence scoring                                       |
| `max_sources`                  | `20`                               | Brave results requested per sub-query (Brave API cap: 20/request)                                    |
| `max_workers`                  | `3`                                | Number of parallel sub-queries (each spawns one web + one ArXiv + one S2 worker)                    |
| `arxiv_max_results`            | `5`                                | ArXiv papers fetched per sub-query                                                                   |
| `semantic_scholar_max_results` | `5`                                | Semantic Scholar snippets fetched per sub-query                                                      |
| `semantic_scholar_api_key`     | `""` (`S2_API_KEY`)                | Semantic Scholar API key. Optional but recommended — avoids shared rate limits under concurrent load |
| `brave_freshness`              | `""`                               | Optional freshness filter: `pd` (24h), `pw` (7d), `pm` (31d), `py` (1y), or `YYYY-MM-DDtoYYYY-MM-DD` |
| `max_retrieval_candidates`     | `150`                              | Retrieval pool size; reserve for future reranking stage                                              |
| `max_claim_sources`            | `50`                               | Leaves passed to claim extraction after retrieval (≤ candidates)                                     |
| `max_chunks_per_url`           | `3`                                | Maximum chunks from any single source URL admitted to the extraction window. Prevents high-volume commercial sources from crowding out lower-volume sources such as ArXiv papers. |
| `max_extracted_claims`         | `100`                              | Maximum claims the LLM extracts per run (injected into the prompt)                                   |
| `max_corrective_rag_loops`     | `2`                                | Max corrective RAG retries before routing to HITL                                                    |
| `n_leaves_contested_threshold` | `5`                                | `n_unique_urls ≥` this with low SA → contested (sources disagree), not insufficient data             |
| `max_new_pages_per_round`      | `5`                                | Max new pages scraped per corrective sub-query per round                                             |
| `high_confidence_threshold`    | `0.80`                             | SA score above which claims are auto-approved                                                        |
| `low_confidence_threshold`     | `0.55`                             | SA score below which claims trigger corrective retrieval or HITL                                     |
| `similarity_support_threshold` | `0.60`                             | Cosine similarity (exclusive) for a leaf to count as corroborating a claim                          |
| `query_planner_max_tokens`     | `1024`                             | Max new tokens for the query planner LLM call                                                        |
| `claim_extractor_max_tokens`   | `16384`                            | Max new tokens for the claim extractor LLM call                                                      |
| `claim_extractor_min_leaves`   | `10`                               | Minimum leaf window for claim extraction retries; halving stops here and returns empty claims         |
| `report_synthesizer_max_tokens`| `8192`                             | Max new tokens for the report synthesizer LLM call                                                   |
| `corrective_retriever_max_tokens` | `512`                           | Max new tokens for corrective sub-query generation per failing claim                                 |
| `hash_algorithm`               | `sha256`                           | Hash function for Merkle leaves (extensible)                                                         |
| `chunk_size`                   | `1000`                             | Fixed character chunk size for source text splitting                                                 |
| `chunk_overlap`                | `200`                              | Character overlap between consecutive chunks                                                         |
| `checkpointer`                 | `memory`                           | `memory` or `postgres`                                                                               |
| `leaf_db_path`                 | `~/.mara/leaves.db`                | Path to the SQLite leaf database (tilde-expanded at open time)                                       |
| `leaf_cache_max_age_hours`     | `336.0` (14 days)                  | Default TTL for URLs that don't match a specific tier. Immutable URLs use `float('inf')`; academic publishers use 1 year; Wikipedia/S2/ResearchGate use 30 days. See [Freshness cache](#freshness-cache). |
| `leaf_db_enabled`              | `True`                             | Set to `False` to disable all DB reads/writes (CI / unit tests)                                      |

---

## Project Structure

```
mara/
├── agent/
│   ├── graph.py              # StateGraph definition and compilation (12 nodes)
│   ├── state.py              # MARAState TypedDict and all shared data classes
│   ├── run_context.py        # RunContext dataclass — carries leaf embeddings between retriever and scorer
│   ├── nodes/
│   │   ├── query_planner.py      # Sub-query decomposition node
│   │   ├── search_worker/        # Retrieval subgraphs (per sub-query, dispatched via Send)
│   │   │   ├── graph.py          #   Builds search_worker, arxiv_worker, and semantic_scholar_worker subgraphs
│   │   │   ├── brave_search.py   #   Brave Search API → ranked URLs + metadata
│   │   │   ├── arxiv_search.py   #   ArXiv API → versioned PDF URLs + abstracts
│   │   │   ├── firecrawl_scrape.py #   Firecrawl scrape → full page/PDF text → chunks (+ per-URL TTL cache)
│   │   │   ├── semantic_scholar_search.py # S2 /snippet/search → SourceChunks directly (no scrape needed)
│   │   │   └── url_ttl.py        #   url_ttl_hours(): source-type TTL classifier (immutable/1yr/30d/default)
│   │   ├── source_hasher.py      # SHA-256 hash of each chunk; builds and persists Merkle leaves
│   │   ├── merkle_builder.py     # Assembles MerkleTree from leaf hashes
│   │   ├── retriever.py          # Hybrid BM25+semantic RRF retrieval: all leaves → top-K
│   │   ├── claim_extractor.py    # LLM extraction: retrieved text → atomic claims (capped at max_extracted_claims)
│   │   ├── confidence_scorer.py  # Beta-Binomial SA (1+k)/(2+k) → ScoredClaim for each claim
│   │   ├── corrective_retriever.py # DB-first + live scrape for low-confidence claims; back-edge to merkle_builder
│   │   ├── hitl_checkpoint.py    # Contested flagging + confidence stats + interrupt() + human approval
│   │   ├── report_synthesizer.py # Approved claims → prose with Merkle leaf citations
│   │   └── certified_output.py   # Assembles final CertifiedReport; closes DB run
│   └── edges/
│       └── routing.py        # dispatch_search_workers (Send fan-out), route_after_scoring (corrective RAG vs HITL)
├── db/                       # Persistence layer — repository pattern, SQLite implementation
│   ├── __init__.py           #   Exports LeafRepository, SQLiteLeafRepository
│   ├── repository.py         #   LeafRepository Protocol (structural subtyping; Postgres-ready)
│   ├── sqlite_repository.py  #   Concrete SQLite implementation (WAL mode, FTS5 BM25)
│   ├── schema.sql            #   Raw DDL — auditable via git diff
│   └── migrations/
│       └── 001_initial.sql   #   Same as schema.sql; baseline for future ALTER TABLE migrations
├── merkle/
│   ├── tree.py               # MerkleTree builder (bottom-up SHA-256 construction)
│   ├── proof.py              # Merkle proof generation and path verification
│   └── hasher.py             # canonical_serialise() — deterministic json.dumps for hashing
├── confidence/               # Pure logic layer — no LangGraph imports
│   ├── scorer.py             # Computes SA (Beta-Binomial posterior mean); returns ScoredClaim
│   ├── embeddings.py         # SentenceTransformer model loading and embedding cache
│   └── signals.py            # compute_sa: Beta-Binomial SA signal computation
├── cli/
│   └── run.py                # Typer CLI: `mara run QUERY` and `mara info`
├── api/                      # Planned — thin FastAPI adapter over the same agent core
├── config.py                 # ResearchConfig (BaseSettings) — loads .env, validates all config
├── logging.py                # get_logger() factory — mara.* logger hierarchy
└── prompts/
    ├── query_planner.py
    ├── claim_extractor.py        # includes max_extracted_claims injected into system prompt
    ├── corrective_retriever.py
    └── report_synthesizer.py

tests/
├── test_config.py                    # ResearchConfig validation, env loading, nested model
├── test_logging.py                   # get_logger hierarchy
├── test_report_store.py              # Report save/load round-trip
├── test_verifier.py                  # Merkle integrity verification
├── db/
│   └── test_sqlite_repository.py     # Repository tests (in-memory SQLite); includes float('inf') TTL cases
├── merkle/
│   ├── test_hasher.py                # canonical_serialise determinism; hash stability
│   ├── test_tree.py                  # Tree construction, odd-leaf duplication, root hash
│   └── test_proof.py                 # Proof generation and path verification
├── confidence/
│   ├── test_signals.py               # SA Beta-Binomial formula numerically verified
│   ├── test_scorer.py                # Composite score computation; threshold routing
│   └── test_embeddings.py            # Embedding cache; model loading
├── agent/
│   ├── test_state.py                 # TypedDict field presence and defaults
│   ├── test_graph.py                 # Full graph compilation; expected node topology
│   ├── test_llm.py                   # make_llm factory; strip_think helper
│   ├── nodes/
│   │   ├── test_query_planner.py     # Sub-query decomposition with mocked LLM
│   │   ├── test_source_hasher.py     # Chunk → MerkleLeaf pipeline; DB persistence path
│   │   ├── test_merkle_builder.py    # MerkleTree assembly from leaves
│   │   ├── test_retriever.py         # Hybrid RRF + embedding cache; pure-semantic fallback
│   │   ├── test_claim_extractor.py   # Claim extraction with mocked LLM
│   │   ├── test_confidence_scorer.py # Node wiring: state in → scored claims out
│   │   ├── test_corrective_retriever.py # DB-first + scrape paths; loop_count increments; URL dedup
│   │   ├── test_hitl_checkpoint.py   # interrupt() dispatch; approve/auto-approve paths
│   │   ├── test_report_synthesizer.py # Claim-to-prose with mocked LLM
│   │   ├── test_certified_output.py  # CertifiedReport assembly; complete_run DB call
│   │   └── search_worker/
│   │       ├── test_brave_search.py  # Brave API calls with mocked httpx client
│   │       ├── test_firecrawl_scrape.py # Firecrawl scrape with mocked client; cache hit/miss
│   │       ├── test_semantic_scholar_search.py # S2 snippet search; rate limiter; error handling
│   │       ├── test_url_ttl.py       # url_ttl_hours() classification across all tiers and edge cases
│   │       └── test_graph.py         # Search worker subgraph topology
│   └── edges/
│       └── test_routing.py           # dispatch_search_workers, route_after_scoring
├── cli/
│   ├── test_run.py                   # CLI: arg parsing, HITL loop, report display; DB disabled path
│   └── test_verify.py                # CLI verify command; pass/fail exit codes
└── integration/                      # Planned — full end-to-end graph tests with mocked I/O
```

The `confidence/` module is intentionally decoupled from LangGraph — it contains pure Python functions that take claims and source chunks and return scores. `agent/nodes/confidence_scorer.py` is the thin wrapper that pulls scored claims from state, calls into `confidence/scorer.py`, and writes results back to state. This separation means the scoring logic can be tested, benchmarked, and iterated on without running the full graph.

---

## Dependencies

Runtime dependencies are declared in `[project.dependencies]`. Test and dev tooling live in `[dependency-groups]` and are never included in the published package.

| Package                 | Role                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `langgraph>=0.2`        | Agent graph orchestration, checkpointing, HITL                                           |
| `langchain>=0.3`        | LLM abstractions, tool calling, structured output                                        |
| `langchain-huggingface` | `ChatHuggingFace` + `HuggingFaceEndpoint` for Qwen3 inference via HF Inference Providers |
| `langsmith`             | Tracing, evaluation, experiment tracking                                                 |
| `sentence-transformers` | Local embeddings via `all-MiniLM-L6-v2` for retrieval and confidence scoring              |
| `firecrawl-py`          | Full-text page scraping for source extraction and hashing; citation crawl                |
| `httpx`                 | Async HTTP client for Brave Search API and ArXiv API — no wrapper package needed         |
| `pydantic>=2`           | State schema validation and serialisation                                                |
| `pydantic-settings`     | `BaseSettings` for `ResearchConfig`; loads `.env`, validates env vars with type checking |
| `typer`                 | CLI framework for `cli/`; chosen for Pydantic compatibility and clean `--help` output    |
| `psycopg[binary]`       | PostgreSQL connection for production checkpointer                                        |
| `json`, `hashlib`       | Stdlib — deterministic serialisation and SHA-256 hashing; no extra dependency            |

**API group** (`uv sync --group api` — planned, required once `api/` is implemented):

| Package             | Role                                                                               |
| ------------------- | ---------------------------------------------------------------------------------- |
| `fastapi[standard]` | REST API framework; `[standard]` bundles uvicorn (ASGI server) and related tooling |

**Test group** (`uv sync --group test`):

| Package          | Role                         |
| ---------------- | ---------------------------- |
| `pytest>=8`      | Test runner                  |
| `pytest-cov`     | Coverage via `coverage.py`   |
| `pytest-asyncio` | Async node and graph tests   |
| `pytest-mock`    | Mocking external API clients |

---

## Design Decisions & Tradeoffs

**Why SHA-256 and not a content-addressed store like IPFS?**
IPFS would provide globally decentralised content addressing, which is an even stronger integrity guarantee. However, it introduces significant operational complexity (running an IPFS node, pinning content) and is overkill for a local research session. SHA-256 leaf hashes embedded in the report are sufficient for the primary use case: a reader verifying that a specific agent session used specific source text. IPFS integration is a natural extension for a publishing use case where permanence matters.

**Why use a Beta-Binomial model for source agreement rather than a trained classifier?**
A trained classifier would require a labelled dataset of (claim, sources, grounded/hallucinated) examples, which is expensive to produce. The Beta-Binomial model is well-established, requires no training data, produces a principled probability estimate with appropriate uncertainty for small sample sizes (via the Laplace-smoothed posterior mean), and remains interpretable — you can see the raw k/n counts that drove the score.

**Why extract atomic claims rather than score paragraphs?**
Paragraph-level scoring would mask within-paragraph variance. A paragraph that is 80% well-supported but contains one hallucinated number would score 0.8 and pass to the report. Atomic claim extraction surfaces the hallucinated number as a single low-scoring claim that can be flagged or removed without discarding the surrounding well-supported text.

**Why not just fine-tune the LLM to not hallucinate?**
Fine-tuning reduces hallucination rates but does not eliminate them and provides no formal guarantee. The Merkle layer provides a _structural_ guarantee — independent of the LLM's behaviour — that is robust to model changes and model failures. The confidence scorer and HITL checkpoint are the probabilistic layer; the Merkle tree is the cryptographic layer. They operate at different levels and are complementary.

**CLI-first, API-ready architecture**

MARA launches as a CLI but the agent core is kept interface-agnostic so a REST API can be added without restructuring anything.

The constraint: nothing inside `mara/agent/` imports from `cli/` or any future `api/` layer. The agent exposes a single async entry point that accepts a `ResearchConfig` and returns a `CertifiedReport` — both are already typed Pydantic models, so FastAPI can serve them as request/response schemas directly with no transformation layer.

```
invoke(config: ResearchConfig) → CertifiedReport
         ▲                              │
         │                              ▼
   cli/run.py              stdout / file (current)
   api/routes.py           JSON over HTTP (planned)
```

`cli/run.py` is a thin Typer wrapper: parse args → build `ResearchConfig` → call agent → stream output. The planned `api/routes.py` will be an equally thin FastAPI wrapper doing the same over HTTP. The graph compiles once at process startup and is reused across invocations — never re-compiled per request.

---

## Testing

### Tools

| Tool             | Role                                                   |
| ---------------- | ------------------------------------------------------ |
| `pytest`         | Test runner                                            |
| `pytest-cov`     | Coverage measurement via `coverage.py`                 |
| `pytest-asyncio` | Async test support (I/O-bound nodes use `async/await`) |
| `pytest-mock`    | `mocker` fixture for patching external API clients     |

All test dependencies are managed as a `test` dependency group in `pyproject.toml` and never end up in the production lockfile.

### `pyproject.toml` configuration

```toml
[dependency-groups]
test = [
    "pytest>=8",
    "pytest-cov",
    "pytest-asyncio",
    "pytest-mock",
]
dev = [
    {include-group = "test"},
    "ruff",
]
api = [
    "fastapi[standard]",  # planned — enables api/ layer; fastapi[standard] bundles uvicorn
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=mara --cov-report=term-missing --cov-fail-under=98"

[tool.coverage.run]
branch = true          # measure branch coverage, not just line coverage
source = ["mara"]
omit = [
    "mara/prompts/*",  # prompt strings have no testable branching logic
]

[tool.coverage.report]
show_missing = true
skip_covered = false
```

`branch = true` is important for MARA specifically: the confidence routing logic has three distinct branches (fast path, corrective RAG, HITL), and the corrective RAG loop has an iteration limit. Line coverage alone would miss untested branch paths through these conditionals.

### Running tests

```bash
# Install dependencies (uv creates and manages the virtualenv automatically)
uv sync --group test

# Run the full suite with coverage
uv run pytest

# Run a specific module
uv run pytest tests/confidence/

# Run with an HTML coverage report
uv run pytest --cov-report=html
open htmlcov/index.html
```

`uv run` ensures the command executes in an environment with all locked dependencies — no manual `source .venv/bin/activate` required.

### Coverage target: 98%

The 98% floor is enforced by `--cov-fail-under=98` in `addopts`, so CI fails automatically if coverage drops. The two modules most likely to challenge this target and their strategies:

**`agent/nodes/hitl_checkpoint.py`** — the HITL interrupt/resume cycle requires testing `interrupt()` dispatch and all three human decision paths (approve, reject, retry). Use a `MemorySaver` checkpointer in tests and drive decisions via `Command(resume=...)` directly, without needing a real UI.

**`agent/nodes/search_worker/`** — Brave and Firecrawl clients must be mocked at the HTTP client level. Fixture a small set of deterministic mock responses in `conftest.py` and reuse them across the node tests and integration tests.

### Testing philosophy

The module structure is deliberately shaped to make testing tractable:

- `merkle/` and `confidence/` are pure Python with no external I/O — test them exhaustively with exact numerical assertions. The SA Beta-Binomial formula, for example, can be verified to floating-point precision given known k and n values.
- `agent/nodes/` tests verify LangGraph wiring: given a known input state, does the node return the expected output state? External API calls are always mocked.
- `tests/integration/` (planned) will run the complete graph with mocked Brave, Firecrawl, and LLM calls, asserting that routing decisions (corrective RAG firing, HITL triggering) behave correctly for crafted confidence scenarios.

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys for Brave Search, Firecrawl, and HuggingFace (Pro account recommended for inference credits)

### Install

```bash
git clone <repo>
cd MARA
uv sync --group test   # installs runtime + test deps
uv pip install -e .    # registers the `mara` console script
```

### Configure

Create a `.env` file in the project root:

```bash
BRAVE_API_KEY=your_brave_key
FIRECRAWL_API_KEY=your_firecrawl_key
HF_TOKEN=your_huggingface_token      # used for model downloads and inference
S2_API_KEY=your_semantic_scholar_key # optional — raises S2 rate limit from shared to per-key

# Optional overrides (shown with defaults)
MARA_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
MARA_HF_PROVIDER=featherless-ai
MARA_EMBEDDING_MODEL=all-MiniLM-L6-v2
MARA_MAX_WORKERS=3
MARA_ARXIV_MAX_RESULTS=5
MARA_SEMANTIC_SCHOLAR_MAX_RESULTS=5
MARA_HIGH_CONFIDENCE_THRESHOLD=0.80
MARA_LOW_CONFIDENCE_THRESHOLD=0.55
MARA_MAX_CLAIM_SOURCES=50
MARA_MAX_EXTRACTED_CLAIMS=100
MARA_MAX_CORRECTIVE_RAG_LOOPS=2
MARA_N_LEAVES_CONTESTED_THRESHOLD=15

# Leaf database (set to false to run without SQLite)
MARA_LEAF_DB_PATH=~/.mara/leaves.db
MARA_LEAF_CACHE_MAX_AGE_HOURS=336    # default TTL; immutable/academic/semi-stable URLs use longer tiers
MARA_LEAF_DB_ENABLED=true
```

### Run

```bash
# Run the full research pipeline
mara run "What are the long-term effects of remote work on productivity?"

# Enable debug logging
mara run "Your research question" --verbose

# Use a named thread (for checkpointer continuity)
mara run "Your research question" --thread-id my-session-1

# Print current configuration and graph node list
mara info
```

When one or more claims score below `low_confidence_threshold`, the pipeline pauses and presents them in the terminal for review. Enter a comma-separated list of indices to approve, or press Enter to skip all. The graph then resumes with only the approved claims.

### Run tests

```bash
uv run pytest                      # full suite with coverage
uv run pytest tests/agent/         # agent layer only
uv run pytest --cov-report=html    # HTML coverage report
open htmlcov/index.html
```

---

_MARA is a personal research project exploring the intersection of cryptographic data structures, statistical reasoning, and LLM agent systems. It is not affiliated with HuggingFace, LangChain, or any other organisation._
