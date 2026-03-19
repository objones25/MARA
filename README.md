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
    - [5. Confidence Scorer](#5-confidence-scorer)
      - [The scoring model](#the-scoring-model)
      - [Routing based on confidence](#routing-based-on-confidence)
    - [6. HITL Checkpoint](#6-hitl-checkpoint)
    - [7. Report Synthesizer](#7-report-synthesizer)
    - [8. Certified Output](#8-certified-output)
  - [The Merkle Integrity Protocol](#the-merkle-integrity-protocol)
  - [Retrieval Strategy](#retrieval-strategy)
    - [Current approach](#current-approach)
    - [The chunking constraint](#the-chunking-constraint)
    - [Planned: local corpus retrieval](#planned-local-corpus-retrieval)
    - [Retrieval experimentation](#retrieval-experimentation)
  - [Statistical Confidence Scoring](#statistical-confidence-scoring)
    - [Model framing](#model-framing)
    - [Corrective RAG loop](#corrective-rag-loop)
    - [Claim extraction](#claim-extraction)
  - [LangGraph Implementation Details](#langgraph-implementation-details)
    - [State schema](#state-schema)
    - [Checkpointing and durability](#checkpointing-and-durability)
    - [Subgraph isolation](#subgraph-isolation)
  - [LangSmith Observability](#langsmith-observability)
    - [What LangSmith captures](#what-langsmith-captures)
    - [Custom evaluators](#custom-evaluators)
    - [Experiment tracking](#experiment-tracking)
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
| Source-absent halluc.     | Corrective RAG loop      | ✅ Corrective RAG triggered by low confidence               |
| Source-inconsist. halluc. | None                     | ✅ HITL checkpoint + multi-source agreement score           |
| Audit trail               | LLM output only          | Full Merkle tree + LangSmith trace                          |
| Reproducibility           | None                     | Same inputs → same root hash (deterministic)                |

---

## Architecture

The agent is implemented as a directed `StateGraph` in LangGraph. Control flows through eight nodes, some of which fan out into parallel subgraphs using the `Send` API. The Merkle integrity layer is a data structure — not a separate node — that is built incrementally as sources are hashed and committed into the agent state.

```
┌─────────────────────────────────────────────────────────┐
│                     MARA StateGraph                     │
│                                                         │
│  [Query Planner]                                        │
│       │                                                 │
│       ▼                                                 │
│  [Search Workers ×N] ◄── parallel fan-out via Send()   │
│       │                                                 │
│       ▼                                                 │
│  [Source Hasher] ──────────────► [Merkle Tree Builder] │
│       │                                   │            │
│       ▼                                   │            │
│  [Confidence Scorer] ◄────────── validates│            │
│       │                                               │
│    ┌──┴──────────────────┬──────────────────┐         │
│    │                     │                  │         │
│ conf ≥ 0.80        0.55–0.80           conf < 0.55    │
│    │                     │                  │         │
│    │            [Corrective RAG]    [HITL Checkpoint] │
│    │             re-fetch + re-score    approve/      │
│    │                     │             reject/retry   │
│    │                     │                  │         │
│    └─────────────────────┴──────────────────┘         │
│                          ▼                             │
│                 [Report Synthesizer]                   │
│                          │                             │
│                 [Certified Output] ◄── Merkle root ───┘
└─────────────────────────────────────────────────────────┘
```

LangSmith traces every node invocation, every LLM call, and every routing decision across the entire graph.

---

## Core Components

### 1. Query Planner

**Role:** Receives the user's natural language research question and decomposes it into a set of focused sub-queries that can be dispatched to the search workers independently.

**Why this matters:** A top-level question like _"What are the long-term economic effects of universal basic income?"_ is too broad for any single search. Breaking it into sub-queries — covering empirical pilots, theoretical models, criticisms, and regional case studies — gives the search workers tighter retrieval targets, which in turn improves the signal-to-noise ratio of retrieved chunks and makes the confidence scorer's job easier.

**Implementation note:** The planner uses a structured output schema (a list of `SubQuery` objects, each with a `query` string and a `domain` hint). The structured output is enforced using LangChain's `.with_structured_output()` on the LLM call, ensuring the downstream `Send()` fan-out always receives well-formed payloads.

**LangGraph pattern used:** This is a standard single node with a conditional edge that either proceeds to the search fan-out or, if the query is judged to be unanswerable (e.g., a real-time data request), exits early with an explanation.

---

### 2. Parallel Search Workers

**Role:** Execute retrieval for each sub-query simultaneously. Each worker is a compiled subgraph (`agent/nodes/search_worker/`) with two internal nodes running in sequence:

1. **`brave_search`:** Calls the Brave Search API with the sub-query and returns a ranked list of URLs. Brave operates an independent crawl index and provides clean structured results without requiring any scraping.
2. **`firecrawl_scrape`:** For each URL returned by `brave_search`, fetches the full page text via Firecrawl (handling JavaScript rendering and anti-bot measures) and splits it into deterministic fixed-size chunks. These chunks — not search snippets — are what get passed to the Source Hasher. See the [Retrieval Strategy](#retrieval-strategy) section for why full-text scraping rather than snippets is required for Merkle hash integrity.

A third retrieval strategy — citation graph traversal via Firecrawl's crawl endpoint — is planned but not yet implemented. See `agent/nodes/search_worker/graph.py`.

**Why run them in parallel?** LLM-based agents spend most of their wall-clock time waiting on I/O — search APIs, scrape requests, LLM inference. Running multiple sub-queries simultaneously collapses this latency dramatically.

**LangGraph pattern used — the `Send` API:** LangGraph's `Send` API enables fan-out by dispatching one `Send` object per sub-query. The parent graph waits for all workers to finish before proceeding.

```python
# Fan-out: one Send per sub-query
def dispatch_search_workers(state: MARState):
    return [
        Send("search_worker", {"sub_query": q, "thread_id": state["thread_id"]})
        for q in state["sub_queries"]
    ]

graph.add_conditional_edges("query_planner", dispatch_search_workers)
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
@dataclass
class MerkleLeaf:
    index: int               # position in the leaf array
    url: str                 # source URL
    text: str                # chunk text
    retrieved_at: str        # ISO timestamp
    hash: str                # SHA-256 hex digest
    claim_indices: list[int] # claims derived from this leaf (populated later)
```

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

### 5. Confidence Scorer

**Role:** For each factual claim extracted from the retrieved evidence, computes a confidence score estimating the probability the claim is genuinely grounded in the retrieved sources.

#### The scoring model

The confidence score for claim `c` is a weighted combination of three signals.

---

**Signal 1 — Source agreement rate (SA):**

SA is modelled using a **Beta-Binomial conjugate model** — the standard Bayesian approach for estimating a proportion from count data. We treat each source as an independent Bernoulli trial: does this source support the claim?

Prior: `Beta(α=1, β=1)` — a uniform (maximally uninformative) prior, making no prior assumption about support probability.

After observing `k` supporting sources out of `n` total retrieved, the posterior is:

```
posterior = Beta(1 + k, 1 + (n − k))
SA(c)     = (1 + k) / (2 + n)        ← posterior mean (Laplace-smoothed)
```

This is the Laplace-smoothed estimator. It avoids boundary values of 0 and 1, which ensures that "zero supporting sources from a small sample" is not treated with the same certainty as "zero supporting sources from a large sample."

A source "supports" the claim if `model.similarity(claim_embedding, source_embedding) > τ` (default τ = 0.72), using `SentenceTransformer("all-MiniLM-L6-v2")`.

---

**Signal 2 — Cross-source consistency (CSC):**

Among the supporting sources, we measure how uniformly they support the claim using the **coefficient of variation (CV)** — a standard normalised dispersion measure.

```
CSC(c) = 1 − CV  =  1 − (std(similarities) / mean(similarities))
         where similarities = {sim(c, s_i) : s_i ∈ supporting_sources}
```

High CV means supporting sources vary widely in how strongly they support the claim, which may indicate the claim is a generalisation that individual sources endorse only partially. CSC defaults to 0.5 when fewer than 2 sources support the claim (insufficient data for a consistency estimate).

---

**Signal 3 — LLM self-assessment (LSA):**

A separate LLM call asks: _"Given only the following retrieved sources, is this claim directly supported, partially supported, or unsupported?"_ The response maps to 1.0, 0.5, or 0.0. This is an independent signal that catches cases where semantic similarity is high but the logical relationship is wrong — for example, a claim that misquotes a number from a source that does contain a number on that topic. LSA is a calibration signal, not a probabilistic estimate.

---

**Final composite score:**

```
confidence(c) = α·SA(c) + β·CSC(c) + γ·LSA(c)
```

Default weights: `α=0.4, β=0.2, γ=0.4`. These are configurable and are candidates for learned optimisation using accumulated LangSmith trace data.

#### Routing based on confidence

All routing decisions use the **composite confidence score**. The individual signal values (SA, CSC, LSA) are surfaced in the HITL interface to help the human reviewer diagnose _why_ a claim scored low — they do not independently gate routing branches.

```
confidence ≥ 0.80  →  fast path to Report Synthesizer
0.55 ≤ confidence < 0.80  →  Corrective RAG: retrieve more sources, re-score claim
confidence < 0.55  →  HITL Checkpoint: human review required
```

---

### 6. HITL Checkpoint

**Role:** When one or more claims score below the confidence floor, the graph pauses execution using LangGraph's `interrupt()` mechanism and surfaces the problematic claims to a human reviewer.

The checkpoint presents:

- The claim text
- Its confidence score and which signals dragged it down
- The sources that were retrieved (with their Merkle leaf indices)
- Three options: **Approve** (the human overrides and accepts the claim), **Reject** (the claim is dropped from the report), or **Retry** (the human can provide an amended search query to retrieve better sources)

```python
def hitl_checkpoint(state: MARState) -> MARState:
    low_confidence_claims = [
        c for c in state["scored_claims"]
        if c.confidence < state["config"]["low_confidence_threshold"]
    ]
    if low_confidence_claims:
        human_decisions = interrupt({
            "kind": "review_low_confidence_claims",
            "claims": [c.to_dict() for c in low_confidence_claims],
            "instructions": "For each claim: approve | reject | retry(query=...)"
        })
        # Resume with human_decisions applied to state
        return apply_human_decisions(state, human_decisions)
    return state
```

This is implemented using LangGraph's `interrupt()` + `Command(resume=...)` pattern, which persists the full graph state to the configured checkpointer so the graph can be resumed hours or days later — even after the server restarts. The durable execution guarantee comes from LangGraph's checkpointing infrastructure.

---

### 7. Report Synthesizer

**Role:** Receives the approved, high-confidence claims along with the full set of Merkle leaves and synthesises a coherent prose report with inline citations.

Each citation in the report is not a simple URL — it is a structured reference that includes the Merkle leaf index:

```
[1:H(src1)] → Merkle leaf index 1, hash H(src1)
```

The synthesizer is prompted to reference claims by their leaf index rather than by URL alone. This means the final report contains explicit pointers into the Merkle tree, making every sentence that makes a factual claim independently verifiable.

**Key constraint:** The synthesizer is instructed _not_ to introduce any new factual claims that are not present in the approved, scored claim set. It may only rephrase, combine, and structure claims that have already passed the confidence filter. This is enforced by including the scored claim set in the prompt context and explicitly prohibiting the addition of "background knowledge."

---

### 8. Certified Output

**Role:** The final node. It assembles the research report, the complete Merkle tree (all leaf hashes, all intermediate hashes, the root hash), the Merkle proofs for each cited leaf, and the full LangSmith run URL into a single `CertifiedReport` object.

```python
@dataclass
class CertifiedReport:
    title: str
    report_text: str                    # prose with inline [leaf_index:hash] citations
    merkle_root: str                    # 64-char SHA-256 hex digest
    merkle_tree: MerkleTree             # full tree for independent verification
    merkle_proofs: dict[int, list]      # {leaf_index: proof_path}
    leaves: list[MerkleLeaf]            # all source chunks + hashes
    scored_claims: list[ScoredClaim]    # all claims with confidence scores
    langsmith_run_url: str              # direct link to LangSmith trace
    generated_at: str                   # ISO timestamp
    config_snapshot: dict               # the exact config used (thresholds, model, etc.)
```

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

## Retrieval Strategy

### Current approach

MARA's retrieval pipeline runs two sequential steps per sub-query, implemented as distinct nodes inside the search worker subgraph:

1. **`brave_search` (Brave Search API):** Returns ranked result URLs for the sub-query. Brave operates an independent crawl index rather than repackaging Google/Bing results, which reduces SEO-optimised noise in research contexts.
2. **`firecrawl_scrape` (Firecrawl):** Fetches the full page text for each URL. Firecrawl handles JavaScript-rendered pages and anti-bot measures, returning clean markdown. This full text is chunked and hashed — not a snippet controlled by the search provider.

Separating search and scraping into distinct nodes means their failure modes, retry logic, and LangSmith traces are independent. Swapping out either provider does not touch the other node.

### The chunking constraint

For Merkle integrity to hold, chunking must be **deterministic and reproducible**: the same source page must always be split into the same chunks. This rules out chunking strategies that depend on model tokenisation windows, probabilistic sentence boundary detection, or any state not encoded in the source text itself.

The current approach uses fixed-size character chunking with overlap, implemented in `agent/nodes/source_hasher.py`. The chunk size and overlap are configurable parameters. This is the simplest chunking strategy that satisfies the determinism requirement.

### Planned: local corpus retrieval

Web search covers recent public-web sources. For research over a pre-indexed private corpus (internal documents, downloaded papers, licensed databases), a vector store retrieval step will be added as a third retrieval strategy in the search worker subgraph, sitting alongside web search and citation traversal.

The same determinism constraint applies: chunks stored in the vector index must be produced by the same chunking function used at hash time. The vector store is then queried for semantic neighbours, but the retrieved text is re-hashed against the stored leaf to verify it hasn't changed since indexing.

### Retrieval experimentation

The retrieval pipeline is one of the primary areas for experimentation as the project develops. Approaches worth evaluating:

- **Hybrid search:** Combining dense (embedding) retrieval with sparse (BM25/keyword) retrieval and rank fusion. Tends to outperform either alone, especially for precise factual queries where exact term match matters.
- **Contextual retrieval:** Before embedding each chunk, prepend a short LLM-generated summary of where the chunk sits within the broader document (Anthropic's approach). Improves retrieval precision for long documents where individual chunks lose their referential context.
- **HyDE (Hypothetical Document Embeddings):** Instead of embedding the query directly, generate a hypothetical answer to the query and embed that. The hypothesis embedding is often closer in vector space to real answer documents than the raw question embedding is.

Any retrieval improvement must still produce deterministic chunks that can be hashed and committed to the Merkle tree. Retrieval quality improvements affect which chunks are found; the hash commitment records exactly which chunks were used, whatever the strategy.

---

## Statistical Confidence Scoring

### Model framing

The SA signal uses a Beta-Binomial conjugate model, which is covered in detail in the Confidence Scorer node section above. To summarise: SA is the posterior mean of a `Beta(1 + k, 1 + n − k)` distribution, where `k` sources out of `n` retrieved exceed the cosine similarity threshold. This is a principled and well-established Bayesian estimator for proportions.

CSC and LSA are not probabilistic models — they are heuristic signals that provide independent perspectives on claim quality. The weighted sum that combines all three is a heuristic aggregator, not a full Bayesian update. The weights `α, β, γ` are configurable and are the primary targets for learned optimisation as LangSmith traces accumulate.

### Corrective RAG loop

The corrective RAG loop fires when the **composite confidence score** falls in the middle band (`0.55 ≤ confidence < 0.80`). The loop re-runs the search worker with a refined query — the Query Planner is invoked again in "refine" mode with the failing claim as additional context. New chunks are hashed, added to the Merkle tree as new leaves, and the confidence scorer re-evaluates the claim against the augmented source set.

The composite score determines routing in all cases. The individual signal breakdown (which of SA, CSC, LSA dragged the score down) is surfaced in the HITL interface as diagnostic context, not as a separate routing condition.

The loop has a configurable maximum iteration count (default: 2) to prevent runaway retrieval.

### Claim extraction

Before scoring, an LLM extraction step converts raw retrieved text into a structured set of atomic claims. Each claim is:

- A single factual assertion (not a compound statement)
- Tagged with the source chunk indices it was extracted from
- Stored with its original sentence span for attribution

Atomic claim extraction is important because compound statements (e.g., _"Study X found Y, and also concluded Z"_) can have mixed support — Y might be well-supported but Z might not. Treating them as one claim would average away a real signal.

---

## LangGraph Implementation Details

### State schema

The graph is typed using Python `TypedDict` to ensure every node receives and returns well-formed state:

```python
import operator
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class MARState(TypedDict):
    # Input
    query: str
    config: ResearchConfig

    # Planner output
    sub_queries: list[SubQuery]

    # Search worker output (reduced from parallel runs)
    raw_chunks: Annotated[list[SourceChunk], operator.add]  # fan-in: add lists

    # Hasher output
    merkle_leaves: list[MerkleLeaf]
    merkle_tree: MerkleTree

    # Scorer output
    extracted_claims: list[Claim]
    scored_claims: list[ScoredClaim]

    # HITL output
    human_approved_claims: list[ScoredClaim]

    # Synthesizer output
    report_draft: str

    # Final output
    certified_report: CertifiedReport | None

    # Internal
    messages: Annotated[list, add_messages]
    loop_count: int  # for corrective RAG iteration limit
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

### Custom evaluators

MARA ships with three LangSmith evaluators that can be run against any set of completed research runs:

**Faithfulness evaluator:** Checks that every claim in the certified report is attributable to a Merkle leaf. Flags any claim with a leaf index that doesn't exist in the tree.

```python
def faithfulness_evaluator(run, example):
    report = run.outputs["certified_report"]
    tree_indices = {leaf.index for leaf in report.leaves}
    cited_indices = extract_citation_indices(report.report_text)
    unfaithful = cited_indices - tree_indices
    return {
        "key": "faithfulness",
        "score": 1.0 if not unfaithful else 0.0,
        "comment": f"Uncited leaf indices: {unfaithful}"
    }
```

**Merkle integrity evaluator:** Recomputes the entire Merkle tree from the stored leaves and checks that the resulting root hash matches the root hash embedded in the report.

```python
def merkle_integrity_evaluator(run, example):
    report = run.outputs["certified_report"]
    recomputed_root = build_merkle_tree(report.leaves).root
    match = recomputed_root == report.merkle_root
    return {
        "key": "merkle_integrity",
        "score": 1.0 if match else 0.0,
        "comment": f"Root match: {match}"
    }
```

**Confidence coverage evaluator:** Measures the fraction of claims in the report that have a confidence score above the configured high threshold, as a measure of overall report quality.

```python
def confidence_coverage_evaluator(run, example):
    claims = run.outputs["certified_report"].scored_claims
    high_conf = sum(1 for c in claims if c.confidence >= HIGH_THRESHOLD)
    score = high_conf / len(claims) if claims else 0.0
    return {"key": "confidence_coverage", "score": score}
```

### Experiment tracking

Different configurations — different confidence thresholds, different LLMs, different retrieval strategies — can be compared against the same set of research questions using LangSmith's `evaluate()` function. This makes MARA's confidence and integrity pipeline fully benchmarkable.

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
mara verify --report report.json

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
MARA_MODEL=claude-sonnet-4-6
MARA_HIGH_CONFIDENCE_THRESHOLD=0.80
MARA_CONFIDENCE_WEIGHTS__ALPHA=0.4
```

| Parameter                      | Default             | Description                                          |
| ------------------------------ | ------------------- | ---------------------------------------------------- |
| `model`                        | `claude-sonnet-4-6` | LLM for planning, synthesis, LSA scoring             |
| `embedding_model`              | `all-MiniLM-L6-v2`  | `sentence-transformers` model for SA and CSC scoring |
| `max_sources`                  | `30`                | Hard cap on total retrieved chunks                   |
| `max_workers`                  | `3`                 | Number of parallel search workers                    |
| `max_corrective_rag_loops`     | `2`                 | Max corrective RAG retries per low-confidence claim  |
| `high_confidence_threshold`    | `0.80`              | Composite score above which claims go to report      |
| `low_confidence_threshold`     | `0.55`              | Composite score below which claims go to HITL        |
| `similarity_support_threshold` | `0.72`              | Cosine similarity for a source to "support" a claim  |
| `confidence_weights.alpha`     | `0.4`               | Weight for source agreement rate (SA)                |
| `confidence_weights.beta`      | `0.2`               | Weight for cross-source consistency (CSC)            |
| `confidence_weights.gamma`     | `0.4`               | Weight for LLM self-assessment (LSA)                 |
| `hash_algorithm`               | `sha256`            | Hash function for Merkle leaves (extensible)         |
| `chunk_size`                   | `1000`              | Fixed character chunk size for source text splitting |
| `chunk_overlap`                | `200`               | Character overlap between consecutive chunks         |
| `checkpointer`                 | `memory`            | `memory` or `postgres`                               |

---

## Project Structure

```
mara/
├── agent/
│   ├── graph.py              # StateGraph definition and compilation
│   ├── state.py              # MARState TypedDict and all shared data classes
│   ├── nodes/
│   │   ├── query_planner.py      # Sub-query decomposition node
│   │   ├── search_worker/        # Retrieval subgraph (one per sub-query, dispatched via Send)
│   │   │   ├── graph.py          #   Subgraph definition: brave_search → firecrawl_scrape
│   │   │   ├── brave.py          #   Brave Search API calls → ranked URLs
│   │   │   └── firecrawl.py      #   Firecrawl scrape → full page text → deterministic chunks
│   │   ├── source_hasher.py      # SHA-256 hash of each chunk; builds Merkle leaves
│   │   ├── confidence_scorer.py  # LangGraph node: calls into confidence/ for scoring
│   │   ├── hitl_checkpoint.py    # interrupt() + human decision handling
│   │   ├── report_synthesizer.py # Claim-to-prose with inline Merkle leaf citations
│   │   └── certified_output.py   # Assembles final CertifiedReport
│   └── edges/
│       └── routing.py        # Conditional edge functions (confidence thresholds → routing)
├── merkle/
│   ├── tree.py               # MerkleTree builder (bottom-up SHA-256 construction)
│   ├── proof.py              # Merkle proof generation and path verification
│   └── hasher.py             # canonical_serialise() — deterministic json.dumps for hashing
├── confidence/               # Pure logic layer — no LangGraph imports
│   ├── scorer.py             # Computes SA (Beta-Binomial), CSC (CV), LSA; returns ScoredClaim
│   ├── embeddings.py         # SentenceTransformer model loading and embedding cache
│   └── signals.py            # Individual SA, CSC, LSA signal computation functions
├── evaluation/
│   ├── evaluators.py         # LangSmith custom evaluators (faithfulness, integrity, coverage)
│   └── datasets.py           # Test question datasets
├── cli/
│   ├── run.py                # Thin Typer adapter: args → ResearchConfig → agent → stdout
│   └── verify.py             # mara verify --report report.json
├── api/                      # Planned — thin FastAPI adapter over the same agent core
│   └── routes.py             #   ResearchConfig (request body) → agent → CertifiedReport (response)
├── config.py                 # ResearchConfig (BaseSettings) — loads .env, validates all config
└── prompts/
    ├── query_planner.py
    ├── claim_extractor.py
    ├── lsa_scorer.py
    └── report_synthesizer.py

tests/
├── conftest.py               # Shared fixtures: mock Brave client, mock Firecrawl, sample MARState
├── merkle/
│   ├── test_hasher.py        # canonical_serialise determinism; hash stability across platforms
│   ├── test_tree.py          # Tree construction, odd-leaf duplication, root hash correctness
│   └── test_proof.py         # Proof generation and path verification
├── confidence/
│   ├── test_signals.py       # SA Beta-Binomial formula (numerically verified), CSC edge cases
│   ├── test_scorer.py        # Composite score computation; routing threshold coverage
│   └── test_embeddings.py    # Embedding cache behaviour; model loading
├── agent/
│   ├── nodes/
│   │   ├── test_query_planner.py    # Sub-query decomposition with mocked LLM
│   │   ├── test_search_worker.py    # brave.py and firecrawl.py with mocked API clients
│   │   ├── test_source_hasher.py    # Chunk → MerkleLeaf pipeline
│   │   ├── test_confidence_scorer.py # Node wiring: state in → scored claims out
│   │   ├── test_hitl_checkpoint.py  # interrupt() dispatch; approve/reject/retry paths
│   │   └── test_routing.py          # All three confidence routing branches
│   └── test_graph.py         # Full graph compilation smoke test
├── cli/
│   ├── test_run.py           # CLI adapter: arg parsing, ResearchConfig construction, output formatting
│   └── test_verify.py        # CLI verification against a known-good CertifiedReport fixture
└── integration/
    └── test_full_run.py      # End-to-end graph run with all external calls mocked

# tests/api/ — planned, mirrors cli/ structure once api/ is implemented
```

The `confidence/` module is intentionally decoupled from LangGraph — it contains pure Python functions that take claims and source chunks and return scores. `agent/nodes/confidence_scorer.py` is the thin wrapper that pulls scored claims from state, calls into `confidence/scorer.py`, and writes results back to state. This separation means the scoring logic can be tested, benchmarked, and iterated on without running the full graph.

---

## Dependencies

Runtime dependencies are declared in `[project.dependencies]`. Test and dev tooling live in `[dependency-groups]` and are never included in the published package.

| Package                 | Role                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| `langgraph>=1.0`        | Agent graph orchestration, checkpointing, HITL                                                              |
| `langchain>=1.0`        | LLM abstractions, tool calling, structured output                                                           |
| `langsmith`             | Tracing, evaluation, experiment tracking                                                                    |
| `sentence-transformers` | Local embeddings via `all-MiniLM-L6-v2` for SA and CSC scoring                                              |
| `firecrawl-py`          | Full-text page scraping for source extraction and hashing; citation crawl                                   |
| `httpx`                 | Async HTTP client for direct Brave Search API calls in `search_worker/brave.py` — no wrapper package needed |
| `pydantic>=2`           | State schema validation and serialisation                                                                   |
| `pydantic-settings`     | `BaseSettings` for `ResearchConfig`; loads `.env`, validates env vars with type checking                    |
| `typer`                 | CLI framework for `cli/`; chosen for Pydantic compatibility and clean `--help` output                       |
| `psycopg[binary]`       | PostgreSQL connection for production checkpointer                                                           |
| `json`, `hashlib`       | Stdlib — deterministic serialisation and SHA-256 hashing; no extra dependency                               |

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

**Why use a Beta-Binomial model for source agreement (SA) rather than a trained classifier?**
A trained classifier would require a labelled dataset of (claim, sources, grounded/hallucinated) examples, which is expensive to produce. The Beta-Binomial model is well-established, requires no training data, produces a principled probability estimate with appropriate uncertainty for small sample sizes (via the Laplace-smoothed posterior mean), and remains interpretable — you can see the raw k/n counts that drove the score. The weighted aggregation across SA, CSC, and LSA is a heuristic combinator on top of that, not a full Bayesian model. As LangSmith traces accumulate, the weights α, β, γ are candidates for learned optimisation.

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
- `tests/integration/test_full_run.py` runs the complete graph with mocked Brave, Firecrawl, and LLM calls, and asserts that routing decisions (corrective RAG firing, HITL triggering) behave correctly for crafted confidence scenarios.

---

## Getting Started

> **TBD** — setup instructions, environment variables, and quickstart guide will be added once the initial implementation is stable.

---

_MARA is a personal research project exploring the intersection of cryptographic data structures, statistical reasoning, and LLM agent systems. It is not affiliated with Anthropic, LangChain, or any other organisation._
