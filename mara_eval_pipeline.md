# MARA — Prompt Evaluation Pipeline

> How to systematically evaluate the RAG pipeline, LLM prompts, confidence scoring, and Merkle integrity in MARA using LangSmith, pytest, and DeepEval.

---

## Table of Contents

- [MARA — Prompt Evaluation Pipeline](#mara--prompt-evaluation-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [What We Are Evaluating](#what-we-are-evaluating)
  - [Evaluation Architecture](#evaluation-architecture)
  - [Golden Dataset Construction](#golden-dataset-construction)
  - [LangSmith Evaluators](#langsmith-evaluators)
    - [1. RAG Groundedness Evaluator](#1-rag-groundedness-evaluator)
    - [2. RAG Correctness Evaluator](#2-rag-correctness-evaluator)
    - [3. RAG Retrieval Relevance Evaluator](#3-rag-retrieval-relevance-evaluator)
    - [4. Merkle Integrity Evaluator](#4-merkle-integrity-evaluator)
    - [5. Faithfulness (Citation Validity) Evaluator](#5-faithfulness-citation-validity-evaluator)
    - [6. Confidence Coverage Evaluator](#6-confidence-coverage-evaluator)
    - [7. HITL Trigger Rate Evaluator](#7-hitl-trigger-rate-evaluator)
  - [Prompt-Specific Evaluators](#prompt-specific-evaluators)
    - [Query Planner Prompt](#query-planner-prompt)
    - [Claim Extractor Prompt](#claim-extractor-prompt)
    - [LSA Scorer Prompt](#lsa-scorer-prompt)
    - [Report Synthesizer Prompt](#report-synthesizer-prompt)
  - [Running LangSmith Experiments](#running-langsmith-experiments)
  - [LangGraph Node Unit Tests](#langgraph-node-unit-tests)
  - [Integration Test Strategy](#integration-test-strategy)
  - [Offline vs. Online Evaluation](#offline-vs-online-evaluation)
  - [Confidence Weight Optimisation](#confidence-weight-optimisation)
  - [Evaluation Workflow Summary](#evaluation-workflow-summary)

---

## Overview

MARA has a layered evaluation problem. Unlike a simple chatbot, it combines a multi-stage retrieval pipeline, cryptographic hash commitments, a statistical confidence model, and four distinct LLM prompt templates. A regression in any one layer can silently degrade the final report quality without being visible from the outside.

This document describes a systematic evaluation pipeline that covers all four layers:

1. **RAG quality** — does the retrieval-and-generation pipeline produce accurate, grounded, relevant answers?
2. **Prompt quality** — do each of the four prompts (query planner, claim extractor, LSA scorer, report synthesizer) behave correctly?
3. **Confidence model calibration** — does the weighted composite score correlate with actual claim accuracy?
4. **Merkle integrity** — does the cryptographic layer remain internally consistent across all runs?

All evaluations are wired into LangSmith so that experiments are versioned, comparable, and reproducible.

---

## What We Are Evaluating

```
┌─────────────────────────────────────────────────────────────────┐
│  Eval Layer       │  What can go wrong                          │
├───────────────────┼─────────────────────────────────────────────┤
│  RAG pipeline     │  Irrelevant chunks retrieved; hallucinated  │
│                   │  synthesis despite real sources             │
├───────────────────┼─────────────────────────────────────────────┤
│  Query Planner    │  Sub-queries too broad/narrow; missing      │
│  prompt           │  domain hints; compound queries not split   │
├───────────────────┼─────────────────────────────────────────────┤
│  Claim Extractor  │  Compound claims not atomised; incorrect    │
│  prompt           │  source-chunk attribution                   │
├───────────────────┼─────────────────────────────────────────────┤
│  LSA Scorer       │  Self-reports "supported" for hallucinated  │
│  prompt           │  claims; false negatives on real support    │
├───────────────────┼─────────────────────────────────────────────┤
│  Report           │  Introduces new facts not in approved       │
│  Synthesizer      │  claim set; drops Merkle leaf citations;    │
│  prompt           │  incorrect leaf index in inline citation    │
├───────────────────┼─────────────────────────────────────────────┤
│  Merkle integrity │  Root hash mismatch; proof path broken;     │
│                   │  non-deterministic serialisation            │
├───────────────────┼─────────────────────────────────────────────┤
│  Confidence       │  Weights α,β,γ miscalibrated; routing to   │
│  model            │  fast path when HITL should fire            │
└───────────────────┴─────────────────────────────────────────────┘
```

---

## Evaluation Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MARA Evaluation Pipeline                       │
│                                                                   │
│  Golden Dataset (LangSmith)                                       │
│  ├── research_questions.json  (inputs: question, domain)         │
│  ├── golden_answers.json      (reference outputs: claims, urls)  │
│  └── adversarial_cases.json   (edge cases, known failure modes)  │
│                                                                   │
│  Evaluators                                                       │
│  ├── groundedness()        RAG: answer grounded in retrieved docs │
│  ├── correctness()         RAG: answer matches ground truth       │
│  ├── retrieval_relevance() RAG: retrieved chunks match question   │
│  ├── merkle_integrity()    Crypto: root hash recomputes correctly │
│  ├── faithfulness()        Citations: all leaf indices exist      │
│  ├── confidence_coverage() Scoring: fraction above high threshold │
│  └── hitl_trigger_rate()   Routing: low-conf claims flagged      │
│                                                                   │
│  LangSmith Experiments (client.evaluate)                         │
│  ├── Experiment A — baseline (claude-sonnet-4-6, α=0.4)          │
│  ├── Experiment B — prompt v2 (updated query planner)            │
│  └── Experiment C — weight sweep (α=0.5, β=0.1, γ=0.4)          │
│                                                                   │
│  pytest suite (offline, CI-gated)                                │
│  ├── tests/merkle/           Pure hash / tree / proof tests      │
│  ├── tests/confidence/       Signal formula tests                 │
│  ├── tests/agent/nodes/      Per-node wiring tests                │
│  └── tests/integration/      Full graph with mocked I/O          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Golden Dataset Construction

The golden dataset lives in LangSmith and is the single source of truth for all offline experiments. It is built from three sources:

**1. Curated research questions (20–50 items to start)**

These should span the domains MARA is intended to handle and include questions at three difficulty levels:

- _Narrow factual_ — single-source questions with a verifiable correct answer (e.g. "What was GDP growth in Germany in 2022?"). These stress-test the retrieval pipeline and correctness evaluator.
- _Synthesis questions_ — questions requiring claims from multiple independent sources (e.g. "What are the main criticisms of universal basic income pilots?"). These stress-test the confidence model and report synthesizer.
- _Adversarial questions_ — questions likely to surface hallucination or HITL triggering (e.g. recent events where evidence is sparse or conflicting).

**2. Production failures (added continuously)**

Any run where the Merkle integrity evaluator or faithfulness evaluator returns 0 becomes a permanent test case. This is the MARA equivalent of a bug report becoming a regression test.

**3. Edge cases for each confidence routing branch**

Craft examples where:

- The composite confidence should clearly be ≥ 0.80 (fast path)
- The composite confidence should be in the 0.55–0.80 band (corrective RAG)
- The composite confidence should be < 0.55 (HITL required)

These validate that the routing logic fires correctly under controlled conditions.

```python
from langsmith import Client

client = Client()

# Create the primary research quality dataset
dataset = client.create_dataset(
    dataset_name="mara-research-quality-v1",
    description="Golden dataset for MARA RAG pipeline and prompt evaluation.",
)

examples = [
    {
        "inputs": {
            "question": "What were the main findings of the Finland UBI pilot (2017–2018)?",
            "domain": "economics",
        },
        "outputs": {
            # Reference answer for correctness evaluator
            "answer": (
                "The Finland pilot gave 2,000 unemployed people €560/month unconditionally. "
                "Participants reported higher wellbeing and trust in institutions. "
                "Employment effects were modest but slightly positive vs. control group."
            ),
            # For retrieval_relevance: URLs that should be in the retrieved set
            "expected_urls": [
                "https://www.kela.fi/web/en/basic-income-experiment",
            ],
            # For faithfulness: every claim in the answer must cite a real leaf
            "claims": [
                "2,000 unemployed people received €560/month",
                "Participants reported higher wellbeing",
                "Participants reported higher trust in institutions",
                "Employment effects were modest but slightly positive",
            ],
        },
    },
    {
        "inputs": {
            "question": "What is the current scientific consensus on ocean acidification rates?",
            "domain": "climate science",
        },
        "outputs": {
            "answer": (
                "Ocean pH has dropped by approximately 0.1 units since industrialisation, "
                "representing a 26% increase in acidity. Current rate of acidification "
                "is faster than any time in the past 300 million years."
            ),
            "expected_urls": [],
            "claims": [
                "Ocean pH has dropped by approximately 0.1 units since industrialisation",
                "This represents a 26% increase in acidity",
                "Current rate is faster than any time in the past 300 million years",
            ],
        },
    },
    # --- adversarial: sparse evidence, should trigger HITL ---
    {
        "inputs": {
            "question": "What are the long-term cognitive effects of microdosing psilocybin?",
            "domain": "neuroscience",
            "expected_routing": "hitl",  # metadata hint for routing evaluator
        },
        "outputs": {
            "answer": "Evidence is preliminary; no large RCT data available as of 2024.",
            "expected_urls": [],
            "claims": [],
        },
    },
]

client.create_examples(dataset_id=dataset.id, examples=examples)
```

---

## LangSmith Evaluators

All evaluators follow the LangSmith convention: they accept `inputs`, `outputs`, and optionally `reference_outputs` dicts, and return a `bool`, `float`, or `EvaluationResult`.

### 1. RAG Groundedness Evaluator

Checks that the final report text does not contain claims that are absent from the retrieved source chunks. This catches _source-inconsistent hallucination_ — the failure mode where the model misreads or ignores what it retrieved.

```python
# evaluation/evaluators.py

from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langsmith.evaluation import EvaluationResult

class GroundedGrade(TypedDict):
    explanation: str
    grounded: bool

_GROUNDEDNESS_PROMPT = """\
You are an auditor reviewing an AI-generated research report.

You will be given:
  SOURCES: the raw text chunks the agent retrieved
  REPORT:  the report the agent produced

Your task:
  (1) Check that every factual claim in the REPORT is directly supported by SOURCES.
  (2) Flag any claim that introduces information not present in SOURCES.

grounded = True  means ALL claims in the REPORT are supported by SOURCES.
grounded = False means at least one claim is unsupported or contradicts SOURCES.

Reason step-by-step before giving your verdict.
"""

_grounded_llm = ChatAnthropic(
    model="claude-sonnet-4-6", temperature=0
).with_structured_output(GroundedGrade, method="json_schema")


def groundedness(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    LangSmith evaluator: is the report grounded in the retrieved sources?

    Expects outputs to contain:
        outputs["certified_report"].leaves       — list[MerkleLeaf]
        outputs["certified_report"].report_text  — str
    """
    report = outputs["certified_report"]
    source_text = "\n\n---\n\n".join(
        f"[Leaf {leaf.index}] {leaf.url}\n{leaf.text}"
        for leaf in report.leaves
    )
    user_msg = f"SOURCES:\n{source_text}\n\nREPORT:\n{report.report_text}"

    grade = _grounded_llm.invoke(
        [
            {"role": "system", "content": _GROUNDEDNESS_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    return EvaluationResult(
        key="groundedness",
        score=1.0 if grade["grounded"] else 0.0,
        comment=grade["explanation"],
    )
```

### 2. RAG Correctness Evaluator

Measures factual accuracy against the reference answer in the golden dataset. Only usable for examples that have a `reference_outputs["answer"]`.

```python
class CorrectnessGrade(TypedDict):
    explanation: str
    correct: bool

_CORRECTNESS_PROMPT = """\
You are a grader checking a research report against a reference answer.

Grade criteria:
  (1) The REPORT must not contain any statement that directly conflicts with the REFERENCE.
  (2) The REPORT may contain MORE information than the REFERENCE — that is acceptable.
  (3) Key facts in the REFERENCE must appear, at minimum implicitly, in the REPORT.

correct = True  means the report meets all criteria.
correct = False means the report contradicts or omits key facts from the reference.
"""

_correctness_llm = ChatAnthropic(
    model="claude-sonnet-4-6", temperature=0
).with_structured_output(CorrectnessGrade, method="json_schema")


def correctness(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> EvaluationResult:
    """
    LangSmith evaluator: does the report match the reference answer?
    Skips gracefully if no reference answer is provided.
    """
    if not reference_outputs.get("answer"):
        return EvaluationResult(key="correctness", score=None, comment="No reference answer provided.")

    report = outputs["certified_report"]
    user_msg = (
        f"QUESTION: {inputs['question']}\n\n"
        f"REFERENCE ANSWER:\n{reference_outputs['answer']}\n\n"
        f"REPORT:\n{report.report_text}"
    )
    grade = _correctness_llm.invoke(
        [
            {"role": "system", "content": _CORRECTNESS_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    return EvaluationResult(
        key="correctness",
        score=1.0 if grade["correct"] else 0.0,
        comment=grade["explanation"],
    )
```

### 3. RAG Retrieval Relevance Evaluator

Checks whether the chunks the agent committed to the Merkle tree are actually relevant to the research question. A high retrieval relevance score with a low groundedness score indicates the agent retrieved the right sources but misread them.

```python
class RelevanceGrade(TypedDict):
    explanation: str
    relevant: bool
    relevant_leaf_count: int
    total_leaf_count: int

_RETRIEVAL_RELEVANCE_PROMPT = """\
You are auditing the retrieval stage of a research agent.

You will be given a QUESTION and a list of SOURCE CHUNKS retrieved by the agent.

For each chunk, decide:
  - Is it directly relevant to answering the QUESTION?
  - Would a human researcher consider it useful for this topic?

Count the number of relevant chunks and return:
  relevant = True   if at least 60% of chunks are relevant.
  relevant = False  if fewer than 60% of chunks are relevant.
"""

_relevance_llm = ChatAnthropic(
    model="claude-sonnet-4-6", temperature=0
).with_structured_output(RelevanceGrade, method="json_schema")


def retrieval_relevance(inputs: dict, outputs: dict) -> EvaluationResult:
    """LangSmith evaluator: are the retrieved chunks relevant to the question?"""
    report = outputs["certified_report"]
    chunks_text = "\n\n---\n\n".join(
        f"[Leaf {leaf.index}]\n{leaf.text[:500]}..."
        for leaf in report.leaves
    )
    user_msg = f"QUESTION: {inputs['question']}\n\nSOURCE CHUNKS:\n{chunks_text}"

    grade = _relevance_llm.invoke(
        [
            {"role": "system", "content": _RETRIEVAL_RELEVANCE_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )
    precision = (
        grade["relevant_leaf_count"] / grade["total_leaf_count"]
        if grade["total_leaf_count"] > 0
        else 0.0
    )
    return EvaluationResult(
        key="retrieval_relevance",
        score=precision,
        comment=grade["explanation"],
    )
```

### 4. Merkle Integrity Evaluator

A deterministic (code-based) evaluator — no LLM involved. Recomputes the entire Merkle tree from the stored leaves and verifies the root hash. This must always score 1.0; any failure is a critical bug.

```python
from mara.merkle.tree import build_merkle_tree

def merkle_integrity(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    LangSmith evaluator: does the Merkle tree recompute to the stored root?
    Deterministic. Must always score 1.0 in correct runs.
    """
    report = outputs["certified_report"]
    recomputed_tree = build_merkle_tree(report.leaves)
    match = recomputed_tree.root == report.merkle_root

    return EvaluationResult(
        key="merkle_integrity",
        score=1.0 if match else 0.0,
        comment=(
            "Root hash matches."
            if match
            else (
                f"Root mismatch. Expected {report.merkle_root[:16]}... "
                f"got {recomputed_tree.root[:16]}..."
            )
        ),
    )
```

### 5. Faithfulness (Citation Validity) Evaluator

Checks that every `[ML:N:hash]` inline citation in the report text refers to a leaf index that exists in the Merkle tree. Catches cases where the synthesizer fabricated a citation index or hallucinated a leaf reference.

```python
import re

def faithfulness(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    LangSmith evaluator: do all inline citations point to real Merkle leaves?
    """
    report = outputs["certified_report"]
    # Extract all [ML:<index>:<hash_prefix>] citations
    cited_indices = set(
        int(m.group(1))
        for m in re.finditer(r"\[ML:(\d+):[0-9a-f]+\]", report.report_text)
    )
    tree_indices = {leaf.index for leaf in report.leaves}
    dangling = cited_indices - tree_indices

    return EvaluationResult(
        key="faithfulness",
        score=1.0 if not dangling else 0.0,
        comment=(
            f"All {len(cited_indices)} citations valid."
            if not dangling
            else f"Dangling citation indices: {sorted(dangling)}"
        ),
    )
```

### 6. Confidence Coverage Evaluator

Measures the fraction of claims that cleared the high-confidence threshold and went directly to the report without triggering corrective RAG or HITL. Higher is better, but very high values (> 0.95) on adversarial questions may indicate the confidence model is over-optimistic.

```python
def confidence_coverage(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    LangSmith evaluator: what fraction of claims exceeded the high threshold?
    """
    report = outputs["certified_report"]
    config = report.config_snapshot
    high_threshold = config.get("high_confidence_threshold", 0.80)

    claims = report.scored_claims
    if not claims:
        return EvaluationResult(key="confidence_coverage", score=None, comment="No claims.")

    high_conf_count = sum(1 for c in claims if c.confidence >= high_threshold)
    score = high_conf_count / len(claims)

    return EvaluationResult(
        key="confidence_coverage",
        score=score,
        comment=(
            f"{high_conf_count}/{len(claims)} claims above threshold {high_threshold}. "
            f"Mean confidence: {sum(c.confidence for c in claims) / len(claims):.3f}"
        ),
    )
```

### 7. HITL Trigger Rate Evaluator

For adversarial cases in the golden dataset where `inputs["expected_routing"] == "hitl"`, verifies that the graph did actually pause for human review. This ensures the safety net fires correctly on low-evidence questions.

```python
def hitl_trigger_rate(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    LangSmith evaluator: for questions labelled expected_routing=hitl,
    did the graph trigger the HITL checkpoint?
    """
    expected = inputs.get("expected_routing")
    if expected != "hitl":
        return EvaluationResult(key="hitl_trigger_rate", score=None, comment="N/A — not an adversarial case.")

    report = outputs["certified_report"]
    # Any claim below low_confidence_threshold means HITL was triggered
    config = report.config_snapshot
    low_threshold = config.get("low_confidence_threshold", 0.55)
    hitl_fired = any(c.confidence < low_threshold for c in report.scored_claims)

    return EvaluationResult(
        key="hitl_trigger_rate",
        score=1.0 if hitl_fired else 0.0,
        comment="HITL fired as expected." if hitl_fired else "HITL did not fire — possible over-confidence.",
    )
```

---

## Prompt-Specific Evaluators

Each of the four LLM prompts in MARA needs its own focused evaluation. These run against smaller, prompt-specific datasets rather than the full research quality dataset.

### Query Planner Prompt

**What can go wrong:** Sub-queries are too broad (returns noise), too narrow (misses coverage), duplicated, or fail to decompose a compound question into independent sub-queries.

**Dataset:** 30 research questions with reference decompositions created by a domain expert.

**Evaluator:**

```python
class PlannerGrade(TypedDict):
    explanation: str
    coverage_score: float   # 0.0–1.0: do sub-queries cover the full question?
    redundancy_score: float # 0.0–1.0: are sub-queries independent (low = better)?
    parseable: bool         # did the structured output schema validate?

_PLANNER_EVAL_PROMPT = """\
You are evaluating the output of a query decomposition step.

ORIGINAL QUESTION: {question}
GENERATED SUB-QUERIES: {sub_queries}
REFERENCE SUB-QUERIES (expert-written): {reference_sub_queries}

Score two dimensions (0.0–1.0):
  coverage_score: What fraction of the REFERENCE sub-queries are meaningfully
                  addressed by the generated set? (1.0 = full coverage)
  redundancy_score: How much overlap is there between the generated sub-queries?
                    (1.0 = high overlap = bad; 0.0 = fully independent = good)

Return your scores and a brief explanation.
"""

def query_planner_quality(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> list[EvaluationResult]:
    """Evaluates query planner output on coverage and redundancy."""
    grade_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).with_structured_output(
        PlannerGrade, method="json_schema"
    )
    sub_queries_text = "\n".join(
        f"  - [{sq.domain}] {sq.query}" for sq in outputs["sub_queries"]
    )
    ref_text = "\n".join(
        f"  - {q}" for q in reference_outputs.get("sub_queries", [])
    )
    grade = grade_llm.invoke(
        _PLANNER_EVAL_PROMPT.format(
            question=inputs["question"],
            sub_queries=sub_queries_text,
            reference_sub_queries=ref_text,
        )
    )
    return [
        EvaluationResult(key="planner_coverage", score=grade["coverage_score"], comment=grade["explanation"]),
        EvaluationResult(key="planner_redundancy", score=1.0 - grade["redundancy_score"], comment=grade["explanation"]),
        EvaluationResult(key="planner_parseable", score=1.0 if grade["parseable"] else 0.0),
    ]
```

### Claim Extractor Prompt

**What can go wrong:** Claims are compound (multiple assertions in one), claim boundaries are wrong, source-chunk attribution is incorrect, claims are hallucinated (not actually in the source text).

**Evaluators:**

```python
def claim_atomicity(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    Code-based evaluator: are extracted claims single-assertion sentences?
    Heuristic: a claim containing ' and ' or '; ' or multiple verbs is likely compound.
    Flags any claim with more than one main verb clause.
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")

    claims = outputs["extracted_claims"]
    compound_count = 0
    for claim in claims:
        doc = nlp(claim.text)
        verb_count = sum(1 for token in doc if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj"))
        if verb_count > 1:
            compound_count += 1

    score = 1.0 - (compound_count / len(claims)) if claims else 1.0
    return EvaluationResult(
        key="claim_atomicity",
        score=score,
        comment=f"{compound_count}/{len(claims)} claims appear compound.",
    )


def claim_attribution_validity(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    Code-based evaluator: does every claim correctly cite at least one source chunk?
    Checks that claim.source_chunk_indices are non-empty and within the valid range.
    """
    leaves = outputs["certified_report"].leaves
    valid_indices = {leaf.index for leaf in leaves}
    claims = outputs["extracted_claims"]

    invalid_count = sum(
        1 for c in claims
        if not c.source_chunk_indices or not set(c.source_chunk_indices).issubset(valid_indices)
    )
    score = 1.0 - (invalid_count / len(claims)) if claims else 1.0
    return EvaluationResult(
        key="claim_attribution_validity",
        score=score,
        comment=f"{invalid_count}/{len(claims)} claims have invalid or missing source attribution.",
    )
```

### LSA Scorer Prompt

**What can go wrong:** The LSA prompt returns "supported" for claims that the retrieved sources actually contradict. Or it returns "unsupported" for claims that are clearly in the sources (false negative, wastes HITL capacity).

**Dataset:** 50 (claim, sources, correct_label) triples, manually labelled as `supported | partially_supported | unsupported`.

**Evaluator:**

```python
def lsa_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluationResult:
    """
    Evaluates LSA prompt accuracy on a labelled dataset of claim-source pairs.
    Expected reference_outputs: {"lsa_label": "supported" | "partially_supported" | "unsupported"}
    Expected outputs: {"lsa_raw_score": 1.0 | 0.5 | 0.0}
    """
    label_to_score = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}
    expected = label_to_score.get(reference_outputs.get("lsa_label"), None)
    actual = outputs.get("lsa_raw_score")

    if expected is None or actual is None:
        return EvaluationResult(key="lsa_accuracy", score=None, comment="Missing label or score.")

    exact_match = abs(expected - actual) < 0.01
    return EvaluationResult(
        key="lsa_accuracy",
        score=1.0 if exact_match else 0.0,
        comment=f"Expected {expected}, got {actual}.",
    )
```

### Report Synthesizer Prompt

**What can go wrong:** The synthesizer introduces new background-knowledge facts not in the approved claim set, drops claims that should appear, or malforms inline `[ML:N:hash]` citations.

```python
import re

_NEW_CLAIMS_PROMPT = """\
You are an auditor.

APPROVED CLAIMS (the only facts the report is allowed to contain):
{approved_claims}

REPORT:
{report_text}

Does the report contain any factual statement that is NOT derivable from the approved claims?
A fact is "new" if it cannot be logically inferred from the approved claims alone.
Return a list of new (unapproved) facts found, or an empty list if none.
"""

class NewFactsGrade(TypedDict):
    new_facts: list[str]

_synth_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).with_structured_output(
    NewFactsGrade, method="json_schema"
)


def synthesizer_no_new_facts(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    LLM evaluator: does the report synthesizer stay within the approved claim set?
    """
    report = outputs["certified_report"]
    approved_text = "\n".join(
        f"  - {c.text}" for c in report.scored_claims
        if c.confidence >= report.config_snapshot.get("high_confidence_threshold", 0.80)
        or c.human_approved
    )
    grade = _synth_llm.invoke(
        _NEW_CLAIMS_PROMPT.format(
            approved_claims=approved_text,
            report_text=report.report_text,
        )
    )
    new_facts = grade["new_facts"]
    return EvaluationResult(
        key="synthesizer_no_new_facts",
        score=1.0 if not new_facts else 0.0,
        comment=(
            "No new facts introduced."
            if not new_facts
            else f"New facts found: {new_facts}"
        ),
    )


def synthesizer_citation_format(inputs: dict, outputs: dict) -> EvaluationResult:
    """
    Code-based evaluator: are all inline citations correctly formatted?
    Valid format: [ML:<int>:[0-9a-f]{6,}]
    """
    report = outputs["certified_report"]
    # Find all intended citation markers (any square-bracket content referencing ML)
    all_brackets = re.findall(r"\[ML:[^\]]+\]", report.report_text)
    valid_pattern = re.compile(r"^\[ML:\d+:[0-9a-f]{6,}\]$")
    malformed = [b for b in all_brackets if not valid_pattern.match(b)]

    return EvaluationResult(
        key="synthesizer_citation_format",
        score=1.0 if not malformed else 0.0,
        comment=(
            f"All {len(all_brackets)} citations correctly formatted."
            if not malformed
            else f"Malformed citations: {malformed}"
        ),
    )
```

---

## Running LangSmith Experiments

```python
# evaluation/run_experiment.py
from langsmith import Client
from mara.agent.graph import compile_graph
from mara.config import ResearchConfig

client = Client()


def run_mara(inputs: dict) -> dict:
    """Adapter: LangSmith inputs dict → CertifiedReport dict."""
    config = ResearchConfig(
        model="claude-sonnet-4-6",
        high_confidence_threshold=0.80,
        low_confidence_threshold=0.55,
    )
    graph = compile_graph(config)
    result = graph.invoke(
        {"query": inputs["question"], "config": config},
        config={"configurable": {"thread_id": inputs.get("thread_id", "eval-run")}},
    )
    return {"certified_report": result["certified_report"]}


# Baseline experiment
experiment_results = client.evaluate(
    run_mara,
    data="mara-research-quality-v1",
    evaluators=[
        groundedness,
        correctness,
        retrieval_relevance,
        merkle_integrity,
        faithfulness,
        confidence_coverage,
        hitl_trigger_rate,
    ],
    experiment_prefix="mara-baseline",
    metadata={
        "model": "claude-sonnet-4-6",
        "confidence_weights": {"alpha": 0.4, "beta": 0.2, "gamma": 0.4},
        "version": "v1.0",
    },
    max_concurrency=4,
)

# View results
df = experiment_results.to_pandas()
print(df[["input.question", "groundedness", "correctness", "merkle_integrity"]].to_string())
```

To compare a prompt change against the baseline, run a second experiment with only the changed config and use LangSmith's experiment comparison view to diff per-question scores.

---

## LangGraph Node Unit Tests

These tests exercise individual nodes in isolation using `MemorySaver` and mocked external API clients. They run in CI on every commit and gate merges via `--cov-fail-under=98`.

```python
# tests/agent/nodes/test_confidence_scorer.py
import pytest
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from mara.agent.state import MARState
from mara.agent.nodes.confidence_scorer import confidence_scorer_node

@pytest.fixture
def sample_state_with_claims():
    """State with pre-extracted claims and source chunks."""
    return {
        "query": "What are the effects of ocean acidification on coral?",
        "config": {"model": "claude-sonnet-4-6", "similarity_support_threshold": 0.72,
                   "confidence_weights": {"alpha": 0.4, "beta": 0.2, "gamma": 0.4}},
        "extracted_claims": [
            MagicMock(text="Coral calcification rates decrease in acidic water.", source_chunk_indices=[0, 1]),
            MagicMock(text="Some coral species show adaptation responses.", source_chunk_indices=[2]),
        ],
        "merkle_leaves": [
            MagicMock(index=0, text="Studies show coral calcification drops 15–30% under pH 7.9..."),
            MagicMock(index=1, text="Coral reef dissolution rates accelerate in acidic conditions..."),
            MagicMock(index=2, text="A subset of Porites corals exhibited partial resilience..."),
        ],
        "scored_claims": [],
        "loop_count": 0,
        "messages": [],
    }


def test_confidence_scorer_returns_scored_claims(sample_state_with_claims):
    """Confidence scorer must populate scored_claims for every extracted claim."""
    with patch("mara.agent.nodes.confidence_scorer.call_lsa") as mock_lsa:
        mock_lsa.return_value = 1.0  # all claims "supported" by LSA
        result = confidence_scorer_node(sample_state_with_claims)

    assert len(result["scored_claims"]) == 2
    for claim in result["scored_claims"]:
        assert 0.0 <= claim.confidence <= 1.0
        assert claim.sa is not None
        assert claim.csc is not None
        assert claim.lsa is not None


def test_confidence_scorer_routes_low_score_to_hitl(sample_state_with_claims):
    """A claim with 0 supporting sources must score below low_confidence_threshold."""
    # Remove all leaves so no source supports any claim
    sample_state_with_claims["merkle_leaves"] = []
    with patch("mara.agent.nodes.confidence_scorer.call_lsa") as mock_lsa:
        mock_lsa.return_value = 0.0
        result = confidence_scorer_node(sample_state_with_claims)

    low_threshold = 0.55
    assert all(c.confidence < low_threshold for c in result["scored_claims"]), (
        "With zero supporting sources and LSA=0, all claims should score below the HITL threshold."
    )
```

```python
# tests/agent/nodes/test_hitl_checkpoint.py
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from mara.agent.graph import compile_graph
from mara.config import ResearchConfig


def test_hitl_interrupt_fires_and_resumes():
    """
    When a claim's confidence is below the low threshold, the graph must
    interrupt at hitl_checkpoint. Resume with 'approve' must complete the run.
    """
    config = ResearchConfig(low_confidence_threshold=0.99)  # force all claims to HITL
    graph = compile_graph(config, checkpointer=MemorySaver())
    thread_cfg = {"configurable": {"thread_id": "test-hitl-1"}}

    # Inject a pre-scored state that will trigger HITL
    graph.update_state(
        thread_cfg,
        values={
            "scored_claims": [
                type("ScoredClaim", (), {"confidence": 0.3, "text": "Test claim.", "human_approved": False})()
            ],
            "loop_count": 0,
            "messages": [],
        },
        as_node="confidence_scorer",
    )

    # Run up to the interrupt
    snapshot = graph.invoke(None, thread_cfg)
    assert graph.get_state(thread_cfg).next == ("hitl_checkpoint",)

    # Resume with human approval
    final_state = graph.invoke(
        Command(resume={"decisions": [{"action": "approve", "claim_index": 0}]}),
        thread_cfg,
    )
    assert final_state["certified_report"] is not None
```

---

## Integration Test Strategy

The integration test suite in `tests/integration/test_full_run.py` runs the complete graph end-to-end with all external I/O mocked. It validates three critical routing scenarios:

```python
# tests/integration/test_full_run.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langgraph.checkpoint.memory import MemorySaver
from mara.agent.graph import compile_graph
from mara.config import ResearchConfig


@pytest.fixture
def mock_brave_results():
    return [MagicMock(url="https://example.com/source-1", title="Test Source", description="...")]


@pytest.fixture
def mock_firecrawl_chunks():
    return [
        MagicMock(url="https://example.com/source-1", text="Finland's UBI pilot ran from 2017–2018 and gave €560/month to 2,000 unemployed people.", retrieved_at="2024-01-01T00:00:00Z"),
    ]


@pytest.mark.asyncio
async def test_full_run_fast_path(mock_brave_results, mock_firecrawl_chunks):
    """
    A well-evidenced question should complete on the fast path:
    confidence ≥ 0.80, no corrective RAG, no HITL.
    """
    config = ResearchConfig(high_confidence_threshold=0.70, low_confidence_threshold=0.40)
    graph = compile_graph(config, checkpointer=MemorySaver())

    with (
        patch("mara.agent.nodes.search_worker.brave.BraveClient.search", return_value=mock_brave_results),
        patch("mara.agent.nodes.search_worker.firecrawl.FirecrawlClient.scrape_batch", return_value=mock_firecrawl_chunks),
        patch("mara.agent.nodes.confidence_scorer.call_lsa", return_value=1.0),
    ):
        result = graph.invoke(
            {"query": "What were the results of Finland's UBI pilot?", "config": config},
            config={"configurable": {"thread_id": "test-fast-1"}},
        )

    report = result["certified_report"]
    assert report is not None
    assert report.merkle_root, "Merkle root must be set."
    assert len(report.leaves) > 0, "At least one source chunk must be committed."

    # Verify Merkle integrity inline
    from mara.merkle.tree import build_merkle_tree
    assert build_merkle_tree(report.leaves).root == report.merkle_root


@pytest.mark.asyncio
async def test_full_run_corrective_rag_fires(mock_brave_results, mock_firecrawl_chunks):
    """
    Mid-confidence claims (0.55–0.80) should trigger a corrective RAG loop
    and re-score, not proceed directly to the report.
    """
    config = ResearchConfig(
        high_confidence_threshold=0.90,  # set high so initial run falls into corrective range
        low_confidence_threshold=0.40,
        max_corrective_rag_loops=1,
    )
    graph = compile_graph(config, checkpointer=MemorySaver())
    call_counts = {"lsa": 0}

    def counting_lsa(*args, **kwargs):
        call_counts["lsa"] += 1
        return 0.70  # forces corrective RAG on first pass, then passes

    with (
        patch("mara.agent.nodes.search_worker.brave.BraveClient.search", return_value=mock_brave_results),
        patch("mara.agent.nodes.search_worker.firecrawl.FirecrawlClient.scrape_batch", return_value=mock_firecrawl_chunks),
        patch("mara.agent.nodes.confidence_scorer.call_lsa", side_effect=counting_lsa),
    ):
        result = graph.invoke(
            {"query": "Complex synthesis question requiring multiple sources.", "config": config},
            config={"configurable": {"thread_id": "test-corrective-1"}},
        )

    # LSA should have been called more than once (initial pass + corrective loop)
    assert call_counts["lsa"] > 1, "Corrective RAG should have triggered at least one re-score."
```

---

## Offline vs. Online Evaluation

| Stage               | What                                           | When                            | Tool                        |
| ------------------- | ---------------------------------------------- | ------------------------------- | --------------------------- |
| **Pre-commit**      | Merkle unit tests, confidence formula tests    | Every commit                    | pytest                      |
| **CI (merge gate)** | All node tests, integration test, 98% coverage | Every PR                        | pytest + GitHub Actions     |
| **Pre-deploy**      | Full LangSmith experiment on golden dataset    | Before each model/prompt update | `client.evaluate()`         |
| **Post-deploy**     | Online eval on sampled live traffic            | Continuously                    | LangSmith online evaluators |
| **Weekly**          | Human spot-check of 10 random CertifiedReports | Weekly                          | Annotation queues           |

The online evaluators run asynchronously on a 10% sample of production traces. The `groundedness` and `faithfulness` evaluators are best suited for online use as they require only the certified report output — no reference answer. The `correctness` evaluator requires a reference answer and is therefore offline-only.

---

## Confidence Weight Optimisation

As LangSmith traces accumulate, the confidence weights `α` (SA), `β` (CSC), and `γ` (LSA) can be tuned empirically. The process:

**1. Collect ground truth labels from human reviewers.**

For each completed run, ask domain experts: "Is this claim accurate given the cited sources?" Record `human_accurate: bool` for each `ScoredClaim`. Store this in the LangSmith run annotations.

**2. Define the calibration objective.**

A well-calibrated confidence model should have: claims with `confidence ≥ 0.80` are accurate ≥ 85% of the time, and claims with `confidence < 0.55` are accurate < 50% of the time. Use Brier Score to measure calibration.

**3. Run a weight grid search.**

```python
# evaluation/weight_optimisation.py
import itertools
import numpy as np
from mara.confidence.scorer import recompute_confidence

alphas = [0.3, 0.4, 0.5]
betas  = [0.1, 0.2, 0.3]
gammas = [0.3, 0.4, 0.5]

best_weights = None
best_brier = float("inf")

for alpha, beta, gamma in itertools.product(alphas, betas, gammas):
    if abs(alpha + beta + gamma - 1.0) > 0.01:
        continue  # weights must sum to 1.0
    scores = [
        recompute_confidence(claim, alpha=alpha, beta=beta, gamma=gamma)
        for claim in annotated_claims
    ]
    labels = [c.human_accurate for c in annotated_claims]
    brier = np.mean([(s - int(l)) ** 2 for s, l in zip(scores, labels)])
    if brier < best_brier:
        best_brier = brier
        best_weights = (alpha, beta, gamma)

print(f"Best weights: α={best_weights[0]}, β={best_weights[1]}, γ={best_weights[2]}")
print(f"Best Brier score: {best_brier:.4f}")
```

**4. Validate on held-out dataset before deploying updated weights.**

Run a new LangSmith experiment with the optimised weights and confirm that `confidence_coverage` improves and `hitl_trigger_rate` still fires correctly on adversarial cases.

---

## Evaluation Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│  Change type              → Evaluation action                   │
├───────────────────────────┼─────────────────────────────────────┤
│  Any code change          → pytest (CI gate)                    │
│  Prompt update            → LangSmith experiment vs. baseline   │
│  Model upgrade            → Full golden dataset experiment       │
│  Confidence weight change → Weight grid search + held-out eval  │
│  New failure in prod      → Add to golden dataset → rerun       │
│  Weekly cadence           → Human annotation spot-check         │
└───────────────────────────┴─────────────────────────────────────┘
```

The guiding principle: **a run that passes `merkle_integrity` and `faithfulness` but fails `groundedness` or `correctness` is a prompt problem, not a cryptography problem.** Keeping these evaluation layers separate makes it straightforward to diagnose which part of the stack degraded.

---

_Sources consulted: LangSmith evaluation docs (RAG evaluation tutorial, custom evaluators), LangGraph testing docs (MemorySaver, partial execution, subgraph testing), MARA architecture specification._
