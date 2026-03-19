"""Cryptographic integrity verification for CertifiedReport objects.

Verification lifecycle
----------------------
1. For each leaf in the report, recompute hash_chunk(url, text, retrieved_at,
   algorithm) using the same canonical_serialise function used at research time.
2. Compare the recomputed hash to the stored leaf hash.
3. Build a Merkle tree from the recomputed leaf hashes (same algorithm and
   same leaf order as at research time).
4. Compare the recomputed root to report.merkle_root.

A report passes verification when:
  - Every leaf's recomputed hash matches its stored hash (no leaf was altered).
  - The recomputed Merkle root matches report.merkle_root (the root was produced
    from exactly these leaves, in this order, with this algorithm).

What verification proves
------------------------
- The source chunks existed at the recorded URLs at retrieval time.
- The agent did not fabricate the text of any citation.
- The Merkle root embedded in the report was produced from these exact sources.

What it does not prove
-----------------------
- That the live source URL still returns the same content (pages change over time).
  Use the --live flag of ``mara verify`` to check for content drift separately.
- That the agent's interpretation of the sources is correct (that is what the
  confidence scorer and HITL checkpoint address).
"""

from dataclasses import dataclass, field

from mara.agent.state import CertifiedReport
from mara.logging import get_logger
from mara.merkle.hasher import hash_chunk
from mara.merkle.tree import build_merkle_tree

_log = get_logger(__name__)


@dataclass
class LeafVerification:
    """Verification result for a single Merkle leaf.

    Attributes:
        index:          The leaf's position index (from leaf["index"]).
        url:            The source URL.
        expected_hash:  The hash stored in the report.
        computed_hash:  The hash recomputed from (url, text, retrieved_at).
        match:          True when expected_hash == computed_hash.
    """

    index: int
    url: str
    expected_hash: str
    computed_hash: str
    match: bool


@dataclass
class VerificationResult:
    """Aggregated result of verifying a CertifiedReport.

    Attributes:
        report_query:           The original research question.
        merkle_root_expected:   Root stored in the report.
        merkle_root_computed:   Root recomputed from recomputed leaf hashes.
        root_match:             True when both roots are identical.
        leaf_results:           Per-leaf verification outcomes (ordered as in
                                report.leaves).
    """

    report_query: str
    merkle_root_expected: str
    merkle_root_computed: str
    root_match: bool
    leaf_results: list[LeafVerification] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every leaf matches and the Merkle root matches."""
        return self.root_match and all(r.match for r in self.leaf_results)

    @property
    def failed_leaves(self) -> list[LeafVerification]:
        """Leaves whose recomputed hash did not match the stored hash."""
        return [r for r in self.leaf_results if not r.match]


def verify_report(report: CertifiedReport) -> VerificationResult:
    """Verify the cryptographic integrity of a CertifiedReport.

    Recomputes every leaf hash from its stored (url, text, retrieved_at) triple,
    rebuilds the Merkle tree, and checks the root against report.merkle_root.

    Args:
        report: A CertifiedReport loaded from disk or produced by the pipeline.

    Returns:
        A VerificationResult with per-leaf outcomes and an overall pass/fail.
    """
    algorithm = report.hash_algorithm
    _log.info(
        "Verifying report '%s…' — %d leaf/leaves, algorithm=%s",
        report.query[:40],
        len(report.leaves),
        algorithm,
    )

    leaf_results: list[LeafVerification] = []
    computed_hashes: list[str] = []

    for leaf in report.leaves:
        computed = hash_chunk(leaf["url"], leaf["text"], leaf["retrieved_at"], algorithm)
        computed_hashes.append(computed)
        leaf_results.append(
            LeafVerification(
                index=leaf["index"],
                url=leaf["url"],
                expected_hash=leaf["hash"],
                computed_hash=computed,
                match=computed == leaf["hash"],
            )
        )

    if computed_hashes:
        merkle_root_computed = build_merkle_tree(computed_hashes, algorithm).root
    else:
        merkle_root_computed = ""

    root_match = merkle_root_computed == report.merkle_root

    result = VerificationResult(
        report_query=report.query,
        merkle_root_expected=report.merkle_root,
        merkle_root_computed=merkle_root_computed,
        root_match=root_match,
        leaf_results=leaf_results,
    )

    if result.passed:
        _log.info("Verification PASSED — %d leaf/leaves verified", len(leaf_results))
    else:
        _log.warning(
            "Verification FAILED — %d leaf/leaves failed, root_match=%s",
            len(result.failed_leaves),
            root_match,
        )

    return result
