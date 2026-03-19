"""CertifiedReport serialization and persistence.

Provides save/load for CertifiedReport objects as self-contained JSON files.

JSON layout mirrors the CertifiedReport dataclass exactly:
  - leaves:         list of MerkleLeaf TypedDicts (JSON objects)
  - scored_claims:  list of ScoredClaim dataclasses (JSON objects)
  - all other fields are plain strings

A saved report is fully self-contained for offline verification: any reader can
recompute leaf hashes from (url, text, retrieved_at), rebuild the Merkle tree,
and confirm the root matches merkle_root without running the full agent.

Filename convention:
    YYYY-MM-DD_HHMMSS_<query_slug>_<root8>.json

where <root8> is the first 8 hex chars of the Merkle root — enough to
distinguish concurrent runs of the same query.
"""

import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path

from mara.agent.state import CertifiedReport
from mara.confidence.scorer import ScoredClaim
from mara.logging import get_logger

_log = get_logger(__name__)

DEFAULT_REPORT_DIR: Path = Path.home() / ".mara" / "reports"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def report_to_dict(report: CertifiedReport) -> dict:
    """Convert a CertifiedReport to a JSON-serializable dict.

    Uses ``dataclasses.asdict`` which recursively converts nested dataclasses
    (ScoredClaim) and leaves TypedDicts as plain dicts.
    """
    return dataclasses.asdict(report)


def report_from_dict(data: dict) -> CertifiedReport:
    """Reconstruct a CertifiedReport from a deserialised JSON dict.

    ``leaves`` are TypedDicts (plain dicts) and need no reconstruction.
    ``scored_claims`` are ScoredClaim dataclasses and must be rebuilt.
    """
    scored_claims = [ScoredClaim(**c) for c in data["scored_claims"]]
    return CertifiedReport(
        query=data["query"],
        report_text=data["report_text"],
        merkle_root=data["merkle_root"],
        leaves=data["leaves"],
        scored_claims=scored_claims,
        hash_algorithm=data.get("hash_algorithm", "sha256"),
        generated_at=data["generated_at"],
    )


# ---------------------------------------------------------------------------
# Filename
# ---------------------------------------------------------------------------


def _report_filename(report: CertifiedReport) -> str:
    """Generate a deterministic, human-readable filename for a report.

    Format: ``YYYY-MM-DD_HHMMSS_<query_slug>_<root8>.json``
    """
    ts = datetime.fromisoformat(report.generated_at).strftime("%Y-%m-%d_%H%M%S")
    slug = re.sub(r"[^\w]+", "_", report.query)[:40].strip("_").lower()
    short_root = report.merkle_root[:8] if report.merkle_root else "no_root"
    return f"{ts}_{slug}_{short_root}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_report(
    report: CertifiedReport,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> Path:
    """Serialise a CertifiedReport to a JSON file and return the saved path.

    Creates ``report_dir`` (and any parents) if it does not exist.

    Args:
        report:     The CertifiedReport to persist.
        report_dir: Directory in which to write the file.

    Returns:
        The absolute path of the written file.
    """
    report_dir = report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / _report_filename(report)
    path.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _log.info("Report saved: %s", path)
    return path


def load_report(path: Path) -> CertifiedReport:
    """Load a CertifiedReport from a JSON file produced by ``save_report``.

    Args:
        path: Path to the JSON file.

    Returns:
        A fully reconstructed CertifiedReport.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the JSON is missing required fields.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return report_from_dict(data)
