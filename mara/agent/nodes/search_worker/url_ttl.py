"""URL-based TTL classifier for the leaf freshness cache.

Returns how long (in hours) leaves scraped from a given URL should be
considered fresh.  Callers pass their configured default so this function
never hard-codes a fallback — the default comes from ResearchConfig.

TTL tiers
---------
float('inf') — immutable
    Versioned ArXiv PDFs (``/abs/XXXX.XXXXXvN``) and DOI-resolved URLs
    (``doi.org/``) are canonical, versioned identifiers: the content at that
    URL is frozen by design.  We never need to re-scrape them.

8 760 h (≈ 1 year) — long-lived academic content
    Direct PDF links and major peer-reviewed publisher domains
    (nature.com, science.org, cell.com, ncbi.nlm.nih.gov, pubmed.ncbi).
    These change only on publisher corrections, which are rare.

720 h (30 days) — semi-stable reference pages
    Wikipedia articles, Semantic Scholar paper pages, ResearchGate, and
    unversioned ArXiv abstract pages.  Content can change (Wikipedia edits,
    S2 metadata updates) but typically stays stable for weeks.

default — everything else
    Caller-supplied value (``ResearchConfig.leaf_cache_max_age_hours``).
    Defaults to 336 h (14 days) in production.

Design notes
------------
- Patterns are tested in TTL-descending order so the cheapest early-exit
  (immutable) fires first.
- ``float('inf')`` is a sentinel understood by
  ``SQLiteLeafRepository.get_fresh_leaves_for_url``: when TTL is infinite
  the cutoff comparison is skipped and cached leaves are always returned.
- The function takes ``default`` as a parameter rather than importing
  ResearchConfig to keep it dependency-free and trivially testable.
"""

import re

# ---------------------------------------------------------------------------
# Compiled patterns — ordered from most-specific / highest-TTL to least.
# ---------------------------------------------------------------------------

# Versioned ArXiv abstract page or PDF: .../abs/2301.00001v2 or .../pdf/...v2
_RE_ARXIV_VERSIONED = re.compile(
    r"arxiv\.org/(?:abs|pdf)/\d{4}\.\d{4,5}v\d+", re.IGNORECASE
)

# DOI resolver — doi.org/10.XXXX/... or dx.doi.org/...
_RE_DOI = re.compile(r"(?:dx\.)?doi\.org/10\.", re.IGNORECASE)

# Direct PDF link (any domain)
_RE_PDF = re.compile(r"\.pdf(?:[?#].*)?$", re.IGNORECASE)

# Major peer-reviewed publisher / biomedical databases
# Uses negative lookbehind so the pattern matches whether the domain is
# the full host (after "://") or a subdomain (after ".").
_RE_ACADEMIC_DOMAINS = re.compile(
    r"(?<![a-zA-Z0-9-])(?:"
    r"nature\.com|"
    r"science\.org|"
    r"cell\.com|"
    r"ncbi\.nlm\.nih\.gov|"
    r"pubmed\.ncbi\.nlm\.nih\.gov|"
    r"jamanetwork\.com|"
    r"nejm\.org|"
    r"thelancet\.com"
    r")(?:/|$)",
    re.IGNORECASE,
)

# Semi-stable reference pages (30 days)
_RE_SEMI_STABLE = re.compile(
    r"(?<![a-zA-Z0-9-])(?:"
    r"wikipedia\.org|"
    r"semanticscholar\.org|"
    r"researchgate\.net"
    r")(?:/|$)",
    re.IGNORECASE,
)

_HOURS_1_YEAR: float = 8_760.0
_HOURS_30_DAYS: float = 720.0


def url_ttl_hours(url: str, default: float) -> float:
    """Return the cache TTL in hours for *url*.

    Args:
        url:     The source URL to classify.
        default: Fallback TTL (hours) for URLs that match no pattern.
                 Should come from ``ResearchConfig.leaf_cache_max_age_hours``.

    Returns:
        ``float('inf')`` for immutable URLs, or a positive finite number of
        hours for everything else.
    """
    if _RE_ARXIV_VERSIONED.search(url):
        return float("inf")
    if _RE_DOI.search(url):
        return float("inf")
    if _RE_PDF.search(url):
        return _HOURS_1_YEAR
    if _RE_ACADEMIC_DOMAINS.search(url):
        return _HOURS_1_YEAR
    if _RE_SEMI_STABLE.search(url):
        return _HOURS_30_DAYS
    return default
