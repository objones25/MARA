# MARA Roadmap

## Additional Data Sources

Potential source integrations in rough priority order.

### OpenAlex

- **Coverage**: Fully open catalog of scholarly works across all disciplines (economics, social science, medicine, STEM)
- **API**: Free, no auth required
- **Value**: Complements Semantic Scholar; broader discipline coverage than ArXiv alone
- **TTL**: Effectively permanent per versioned paper URL

### Wikipedia

- **Coverage**: Encyclopedic background and definitional content
- **API**: Free, structured access to full articles
- **Value**: High-quality, stable reference content not always surfaced by Brave. Useful for grounding definitional claims that currently score low in confidence scoring.
- **TTL**: 30-90 days (articles are edited frequently but stable enough for weeks)

### SSRN

- **Coverage**: Working papers in economics, finance, law, and social science
- **API**: No official API; accessible via scraping
- **Value**: Pre-publication research not on ArXiv; strong overlap with economic and policy topics

### PubMed

- **Coverage**: Biomedical and life science literature
- **API**: Free NCBI E-utilities API
- **Value**: Relevant if health, clinical, or biological economics topics are in scope

---

## TTL Improvements

Currently all leaves share a single `leaf_cache_max_age_hours` (default: 7 days). This is appropriate for news and general web pages but wasteful for stable content — ArXiv PDFs are versioned by URL and never change, yet are re-scraped every week.

**Proposed approach**: URL-pattern-based TTL overrides at freshness-check time in `sqlite_repository.py`. `leaf_cache_max_age_hours` remains the default fallback for all unrecognized URLs — no breaking changes, no schema changes.

| URL pattern | TTL |
| --- | --- |
| `arxiv.org/pdf/*v*` | Permanent (never expires) |
| `arxiv.org/html/*v*` | Permanent |
| Academic PDF (heuristic: `.pdf` on non-news domain) | 1 year |
| `wikipedia.org` | 60 days |
| Everything else | `leaf_cache_max_age_hours` (config default) |
