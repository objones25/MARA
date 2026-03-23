"""Tests for mara.agent.nodes.search_worker.url_ttl."""

import math

import pytest

from mara.agent.nodes.search_worker.url_ttl import url_ttl_hours


class TestImmutableUrls:
    """Versioned ArXiv and DOI URLs should return float('inf')."""

    def test_arxiv_versioned_abs(self):
        url = "https://arxiv.org/abs/2301.00001v2"
        assert math.isinf(url_ttl_hours(url, 336.0))

    def test_arxiv_versioned_pdf(self):
        url = "https://arxiv.org/pdf/2301.00001v3"
        assert math.isinf(url_ttl_hours(url, 336.0))

    def test_arxiv_versioned_longer_id(self):
        url = "https://arxiv.org/abs/2301.12345v10"
        assert math.isinf(url_ttl_hours(url, 336.0))

    def test_doi_org(self):
        url = "https://doi.org/10.1038/s41586-021-03350-4"
        assert math.isinf(url_ttl_hours(url, 336.0))

    def test_dx_doi_org(self):
        url = "https://dx.doi.org/10.1016/j.cell.2021.01.001"
        assert math.isinf(url_ttl_hours(url, 336.0))


class TestUnversionedArxiv:
    """Unversioned ArXiv URLs are semi-stable, not immutable."""

    def test_unversioned_arxiv_abs(self):
        url = "https://arxiv.org/abs/2301.00001"
        ttl = url_ttl_hours(url, 336.0)
        assert not math.isinf(ttl)

    def test_unversioned_arxiv_falls_through_to_pdf_pattern(self):
        # arxiv.org/pdf/XXXX.XXXX (no version) ends in a path, not .pdf
        # so it uses the default
        url = "https://arxiv.org/pdf/2301.00001"
        ttl = url_ttl_hours(url, 336.0)
        # No version → not immutable; no .pdf extension → not 1-year
        # arxiv.org is not in academic domains list → falls to default
        assert ttl == 336.0


class TestLongLivedAcademicContent:
    """PDF links and major publisher domains → 8760 h (1 year)."""

    def test_direct_pdf_link(self):
        url = "https://example.com/papers/report.pdf"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_pdf_with_query_string(self):
        url = "https://example.com/doc.pdf?token=abc"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_pdf_with_fragment(self):
        url = "https://example.com/doc.pdf#page=3"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_nature_com(self):
        url = "https://www.nature.com/articles/s41586-021-03350-4"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_science_org(self):
        url = "https://www.science.org/doi/10.1126/science.abn2027"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_cell_com(self):
        url = "https://www.cell.com/cell/fulltext/S0092-8674(21)00007-1"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_ncbi_nlm_nih_gov(self):
        url = "https://ncbi.nlm.nih.gov/pmc/articles/PMC7930601/"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_pubmed_ncbi(self):
        url = "https://pubmed.ncbi.nlm.nih.gov/33674742/"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_nejm_org(self):
        url = "https://www.nejm.org/doi/full/10.1056/NEJMoa2034577"
        assert url_ttl_hours(url, 336.0) == 8_760.0

    def test_thelancet_com(self):
        url = "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)00796-X/fulltext"
        assert url_ttl_hours(url, 336.0) == 8_760.0


class TestSemiStableUrls:
    """Wikipedia, Semantic Scholar, ResearchGate → 720 h (30 days)."""

    def test_wikipedia(self):
        url = "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"
        assert url_ttl_hours(url, 336.0) == 720.0

    def test_wikipedia_other_lang(self):
        url = "https://de.wikipedia.org/wiki/Maschinelles_Lernen"
        assert url_ttl_hours(url, 336.0) == 720.0

    def test_semantic_scholar_paper(self):
        url = "https://www.semanticscholar.org/paper/CorpusId:12345678"
        assert url_ttl_hours(url, 336.0) == 720.0

    def test_researchgate(self):
        url = "https://www.researchgate.net/publication/123456789"
        assert url_ttl_hours(url, 336.0) == 720.0


class TestDefaultFallback:
    """URLs matching no pattern use the caller-supplied default."""

    def test_generic_news_site(self):
        assert url_ttl_hours("https://techcrunch.com/article/foo", 336.0) == 336.0

    def test_blog_post(self):
        assert url_ttl_hours("https://example.com/blog/post-title", 168.0) == 168.0

    def test_github_readme(self):
        assert url_ttl_hours("https://github.com/owner/repo", 336.0) == 336.0

    def test_default_propagated(self):
        """Callers can pass any default and it is returned unchanged."""
        assert url_ttl_hours("https://random.org/page", 999.5) == 999.5


class TestPatternPriority:
    """Immutable patterns take priority over less-specific ones."""

    def test_versioned_arxiv_pdf_not_classified_as_pdf_tier(self):
        # A versioned ArXiv PDF URL ends in a path component like /pdf/XXXX.XXXXvN
        # The versioned pattern fires first → float('inf'), not 8760
        url = "https://arxiv.org/pdf/2301.00001v2.pdf"
        assert math.isinf(url_ttl_hours(url, 336.0))

    def test_doi_on_academic_domain_returns_inf_not_1year(self):
        # nature.com article with DOI path: DOI pattern fires first
        url = "https://doi.org/10.1038/s41586-021-03350-4"
        assert math.isinf(url_ttl_hours(url, 336.0))
