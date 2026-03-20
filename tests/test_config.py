"""Tests for mara.config — ResearchConfig validators."""

import pytest
from pydantic import ValidationError

from mara.config import ResearchConfig


# ---------------------------------------------------------------------------
# ResearchConfig validators
# ---------------------------------------------------------------------------

class TestResearchConfigValidators:
    def test_defaults_are_valid(self):
        config = ResearchConfig()
        assert config.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def test_low_threshold_above_high_raises(self):
        with pytest.raises(ValidationError, match="low_confidence_threshold"):
            ResearchConfig(low_confidence_threshold=0.9, high_confidence_threshold=0.8)

    def test_equal_thresholds_raise(self):
        with pytest.raises(ValidationError, match="low_confidence_threshold"):
            ResearchConfig(low_confidence_threshold=0.75, high_confidence_threshold=0.75)

    def test_postgres_checkpointer_without_dsn_raises(self):
        with pytest.raises(ValidationError, match="postgres_dsn"):
            ResearchConfig(checkpointer="postgres", postgres_dsn="")

    def test_postgres_checkpointer_with_dsn_is_valid(self):
        config = ResearchConfig(
            checkpointer="postgres",
            postgres_dsn="postgresql://user:pass@localhost:5432/mara",
        )
        assert config.checkpointer == "postgres"

    def test_chunk_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValidationError, match="chunk_overlap"):
            ResearchConfig(chunk_size=500, chunk_overlap=500)

    def test_chunk_overlap_above_chunk_size_raises(self):
        with pytest.raises(ValidationError, match="chunk_overlap"):
            ResearchConfig(chunk_size=500, chunk_overlap=600)

    def test_invalid_checkpointer_value_raises(self):
        with pytest.raises(ValidationError):
            ResearchConfig(checkpointer="redis")

    def test_max_claim_sources_greater_than_max_retrieval_candidates_raises(self):
        with pytest.raises(ValidationError, match="max_claim_sources"):
            ResearchConfig(max_claim_sources=200, max_retrieval_candidates=100)

    def test_max_claim_sources_equal_to_max_retrieval_candidates_is_valid(self):
        config = ResearchConfig(max_claim_sources=100, max_retrieval_candidates=100)
        assert config.max_claim_sources == 100

    def test_max_claim_sources_less_than_max_retrieval_candidates_is_valid(self):
        config = ResearchConfig(max_claim_sources=50, max_retrieval_candidates=150)
        assert config.max_claim_sources == 50
