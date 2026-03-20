from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfidenceWeights(BaseModel):
    """Weights for the three confidence signals. Must sum to 1.0.

    Loaded as a nested model via MARA_CONFIDENCE_WEIGHTS__ALPHA etc.
    Sub-models must inherit BaseModel (not BaseSettings) when populated
    through the parent's env_nested_delimiter. See pydantic-settings docs.
    """

    alpha: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for source agreement rate (SA)")
    beta: float = Field(default=0.2, ge=0.0, le=1.0, description="Weight for cross-source consistency (CSC)")
    gamma: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for LLM self-assessment (LSA)")

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ConfidenceWeights":
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")
        return self


class ResearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MARA_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        populate_by_name=True,
        extra="ignore",
    )

    # API keys (no MARA_ prefix — standard names)
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    firecrawl_api_key: str = Field(default="", alias="FIRECRAWL_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN", description="HuggingFace Hub token for authenticated model downloads.")

    # LLM / embedding
    model: str = "Qwen/Qwen3-30B-A3B-Instruct"
    lsa_model: str = Field(default="Qwen/Qwen3-32B-Instruct", description="Model used for LLM Self-Assessment (LSA) scoring; Qwen3-32B offers better structured-output reliability.")
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    max_sources: int = Field(default=20, gt=0, description="Brave results requested per sub-query (Brave API hard cap is 20 per request)")
    max_workers: int = Field(default=3, gt=0)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    # Brave Search parameters
    brave_freshness: str = Field(
        default="",
        description="Optional freshness filter: 'pd' (24h), 'pw' (7d), 'pm' (31d), 'py' (1y), or 'YYYY-MM-DDtoYYYY-MM-DD'. Empty = no filter.",
    )

    # Confidence routing
    high_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    low_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    similarity_support_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    max_corrective_rag_loops: int = Field(default=2, ge=0)
    max_retrieval_candidates: int = Field(default=150, gt=0, description="Retrieval pool size; fed into reranker once implemented")
    max_claim_sources: int = Field(default=50, gt=0, description="Leaves passed to claim extraction after retrieval (and eventual reranking)")

    # Confidence weights (nested — populated via MARA_CONFIDENCE_WEIGHTS__*)
    confidence_weights: ConfidenceWeights = Field(default_factory=ConfidenceWeights)

    # Cryptography
    hash_algorithm: str = "sha256"

    # Infrastructure
    checkpointer: str = Field(default="memory", pattern="^(memory|postgres)$")
    postgres_dsn: str = ""

    # Leaf database
    leaf_db_path: str = Field(default="~/.mara/leaves.db", description="Path to the SQLite leaf database. Tilde-expanded at open time.")
    leaf_cache_max_age_hours: float = Field(default=168.0, gt=0.0, description="How long a scraped URL's leaves are considered fresh (default: 7 days).")
    leaf_db_enabled: bool = Field(default=True, description="Set to False to disable all DB reads/writes (useful in tests and CI).")

    # ArXiv
    arxiv_max_results: int = Field(default=5, gt=0, description="ArXiv papers requested per sub-query.")

    @model_validator(mode="after")
    def claim_sources_le_candidates(self) -> "ResearchConfig":
        if self.max_claim_sources > self.max_retrieval_candidates:
            raise ValueError(
                f"max_claim_sources ({self.max_claim_sources}) must be ≤ "
                f"max_retrieval_candidates ({self.max_retrieval_candidates})"
            )
        return self

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> "ResearchConfig":
        if self.low_confidence_threshold >= self.high_confidence_threshold:
            raise ValueError(
                f"low_confidence_threshold ({self.low_confidence_threshold}) must be "
                f"less than high_confidence_threshold ({self.high_confidence_threshold})"
            )
        return self

    @model_validator(mode="after")
    def postgres_dsn_required_when_postgres(self) -> "ResearchConfig":
        if self.checkpointer == "postgres" and not self.postgres_dsn:
            raise ValueError("postgres_dsn is required when checkpointer='postgres'")
        return self

    @model_validator(mode="after")
    def chunk_overlap_less_than_chunk_size(self) -> "ResearchConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self
