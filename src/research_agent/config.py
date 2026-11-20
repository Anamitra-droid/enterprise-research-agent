"""Central configuration, loaded from environment variables / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Providers -------------------------------------------------------
    # "auto" = use the real provider when its API key is present, otherwise the offline mock.
    llm_provider: Literal["auto", "openrouter", "mock"] = "auto"
    search_provider: Literal["auto", "tavily", "mock"] = "auto"
    embedding_provider: Literal["huggingface", "hash"] = "huggingface"

    openrouter_api_key: str | None = None
    tavily_api_key: str | None = None

    # --- OpenRouter (OpenAI-compatible gateway to many model providers) ---
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Optional attribution headers shown on openrouter.ai (HTTP-Referer / X-Title).
    openrouter_app_url: str = ""
    openrouter_app_name: str = "Enterprise Research Agent"
    # Extra models OpenRouter falls back to if the primary is unavailable/refuses (in order).
    openrouter_fallback_models: list[str] = Field(default_factory=list)

    # Smart tier: synthesis, conflict detection. Fast tier: classification, grading, checks.
    # Any OpenRouter model id with structured-output support works (see openrouter.ai/models).
    smart_model: str = "anthropic/claude-sonnet-5.5"
    fast_model: str = "anthropic/claude-haiku-5.5"
    llm_max_tokens: int = 4096
    llm_timeout_s: float = 60.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Storage ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    chroma_dir: Path = PROJECT_ROOT / "var" / "chroma"
    sqlite_path: Path = PROJECT_ROOT / "var" / "app.db"
    checkpoint_path: Path = PROJECT_ROOT / "var" / "checkpoints.db"
    collection_name: str = "ledgerleaf_kb"

    # --- Retrieval -------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_k: int = 6
    web_max_results: int = 5

    # --- Agent behaviour -------------------------------------------------
    max_retrieval_attempts: int = 2  # initial + 1 reformulated retry
    max_synthesis_attempts: int = 2  # initial + 1 regeneration on hallucination
    max_clarifications: int = 2
    tool_max_retries: int = 3
    tool_retry_base_delay_s: float = 0.5
    groundedness_threshold: float = 0.7
    low_confidence_threshold: float = 0.45
    history_summarize_after: int = 12  # messages
    history_keep_last: int = 6

    # Human-in-the-loop toggles
    hitl_confidential: bool = True
    hitl_low_confidence: bool = True
    hitl_web_privacy: bool = True
    hitl_kb_writes: bool = True

    # Terms that must never leak to external search without approval
    # (merged with `sensitive_terms` declared in confidential document front-matter).
    sensitive_terms: list[str] = Field(default_factory=lambda: ["Project Falcon", "salary band"])

    # --- Observability ---------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openrouter" if (self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")) else "mock"

    @property
    def resolved_search_provider(self) -> str:
        if self.search_provider != "auto":
            return self.search_provider
        return "tavily" if (self.tavily_api_key or os.getenv("TAVILY_API_KEY")) else "mock"

    def ensure_dirs(self) -> None:
        for p in (self.chroma_dir, self.sqlite_path.parent, self.checkpoint_path.parent):
            Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    # Make keys visible to SDKs that read env vars directly.
    if s.openrouter_api_key:
        os.environ.setdefault("OPENROUTER_API_KEY", s.openrouter_api_key)
    if s.tavily_api_key:
        os.environ.setdefault("TAVILY_API_KEY", s.tavily_api_key)
    return s
