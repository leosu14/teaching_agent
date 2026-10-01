"""Application settings. Everything provider-specific comes from the environment or config files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.routing import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "var"
    database_url: str | None = None
    object_store_dir: Path | None = None
    routing_file: Path = REPO_ROOT / "config" / "routing.toml"

    llm_providers: list[str] = Field(default_factory=lambda: ["mock"])
    search_provider: Literal["mock"] = "mock"
    retrieval_provider: Literal["local"] = "local"
    image_provider: Literal["mock"] = "mock"  # image generation
    image_search_provider: Literal["mock"] = "mock"
    tts_provider: Literal["mock"] = "mock"
    video_provider: Literal["mock"] = "mock"
    corpus_dir: Path = REPO_ROOT / "fixtures" / "demo"

    max_revisions: int = Field(default=2, ge=0, le=10)
    revision_exhausted_policy: Literal["fail", "accept_with_warnings"] = "fail"
    diagnostic_max_rounds: int = Field(default=2, ge=1, le=5)
    diagnostic_memory_confidence: float = Field(default=0.6, ge=0, le=1)
    research_requirement: Literal["mandatory", "optional"] = "mandatory"
    research_max_results: int = Field(default=5, ge=1, le=50)
    research_max_sources: int = Field(default=6, ge=1, le=50)
    research_min_reliability: float = Field(default=0.5, ge=0, le=1)
    research_cache: bool = True
    visual_failure_policy: Literal["fail", "continue"] = "fail"
    visual_max_per_lesson: int = Field(default=6, ge=0, le=20)
    visual_max_candidates: int = Field(default=3, ge=1, le=10)

    log_level: str = "INFO"
    log_json: bool = True

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'teaching_agent.db'}"

    @property
    def resolved_object_store_dir(self) -> Path:
        return self.object_store_dir or self.data_dir / "objects"

    def validate_runtime(self) -> None:
        if not self.llm_providers:
            raise ConfigError("TA_LLM_PROVIDERS must name at least one LLM provider")
        if self.search_provider == "mock" or self.retrieval_provider == "local":
            for name in ("web_corpus.json", "knowledge_base.json", "image_catalog.json"):
                if not (self.corpus_dir / name).exists():
                    raise ConfigError(f"TA_CORPUS_DIR={self.corpus_dir} is missing {name}")
