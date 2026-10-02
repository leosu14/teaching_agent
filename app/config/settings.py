"""Application settings. Everything provider-specific comes from the environment or config files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.providers import ProviderSettings
from app.config.routing import ConfigError
from app.schemas.video import VideoConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TA_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "var"
    database_url: str | None = None
    object_store_dir: Path | None = None
    routing_file: Path = REPO_ROOT / "config" / "routing.toml"

    # LLM, TTS, image and search providers: unprefixed variables (LLM_PROVIDER, TTS_PROVIDER, ...), see
    # app/config/providers.py. Every capability defaults to its mock provider.
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    retrieval_provider: Literal["local"] = "local"
    video_composer: Literal["ffmpeg", "mock"] = "ffmpeg"  # ffmpeg: real MP4; mock: a manifest, for tests
    presentation_renderer: Literal["pptx", "mock"] = "pptx"  # pptx: local python-pptx; mock: JSON for tests
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
    presentation_aspect_ratio: Literal["16:9", "4:3"] = "16:9"
    presentation_max_slides: int = Field(default=20, ge=2, le=30)
    audio_failure_policy: Literal["fail", "continue"] = "fail"
    audio_language: str | None = Field(default=None, pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    audio_voice: str | None = None
    audio_speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    audio_format: str = "wav"
    audio_sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    audio_max_words_per_segment: int = Field(default=80, ge=5, le=400)
    audio_silent_slide_seconds: float = Field(default=3.0, gt=0, le=60)
    video_failure_policy: Literal["fail", "continue"] = "fail"  # fail: video is required; continue: optional
    # Output format overrides. Unset values keep the VideoConfig defaults (1920x1080, 30 fps, H.264/AAC in MP4).
    video_width: int | None = None
    video_height: int | None = None
    video_fps: int | None = None
    video_bitrate_kbps: int | None = None
    video_background: str | None = None
    video_transition: Literal["cut", "fade"] | None = None
    video_fade_seconds: float | None = None
    video_duration_tolerance: float | None = None
    video_subtitles: bool | None = None
    video_subtitle_max_chars: int | None = None
    video_font_path: Path | None = None  # a TrueType font for slide text and subtitles; default: Pillow's bundled one
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    video_timeout_seconds: float = Field(default=1200.0, gt=0)
    video_work_dir: Path | None = None  # scratch space for composition; default: <data_dir>/work
    video_keep_failed_work: bool = True  # keep a failed composition's scratch directory for diagnosis

    log_level: str = "INFO"
    log_json: bool = True

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'teaching_agent.db'}"

    @property
    def resolved_object_store_dir(self) -> Path:
        return self.object_store_dir or self.data_dir / "objects"

    @property
    def resolved_video_work_dir(self) -> Path:
        return self.video_work_dir or self.data_dir / "work"

    def video_config(self) -> VideoConfig:
        """VideoConfig with the configured overrides; everything unset keeps its single default in VideoConfig."""
        overrides = {"width": self.video_width, "height": self.video_height, "fps": self.video_fps,
                     "bitrate_kbps": self.video_bitrate_kbps, "background": self.video_background,
                     "transition": self.video_transition, "fade_seconds": self.video_fade_seconds,
                     "duration_tolerance": self.video_duration_tolerance}
        config = VideoConfig(**{k: v for k, v in overrides.items() if v is not None})
        subtitles = {"enabled": self.video_subtitles, "max_chars_per_line": self.video_subtitle_max_chars}
        subtitles = {k: v for k, v in subtitles.items() if v is not None}
        if subtitles:
            config = VideoConfig.model_validate({**config.model_dump(),
                                                 "subtitles": {**config.subtitles.model_dump(), **subtitles}})
        return config

    def validate_runtime(self) -> None:
        """Startup validation: provider configuration first (every problem at once, secrets never shown), then the
        local corpora the mock and local providers read."""
        self.providers.validate_startup()
        needed = {"knowledge_base.json"} if self.retrieval_provider == "local" else set()
        if self.providers.search_provider == "mock":
            needed.add("web_corpus.json")
        if self.providers.image_search_provider == "mock":
            needed.add("image_catalog.json")
        for name in sorted(needed):
            if not (self.corpus_dir / name).exists():
                raise ConfigError(f"TA_CORPUS_DIR={self.corpus_dir} is missing {name}")
