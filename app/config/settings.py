"""Application settings. Everything provider-specific comes from the environment or config files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.production import ProductionSettings
from app.config.providers import ProviderSettings
from app.config.routing import ConfigError
from app.schemas.curriculum import CurriculumConfig, FeasibilityConfig
from app.schemas.generative_video import GeneratedVideoConfig
from app.schemas.pedagogy import (
    AdaptiveQuestioningPolicy,
    DifficultyBands,
    PedagogyConfig,
    PlannerConfig,
)
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
    # Budget of a production task (MAX_LLM_REQUESTS, MAX_GENERATED_IMAGES, ...): see app/config/production.py.
    production: ProductionSettings = Field(default_factory=ProductionSettings)
    retrieval_provider: Literal["local"] = "local"
    video_composer: Literal["ffmpeg", "mock"] = "ffmpeg"  # ffmpeg: real MP4; mock: a manifest, for tests
    presentation_renderer: Literal["pptx", "mock"] = "pptx"  # pptx: local python-pptx; mock: JSON for tests
    corpus_dir: Path = REPO_ROOT / "fixtures" / "demo"

    max_revisions: int = Field(default=2, ge=0, le=10)
    revision_exhausted_policy: Literal["fail", "accept_with_warnings"] = "fail"
    diagnostic_max_rounds: int = Field(default=2, ge=1, le=5)
    diagnostic_memory_confidence: float = Field(default=0.6, ge=0, le=1)
    diagnostic_max_questions: int = Field(default=12, ge=1, le=50)  # adaptive questioning budget per diagnostic
    diagnostic_max_follow_ups: int = Field(default=1, ge=0, le=5)  # follow-up questions per missed concept
    # Adaptive pedagogy (see PedagogyConfig): difficulty bands, the mastery target and the lesson time budget.
    pedagogy_band_guided: float = Field(default=0.3, gt=0, lt=1)
    pedagogy_band_independent: float = Field(default=0.6, gt=0, lt=1)
    pedagogy_band_consolidation: float = Field(default=0.8, gt=0, lt=1)
    pedagogy_mastery_target: float = Field(default=0.8, gt=0, le=1)
    pedagogy_max_target_concepts: int = Field(default=2, ge=1, le=10)
    pedagogy_lesson_minutes: int | None = Field(default=None, ge=5, le=240)  # unset: the learner's session length
    curriculum_evidence_required: int = Field(default=2, ge=1, le=20)  # evidence before an objective is mastered
    curriculum_completion_rule: Literal["all_required_mastered", "targets_mastered"] = "all_required_mastered"
    curriculum_sessions_per_week: float = Field(default=3, gt=0, le=21)  # for target-date feasibility warnings
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

    # Optional generated video segments (a lesson asks for them with the "video.generated_segments" capability).
    # Budgets: MAX_GENERATED_VIDEO_SEGMENTS, MAX_GENERATED_VIDEO_SECONDS, MAX_VIDEO_GENERATION_COST_USD; the
    # provider: VIDEO_GENERATION_PROVIDER (see app/config/providers.py). Unset values keep GeneratedVideoConfig's.
    generated_video_enabled: bool = True
    generated_video_min_seconds: float | None = Field(default=None, gt=0, le=60)
    generated_video_max_seconds: float | None = Field(default=None, gt=0, le=60)
    generated_video_required: bool = False
    generated_video_failure_policy: Literal["fail", "continue"] = "fail"  # for required segments
    generated_video_strategy: Literal["full_frame_replace", "inset"] = "full_frame_replace"
    generated_video_poll_interval_seconds: float | None = Field(default=None, ge=0, le=600)
    generated_video_poll_timeout_seconds: float | None = Field(default=None, gt=0, le=86400)
    generated_video_poll_max_attempts: int | None = Field(default=None, ge=1, le=10000)

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

    def questioning_policy(self) -> AdaptiveQuestioningPolicy:
        return AdaptiveQuestioningPolicy(max_questions=self.diagnostic_max_questions,
                                         max_follow_ups_per_concept=self.diagnostic_max_follow_ups)

    def pedagogy_config(self) -> PedagogyConfig:
        """PedagogyConfig with the configured overrides; everything else keeps its single default."""
        return PedagogyConfig(
            bands=DifficultyBands(guided=self.pedagogy_band_guided, independent=self.pedagogy_band_independent,
                                  consolidation=self.pedagogy_band_consolidation),
            mastery_target=self.pedagogy_mastery_target,
            planner=PlannerConfig(max_target_concepts=self.pedagogy_max_target_concepts),
            questioning=self.questioning_policy())

    def curriculum_config(self) -> CurriculumConfig:
        """CurriculumConfig sharing the adaptive engine's thresholds (mastery target, prerequisite threshold,
        practice band, repeated failure), so a curriculum and the lessons it starts never disagree."""
        pedagogy = self.pedagogy_config()
        return CurriculumConfig(
            mastery_target=pedagogy.mastery_target, prerequisite_threshold=pedagogy.prerequisite_threshold,
            practice_from=min(pedagogy.bands.independent, pedagogy.mastery_target),
            repeated_failure_streak=pedagogy.repeated_failure_streak,
            evidence_required=self.curriculum_evidence_required, completion_rule=self.curriculum_completion_rule,
            feasibility=FeasibilityConfig(sessions_per_week=self.curriculum_sessions_per_week))

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

    def generated_video_config(self) -> GeneratedVideoConfig:
        """GeneratedVideoConfig: the production budget, the provider's configured price, then the overrides."""
        budget = self.production
        overrides = {
            "min_segment_seconds": self.generated_video_min_seconds,
            "max_segment_seconds": self.generated_video_max_seconds,
            "poll_interval_seconds": self.generated_video_poll_interval_seconds,
            "poll_timeout_seconds": self.generated_video_poll_timeout_seconds,
            "poll_max_attempts": self.generated_video_poll_max_attempts,
        }
        return GeneratedVideoConfig(
            max_segments=budget.max_generated_video_segments, max_total_seconds=budget.max_generated_video_seconds,
            max_cost_usd=budget.max_video_generation_cost_usd,
            price_per_second_usd=self.providers.video_generation_price_per_second,
            required=self.generated_video_required, failure_policy=self.generated_video_failure_policy,
            strategy=self.generated_video_strategy, **{k: v for k, v in overrides.items() if v is not None})

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
