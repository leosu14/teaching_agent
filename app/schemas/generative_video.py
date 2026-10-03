"""Generative video schemas: optional short AI-generated clips inside a lesson video. Provider-independent.

Chain: the video strategy decides, per lesson section, whether a generated clip is worth it (a VideoSegmentDecision,
recorded for every section) and plans the selected ones (VideoSegmentPlan, inside a VideoSegmentPlanSet) -> each
plan becomes one provider request (VideoGenerationRequest) run as an asynchronous job (VideoGenerationJob) -> the
provider's file is validated from its bytes, normalised to the platform format and stored as a
GENERATED_VIDEO_ASSET artifact (VideoSegmentAsset) -> the existing VideoPlan places it on its slide
(GeneratedClip) -> the existing VideoComposer composes the final MP4.

Nothing here is specific to a vendor: provider adapters map their API onto these schemas. Generated video never
replaces narration or subtitles (the PresentationTimeline stays authoritative) and every segment has a fallback
(an existing IMAGE_ASSET or the slide itself).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.artifact import Artifact, StoredObject
from app.schemas.common import Schema, utcnow

# --- Provider-level request, job and result --------------------------------------------------------------------


class VideoGenerationStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (VideoGenerationStatus.COMPLETED, VideoGenerationStatus.FAILED,
                        VideoGenerationStatus.CANCELLED)


class VideoGenerationRequest(Schema):
    """One clip to generate. `prompt` comes from the controlled prompt builder only: educational content in a fixed
    structure, never raw user input and never learner data. `metadata` is ours (e.g. the segment id); adapters do
    not send it to the vendor."""

    prompt: str = Field(min_length=1, max_length=2000)
    duration: float = Field(gt=0, le=60)  # seconds
    aspect_ratio: str = Field(default="16:9", pattern=r"^[1-9]\d?:[1-9]\d?$")
    width: int = Field(ge=16, le=7680)  # the resolution asked for; a provider may only offer a few sizes
    height: int = Field(ge=16, le=4320)
    fps: int = Field(default=24, ge=1, le=120)
    seed: int | None = Field(default=None, ge=0)  # reproducibility, where the provider supports it
    with_audio: bool = False
    metadata: dict = Field(default_factory=dict)

    def request_hash(self) -> str:
        """Identity of what is asked: every field except our own metadata."""
        body = self.model_dump(mode="json", exclude={"metadata"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class VideoGenerationJob(Schema):
    """A provider-independent generation job. `job_id` is ours and deterministic for a generation key, so the same
    request is never submitted twice for the same task; `provider_job_id` is the vendor's."""

    job_id: str
    provider: str
    provider_job_id: str
    status: VideoGenerationStatus
    request: VideoGenerationRequest
    generation_key: str  # request + provider configuration (see GenerationKey)
    model: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    polls: int = Field(default=0, ge=0)
    error: str | None = None
    cancellation: Literal["provider", "local"] | None = None  # how a cancelled job was cancelled
    metadata: dict = Field(default_factory=dict)  # what the provider reported about the job (never credentials)

    def advanced(self, status: VideoGenerationStatus, *, error: str | None = None, **metadata) -> VideoGenerationJob:
        return self.model_copy(update={"status": status, "updated_at": utcnow(), "error": error,
                                       "metadata": {**self.metadata, **metadata}})


class VideoGenerationUsage(Schema):
    requests: int = Field(default=1, ge=0)
    seconds: float = Field(default=0.0, ge=0)  # generated seconds billed
    cost_usd: float | None = None  # only what the provider itself reports


class GeneratedVideoFile(Schema):
    """The bytes a provider returned for a completed job, plus what it claimed about them. The claims are recorded,
    never trusted: validation measures the bytes."""

    content: bytes
    media_type: str
    model: str
    reported_duration: float | None = None
    reported_width: int | None = None
    reported_height: int | None = None
    reported_fps: float | None = None
    usage: VideoGenerationUsage = Field(default_factory=VideoGenerationUsage)
    metadata: dict = Field(default_factory=dict)


class VideoGenerationResult(Schema):
    """A finished generation. The provider layer fills `content`; the generation service stores it and replaces it
    with `object`, the content-addressed reference. Measured values come from validation, not from the provider."""

    provider: str
    provider_job_id: str
    status: VideoGenerationStatus
    object: StoredObject | None = None
    content: bytes | None = Field(default=None, exclude=True)
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    format: str | None = None  # container, e.g. "mp4"
    media_type: str | None = None
    model: str | None = None
    usage: VideoGenerationUsage = Field(default_factory=VideoGenerationUsage)
    metadata: dict = Field(default_factory=dict)


# --- Planning ----------------------------------------------------------------------------------------------------


class VideoPurpose(str, Enum):
    """Why a section would benefit from moving pictures. Every value is subject-independent."""

    PHYSICAL_PROCESS = "physical_process"
    SCIENTIFIC_PROCESS = "scientific_process"
    HISTORICAL_SCENE = "historical_scene"
    GEOGRAPHICAL_MOVEMENT = "geographical_movement"
    PRONUNCIATION = "pronunciation"  # mouth/articulation demonstration
    VISUAL_STORYTELLING = "visual_storytelling"


class InsertionStrategy(str, Enum):
    """Where a generated clip appears in the composed video. Only the safest strategies are implemented
    (IMPLEMENTED_STRATEGIES); the others are reserved names so plans stay forward-compatible."""

    FULL_FRAME_REPLACE = "full_frame_replace"  # the clip replaces the slide picture; subtitles stay on top
    INSET = "inset"  # the clip plays in a box on the slide card
    OVERLAY = "overlay"
    INTRO = "intro"
    OUTRO = "outro"


IMPLEMENTED_STRATEGIES = frozenset({InsertionStrategy.FULL_FRAME_REPLACE, InsertionStrategy.INSET})
ClipAudio = Literal["muted", "mixed"]  # muted: narration only (the default); mixed: the clip's own sound under it


class SegmentFallback(Schema):
    """What the slide shows when its generated clip is not available: an existing IMAGE_ASSET, else the slide."""

    kind: Literal["image_asset", "slide"]
    slide_id: str
    artifact_id: str | None = None  # the IMAGE_ASSET, for kind image_asset

    @model_validator(mode="after")
    def _image(self) -> SegmentFallback:
        if (self.kind == "image_asset") != (self.artifact_id is not None):
            raise ValueError("an image_asset fallback names its IMAGE_ASSET and a slide fallback does not")
        return self


class ProviderPreferences(Schema):
    provider: str | None = None  # None: the configured provider
    with_audio: bool = False
    seed: int | None = None


class VideoPromptContent(Schema):
    """The educational content a prompt is built from: lesson material only, separated from provider instructions."""

    subject: str = Field(min_length=1, max_length=200)  # what is shown, e.g. the section heading
    visual_description: str = Field(min_length=1, max_length=600)  # what the viewer should see happen
    purpose: VideoPurpose
    language: str | None = None  # only when on-screen speech or text in that language matters (pronunciation)


class VideoCandidate(Schema):
    """A suggestion (e.g. from a model) that a section could use a generated clip. Suggestions only add weight:
    the deterministic policy decides."""

    lesson_section_id: str
    purpose: VideoPurpose
    rationale: str = ""


SkipReason = Literal[
    "exercise_text_sufficient", "review_text_sufficient", "short_factual_text", "definition_or_grammar",
    "no_visual_motion", "existing_image_sufficient", "no_slide", "duplicate_slide", "duration_unsupported",
    "format_unsupported", "budget_segments", "budget_seconds", "budget_cost",
]


class VideoSegmentDecision(Schema):
    """Why a section did or did not get a generated clip. One per lesson section, always recorded."""

    lesson_section_id: str
    slide_id: str | None = None
    selected: bool
    purpose: VideoPurpose | None = None
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)  # the cues that counted, in order
    skip_reason: SkipReason | None = None
    suggested: bool = False  # a candidate suggestion named this section


class VideoSegmentPlan(Schema):
    segment_id: str
    lesson_section_id: str
    slide_id: str
    purpose: VideoPurpose
    prompt: str = Field(min_length=1, max_length=2000)  # built by the prompt builder from `content`
    content: VideoPromptContent
    duration: float = Field(gt=0)
    aspect_ratio: str = "16:9"
    width: int = Field(ge=16)  # the platform resolution the clip is normalised to
    height: int = Field(ge=16)
    fps: int = Field(ge=1)
    priority: int = Field(default=0, ge=0)  # higher first when the budget cannot take every segment
    required: bool = False
    provider_preferences: ProviderPreferences = Field(default_factory=ProviderPreferences)
    insertion_strategy: InsertionStrategy = InsertionStrategy.FULL_FRAME_REPLACE
    audio: ClipAudio = "muted"
    fallback: SegmentFallback
    estimated_cost_usd: float | None = None  # None: the provider's price is not configured

    def request(self, *, seed: int | None = None) -> VideoGenerationRequest:
        """The provider request: the prompt and technical parameters, nothing about the learner or the task."""
        return VideoGenerationRequest(
            prompt=self.prompt, duration=self.duration, aspect_ratio=self.aspect_ratio, width=self.width,
            height=self.height, fps=self.fps, seed=self.provider_preferences.seed if seed is None else seed,
            with_audio=self.audio == "mixed", metadata={"segment_id": self.segment_id})


class VideoBudgetSummary(Schema):
    segments: int = 0
    seconds: float = 0.0
    estimated_cost_usd: float | None = 0.0  # None: some segment has no known price
    cost_known: bool = True
    max_segments: int
    max_seconds: float
    max_cost_usd: float | None = None


class VideoSegmentPlanSet(Schema):
    """The video strategy's output for one lesson: a decision per section and the planned segments."""

    plan_id: str  # deterministic: lesson, deck, assets, configuration and provider limits
    lesson_title: str
    language: str
    deck_id: str
    provider: str
    strategy: str  # name/version of the strategy
    decisions: list[VideoSegmentDecision] = Field(default_factory=list)
    segments: list[VideoSegmentPlan] = Field(default_factory=list)
    budget: VideoBudgetSummary
    warnings: list[str] = Field(default_factory=list)
    over_budget_required: list[str] = Field(default_factory=list)  # required sections the budget could not take

    def segment(self, segment_id: str) -> VideoSegmentPlan:
        return next(s for s in self.segments if s.segment_id == segment_id)


# --- Configuration ------------------------------------------------------------------------------------------------


class GeneratedVideoConfig(Schema):
    """Every generative-video default lives here. Budgets come from MAX_GENERATED_VIDEO_SEGMENTS,
    MAX_GENERATED_VIDEO_SECONDS and MAX_VIDEO_GENERATION_COST_USD; the rest from TA_GENERATED_VIDEO_* settings."""

    min_segment_seconds: float = Field(default=3.0, gt=0, le=60)
    max_segment_seconds: float = Field(default=10.0, gt=0, le=60)
    max_total_seconds: float = Field(default=20.0, ge=0, le=600)  # per lesson
    max_segments: int = Field(default=2, ge=0, le=20)  # per lesson
    max_cost_usd: float | None = Field(default=None, ge=0)  # per lesson; None: no cost limit
    price_per_second_usd: float | None = Field(default=None, ge=0)  # the provider's price; None: unknown
    required: bool = False  # planned segments are required (a failure follows the failure policy)
    failure_policy: Literal["fail", "continue"] = "fail"  # what a failed or over-budget required segment does
    strategy: InsertionStrategy = InsertionStrategy.FULL_FRAME_REPLACE
    poll_interval_seconds: float = Field(default=5.0, ge=0, le=600)
    poll_timeout_seconds: float = Field(default=600.0, gt=0, le=86400)  # per run of the stage; then the task waits
    poll_max_attempts: int = Field(default=120, ge=1, le=10000)
    min_width: int = Field(default=128, ge=16)  # smallest provider output accepted before normalisation
    min_height: int = Field(default=72, ge=16)
    duration_tolerance: float = Field(default=0.5, gt=0, le=10)  # seconds between asked and measured clip length

    @model_validator(mode="after")
    def _limits(self) -> GeneratedVideoConfig:
        if self.min_segment_seconds > self.max_segment_seconds:
            raise ValueError("the minimum segment duration is longer than the maximum")
        if self.strategy not in IMPLEMENTED_STRATEGIES:
            raise ValueError(f"insertion strategy '{self.strategy.value}' is not implemented yet")
        return self


class ProviderVideoLimits(Schema):
    """What the configured provider can generate, as the strategy needs to know it."""

    provider: str
    model: str | None = None
    durations: list[float] | None = None  # None: any duration up to max_duration
    max_duration: float = Field(default=10.0, gt=0)
    aspect_ratios: list[str] = Field(default_factory=lambda: ["16:9"])
    supports_audio: bool = False
    supports_seed: bool = False
    supports_cancel: bool = False
    requires_network: bool = False


# --- Validation and normalisation ---------------------------------------------------------------------------------

ClipValidationCode = Literal[
    "empty_file", "checksum_mismatch", "unreadable", "corrupted", "missing_video_stream", "unsupported_codec",
    "unsupported_container", "zero_duration", "duration_mismatch", "too_small", "aspect_mismatch", "fps_out_of_range",
    "resolution_mismatch", "fps_mismatch", "missing_audio", "unexpected_audio",
]


class ClipValidationError(Schema):
    code: ClipValidationCode
    field: str
    message: str
    expected: str | None = None
    actual: str | None = None


class ClipExpectation(Schema):
    """What a clip must be. Raw provider output is checked loosely (decodable, long enough, the right shape);
    a normalised clip exactly (the platform's codec, container, resolution and frame rate)."""

    stage: Literal["raw", "normalized"]
    duration: float = Field(gt=0)
    duration_tolerance: float = Field(gt=0)
    aspect_ratio: str = "16:9"
    min_width: int = 16
    min_height: int = 16
    width: int | None = None  # exact, for a normalised clip
    height: int | None = None
    fps: float | None = None
    codecs: list[str] = Field(default_factory=list)  # empty: any decodable codec
    containers: list[str] = Field(default_factory=list)  # empty: any container the parser recognises
    audio: Literal["required", "absent", "any"] = "any"


class ClipValidationReport(Schema):
    valid: bool
    stage: Literal["raw", "normalized"]
    checksum: str
    errors: list[ClipValidationError] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    container: str | None = None
    codec: str | None = None
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool = False
    validator: str


# --- Stage results -------------------------------------------------------------------------------------------------


class VideoSegmentAsset(Schema):
    """A GENERATED_VIDEO_ASSET: a normalised, validated clip in the object store."""

    asset_id: str
    segment_id: str
    artifact_id: str
    uri: str
    object_key: str
    checksum: str
    media_type: str
    duration: float
    width: int
    height: int
    fps: float
    format: str
    has_audio: bool
    provider: str
    model: str | None = None
    generation_job_id: str
    provider_job_id: str
    generation_key: str
    source_checksum: str  # the provider's original file, before normalisation
    reused: bool = False  # an earlier generation with the same key was reused; nothing was generated
    metadata: dict = Field(default_factory=dict)


class SegmentFailure(Schema):
    segment_id: str
    lesson_section_id: str
    required: bool
    stage: Literal["budget", "submit", "poll", "download", "validate", "normalize", "artifact", "cancelled"]
    error: str
    fallback: SegmentFallback


class GeneratedVideoResult(Schema):
    """The outcome of the generation stage after its failure policy. Every planned segment ends as an asset or as a
    failure with its fallback; nothing fake is ever stored in place of a clip."""

    status: Literal["complete", "partial", "failed", "cancelled", "skipped"]
    plan_id: str
    assets: list[VideoSegmentAsset] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    failures: list[SegmentFailure] = Field(default_factory=list)
    jobs: list[VideoGenerationJob] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def asset_for(self, segment_id: str) -> VideoSegmentAsset | None:
        return next((a for a in self.assets if a.segment_id == segment_id), None)


# --- Tool inputs ----------------------------------------------------------------------------------------------------


class GenerationSubmitRequest(Schema):
    segment: VideoSegmentPlan
    generation_key: str | None = None  # None: computed from the request, the provider and its model


class GenerationJobRequest(Schema):
    job: VideoGenerationJob


class GenerationLookupRequest(Schema):
    generation_key: str


class GenerationLookup(Schema):
    """What the generation ledger knows about a key: a job (pending or done) and, once done, the stored raw file."""

    generation_key: str
    job: VideoGenerationJob | None = None
    raw: StoredObject | None = None


class ClipAssetRequest(Schema):
    """Download (or reuse), validate, normalise and store one completed job's clip as a GENERATED_VIDEO_ASSET."""

    segment: VideoSegmentPlan
    job: VideoGenerationJob
    plan_id: str
    parent_ids: list[str] = Field(default_factory=list)


class ClipAssetResult(Schema):
    """A GENERATED_VIDEO_ASSET and the two validations behind it (the provider's file, then the normalised clip)."""

    asset: VideoSegmentAsset
    artifact: Artifact
    raw_validation: ClipValidationReport
    validation: ClipValidationReport


class GeneratedVideoStageRequest(Schema):
    """Input of the generation stage: the stored segment plan, its artifact and the configuration to apply."""

    plan: VideoSegmentPlanSet
    plan_artifact_id: str
    config: GeneratedVideoConfig = Field(default_factory=GeneratedVideoConfig)
