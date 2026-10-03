"""Video schemas. Renderer- and composer-independent.

- VideoConfig: the output format (resolution, frame rate, codecs, container, subtitle style). Every default lives
  here and nowhere else.
- VideoPlan: WHAT the video shows and when. One VideoSegment per slide, timed from the PresentationTimeline, with
  its visual (an existing IMAGE_ASSET or a slide card the compositor draws from the slide's text), its narration
  (AUDIO_ASSET references placed at their timeline positions), its subtitles and the transition into it. Assets are
  referenced by artifact id, object URI and checksum, never by local file name, and nothing in the plan is
  specific to a command-line tool.
- ComposedVideo / VideoProbe / VideoValidationReport: what a composer produced and what a parser measured in it.
- GeneratedClip: an optional GENERATED_VIDEO_ASSET shown during part of a segment (full frame or inset). Narration,
  subtitles and timing stay exactly as the PresentationTimeline says; without clips a plan is what it always was.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from app.schemas.artifact import Artifact, StoredObject
from app.schemas.audio import PresentationTimeline
from app.schemas.common import Schema
from app.schemas.generative_video import ClipAudio, InsertionStrategy
from app.schemas.presentation import Presentation

TIME_EPSILON = 0.0015  # seconds: timeline values are whole milliseconds; this absorbs float rounding

Color = Annotated[str, StringConstraints(pattern=r"^[0-9A-Fa-f]{6}$")]


class VideoCodec(str, Enum):
    H264 = "h264"


class AudioCodec(str, Enum):
    AAC = "aac"


class VideoContainer(str, Enum):
    MP4 = "mp4"


CONTAINER_MEDIA_TYPES = {VideoContainer.MP4: "video/mp4"}
IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg"})  # what the compositor can draw from an IMAGE_ASSET


class TransitionType(str, Enum):
    CUT = "cut"
    FADE = "fade"  # fade through the background colour: out at the end of a slide, in at the start of the next


class SubtitleStyle(Schema):
    """How burned-in subtitles look. Sizes are fractions of the frame height so they scale with the resolution."""

    enabled: bool = True
    font_size: float = Field(default=0.045, gt=0, le=0.2)
    max_chars_per_line: int = Field(default=42, ge=10, le=120)
    max_lines: int = Field(default=2, ge=1, le=4)
    bottom_margin: float = Field(default=0.06, ge=0, le=0.4)  # safe area under the subtitle box
    text_color: Color = "FFFFFF"
    box_color: Color = "000000"
    box_opacity: float = Field(default=0.72, ge=0, le=1)


class VideoConfig(Schema):
    width: int = Field(default=1920, ge=160, le=7680)
    height: int = Field(default=1080, ge=90, le=4320)
    fps: int = Field(default=30, ge=1, le=120)
    codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    container: VideoContainer = VideoContainer.MP4
    bitrate_kbps: int | None = Field(default=None, ge=100, le=200_000)  # None: constant quality
    audio_bitrate_kbps: int = Field(default=128, ge=32, le=512)
    audio_sample_rate: int = Field(default=48000, ge=8000, le=96000)
    audio_channels: int = Field(default=2, ge=1, le=2)
    pixel_format: Literal["yuv420p"] = "yuv420p"
    background: Color = "F4F6F8"
    text_color: Color = "1F2933"
    accent_color: Color = "2563EB"
    transition: TransitionType = TransitionType.CUT
    fade_seconds: float = Field(default=0.5, gt=0, le=5)
    allow_audio_overlap: bool = False
    duration_tolerance: float = Field(default=0.1, gt=0, le=5)  # seconds between the MP4 and the timeline
    subtitles: SubtitleStyle = Field(default_factory=SubtitleStyle)

    @model_validator(mode="after")
    def _even(self) -> VideoConfig:
        if self.pixel_format == "yuv420p" and (self.width % 2 or self.height % 2):
            raise ValueError("yuv420p needs an even width and height")
        return self

    @property
    def media_type(self) -> str:
        return CONTAINER_MEDIA_TYPES[self.container]

    @property
    def resolution(self) -> Resolution:
        return Resolution(width=self.width, height=self.height)


class Resolution(Schema):
    width: int = Field(ge=1)
    height: int = Field(ge=1)


# --- Inputs: the assets a plan may reference ----------------------------------------------------


class ImageAssetInput(Schema):
    """An IMAGE_ASSET artifact as the video planner sees it: a reference, never the bytes."""

    artifact_id: str
    asset_id: str
    uri: str
    checksum: str
    media_type: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    alt_text: str = ""


class AudioAssetInput(Schema):
    """An AUDIO_ASSET artifact: its object reference, measured duration and the narration text it speaks."""

    artifact_id: str
    segment_id: str
    slide_id: str
    uri: str
    checksum: str
    media_type: str
    duration: float = Field(gt=0)
    text: str
    language: str


class GeneratedClipInput(Schema):
    """A GENERATED_VIDEO_ASSET as the video planner sees it: a normalised clip planned for one slide."""

    segment_id: str  # the VideoSegmentPlan it fulfils
    slide_id: str
    artifact_id: str
    uri: str
    checksum: str
    media_type: str
    duration: float = Field(gt=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(gt=0)
    has_audio: bool = False
    strategy: InsertionStrategy = InsertionStrategy.FULL_FRAME_REPLACE
    audio: ClipAudio = "muted"


# --- The plan -----------------------------------------------------------------------------------


class SlideCard(Schema):
    """Text the compositor draws on a slide: the deck title, the slide title and a few lines of content."""

    deck_title: str
    title: str
    lines: list[str] = Field(default_factory=list)
    footer: str = ""


class VisualRef(Schema):
    """What is on screen during a segment. `image` is an existing IMAGE_ASSET placed on the card; without one the
    compositor draws the card alone (a deterministic slide background, never a generated image)."""

    kind: Literal["image_asset", "slide_card"]
    card: SlideCard
    image: ImageAssetInput | None = None

    @model_validator(mode="after")
    def _image(self) -> VisualRef:
        if (self.kind == "image_asset") != (self.image is not None):
            raise ValueError("an image_asset visual has an image and a slide_card visual has none")
        return self


class GeneratedClip(Schema):
    """A generated clip shown from `start_time` to `end_time` (inside its segment, on the video timeline).
    full_frame_replace: the clip fills the frame instead of the slide; inset: it plays in the slide's media box.
    Subtitles are drawn over it as on any slide. Its own sound is muted unless `audio` is "mixed"."""

    segment_id: str  # the VideoSegmentPlan id
    artifact_id: str  # the GENERATED_VIDEO_ASSET
    uri: str
    checksum: str
    media_type: str
    asset_duration: float = Field(gt=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps: float = Field(gt=0)
    has_audio: bool = False
    strategy: InsertionStrategy = InsertionStrategy.FULL_FRAME_REPLACE
    audio: ClipAudio = "muted"
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 3)


class AudioTrack(Schema):
    """One AUDIO_ASSET placed on the video timeline at the position the PresentationTimeline gives it."""

    track_id: str  # the narration segment id
    slide_id: str
    artifact_id: str
    uri: str
    checksum: str
    media_type: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    duration: float = Field(gt=0)
    asset_duration: float = Field(gt=0)  # measured length of the audio file; validated against `duration`


class Transition(Schema):
    type: TransitionType = TransitionType.CUT
    duration: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _cut(self) -> Transition:
        if self.type == TransitionType.CUT and self.duration != 0:
            raise ValueError("a cut has no duration")
        if self.type == TransitionType.FADE and self.duration <= 0:
            raise ValueError("a fade needs a duration")
        return self


class TransitionPoint(Schema):
    """The boundary between two segments and how it is crossed."""

    from_segment: str
    to_segment: str
    at: float = Field(ge=0)
    transition: Transition


class Subtitle(Schema):
    subtitle_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str = Field(min_length=1)
    language: str
    segment_ref: str  # the narration segment (AUDIO_ASSET) the text comes from

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")


class SubtitleTrack(Schema):
    track_id: str
    language: str
    source: Literal["narration_text"] = "narration_text"  # known text, never speech-to-text
    burned_in: bool = True
    subtitles: list[Subtitle] = Field(default_factory=list)

    def to_webvtt(self) -> str:
        lines = ["WEBVTT", ""]
        for s in self.subtitles:
            lines += [s.subtitle_id, f"{_vtt_time(s.start_time)} --> {_vtt_time(s.end_time)}", s.text, ""]
        return "\n".join(lines)


def _vtt_time(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class VideoSegment(Schema):
    segment_id: str
    slide_id: str
    order: int = Field(ge=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    duration: float
    visual_ref: VisualRef
    audio_refs: list[str] = Field(default_factory=list)  # AudioTrack ids, in playback order; empty: silence
    subtitle_refs: list[str] = Field(default_factory=list)  # Subtitle ids shown during this segment
    transition: Transition = Field(default_factory=Transition)  # into this segment
    generated: GeneratedClip | None = None  # an optional generated clip shown during part of the segment


class VideoPlan(Schema):
    video_plan_id: str  # derived from every input checksum and the configuration
    task_id: str
    presentation_ref: str  # PRESENTATION artifact id
    timeline_ref: str  # PRESENTATION_TIMELINE artifact id
    timeline_id: str
    timeline_checksum: str
    deck_id: str
    title: str
    language: str
    resolution: Resolution
    fps: int = Field(ge=1)
    duration: float = Field(ge=0)
    config: VideoConfig
    slides: list[VideoSegment] = Field(default_factory=list)
    audio_tracks: list[AudioTrack] = Field(default_factory=list)
    subtitle_track: SubtitleTrack | None = None
    transitions: list[TransitionPoint] = Field(default_factory=list)
    planner: str
    metadata: dict = Field(default_factory=dict)

    def track(self, track_id: str) -> AudioTrack:
        return next(t for t in self.audio_tracks if t.track_id == track_id)

    def subtitles_for(self, segment: VideoSegment) -> list[Subtitle]:
        if self.subtitle_track is None:
            return []
        wanted = set(segment.subtitle_refs)
        return [s for s in self.subtitle_track.subtitles if s.subtitle_id in wanted]

    def image_artifact_ids(self) -> list[str]:
        return list(dict.fromkeys(s.visual_ref.image.artifact_id for s in self.slides if s.visual_ref.image))

    def audio_artifact_ids(self) -> list[str]:
        return list(dict.fromkeys(t.artifact_id for t in self.audio_tracks))

    def generated_clips(self) -> list[GeneratedClip]:
        return [s.generated for s in self.slides if s.generated is not None]


class VideoPlanningRequest(Schema):
    """What the VideoAgent plans from: the presentation, its timeline and the asset references."""

    task_id: str
    presentation: Presentation  # the built, renderer-independent presentation (not the PPTX file)
    presentation_artifact_id: str
    timeline: PresentationTimeline
    timeline_artifact_id: str
    timeline_checksum: str
    image_assets: list[ImageAssetInput] = Field(default_factory=list)
    audio_assets: list[AudioAssetInput] = Field(default_factory=list)
    generated_clips: list[GeneratedClipInput] = Field(default_factory=list)  # optional GENERATED_VIDEO_ASSETs
    config: VideoConfig = Field(default_factory=VideoConfig)

    def validation_request(self, plan: VideoPlan, *, enforce: bool = False) -> VideoPlanValidationRequest:
        return VideoPlanValidationRequest(
            plan=plan.model_dump(mode="json"), timeline=self.timeline,
            presentation_artifact_id=self.presentation_artifact_id, timeline_artifact_id=self.timeline_artifact_id,
            slide_ids=[s.slide_id for s in self.presentation.slides],
            image_assets=self.image_assets, audio_assets=self.audio_assets, generated_clips=self.generated_clips,
            enforce=enforce)


# --- Plan validation ----------------------------------------------------------------------------

VideoPlanIssueCode = Literal[
    "invalid_schema", "no_segments", "duplicate_segment_id", "unknown_slide", "segment_order", "negative_duration",
    "timeline_mismatch", "duration_mismatch", "unknown_image", "unsupported_image", "unknown_audio",
    "audio_outside_segment", "audio_overlap", "audio_duration_mismatch", "unknown_reference", "subtitle_timing",
    "subtitle_overlap", "invalid_transition", "unsupported_config", "reference_mismatch",
    "unknown_generated_clip", "unsupported_strategy", "clip_timing", "clip_format",
]


class VideoPlanIssue(Schema):
    code: VideoPlanIssueCode
    message: str
    segment_id: str | None = None
    field: str | None = None

    def describe(self) -> str:
        where = f"segment {self.segment_id}" if self.segment_id else "plan"
        return f"[{self.code}] {where}{f' ({self.field})' if self.field else ''}: {self.message}"


class VideoPlanValidationRequest(Schema):
    plan: dict  # raw JSON, so a malformed plan is reported rather than crashed on
    timeline: PresentationTimeline
    presentation_artifact_id: str
    timeline_artifact_id: str
    slide_ids: list[str] = Field(default_factory=list)  # the presentation's slides, in deck order
    image_assets: list[ImageAssetInput] = Field(default_factory=list)
    audio_assets: list[AudioAssetInput] = Field(default_factory=list)
    generated_clips: list[GeneratedClipInput] = Field(default_factory=list)
    enforce: bool = False


class VideoPlanValidationReport(Schema):
    valid: bool
    video_plan_id: str | None = None
    errors: list[VideoPlanIssue] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    validator: str
    plan: VideoPlan | None = None

    @model_validator(mode="after")
    def _consistent(self) -> VideoPlanValidationReport:
        if self.valid == bool(self.errors):
            raise ValueError("a report is valid exactly when it has no errors")
        if self.valid and self.plan is None:
            raise ValueError("a valid report carries the plan")
        return self


# --- Composition --------------------------------------------------------------------------------


class ComposedVideo(Schema):
    """A file a composer wrote into the workspace it was given."""

    path: str
    media_type: str
    composer: str  # name/version
    frames: int = Field(ge=0)
    duration: float = Field(ge=0)  # what the composer encoded: frames / fps
    render_seconds: float = Field(ge=0)
    cpu_seconds: float | None = None
    subtitles_burned: int = Field(default=0, ge=0)
    metadata: dict = Field(default_factory=dict)


class VideoComposeRequest(Schema):
    plan: VideoPlan
    name: str = "video"  # the VIDEO artifact name, for reuse lookup


class VideoComposeResult(Schema):
    composition_key: str  # identifies equivalent outputs: plan identity, configuration and composer
    object: StoredObject
    composer: str
    frames: int = 0
    duration: float = 0.0
    render_seconds: float = 0.0
    cpu_seconds: float | None = None
    subtitles_burned: int = 0
    reused: bool = False  # an existing VIDEO artifact with the same composition key was found; nothing was composed
    artifact: Artifact | None = None  # the reused VIDEO artifact


# --- Output validation --------------------------------------------------------------------------


class VideoStreamProbe(Schema):
    codec: str
    width: int
    height: int
    fps: float
    pixel_format: str | None = None
    frames: int | None = None
    duration: float | None = None


class AudioStreamProbe(Schema):
    codec: str
    sample_rate: int
    channels: int
    duration: float | None = None


class ClipNormalization(Schema):
    """The platform format a generated clip is converted to before composition."""

    width: int = Field(ge=2)
    height: int = Field(ge=2)
    fps: int = Field(ge=1)
    duration: float = Field(gt=0)  # the clip is cut to at most this long
    codec: VideoCodec = VideoCodec.H264
    pixel_format: Literal["yuv420p"] = "yuv420p"
    container: VideoContainer = VideoContainer.MP4
    keep_audio: bool = False  # muted clips carry no audio stream at all
    audio_codec: AudioCodec = AudioCodec.AAC
    audio_sample_rate: int = 48000
    audio_channels: int = 2

    @property
    def media_type(self) -> str:
        return CONTAINER_MEDIA_TYPES[self.container]


class NormalizedClip(Schema):
    path: str
    media_type: str
    normalizer: str  # name/version
    render_seconds: float = Field(ge=0)
    metadata: dict = Field(default_factory=dict)


class VideoProbe(Schema):
    """What a container parser measured in a file."""

    container: str  # format names the parser reported, e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    brand: str | None = None
    duration: float
    size_bytes: int
    video: VideoStreamProbe | None = None
    audio: list[AudioStreamProbe] = Field(default_factory=list)
    prober: str


class AudioWindow(Schema):
    """A stretch of the timeline and whether narration is expected there."""

    label: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    expect_sound: bool


class FrameSample(Schema):
    label: str
    time: float = Field(ge=0)
    expect_change: bool = True  # the picture should differ from the previous sample's (a new slide)


VideoValidationCode = Literal[
    "empty_file", "checksum_mismatch", "not_mp4", "unreadable", "zero_duration", "duration_mismatch",
    "missing_video_stream", "resolution_mismatch", "fps_mismatch", "codec_mismatch", "missing_audio_stream",
    "audio_codec_mismatch", "missing_narration", "unexpected_sound", "empty_frame", "no_transition",
]


class VideoValidationError(Schema):
    code: VideoValidationCode
    field: str
    message: str
    expected: str | None = None
    actual: str | None = None


class VideoValidationRequest(Schema):
    object: StoredObject
    config: VideoConfig
    expected_duration: float = Field(ge=0)
    audio_windows: list[AudioWindow] = Field(default_factory=list)
    frame_samples: list[FrameSample] = Field(default_factory=list)  # one per segment, in order


class VideoValidationReport(Schema):
    valid: bool
    errors: list[VideoValidationError] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    probe: VideoProbe | None = None
    audio_levels: dict[str, float] = Field(default_factory=dict)  # window label -> peak level in dBFS
    validator: str


class VideoValidationInput(Schema):
    plan: VideoPlan
    composed: VideoComposeResult


# --- VIDEO artifact -----------------------------------------------------------------------------


class VideoArtifactRequest(Schema):
    plan: VideoPlan
    composed: VideoComposeResult
    validation: VideoValidationReport
    parent_ids: list[str] = Field(default_factory=list)
    name: str = "video"


class VideoArtifactMetadata(Schema):
    video_plan_id: str
    composition_key: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: str | None
    container: str
    media_type: str
    file_size: int
    checksum: str
    object_key: str
    timeline_ref: str
    presentation_ref: str
    timeline_duration: float
    segments: int
    audio_tracks: int
    audio_stream: bool
    subtitles: int
    subtitles_burned: bool
    transition: str
    composer: str
    render_seconds: float
    validation: VideoValidationReport
    generated_segments: int = 0  # generated clips composed into the video
    generated_artifact_ids: list[str] = Field(default_factory=list)  # their GENERATED_VIDEO_ASSET artifacts


class VideoStageRequest(Schema):
    """Input of the composition stage: the validated plan and the artifacts the VIDEO artifact will link."""

    plan: VideoPlan
    parent_ids: list[str] = Field(default_factory=list)  # video plan, timeline, presentation
    name: str = "video"


VideoFailureStage = Literal["compose", "validate", "artifact"]


class VideoResult(Schema):
    """The outcome of the composition stage after the video failure policy."""

    status: Literal["complete", "failed"]
    video_plan_id: str
    artifact: Artifact | None = None  # the VIDEO artifact; None when composition failed
    composed: VideoComposeResult | None = None
    validation: VideoValidationReport | None = None
    failed_stage: VideoFailureStage | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
