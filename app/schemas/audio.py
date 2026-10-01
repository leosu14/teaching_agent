"""Audio schemas. Provider- and format-independent.

Four representations, kept apart on purpose:

- AudioPlan: WHAT is spoken over WHICH slide (segments mapped to slide ids, in order). No provider details; the
  voice is a provider-independent voice id and the language a BCP 47 tag.
- TTSRequest / TTSResult: one synthesis call through the TTS tool, and the stored audio object it produced.
- AUDIO_ASSET metadata: a validated audio object linked to its segment and slide.
- PresentationTimeline: when each slide and segment plays, resolved from the measured audio durations. It
  references the PRESENTATION artifact and the slide ids, never the PPTX file itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from app.schemas.artifact import Artifact, StoredObject
from app.schemas.common import Schema
from app.schemas.lesson import LessonContent, LessonRequest
from app.schemas.presentation import SlideDeckPlan
from app.schemas.research import ResearchBundle

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LanguageTag = Annotated[str, StringConstraints(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")]
SegmentId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,96}$")]
SlideRef = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]  # a slide id, as SlidePlan has it

WORDS_PER_SECOND = 2.5  # ~150 words per minute: only for the planning estimate, never for the timeline

# Known audio formats. A provider declares which ones it can produce; the validator can only measure the ones
# app/utils/audio.py has a reader for, and rejects the rest instead of trusting them.
AUDIO_MEDIA_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


# --- Language and voice -----------------------------------------------------------------------


def normalize_language(tag: str) -> str:
    """Canonical case for a BCP 47 tag: language lower, script title, region upper (zh-hans-cn -> zh-Hans-CN)."""
    parts = tag.replace("_", "-").split("-")
    out = [parts[0].lower()]
    for p in parts[1:]:
        out.append(p.title() if len(p) == 4 else p.upper() if len(p) in (2, 3) else p.lower())
    return "-".join(out)


def language_matches(requested: str, offered: str) -> bool:
    """`en` is served by any English voice; `en-US` only by an en-US voice."""
    r, o = normalize_language(requested).split("-"), normalize_language(offered).split("-")
    return r == o or (len(r) == 1 and r[0] == o[0])


class Voice(Schema):
    """A voice as the provider exposes it. Optional fields stay None unless the provider reports them."""

    voice_id: Text
    language: LanguageTag
    display_name: Text
    provider: Text
    gender: Literal["female", "male", "neutral"] | None = None
    metadata: dict = Field(default_factory=dict)  # anything else the provider reported, verbatim

    def supports(self, language: str) -> bool:
        return language_matches(language, self.language)


def voices_for(language: str, voices: list[Voice]) -> list[Voice]:
    """Voices that can speak `language`: exact tag matches first, otherwise in the provider's catalog order (a
    provider lists its default voice first). Deterministic."""
    exact = normalize_language(language)
    return sorted((v for v in voices if v.supports(language)), key=lambda v: normalize_language(v.language) != exact)


class VoiceQuery(Schema):
    language: LanguageTag | None = None


class VoiceCatalog(Schema):
    provider: str
    voices: list[Voice] = Field(default_factory=list)


# --- The audio plan ---------------------------------------------------------------------------


class AudioSourceType(str, Enum):
    """What a segment narrates. Images are never a source: only text the lesson and slides already contain."""

    SLIDE_TITLE = "slide_title"
    SLIDE_CONTENT = "slide_content"
    SPEAKER_NOTES = "speaker_notes"
    EXERCISE_INSTRUCTIONS = "exercise_instructions"
    ANSWER_EXPLANATION = "answer_explanation"


def estimate_duration(text: str, speaking_rate: float = 1.0) -> float:
    return round(max(0.5, len(text.split()) / (WORDS_PER_SECOND * speaking_rate)), 1)


class AudioSegment(Schema):
    segment_id: SegmentId
    order: int = Field(ge=1)
    slide_id: Text
    source_type: AudioSourceType
    source_ref: Text  # e.g. "s03.notes" or a question id: where on the slide the text comes from
    text: str  # what is spoken; the validator rejects empty text
    language: LanguageTag
    voice: Text  # a voice id from the provider's catalog
    speaking_rate: float = Field(default=1.0, gt=0, le=4)  # 1.0 is the voice's normal rate
    pitch: float | None = Field(default=None, ge=-24, le=24)  # semitones, where the provider supports it
    pause_before: float = Field(default=0.0, ge=0, le=10)  # seconds of silence before the segment
    pause_after: float = Field(default=0.0, ge=0, le=10)  # seconds of silence after it
    expected_duration: float = Field(gt=0)  # planning estimate in seconds; the timeline uses the measured one
    required: bool = True  # an optional segment may fail without failing the task (with a warning)
    # Filled by the timing resolver from the generated audio; None in a plan that has not been voiced yet.
    start_time: float | None = None
    end_time: float | None = None
    duration: float | None = None

    def tts_request(self, output_format: str = "wav", sample_rate: int | None = None) -> TTSRequest:
        return TTSRequest(text=self.text, language=self.language, voice=self.voice, speaking_rate=self.speaking_rate,
                          pitch=self.pitch, output_format=output_format, sample_rate=sample_rate)


class AudioPlan(Schema):
    audio_plan_id: Text
    task_id: Text
    deck_id: Text
    language: LanguageTag
    voice: Text  # the plan's default voice
    segments: list[AudioSegment] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def for_slide(self, slide_id: str) -> list[AudioSegment]:
        return [s for s in self.segments if s.slide_id == slide_id]

    def expected_duration(self) -> float:
        return round(sum(s.pause_before + s.expected_duration + s.pause_after for s in self.segments), 3)


class NarrationProposal(Schema):
    """One segment as the model proposes it. The system assigns ids, order, voice, language and rate."""

    slide_id: SlideRef
    source_type: AudioSourceType
    source_ref: Text
    text: Text
    required: bool = True
    pause_before: float = Field(default=0.0, ge=0, le=10)
    pause_after: float = Field(default=0.5, ge=0, le=10)


class AudioPlanProposal(Schema):
    segments: list[NarrationProposal] = Field(min_length=1)
    rationale: str = ""


# --- Planning input ----------------------------------------------------------------------------


class SpokenQuestion(Schema):
    question_id: str
    prompt: str
    choices: list[str] = Field(default_factory=list)


class SpokenAnswer(Schema):
    question_id: str
    answer: str
    explanation: str = ""


class SpeakableSlide(Schema):
    """The text of a slide that may be spoken. Images, captions, tables, citations and footers are left out:
    they are visual or reference material, not narration."""

    slide_id: str
    order: int
    slide_type: str
    title: str
    content: list[str] = Field(default_factory=list)  # text, bullet and vocabulary lines
    speaker_notes: str = ""
    questions: list[SpokenQuestion] = Field(default_factory=list)
    answers: list[SpokenAnswer] = Field(default_factory=list)


class AudioPlanningInput(Schema):
    """What the model sees when planning narration."""

    language: str
    lesson_title: str
    level: str
    slides: list[SpeakableSlide]
    max_words_per_segment: int = Field(ge=5)
    corrections: list[str] = Field(default_factory=list)  # validation errors of the previous proposal


def speakable(deck: SlideDeckPlan) -> list[SpeakableSlide]:
    slides = []
    for s in deck.slides:
        content: list[str] = []
        questions, answers = [], []
        for b in s.content_blocks:
            match b.kind:
                case "text":
                    content.append(b.text)
                case "bullets":
                    content.extend(b.items)
                case "vocabulary":
                    content.extend(f"{e.term}: {e.meaning}" for e in b.entries)
                case "question":
                    questions.append(SpokenQuestion(question_id=b.question_id, prompt=b.prompt, choices=b.choices))
                case "answer":
                    answers.append(SpokenAnswer(question_id=b.question_id, answer=b.answer, explanation=b.explanation))
        slides.append(SpeakableSlide(slide_id=s.slide_id, order=s.order, slide_type=s.slide_type.value, title=s.title,
                                     content=content, speaker_notes=s.speaker_notes, questions=questions,
                                     answers=answers))
    return slides


class AudioPlanningRequest(Schema):
    """Input of the AudioPlannerAgent: an approved lesson, its rendered presentation's deck, and audio settings."""

    task_id: str
    request: LessonRequest
    lesson: LessonContent
    deck: SlideDeckPlan
    research: ResearchBundle | None = None
    presentation_artifact_id: str
    language: LanguageTag
    voice_id: str | None = None  # a configured voice; otherwise the provider's first voice for the language
    speaking_rate: float = Field(default=1.0, gt=0, le=4)
    pitch: float | None = Field(default=None, ge=-24, le=24)
    max_words_per_segment: int = Field(default=80, ge=5, le=400)
    speak_citations: bool = False  # citation metadata is only read aloud when explicitly required

    def planning_input(self, corrections: list[str] | None = None) -> AudioPlanningInput:
        return AudioPlanningInput(language=self.language, lesson_title=self.lesson.title, level=self.lesson.level,
                                  slides=speakable(self.deck), max_words_per_segment=self.max_words_per_segment,
                                  corrections=corrections or [])

    def unspoken_terms(self) -> list[str]:
        """Citation metadata the narration must not read aloud: citation ids, source titles and URLs."""
        if self.speak_citations or self.research is None:
            return []
        terms = [c.citation_id for c in self.research.citations]
        for source in self.research.sources:
            terms += [source.title, source.url]
        return sorted({t for t in terms if t and len(t) >= 4})

    def validation_request(self, plan: AudioPlan, *, enforce: bool = False) -> AudioPlanValidationRequest:
        return AudioPlanValidationRequest(plan=plan.model_dump(mode="json"),
                                          slide_ids=[s.slide_id for s in self.deck.slides],
                                          unspoken_terms=self.unspoken_terms(),
                                          max_words_per_segment=self.max_words_per_segment, enforce=enforce)


# --- Deterministic plan validation -------------------------------------------------------------

AudioPlanIssueCode = Literal[
    "invalid_schema", "no_segments", "duplicate_segment_id", "unknown_slide", "segment_order", "slide_order",
    "empty_text", "too_long", "unsupported_language", "unknown_voice", "voice_language_mismatch", "spoken_metadata",
    "incomplete_timing", "negative_duration", "timing_mismatch", "overlapping_timing",
]


class AudioPlanIssue(Schema):
    code: AudioPlanIssueCode
    message: str
    segment_id: str | None = None
    field: str | None = None

    def describe(self) -> str:
        where = f"segment {self.segment_id}" if self.segment_id else "plan"
        return f"[{self.code}] {where}{f' ({self.field})' if self.field else ''}: {self.message}"


class AudioPlanValidationRequest(Schema):
    """A plan (as raw JSON, so malformed segments are reported rather than crashed on) and what it may reference.
    Voices and languages are checked against the TTS provider's live catalog unless `check_voices` is off."""

    plan: dict
    slide_ids: list[str] = Field(default_factory=list)  # in deck order
    unspoken_terms: list[str] = Field(default_factory=list)
    max_words_per_segment: int | None = None
    check_voices: bool = True
    enforce: bool = False


class AudioPlanValidationReport(Schema):
    valid: bool
    audio_plan_id: str | None = None
    errors: list[AudioPlanIssue] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    validator: str
    plan: AudioPlan | None = None

    @model_validator(mode="after")
    def _consistent(self) -> AudioPlanValidationReport:
        if self.valid == bool(self.errors):
            raise ValueError("a report is valid exactly when it has no errors")
        if self.valid and self.plan is None:
            raise ValueError("a valid report carries the plan")
        return self


# --- TTS ---------------------------------------------------------------------------------------


class TTSRequest(Schema):
    text: Text
    language: LanguageTag
    voice: Text
    speaking_rate: float = Field(default=1.0, gt=0, le=4)
    pitch: float | None = Field(default=None, ge=-24, le=24)
    output_format: str = "wav"
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)  # None: the provider's default

    def fingerprint(self) -> str:
        """sha256 of everything that determines the audio. Equal fingerprints may reuse an existing asset."""
        body = json.dumps({**self.model_dump(mode="json"), "language": normalize_language(self.language)},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


class TTSUsage(Schema):
    """What a provider reports per request. Costs stay None unless the provider reports them."""

    requests: int = Field(default=1, ge=0)
    characters: int = Field(default=0, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    seconds: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = None
    cost_usd: float | None = None


class TTSResult(Schema):
    audio: StoredObject  # the content-addressed object in the object store
    duration: float  # seconds, as the provider declared it; the validator measures the bytes
    sample_rate: int
    channels: int
    format: str
    media_type: str
    provider: str
    model: str
    voice: str
    language: str
    input_hash: str  # TTSRequest.fingerprint()
    provider_metadata: dict = Field(default_factory=dict)
    usage: TTSUsage
    ignored_parameters: list[str] = Field(default_factory=list)  # requested but not supported by the provider


# --- Audio validation --------------------------------------------------------------------------

AudioValidationCode = Literal[
    "empty_audio", "checksum_mismatch", "unsupported_media_type", "unreadable_audio", "format_mismatch",
    "zero_duration", "duration_mismatch", "sample_rate_mismatch", "channel_mismatch",
]


class AudioValidationError(Schema):
    code: AudioValidationCode
    field: str
    message: str
    expected: str | None = None
    actual: str | None = None


class MeasuredAudio(Schema):
    """What the bytes actually contain, read with a container parser."""

    format: str
    media_type: str
    duration: float
    sample_rate: int
    channels: int
    sample_width: int  # bytes per sample
    frames: int
    checksum: str
    size_bytes: int


class AudioValidationRequest(Schema):
    object: StoredObject
    declared_format: str
    declared_duration: float
    declared_sample_rate: int
    declared_channels: int
    expected_sample_rate: int | None = None
    expected_channels: int | None = None


class AudioValidationReport(Schema):
    valid: bool
    errors: list[AudioValidationError] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)
    measured: MeasuredAudio | None = None
    validator: str


# --- AUDIO_ASSET -------------------------------------------------------------------------------


class AudioAssetRequest(Schema):
    name: str
    segment: AudioSegment
    audio_plan_id: str
    tts: TTSResult
    expected_sample_rate: int | None = None
    expected_channels: int | None = None
    parent_ids: list[str] = Field(default_factory=list)


class AudioAssetMetadata(Schema):
    """Metadata of an AUDIO_ASSET artifact. The bytes are a content-addressed object in the object store."""

    asset_id: str
    segment_id: str
    slide_id: str
    audio_plan_id: str
    source_type: AudioSourceType
    source_ref: str
    text: str
    input_hash: str
    duration: float  # measured from the bytes
    format: str
    media_type: str
    sample_rate: int
    channels: int
    language: str
    voice: str
    provider: str
    model: str | None = None
    checksum: str
    size_bytes: int
    object_key: str
    usage: TTSUsage
    validation: AudioValidationReport


class AudioAssetLookup(Schema):
    name: str
    input_hash: str


class AudioAssetLookupResult(Schema):
    artifact: Artifact | None = None
    reason: str = ""


class AudioAssetRef(Schema):
    segment_id: str
    slide_id: str
    artifact_id: str
    asset_id: str
    checksum: str
    uri: str
    media_type: str
    duration: float
    reused: bool = False  # an existing asset with the same inputs and checksum was reused


class AudioSegmentFailure(Schema):
    segment_id: str
    slide_id: str
    required: bool
    stage: Literal["tts", "validate", "store"]
    reason: str
    errors: list[AudioValidationError] = Field(default_factory=list)


class NarrationRequest(Schema):
    plan: AudioPlan
    audio_plan_artifact_id: str
    output_format: str = "wav"
    sample_rate: int | None = None


class NarrationResult(Schema):
    status: Literal["complete", "partial", "failed"]
    audio_plan_id: str
    assets: list[AudioAssetRef] = Field(default_factory=list)  # one per voiced segment, in plan order
    artifacts: list[Artifact] = Field(default_factory=list)  # the AUDIO_ASSET artifacts, each once
    failures: list[AudioSegmentFailure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def durations(self) -> dict[str, float]:
        return {a.segment_id: a.duration for a in self.assets}


# --- Timing ------------------------------------------------------------------------------------


class SegmentTiming(Schema):
    segment_id: str
    slide_id: str
    order: int
    audio_artifact_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    duration: float = Field(gt=0)


class SlideTiming(Schema):
    slide_id: str
    order: int = Field(ge=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    duration: float = Field(gt=0)
    audio_segment_refs: list[str] = Field(default_factory=list)  # segment ids, in playback order


class PresentationTimeline(Schema):
    """When each slide and narration segment plays. Independent of the PPTX file: it references the PRESENTATION
    artifact and slide ids, and every segment's AUDIO_ASSET artifact."""

    timeline_id: str
    presentation_artifact_id: str
    deck_id: str
    audio_plan_id: str
    audio_plan_artifact_id: str
    language: str
    voice: str
    duration: float = Field(ge=0)
    slides: list[SlideTiming] = Field(min_length=1)
    segments: list[SegmentTiming] = Field(default_factory=list)
    missing_segments: list[str] = Field(default_factory=list)  # planned segments without audio (failed, optional)
    resolver: str
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ordered(self) -> PresentationTimeline:
        if [s.order for s in self.slides] != list(range(1, len(self.slides) + 1)):
            raise ValueError("slides must be ordered 1..n")
        cursor = 0.0
        for s in self.slides:
            if not _same(s.start_time, cursor) or not _same(s.duration, s.end_time - s.start_time):
                raise ValueError(f"slide {s.slide_id} timing is not contiguous")
            cursor = s.end_time
        if not _same(self.duration, cursor):
            raise ValueError("timeline duration differs from the end of the last slide")
        slides = {s.slide_id: s for s in self.slides}
        for seg in self.segments:
            slide = slides.get(seg.slide_id)
            if slide is None or seg.start_time < slide.start_time or seg.end_time > slide.end_time + 1e-9:
                raise ValueError(f"segment {seg.segment_id} lies outside its slide")
        return self

    def slide(self, slide_id: str) -> SlideTiming:
        return next(s for s in self.slides if s.slide_id == slide_id)

    def audio_artifact_ids(self) -> list[str]:
        return list(dict.fromkeys(s.audio_artifact_id for s in self.segments))


def _same(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


class TimelineRequest(Schema):
    plan: AudioPlan
    narration: NarrationResult
    slide_ids: list[str] = Field(min_length=1)  # deck order
    presentation_artifact_id: str
    audio_plan_artifact_id: str
    silent_slide_seconds: float = Field(default=3.0, gt=0, le=60)  # how long a slide without narration is shown
    name: str = "presentation_timeline"


class TimelineResult(Schema):
    timeline: PresentationTimeline
    plan: AudioPlan  # the plan with every voiced segment's start_time, end_time and duration
    artifact: Artifact


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def concise(text: str, max_words: int) -> str:
    """Whole sentences up to `max_words`; a first sentence that is longer is cut at a word boundary."""
    out: list[str] = []
    for sentence in SENTENCE_RE.split(" ".join(text.split())):
        if len(" ".join([*out, sentence]).split()) > max_words:
            break
        out.append(sentence)
    if not out:
        return " ".join(text.split()[:max_words])
    return " ".join(out)
