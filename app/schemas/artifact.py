"""Artifact metadata. Binary content lives in the object store; only metadata is in the database."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.common import Schema, utcnow


class ArtifactType(str, Enum):
    LESSON = "LESSON"
    SCRIPT = "SCRIPT"
    SLIDE = "SLIDE"
    SLIDE_PLAN = "SLIDE_PLAN"
    PRESENTATION = "PRESENTATION"
    IMAGE_ASSET = "IMAGE_ASSET"
    AUDIO_PLAN = "AUDIO_PLAN"
    AUDIO_ASSET = "AUDIO_ASSET"
    PRESENTATION_TIMELINE = "PRESENTATION_TIMELINE"
    VIDEO_PLAN = "VIDEO_PLAN"
    VIDEO = "VIDEO"
    VIDEO_SEGMENT_PLAN = "VIDEO_SEGMENT_PLAN"
    GENERATED_VIDEO_ASSET = "GENERATED_VIDEO_ASSET"
    CODE = "CODE"
    EXERCISE = "EXERCISE"
    REPORT = "REPORT"
    SUBTITLE = "SUBTITLE"
    LESSON_PLAN = "LESSON_PLAN"
    LEARNER_EVALUATION = "LEARNER_EVALUATION"
    RESEARCH_BUNDLE = "RESEARCH_BUNDLE"
    VISUAL_PLAN = "VISUAL_PLAN"
    LEARNING_EVIDENCE = "LEARNING_EVIDENCE"
    LEARNER_MODEL = "LEARNER_MODEL"
    KNOWLEDGE_GAPS = "KNOWLEDGE_GAPS"
    PEDAGOGICAL_PLAN = "PEDAGOGICAL_PLAN"
    LEARNING_RECOMMENDATION = "LEARNING_RECOMMENDATION"
    LEARNING_GOAL = "LEARNING_GOAL"
    CURRICULUM = "CURRICULUM"
    CURRICULUM_VERSION = "CURRICULUM_VERSION"
    LEARNING_OBJECTIVE = "LEARNING_OBJECTIVE"
    LEARNING_ACTION = "LEARNING_ACTION"
    TEACHING_SESSION = "TEACHING_SESSION"
    TEACHING_TURN = "TEACHING_TURN"
    TEACHING_SESSION_SUMMARY = "TEACHING_SESSION_SUMMARY"
    INTERACTION_EVIDENCE = "INTERACTION_EVIDENCE"
    ASSESSMENT_ITEM = "ASSESSMENT_ITEM"
    ASSESSMENT_RUBRIC = "ASSESSMENT_RUBRIC"
    ASSESSMENT_GRADE = "ASSESSMENT_GRADE"
    ASSESSMENT_FEEDBACK = "ASSESSMENT_FEEDBACK"
    LEARNING_CYCLE = "LEARNING_CYCLE"


class Artifact(Schema):
    artifact_id: str
    task_id: str
    type: ArtifactType
    name: str
    uri: str
    media_type: str
    content_hash: str
    size_bytes: int
    version: int = Field(ge=1)
    provider: str
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class StoredObject(Schema):
    """A content-addressed object in the object store: identical bytes are stored once and share this reference."""

    uri: str
    checksum: str  # sha256 of the content, hex
    key: str | None = None  # storage-relative key, e.g. objects/sha256/ab/<sha256>.png
    media_type: str
    size_bytes: int = Field(ge=0)
    reused: bool = False  # the object already existed when it was stored


class ArtifactDraft(Schema):
    """A request to store an artifact. `key` is local to one batch and lets drafts reference each other."""

    key: str
    name: str
    type: ArtifactType
    media_type: str
    content: str
    provider: str = "teaching-agent"
    parent_keys: list[str] = Field(default_factory=list)
    parent_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ArtifactBatch(Schema):
    drafts: list[ArtifactDraft] = Field(min_length=1)


class StoredArtifacts(Schema):
    artifacts: list[Artifact]
    by_key: dict[str, str]


class ArtifactView(Schema):
    """An artifact as the API serves it: no storage locations (an object store `uri` is a host path), no content digest
    and no byte length (`content_hash`, `size_bytes`): an artifact can hold an answer key (an open teaching question is
    stored as an ASSESSMENT_ITEM whose other fields the learner already sees), so its hash would verify guessed keys
    offline and its size would give the key's length. Its metadata is served through `PUBLIC_METADATA` only (see
    there). The internal `Artifact` keeps everything for storage, deduplication, integrity checks and the pipeline."""

    artifact_id: str
    task_id: str
    type: ArtifactType
    name: str
    media_type: str
    version: int
    provider: str
    parent_ids: list[str]
    metadata: dict
    created_at: datetime

    @classmethod
    def of(cls, artifact: Artifact) -> ArtifactView:
        data = artifact.model_dump(include=set(cls.model_fields))
        return cls(**data | {"metadata": public_metadata(artifact.type, data["metadata"])})


# Type -> the metadata keys a learner may see. Metadata is served by allowlist, never by removing known-bad keys:
# much of it is written from the lesson, whose exercises and check questions carry their answers, or by a model that
# was given them (an AUDIO_ASSET's narration `text` speaks the answers aloud, issue #28). A key not listed here, a
# type without an entry, and any value that is not a JSON scalar or a list of scalars are not served, so a new field
# stays private until someone reviews it. Never listed:
# - free text written by a model or a third party: narration text, descriptions, purposes, rationales, warnings,
#   reasons, provider titles and credits, selection and rejection records, search queries and prompts;
# - ids a model chooses or that are built from one (an image's `visual_id`, lesson section ids, slide ids and the
#   narration and video segment ids built from them, a narration segment's `source_ref`): a model can spell an answer
#   in an id; artifact ids and the code's content-derived plan ids are served instead;
# - length signals of answer-bearing text: a narration clip's `duration` and its TTS `usage` (characters, seconds);
# - storage locations, digests and byte lengths (`uri`, `object_key`, `checksum`, `*_hash`, `size_bytes`), and ids
#   that are digests of one answer-bearing object (an audio asset's `asset_id` is a prefix of its WAV's sha256).
# A test checks that every `ArtifactType` has an entry and that no listed key is a server-only key.
PUBLIC_METADATA: Mapping[ArtifactType, frozenset[str]] = {
    ArtifactType.LESSON: frozenset({"title", "level", "sections", "research_id", "research_status", "references",
                                    "visuals", "visual_status"}),
    ArtifactType.SCRIPT: frozenset(),
    ArtifactType.SLIDE: frozenset(),
    ArtifactType.SLIDE_PLAN: frozenset({"deck_id", "slides", "slide_types", "image_artifact_ids", "citation_ids"}),
    ArtifactType.PRESENTATION: frozenset({"presentation_id", "deck_id", "title", "language", "slide_plan_artifact_id",
                                          "lesson_artifact_id", "renderer", "format", "media_type", "slides",
                                          "elements", "image_artifact_ids", "citation_ids"}),
    ArtifactType.IMAGE_ASSET: frozenset({"asset_id", "plan_id", "visual_type", "origin", "width", "height", "format"}),
    ArtifactType.AUDIO_PLAN: frozenset({"audio_plan_id", "deck_id", "language", "voice", "segments", "required"}),
    ArtifactType.AUDIO_ASSET: frozenset({"audio_plan_id", "source_type", "format", "media_type", "sample_rate",
                                         "channels", "language", "voice", "provider", "model"}),
    ArtifactType.PRESENTATION_TIMELINE: frozenset({"timeline_id", "deck_id", "presentation_artifact_id",
                                                   "audio_plan_id", "duration", "slides", "segments",
                                                   "audio_artifact_ids"}),
    ArtifactType.VIDEO_PLAN: frozenset({"video_plan_id", "segments", "generated_clips", "audio_tracks", "duration",
                                        "width", "height", "fps", "transition", "timeline_ref", "presentation_ref",
                                        "subtitles"}),
    ArtifactType.VIDEO: frozenset({"video_plan_id", "duration", "width", "height", "fps", "codec", "audio_codec",
                                   "container", "media_type", "timeline_ref", "presentation_ref", "timeline_duration",
                                   "segments", "audio_tracks", "audio_stream", "subtitles", "subtitles_burned",
                                   "transition", "composer"}),
    ArtifactType.VIDEO_SEGMENT_PLAN: frozenset({"plan_id", "provider", "strategy", "seconds", "estimated_cost_usd",
                                                "cost_known"}),
    ArtifactType.GENERATED_VIDEO_ASSET: frozenset({"asset_id", "plan_id", "purpose", "insertion_strategy", "audio",
                                                   "width", "height", "fps", "format", "codec", "has_audio",
                                                   "provider", "model", "reused"}),
    ArtifactType.CODE: frozenset(),
    ArtifactType.EXERCISE: frozenset(),
    ArtifactType.REPORT: frozenset({"kind", "status", "revisions"}),
    ArtifactType.SUBTITLE: frozenset({"format", "language", "cues", "source", "burned_in"}),
    ArtifactType.LESSON_PLAN: frozenset({"objectives", "pedagogical_plan_id"}),
    ArtifactType.LEARNER_EVALUATION: frozenset({"score", "remaining_gaps", "action"}),
    ArtifactType.RESEARCH_BUNDLE: frozenset({"research_id", "status", "sources", "evidence", "citations", "warnings"}),
    ArtifactType.VISUAL_PLAN: frozenset({"plan_id", "visuals", "required"}),
    ArtifactType.LEARNING_EVIDENCE: frozenset({"source", "items", "concepts", "attempt_id", "misconceptions",
                                               "session_id"}),
    ArtifactType.LEARNER_MODEL: frozenset({"domain", "mastered", "developing", "weak", "evidence", "session_id"}),
    ArtifactType.KNOWLEDGE_GAPS: frozenset({"gap_set_id", "goal_id", "gaps"}),
    ArtifactType.PEDAGOGICAL_PLAN: frozenset({"plan_id", "targets", "prerequisites", "minutes", "strategy"}),
    ArtifactType.LEARNING_RECOMMENDATION: frozenset({"concepts", "goal_achieved", "plan_id"}),
    ArtifactType.LEARNING_GOAL: frozenset({"goal_id", "domain", "target_level", "status"}),
    ArtifactType.CURRICULUM: frozenset({"goal_id", "curriculum_id", "version", "version_id", "objectives"}),
    ArtifactType.CURRICULUM_VERSION: frozenset({"goal_id", "curriculum_id", "version", "version_id", "base_version"}),
    ArtifactType.LEARNING_OBJECTIVE: frozenset({"goal_id", "curriculum_id", "version", "version_id", "objective_id",
                                                "concept_id", "order", "role", "mode"}),
    ArtifactType.LEARNING_ACTION: frozenset({"action", "action_id", "goal_id", "objective_id", "concept_id",
                                             "curriculum_version", "session_id"}),
    ArtifactType.TEACHING_SESSION: frozenset({"session_id", "action", "mode", "concept_id", "objective_id"}),
    ArtifactType.TEACHING_TURN: frozenset({"session_id", "sequence", "speaker", "turn_type"}),
    ArtifactType.TEACHING_SESSION_SUMMARY: frozenset({"session_id", "completion_reason", "recommended_next_action",
                                                      "practice"}),
    ArtifactType.INTERACTION_EVIDENCE: frozenset({"session_id", "evidence_type", "concept_id", "correct", "practice"}),
    ArtifactType.ASSESSMENT_ITEM: frozenset({"concept_id", "objective_id", "response_type", "rubric_id"}),
    ArtifactType.ASSESSMENT_RUBRIC: frozenset({"criteria", "passing_threshold", "deterministic"}),
    ArtifactType.ASSESSMENT_GRADE: frozenset({"attempt_id", "assessment_item_id", "outcome", "score", "grader_type",
                                              "concept_id", "attempt_number"}),
    ArtifactType.ASSESSMENT_FEEDBACK: frozenset({"grade_id", "outcome"}),
    ArtifactType.LEARNING_CYCLE: frozenset({"cycle_id", "action", "goal_id", "objective_id", "concept_id"}),
}

# Keys that are never served whatever the allowlist says: storage locations, and digests or byte lengths of stored
# content (an object store key embeds the content's sha256; generation and composition keys are digests of inputs).
SERVER_ONLY_METADATA = frozenset({
    "uri", "key", "object_key", "checksum", "content_hash", "hash", "digest", "sha256", "generation_key",
    "composition_key", "size_bytes", "bytes", "file_size", "content_length", "byte_length",
})
SERVER_ONLY_SUFFIXES = ("_checksum", "_hash", "_digest", "_sha256", "_size_bytes")

_SCALAR = (str, int, float, bool, type(None))


def server_only(key: str) -> bool:
    return key in SERVER_ONLY_METADATA or key.endswith(SERVER_ONLY_SUFFIXES)


def _scalar(value) -> bool:
    return isinstance(value, _SCALAR) or (isinstance(value, list) and all(isinstance(v, _SCALAR) for v in value))


def public_metadata(type_: ArtifactType, metadata: Mapping) -> dict:
    """An artifact's metadata as the API serves it (ArtifactView): only the keys `PUBLIC_METADATA` lists for its type,
    and only flat values (a scalar, or a list of scalars). Nothing for a type without an entry."""
    allowed = PUBLIC_METADATA.get(type_, frozenset())
    return {k: v for k, v in metadata.items() if k in allowed and not server_only(k) and _scalar(v)}
