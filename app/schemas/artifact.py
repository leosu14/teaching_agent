"""Artifact metadata. Binary content lives in the object store; only metadata is in the database."""

from __future__ import annotations

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
