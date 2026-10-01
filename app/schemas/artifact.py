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
    VIDEO = "VIDEO"
    CODE = "CODE"
    EXERCISE = "EXERCISE"
    REPORT = "REPORT"
    SUBTITLE = "SUBTITLE"
    LESSON_PLAN = "LESSON_PLAN"
    LEARNER_EVALUATION = "LEARNER_EVALUATION"
    RESEARCH_BUNDLE = "RESEARCH_BUNDLE"
    VISUAL_PLAN = "VISUAL_PLAN"


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
