"""A lesson as the learner reads it in the student web app (GET /lessons/{lesson_id}/reading).

The stored LESSON artifact is never served (app/services/artifact_content.py): its exercises and check questions
carry their answers. The reading is built from it by code (app/services/lesson_reading.py), field by field, so a new
lesson field never reaches a learner by default."""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import Schema
from app.schemas.lesson import SectionPurpose


class ReadingImage(Schema):
    """An answer-isolated IMAGE_ASSET of the section: its content is GET /artifacts/{artifact_id}/content."""

    artifact_id: str
    description: str | None = None


class ReadingSection(Schema):
    section_id: str
    purpose: SectionPurpose
    heading: str | None = None
    paragraphs: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    analogy: str | None = None
    images: list[ReadingImage] = Field(default_factory=list)


class ReadingReference(Schema):
    title: str
    publisher: str | None = None
    url: str | None = None  # http(s) only


class LessonReading(Schema):
    lesson_id: str  # the LESSON artifact
    task_id: str  # the lesson task
    title: str | None = None
    level: str | None = None
    objectives: list[str] = Field(default_factory=list)
    introduction: list[str] = Field(default_factory=list)
    sections: list[ReadingSection] = Field(default_factory=list)
    summary: list[str] = Field(default_factory=list)
    references: list[ReadingReference] = Field(default_factory=list)
    # Pieces held back: every sentence or field that states an answer the learner is (or will be) graded on, and
    # images the content policy would not serve. `question_open`: some are held back only while a question is open.
    withheld: int = Field(default=0, ge=0)
    question_open: bool = False
