"""Authored courses: the source format of the knowledge base (MVP item 5).

A course is one JSON file a course author edits: a subject, a level framework and the levels it covers, and units.
Each unit is one lesson topic at one level; each concept in it carries what the engine needs (id, name, level,
prerequisites), the teaching material research quotes (explanation, examples, a practice item) and the diagnostic
probes. `app/content/course.py` validates a course and compiles it to knowledge-base documents; nothing in the
engine reads this format directly.

Ids are lowercase. A concept id has no `/` or `:` and a course id has no `.`, so the compiled document ids
(`<course>/<concept>` and `<course>/<concept>/reference`) never collide.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas.common import Schema

COURSE_ID_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
CONCEPT_ID_PATTERN = r"^[a-z0-9_]+(?:\.[a-z0-9_]+)+$"
UNIT_ID_PATTERN = COURSE_ID_PATTERN


class CourseProbe(Schema):
    """A diagnostic question with its key. Probes never reach a learner with their answers: the diagnostic and the
    teaching agent ask the prompt and grade against `answer` and `accepted`."""

    prompt: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    accepted: list[str] = Field(default_factory=list)  # other correct answers (never a looser spelling)
    difficulty: float = Field(default=0.5, ge=0, le=1)


class CoursePractice(Schema):
    """One worked practice item for the lesson writer: quoted with the explanation in the concept's reference."""

    prompt: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class CourseConcept(Schema):
    concept_id: str = Field(pattern=CONCEPT_ID_PATTERN)
    name: str = Field(min_length=1)
    level: str | None = None  # None: the unit's level
    prerequisites: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    examples: list[str] = Field(min_length=1)
    practice: CoursePractice
    probes: list[CourseProbe] = Field(default_factory=list)


class CourseUnit(Schema):
    unit_id: str = Field(pattern=UNIT_ID_PATTERN)
    title: str = Field(min_length=1)
    topic: str = Field(min_length=1)  # what a lesson asks for: concept maps are looked up by subject and topic
    level: str = Field(min_length=1)
    summary: str = ""
    concepts: list[CourseConcept] = Field(default_factory=list)


class Course(Schema):
    schema_version: Literal[1] = 1
    course_id: str = Field(pattern=COURSE_ID_PATTERN)
    title: str = Field(min_length=1)
    subject: str = Field(min_length=1)  # the knowledge-base domain goals and lessons name
    framework: str = "cefr"
    levels: list[str] = Field(min_length=1)  # the framework's levels this course covers, lowest first
    language: str = "en"  # language of the explanations
    publisher: str = Field(min_length=1)
    description: str = ""
    units: list[CourseUnit] = Field(min_length=1)

    def concepts(self) -> list[tuple[CourseUnit, CourseConcept]]:
        return [(unit, concept) for unit in self.units for concept in unit.concepts]

    def level_of(self, unit: CourseUnit, concept: CourseConcept) -> str:
        return concept.level or unit.level


class CourseSummary(Schema):
    """What a loaded course holds, for the CLI and logs. No probes or answers."""

    course_id: str
    title: str
    subject: str
    framework: str
    levels: list[str]
    units: int
    concepts: int
    probes: int
    concepts_per_level: dict[str, int]
    topics: list[str]
