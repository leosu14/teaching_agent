"""Teacher review (Important item 2): lesson approval before release, and overrides of UNCERTAIN grades.

A class can require its teacher's approval before a lesson reaches its learners (`lesson_approval`, off by default).
The decision is taken when a lesson is requested: a lesson task created for a learner on the roster of such a class
gets a `LessonRelease` row, PENDING, for that class. No row means released, so lessons of learners in no such class,
and every lesson requested before the setting was turned on, are released as before.

An override is a teacher's outcome for an answer the grader could not decide (UNCERTAIN). The stored grade stays as
it was (grades are immutable); the override is a record of its own, one per attempt, and becomes one piece of
learning evidence through learner memory's MasteryUpdater."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field

from app.schemas.common import Schema

NOTE_MAX = 500


class ReleaseStatus(str, Enum):
    PENDING = "PENDING"  # waiting for the teacher: held from the learner
    APPROVED = "APPROVED"  # released
    REJECTED = "REJECTED"  # not released: held from the learner, and never reused by a learning cycle


class LessonRelease(Schema):
    """A lesson requested for a learner while a class of theirs required approval. Final once decided."""

    task_id: str  # the lesson task
    learner_id: str
    class_id: str
    status: ReleaseStatus
    requested_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None  # the teacher's user id
    note: str | None = None


class LessonReleaseView(Schema):
    """A release as the class's teacher sees it: the learner, the lesson's topic and how far its generation got."""

    task_id: str
    lesson_id: str | None = None  # the LESSON artifact, once the task completed
    learner_id: str
    display_name: str
    topic: str
    subject: str
    task_status: str
    status: ReleaseStatus
    requested_at: datetime
    decided_at: datetime | None = None
    note: str | None = None


class ReleaseDecision(Schema):
    note: str | None = Field(default=None, max_length=NOTE_MAX)


class ClassSettings(Schema):
    """`PATCH /classes/{class_id}`: what the class's teacher may change."""

    lesson_approval: bool | None = None


OverrideOutcome = Literal["CORRECT", "PARTIAL", "INCORRECT"]
# The score each outcome counts with (CORRECT and INCORRECT are full and no credit; PARTIAL the middle).
OVERRIDE_SCORE: dict[str, float] = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


class OverrideGrade(Schema):
    outcome: OverrideOutcome
    note: str | None = Field(default=None, max_length=NOTE_MAX)


class GradeOverride(Schema):
    """A teacher's outcome for an UNCERTAIN grade. One per attempt, immutable."""

    attempt_id: str
    learner_id: str
    class_id: str
    teacher_id: str
    outcome: OverrideOutcome
    note: str | None = None
    evidence_id: str
    created_at: datetime


class OverrideView(Schema):
    """An override as the learner (and the teacher) sees it."""

    outcome: OverrideOutcome
    note: str | None = None
    created_at: datetime


class OverrideResult(Schema):
    attempt_id: str
    override: OverrideView
    replayed: bool
    evidence_ids: list[str]
    mastery_changes: list[dict] = Field(default_factory=list)


class LessonHeld(Exception):
    """The lesson waits for (or was refused) its teacher's release: nothing of it is served to the learner (409)."""


class ReleaseConflict(Exception):
    """The release was already decided otherwise, or the lesson is not ready to decide on (409)."""


class OverrideConflict(Exception):
    """The attempt already has a different override, or its grade is not UNCERTAIN (409)."""


HELD_MESSAGE = "this lesson is waiting for your teacher to release it"
REJECTED_MESSAGE = "your teacher did not release this lesson"
