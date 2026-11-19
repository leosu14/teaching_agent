"""The teacher dashboard (MVP item 7): a class's learners at a glance, and the answers that could not be graded.

Everything here is a read of what the learners already have, aggregated across a class's roster for the class's
teacher. It carries the same learner-safe information the learner's own views do: mastery numbers, counts, goal and
curriculum progress, cycle states, and for an UNCERTAIN grade the question as the learner saw it and the learner's own
answer. Never an answer key (`expected_answer`, acceptable answers, rubrics, known errors), the grader's
`uncertainty_reason` (raw grader text, which can quote the expected answer) or internal workflow state."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.assessment import AttemptSource, GraderType, ResponseType
from app.schemas.common import Schema
from app.schemas.review import OverrideView


class GoalOverview(Schema):
    """A goal and, once a path is planned, its curriculum progress now (computed read-only)."""

    goal_id: str
    title: str
    domain: str
    target_level: str | None = None
    target_date: datetime | None = None
    status: str
    planned: bool  # a curriculum exists
    objectives_mastered: int = 0
    objectives_required: int = 0
    percent_complete: float = 0.0
    goal_complete: bool = False
    assignment_id: str | None = None  # set by an assignment of this class (#62)
    overdue: bool = False  # an assigned goal past its deadline and not complete (display only, D4)


class LearnerOverview(Schema):
    """One learner on the roster."""

    learner_id: str
    display_name: str
    joined_at: datetime
    average_mastery: float
    mastered: int  # concepts at or above the mastered threshold
    weak: int  # concepts with evidence and below the weak threshold
    due_for_review: int
    lessons_completed: int
    assessments_taken: int
    goals: list[GoalOverview] = Field(default_factory=list)  # not cancelled ones
    cycles: int  # learning cycles started
    current_cycle_status: str | None = None  # RUNNING, WAITING or BLOCKED: a cycle is under way
    last_cycle_status: str | None = None  # the latest cycle's status, whatever it is
    last_active_at: datetime | None = None  # the latest cycle's last change, else the profile's
    uncertain_open: int = 0  # UNCERTAIN grades not settled by a later graded attempt
    assignments_overdue: int = 0  # this class's assigned goals past their deadline and not complete (#62)


class ClassDashboard(Schema):
    class_id: str
    name: str
    created_at: datetime
    as_of: datetime
    learners: list[LearnerOverview] = Field(default_factory=list)
    uncertain_open: int = 0


class UncertainGrade(Schema):
    """An answer the grader could not decide (UNCERTAIN: never evidence, never a mastery change), for the teacher to
    look at. `resolved` when the learner answered the same item again later and that answer was graded."""

    attempt_id: str
    learner_id: str
    display_name: str
    assessment_item_id: str
    concept_id: str
    concept_name: str
    prompt: str  # the question as the learner saw it
    response_type: ResponseType
    choices: list[str] = Field(default_factory=list)  # a multiple-choice item's choices, as the learner saw them
    learner_answer: str
    source: AttemptSource
    attempt_number: int
    submitted_at: datetime
    grader_type: GraderType
    confidence: float
    later_attempts: int = 0  # attempts on the same item after this one
    resolved: bool = False  # a later attempt on the item was graded, or the teacher overrode this grade
    override: OverrideView | None = None  # the teacher's outcome (Important item 2)
