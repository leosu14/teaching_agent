"""Class assignments (Important item 3, issue #62): a teacher assigns a goal, or course topics, with a deadline to a
whole class.

An assignment is a class-level template. For each learner on the class roster it is materialized as an ordinary
`LearningGoal` (`target_date = due_at`, `metadata.assignment_id` / `metadata.class_id`), so curricula, next actions,
deadline pressure, completion, learning cycles, the dashboard and learner deletion all work as they do for a goal the
learner set. There is no second goal or lesson concept: "assigned lessons" are course topics (units), resolved once,
at creation, to the knowledge-base concepts the goal targets (D1).

- Whole class only (D2): every learner on the roster when it is created, and every learner who joins later (D5, with
  the original `due_at`).
- `due_at` is an instant, stored in UTC (D3). Overdue is display only (D4).
- Targets cannot be changed (D6); title, description and `due_at` can, and reach the learners' goals without any
  replanning or model call (D10).
- On an assigned goal the learner may change only its priority, and pause or resume it (`ACTIVE` <-> `PAUSED`, D7 and
  D13), while the assignment is active and they are on the class roster. A learner who leaves keeps the goal,
  unlocked (D9).
- Completion is mastery (D8): the goal's completion rule. `completed_at` is the time of the latest evidence that
  supports it (D11).
- A late joiner whose assigned concepts are no longer all in the knowledge base still joins; that assignment is
  `NOT_ASSIGNED` for them (D12).

Status is derived when it is read, never stored as a flag that could go stale."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import AwareDatetime, Field, field_validator, model_validator

from app.schemas.common import Schema

ASSIGNMENT_KEY = "assignment_id"  # LearningGoal.metadata: the assignment the goal was materialized from
CLASS_KEY = "class_id"  # LearningGoal.metadata: that assignment's class
LOCKED_MESSAGE = "this goal is set by your teacher: only its priority, and pausing or resuming it, can be changed"
TOPICS_MAX = 50


def utc(value: datetime) -> datetime:
    """The same instant in UTC (an aware datetime is required)."""
    return value.astimezone(timezone.utc)


class AssignmentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"


class MaterializedState(str, Enum):
    ASSIGNED = "ASSIGNED"  # the learner's goal exists
    NOT_ASSIGNED = "NOT_ASSIGNED"  # a late join after the course changed: some assigned concepts are gone (D12)


class LearnerAssignmentStatus(str, Enum):
    """A learner's status on an assignment, derived when read (section 4 of #62)."""

    CANCELLED = "CANCELLED"  # the assignment, or the learner's goal, is cancelled
    NOT_ASSIGNED = "NOT_ASSIGNED"  # D12: no goal was made for this learner
    COMPLETED = "COMPLETED"  # the goal is COMPLETED, or its mastery completion rule holds now
    OVERDUE = "OVERDUE"  # not completed and past `due_at`
    IN_PROGRESS = "IN_PROGRESS"  # a curriculum exists and some objective has evidence or was taught
    NOT_STARTED = "NOT_STARTED"


class AssignedTopic(Schema):
    """One assigned course topic (a unit) and the concepts it was resolved to at creation."""

    topic: str
    concept_ids: list[str] = Field(min_length=1)


class ClassAssignment(Schema):
    """The stored assignment: teacher data. It names no learner, so a learner's deletion leaves nothing of them here."""

    assignment_id: str
    class_id: str
    created_by: str  # the teacher's user id
    title: str
    description: str = ""
    domain: str
    framework_id: str
    target_level: str | None = None  # a level goal ...
    topics: list[AssignedTopic] = Field(default_factory=list)  # ... or course topics (exactly one of the two)
    target_concepts: list[str] = Field(min_length=1)  # resolved once, at creation
    due_at: AwareDatetime
    status: AssignmentStatus = AssignmentStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    cancelled_at: datetime | None = None
    version: int = 1

    @field_validator("due_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return utc(value)

    @property
    def is_active(self) -> bool:
        return self.status == AssignmentStatus.ACTIVE


class AssignmentLearner(Schema):
    """Which learner got an assignment, and their goal: learner data (erased with the learner)."""

    assignment_id: str
    learner_id: str
    goal_id: str | None = None  # None when NOT_ASSIGNED
    state: MaterializedState = MaterializedState.ASSIGNED
    assigned_at: datetime
    source: str = "created"  # "created" (on the roster when it was made) or "joined" (joined later)


# --- requests ---------------------------------------------------------------------------------------------------------


class CreateAssignment(Schema):
    """POST /classes/{class_id}/assignments. Exactly one target: a level, or course topics. The same
    `idempotency_key` never creates a second assignment. There is no `learner_ids`: assignments are for the whole
    class (D2), and an unknown field is refused (422)."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    domain: str = Field(min_length=1, max_length=128)
    framework_id: str | None = Field(default=None, min_length=1, max_length=64)
    target_level: str | None = Field(default=None, min_length=1, max_length=32)
    topics: list[str] = Field(default_factory=list, max_length=TOPICS_MAX)
    due_at: AwareDatetime  # a naive time is refused (422): it names no instant (D3)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("due_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return utc(value)

    @field_validator("topics")
    @classmethod
    def _topics(cls, value: list[str]) -> list[str]:
        cleaned = [t.strip() for t in value]
        if any(not t or len(t) > 200 for t in cleaned):
            raise ValueError("each topic needs 1 to 200 characters")
        if len({t.casefold() for t in cleaned}) != len(cleaned):
            raise ValueError("topics must be unique")
        return cleaned

    @model_validator(mode="after")
    def _one_target(self) -> CreateAssignment:
        if bool(self.target_level) == bool(self.topics):
            raise ValueError("an assignment needs exactly one target: a target level, or course topics")
        return self


class UpdateAssignment(Schema):
    """PATCH /classes/{class_id}/assignments/{assignment_id}: title, description and deadline only. The targets
    (domain, level, topics) cannot be changed (D6): a body naming them is refused (422) by `extra="forbid"`."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    due_at: AwareDatetime | None = None

    @field_validator("due_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None


# --- views ------------------------------------------------------------------------------------------------------------


class TopicProgress(Schema):
    topic: str
    mastered: int
    total: int


class AssignmentTotals(Schema):
    learners: int = 0
    completed: int = 0
    in_progress: int = 0
    not_started: int = 0
    overdue: int = 0
    not_assigned: int = 0
    cancelled: int = 0


class AssignmentView(Schema):
    """An assignment as its teacher sees it, with totals over the learners still on the roster."""

    assignment_id: str
    class_id: str
    title: str
    description: str = ""
    domain: str
    framework_id: str
    target_level: str | None = None
    topics: list[str] = Field(default_factory=list)
    due_at: datetime
    status: AssignmentStatus
    created_at: datetime
    updated_at: datetime
    cancelled_at: datetime | None = None
    overdue: bool  # past due_at (display only, D4)
    totals: AssignmentTotals = Field(default_factory=AssignmentTotals)


class LearnerAssignmentProgress(Schema):
    """One learner on an assignment (computed read-only, as the dashboard does: nothing is refreshed or recorded)."""

    learner_id: str
    display_name: str
    goal_id: str | None = None
    status: LearnerAssignmentStatus
    paused: bool = False
    overdue: bool = False
    planned: bool = False  # a curriculum exists
    percent_complete: float = 0.0
    objectives_mastered: int = 0
    objectives_required: int = 0
    topics: list[TopicProgress] = Field(default_factory=list)
    completed_at: datetime | None = None
    completed_late: bool | None = None  # None while the completion time is unknown
    assigned_at: datetime
    source: str


class AssignmentDetail(Schema):
    assignment: AssignmentView
    as_of: datetime
    learners: list[LearnerAssignmentProgress] = Field(default_factory=list)


class LearnerAssignment(Schema):
    """An assignment as its learner sees it (GET /learners/{id}/assignments): the class, not its other learners."""

    assignment_id: str
    class_id: str
    class_name: str
    title: str
    description: str = ""
    domain: str
    target_level: str | None = None
    topics: list[str] = Field(default_factory=list)
    due_at: datetime
    assignment_status: AssignmentStatus
    progress: LearnerAssignmentProgress


class AssignmentConflict(Exception):
    """The idempotency key was used for a different assignment, or the assignment is cancelled (409)."""


class AssignmentNotFound(Exception):
    """No such assignment in this class: the uniform 404."""

    def __init__(self) -> None:
        super().__init__("not found")


class InvalidAssignment(ValueError):
    """The target cannot be resolved: an unknown domain, level or topic (422)."""


class GoalLocked(Exception):
    """The learner tried to change what their teacher controls on an assigned goal (409, D7 and D13)."""

    def __init__(self) -> None:
        super().__init__(LOCKED_MESSAGE)
