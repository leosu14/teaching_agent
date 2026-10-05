"""Learning cycles: the execution record of one curriculum NextLearningAction, end to end.

A cycle orchestrates what already exists (the lesson workflow, the interactive session, the evaluation workflow); it
owns no educational decision and no learner state. Its record says which action it executes, which child (lesson
task, teaching session, evaluation task) each step is, where the learner is needed, and what the action produced.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import Schema
from app.schemas.curriculum import NextLearningAction
from app.schemas.lesson import LearnerAnswer
from app.schemas.teaching import LearnerInputKind, PublicQuestion, TeachingTurn


class CycleStatus(str, Enum):
    RUNNING = "RUNNING"  # the cycle owes work: a step to start or a child to observe
    WAITING = "WAITING"  # a child waits for the learner (diagnostic, session question, evaluation answers)
    BLOCKED = "BLOCKED"  # a retryable failure; `resume` retries through the child's own recovery
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"  # a permanent failure (invalid action, invalid state, a workflow that cannot be retried)
    CANCELLED = "CANCELLED"


ACTIVE_CYCLE_STATUSES = frozenset({CycleStatus.RUNNING, CycleStatus.WAITING, CycleStatus.BLOCKED})
TERMINAL_CYCLE_STATUSES = frozenset({CycleStatus.COMPLETED, CycleStatus.FAILED, CycleStatus.CANCELLED})


class StepKind(str, Enum):
    LESSON = "LESSON"  # the lesson workflow (generated for the cycle, or an existing lesson reused)
    TEACHING_SESSION = "TEACHING_SESSION"  # the interactive session on that lesson (graded by the AssessmentService)
    EVALUATION = "EVALUATION"  # the evaluation workflow on that lesson


class StepStatus(str, Enum):
    PENDING = "PENDING"  # not started
    STARTED = "STARTED"  # its child key is recorded; the child may not exist yet
    RUNNING = "RUNNING"  # its child exists
    WAITING = "WAITING"  # its child waits for the learner
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class FailureKind(str, Enum):
    """Learner input is not a failure: it is WAITING."""

    VALIDATION = "VALIDATION"  # deterministic: a stale or invalid action, an objective already mastered, bad state
    PROVIDER = "PROVIDER"  # a model/provider failure (retryable under the existing provider policy)
    WORKFLOW = "WORKFLOW"  # a child workflow failed; retryable only where its own recovery allows it


class CycleResult(str, Enum):
    TAUGHT = "TAUGHT"  # a lesson and an interactive session
    EVALUATED = "EVALUATED"  # a lesson's evaluation
    GOAL_COMPLETED = "GOAL_COMPLETED"  # COMPLETE: the curriculum's completion rule holds
    NOTHING_DUE = "NOTHING_DUE"  # WAIT: nothing to do now


WaitKind = Literal["diagnostic_answers", "session_answer", "assessment_answers"]
PromptKind = Literal["DIAGNOSTIC_QUESTIONS", "SESSION_QUESTION", "ASSESSMENT_QUESTIONS"]


class CycleStep(Schema):
    step_id: str
    index: int = Field(ge=0)
    kind: StepKind
    status: StepStatus = StepStatus.PENDING
    child_key: str  # deterministic: the idempotency key the child is created (and found again) with
    child_id: str | None = None  # the lesson task, teaching session or evaluation task
    reuse_lesson: bool = False  # LESSON: an existing lesson for the objective may serve (REVIEW, PRACTICE, EVALUATE)
    reused: bool = False  # LESSON: an existing lesson did serve
    session_action: Literal["LEARN", "REVIEW", "PRACTICE", "EVALUATE"] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class CycleWaiting(Schema):
    """Where the learner is needed. `ref` is the child's waiting point (the task's waiting node, the session's open
    question turn), so a response is delivered to exactly that point once."""

    step_id: str
    kind: WaitKind
    child_id: str
    ref: str
    since: datetime


class CycleFailure(Schema):
    kind: FailureKind
    retryable: bool
    message: str = Field(max_length=500)
    step_id: str | None = None
    category: str | None = None  # the child task's failure category (ProviderError, BudgetExceededError, ...)
    at: datetime


class CycleOutcome(Schema):
    """What the executed action produced, copied from the children (never recomputed differently)."""

    result: CycleResult
    learning_evidence_ids: list[str] = Field(default_factory=list)
    mastery_changes: list[dict] = Field(default_factory=list)  # MasteryChange dumps from learner memory
    objective_progress: dict | None = None  # ObjectiveProgress dump after the update
    goal_complete: bool = False
    next_action: NextLearningAction | None = None  # the curriculum's next action after this cycle
    artifact_ids: dict[str, str] = Field(default_factory=dict)


class CycleEvent(Schema):
    """An event written with the cycle change and published after it (the cycle's outbox): stable id, once."""

    event_id: str
    type: str
    data: dict = Field(default_factory=dict)


class CycleLease(Schema):
    """Who may drive the cycle's children now: one request (or, later, one worker) at a time. Taken in a
    version-checked write before a child is created, run, resumed or given a response, released when the request
    returns; a lease left by a crashed process expires at `until` and can then be taken over."""

    token: str
    until: datetime


class LearningCycle(Schema):
    cycle_id: str  # stable: learner + idempotency key
    learner_id: str
    user_id: str
    idempotency_key: str
    action: NextLearningAction  # selected by the curriculum when the cycle started; never changed afterwards
    status: CycleStatus = CycleStatus.RUNNING
    steps: list[CycleStep] = Field(default_factory=list)
    waiting: CycleWaiting | None = None
    failure: CycleFailure | None = None
    outcome: CycleOutcome | None = None
    artifact_ids: dict[str, str] = Field(default_factory=dict)  # "cycle" (at start), "outcome" (at completion)
    pending_events: list[CycleEvent] = Field(default_factory=list)
    lease: CycleLease | None = None
    version: int = 1  # optimistic lock
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    def step(self, step_id: str) -> CycleStep:
        return next(s for s in self.steps if s.step_id == step_id)

    @property
    def current_step(self) -> CycleStep | None:
        """The first step that is not completed."""
        return next((s for s in self.steps if s.status != StepStatus.COMPLETED), None)


class CycleRequestRecord(Schema):
    """A learner response by its client_response_id: replayed, never delivered to a child twice."""

    cycle_id: str
    client_response_id: str
    request_hash: str
    step_id: str
    waiting_ref: str
    applied: bool = False
    created_at: datetime


class CycleConflict(ValueError):
    """Another cycle is active for the learner, the cycle changed concurrently, or a client_response_id was reused for
    a different response (409)."""


# --- API ---------------------------------------------------------------------------------------------------------


class StartLearningCycle(Schema):
    idempotency_key: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)


class CycleResponse(Schema):
    """The learner's response to the current prompt: `answers` for question sheets (diagnostic, evaluation), `answer`
    (and `kind`) for a session question."""

    client_response_id: str = Field(min_length=1, max_length=128)
    answers: list[LearnerAnswer] | None = Field(default=None, min_length=1)
    answer: str | None = Field(default=None, min_length=1, max_length=2000)
    kind: LearnerInputKind = "answer"

    @model_validator(mode="after")
    def _one_shape(self) -> CycleResponse:
        if (self.answers is None) == (self.answer is None):
            raise ValueError("send either `answers` (a question sheet) or `answer` (a session question)")
        return self


class LearnerPrompt(Schema):
    """What the learner must do now. Never an answer key: question sheets as the workflows publish them, the session's
    public question and the teacher turns since the learner last spoke."""

    kind: PromptKind
    step_id: str
    questions: list[dict] = Field(default_factory=list)
    question: PublicQuestion | None = None
    teacher_turns: list[TeachingTurn] = Field(default_factory=list)


class LearningCycleView(Schema):
    cycle_id: str
    learner_id: str
    idempotency_key: str
    status: CycleStatus
    action: NextLearningAction
    steps: list[CycleStep]
    waiting: CycleWaiting | None = None
    prompt: LearnerPrompt | None = None
    failure: CycleFailure | None = None
    outcome: CycleOutcome | None = None
    artifact_ids: dict[str, str] = Field(default_factory=dict)
    created: bool = False  # this request created the cycle
    replayed: bool = False  # this response had been received before
    busy_until: datetime | None = None  # another request is driving the cycle (its lease); read it again later
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
