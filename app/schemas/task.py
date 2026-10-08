"""First-class Task model."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.artifact import ArtifactType
from app.schemas.common import CostSummary, Schema, utcnow
from app.schemas.curriculum import CurriculumPlan, NextLearningAction
from app.schemas.evaluation import LearningRecommendation
from app.schemas.learner import MasteryChange
from app.schemas.lesson import LessonRequest
from app.schemas.pedagogy import EvaluationFeedback, NextLearningRecommendation
from app.schemas.workflow import NodeStatus, WorkflowState


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    REVIEWING = "REVIEWING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


TERMINAL_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})


class TaskError(Schema):
    kind: str
    message: str
    node_id: str | None = None
    category: str | None = None  # failure classification, e.g. BudgetExceededError, ProviderError, AudioError
    stage: str | None = None  # the workflow stage the failure happened in
    at: datetime = Field(default_factory=utcnow)


class TaskPlan(Schema):
    lesson_request: LessonRequest
    workflow_id: str
    steps: list[str]
    estimated_cost_usd: float
    inputs: dict[str, str] = Field(default_factory=dict)  # e.g. the lesson task an evaluation refers to


class WaitRequest(Schema):
    node_id: str
    kind: str
    prompt: dict


class ArtifactSummary(Schema):
    artifact_id: str
    type: ArtifactType
    name: str
    version: int
    uri: str
    parent_ids: list[str]


class StepSummary(Schema):
    node_id: str
    status: NodeStatus
    attempts: int
    duration_ms: float | None


class TaskResult(Schema):
    title: str
    artifacts: list[ArtifactSummary]
    mastery_changes: list[MasteryChange]
    review_verdict: str = ""
    revisions: int = 0
    estimated_level: str | None = None
    score: float | None = None
    remaining_gaps: list[str] = Field(default_factory=list)
    recommendation: LearningRecommendation | None = None
    next_recommendation: NextLearningRecommendation | None = None
    feedback: EvaluationFeedback | None = None
    warnings: list[str] = Field(default_factory=list)
    curriculum: CurriculumPlan | None = None  # a curriculum planning task: the validated (and stored) curriculum
    learning_action: NextLearningAction | None = None  # the next action after an evaluation, for curriculum learners


class TaskControl(Schema):
    pause_requested: bool = False
    cancel_requested: bool = False


class Task(Schema):
    task_id: str
    user_id: str
    learner_id: str
    request: str
    status: TaskStatus = TaskStatus.CREATED
    plan: TaskPlan | None = None
    current_step: str | None = None
    workflow: WorkflowState | None = None
    waiting: WaitRequest | None = None
    control: TaskControl = Field(default_factory=TaskControl)
    errors: list[TaskError] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    cost: CostSummary = Field(default_factory=CostSummary)
    result: TaskResult | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def steps(self) -> list[StepSummary]:
        if self.workflow is None:
            return []
        return [
            StepSummary(node_id=nid, status=s.status, attempts=s.attempts, duration_ms=s.duration_ms)
            for nid, s in self.workflow.node_states.items()
        ]


# --- What a learner sees -----------------------------------------------------------------------------------------
# `Task` is the internal record: its workflow state holds every node's output, answer keys included (diagnostic items
# and evaluation assessments carry `expected_answer` and `accepted_answers`), plus provider requests and metadata.
# Learner-facing routes serve `TaskView` instead, built field by field so a new internal field never leaks by default.


class ArtifactRef(Schema):
    """An artifact as a task result names it: no storage location."""

    artifact_id: str
    type: ArtifactType
    name: str
    version: int
    parent_ids: list[str]


class TaskResultView(Schema):
    title: str
    artifacts: list[ArtifactRef]
    mastery_changes: list[MasteryChange]
    review_verdict: str = ""
    revisions: int = 0
    estimated_level: str | None = None
    score: float | None = None
    remaining_gaps: list[str] = Field(default_factory=list)
    recommendation: LearningRecommendation | None = None
    next_recommendation: NextLearningRecommendation | None = None
    feedback: EvaluationFeedback | None = None
    warnings: list[str] = Field(default_factory=list)
    curriculum: CurriculumPlan | None = None
    learning_action: NextLearningAction | None = None

    @classmethod
    def of(cls, result: TaskResult) -> TaskResultView:
        data = result.model_dump(include=set(cls.model_fields) - {"artifacts"})
        return cls(**data, artifacts=[ArtifactRef(**a.model_dump(include=set(ArtifactRef.model_fields)))
                                      for a in result.artifacts])


class WaitingView(Schema):
    """What the task is waiting for. The prompt is the public request a waiting node publishes (question sheets
    without answer keys)."""

    kind: str
    prompt: dict


class TaskCostView(Schema):
    estimated_cost_usd: float
    actual_cost_usd: float


class TaskView(Schema):
    task_id: str
    learner_id: str
    request: str
    status: TaskStatus
    current_step: str | None
    waiting: WaitingView | None
    errors: list[TaskError]
    artifact_ids: list[str]
    cost: TaskCostView
    result: TaskResultView | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, task: Task) -> TaskView:
        return cls(
            task_id=task.task_id, learner_id=task.learner_id, request=task.request, status=task.status,
            current_step=task.current_step,
            waiting=WaitingView(kind=task.waiting.kind, prompt=task.waiting.prompt) if task.waiting else None,
            errors=task.errors, artifact_ids=task.artifact_ids,
            cost=TaskCostView(estimated_cost_usd=task.cost.estimated_cost_usd,
                              actual_cost_usd=task.cost.actual_cost_usd),
            result=TaskResultView.of(task.result) if task.result is not None else None,
            created_at=task.created_at, updated_at=task.updated_at)
