"""First-class Task model."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.artifact import ArtifactType
from app.schemas.common import CostSummary, Schema, utcnow
from app.schemas.evaluation import LearningRecommendation
from app.schemas.learner import MasteryChange
from app.schemas.lesson import LessonRequest
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
