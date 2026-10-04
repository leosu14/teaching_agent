"""Production run schemas: the user's task, the preflight plan of a run and the machine-readable run report
(production_run.json). Nothing here ever holds a credential."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.artifact import ArtifactType
from app.schemas.common import Schema
from app.schemas.lesson import LessonRequest
from app.schemas.providers import HealthStatus
from app.schemas.task import TaskError
from app.schemas.usage import TaskBudget, TaskUsage

RUN_KEY = "production_run_key"  # Task.metadata key of a production task's identity
PRODUCTION = "production"  # Task.metadata key of the run's provider/model selection

# The artifact chain a production lesson must produce, in order.
REQUIRED_CHAIN: tuple[ArtifactType, ...] = (
    ArtifactType.RESEARCH_BUNDLE, ArtifactType.LESSON_PLAN, ArtifactType.LESSON, ArtifactType.IMAGE_ASSET,
    ArtifactType.PRESENTATION, ArtifactType.AUDIO_ASSET, ArtifactType.PRESENTATION_TIMELINE, ArtifactType.VIDEO,
)


class ProductionTask(Schema):
    """What the user asked for. The level framework and subject are resolved from it, never assumed."""

    level: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    language: str = Field(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")  # language of the lesson (BCP 47)
    learner_id: str = Field(min_length=1, max_length=128)
    subject: str | None = None  # default: the language itself, for a language level framework (CEFR)
    user_id: str = "production"
    generated_video: bool = False  # ask for optional generated video segments


class ProviderChoice(Schema):
    capability: str
    provider: str
    model: str | None = None
    fallbacks: list[str] = Field(default_factory=list)
    required: bool = False
    real: bool = False  # needs the network (not a mock)


class VideoGenerationPlan(Schema):
    """What generated video segments may cost a run, known before any provider call. The estimate is an upper
    bound: the strategy only plans clips for sections that need moving pictures."""

    requested: bool  # the task asked for generated segments
    enabled: bool  # requested, allowed by the settings and by the budget
    provider: str
    model: str | None = None
    real: bool = False  # needs the network (not the mock)
    max_segments: int
    max_seconds: float
    segment_seconds: tuple[float, float]  # (min, max) per clip
    estimated_seconds: float  # at most this many seconds are generated
    estimated_cost_usd: float | None = None  # None: the provider's price is not configured
    max_cost_usd: float | None = None
    required: bool = False
    failure_policy: str  # what a failed required clip does: fail the task, or continue with its fallback
    fallback: str = "the slide's existing image, else the slide itself"


class ProductionPlan(Schema):
    """Everything known before a single provider call: the resolved request, providers, budget and stages."""

    mode: Literal["offline", "production"]
    ready: bool  # a production run may start (configuration is complete)
    problems: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    task: ProductionTask
    lesson_request: LessonRequest
    workflow_id: str
    stages: list[str]
    providers: list[ProviderChoice]
    llm_routes: dict[str, str] = Field(default_factory=dict)  # agent id -> provider/model
    required_capabilities: list[str]
    budget: TaskBudget
    estimated_llm_cost_usd: float | None = None  # from configured pricing; None when the models are unpriced
    knowledge_concepts: int = 0  # concepts the knowledge base holds for the subject and topic
    video_generation: VideoGenerationPlan | None = None
    run_key: str
    # The long-term learning loop the lesson belongs to, and the curriculum-planning workflow's stages (used for a
    # learner with a learning goal; goals are optional).
    learning_loop: list[str] = Field(default_factory=list)
    curriculum_stages: list[str] = Field(default_factory=list)


class ArtifactNode(Schema):
    artifact_id: str
    type: ArtifactType
    name: str
    version: int
    uri: str
    content_hash: str
    size_bytes: int
    provider: str
    parent_ids: list[str]
    node_id: str | None = None  # the workflow node that created it
    request_ids: list[str] = Field(default_factory=list)  # provider requests made by that node


class NodeTrace(Schema):
    node_id: str
    status: str
    attempts: int
    duration_ms: float | None = None
    request_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)


class FinalVideo(Schema):
    artifact_id: str
    uri: str
    content_hash: str
    size_bytes: int
    duration: float | None = None
    width: int | None = None
    height: int | None = None


class GraphCheck(Schema):
    complete: bool
    missing_types: list[str] = Field(default_factory=list)
    video_ancestor_types: list[str] = Field(default_factory=list)


class ProductionReport(Schema):
    report_version: int = 1
    task_id: str
    run_key: str
    reused: bool = False  # an identical earlier run was reused: no provider call was made
    resumed: bool = False
    mode: str
    level: str
    topic: str
    language: str
    subject: str
    learner_id: str
    start_time: datetime
    end_time: datetime
    status: str
    providers: list[ProviderChoice]
    llm_routes: dict[str, str] = Field(default_factory=dict)
    budget: TaskBudget | None = None
    health: list[HealthStatus] = Field(default_factory=list)
    artifact_graph: list[ArtifactNode]
    graph_check: GraphCheck
    trace: list[NodeTrace]
    usage: TaskUsage
    estimated_cost_usd: float
    cost_complete: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[TaskError] = Field(default_factory=list)
    final_video: FinalVideo | None = None
    summary: dict = Field(default_factory=dict)  # research, lesson, visual, presentation, audio and video counts
    evaluation_task_id: str | None = None
