"""Failure classification: one category per failed task, from the typed errors that already exist.

The category names what to look at first. A budget stop, a configuration problem or a provider failure is named as
such wherever it surfaced (they are found anywhere in the exception's cause chain); any other failure is named after
the workflow stage it happened in. The stage is always recorded too.
"""

from __future__ import annotations

from enum import Enum

from app.config.routing import ConfigError
from app.observability.budget import BudgetExceededError
from app.providers.core.errors import ProviderError
from app.runtime.workflow.nodes import ReviewRejected


class FailureCategory(str, Enum):
    CONFIGURATION = "ConfigurationError"
    BUDGET = "BudgetExceededError"
    PROVIDER = "ProviderError"
    PLANNING = "PlanningError"  # request interpretation, diagnostic and lesson planning
    RESEARCH = "ResearchError"
    TEACHING = "TeachingError"
    REVIEW = "ReviewError"
    VISUAL = "VisualError"
    PRESENTATION = "PresentationError"
    AUDIO = "AudioError"
    VIDEO = "VideoError"
    LEARNER = "LearnerMemoryError"
    EVALUATION = "EvaluationError"
    WORKFLOW = "WorkflowError"


# Workflow stage of a node, by node id prefix (first match wins).
STAGES: tuple[tuple[str, FailureCategory], ...] = (
    ("interpret_request", FailureCategory.PLANNING),
    ("learner_snapshot", FailureCategory.LEARNER),
    ("knowledge_graph", FailureCategory.LEARNER),
    ("load_goal", FailureCategory.LEARNER),
    ("record_diagnostic", FailureCategory.LEARNER),
    ("learner_model", FailureCategory.LEARNER),
    ("knowledge_gaps", FailureCategory.PLANNING),
    ("pedagogical_plan", FailureCategory.PLANNING),
    ("store_pedagogy", FailureCategory.PLANNING),
    ("diagnos", FailureCategory.PLANNING),
    ("answers_", FailureCategory.PLANNING),
    ("research", FailureCategory.RESEARCH),
    ("store_research", FailureCategory.RESEARCH),
    ("plan", FailureCategory.PLANNING),
    ("teach_review", FailureCategory.TEACHING),
    ("visual", FailureCategory.VISUAL),
    ("package_artifacts", FailureCategory.TEACHING),
    ("store_artifacts", FailureCategory.TEACHING),
    ("presentation_gate", FailureCategory.PRESENTATION),
    ("slide_plan", FailureCategory.PRESENTATION),
    ("validate_slide_plan", FailureCategory.PRESENTATION),
    ("store_slide_plan", FailureCategory.PRESENTATION),
    ("build_presentation", FailureCategory.PRESENTATION),
    ("render_presentation", FailureCategory.PRESENTATION),
    ("audio", FailureCategory.AUDIO),
    ("validate_audio_plan", FailureCategory.AUDIO),
    ("store_audio_plan", FailureCategory.AUDIO),
    ("synthesize_audio", FailureCategory.AUDIO),
    ("video", FailureCategory.VIDEO),
    ("validate_video_plan", FailureCategory.VIDEO),
    ("store_video_plan", FailureCategory.VIDEO),
    ("compose_video", FailureCategory.VIDEO),
    ("update_learner", FailureCategory.LEARNER),
    ("load_lesson", FailureCategory.EVALUATION),
    ("assess", FailureCategory.EVALUATION),
    ("evaluate", FailureCategory.EVALUATION),
    ("store_evaluation", FailureCategory.EVALUATION),
    ("next_recommendation", FailureCategory.EVALUATION),
    ("feedback", FailureCategory.EVALUATION),
)


def stage_of(node_id: str | None) -> FailureCategory:
    for prefix, category in STAGES:
        if node_id and node_id.startswith(prefix):
            return category
    return FailureCategory.WORKFLOW


def causes(exc: BaseException) -> list[BaseException]:
    """The exception and everything it was raised from, outermost first."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify(exc: BaseException, node_id: str | None) -> tuple[FailureCategory, FailureCategory]:
    """(category, stage) of a failure raised by `node_id`."""
    stage = stage_of(node_id)
    chain = causes(exc)
    for kind, category in ((BudgetExceededError, FailureCategory.BUDGET), (ConfigError, FailureCategory.CONFIGURATION),
                           (ProviderError, FailureCategory.PROVIDER)):
        if any(isinstance(e, kind) for e in chain):
            return category, stage
    if isinstance(exc, ReviewRejected):
        return FailureCategory.REVIEW, FailureCategory.REVIEW
    return stage, stage
