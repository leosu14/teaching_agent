"""Event schema shared by the event bus, persistence, API and logs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.common import Schema, utcnow


class EventType:
    TASK_CREATED = "task.created"
    TASK_PLANNED = "task.planned"
    TASK_STARTED = "task.started"
    TASK_WAITING = "task.waiting"
    TASK_RESUMED = "task.resumed"
    TASK_PAUSED = "task.paused"
    TASK_CANCELLED = "task.cancelled"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_LEASE_LOST = "task.lease_lost"  # a worker's write was refused: its lease expired or was taken over
    NODE_STARTED = "node.started"
    NODE_FINISHED = "node.finished"
    NODE_FAILED = "node.failed"
    NODE_SKIPPED = "node.skipped"
    NODE_RETRY = "node.retry"
    AGENT_STARTED = "agent.started"
    AGENT_FINISHED = "agent.finished"
    AGENT_FAILED = "agent.failed"
    AGENT_VALIDATION_FAILED = "agent.validation_failed"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    TOOL_FAILED = "tool.failed"
    LLM_CALL = "llm.call"
    LLM_FAILED = "llm.failed"
    PROVIDER_REQUEST_STARTED = "provider.request_started"
    PROVIDER_REQUEST_COMPLETED = "provider.request_completed"
    PROVIDER_REQUEST_FAILED = "provider.request_failed"
    PROVIDER_FALLBACK = "provider.fallback"
    PROVIDER_RATE_LIMITED = "provider.rate_limited"
    # Spend ledger (ids and numbers only; stored even when the emitting task worker has lost its lease)
    BUDGET_EXHAUSTED = "budget.exhausted"  # a reservation was refused: nothing was sent
    SPEND_UNKNOWN = "spend.unknown"  # an attempt that may have been billed (timeout, cancellation, unusable answer)
    SPEND_ABANDONED = "spend.abandoned"  # a reservation nobody settled in time: its estimate was charged
    SPEND_AFTER_LEASE_LOSS = "spend.after_lease_loss"  # settled after the worker lost its task lease (kept)
    SPEND_SETTLE_FAILED = "spend.settle_failed"  # the outcome could not be stored; the sweep will charge the estimate
    SPEND_SETTLE_CONFLICT = "spend.settle_conflict"  # a second, different settlement of one attempt (ignored)
    REVIEW_STARTED = "review.started"
    REVIEW_COMPLETED = "review.completed"
    ARTIFACT_CREATED = "artifact.created"
    LEARNER_UPDATED = "learner.updated"
    LEARNER_MASTERY_UPDATED = "learner.mastery_updated"
    LEARNER_EVIDENCE_RECORDED = "learner.evidence_recorded"
    LEARNER_MODEL_BUILT = "learner.model_built"
    LEARNING_GOAL_RESOLVED = "learning_goal.resolved"
    KNOWLEDGE_GAPS_ANALYZED = "knowledge_gaps.analyzed"
    PEDAGOGICAL_PLAN_CREATED = "pedagogical_plan.created"
    NEXT_RECOMMENDATION_CREATED = "next_recommendation.created"
    GOAL_CREATED = "goal.created"
    GOAL_UPDATED = "goal.updated"
    GOAL_COMPLETED = "goal.completed"
    CURRICULUM_CREATED = "curriculum.created"
    CURRICULUM_REPLANNED = "curriculum.replanned"
    OBJECTIVE_STARTED = "objective.started"
    OBJECTIVE_MASTERED = "objective.mastered"
    LEARNING_ACTION_SELECTED = "learning_action.selected"
    ASSESSMENT_CREATED = "assessment.created"
    ASSESSMENT_WAITING = "assessment.waiting"
    ASSESSMENT_SUBMITTED = "assessment.submitted"
    # Semantic assessment (the AssessmentService): every graded answer, whoever asked for it.
    ASSESSMENT_STARTED = "assessment.started"
    ASSESSMENT_GRADED = "assessment.graded"
    ASSESSMENT_UNCERTAIN = "assessment.uncertain"
    ASSESSMENT_COMPLETED = "assessment.completed"
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_COMPLETED = "evaluation.completed"
    RECOMMENDATION_CREATED = "recommendation.created"
    RESEARCH_STARTED = "research.started"
    RESEARCH_QUERY_CREATED = "research.query_created"
    RESEARCH_SEARCH_COMPLETED = "research.search_completed"
    RESEARCH_SOURCE_SELECTED = "research.source_selected"
    RESEARCH_COMPLETED = "research.completed"
    RESEARCH_FAILED = "research.failed"
    VISUAL_STARTED = "visual.started"
    VISUAL_PLAN_CREATED = "visual.plan_created"
    IMAGE_SEARCH_STARTED = "image.search_started"
    IMAGE_SEARCH_COMPLETED = "image.search_completed"
    IMAGE_SELECTED = "image.selected"
    IMAGE_GENERATION_STARTED = "image.generation_started"
    IMAGE_GENERATION_COMPLETED = "image.generation_completed"
    IMAGE_VALIDATION_FAILED = "image.validation_failed"
    IMAGE_ASSET_CREATED = "image.asset_created"
    VISUAL_COMPLETED = "visual.completed"
    VISUAL_FAILED = "visual.failed"
    SLIDE_PLANNING_STARTED = "slide_planning.started"
    SLIDE_PLAN_CREATED = "slide_plan.created"
    SLIDE_PLAN_VALIDATED = "slide_plan.validated"
    PRESENTATION_BUILD_STARTED = "presentation.build_started"
    PRESENTATION_BUILD_COMPLETED = "presentation.build_completed"
    PRESENTATION_RENDER_STARTED = "presentation.render_started"
    PRESENTATION_RENDER_COMPLETED = "presentation.render_completed"
    PRESENTATION_ARTIFACT_CREATED = "presentation.artifact_created"
    PRESENTATION_FAILED = "presentation.failed"
    AUDIO_PLANNING_STARTED = "audio_planning.started"
    AUDIO_PLAN_CREATED = "audio_plan.created"
    AUDIO_PLAN_VALIDATED = "audio_plan.validated"
    TTS_STARTED = "tts.started"
    TTS_COMPLETED = "tts.completed"
    AUDIO_VALIDATION_FAILED = "audio.validation_failed"
    AUDIO_ASSET_CREATED = "audio.asset_created"
    TIMELINE_CREATED = "timeline.created"
    AUDIO_COMPLETED = "audio.completed"
    AUDIO_FAILED = "audio.failed"
    VIDEO_PLANNING_STARTED = "video_planning.started"
    VIDEO_PLAN_CREATED = "video_plan.created"
    VIDEO_PLAN_VALIDATED = "video_plan.validated"
    VIDEO_COMPOSITION_STARTED = "video_composition.started"
    VIDEO_COMPOSITION_COMPLETED = "video_composition.completed"
    VIDEO_VALIDATION_STARTED = "video.validation_started"
    VIDEO_VALIDATION_COMPLETED = "video.validation_completed"
    VIDEO_ARTIFACT_CREATED = "video.artifact_created"
    VIDEO_FAILED = "video.failed"
    VIDEO_STRATEGY_COMPLETED = "video_strategy.completed"
    VIDEO_GENERATION_SUBMITTED = "video_generation.submitted"
    VIDEO_GENERATION_POLLED = "video_generation.polled"
    VIDEO_GENERATION_COMPLETED = "video_generation.completed"
    VIDEO_GENERATION_FAILED = "video_generation.failed"
    VIDEO_GENERATION_CANCELLED = "video_generation.cancelled"
    VIDEO_GENERATION_REUSED = "video_generation.reused"
    VIDEO_GENERATION_WAITING = "video_generation.waiting"
    GENERATED_VIDEO_VALIDATED = "generated_video.validated"
    GENERATED_VIDEO_ASSET_CREATED = "generated_video.asset_created"
    GENERATED_VIDEO_FALLBACK = "generated_video.fallback"
    TEACHING_SESSION_STARTED = "teaching_session.started"
    TEACHING_TURN_CREATED = "teaching_turn.created"
    LEARNER_ANSWER_RECEIVED = "learner_answer.received"
    HINT_GIVEN = "hint.given"
    MISCONCEPTION_DETECTED = "misconception.detected"
    DIFFICULTY_CHANGED = "difficulty.changed"
    TEACHING_SESSION_PAUSED = "teaching_session.paused"
    TEACHING_SESSION_RESUMED = "teaching_session.resumed"
    TEACHING_SESSION_COMPLETED = "teaching_session.completed"
    TEACHING_SESSION_CANCELLED = "teaching_session.cancelled"
    TEACHING_SESSION_FAILED = "teaching_session.failed"
    # Learning cycles: one curriculum action executed end to end (stream key: the cycle id).
    LEARNING_CYCLE_STARTED = "learning_cycle.started"
    LEARNING_CYCLE_ACTION_SELECTED = "learning_cycle.action_selected"
    LEARNING_CYCLE_STEP_STARTED = "learning_cycle.step_started"
    LEARNING_CYCLE_STEP_COMPLETED = "learning_cycle.step_completed"
    LEARNING_CYCLE_WAITING = "learning_cycle.waiting"
    LEARNING_CYCLE_RESPONSE_RECEIVED = "learning_cycle.response_received"
    LEARNING_CYCLE_ACTION_COMPLETED = "learning_cycle.action_completed"
    LEARNING_CYCLE_RESUMED = "learning_cycle.resumed"
    LEARNING_CYCLE_COMPLETED = "learning_cycle.completed"
    LEARNING_CYCLE_FAILED = "learning_cycle.failed"
    LEARNING_CYCLE_CANCELLED = "learning_cycle.cancelled"


class Event(Schema):
    event_id: str
    type: str
    task_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    tool: str | None = None
    at: datetime = Field(default_factory=utcnow)
    data: dict = Field(default_factory=dict)


# --- What a learner sees -----------------------------------------------------------------------------------------
# `Event` is the internal diagnostic record: its free-form `data` carries whatever the emitter attached, including raw
# exception and validation text (which can quote model output, answer keys among it), storage URIs, provider and model
# names, token counts and costs. Learner-facing routes serve `LearnerEvent` instead: the type, a status and a fixed
# message chosen from that type, and the event's ids. Nothing is copied out of `data`, so a new payload field never
# leaks by default. The internal event log is unchanged.

KNOWN_EVENT_TYPES = frozenset(v for k, v in vars(EventType).items() if k.isupper() and isinstance(v, str))


class EventStatus(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    WAITING = "waiting"
    RETRYING = "retrying"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


# A step that is retried or replaced is not a failure the learner needs to act on.
_RETRY_TYPES = frozenset({EventType.NODE_RETRY, EventType.AGENT_VALIDATION_FAILED, EventType.IMAGE_VALIDATION_FAILED,
                          EventType.AUDIO_VALIDATION_FAILED, EventType.PROVIDER_FALLBACK,
                          EventType.PROVIDER_RATE_LIMITED, EventType.GENERATED_VIDEO_FALLBACK})
# The last word of an event type names its outcome.
_STATUS_BY_OUTCOME: dict[str, EventStatus] = {
    "started": EventStatus.STARTED, "waiting": EventStatus.WAITING, "paused": EventStatus.PAUSED,
    "resumed": EventStatus.RESUMED, "skipped": EventStatus.SKIPPED, "failed": EventStatus.FAILED,
    "cancelled": EventStatus.CANCELLED, "completed": EventStatus.COMPLETED, "finished": EventStatus.COMPLETED,
}
LEARNER_EVENT_MESSAGES: dict[EventStatus, str] = {
    EventStatus.STARTED: "A step started.",
    EventStatus.PROGRESS: "Work is in progress.",
    EventStatus.WAITING: "Waiting for your answer.",
    EventStatus.RETRYING: "A step is being retried.",
    EventStatus.PAUSED: "Paused.",
    EventStatus.RESUMED: "Resumed.",
    EventStatus.COMPLETED: "A step completed.",
    EventStatus.SKIPPED: "A step was skipped.",
    EventStatus.FAILED: "A step failed.",
    EventStatus.CANCELLED: "Cancelled.",
}
OTHER_EVENT_TYPE = "other"


def event_status(event_type: str) -> EventStatus:
    if event_type in _RETRY_TYPES:
        return EventStatus.RETRYING
    return _STATUS_BY_OUTCOME.get(event_type.rsplit(".", 1)[-1].rsplit("_", 1)[-1], EventStatus.PROGRESS)


class LearnerEvent(Schema):
    """An event as a learner sees it: built from approved fields and fixed messages only, never from `data`."""

    event_id: str
    type: str
    status: EventStatus
    message: str
    task_id: str | None = None
    node_id: str | None = None
    at: datetime

    @classmethod
    def of(cls, event: Event) -> LearnerEvent:
        known = event.type in KNOWN_EVENT_TYPES
        status = event_status(event.type) if known else EventStatus.PROGRESS
        return cls(event_id=event.event_id, type=event.type if known else OTHER_EVENT_TYPE, status=status,
                   message=LEARNER_EVENT_MESSAGES[status], task_id=event.task_id, node_id=event.node_id,
                   at=event.at)
