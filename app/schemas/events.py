"""Event schema shared by the event bus, persistence, API and logs."""

from __future__ import annotations

from datetime import datetime

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
    REVIEW_STARTED = "review.started"
    REVIEW_COMPLETED = "review.completed"
    ARTIFACT_CREATED = "artifact.created"
    LEARNER_UPDATED = "learner.updated"
    LEARNER_MASTERY_UPDATED = "learner.mastery_updated"
    ASSESSMENT_CREATED = "assessment.created"
    ASSESSMENT_WAITING = "assessment.waiting"
    ASSESSMENT_SUBMITTED = "assessment.submitted"
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


class Event(Schema):
    event_id: str
    type: str
    task_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    tool: str | None = None
    at: datetime = Field(default_factory=utcnow)
    data: dict = Field(default_factory=dict)
