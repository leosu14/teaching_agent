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


class Event(Schema):
    event_id: str
    type: str
    task_id: str | None = None
    node_id: str | None = None
    agent_id: str | None = None
    tool: str | None = None
    at: datetime = Field(default_factory=utcnow)
    data: dict = Field(default_factory=dict)
