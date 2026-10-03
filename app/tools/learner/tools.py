"""Learner-memory tools: the only way agents and workflows read or write learner history."""

from __future__ import annotations

from pydantic import Field

from app.learner.memory import LearnerMemoryService, UnknownGoal
from app.observability.scope import ExecutionScope
from app.schemas.common import Schema
from app.schemas.events import EventType
from app.schemas.evaluation import EvaluationOutcome
from app.schemas.learner import LearnerSnapshot, LearnerSummary, LearningGoal, MasteryUpdate
from app.schemas.lesson import DiagnosticOutcome, LessonOutcome
from app.schemas.pedagogy import LearnerModel
from app.tools.base import Tool, ToolError


class LearnerRef(Schema):
    learner_id: str


class SnapshotInput(Schema):
    learner_id: str
    subject: str
    framework_id: str
    target_level: str | None = None


class LearnerModelQuery(Schema):
    learner_id: str
    domain: str
    framework_id: str
    target_level: str | None = None
    concept_ids: list[str] = Field(default_factory=list)  # the domain's known concepts (to list the unknown ones)


class GoalQuery(Schema):
    learner_id: str
    domain: str
    topic: str
    topic_concepts: list[str] = Field(default_factory=list)
    target_level: str | None = None
    goal_id: str | None = None


class LearnerSummaryTool(Tool[LearnerRef, LearnerSummary]):
    name = "learner.summary"
    description = "Profile summary: subjects studied, levels and preferences."
    input_model = LearnerRef
    output_model = LearnerSummary
    permissions = frozenset({"learner:read"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: LearnerRef, scope: ExecutionScope) -> LearnerSummary:
        return self._memory.summary(data.learner_id)


class LearnerSnapshotTool(Tool[SnapshotInput, LearnerSnapshot]):
    name = "learner.snapshot"
    description = "What the learner knows, probably does not know, should learn next and should review."
    input_model = SnapshotInput
    output_model = LearnerSnapshot
    permissions = frozenset({"learner:read"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: SnapshotInput, scope: ExecutionScope) -> LearnerSnapshot:
        return self._memory.snapshot(data.learner_id, data.subject, data.framework_id, data.target_level)


class RecordLessonTool(Tool[LessonOutcome, MasteryUpdate]):
    name = "learner.record_lesson"
    description = "Record a completed lesson and its assessment evidence in long-term memory."
    input_model = LessonOutcome
    output_model = MasteryUpdate
    permissions = frozenset({"learner:write"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: LessonOutcome, scope: ExecutionScope) -> MasteryUpdate:
        return self._memory.record_lesson(data, scope)


class RecordEvaluationTool(Tool[EvaluationOutcome, MasteryUpdate]):
    name = "learner.record_evaluation"
    description = "Record post-lesson assessment evidence in long-term memory and return the mastery changes."
    input_model = EvaluationOutcome
    output_model = MasteryUpdate
    permissions = frozenset({"learner:write"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: EvaluationOutcome, scope: ExecutionScope) -> MasteryUpdate:
        return self._memory.record_evaluation(data, scope)


class RecordDiagnosticTool(Tool[DiagnosticOutcome, MasteryUpdate]):
    name = "learner.record_diagnostic"
    description = "Record a finished diagnostic's graded answers as learning evidence and update mastery from it."
    input_model = DiagnosticOutcome
    output_model = MasteryUpdate
    permissions = frozenset({"learner:write"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: DiagnosticOutcome, scope: ExecutionScope) -> MasteryUpdate:
        return self._memory.record_diagnostic(data, scope)


class LearnerModelTool(Tool[LearnerModelQuery, LearnerModel]):
    name = "learner.model"
    description = "The learner's current model in a domain, built deterministically from mastery, evidence and history."
    input_model = LearnerModelQuery
    output_model = LearnerModel
    permissions = frozenset({"learner:read"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: LearnerModelQuery, scope: ExecutionScope) -> LearnerModel:
        model = self._memory.learner_model(data.learner_id, data.domain, data.framework_id, data.target_level,
                                           data.concept_ids)
        scope.emit(EventType.LEARNER_MODEL_BUILT, learner_id=model.learner_id, domain=model.domain,
                   mastered=model.mastered_concepts, developing=model.developing_concepts,
                   weak=model.weak_concepts, evidence=model.evidence_count)
        return model


class LearningGoalTool(Tool[GoalQuery, LearningGoal]):
    name = "learning_goal.resolve"
    description = "The learning goal a lesson is planned against: named, the learner's active goal, or implicit."
    input_model = GoalQuery
    output_model = LearningGoal
    permissions = frozenset({"learner:read"})

    def __init__(self, memory: LearnerMemoryService) -> None:
        self._memory = memory

    async def run(self, data: GoalQuery, scope: ExecutionScope) -> LearningGoal:
        try:
            goal = self._memory.resolve_goal(data.learner_id, data.domain, data.topic, data.topic_concepts,
                                             data.target_level, data.goal_id)
        except (UnknownGoal, ValueError) as exc:
            raise ToolError(f"no learning goal: {exc}") from exc
        scope.emit(EventType.LEARNING_GOAL_RESOLVED, goal_id=goal.goal_id, targets=goal.target_concepts)
        return goal
