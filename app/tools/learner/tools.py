"""Learner-memory tools: the only way agents and workflows read or write learner history."""

from __future__ import annotations

from app.learner.memory import LearnerMemoryService
from app.observability.scope import ExecutionScope
from app.schemas.common import Schema
from app.schemas.learner import LearnerSnapshot, LearnerSummary, MasteryUpdate
from app.schemas.evaluation import EvaluationOutcome
from app.schemas.lesson import LessonOutcome
from app.tools.base import Tool


class LearnerRef(Schema):
    learner_id: str


class SnapshotInput(Schema):
    learner_id: str
    subject: str
    framework_id: str
    target_level: str | None = None


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
