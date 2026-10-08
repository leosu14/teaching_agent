"""Resource ownership: which learner a resource belongs to. The API's authorization boundary asks this before it lets
a caller read or change anything, so ownership is decided by the stored data, never by ids the client sends.

Every lookup returns the owning learner id, or None when the resource does not exist. Lookups are reads only."""

from __future__ import annotations

from app.artifacts.service import ArtifactService
from app.learner.memory import LearnerMemoryService, UnknownGoal
from app.schemas.artifact import ArtifactType
from app.services.assessment import AssessmentService
from app.services.learning_cycles import LearningCycleService
from app.services.tasks import TaskService
from app.services.teaching import TeachingSessionService
from app.storage.repositories import NotFound


class OwnershipService:
    def __init__(self, *, tasks: TaskService, artifacts: ArtifactService, memory: LearnerMemoryService,
                 teaching: TeachingSessionService, assessment: AssessmentService,
                 cycles: LearningCycleService) -> None:
        self._tasks = tasks
        self._artifacts = artifacts
        self._memory = memory
        self._teaching = teaching
        self._assessment = assessment
        self._cycles = cycles

    def task(self, task_id: str) -> str | None:
        try:
            return self._tasks.get(task_id).learner_id
        except NotFound:
            return None

    def goal(self, goal_id: str) -> str | None:
        try:
            return self._memory.goal(goal_id).learner_id
        except UnknownGoal:
            return None

    def lesson(self, lesson_id: str) -> str | None:
        """A LESSON artifact id or a lesson task id (the two ways a lesson is addressed)."""
        try:
            artifact = self._artifacts.get(lesson_id)
        except (NotFound, KeyError):
            return self.task(lesson_id)
        return self.task(artifact.task_id) if artifact.type == ArtifactType.LESSON else None

    def teaching_session(self, session_id: str) -> str | None:
        session = self._teaching.repository.get(session_id)
        return session.learner_id if session is not None else None

    def assessment_item(self, item_id: str) -> str | None:
        """An item belongs to the learner whose lesson it was registered on."""
        item = self._assessment.repository.item(item_id)
        return self.lesson(item.lesson_id) if item is not None and item.lesson_id else None

    def assessment_attempt(self, attempt_id: str) -> str | None:
        attempt = self._assessment.repository.attempt(attempt_id)
        return attempt.learner_id if attempt is not None else None

    def learning_cycle(self, cycle_id: str) -> str | None:
        cycle = self._cycles.repository.get(cycle_id)
        return cycle.learner_id if cycle is not None else None
