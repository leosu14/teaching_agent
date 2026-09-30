"""Task application service: the single entry point used by the API and the CLI."""

from __future__ import annotations

from app.artifacts.service import ArtifactService
from app.runtime.orchestrator.orchestrator import Orchestrator
from app.schemas.artifact import Artifact
from app.schemas.events import Event
from app.schemas.lesson import DiagnosticAnswers
from app.schemas.task import Task, TaskStatus
from app.runtime.tasks.state_machine import InvalidTransition
from app.storage.repositories import NotFound, SqlEventRepository, SqlTaskRepository


class TaskService:
    def __init__(self, orchestrator: Orchestrator, tasks: SqlTaskRepository, events: SqlEventRepository,
                 artifacts: ArtifactService) -> None:
        self._orchestrator = orchestrator
        self._tasks = tasks
        self._events = events
        self._artifacts = artifacts

    async def create_and_run(self, *, request: str, learner_id: str, user_id: str) -> Task:
        task = self._orchestrator.create_task(request=request, learner_id=learner_id, user_id=user_id)
        return await self._orchestrator.run(task.task_id)

    async def submit_assessment_for(self, learner_id: str, task_id: str, answers: DiagnosticAnswers) -> Task:
        task = self._tasks.get(task_id)
        if task.learner_id != learner_id:
            raise NotFound(f"task {task_id} does not belong to learner {learner_id}")
        return await self.submit_assessment(task_id, answers)

    async def submit_assessment(self, task_id: str, answers: DiagnosticAnswers) -> Task:
        task = self._tasks.get(task_id)
        if task.status != TaskStatus.WAITING or task.waiting is None:
            raise InvalidTransition(f"task {task_id} is not waiting for an assessment")
        return await self._orchestrator.submit_input(task_id, task.waiting.node_id, answers.model_dump(mode="json"))

    async def resume(self, task_id: str) -> Task:
        return await self._orchestrator.resume(task_id)

    def pause(self, task_id: str) -> Task:
        return self._orchestrator.request_pause(task_id)

    def cancel(self, task_id: str) -> Task:
        return self._orchestrator.request_cancel(task_id)

    def get(self, task_id: str) -> Task:
        return self._tasks.get(task_id)

    def artifacts(self, task_id: str) -> list[Artifact]:
        self._tasks.get(task_id)
        return self._artifacts.list_for_task(task_id)

    def artifact_content(self, artifact_id: str) -> tuple[Artifact, bytes]:
        return self._artifacts.get(artifact_id), self._artifacts.read(artifact_id)

    def events(self, task_id: str) -> list[Event]:
        self._tasks.get(task_id)
        return self._events.list_for_task(task_id)
