"""Task application service: the single entry point used by the API and the CLI."""

from __future__ import annotations

from pathlib import Path

from app.artifacts.service import ArtifactService
from app.runtime.orchestrator.orchestrator import Orchestrator
from app.runtime.workflows.lesson_evaluation import LESSON_TASK
from app.runtime.workflows.lesson_evaluation import WORKFLOW_ID as EVALUATION_WORKFLOW
from app.runtime.workflows.lesson_generation import WORKFLOW_ID as LESSON_WORKFLOW
from app.schemas.artifact import Artifact
from app.schemas.events import Event
from app.schemas.lesson import DiagnosticAnswers, LessonRequest
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

    def create_lesson(self, *, lesson_request: LessonRequest, learner_id: str, user_id: str,
                      metadata: dict | None = None) -> Task:
        """A lesson task whose request is already structured (no interpretation step). `metadata` may carry the
        task's budget."""
        return self._orchestrator.create_planned_task(
            request=lesson_request.raw_request, learner_id=learner_id, user_id=user_id, workflow_id=LESSON_WORKFLOW,
            lesson_request=lesson_request, inputs={}, metadata=metadata)

    async def run(self, task_id: str) -> Task:
        """Execute a created task until it completes, waits, pauses or fails."""
        return await self._orchestrator.run(task_id)

    async def submit_assessment_for(self, learner_id: str, task_id: str, answers: DiagnosticAnswers) -> Task:
        task = self._tasks.get(task_id)
        if task.learner_id != learner_id:
            raise NotFound(f"task {task_id} does not belong to learner {learner_id}")
        return await self.submit_assessment(task_id, answers)

    async def submit_assessment(self, task_id: str, answers: DiagnosticAnswers) -> Task:
        return await self.submit_answers(task_id, answers.model_dump(mode="json"))

    async def submit_answers(self, task_id: str, payload: dict) -> Task:
        """Attach the learner's answers to whatever the task is waiting on, then resume it.

        The waiting node validates the payload (diagnostic answers or assessment answers)."""
        task = self._tasks.get(task_id)
        if task.status != TaskStatus.WAITING or task.waiting is None:
            raise InvalidTransition(f"task {task_id} is not waiting for answers")
        return await self._orchestrator.submit_input(task_id, task.waiting.node_id, payload)

    async def start_evaluation(self, lesson_task_id: str, *, user_id: str) -> Task:
        """Start the post-lesson evaluation of a completed lesson task. Runs until it waits for answers."""
        lesson = self._tasks.get(lesson_task_id)
        if lesson.status != TaskStatus.COMPLETED or lesson.plan is None or lesson.plan.workflow_id != LESSON_WORKFLOW:
            raise InvalidTransition(f"task {lesson_task_id} is not a completed lesson")
        task = self._orchestrator.create_planned_task(
            request=f"Evaluate lesson {lesson_task_id}", learner_id=lesson.learner_id, user_id=user_id,
            workflow_id=EVALUATION_WORKFLOW, lesson_request=lesson.plan.lesson_request,
            inputs={LESSON_TASK: lesson_task_id},
        )
        return await self._orchestrator.run(task.task_id)

    async def resume(self, task_id: str) -> Task:
        return await self._orchestrator.resume(task_id)

    def invalidate(self, task_id: str, node_ids: list[str]) -> list[str]:
        return self._orchestrator.invalidate(task_id, node_ids)

    def save(self, task: Task) -> None:
        self._tasks.save(task)

    def list_for_learner(self, learner_id: str) -> list[Task]:
        return self._tasks.list_for_learner(learner_id)

    def verify_artifact(self, artifact: Artifact) -> str | None:
        return self._artifacts.verify(artifact)

    def discard_artifact_object(self, artifact: Artifact) -> None:
        self._artifacts.discard_object(artifact)

    def export_artifact(self, artifact_id: str, target: Path) -> Path:
        """Copy an artifact's content to a local file (streamed). The artifact itself stays in the object store."""
        artifact = self._artifacts.get(artifact_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts.copy_object_to(artifact.uri, target)
        return target

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
