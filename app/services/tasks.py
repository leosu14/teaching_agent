"""Task application service: the single entry point used by the API and the CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Coroutine
from pathlib import Path

from app.artifacts.service import ArtifactService
from app.runtime.orchestrator.orchestrator import RESUMABLE, Orchestrator, TaskBusy
from app.runtime.workflows.lesson_evaluation import LESSON_TASK
from app.runtime.workflows.lesson_evaluation import WORKFLOW_ID as EVALUATION_WORKFLOW
from app.runtime.workflows.lesson_generation import WORKFLOW_ID as LESSON_WORKFLOW
from app.schemas.artifact import Artifact
from app.schemas.common import new_id
from app.schemas.events import Event
from app.schemas.job import JobConflict, JobKind, TaskJob
from app.schemas.lesson import DiagnosticAnswers, LessonRequest
from app.schemas.task import Task, TaskStatus
from app.runtime.tasks.state_machine import InvalidTransition
from app.storage.repositories import NotFound, SqlEventRepository, SqlTaskJobRepository, SqlTaskRepository

__all__ = ["JobConflict", "TaskBusy", "TaskService"]


class TaskService:
    """Two ways to execute a task. The `enqueue_*` methods (the API) validate, persist a job and return at once; the
    worker (app/services/worker.py) runs the job. The awaiting methods (`create_and_run`, `run`, `submit_answers`,
    `resume`, ...) run the workflow in the caller, for the CLI demos and the services that drive tasks themselves."""

    def __init__(self, orchestrator: Orchestrator, tasks: SqlTaskRepository, events: SqlEventRepository,
                 artifacts: ArtifactService, jobs: SqlTaskJobRepository | None = None,
                 job_max_attempts: int = 3) -> None:
        self._orchestrator = orchestrator
        self._tasks = tasks
        self._events = events
        self._artifacts = artifacts
        self._jobs = jobs
        self._job_max_attempts = job_max_attempts

    # --- background execution: validate, enqueue, return ----------------------------------------------------------

    def enqueue_task(self, *, request: str, learner_id: str, user_id: str,
                     idempotency_key: str | None = None) -> tuple[Task, TaskJob]:
        """Create a lesson task from its request and queue its execution. With an `idempotency_key`, a repeated
        call by the same user returns the same task and job (JobConflict when the key was used for another request)."""
        task_id = None
        if idempotency_key is not None:
            digest = hashlib.sha256(json.dumps(["task", user_id, idempotency_key]).encode()).hexdigest()
            task_id = f"task_{digest[:24]}"
        task = self._orchestrator.create_task(request=request, learner_id=learner_id, user_id=user_id,
                                              task_id=task_id)
        if task_id is not None and (task.request != request or task.learner_id != learner_id
                                    or task.user_id != user_id):
            raise JobConflict(task.task_id)
        if task_id is not None and task.status != TaskStatus.CREATED:  # a retry of a request that already ran
            job = self._job_store().latest_for_task(task.task_id)
            if job is not None:
                return task, job
        return task, self._enqueue(task.task_id, JobKind.RUN, {})

    def enqueue_answers(self, task_id: str, payload: dict) -> tuple[Task, TaskJob]:
        """Validate the answers to whatever the task is waiting on (InvalidInput, InvalidTransition) and queue them.
        The same answers submitted again while queued return the same job; different ones get JobConflict."""
        task = self._tasks.get(task_id)
        if task.status != TaskStatus.WAITING or task.waiting is None:
            raise InvalidTransition(f"task {task_id} is not waiting for answers")
        node_id = task.waiting.node_id
        data = self._orchestrator.check_input(task, node_id, payload)
        return task, self._enqueue(task_id, JobKind.SUBMIT_INPUT,
                                   {"node_id": node_id, "answers": data, "task_version": task.version})

    def enqueue_assessment_for(self, learner_id: str, task_id: str,
                               answers: DiagnosticAnswers) -> tuple[Task, TaskJob]:
        task = self._tasks.get(task_id)
        if task.learner_id != learner_id:
            raise NotFound(f"task {task_id} does not belong to learner {learner_id}")
        return self.enqueue_answers(task_id, answers.model_dump(mode="json"))

    def enqueue_evaluation(self, lesson_task_id: str, *, user_id: str) -> tuple[Task, TaskJob]:
        """Create the evaluation task of a completed lesson and queue its execution."""
        task = self.create_evaluation(lesson_task_id, user_id=user_id)
        return task, self._enqueue(task.task_id, JobKind.RUN, {})

    def enqueue_resume(self, task_id: str) -> tuple[Task, TaskJob]:
        """Queue the continuation of a paused, failed or abandoned task (InvalidTransition when it cannot resume)."""
        task = self._tasks.get(task_id)
        if task.status not in RESUMABLE and not self._orchestrator.waiting_on_work(task):
            raise InvalidTransition(f"task {task_id} in status {task.status.value} cannot be resumed")
        return task, self._enqueue(task_id, JobKind.RESUME, {})

    def enqueue_recovery(self, task_id: str) -> tuple[TaskJob, bool]:
        """Queue the continuation of a task left mid-run by a worker that died with no job left to reclaim (the
        sweeper). Returns (the task's active job, whether this call queued it): when the task already has an active
        job, that job will continue it and nothing is added."""
        job = TaskJob(job_id=new_id("job"), task_id=task_id, kind=JobKind.RECOVER, dedupe_key="RECOVER",
                      max_attempts=self._job_max_attempts)
        return self._job_store().enqueue(job)

    def job(self, task_id: str) -> TaskJob | None:
        """The task's latest job (None for a task that was only ever run in-process)."""
        return self._jobs.latest_for_task(task_id) if self._jobs is not None else None

    def jobs(self, task_id: str) -> list[TaskJob]:
        return self._jobs.for_task(task_id) if self._jobs is not None else []

    def _enqueue(self, task_id: str, kind: JobKind, payload: dict) -> TaskJob:
        """One active job per task: the same work again returns the queued job; other work is a JobConflict."""
        body = json.dumps({k: v for k, v in payload.items() if k != "task_version"}, sort_keys=True)
        dedupe_key = f"{kind.value}:{hashlib.sha256(body.encode()).hexdigest()[:32]}"
        job = TaskJob(job_id=new_id("job"), task_id=task_id, kind=kind, payload=payload, dedupe_key=dedupe_key,
                      max_attempts=self._job_max_attempts)
        active, inserted = self._job_store().enqueue(job)
        if not inserted and active.dedupe_key != dedupe_key:
            raise JobConflict(task_id)
        return active

    def _job_store(self) -> SqlTaskJobRepository:
        if self._jobs is None:
            raise RuntimeError("no job queue is configured")
        return self._jobs

    # --- in-process execution --------------------------------------------------------------------------------------

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

    def create_planned(self, *, request: LessonRequest, learner_id: str, user_id: str, workflow_id: str,
                       inputs: dict[str, str], metadata: dict | None = None, task_id: str | None = None) -> Task:
        """A task for a known workflow (e.g. curriculum planning) with its inputs; not run yet. With a deterministic
        `task_id`, an existing task of that id is returned instead of a second one being created."""
        return self._orchestrator.create_planned_task(
            request=request.raw_request, learner_id=learner_id, user_id=user_id, workflow_id=workflow_id,
            lesson_request=request, inputs=inputs, metadata=metadata, task_id=task_id)

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
        task = self.create_evaluation(lesson_task_id, user_id=user_id)
        return await self._orchestrator.run(task.task_id)

    def create_evaluation(self, lesson_task_id: str, *, user_id: str, metadata: dict | None = None) -> Task:
        """The evaluation task of a completed lesson task, created but not run (`metadata`: e.g. a learning cycle's
        step key)."""
        lesson = self._tasks.get(lesson_task_id)
        if lesson.status != TaskStatus.COMPLETED or lesson.plan is None or lesson.plan.workflow_id != LESSON_WORKFLOW:
            raise InvalidTransition(f"task {lesson_task_id} is not a completed lesson")
        return self._orchestrator.create_planned_task(
            request=f"Evaluate lesson {lesson_task_id}", learner_id=lesson.learner_id, user_id=user_id,
            workflow_id=EVALUATION_WORKFLOW, lesson_request=lesson.plan.lesson_request,
            inputs={LESSON_TASK: lesson_task_id}, metadata=metadata,
        )

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
        """Cancel a task. A task waiting on outside work (a video generation job) also has that work released: the
        job is cancelled at the provider when it supports it, otherwise recorded as cancelled locally."""
        before = self._tasks.get(task_id)
        on_work = self._orchestrator.waiting_on_work(before)
        task = self._orchestrator.request_cancel(task_id)
        if self._jobs is not None:  # queued work for a cancelled task never starts
            self._jobs.cancel_queued(task_id)
        if on_work:
            _run_sync(self._orchestrator.release(task_id))
            task = self._tasks.get(task_id)
        return task

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


def _run_sync(coro: Coroutine) -> object:
    """Run a coroutine to completion from synchronous code, even when the caller's thread runs an event loop."""
    result: dict = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target, name="task-release")
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
