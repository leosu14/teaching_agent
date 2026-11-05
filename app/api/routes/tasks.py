"""Lesson and evaluation tasks. Every route serves `TaskView`: the internal `Task` (workflow state with answer keys,
provider requests, metadata) never leaves the service layer.

Routes that would run a task's workflow (create, answers, evaluation, resume) only validate and queue the work, then
answer 202 Accepted with the task and its job; the worker runs it (app/services/worker.py). Clients poll
`GET /tasks/{task_id}` (the `Location` header) until `job.status` is no longer QUEUED or RUNNING."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Response
from pydantic import Field

from app.api.auth import Principal
from app.api.authorization import authorize, owned_task, principal
from app.api.deps import container
from app.schemas.artifact import ArtifactView
from app.schemas.common import Schema
from app.schemas.events import LearnerEvent
from app.schemas.job import TaskJob
from app.schemas.lesson import LearnerAnswer
from app.schemas.task import Task, TaskView
from app.services.container import Container

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTask(Schema):
    request: str = Field(min_length=1, max_length=2000)
    learner_id: str = Field(min_length=1, max_length=128)
    # Deprecated and ignored: the task's user is the authenticated principal.
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


def accepted(response: Response, queued: tuple[Task, TaskJob]) -> TaskView:
    task, job = queued
    response.headers["Location"] = f"/tasks/{task.task_id}"
    return TaskView.of(task, job)


@router.post("", response_model=TaskView, status_code=202)
def create_task(body: CreateTask, response: Response, p: Principal = Depends(principal),
                c: Container = Depends(container),
                idempotency_key: str | None = Header(default=None, min_length=1, max_length=128)) -> TaskView:
    """Create a lesson task and queue it. With an `Idempotency-Key` header, a retried request returns the same task
    instead of creating another (409 when the key was already used for a different request)."""
    authorize(p, body.learner_id)
    return accepted(response, c.task_service.enqueue_task(request=body.request, learner_id=body.learner_id,
                                                          user_id=p.user_id, idempotency_key=idempotency_key))


class StartEvaluation(Schema):
    # Deprecated and ignored: the evaluation's user is the authenticated principal.
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


class SubmitAnswers(Schema):
    answers: list[LearnerAnswer] = Field(min_length=1)


@router.post("/{task_id}/evaluation", response_model=TaskView, status_code=202, dependencies=[Depends(owned_task)])
def start_evaluation(task_id: str, response: Response, body: StartEvaluation | None = None,
                     p: Principal = Depends(principal), c: Container = Depends(container)) -> TaskView:
    """Create the post-lesson evaluation task of a completed lesson task and queue it."""
    return accepted(response, c.task_service.enqueue_evaluation(task_id, user_id=p.user_id))


@router.post("/{task_id}/answers", response_model=TaskView, status_code=202, dependencies=[Depends(owned_task)])
def submit_answers(task_id: str, body: SubmitAnswers, response: Response,
                   c: Container = Depends(container)) -> TaskView:
    """Validate the answers to whatever the task is WAITING on (422 when they do not fit) and queue its
    continuation."""
    return accepted(response, c.task_service.enqueue_answers(task_id, body.model_dump(mode="json")))


@router.get("/{task_id}", response_model=TaskView, dependencies=[Depends(owned_task)])
def get_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(c.task_service.get(task_id), c.task_service.job(task_id))


@router.post("/{task_id}/pause", response_model=TaskView, dependencies=[Depends(owned_task)])
def pause_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(c.task_service.pause(task_id), c.task_service.job(task_id))


@router.post("/{task_id}/resume", response_model=TaskView, status_code=202, dependencies=[Depends(owned_task)])
def resume_task(task_id: str, response: Response, c: Container = Depends(container)) -> TaskView:
    """Queue the continuation of a paused, failed or abandoned task from its last checkpoint."""
    return accepted(response, c.task_service.enqueue_resume(task_id))


@router.post("/{task_id}/cancel", response_model=TaskView, dependencies=[Depends(owned_task)])
def cancel_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(c.task_service.cancel(task_id), c.task_service.job(task_id))


@router.get("/{task_id}/artifacts", response_model=list[ArtifactView], dependencies=[Depends(owned_task)])
def task_artifacts(task_id: str, c: Container = Depends(container)) -> list[ArtifactView]:
    return [ArtifactView.of(a) for a in c.task_service.artifacts(task_id)]


@router.get("/{task_id}/events", response_model=list[LearnerEvent], dependencies=[Depends(owned_task)])
def task_events(task_id: str, c: Container = Depends(container)) -> list[LearnerEvent]:
    return [LearnerEvent.of(e) for e in c.task_service.events(task_id)]
