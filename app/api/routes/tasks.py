"""Lesson and evaluation tasks. Every route serves `TaskView`: the internal `Task` (workflow state with answer keys,
provider requests, metadata) never leaves the service layer."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import Field

from app.api.auth import Principal
from app.api.authorization import authorize, owned_task, principal
from app.api.deps import container
from app.schemas.artifact import ArtifactView
from app.schemas.common import Schema
from app.schemas.events import Event
from app.schemas.lesson import LearnerAnswer
from app.schemas.task import TaskView
from app.services.container import Container

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTask(Schema):
    request: str = Field(min_length=1, max_length=2000)
    learner_id: str = Field(min_length=1, max_length=128)
    # Deprecated and ignored: the task's user is the authenticated principal.
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


@router.post("", response_model=TaskView, status_code=201)
async def create_task(body: CreateTask, p: Principal = Depends(principal),
                      c: Container = Depends(container)) -> TaskView:
    authorize(p, body.learner_id)
    return TaskView.of(await c.task_service.create_and_run(request=body.request, learner_id=body.learner_id,
                                                           user_id=p.user_id))


class StartEvaluation(Schema):
    # Deprecated and ignored: the evaluation's user is the authenticated principal.
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


class SubmitAnswers(Schema):
    answers: list[LearnerAnswer] = Field(min_length=1)


@router.post("/{task_id}/evaluation", response_model=TaskView, status_code=201, dependencies=[Depends(owned_task)])
async def start_evaluation(task_id: str, body: StartEvaluation | None = None, p: Principal = Depends(principal),
                           c: Container = Depends(container)) -> TaskView:
    """Start a post-lesson evaluation task for a completed lesson task."""
    return TaskView.of(await c.task_service.start_evaluation(task_id, user_id=p.user_id))


@router.post("/{task_id}/answers", response_model=TaskView, dependencies=[Depends(owned_task)])
async def submit_answers(task_id: str, body: SubmitAnswers, c: Container = Depends(container)) -> TaskView:
    """Submit answers for whatever the task is WAITING on and resume it."""
    return TaskView.of(await c.task_service.submit_answers(task_id, body.model_dump(mode="json")))


@router.get("/{task_id}", response_model=TaskView, dependencies=[Depends(owned_task)])
def get_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(c.task_service.get(task_id))


@router.post("/{task_id}/pause", response_model=TaskView, dependencies=[Depends(owned_task)])
def pause_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(c.task_service.pause(task_id))


@router.post("/{task_id}/resume", response_model=TaskView, dependencies=[Depends(owned_task)])
async def resume_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(await c.task_service.resume(task_id))


@router.post("/{task_id}/cancel", response_model=TaskView, dependencies=[Depends(owned_task)])
def cancel_task(task_id: str, c: Container = Depends(container)) -> TaskView:
    return TaskView.of(c.task_service.cancel(task_id))


@router.get("/{task_id}/artifacts", response_model=list[ArtifactView], dependencies=[Depends(owned_task)])
def task_artifacts(task_id: str, c: Container = Depends(container)) -> list[ArtifactView]:
    return [ArtifactView.of(a) for a in c.task_service.artifacts(task_id)]


@router.get("/{task_id}/events", response_model=list[Event], dependencies=[Depends(owned_task)])
def task_events(task_id: str, c: Container = Depends(container)) -> list[Event]:
    return c.task_service.events(task_id)
