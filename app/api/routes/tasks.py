from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import Field

from app.api.deps import container
from app.schemas.artifact import Artifact
from app.schemas.common import Schema
from app.schemas.events import Event
from app.schemas.lesson import LearnerAnswer
from app.schemas.task import Task
from app.services.container import Container

router = APIRouter(prefix="/tasks", tags=["tasks"])


class CreateTask(Schema):
    request: str = Field(min_length=1, max_length=2000)
    learner_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)


@router.post("", response_model=Task, status_code=201)
async def create_task(body: CreateTask, c: Container = Depends(container)) -> Task:
    return await c.task_service.create_and_run(request=body.request, learner_id=body.learner_id, user_id=body.user_id)


class StartEvaluation(Schema):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)


class SubmitAnswers(Schema):
    answers: list[LearnerAnswer] = Field(min_length=1)


@router.post("/{task_id}/evaluation", response_model=Task, status_code=201)
async def start_evaluation(task_id: str, body: StartEvaluation | None = None,
                           c: Container = Depends(container)) -> Task:
    """Start a post-lesson evaluation task for a completed lesson task."""
    return await c.task_service.start_evaluation(task_id, user_id=(body or StartEvaluation()).user_id)


@router.post("/{task_id}/answers", response_model=Task)
async def submit_answers(task_id: str, body: SubmitAnswers, c: Container = Depends(container)) -> Task:
    """Submit answers for whatever the task is WAITING on and resume it."""
    return await c.task_service.submit_answers(task_id, body.model_dump(mode="json"))


@router.get("/{task_id}", response_model=Task)
def get_task(task_id: str, c: Container = Depends(container)) -> Task:
    return c.task_service.get(task_id)


@router.post("/{task_id}/pause", response_model=Task)
def pause_task(task_id: str, c: Container = Depends(container)) -> Task:
    return c.task_service.pause(task_id)


@router.post("/{task_id}/resume", response_model=Task)
async def resume_task(task_id: str, c: Container = Depends(container)) -> Task:
    return await c.task_service.resume(task_id)


@router.post("/{task_id}/cancel", response_model=Task)
def cancel_task(task_id: str, c: Container = Depends(container)) -> Task:
    return c.task_service.cancel(task_id)


@router.get("/{task_id}/artifacts", response_model=list[Artifact])
def task_artifacts(task_id: str, c: Container = Depends(container)) -> list[Artifact]:
    return c.task_service.artifacts(task_id)


@router.get("/{task_id}/events", response_model=list[Event])
def task_events(task_id: str, c: Container = Depends(container)) -> list[Event]:
    return c.task_service.events(task_id)
