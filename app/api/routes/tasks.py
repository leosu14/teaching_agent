from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import Field

from app.api.deps import container
from app.schemas.artifact import Artifact
from app.schemas.common import Schema
from app.schemas.events import Event
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
