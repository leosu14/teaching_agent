"""Persistence interface the runtime needs for tasks (implemented by the storage layer)."""

from __future__ import annotations

from typing import Protocol

from app.schemas.task import Task


class TaskRepository(Protocol):
    def save(self, task: Task) -> None: ...

    def get(self, task_id: str) -> Task: ...
