"""Persistence interface the runtime needs for tasks (implemented by the storage layer)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Protocol

from app.schemas.lease import Lease
from app.schemas.task import Task, TaskControl


class TaskRepository(Protocol):
    def create(self, task: Task) -> bool: ...

    def save(self, task: Task, *, lease: Lease | None = None, clear_control: Iterable[str] = ()) -> None: ...

    def get(self, task_id: str) -> Task: ...

    def claim(self, task_id: str, owner: str, ttl: timedelta) -> Lease | None: ...

    def renew(self, lease: Lease, ttl: timedelta) -> Lease | None: ...

    def release(self, lease: Lease) -> bool: ...

    def holds(self, lease: Lease) -> bool: ...

    def control(self, task_id: str) -> TaskControl: ...

    def request_control(self, task_id: str, *, pause: bool = False, cancel: bool = False) -> None: ...
