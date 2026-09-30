"""Task lifecycle: the allowed status transitions, enforced in one place."""

from __future__ import annotations

from app.schemas.common import utcnow
from app.schemas.task import Task, TaskStatus

S = TaskStatus
ALLOWED: dict[TaskStatus, frozenset[TaskStatus]] = {
    S.CREATED: frozenset({S.PLANNING, S.CANCELLED, S.FAILED}),
    S.PLANNING: frozenset({S.RUNNING, S.FAILED, S.CANCELLED, S.PAUSED}),
    S.RUNNING: frozenset({S.WAITING, S.REVIEWING, S.COMPLETED, S.FAILED, S.CANCELLED, S.PAUSED}),
    S.REVIEWING: frozenset({S.RUNNING, S.COMPLETED, S.FAILED, S.CANCELLED, S.PAUSED}),
    S.WAITING: frozenset({S.RUNNING, S.CANCELLED, S.FAILED}),
    S.PAUSED: frozenset({S.RUNNING, S.PLANNING, S.CANCELLED}),
    S.FAILED: frozenset({S.RUNNING, S.PLANNING}),  # recovery: resume from the last checkpoint
    S.COMPLETED: frozenset(),
    S.CANCELLED: frozenset(),
}


class InvalidTransition(ValueError):
    pass


def transition(task: Task, new: TaskStatus) -> None:
    if task.status == new:
        return
    if new not in ALLOWED[task.status]:
        raise InvalidTransition(f"task {task.task_id}: cannot go from {task.status.value} to {new.value}")
    task.status = new
    task.updated_at = utcnow()
