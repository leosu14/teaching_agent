"""Leases and fencing for executable workflow units (tasks; learning cycles keep theirs in the cycle record).

One worker at a time may execute a task. It claims the task's lease, a persisted row update that raises the task's
fencing generation (`fence`) by one, so every claim, including a takeover after expiry, makes all earlier holders
stale. The task record, its events, its artifacts and its assessment attempts and grades are written only if, at
commit time and in the same transaction, the stored generation is still `fence` and the lease has not expired, so a
worker that lost its lease cannot commit any of them, however long its slow call took; expiry alone stops it.

Not yet fenced at commit: learner-memory and curriculum writes made by a task's tools. The tool manager checks the
lease just before each such tool runs (a separate read, not part of the write); those writes are idempotent per task,
but making them atomic with the fence is a follow-up required before running workers in several processes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import Schema


class Lease(Schema):
    """The right to execute one resource, held by `owner` until `until` under fencing generation `fence`."""

    resource: str = "task"
    resource_id: str
    owner: str
    fence: int = Field(ge=1)
    until: datetime


class LeaseLost(RuntimeError):
    """A fenced write was refused: the writer's lease expired or another worker has claimed the resource since.
    Nothing of that write was stored. The message is fixed text; the generation and owner stay in the internal
    diagnostics."""

    MESSAGE = "This work was taken over by another request. Read it again for its current state."

    def __init__(self, lease: Lease | None = None) -> None:
        super().__init__(self.MESSAGE)
        self.lease = lease


class WriteConflict(RuntimeError):
    """An unleased, version-checked write lost to a newer version, or the resource is leased by a running worker."""
