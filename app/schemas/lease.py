"""Leases and fencing for executable workflow units (tasks; learning cycles keep theirs in the cycle record).

One worker at a time may execute a task. It claims the task's lease, a persisted row update that raises the task's
fencing generation (`fence`) by one, so every claim, including a takeover after expiry, makes all earlier holders
stale. Every write the holder makes (the task record, its events, its artifacts) is applied only if, at commit time
and in the same transaction, the stored generation is still `fence` and the lease has not expired. A worker that lost
its lease therefore cannot commit anything, however long its slow call took, and expiry alone never lets it go on.
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
    """A protected write was refused: the writer's lease expired or another worker has claimed the resource since.
    Nothing was written. The message is fixed text; the generation and owner stay in the internal diagnostics."""

    MESSAGE = "This work was taken over by another request. Read it again for its current state."

    def __init__(self, lease: Lease | None = None) -> None:
        super().__init__(self.MESSAGE)
        self.lease = lease


class WriteConflict(RuntimeError):
    """An unleased, version-checked write lost to a newer version, or the resource is leased by a running worker."""
