"""Leases and fencing for executable workflow units (tasks; learning cycles keep theirs in the cycle record).

One worker at a time may execute a task. It claims the task's lease, a persisted row update that raises the task's
fencing generation (`fence`) by one, so every claim, including a takeover after expiry, makes all earlier holders
stale. The task record, its events, its artifacts, its assessment attempts and grades, and every learner-memory and
curriculum write made while the task runs are written only if, at commit time and in the same transaction, the stored
generation is still `fence` and the lease has not expired, so a worker that lost its lease cannot commit any of them,
however long its slow call took; expiry alone stops it.

The task record, events, artifacts and grades take the lease as an argument. Learner memory and curricula are written
by services several layers below the task (tools, the curriculum tracker), so the orchestrator instead names the lease
for everything that runs under it (`fenced_writes`), and those repositories fence each write by `write_fence()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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


# The lease that fences learner-memory and curriculum writes made by the work running in this asyncio task (and the
# tasks and threads it starts, which copy the context). None outside a task's execution: writes are then unfenced
# (API requests, teaching sessions), and version checks alone protect them.
_write_fence: ContextVar[Lease | None] = ContextVar("write_fence", default=None)


def write_fence() -> Lease | None:
    return _write_fence.get()


@contextmanager
def fenced_writes(lease: Lease | None) -> Iterator[None]:
    """Fence the learner-memory and curriculum writes made inside the block by `lease`: each is committed only while
    the lease is still current and unexpired (LeaseLost otherwise, and nothing of that write is stored)."""
    token = _write_fence.set(lease)
    try:
        yield
    finally:
        _write_fence.reset(token)
