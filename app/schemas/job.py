"""Task jobs: the background work queue that executes tasks outside the HTTP request.

An API request that would run a task's workflow (create a lesson, submit answers, start an evaluation, resume) only
validates and enqueues a `TaskJob`, then answers 202. A worker process claims queued jobs one at a time under a job
lease and runs them through the orchestrator, which still takes the task's own lease and fences every task write. A
task has at most one active (QUEUED or RUNNING) job; re-submitting the same work returns that job.

A job is DONE once the worker ran its task to a stopping point: the task's own status (COMPLETED, WAITING, FAILED,
...) says how the task ended. A job is FAILED when the worker itself could not run it (an unexpected error on every
attempt, or work that no longer applies to the task); the task then stays at its last checkpoint and can be resumed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.common import Schema, utcnow


class JobKind(str, Enum):
    RUN = "RUN"  # plan (if needed) and execute a created task
    SUBMIT_INPUT = "SUBMIT_INPUT"  # attach a waiting task's answers and continue it
    RESUME = "RESUME"  # continue a paused, failed or abandoned task from its last checkpoint


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ACTIVE_JOB_STATUSES = frozenset({JobStatus.QUEUED, JobStatus.RUNNING})


class JobErrorCategory(str, Enum):
    INVALID_STATE = "invalid_state"  # the task no longer accepts this work (e.g. no longer waiting at that step)
    INVALID_INPUT = "invalid_input"  # the answers were refused when the worker applied them
    WORKER_ERROR = "worker_error"  # an unexpected error on every attempt


JOB_ERROR_MESSAGES: dict[JobErrorCategory, str] = {
    JobErrorCategory.INVALID_STATE: "The task no longer accepts this request. Read the task for its current state.",
    JobErrorCategory.INVALID_INPUT: "The answers could not be applied to the task.",
    JobErrorCategory.WORKER_ERROR: "The task could not be run. It stays at its last checkpoint and can be resumed.",
}


class TaskJob(Schema):
    """The internal job record. `error` is redacted exception text for trusted diagnostics, never served."""

    job_id: str
    task_id: str
    kind: JobKind
    status: JobStatus = JobStatus.QUEUED
    payload: dict = Field(default_factory=dict)
    dedupe_key: str  # what makes two submissions the same work (kind plus a hash of the payload)
    attempts: int = 0  # claims so far, including ones whose worker died
    max_attempts: int = Field(default=3, ge=1)
    available_at: datetime = Field(default_factory=utcnow)  # not claimed before this (retry backoff)
    lease_owner: str | None = None
    lease_until: datetime | None = None
    task_status: str | None = None  # the task's status (a TaskStatus value) when the job finished
    error: str | None = None
    error_category: JobErrorCategory | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_JOB_STATUSES


class TaskJobView(Schema):
    """A job as a client polling its task sees it: no payload, no worker identity, no exception text."""

    job_id: str
    kind: JobKind
    status: JobStatus
    attempts: int
    message: str | None  # a fixed message when the job FAILED (by its category), never the exception text
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None

    @classmethod
    def of(cls, job: TaskJob) -> TaskJobView:
        return cls(job_id=job.job_id, kind=job.kind, status=job.status, attempts=job.attempts,
                   message=JOB_ERROR_MESSAGES[job.error_category] if job.error_category is not None else None,
                   created_at=job.created_at, updated_at=job.updated_at, finished_at=job.finished_at)


class JobConflict(RuntimeError):
    """The task already has different work queued or running (409). Fixed text."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"task {task_id} already has other work queued or running: read it again later")
        self.task_id = task_id
