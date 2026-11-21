"""The sweeper: finds work a killed worker left mid-run and queues its continuation (MVP item 2).

    python -m app.services.worker --sweep    # one pass, then exit (cron-able)

The task worker also runs a pass every `TA_WORKER_SWEEP_SECONDS` while it serves.

What a killed worker leaves behind, and who recovers it:
- A queued job's worker died: the job's lease runs out and the next worker claims it again (the job queue does this).
  A job whose worker died on its last allowed attempt is FAILED instead (`fail_exhausted`, also run here, so it
  happens even while no worker is claiming).
- A task left PLANNING, RUNNING or REVIEWING with no job to reclaim: run in-process (a learning cycle's child task, a
  CLI demo) by a process that was killed, or its job lost. Its task lease runs out, then nothing would ever continue it
  and it looks stuck to the learner. The sweeper queues a RECOVER job for it, which resumes the task from its last
  checkpoint (completed nodes and stored artifacts are not produced again) only if it is still stranded when a worker
  picks it up. A task whose lease is live is being executed and is never touched, so the sweeper cannot cause a second
  execution; a task that holds no lease at all is swept only once it has not changed for `idle` (a lease's length).
- A task whose latest job FAILED is left alone: it already failed on every attempt, and is resumed only on request.
- A stored object a learner's data deletion queued but the store refused to delete (MVP item 9) is tried again
  (`purge_objects`), until it is deleted, or found shared and kept.
- A stranded task whose learner's data is deleted meanwhile gets no job: the recovery is a write for the learner, and
  is refused like any other once their rows are gone.

A stale worker that is still alive somewhere cannot interfere with the recovery: once its lease ran out, each of its
writes (task, events, artifacts, grades, learner memory, curricula) is refused at commit.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from app.runtime.orchestrator.orchestrator import TASK_LEASE
from app.schemas.job import JobStatus
from app.schemas.task import TaskStatus
from app.services.tasks import TaskService
from app.schemas.lease import LearnerDataGone
from app.storage.repositories import NotFound, ObjectPurge, SqlTaskJobRepository, SqlTaskRepository

logger = logging.getLogger(__name__)

# States a task is in only while a worker executes it: one left in them without a live lease was abandoned mid-run.
STRANDED = frozenset({TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.REVIEWING})


@dataclass
class SweepReport:
    failed_jobs: int = 0  # jobs FAILED because their worker died on the last allowed attempt
    recovered: list[str] = field(default_factory=list)  # tasks a RECOVER job was queued for
    left_failed: list[str] = field(default_factory=list)  # stranded tasks whose latest job FAILED: left alone
    objects_deleted: int = 0  # queued object deletions done on this pass
    objects_pending: int = 0  # queued object deletions the store refused again


class TaskSweeper:
    def __init__(self, tasks: SqlTaskRepository, jobs: SqlTaskJobRepository, service: TaskService, *,
                 idle: timedelta = TASK_LEASE, batch: int = 100,
                 purge_objects: Callable[[], ObjectPurge] | None = None) -> None:
        self._tasks = tasks
        self._jobs = jobs
        self._service = service
        self._idle = idle
        self._batch = batch
        self._purge_objects = purge_objects

    def sweep(self) -> SweepReport:
        """One pass. Safe to run from several processes at once: each task gets at most one active job. Which leases
        ran out, and how long a task has been idle, is judged by the repositories' clock: the database's on
        PostgreSQL, so sweepers on machines whose clocks drift agree with every worker on when a lease ended."""
        report = SweepReport(failed_jobs=self._jobs.fail_exhausted())
        if report.failed_jobs:
            logger.warning("sweeper: failed %d job(s) whose worker died on the last attempt", report.failed_jobs)
        for task_id in self._tasks.stranded(STRANDED, idle=self._idle, limit=self._batch):
            latest = self._jobs.latest_for_task(task_id)
            if latest is not None and latest.status == JobStatus.FAILED:
                report.left_failed.append(task_id)
                continue
            try:
                job, queued = self._service.enqueue_recovery(task_id)
            except (NotFound, LearnerDataGone):  # the learner's data was deleted since the task was listed
                continue
            if queued:
                report.recovered.append(task_id)
                logger.warning("sweeper: task %s was left mid-run by a worker that stopped; queued job %s to "
                               "continue it", task_id, job.job_id)
        if self._purge_objects is not None:
            purged = self._purge_objects()
            report.objects_deleted, report.objects_pending = purged.deleted, purged.pending
        return report
