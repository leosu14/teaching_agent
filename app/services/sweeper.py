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
- Expired shared-state rows (issue #66 PR 2): rate-limit starts and slots, stream slots and research-cache entries
  past their expiry in database time, of keys no admission has counted since (`shared_state`).
- A stranded task whose learner's data is deleted meanwhile gets no job: the recovery is a write for the learner, and
  is refused like any other once their rows are gone.

A stale worker that is still alive somewhere cannot interfere with the recovery: once its lease ran out, each of its
writes (task, events, artifacts, grades, learner memory, curricula) is refused at commit.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta

from app.curriculum.cycle import CycleNotFound
from app.runtime.orchestrator.orchestrator import TASK_LEASE
from app.schemas.job import JobStatus
from app.schemas.task import TaskStatus
from app.services.tasks import TaskService
from app.schemas.lease import LearnerDataGone
from app.teaching.errors import TeachingSessionNotFound
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
    shared_state: object = None  # what the shared-state step deleted (app/storage/shared_state.py, SharedStateSweep)


class TaskSweeper:
    def __init__(self, tasks: SqlTaskRepository, jobs: SqlTaskJobRepository, service: TaskService, *,
                 idle: timedelta = TASK_LEASE, batch: int = 100,
                 purge_objects: Callable[[], ObjectPurge] | None = None,
                 shared_state: Callable[[], object] | None = None) -> None:
        self._tasks = tasks
        self._jobs = jobs
        self._service = service
        self._idle = idle
        self._batch = batch
        self._purge_objects = purge_objects
        self._shared_state = shared_state

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
        if self._shared_state is not None:
            report.shared_state = self._shared_state()
        return report


# --- the learning sweeper (issue #66 PR 3) ---------------------------------------------------------------------------

LearnerRefresh = Callable[[str], Awaitable[object]]  # learner id -> its curricula refreshed and stored
IdBatch = Callable[[str | None, int], list[str]]  # (after, limit) -> ids in id order


@dataclass
class _Pass:
    """One kind of sweep: which ids to look at (in id order, from a cursor) and what to do with each one."""

    name: str
    ids: IdBatch
    act: Callable[[str], Awaitable[object]]
    cursor: str | None = None
    log_ids: bool = True  # False for learner ids: a log line never names a learner (as the deletion's does not)


@dataclass
class LearningSweepReport:
    """Counts of one learning-sweeper pass, by kind ("curricula", "cycles", "outboxes")."""

    done: dict[str, int] = field(default_factory=dict)  # ids handled (refreshed, reconciled, published)
    failed: dict[str, int] = field(default_factory=dict)  # ids whose action raised (logged; retried next round)
    wrapped: dict[str, bool] = field(default_factory=dict)  # this pass reached the end of the ids: the next starts over

    @property
    def errors(self) -> int:
        return sum(self.failed.values())

    def summary(self) -> str:
        return "; ".join(f"{k}: {self.done.get(k, 0)} done, {self.failed.get(k, 0)} failed" for k in self.done)


class LearningSweeper:
    """The writes no read makes any more (issue #66 PR 3), recovered in bounded batches:

    - **curricula**: every learner with a curriculum gets its curricula refreshed and stored
      (progress, transitions once, completion with D11's `completed_at`, the curriculum status following the goal):
      what time alone changes (review due dates, mastery decay), and an evidence write's refresh a crash lost;
    - **cycles**: every active learning cycle stores what its children did (`reconcile`): a child a worker finished
      whose hook was lost; and a cycle of any status whose events a crash left unpublished has them published;
    - **outboxes**: every teaching session with unpublished outbox items has them published (events, artifacts, the
      completed session's integration into learner memory and curricula).

    Each kind takes at most `batch` ids per pass, round-robin from a cursor kept in this process, so a pass is bounded
    whatever the number of learners: each one is visited within ceil(N / batch) passes. `full` (the cron entry
    point, a process that runs one pass) walks every id, still `batch` at a time. Every action is idempotent (stable
    ids, version-checked or fenced writes, transitions reported once), so a repeated or concurrent pass on another
    process changes nothing more. One id's failure is logged and counted, and never stops the pass; a learner deleted
    meanwhile is skipped."""

    def __init__(self, passes: list[_Pass], *, batch: int = 100) -> None:
        if batch < 1:
            raise ValueError("batch must be at least 1")
        self._passes = passes
        self._batch = batch

    @classmethod
    def of(cls, *, learners: IdBatch, refresh: LearnerRefresh, cycles: IdBatch,
           reconcile: Callable[[str], Awaitable[object]], outboxes: IdBatch,
           publish: Callable[[str], Awaitable[object]], batch: int = 100) -> LearningSweeper:
        return cls([_Pass("curricula", learners, refresh, log_ids=False), _Pass("cycles", cycles, reconcile),
                    _Pass("outboxes", outboxes, publish)], batch=batch)

    async def sweep(self, *, full: bool = False) -> LearningSweepReport:
        report = LearningSweepReport()
        for p in self._passes:
            report.done[p.name] = report.failed[p.name] = 0
            if full:
                p.cursor = None  # every id, from the first
            while True:
                # One id past the batch tells whether more follow, so a pass that reaches the end wraps at once
                # (never an empty pass: every id within ceil(N / batch) passes).
                found = p.ids(p.cursor, self._batch + 1)
                ids, more = found[:self._batch], len(found) > self._batch
                for item in ids:
                    try:
                        await p.act(item)
                        report.done[p.name] += 1
                    except (NotFound, LearnerDataGone, CycleNotFound, TeachingSessionNotFound):  # deleted meanwhile
                        continue
                    except Exception as exc:  # noqa: BLE001 - logged; the next round tries again
                        report.failed[p.name] += 1
                        # The exception's type only: its message (and a traceback's) can name a learner.
                        logger.warning("sweeper: %s %s failed: %s", p.name, item if p.log_ids else "(a learner)",
                                       type(exc).__name__)
                p.cursor = ids[-1] if more else None
                report.wrapped[p.name] = p.cursor is None
                if not full or p.cursor is None:
                    break
        level = logging.WARNING if report.errors else logging.INFO
        logger.log(level, "sweeper: learning pass: %s", report.summary())
        return report
