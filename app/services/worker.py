"""The task worker: claims queued task jobs and runs them through the orchestrator, outside any HTTP request.

    python -m app.services.worker            # run until interrupted (SIGINT / SIGTERM)
    python -m app.services.worker --drain    # run every job that is available now, then exit
    python -m app.services.worker --sweep    # queue the continuation of work killed workers left mid-run, then exit

Guarantees, layered on the orchestrator's (see docs/architecture.md, background worker):
- One worker per job: a claim is a conditional UPDATE of the job row under a job lease, renewed by a heartbeat while
  the job runs. One worker per task: the orchestrator still claims the task's own lease and fences every task,
  event and artifact write, so a job reclaimed from a hung worker cannot commit twice.
- Recovery: a job whose worker died is claimed again once its lease runs out and continues the task from its last
  checkpoint (completed nodes and stored artifacts are not produced again). A job whose worker died on its last allowed
  attempt is FAILED instead of claimed again. Answers whose submission already reached
  the task are never applied twice: the job carries the task version it was validated against. A task left mid-run
  with no job at all (run in-process by a killed process) is found by the sweeper (app/services/sweeper.py), which
  this worker runs between jobs, and continued through a RECOVER job.
- Failure handling: an unexpected error is retried with exponential backoff up to the job's `max_attempts`, then the
  job is FAILED (the task stays at its checkpoint and can be resumed). A busy task (another worker's live lease) is
  retried later without using an attempt. Work that no longer applies (the task moved on) fails at once.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Callable
from datetime import datetime, timedelta

from app.observability.redaction import redact_text
from app.runtime.orchestrator.orchestrator import TASK_LEASE, InvalidInput, TaskBusy
from app.runtime.tasks.state_machine import InvalidTransition
from app.schemas.common import new_id, utcnow
from app.schemas.job import JobErrorCategory, JobKind, JobStatus, TaskJob
from app.schemas.task import Task, TaskStatus
from app.services.sweeper import STRANDED, TaskSweeper
from app.services.tasks import TaskService
from app.storage.repositories import NotFound, SqlTaskJobRepository

logger = logging.getLogger(__name__)

# How long a worker's claim on a job lasts without renewal (renewed every third of it while the job runs).
JOB_LEASE = TASK_LEASE
# Executable states of a task a RUN job may still execute; any other state means the work already happened or was
# stopped (paused, cancelled) and the job ends without running anything.
RUNNABLE = frozenset({TaskStatus.CREATED, TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.REVIEWING})


class _Refused(Exception):
    """The job's work no longer applies to its task: fail it now, retrying cannot help."""

    def __init__(self, category: JobErrorCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


class TaskWorker:
    def __init__(self, jobs: SqlTaskJobRepository, tasks: TaskService, *, lease_ttl: timedelta = JOB_LEASE,
                 heartbeat_seconds: float | None = None, poll_seconds: float = 1.0,
                 retry_backoff_seconds: float = 5.0, retry_max_backoff_seconds: float = 300.0,
                 busy_retry_seconds: float = 10.0, clock: Callable[[], datetime] = utcnow,
                 worker_id: str | None = None, sweeper: TaskSweeper | None = None,
                 sweep_seconds: float = 60.0) -> None:
        self._jobs = jobs
        self._tasks = tasks
        self._ttl = lease_ttl
        self._heartbeat = heartbeat_seconds if heartbeat_seconds is not None else lease_ttl.total_seconds() / 3
        self._poll = poll_seconds
        self._backoff = retry_backoff_seconds
        self._max_backoff = retry_max_backoff_seconds
        self._busy = busy_retry_seconds
        self._clock = clock
        self.worker_id = worker_id or new_id("worker")
        self._sweeper = sweeper
        self._sweep_seconds = sweep_seconds

    # --- loops ------------------------------------------------------------------------------------------------------

    async def run_once(self) -> TaskJob | None:
        """Claim and run one available job. Returns the job as stored afterwards, or None when none was available."""
        job = self._jobs.claim(self.worker_id, self._ttl)
        if job is None:
            return None
        await self._run(job)
        return self._jobs.get(job.job_id)

    async def drain(self, max_jobs: int = 1000) -> list[TaskJob]:
        """Run jobs until none is available now (jobs waiting out a backoff stay queued)."""
        done = []
        while len(done) < max_jobs:
            job = await self.run_once()
            if job is None:
                break
            done.append(job)
        return done

    def sweep(self):
        """One sweeper pass (app/services/sweeper.py), or None without a sweeper."""
        return self._sweeper.sweep() if self._sweeper is not None else None

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Poll for jobs until `stop` is set. A job that is running when `stop` is set finishes first. With a sweeper,
        a sweep pass runs at start and then every `sweep_seconds`, between jobs."""
        logger.info("task worker %s started", self.worker_id)
        next_sweep = 0.0
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            if self._sweeper is not None and self._sweep_seconds > 0 and loop.time() >= next_sweep:
                next_sweep = loop.time() + self._sweep_seconds
                try:
                    self._sweeper.sweep()
                except Exception:  # e.g. the database is down: the next pass tries again
                    logger.exception("task worker %s: sweeping stranded work failed", self.worker_id)
            try:
                job = await self.run_once()
            except Exception:  # the queue itself failed (e.g. the database is down): wait and try again
                logger.exception("task worker %s: claiming a job failed", self.worker_id)
                job = None
            if job is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._poll)
                except asyncio.TimeoutError:
                    pass
        logger.info("task worker %s stopped", self.worker_id)

    # --- one job ----------------------------------------------------------------------------------------------------

    async def _run(self, job: TaskJob) -> None:
        beat = asyncio.create_task(self._renew(job)) if self._heartbeat > 0 else None
        try:
            await self._attempt(job)
        finally:
            if beat is not None:
                beat.cancel()

    async def _attempt(self, job: TaskJob) -> None:
        try:
            task = await self._execute(job)
        except TaskBusy:  # another worker (or an in-process caller) is executing the task: nothing ran
            self._jobs.retry(job.job_id, self.worker_id, delay=timedelta(seconds=self._busy), count_attempt=False)
            logger.info("job %s: task %s is busy, retrying later", job.job_id, job.task_id)
            return
        except _Refused as exc:
            self._finish(job, JobStatus.FAILED, error=str(exc), category=exc.category)
            return
        except NotFound as exc:
            self._finish(job, JobStatus.FAILED, error=f"task not found: {exc}", category=JobErrorCategory.INVALID_STATE)
            return
        except InvalidInput as exc:
            self._finish(job, JobStatus.FAILED, error=redact_text(str(exc))[:2000],
                         category=JobErrorCategory.INVALID_INPUT)
            return
        except InvalidTransition as exc:
            self._finish(job, JobStatus.FAILED, error=redact_text(str(exc))[:2000],
                         category=JobErrorCategory.INVALID_STATE)
            return
        except Exception as exc:  # LeaseLost included: the next attempt reads the task's current state
            self._failed_attempt(job, exc)
            return
        self._finish(job, JobStatus.CANCELLED if task.status == TaskStatus.CANCELLED else JobStatus.DONE,
                     task_status=task.status)

    async def _execute(self, job: TaskJob) -> Task:
        task = self._tasks.get(job.task_id)
        if job.kind == JobKind.RUN:
            if task.status not in RUNNABLE:  # already ran, or was paused or cancelled meanwhile
                return task
            return await self._tasks.run(task.task_id)
        if job.kind == JobKind.RESUME:
            return await self._tasks.resume(task.task_id)
        if job.kind == JobKind.RECOVER:  # queued by the sweeper: continue only if the task is still mid-run
            if task.status not in STRANDED:
                return task
            return await self._tasks.resume(task.task_id)
        if job.kind == JobKind.SUBMIT_INPUT:
            return await self._submit(job, task)
        raise _Refused(JobErrorCategory.INVALID_STATE, f"unknown job kind {job.kind}")

    async def _submit(self, job: TaskJob, task: Task) -> Task:
        node_id = job.payload["node_id"]
        if task.status == TaskStatus.CANCELLED:
            return task
        unchanged = task.version == job.payload.get("task_version")
        if unchanged and task.status == TaskStatus.WAITING and task.waiting is not None \
                and task.waiting.node_id == node_id:
            return await self._tasks.submit_answers(task.task_id, job.payload["answers"])
        node = task.workflow.node_states.get(node_id) if task.workflow is not None else None
        if task.status == TaskStatus.WAITING and task.waiting is None and node is not None \
                and node.human_input is not None:
            # An earlier attempt stored these answers and died before continuing: continue from the checkpoint.
            return await self._tasks.run(task.task_id)
        if not unchanged:  # the task moved on since the answers were queued: they were applied (or superseded)
            if task.status in RUNNABLE:  # an earlier attempt applied them and stopped mid-run: continue the task
                return await self._tasks.run(task.task_id)
            return task
        raise _Refused(JobErrorCategory.INVALID_STATE, f"task {task.task_id} is not waiting for input at {node_id}")

    def _failed_attempt(self, job: TaskJob, exc: Exception) -> None:
        error = redact_text(f"{type(exc).__name__}: {exc}")[:2000]
        current = self._jobs.get(job.job_id)
        if current.attempts >= current.max_attempts:
            logger.error("job %s (task %s) failed on its last attempt %d: %s", job.job_id, job.task_id,
                         current.attempts, error)
            self._finish(job, JobStatus.FAILED, error=error, category=JobErrorCategory.WORKER_ERROR)
            return
        delay = min(self._max_backoff, self._backoff * 2 ** (current.attempts - 1))
        logger.warning("job %s (task %s) attempt %d failed, retrying in %.0fs: %s", job.job_id, job.task_id,
                       current.attempts, delay, error)
        self._jobs.retry(job.job_id, self.worker_id, delay=timedelta(seconds=delay), error=error)

    def _finish(self, job: TaskJob, status: JobStatus, *, task_status: TaskStatus | None = None,
                error: str | None = None, category: JobErrorCategory | None = None) -> None:
        if task_status is None:
            try:
                task_status = self._tasks.get(job.task_id).status
            except NotFound:
                task_status = None
        if not self._jobs.finish(job.job_id, self.worker_id, status, task_status=task_status, error=error,
                                 error_category=category):
            logger.warning("job %s was reclaimed by another worker; outcome %s not recorded", job.job_id,
                           status.value)

    async def _renew(self, job: TaskJob) -> None:
        while True:
            await asyncio.sleep(self._heartbeat)
            if not self._jobs.renew(job.job_id, self.worker_id, self._ttl):
                logger.warning("job %s: worker %s lost its claim", job.job_id, self.worker_id)
                return


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - process entry point
    from app.config.settings import Settings
    from app.observability.logging import configure_logging
    from app.services.container import build_container

    parser = argparse.ArgumentParser(description="Run the task worker: execute queued lesson and evaluation tasks.")
    parser.add_argument("--drain", action="store_true", help="run every available job, then exit")
    parser.add_argument("--sweep", action="store_true",
                        help="queue the continuation of work killed workers left mid-run (one pass), then exit")
    args = parser.parse_args(argv)
    from app.config.secrets import load_secret_files

    load_secret_files()
    settings = Settings()
    configure_logging(settings.log_level, settings.log_json)
    container = build_container(settings)
    try:
        if args.sweep:
            report = container.task_worker.sweep()
            print(f"failed jobs: {report.failed_jobs}; recovered tasks: {len(report.recovered)}; "
                  f"left failed: {len(report.left_failed)}")
            return 0
        if args.drain:
            asyncio.run(container.task_worker.drain())
            return 0
        stop = asyncio.Event()

        async def serve() -> None:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await container.task_worker.run_forever(stop)

        asyncio.run(serve())
        return 0
    finally:
        container.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
