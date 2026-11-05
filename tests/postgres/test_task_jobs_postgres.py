"""The task job queue on PostgreSQL (MVP item 1), opt-in like the spend tests: skipped unless TA_TEST_POSTGRES_URL
names a database the test may write to.

SQLite serializes writers, so it cannot show that two workers racing on the same rows stay correct. Here row locks
decide: concurrent enqueues of one task insert one job (the partial unique index), concurrent claims hand each job to
exactly one worker (the conditional UPDATE), a reclaimed job's old worker cannot record an outcome, and a lesson runs
end to end through the queue with every artifact stored once. Every run uses fresh ids, so the database can be reused.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
from collections import Counter
from datetime import timedelta

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.common import new_id, utcnow
from app.schemas.job import JobKind, JobStatus, TaskJob
from app.schemas.task import Task, TaskStatus
from app.services.container import build_container
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlTaskJobRepository, SqlTaskRepository
from tests.conftest import LEARNER, add_demo_learner, answers_for

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL job queue tests")
THREADS = 8


class Clock:
    def __init__(self) -> None:
        self.offset = timedelta(0)

    def __call__(self):
        return utcnow() + self.offset


@pytest.fixture
def repos():
    sessions = create_db(URL)
    yield SqlTaskJobRepository(sessions), SqlTaskRepository(sessions), sessions
    dispose(sessions)


def _task(tasks: SqlTaskRepository) -> str:
    task = Task(task_id=new_id("pgtask"), user_id="u", learner_id="l", request="r")
    tasks.create(task)
    return task.task_id


def _job(task_id: str, key: str = "RUN:x") -> TaskJob:
    return TaskJob(job_id=new_id("pgjob"), task_id=task_id, kind=JobKind.RUN, dedupe_key=key)


def test_concurrent_enqueues_of_one_task_insert_one_job(repos) -> None:
    jobs, tasks, _ = repos
    task_id = _task(tasks)
    barrier = threading.Barrier(THREADS)
    results: list[tuple[str, bool]] = []

    def enqueue() -> None:
        barrier.wait()
        job, inserted = jobs.enqueue(_job(task_id))
        results.append((job.job_id, inserted))

    threads = [threading.Thread(target=enqueue) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == THREADS
    assert sum(inserted for _, inserted in results) == 1
    assert len({job_id for job_id, _ in results}) == 1  # every caller got the one active job
    assert len(jobs.for_task(task_id)) == 1


def test_concurrent_claims_hand_each_job_to_one_worker(repos) -> None:
    jobs, tasks, _ = repos
    mine = {jobs.enqueue(_job(_task(tasks)))[0].job_id for _ in range(THREADS * 2)}
    barrier = threading.Barrier(THREADS)
    claimed: list[str] = []
    lock = threading.Lock()

    def claim_all() -> None:
        owner = new_id("w")
        barrier.wait()
        while True:
            job = jobs.claim(owner, timedelta(minutes=5))
            if job is None:
                return
            with lock:
                claimed.append(job.job_id)
            assert jobs.finish(job.job_id, owner, JobStatus.DONE)

    threads = [threading.Thread(target=claim_all) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    ours = [j for j in claimed if j in mine]
    assert Counter(ours) == Counter(mine)  # each of this run's jobs claimed exactly once
    assert all(jobs.get(j).attempts == 1 for j in mine)


def _claim_in_process(url: str, out) -> None:
    sessions = create_db(url)
    try:
        jobs = SqlTaskJobRepository(sessions)
        owner = new_id("proc")
        got = []
        while (job := jobs.claim(owner, timedelta(minutes=5))) is not None:
            got.append(job.job_id)
            jobs.finish(job.job_id, owner, JobStatus.DONE)
        out.put(got)
    finally:
        dispose(sessions)


def test_concurrent_claims_across_processes(repos) -> None:
    jobs, tasks, _ = repos
    mine = {jobs.enqueue(_job(_task(tasks)))[0].job_id for _ in range(12)}
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_claim_in_process, args=(URL, out)) for _ in range(4)]
    for p in procs:
        p.start()
    claimed = [j for _ in procs for j in out.get(timeout=120)]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    ours = [j for j in claimed if j in mine]
    assert Counter(ours) == Counter(mine)


def test_an_expired_claim_is_reclaimed_and_the_old_worker_is_fenced_out(repos) -> None:
    _, tasks, sessions = repos
    clock = Clock()
    jobs = SqlTaskJobRepository(sessions, clock)
    task_id = _task(tasks)
    job, _ = jobs.enqueue(_job(task_id))
    while (first := jobs.claim("dead", timedelta(minutes=5))) is not None and first.job_id != job.job_id:
        jobs.finish(first.job_id, "dead", JobStatus.DONE)  # other runs' leftovers
    assert first is not None and first.job_id == job.job_id
    assert jobs.get(job.job_id).lease_owner == "dead"
    clock.offset = timedelta(minutes=6)
    taken = None
    while (candidate := jobs.claim("alive", timedelta(minutes=5))) is not None:
        if candidate.job_id == job.job_id:
            taken = candidate
            break
        jobs.finish(candidate.job_id, "alive", JobStatus.DONE)
    assert taken is not None and taken.attempts == 2 and taken.lease_owner == "alive"
    assert not jobs.renew(job.job_id, "dead", timedelta(minutes=5))
    assert not jobs.finish(job.job_id, "dead", JobStatus.FAILED)
    assert not jobs.retry(job.job_id, "dead", delay=timedelta(0))
    assert jobs.finish(job.job_id, "alive", JobStatus.DONE, task_status=TaskStatus.WAITING)
    assert jobs.get(job.job_id).status == JobStatus.DONE
    # The task's next job can be queued once this one is no longer active.
    _, inserted = jobs.enqueue(_job(task_id, "RESUME:y"))
    assert inserted
    assert len(jobs.cancel_queued(task_id)) == 1  # leave nothing queued for later runs


def test_a_lesson_runs_end_to_end_through_the_queue_on_postgres(tmp_path) -> None:
    llm = MockLLMProvider(default_responders())
    c = build_container(Settings(data_dir=tmp_path / "data", database_url=URL, log_json=False),
                        llm_providers={"mock": llm})
    try:
        lid = add_demo_learner(c, f"pg-learner-{new_id('l')}")
        task, job = c.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1",
                                                idempotency_key=new_id("key"))
        again, _ = c.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1",
                                                  idempotency_key=None)
        assert again.task_id != task.task_id  # no key: another task (left queued, run by the drain below)
        c.task_service.cancel(again.task_id)

        async def run() -> Task:
            await c.task_worker.drain()
            current = c.task_service.get(task.task_id)
            while current.status == TaskStatus.WAITING:
                c.task_service.enqueue_answers(current.task_id, answers_for(current).model_dump(mode="json"))
                await asyncio.gather(c.task_worker.drain(), c.task_worker.drain())  # two loops, one execution
                current = c.task_service.get(task.task_id)
            return current

        done = asyncio.run(run())
        assert done.status == TaskStatus.COMPLETED, done.errors
        names = Counter(a.name for a in c.task_service.artifacts(task.task_id))
        assert names and all(n == 1 for n in names.values())
        events = Counter(e.type for e in c.task_service.events(task.task_id))
        assert events["task.completed"] == 1 and events["task.started"] == 1
        task_jobs = c.task_service.jobs(task.task_id)
        assert [j.kind for j in task_jobs] == [JobKind.RUN, JobKind.SUBMIT_INPUT, JobKind.SUBMIT_INPUT]
        assert all(j.status == JobStatus.DONE and j.attempts == 1 for j in task_jobs)
        assert c.task_service.job(again.task_id).status == JobStatus.CANCELLED
    finally:
        c.close()
