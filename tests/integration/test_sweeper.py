"""The sweeper (MVP item 2): work a killed worker left mid-run is queued again and finishes exactly once.

A killed worker is a model call parked at a gate that never opens in time; its lease running out is the clock moved
forward (tests/integration/test_task_concurrency.py). A worker cancelled mid-run (its lease given back, the task left
RUNNING) is a cancelled asyncio task.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from app.schemas.job import JobKind, JobStatus
from app.schemas.lease import LeaseLost
from app.schemas.task import TaskStatus
from app.services.sweeper import TaskSweeper
from app.storage.repositories import SqlTaskJobRepository
from tests.conftest import LEARNER, add_demo_learner, answers_for
from tests.integration.test_task_concurrency import AHEAD, Clock, Gate, executions, open_container
from tests.integration.test_task_worker import worker


def sweeper(container, clock: Clock) -> TaskSweeper:
    return TaskSweeper(container.task_service._tasks, SqlTaskJobRepository(container._sessions, clock),
                       container.task_service, clock=clock)


async def parked(container, llm):
    """A lesson whose worker is inside a model call while it holds the task's lease: (task id, the parked run)."""
    lid = add_demo_learner(container)
    task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
    gate = Gate(llm)
    gate.arm()
    run = asyncio.create_task(container.task_service.submit_assessment(task.task_id, answers_for(task)))
    await gate.entered.wait()
    assert container.task_service.get(task.task_id).status == TaskStatus.RUNNING
    return task.task_id, run, gate


async def complete(container, w, task_id: str):
    task = container.task_service.get(task_id)
    while task.status == TaskStatus.WAITING:
        container.task_service.enqueue_answers(task_id, answers_for(task).model_dump(mode="json"))
        await w.run_once()
        task = container.task_service.get(task_id)
    return task


async def test_a_task_left_mid_run_by_a_killed_worker_is_recovered_and_completes_once(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        task_id, run, gate = await parked(container, llm)
        assert sweeper(container, clock).sweep().recovered == []  # its worker's lease is live: being executed
        clock.offset = AHEAD  # the worker was killed: its lease runs out and no job exists to reclaim
        report = sweeper(container, clock).sweep()
        assert report.recovered == [task_id]
        job = container.task_service.job(task_id)
        assert job.kind == JobKind.RECOVER and job.status == JobStatus.QUEUED
        assert sweeper(container, clock).sweep().recovered == []  # one job per task, however often it sweeps

        w = worker(container, clock)
        done = await w.run_once()
        assert done.job_id == job.job_id and done.status == JobStatus.DONE
        gate.open()  # the "killed" worker wakes up: every write it attempts is refused
        with pytest.raises(LeaseLost):
            await run
        final = await complete(container, w, task_id)
        assert final.status == TaskStatus.COMPLETED
        events = container.task_service.events(task_id)
        assert Counter(e.type for e in events)["task.completed"] == 1
        recovered = [e for e in events if e.type == "task.resumed" and e.data.get("from_status") == "RUNNING"]
        assert len(recovered) == 1  # the one recovery
        assert all(n == 1 for n in Counter(a.name for a in container.task_service.artifacts(task_id)).values())
        profile = container.memory.get(LEARNER["learner_id"])
        assert len([x for x in profile.lessons if x.task_id == task_id]) == 1
    finally:
        container.close()


async def test_a_task_whose_worker_stopped_without_a_lease_is_swept_once_idle(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        task_id, run, _ = await parked(container, llm)
        run.cancel()  # the request was cancelled mid-run: the lease is given back, the task stays RUNNING
        with pytest.raises(asyncio.CancelledError):
            await run
        assert container.task_service._tasks.lease(task_id) is None
        assert container.task_service.get(task_id).status == TaskStatus.RUNNING
        assert sweeper(container, clock).sweep().recovered == []  # it may still be picked up by its caller
        clock.offset = AHEAD
        assert sweeper(container, clock).sweep().recovered == [task_id]
        runs = executions(container)
        await worker(container, clock).run_once()
        assert runs[task_id] == 1
        assert container.task_service.get(task_id).status in (TaskStatus.WAITING, TaskStatus.COMPLETED)
    finally:
        container.close()


async def test_the_sweeper_leaves_alone_work_that_is_not_stranded(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        waiting = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid,
                                                              user_id="u1")
        queued, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        task_id, run, gate = await parked(container, llm)
        clock.offset = AHEAD
        container.task_service.enqueue_resume(task_id)  # someone already asked for it to continue
        report = sweeper(container, clock).sweep()
        assert report.recovered == [] and report.left_failed == []
        assert waiting.status == TaskStatus.WAITING and container.task_service.get(queued.task_id).status \
            == TaskStatus.CREATED
        assert [j.kind for j in container.task_service.jobs(task_id)] == [JobKind.RESUME]
        gate.open()
        with pytest.raises(LeaseLost):
            await run
    finally:
        container.close()


async def test_a_task_whose_job_already_failed_is_left_for_a_manual_resume(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        task_id, run, gate = await parked(container, llm)
        clock.offset = AHEAD
        container.task_service.enqueue_resume(task_id)
        jobs = SqlTaskJobRepository(container._sessions, clock)
        job = jobs.claim("w", AHEAD)
        jobs.finish(job.job_id, "w", JobStatus.FAILED)  # e.g. it failed on every attempt
        report = sweeper(container, clock).sweep()
        assert report.recovered == [] and report.left_failed == [task_id]
        assert len(container.task_service.jobs(task_id)) == 1
        gate.open()
        with pytest.raises(LeaseLost):
            await run
    finally:
        container.close()


async def test_a_recovery_job_for_a_task_that_moved_on_runs_nothing(tmp_path) -> None:
    clock = Clock()
    container, _ = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        job, queued = container.task_service.enqueue_recovery(task.task_id)  # it reached WAITING meanwhile
        assert queued
        runs = executions(container)
        done = await worker(container, clock).run_once()
        assert done.job_id == job.job_id and done.status == JobStatus.DONE and done.task_status == "WAITING"
        assert runs[task.task_id] == 0
    finally:
        container.close()


async def test_the_worker_loop_sweeps_while_it_serves(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        task_id, run, gate = await parked(container, llm)
        clock.offset = AHEAD
        w = worker(container, clock, sweeper=sweeper(container, clock), poll_seconds=0.01, sweep_seconds=3600)
        stop = asyncio.Event()
        loop = asyncio.create_task(w.run_forever(stop))
        while (job := container.task_service.job(task_id)) is None or job.status != JobStatus.DONE:
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(loop, 30)
        assert job.kind == JobKind.RECOVER
        assert container.task_service.get(task_id).status in (TaskStatus.WAITING, TaskStatus.COMPLETED)
        gate.open()
        with pytest.raises(LeaseLost):
            await run
    finally:
        container.close()


def test_the_container_worker_has_a_sweeper(tmp_path) -> None:
    container, _ = open_container(tmp_path / "data")
    try:
        report = container.task_worker.sweep()
        assert report is not None and report.recovered == [] and report.failed_jobs == 0
    finally:
        container.close()
