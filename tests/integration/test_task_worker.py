"""The background task worker (MVP item 1): the API queues task work and answers at once; the worker runs it.

What is checked is what ran (workflow executions, model calls) and what was stored (task, job, events, artifacts),
not only what a request returned. Time is a clock moved forward, never a sleep; a slow or dead worker is a model call
parked at a gate (tests/integration/test_task_concurrency.py).
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import timedelta

import pytest

from app.schemas.job import JOB_ERROR_MESSAGES, JobErrorCategory, JobKind, JobStatus
from app.schemas.task import TaskStatus
from app.services.worker import TaskWorker
from app.storage.repositories import SqlTaskJobRepository
from tests.auth import client_for, drain, settle
from tests.conftest import ANSWERS, LEARNER, add_demo_learner, answers_for
from tests.integration.test_task_concurrency import (
    AHEAD,
    Clock,
    Gate,
    event_types,
    executions,
    footprint,
    open_container,
    sequential_lesson,
)

LID = LEARNER["learner_id"]


def answer(task: dict) -> list[dict]:
    sheet = task["waiting"]["prompt"]
    key = ANSWERS["rounds"][sheet["round_number"] - 1]
    return [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]


def worker(container, clock=None, **kw) -> TaskWorker:
    """A second worker on the same queue (its own id), with no heartbeat unless asked: expiry is the clock's."""
    jobs = SqlTaskJobRepository(container._sessions, clock or Clock())
    kw.setdefault("heartbeat_seconds", 0)
    return TaskWorker(jobs, container.task_service, clock=clock or Clock(), **kw)


@pytest.fixture
def env(tmp_path):
    container, llm = open_container(tmp_path / "data")
    add_demo_learner(container)
    with client_for(container, LID, user_id="u1") as client:
        yield container, llm, client
    container.close()


# --- the API only queues --------------------------------------------------------------------------------------------


def test_create_returns_202_without_running_anything(env) -> None:
    container, llm, client = env
    runs = executions(container)
    created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID})
    assert created.status_code == 202, created.text
    body = created.json()
    assert created.headers["location"] == f"/tasks/{body['task_id']}"
    assert body["status"] == "CREATED" and body["job"]["status"] == "QUEUED" and body["job"]["kind"] == "RUN"
    assert sum(llm.calls.values()) == 0 and not runs  # the request ran no workflow and called no model
    polled = client.get(f"/tasks/{body['task_id']}").json()
    assert polled["status"] == "CREATED" and polled["job"]["job_id"] == body["job"]["job_id"]

    task = settle(client, created)
    assert task["status"] == "WAITING" and task["waiting"]["kind"] == "diagnostic_answers"
    assert task["job"]["status"] == "DONE" and task["job"]["attempts"] == 1 and task["job"]["message"] is None
    assert "payload" not in task["job"] and "lease_owner" not in task["job"]


def test_whole_lesson_through_the_queue_matches_an_inline_run(env, tmp_path) -> None:
    container, llm, client = env
    baseline = asyncio.run(sequential_lesson(tmp_path))
    task = settle(client, client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}))
    while task["status"] == "WAITING":
        task = settle(client, client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answer(task)}))
    assert task["status"] == "COMPLETED" and task["result"]["review_verdict"] == "APPROVED"
    assert footprint(container, llm, task["task_id"]) == baseline  # same calls, artifacts and events
    jobs = container.task_service.jobs(task["task_id"])
    assert [j.kind for j in jobs] == [JobKind.RUN, JobKind.SUBMIT_INPUT, JobKind.SUBMIT_INPUT]
    assert all(j.status == JobStatus.DONE for j in jobs)

    evaluation = settle(client, client.post(f"/tasks/{task['task_id']}/evaluation"))
    assert evaluation["status"] == "WAITING" and evaluation["waiting"]["kind"] == "assessment_answers"


def test_bad_answers_are_refused_before_anything_is_queued(env) -> None:
    container, _, client = env
    task = settle(client, client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}))
    while task["status"] == "WAITING":
        task = settle(client, client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answer(task)}))
    evaluation = settle(client, client.post(f"/tasks/{task['task_id']}/evaluation"))
    bad = client.post(f"/tasks/{evaluation['task_id']}/answers",
                      json={"answers": [{"question_id": "x", "answer": "y"}]})
    assert bad.status_code == 422  # validated in the request, as before
    assert len(container.task_service.jobs(evaluation["task_id"])) == 1  # only its RUN: nothing was queued
    # A task that is not waiting refuses answers with 409, as before.
    stale = {"answers": [{"question_id": "r1_x", "answer": "y"}]}
    assert client.post(f"/tasks/{task['task_id']}/answers", json=stale).status_code == 409


def test_queued_answers_are_one_job_until_they_run(env) -> None:
    container, _, client = env
    task = settle(client, client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}))
    done = client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answer(task)})
    assert done.status_code == 202
    assert client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answer(task)}).json()["job"]["job_id"] \
        == done.json()["job"]["job_id"]  # still queued: the same answers are the same job
    drain(container)
    after = client.get(f"/tasks/{task['task_id']}").json()
    assert after["status"] == "WAITING" and after["waiting"]["prompt"]["round_number"] == 2
    assert client.post(f"/tasks/{task['task_id']}/resume").status_code == 409  # waiting on the learner


# --- no duplicate execution --------------------------------------------------------------------------------------


def test_duplicate_submissions_share_one_job_and_one_execution(env) -> None:
    container, llm, client = env
    task = settle(client, client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}))
    runs = executions(container)
    first = client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answer(task)})
    again = client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answer(task)})
    other = [dict(a, answer="something else") for a in answer(task)]
    conflict = client.post(f"/tasks/{task['task_id']}/answers", json={"answers": other})
    assert first.status_code == again.status_code == 202
    assert first.json()["job"]["job_id"] == again.json()["job"]["job_id"]
    assert conflict.status_code == 409 and conflict.json()["error"] == "JobConflict"
    assert client.post(f"/tasks/{task['task_id']}/resume").status_code == 409  # still WAITING, not resumable
    jobs = drain(container)
    assert len(jobs) == 1 and runs == {task["task_id"]: 1}
    after = client.get(f"/tasks/{task['task_id']}").json()
    assert after["waiting"]["prompt"]["round_number"] == 2 and after["job"]["job_id"] == first.json()["job"]["job_id"]
    assert drain(container) == [] and runs == {task["task_id"]: 1}
    assert event_types(container, task["task_id"])["task.started"] == 1


def test_idempotency_key_returns_the_same_task(env) -> None:
    container, _, client = env
    body = {"request": LEARNER["request"], "learner_id": LID}
    one = client.post("/tasks", json=body, headers={"Idempotency-Key": "k-1"})
    two = client.post("/tasks", json=body, headers={"Idempotency-Key": "k-1"})
    assert one.status_code == two.status_code == 202
    assert one.json()["task_id"] == two.json()["task_id"] and one.json()["job"]["job_id"] == two.json()["job"]["job_id"]
    other = client.post("/tasks", json=dict(body, request="Something else"), headers={"Idempotency-Key": "k-1"})
    assert other.status_code == 409
    assert len(drain(container)) == 1
    three = client.post("/tasks", json=body, headers={"Idempotency-Key": "k-1"})  # after it ran: the same, not rerun
    assert three.json()["task_id"] == one.json()["task_id"] and three.json()["status"] == "WAITING"
    assert drain(container) == []
    fresh = client.post("/tasks", json=body)  # no key: a new task
    assert fresh.json()["task_id"] != one.json()["task_id"]
    # Another user's identical key is another task.
    with client_for(container, LID, user_id="u2") as other_user:
        theirs = other_user.post("/tasks", json=body, headers={"Idempotency-Key": "k-1"})
    assert theirs.status_code == 202 and theirs.json()["task_id"] != one.json()["task_id"]


async def test_two_workers_never_run_the_same_job(tmp_path) -> None:
    container, llm = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        runs = executions(container)
        task, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(llm)
        gate.arm()
        a, b = worker(container), worker(container)
        first = asyncio.create_task(a.run_once())
        await gate.entered.wait()  # A holds the job (and the task) inside its first model call
        assert await b.run_once() is None  # nothing claimable for B
        gate.open()
        job = await first
        assert job.status == JobStatus.DONE and job.attempts == 1 and runs == {task.task_id: 1}
        assert container.task_service.get(task.task_id).status == TaskStatus.WAITING
    finally:
        container.close()


# --- failure handling ------------------------------------------------------------------------------------------


async def test_an_unexpected_error_is_retried_with_backoff_then_succeeds(tmp_path) -> None:
    clock = Clock()
    container, _ = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task, job = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        w = worker(container, clock, retry_backoff_seconds=30)
        original, failures = container.task_service.run, []

        async def flaky(task_id):
            if not failures:
                failures.append(task_id)
                raise RuntimeError("database went away: secret=sk-test-123")
            return await original(task_id)

        container.task_service.run = flaky
        retried = await w.run_once()
        assert retried.status == JobStatus.QUEUED and retried.attempts == 1 and retried.lease_owner is None
        assert retried.available_at > clock() and "sk-test-123" not in (retried.error or "")
        assert await w.run_once() is None  # backing off
        clock.offset += timedelta(seconds=31)
        done = await w.run_once()
        assert done.status == JobStatus.DONE and done.attempts == 2
        assert container.task_service.get(task.task_id).status == TaskStatus.WAITING
    finally:
        container.close()


async def test_a_job_that_keeps_failing_ends_failed_with_a_fixed_message(tmp_path) -> None:
    clock = Clock()
    container, _ = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        w = worker(container, clock, retry_backoff_seconds=1)

        async def broken(task_id):
            raise RuntimeError("internal detail expected_answer=fue")

        container.task_service.run = broken
        for _ in range(3):
            job = await w.run_once()
            clock.offset += timedelta(seconds=10)
        assert job.status == JobStatus.FAILED and job.attempts == 3
        assert job.error_category == JobErrorCategory.WORKER_ERROR
        with client_for(container, lid, user_id="u1") as client:
            view = client.get(f"/tasks/{task.task_id}").json()
        assert view["job"]["status"] == "FAILED"
        assert view["job"]["message"] == JOB_ERROR_MESSAGES[JobErrorCategory.WORKER_ERROR]
        assert "expected_answer" not in json.dumps(view)
        assert await w.run_once() is None  # not retried any more
        # The task stays at its checkpoint and can be resumed through the queue.
        with client_for(container, lid, user_id="u1") as client:
            resumed = client.post(f"/tasks/{task.task_id}/resume")
            assert resumed.status_code == 202 and resumed.json()["job"]["kind"] == "RESUME"
    finally:
        container.close()


def test_a_cancelled_task_never_runs_its_queued_job(env) -> None:
    container, llm, client = env
    created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}).json()
    cancelled = client.post(f"/tasks/{created['task_id']}/cancel").json()
    assert cancelled["status"] == "CANCELLED" and cancelled["job"]["status"] == "CANCELLED"
    assert drain(container) == [] and sum(llm.calls.values()) == 0


def test_a_task_paused_before_its_worker_starts_stays_paused_until_resumed(env) -> None:
    container, llm, client = env
    created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}).json()
    assert client.post(f"/tasks/{created['task_id']}/pause").json()["status"] == "PAUSED"
    jobs = drain(container)
    assert [j.status for j in jobs] == [JobStatus.DONE] and sum(llm.calls.values()) == 0
    assert client.get(f"/tasks/{created['task_id']}").json()["status"] == "PAUSED"
    task = settle(client, client.post(f"/tasks/{created['task_id']}/resume"))
    assert task["status"] == "WAITING"


# --- recovery ------------------------------------------------------------------------------------------------------


async def test_a_dead_workers_job_is_reclaimed_and_continues_from_its_checkpoint(tmp_path) -> None:
    """Worker A dies inside a model call (it never returns): its job and task leases run out, worker B claims the
    job as a second attempt and finishes the task from the checkpoint A left. When A's call finally returns, every
    write it attempts is refused, and the lesson is exactly the uncontended one."""
    baseline = await sequential_lesson(tmp_path)
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        await worker(container, clock).run_once()
        task = container.task_service.get(task.task_id)
        assert task.status == TaskStatus.WAITING
        container.task_service.enqueue_answers(task.task_id, answers_for(task).model_dump(mode="json"))
        gate = Gate(llm)
        gate.arm()
        a = worker(container, clock)
        dying = asyncio.create_task(a.run_once())
        await gate.entered.wait()  # A applied the answers and is inside a model call
        b = worker(container, clock)
        assert await b.run_once() is None  # A's claim is live
        clock.offset += AHEAD  # A hangs past both leases
        reclaimed = await b.run_once()
        assert reclaimed.status == JobStatus.DONE and reclaimed.attempts == 2
        gate.open()
        await dying  # A wakes up: its writes are refused and it records nothing over B's outcome
        job = container.task_service.job(task.task_id)
        assert job.job_id == reclaimed.job_id and job.status == JobStatus.DONE and job.attempts == 2
        task = container.task_service.get(task.task_id)
        while task.status == TaskStatus.WAITING:
            container.task_service.enqueue_answers(task.task_id, answers_for(task).model_dump(mode="json"))
            await b.run_once()
            task = container.task_service.get(task.task_id)
        assert task.status == TaskStatus.COMPLETED
        result = footprint(container, llm, task.task_id)
        assert result["artifacts"] == baseline["artifacts"]  # every artifact stored once
        assert all(n == 1 for n in Counter(a.name for a in container.task_service.artifacts(task.task_id)).values())
        assert result["task_events"]["task.completed"] == 1
    finally:
        container.close()


async def test_answers_already_applied_by_a_dead_worker_are_not_applied_again(tmp_path) -> None:
    """A stored the answers (the task left WAITING) and died before finishing its job. The reclaimed job continues
    the task instead of submitting the same answers a second time."""
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        w = worker(container, clock)
        await w.run_once()
        task = container.task_service.get(task.task_id)
        _, job = container.task_service.enqueue_answers(task.task_id, answers_for(task).model_dump(mode="json"))
        jobs = SqlTaskJobRepository(container._sessions, clock)
        assert jobs.claim("dead-worker", timedelta(minutes=5)).job_id == job.job_id
        await container.task_service.submit_answers(task.task_id, job.payload["answers"])  # the dead worker's work
        after_first = container.task_service.get(task.task_id)
        assert after_first.status == TaskStatus.WAITING and after_first.waiting.prompt["round_number"] == 2
        submitted = event_types(container, task.task_id)["task.resumed"]
        clock.offset += AHEAD
        done = await w.run_once()
        assert done.job_id == job.job_id and done.status == JobStatus.DONE and done.attempts == 2
        assert event_types(container, task.task_id)["task.resumed"] == submitted  # not submitted twice
        assert container.task_service.get(task.task_id).version == after_first.version
    finally:
        container.close()


async def test_a_busy_task_is_retried_without_using_an_attempt(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        lease = container.orchestrator._tasks.claim(task.task_id, "inline-caller", timedelta(minutes=5))
        w = worker(container, clock, busy_retry_seconds=10)
        busy = await w.run_once()
        assert busy.status == JobStatus.QUEUED and busy.attempts == 0 and sum(llm.calls.values()) == 0
        container.orchestrator._tasks.release(lease)
        clock.offset += timedelta(seconds=11)
        done = await w.run_once()
        assert done.status == JobStatus.DONE and done.attempts == 1
    finally:
        container.close()


async def test_the_worker_loop_stops_cleanly(tmp_path) -> None:
    container, _ = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        task, _ = container.task_service.enqueue_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        stop = asyncio.Event()
        loop = asyncio.create_task(worker(container, poll_seconds=0.01).run_forever(stop))
        for _ in range(500):
            if container.task_service.job(task.task_id).status == JobStatus.DONE:
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(loop, timeout=10)
        assert container.task_service.get(task.task_id).status == TaskStatus.WAITING
    finally:
        container.close()


def test_an_embedded_worker_runs_queued_tasks_inside_the_api_process(tmp_path) -> None:
    """TA_WORKER_EMBEDDED=true (development): the API's lifespan runs a worker loop; the client only polls."""
    import time

    from app.config.settings import Settings
    from app.providers.llm.mock import MockLLMProvider
    from app.providers.llm.mock_responders import default_responders
    from app.services.container import build_container

    settings = Settings(data_dir=tmp_path / "data", log_json=False, worker_embedded=True, worker_poll_seconds=0.05)
    container = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    try:
        add_demo_learner(container)
        with client_for(container, LID, user_id="u1") as client:
            created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID})
            assert created.status_code == 202
            deadline = time.monotonic() + 60
            while (polled := client.get(f"/tasks/{created.json()['task_id']}").json())["job"]["status"] != "DONE":
                assert time.monotonic() < deadline, polled
                time.sleep(0.05)
            assert polled["status"] == "WAITING"
    finally:
        container.close()
