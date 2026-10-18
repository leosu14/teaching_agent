"""Concurrent task execution (PR18): one worker per task at a time, and a stale worker commits nothing.

Concurrency here is deterministic: a gated provider call parks the first worker at a known point (`Gate`), other
requests run while it is parked, then the gate opens. Lease expiry is a clock moved forward (`Clock`), never a sleep.
Each test counts what actually ran (workflow executions, provider calls) and checks what was persisted (state, events,
artifacts), not just that requests returned.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import timedelta

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.runtime.orchestrator.orchestrator import TASK_LEASE, TaskBusy
from app.runtime.tasks.state_machine import InvalidTransition
from app.schemas.common import utcnow
from app.schemas.lease import LeaseLost
from app.schemas.task import TaskStatus
from app.services.container import Container, build_container
from tests.conftest import LEARNER, add_demo_learner, answers_for

AHEAD = TASK_LEASE + timedelta(seconds=1)


class Clock:
    """A process clock that tests move forward (a slow call outliving the lease, or another replica's clock)."""

    def __init__(self, offset: timedelta = timedelta(0)) -> None:
        self.offset = offset

    def __call__(self):
        return utcnow() + self.offset


class Gate:
    """Parks the next provider call (once armed) until `open()`: the slow model call of a live worker."""

    def __init__(self, llm: MockLLMProvider) -> None:
        self.entered = asyncio.Event()
        self._open = asyncio.Event()
        self._armed = False
        original = llm.generate

        async def generate(request):
            if self._armed:
                self._armed = False
                self.entered.set()
                await self._open.wait()
            return await original(request)

        llm.generate = generate

    def arm(self) -> None:
        self._armed = True

    def open(self) -> None:
        self._open.set()


def executions(container: Container) -> Counter:
    """How many times each task's workflow was actually executed (a claimed lease reaching the engine)."""
    counts: Counter = Counter()
    original = container.orchestrator._run

    async def run(task, lease):
        counts[task.task_id] += 1
        return await original(task, lease)

    container.orchestrator._run = run
    return counts


def open_container(data_dir, *, clock: Clock | None = None) -> tuple[Container, MockLLMProvider]:
    llm = MockLLMProvider(default_responders())
    container = build_container(Settings(data_dir=data_dir, log_json=False), llm_providers={"mock": llm},
                                clock=clock or Clock())
    return container, llm


def event_types(container: Container, task_id: str) -> Counter:
    return Counter(e.type for e in container.task_service.events(task_id))


def snapshot(container: Container, task_id: str) -> tuple:
    task = container.task_service.get(task_id)
    return (task.model_dump_json(), [e.event_id for e in container.task_service.events(task_id)],
            sorted(a.artifact_id for a in container.task_service.artifacts(task_id)))


async def finish(container: Container, task_id: str):
    task = container.task_service.get(task_id)
    while task.status == TaskStatus.WAITING:
        task = await container.task_service.submit_assessment(task_id, answers_for(task))
    return task


async def sequential_lesson(tmp_path) -> dict:
    """What one uncontended lesson costs and stores: the baseline every concurrent run must equal."""
    container, llm = open_container(tmp_path / "baseline")
    try:
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        task = await finish(container, task.task_id)
        return footprint(container, llm, task.task_id)
    finally:
        container.close()


def footprint(container: Container, llm: MockLLMProvider, task_id: str) -> dict:
    task = container.task_service.get(task_id)
    artifacts = container.task_service.artifacts(task_id)
    events = event_types(container, task_id)
    return {"status": task.status, "llm_calls": sum(llm.calls.values()), "llm_by_agent": dict(llm.calls),
            "provider_requests": len(task.cost.provider_requests),
            "artifacts": Counter(a.name for a in artifacts),
            "task_events": {t: n for t, n in events.items() if t.startswith("task.")},
            "node_finished": events["node.finished"]}


# --- simultaneous starts and resumes ------------------------------------------------------------------------------


async def test_simultaneous_resumes_execute_the_task_once(tmp_path) -> None:
    container, llm = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        runs = executions(container)
        gate = Gate(llm)
        task = container.orchestrator.create_task(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate.arm()
        first = asyncio.create_task(container.task_service.resume(task.task_id))
        await gate.entered.wait()  # the first resume is inside its first model call, holding the lease
        others = await asyncio.gather(container.task_service.resume(task.task_id),
                                      container.task_service.run(task.task_id),
                                      container.task_service.resume(task.task_id), return_exceptions=True)
        assert all(isinstance(r, TaskBusy) for r in others), others
        calls_while_parked = sum(llm.calls.values())
        gate.open()
        done = await first
        assert done.status == TaskStatus.WAITING
        assert runs == {task.task_id: 1}  # one execution of the logical action
        assert calls_while_parked == 0  # the refused requests made no model call
        events = event_types(container, task.task_id)
        assert events["task.started"] == 1 and events["task.waiting"] == 1 and events["task.planned"] == 1
        assert all(n == 1 for n in Counter(a.name for a in container.task_service.artifacts(task.task_id)).values())
        assert container.task_service.get(task.task_id).version == done.version
    finally:
        container.close()


async def test_answer_and_resume_racing_on_a_running_task_run_it_once(tmp_path) -> None:
    """The audit's probe 1, deterministic: while one request is executing the lesson after the learner's answers,
    `POST /tasks/{id}/resume` and a second submission arrive. Before PR18 the resume re-ran the running node
    (24 LLM calls for 13, two `task.completed`, 20 artifacts stored twice). Now both are refused and the lesson is
    exactly the uncontended one."""
    baseline = await sequential_lesson(tmp_path)
    container, llm = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(llm)
        refused = 0
        while task.status == TaskStatus.WAITING:
            answers = answers_for(task)
            gate.arm()
            submit = asyncio.create_task(container.task_service.submit_assessment(task.task_id, answers))
            await gate.entered.wait()
            gate.entered.clear()
            racing = await asyncio.gather(container.task_service.resume(task.task_id),
                                          container.task_service.submit_assessment(task.task_id, answers),
                                          return_exceptions=True)
            assert isinstance(racing[0], TaskBusy), racing  # the resume: the task is leased by the answering request
            assert isinstance(racing[1], InvalidTransition), racing  # the answer: it no longer waits for one
            refused += len(racing)
            gate.open()
            task = await submit
            gate = Gate(llm)
        assert refused >= 2 and task.status == TaskStatus.COMPLETED
        assert footprint(container, llm, task.task_id) == baseline
    finally:
        container.close()


# --- lease expiry and takeover -----------------------------------------------------------------------------------


async def test_a_worker_whose_lease_was_taken_over_commits_nothing(tmp_path) -> None:
    """Two processes on one database. A's model call outlives its lease (B's clock is past it); B claims the task and
    runs it. When A's call returns, every write A attempts (artifacts, events, checkpoint, result) is refused: B's
    newer state stands, A publishes no success, and the run ends exactly like an uncontended one."""
    baseline = await sequential_lesson(tmp_path)
    data = tmp_path / "data"
    a, a_llm = open_container(data)
    b, b_llm = open_container(data, clock=Clock(AHEAD))
    try:
        lid = add_demo_learner(a)
        task = await a.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(a_llm)
        gate.arm()
        stale = asyncio.create_task(a.task_service.submit_assessment(task.task_id, answers_for(task)))
        await gate.entered.wait()  # A holds the lease, parked in a slow model call
        a_fence = a.task_service._tasks.lease(task.task_id).fence

        newer = await b.task_service.resume(task.task_id)  # B: A's lease has expired by B's clock
        assert newer.status in (TaskStatus.WAITING, TaskStatus.COMPLETED)
        b_runs = sum(b_llm.calls.values())
        before = snapshot(a, task.task_id)

        gate.open()  # A's slow call returns after the takeover
        with pytest.raises(LeaseLost) as raised:
            await stale
        assert str(raised.value) == LeaseLost.MESSAGE

        after = snapshot(a, task.task_id)
        assert after[0] == before[0]  # B's state (and version) untouched
        assert after[2] == before[2]  # no artifact recorded by A
        new_events = [e for e in a.task_service.events(task.task_id) if e.event_id not in set(before[1])]
        assert [e.type for e in new_events] == ["task.lease_lost"]  # only the internal diagnostic
        assert new_events[0].data["stale_fence"] == a_fence
        assert all(a.artifacts.verify(x) is None for x in a.task_service.artifacts(task.task_id))

        final = await finish(b, task.task_id)
        result = footprint(b, b_llm, task.task_id)
        assert final.status == TaskStatus.COMPLETED
        assert result["task_events"]["task.completed"] == 1
        assert result["artifacts"] == baseline["artifacts"]  # no duplicate artifact versions
        assert result["task_events"] == {**baseline["task_events"],  # plus B's takeover resume and A's diagnostic
                                         "task.resumed": baseline["task_events"].get("task.resumed", 0) + 1,
                                         "task.lease_lost": 1}
        assert b_runs > 0  # B re-ran the interrupted node; A's in-flight call was its only wasted one
    finally:
        a.close()
        b.close()


async def test_expiry_alone_stops_a_slow_worker_and_resume_recovers(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(llm)
        gate.arm()
        slow = asyncio.create_task(container.task_service.submit_assessment(task.task_id, answers_for(task)))
        await gate.entered.wait()
        committed = snapshot(container, task.task_id)
        clock.offset = AHEAD  # nobody takes over; the lease just runs out during the call
        gate.open()
        with pytest.raises(LeaseLost):
            await slow
        after = snapshot(container, task.task_id)
        assert after[0] == committed[0] and after[2] == committed[2]
        assert container.task_service.get(task.task_id).status == TaskStatus.RUNNING  # its last checkpoint

        resumed = await container.task_service.resume(task.task_id)  # safe retry after the expired lease
        final = await finish(container, resumed.task_id)
        events = event_types(container, task.task_id)
        assert final.status == TaskStatus.COMPLETED and events["task.completed"] == 1
        assert all(n == 1 for n in Counter(a.name for a in container.task_service.artifacts(task.task_id)).values())
    finally:
        container.close()


async def test_a_live_worker_renews_its_lease_through_a_slow_call(tmp_path) -> None:
    clock = Clock()
    container, llm = open_container(tmp_path / "data", clock=clock)
    try:
        container.orchestrator._heartbeat = 0.001
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(llm)
        gate.arm()
        slow = asyncio.create_task(container.task_service.submit_assessment(task.task_id, answers_for(task)))
        await gate.entered.wait()
        repo = container.task_service._tasks
        first_until = repo.lease(task.task_id).until
        clock.offset = TASK_LEASE * 0.9
        while repo.lease(task.task_id).until <= first_until:  # the heartbeat renews (a condition, not a delay)
            await asyncio.sleep(0)
        clock.offset = TASK_LEASE * 1.5  # past the original expiry: still held, so nobody else can take it
        with pytest.raises(TaskBusy):
            await container.task_service.resume(task.task_id)
        gate.open()
        done = await slow
        assert done.status in (TaskStatus.WAITING, TaskStatus.COMPLETED)
        assert event_types(container, task.task_id)["task.lease_lost"] == 0
    finally:
        container.close()


async def test_cancel_during_a_run_is_not_lost_to_the_workers_checkpoints(tmp_path) -> None:
    container, llm = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        gate = Gate(llm)
        gate.arm()
        running = asyncio.create_task(container.task_service.submit_assessment(task.task_id, answers_for(task)))
        await gate.entered.wait()
        container.task_service.pause(task.task_id)  # recorded while the worker holds the lease
        gate.open()
        done = await running
        assert done.status == TaskStatus.PAUSED  # the worker saw the request at its next node boundary
        assert not container.task_service._tasks.control(task.task_id).pause_requested
    finally:
        container.close()
