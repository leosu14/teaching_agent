"""Concurrent learning-cycle starts and resumes, and stale cycle drivers (PR18).

Deterministic like tests/integration/test_task_concurrency.py: a gated provider call parks a request at a known point,
lease expiry is a clock moved forward. Every test checks what ran (workflow nodes started, model calls) and what was
persisted (the cycle, its children, events, artifacts, learner state) against an uncontended run.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import timedelta

import pytest

from app.curriculum import cycle as engine
from app.runtime.orchestrator.orchestrator import TASK_LEASE
from app.schemas.learning_cycle import CycleConflict, CycleLeaseLost, CycleStatus, StartLearningCycle
from app.services import learning_cycles as cycle_service
from tests.integration.test_learning_cycle_flow import LIFECYCLE, cycle_events, cycle_id, footprint, lessons
from tests.integration.test_task_concurrency import Clock, Gate
from tests.learning_cycle_fixtures import LEARNER, CycleEnv, ScriptedLearner, answer, open_cycle_env

# Past both leases: the cycle's drive lease and its child task's lease.
AHEAD = max(cycle_service.DRIVE_LEASE, TASK_LEASE) + timedelta(seconds=1)


def node_starts(env: CycleEnv, task_id: str) -> Counter:
    return Counter(e.node_id for e in env.container.task_service.events(task_id) if e.type == "node.started")


def planning_tasks(env: CycleEnv) -> list:
    return [t for t in env.container.task_service.list_for_learner(LEARNER)
            if t.plan and t.plan.workflow_id == "curriculum_planning"]


async def entered(gate: Gate) -> None:
    await asyncio.wait_for(gate.entered.wait(), timeout=60)  # fails instead of hanging if nothing was parked


# --- simultaneous starts -----------------------------------------------------------------------------------------


async def test_a_late_duplicate_start_finds_the_curriculum_planned(cycle_env_at, monkeypatch) -> None:
    """The concurrent-start failure seen on Python 3.13 (`test_concurrent_starts_execute_one_action`), forced.

    Before PR18 two starts with the same key both reached `build_curriculum` before the cycle existed. Both found the
    one planning task by its key; the first ran it to COMPLETED while the second, which had seen it unfinished, then
    called `run`/`resume` on it: `InvalidTransition: cannot go from COMPLETED to RUNNING`, an unmapped 500 (or, with
    the other interleaving, the planning workflow ran twice). Whether it showed depended on the interpreter's
    scheduling. Here the second request is held between reading the task and running it until the first finished,
    which is exactly that interleaving: it now claims the task's lease, sees the task completed, and continues."""
    baseline_env = cycle_env_at("base")
    baseline_view = await baseline_env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    baseline = footprint(baseline_env)
    baseline_calls = sum(baseline_env.llm.calls.values())

    env = cycle_env_at("base")
    tasks = env.container.task_service
    first_done = asyncio.Event()
    order: list[str] = []
    original_run = tasks.run

    async def run(task_id: str):
        order.append(task_id)
        if len(order) == 1:  # the first request: let the second read the task first, then run it to the end
            for _ in range(5):
                await asyncio.sleep(0)
            try:
                return await original_run(task_id)
            finally:
                first_done.set()
        await first_done.wait()  # the second request read the task before the first finished it
        return await original_run(task_id)

    monkeypatch.setattr(tasks, "run", run)
    views = await asyncio.gather(*(env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
                                   for _ in range(2)))
    monkeypatch.undo()
    [plan] = planning_tasks(env)
    assert order[:2] == [plan.task_id, plan.task_id]  # both reached the same planning task (the old race)
    assert {v.cycle_id for v in views} == {baseline_view.cycle_id}
    assert node_starts(env, plan.task_id) == node_starts(baseline_env, planning_tasks(baseline_env)[0].task_id)
    assert all(n == 1 for n in node_starts(env, plan.task_id).values())  # planned once
    assert sum(env.llm.calls.values()) == baseline_calls
    assert footprint(env) == baseline


async def test_a_duplicate_start_during_planning_is_refused(cycle_env_at) -> None:
    baseline_env = cycle_env_at("base")
    await baseline_env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    baseline = footprint(baseline_env)

    env = cycle_env_at("base")
    gate = Gate(env.llm)
    gate.arm()
    first = asyncio.create_task(env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1")))
    await entered(gate)  # parked inside the curriculum planning task, holding its lease
    duplicates = await asyncio.gather(env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1")),
                                      env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="other")),
                                      return_exceptions=True)
    assert all(isinstance(d, CycleConflict) for d in duplicates), duplicates
    calls_while_parked = sum(env.llm.calls.values())
    gate.open()
    view = await first
    assert view.cycle_id == cycle_id(env) and view.prompt is not None
    assert calls_while_parked == 0  # the refused starts executed nothing
    [plan] = planning_tasks(env)
    assert all(n == 1 for n in node_starts(env, plan.task_id).values())
    assert footprint(env) == baseline


# --- simultaneous resumes and stale drivers ----------------------------------------------------------------------


async def test_simultaneous_cycle_resumes_drive_the_lesson_once(cycle_env_at) -> None:
    baseline_env = cycle_env_at("diagnostic")
    cid = cycle_id(baseline_env)
    view = await baseline_env.cycles.get(cid)
    await baseline_env.cycles.respond(cid, ScriptedLearner(prefix="b").respond(view.prompt))
    baseline = footprint(baseline_env)
    baseline_calls = sum(baseline_env.llm.calls.values())
    baseline_nodes = node_starts(baseline_env, lessons(baseline_env)[0].task_id)

    env = cycle_env_at("diagnostic")
    view = await env.cycles.get(cid)
    gate = Gate(env.llm)
    gate.arm()
    responding = asyncio.create_task(env.cycles.respond(cid, ScriptedLearner(prefix="b").respond(view.prompt)))
    await entered(gate)  # the lesson is running inside this request, under the cycle's and the task's leases
    racing = await asyncio.gather(env.cycles.resume(cid), env.cycles.resume(cid), env.cycles.get(cid))
    assert all(v.busy_until is not None for v in racing)  # seen busy: nothing driven
    assert sum(env.llm.calls.values()) == 0
    gate.open()
    await responding
    [lesson] = lessons(env)
    assert node_starts(env, lesson.task_id) == baseline_nodes  # every node ran as often as uncontended
    assert Counter(e.type for e in env.container.task_service.events(lesson.task_id))["task.lease_lost"] == 0
    assert sum(env.llm.calls.values()) == baseline_calls
    assert footprint(env) == baseline


async def test_a_stale_cycle_driver_commits_nothing_after_takeover(cycle_env_at) -> None:
    """The audit's probe 2, deterministic: process A runs the lesson inside a learner's response; its model call
    outlives the leases, and process B (its clock past them) resumes the cycle, takes over the cycle and the lesson
    task, and runs the lesson. Before PR18 both ran it to the end (21 LLM calls for 13, two `task.completed`, 15
    duplicated artifact versions). Now A's writes are refused when its call returns, B's run stands, and the cycle
    ends exactly like an uncontended one."""
    baseline_env = cycle_env_at("diagnostic")
    cid = cycle_id(baseline_env)
    await answer(baseline_env, await baseline_env.cycles.get(cid), ScriptedLearner(prefix="b"))
    baseline = footprint(baseline_env)

    a = cycle_env_at("diagnostic")
    b = open_cycle_env(a.data_dir, copy_from=None, clock=Clock(AHEAD))
    try:
        view = await a.cycles.get(cid)
        learner = ScriptedLearner(prefix="b")
        gate = Gate(a.llm)
        gate.arm()
        stale = asyncio.create_task(a.cycles.respond(cid, learner.respond(view.prompt)))
        await entered(gate)
        old_generation = a.cycles.cycle(cid).lease_generation

        taken = await b.cycles.resume(cid)  # B: A's leases have expired by B's clock
        assert a.cycles.cycle(cid).lease_generation == old_generation + 1 and taken.busy_until is None  # B drove
        [lesson] = lessons(a)
        state_after_b = (a.container.task_service.get(lesson.task_id).model_dump_json(),
                         sorted(x.artifact_id for x in a.container.task_service.artifacts(lesson.task_id)))

        gate.open()
        a_view = await stale  # A's response was received; its stale lesson writes were refused
        assert a_view.cycle_id == cid
        assert (a.container.task_service.get(lesson.task_id).model_dump_json(),
                sorted(x.artifact_id for x in a.container.task_service.artifacts(lesson.task_id))) == state_after_b
        lesson_events = Counter(e.type for e in a.container.task_service.events(lesson.task_id))
        assert lesson_events["task.lease_lost"] == 1

        final = await answer(b, await b.cycles.get(cid), learner)
        assert final.status == CycleStatus.COMPLETED
        lesson_events = Counter(e.type for e in b.container.task_service.events(lesson.task_id))
        assert lesson_events["task.completed"] == 1
        events = cycle_events(b, cid)
        assert all(events[t] == 1 for t in LIFECYCLE), events
        assert Counter(x.name for x in b.cycles.artifacts(cid)) == {"learning_cycle": 1, "learning_cycle_outcome": 1}
        assert footprint(b) == baseline
    finally:
        b.close()


async def test_a_driver_write_is_fenced_by_token_and_expiry(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic")
    repo = env.cycles.repository
    cid = cycle_id(env)
    start = env.cycles._clock()
    cycle = repo.get(cid)
    old = engine.claim(cycle, start, start + cycle_service.DRIVE_LEASE)
    repo.apply(old, expected_version=cycle.version)
    token = old.lease.token

    renewed = engine.renew(old, token, start, start + cycle_service.DRIVE_LEASE)
    repo.apply(renewed, expected_version=old.version, lease_token=token, at=start)  # the holder writes

    later = start + AHEAD
    with pytest.raises(CycleLeaseLost):  # expired, nobody took over: still refused
        repo.apply(engine.record_artifact(renewed, "probe", "art_x", later), expected_version=renewed.version,
                   lease_token=token, at=later)
    new = engine.claim(renewed, later, later + cycle_service.DRIVE_LEASE)
    repo.apply(new, expected_version=renewed.version)
    assert new.lease_generation == renewed.lease_generation + 1 and new.lease.token != token
    with pytest.raises(CycleLeaseLost):  # superseded: even with the current version, the old token is refused
        repo.apply(engine.record_artifact(new, "probe", "art_x", later), expected_version=new.version,
                   lease_token=token, at=later)
    with pytest.raises(CycleConflict):  # a write computed from an old version never overwrites a newer one
        repo.apply(engine.record_artifact(renewed, "probe", "art_y", later), expected_version=renewed.version)
    stored = repo.get(cid)
    assert stored.version == new.version and "probe" not in stored.artifact_ids
