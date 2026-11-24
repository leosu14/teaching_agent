"""Issue #66 PR 3: the writes the GETs used to make, made by their writers and recovered by the sweeper.

- A cycle's child run by the worker (the answers queued through the task API, not the cycle): the worker's hook
  stores what the child did in the cycle, with no read of the cycle.
- The hook lost (the worker died after the job, or the hook failed): the cycle's record lags; reads still show the
  child's progress and store nothing; the sweeper stores it, once.
- The lesson's diagnostic and lesson evidence refresh the learner's curricula in the lesson's own job.
- A cycle whose events a crash left unpublished is found by its stored body, published once by the sweeper.
- `python -m app.services.worker --sweep` (cron) walks everything and exits non-zero when anything failed."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.schemas.curriculum import curriculum_id_for
from app.services import learning_cycles as cycle_service
from app.services import worker as worker_module
from app.storage.orm import LearningCycleRow
from app.storage.repositories import NO_PENDING_EVENTS
from tests.learning_cycle_fixtures import LEARNER, ScriptedLearner


def the_cycle(env):
    [cycle] = env.cycles.repository.for_learner(LEARNER)
    return cycle


def lesson_of(env, cycle) -> str:
    return next(s.child_id for s in cycle.steps if s.kind.value == "LESSON")


async def answer_the_diagnostic_through_the_task(env) -> tuple[str, int]:
    """Answer the lesson's diagnostic as the task API does (queued for the worker), not through the cycle."""
    c = env.container
    cycle = the_cycle(env)
    view = await env.cycles.get(cycle.cycle_id)
    response = ScriptedLearner(prefix="d").respond(view.prompt)
    lesson = lesson_of(env, cycle)
    c.task_service.enqueue_answers(lesson, {"answers": [a.model_dump(mode="json") for a in response.answers]})
    return lesson, cycle.version


async def test_the_worker_stores_a_cycle_childs_outcome_in_the_cycle(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic")
    c = env.container
    lesson, version = await answer_the_diagnostic_through_the_task(env)
    while await c.task_worker.run_once() is not None:
        pass
    assert c.task_service.get(lesson).status.value == "COMPLETED"
    stored = the_cycle(env)
    assert stored.version > version  # the hook stored it, no read did
    assert stored.steps[0].status.value == "COMPLETED"
    # the lesson's own job refreshed the curricula with its diagnostic and lesson evidence (no GET, no sweep)
    goal = next(g for g in c.curriculum_service.goals(LEARNER) if g.status.value == "ACTIVE")
    snapshot = c.curriculum_service.engine.repository.progress(curriculum_id_for(goal.goal_id))
    assert snapshot is not None and any(p.evidence_count > 0 for p in snapshot.objectives)


async def test_a_lost_hook_is_recovered_by_the_sweeper_and_reads_store_nothing(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("diagnostic")
    c = env.container
    lesson, version = await answer_the_diagnostic_through_the_task(env)
    c.task_worker.on_settled = None  # the worker died after its job, before the hook ran
    while await c.task_worker.run_once() is not None:
        pass
    assert c.task_service.get(lesson).status.value == "COMPLETED"
    assert the_cycle(env).version == version  # the cycle's record lags
    view = await env.cycles.get(the_cycle(env).cycle_id)
    assert view.steps[0].status.value == "COMPLETED"  # a read shows the child's progress
    assert the_cycle(env).steps[0].status.value == "WAITING"  # stored: still waiting on the diagnostic
    assert view.version == version and the_cycle(env).version == version  # and stores nothing
    report = await c.task_worker.sweep_learning(full=True)
    assert report.errors == 0 and report.done["cycles"] >= 1, report.summary()
    stored = the_cycle(env)
    assert stored.version > version and stored.steps[0].status.value == "COMPLETED"
    again = await c.task_worker.sweep_learning(full=True)
    assert again.errors == 0 and the_cycle(env).version == stored.version  # idempotent


async def test_a_failing_hook_is_only_logged_and_the_sweeper_recovers(cycle_env_at, caplog) -> None:
    env = cycle_env_at("diagnostic")
    c = env.container
    lesson, version = await answer_the_diagnostic_through_the_task(env)

    async def broken(task_id: str) -> None:
        raise RuntimeError("database blip")

    c.task_worker.on_settled = broken
    while (job := await c.task_worker.run_once()) is not None:
        assert job.status.value == "DONE" or job.task_id != lesson  # the job's own outcome is unaffected
    assert "for its learning cycle failed" in caplog.text
    assert the_cycle(env).version == version
    await c.task_worker.sweep_learning(full=True)
    assert the_cycle(env).steps[0].status.value == "COMPLETED"


async def test_unpublished_cycle_events_are_found_by_the_stored_body(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("done")
    c = env.container
    cycle = the_cycle(env)
    assert cycle.status.value == "COMPLETED" and not cycle.pending_events
    with c._sessions() as s:
        body = s.scalar(select(LearningCycleRow.body).where(LearningCycleRow.cycle_id == cycle.cycle_id))
    assert NO_PENDING_EVENTS in body  # the marker matches the stored form
    assert [n for n in type(cycle).model_fields if "pending_events" in n] == ["pending_events"]
    assert body.count('"pending_events"') == 1  # no nested field of that name
    assert c.learning_cycle_service.repository.active_ids(None, 100) == []  # terminal, nothing owed
    # a crash between a terminal write and its publish leaves events owed
    from tests.integration.test_learning_cycle_flow import SimulatedCrash
    env2 = cycle_env_at("session")
    original = cycle_service.LearningCycleService._publish

    async def crashing(self, cyc):
        if any(e.type == "learning_cycle.completed" for e in cyc.pending_events):
            raise SimulatedCrash("publish")
        return await original(self, cyc)

    from tests.learning_cycle_fixtures import answer
    monkeypatch.setattr(cycle_service.LearningCycleService, "_publish", crashing)
    with pytest.raises(SimulatedCrash):
        await answer(env2, await env2.cycles.get(the_cycle(env2).cycle_id), ScriptedLearner(prefix="s"))
    monkeypatch.undo()
    owed = the_cycle(env2)
    assert owed.status.value == "COMPLETED" and owed.pending_events
    assert env2.container.learning_cycle_service.repository.active_ids(None, 100) == [owed.cycle_id]
    await env2.container.task_worker.sweep_learning(full=True)
    assert not the_cycle(env2).pending_events
    assert env2.container.learning_cycle_service.repository.active_ids(None, 100) == []


def test_the_cron_sweep_exits_non_zero_when_anything_failed(cycle_env_at, monkeypatch, capsys) -> None:
    """`--sweep`'s pass (`sweep_once`): its counts printed; exit code 1 once any learning id failed."""
    env = cycle_env_at("session")
    worker = env.container.task_worker
    assert worker_module.sweep_once(worker) == 0
    out = capsys.readouterr().out
    assert "recovered tasks" in out and "learning: curricula: 1 done, 0 failed" in out

    async def broken(learner_id, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(env.container.curriculum_service, "tracked", broken)
    worker.learning_sweeper = None  # rebuilt below with the broken refresh, as the container wires it
    from app.services.sweeper import LearningSweeper
    c = env.container
    worker.learning_sweeper = LearningSweeper.of(
        learners=c.curriculum_service.engine.repository.learners_to_refresh, refresh=c.curriculum_service.tracked,
        cycles=c.learning_cycle_service.repository.active_ids, reconcile=c.learning_cycle_service.reconcile,
        outboxes=c.teaching_service.repository.with_pending_outbox, publish=c.teaching_service.publish_pending)
    assert worker_module.sweep_once(worker) == 1
    assert "curricula: 0 done, 1 failed" in capsys.readouterr().out
