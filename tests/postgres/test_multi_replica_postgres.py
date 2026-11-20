"""Multi-replica readiness on PostgreSQL (issue #66 PR 1): database-time leases and the teaching-turn lease.

Opt-in like the other PostgreSQL tests (TA_TEST_POSTGRES_URL). Every run uses fresh ids, so the database can be reused.

Two machines whose process clocks are skewed by -10 and +10 minutes are two real processes (spawned, each with its own
engine and its own skewed clock) on one database. For each lease (task, task job, learning-cycle drive, teaching turn):
- only one holds it: the machine whose clock runs 20 minutes ahead cannot take over a live lease;
- takeover happens only once the lease ran out in database time (the test waits on the database's clock, a lease of
  two seconds, never a guess at the process clocks);
- the stale holder's protected write after the takeover is refused and leaves no row.
Racing claims from eight skewed processes: exactly one wins. Spend: a reservation's UTC day and expiry come from the
database's clock whatever the process clock says, and a sweeper whose clock runs ahead abandons nothing live. The
teaching-turn lease across two API replicas: the same answer sent to both makes one teacher call per owed turn, and a
different answer is refused at once. Migration 0005 up and down.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context

import pytest
from sqlalchemy import create_engine, inspect, select, text

from app.schemas.common import new_id, utcnow
from app.storage.db import create_db, dispose
from app.storage.migrate import alembic_config, upgrade

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL multi-replica tests")
SKEW = timedelta(minutes=10)
TTL = timedelta(seconds=2)
KINDS = ["task", "job", "cycle", "teaching"]


# --- what one machine does (runs in its own spawned process) -------------------------------------------------------

_SESSIONS: dict = {}


def _sessions(url: str):
    if url not in _SESSIONS:
        _SESSIONS[url] = create_db(url)
    return _SESSIONS[url]


def _clock(skew_s: float):
    return lambda: utcnow() + timedelta(seconds=skew_s)


def machine(url: str, skew_s: float, kind: str, op: str, resource_id: str, held=None):
    """One step on a machine whose process clock is `skew_s` off. Returns something picklable."""
    from app.observability.events import EventBus
    from app.schemas.events import Event, EventType
    from app.schemas.lease import LeaseLost
    from app.schemas.learning_cycle import CycleConflict
    from app.schemas.teaching import OutboxItem, SessionChange, SessionConflict
    from app.services.learning_cycles import LearningCycleService
    from app.storage.repositories import (
        SqlEventRepository,
        SqlLearningCycleRepository,
        SqlTaskJobRepository,
        SqlTaskRepository,
        SqlTeachingRepository,
    )

    sessions, clock = _sessions(url), _clock(skew_s)
    if op == "warm":
        return True
    if kind == "task":
        tasks = SqlTaskRepository(sessions, clock)
        if op == "claim":
            return tasks.claim(resource_id, new_id("w"), TTL)
        task = tasks.get(resource_id)
        events = SqlEventRepository(sessions, clock)
        refused = []
        try:
            tasks.save(task, lease=held)
        except LeaseLost:
            refused.append("save")
        try:
            events.append(Event(event_id=new_id("stale"), type=EventType.NODE_STARTED, task_id=resource_id,
                                at=clock()), fence=held)
        except LeaseLost:
            refused.append("event")
        return refused
    if kind == "job":
        jobs = SqlTaskJobRepository(sessions, clock)
        if op == "claim":
            owner = new_id("w")
            while (job := jobs.claim(owner, TTL)) is not None:  # other runs' leftovers may come first
                if job.task_id == resource_id:
                    return owner
                jobs.finish(job.job_id, owner, _done())
            return None
        job = jobs.active_for_task(resource_id)
        return [name for name, ok in (("renew", jobs.renew(job.job_id, held, TTL)),
                                      ("finish", jobs.finish(job.job_id, held, _done()))) if not ok]
    if kind == "cycle":
        repo = SqlLearningCycleRepository(sessions, clock)
        service = LearningCycleService(repo, curriculum=None, tasks=None, teaching=None, memory=None, artifacts=None,
                                       events=EventBus(), event_log=None, clock=clock, drive_lease=TTL)
        if op == "claim":
            cycle, token = asyncio.run(service._claim(resource_id))
            return (token, cycle.lease.until) if token else None
        from app.curriculum import cycle as engine

        cycle = repo.get(resource_id)
        try:
            asyncio.run(service._commit(cycle, engine.cancel(cycle, clock()), held))
        except CycleConflict:
            return ["commit"]
        return []
    if kind == "teaching":
        repo = SqlTeachingRepository(sessions, clock)
        if op == "claim":
            return repo.claim_turns(resource_id, TTL)
        session = repo.get(resource_id)
        work = session.model_copy(update={"version": session.version + 1})
        outbox = OutboxItem(item_id=new_id("stale"), session_id=resource_id, kind="event")
        try:
            repo.apply(SessionChange(session=work, outbox=[outbox]), expected_version=session.version,
                       turn_lease=held)
        except SessionConflict:
            return ["apply"]
        return []
    raise AssertionError(kind)


def _done():
    from app.schemas.job import JobStatus

    return JobStatus.DONE


# --- the resources ------------------------------------------------------------------------------------------------


def resource(kind: str, sessions) -> str:
    """A fresh task, queued job, learning cycle or teaching session: its id (a job's: its task's)."""
    from app.curriculum import cycle as engine
    from app.schemas.curriculum import LearningActionType, NextLearningAction
    from app.schemas.job import JobKind, TaskJob
    from app.schemas.task import Task
    from app.schemas.teaching import TeachingConfig
    from app.storage.repositories import (
        SqlLearningCycleRepository,
        SqlTaskJobRepository,
        SqlTaskRepository,
        SqlTeachingRepository,
    )
    from app.teaching import engine as teaching

    learner = new_id("pg-mr")
    if kind in ("task", "job"):
        task_id = new_id("pgtask")
        SqlTaskRepository(sessions).create(Task(task_id=task_id, user_id="u", learner_id=learner, request="r"))
        if kind == "job":
            SqlTaskJobRepository(sessions).enqueue(TaskJob(job_id=new_id("pgjob"), task_id=task_id, kind=JobKind.RUN,
                                                           dedupe_key="RUN:x"))
        return task_id
    if kind == "cycle":
        now = utcnow()
        action = NextLearningAction(action_id=new_id("act"), learner_id=learner, action=LearningActionType.WAIT,
                                    reason="nothing due", as_of=now)
        cycle = engine.start(learner_id=learner, user_id="u", idempotency_key=new_id("k"), action=action, at=now)
        assert SqlLearningCycleRepository(sessions).create(cycle)
        return cycle.cycle_id
    change = teaching.start(session_id=new_id("pgsess"), learner_id=learner, task_id=new_id("t"),
                            lesson_id=new_id("l"), objective_id=None, objective_description="Learn it",
                            concept_id="c", concept_name="C", action="LEARN", mode="assessed",
                            config=TeachingConfig(), at=utcnow(), metadata={})
    assert SqlTeachingRepository(sessions).create(change)
    return change.session.session_id


def database_now(sessions) -> datetime:
    with sessions() as s:
        return s.scalar(text("SELECT clock_timestamp()")).astimezone(timezone.utc)


def until_of(kind: str, claimed) -> datetime:
    return claimed[1] if kind == "cycle" else claimed.until if kind in ("task", "teaching") else None


@pytest.fixture
def machines():
    ctx = get_context("spawn")
    behind = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    ahead = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    for pool, skew in ((behind, -SKEW), (ahead, SKEW)):
        assert pool.submit(machine, URL, skew.total_seconds(), "-", "warm", "").result(timeout=120)
    yield behind, ahead
    behind.shutdown()
    ahead.shutdown()


@pytest.mark.parametrize("kind", KINDS)
def test_skewed_machines_hold_a_lease_one_at_a_time_and_take_over_only_in_database_time(kind, machines) -> None:
    behind, ahead = machines
    sessions = create_db(URL)
    try:
        rid = resource(kind, sessions)

        def on(pool, op: str, held=None, skew: timedelta = SKEW):
            sign = -1 if pool is behind else 1
            return pool.submit(machine, URL, sign * skew.total_seconds(), kind, op, rid, held).result(timeout=60)

        first = on(behind, "claim")
        assert first is not None
        started = database_now(sessions)
        # 20 minutes ahead of the holder's clock, so long past the lease by its own clock: still refused.
        assert on(ahead, "claim") is None, "a machine whose clock runs ahead took over a live lease"
        if kind != "job":
            ends = until_of(kind, first)
            assert started < ends <= started + TTL  # the lease ends at database time plus its TTL
        while database_now(sessions) <= started + TTL:  # until it ran out in database time
            time.sleep(0.05)
        second = on(ahead, "claim")
        assert second is not None, "the lease did not expire in database time"
        if kind == "task":
            assert second.fence == first.fence + 1
        held = first[0] if kind == "cycle" else first
        refused = on(behind, "stale", held)
        assert refused == {"task": ["save", "event"], "job": ["renew", "finish"], "cycle": ["commit"],
                           "teaching": ["apply"]}[kind]
        _no_stale_rows(kind, rid, sessions)
    finally:
        dispose(sessions)


def _no_stale_rows(kind: str, rid: str, sessions) -> None:
    from app.schemas.job import JobStatus
    from app.storage.orm import EventRow, LearningCycleRow, TaskJobRow, TaskRow, TeachingOutboxRow

    with sessions() as s:
        if kind == "task":
            assert s.get(TaskRow, rid).version == 1  # the stale save stored nothing
            assert s.scalars(select(EventRow).where(EventRow.task_id == rid)).all() == []
        elif kind == "job":
            job = s.scalars(select(TaskJobRow).where(TaskJobRow.task_id == rid)).one()
            assert job.status == JobStatus.RUNNING.value and job.attempts == 2  # the new holder's, untouched
        elif kind == "cycle":
            assert s.get(LearningCycleRow, rid).status == "RUNNING"  # the stale cancel stored nothing
        else:
            assert s.scalars(select(TeachingOutboxRow).where(TeachingOutboxRow.session_id == rid,
                                                             TeachingOutboxRow.item_id.like("stale%"))).all() == []


def test_racing_claims_from_skewed_processes_have_exactly_one_winner() -> None:
    sessions = create_db(URL)
    ctx = get_context("spawn")
    try:
        rids = [resource("task", sessions) for _ in range(3)]
        with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as pool:
            list(pool.map(machine, [URL] * 8, [0] * 8, ["-"] * 8, ["warm"] * 8, [""] * 8))
            for rid in rids:
                skews = [(-1) ** i * (i + 1) * 60.0 for i in range(8)]  # -1 min ... +8 min
                won = [r for r in pool.map(machine, [URL] * 8, skews, ["task"] * 8, ["claim"] * 8, [rid] * 8) if r]
                assert len(won) == 1
    finally:
        dispose(sessions)


# --- spend ----------------------------------------------------------------------------------------------------------


def test_a_reservations_day_and_expiry_are_database_time_whatever_the_process_clock() -> None:
    from app.observability.spend import task_account
    from app.schemas.providers import Capability
    from app.schemas.spend import Estimate, SpendState
    from app.schemas.usage import BudgetDefaults, TaskBudget
    from app.storage.repositories import SqlSpendRepository

    sessions = create_db(URL)
    try:
        today = database_now(sessions).date()
        tomorrow = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        # The worker's process clock says it is already tomorrow (UTC); the database says today.
        worker_clock = lambda: tomorrow + timedelta(hours=1)  # noqa: E731
        store = SqlSpendRepository(sessions, worker_clock)
        learner, task_id = new_id("pg-day"), new_id("task")
        defaults = BudgetDefaults(learner_daily=TaskBudget(max_llm_requests=100))
        account = task_account(store, defaults, task_id=task_id, learner_id=learner, budget=None, fence=None,
                               clock=worker_clock)
        before = database_now(sessions)
        r = account.reserve(entry_id=new_id("preq"), operation_id=new_id("pop"), attempt=1, capability=Capability.LLM,
                            operation="generate", provider="p", model="m", estimate=Estimate(llm_tokens=10),
                            timeout_seconds=30)
        assert f"learner:{learner}:{today.isoformat()}" in {s.key for s in r.scopes}
        [entry] = store.entries(task_id=task_id)
        expires = entry.reserved_until
        assert before + timedelta(seconds=30) < expires <= database_now(sessions) + timedelta(seconds=30 + 60)
        # A sweeper whose clock runs a day ahead abandons nothing still live in database time.
        sweeper = SqlSpendRepository(sessions, worker_clock)
        while sweeper.expire_abandoned(worker_clock() + timedelta(days=1)):
            pass
        [entry] = store.entries(task_id=task_id)
        assert entry.state == SpendState.RESERVED
        account.settle(r, None, SpendState.FAILED)
    finally:
        dispose(sessions)


# --- the teaching-turn lease across two API replicas ---------------------------------------------------------------


def test_the_same_answer_on_two_replicas_makes_one_teacher_call_and_another_is_refused(tmp_path) -> None:
    from app.schemas.learning_cycle import LearnerPrompt
    from app.schemas.teaching import LearnerInput
    from app.teaching.errors import TeachingTurnBusy
    from scripts.run_learning_cycle_demo import ScriptedLearner
    from tests.auth import TestAuth, client_for
    from tests.postgres.test_student_app_postgres import placed_learner, replica

    first, second = replica(tmp_path / "data"), replica(tmp_path / "data")
    try:
        learner_id = placed_learner(first)
        auth = TestAuth()
        with client_for(first, learner_id, auth=auth) as a:
            goal = a.post(f"/learners/{learner_id}/goals", json={
                "title": "Reach B1 Spanish", "domain": "spanish", "target_level": "B1",
                "idempotency_key": new_id("goal"), "metadata": {"framework_id": "cefr"}})
            assert a.post(f"/goals/{goal.json()['goal_id']}/curriculum").status_code == 201
            view = a.post(f"/learners/{learner_id}/learning-cycles", json={"idempotency_key": new_id("w")}).json()
            learner = ScriptedLearner(prefix="mr")
            for _ in range(20):  # to the session's first question
                if view["prompt"]["kind"] == "SESSION_QUESTION":
                    break
                response = learner.respond(LearnerPrompt.model_validate(view["prompt"]))
                view = a.post(f"/learning-cycles/{view['cycle_id']}/responses",
                              json=response.model_dump(mode="json", exclude_none=True)).json()
            assert view["prompt"]["kind"] == "SESSION_QUESTION"
        sid = next(s["child_id"] for s in view["steps"] if s["kind"] == "TEACHING_SESSION")
        question = first.teaching_service.repository.get(sid).state.pending_question
        data = LearnerInput(answer=learner.answer(question.prompt), client_turn_id=new_id("same"))

        calls, lock = [], threading.Lock()
        entered, release = threading.Event(), threading.Event()
        for c in (first, second):
            real = c.teaching_service._runtime.teacher_turn

            async def gated(*args, _real=real, **kwargs):
                with lock:
                    calls.append(1)
                    hold = len(calls) == 1
                if hold:
                    entered.set()
                    await asyncio.to_thread(release.wait, 60)
                return await _real(*args, **kwargs)

            c.teaching_service._runtime.teacher_turn = gated

        results: dict = {}

        def submit(name: str, c, payload) -> None:
            try:
                results[name] = asyncio.run(c.teaching_service.submit(sid, payload))
            except BaseException as exc:  # noqa: BLE001 - checked below
                results[name] = exc

        holder = threading.Thread(target=submit, args=("holder", first, data))
        holder.start()
        assert entered.wait(60)  # replica 1 holds the lease, inside the teacher call
        repeat = threading.Thread(target=submit, args=("repeat", second, data))
        repeat.start()
        other = LearnerInput(answer="otra cosa", client_turn_id=new_id("other"))
        submit("other", second, other)  # replica 2, a different answer: refused at once, no grading, no call
        assert isinstance(results["other"], TeachingTurnBusy) and len(calls) == 1
        time.sleep(0.2)
        assert repeat.is_alive() and len(calls) == 1  # the repeat waits on replica 2 and calls nothing
        release.set()
        holder.join(60)
        repeat.join(60)
        a_result, b_result = results["holder"], results["repeat"]
        assert not isinstance(a_result, BaseException), a_result
        assert not isinstance(b_result, BaseException), b_result
        assert sorted([a_result.replayed, b_result.replayed]) == [False, True]
        assert [t.turn_id for t in a_result.teacher_turns] == [t.turn_id for t in b_result.teacher_turns]
        assert len(calls) == len(a_result.teacher_turns)  # one teacher call per owed turn, on either replica
        assert first.teaching_service.repository.turn_lease(sid) is None
    finally:
        first.close()
        second.close()


# --- migration 0005 ------------------------------------------------------------------------------------------------


def test_migration_0005_up_and_down_on_postgresql(fresh_db) -> None:
    from alembic import command

    upgrade(fresh_db)

    def columns() -> set[str]:
        engine = create_engine(fresh_db)
        try:
            return {c["name"] for c in inspect(engine).get_columns("teaching_sessions")}
        finally:
            engine.dispose()

    lease = {"turn_lease_token", "turn_lease_expires_ms"}
    assert lease <= columns()
    command.downgrade(alembic_config(fresh_db), "0004_class_assignments")
    assert not lease & columns()
    command.upgrade(alembic_config(fresh_db), "head")
    assert lease <= columns()

