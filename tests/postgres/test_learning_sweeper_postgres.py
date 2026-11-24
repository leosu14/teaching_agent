"""Issue #66 PR 3 on PostgreSQL: the learning sweeper is safe to run on every worker at once, and the GETs it took the
writes from write nothing.

Opt-in like the other PostgreSQL tests: skipped unless TA_TEST_POSTGRES_URL names a database the test may write to.
Two replicas share the database. Every run uses fresh ids, so the database can be reused; each sweeper here walks
only this run's learners (the database may hold other runs' data).

- Learners whose goals were mastered by evidence recorded without a refresh; four sweeps (two per replica) start at
  once from a barrier, each walking every learner `batch` at a time: no error, every goal COMPLETED exactly once
  (one `goal.completed` event each) at D11's time, the curriculum following the goal; a later pass changes nothing.
- A sweep racing the learner's deletion on the other replica: no error, nothing of the learner left.
- GET next-action, with and without a learner `as_of`, sends no INSERT, UPDATE or DELETE to PostgreSQL."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta

from sqlalchemy import event, func, select

from app.schemas.common import new_id, utcnow
from app.schemas.curriculum import GoalInput, curriculum_id_for, curriculum_status_for
from app.schemas.learner import GoalStatus, LearnerProfileInput
from app.services.sweeper import LearningSweeper
from app.storage.orm import EventRow
from tests.auth import client_for
from tests.postgres.test_class_assignments_postgres import pytestmark, race, replicas  # noqa: F401 - fixtures
from tests.postgres.test_curriculum_status_postgres import mastered

WRITE = re.compile(r"\s*(INSERT|UPDATE|DELETE)\b", re.I)


def learner_with_a_mastered_goal(c) -> tuple[str, str]:
    lid = new_id("l")
    c.learner_service.upsert(lid, LearnerProfileInput(display_name="Swept"))
    goal, _ = asyncio.run(c.curriculum_service.create_goal(lid, GoalInput(
        title="Sweep", domain="spanish", target_concepts=["es.preterite"])))
    mastered(c, lid, goal.goal_id)  # evidence through the learner service: no refresh
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.ACTIVE
    return lid, goal.goal_id


def sweeper_of(c, learners: list[str], batch: int = 2) -> LearningSweeper:
    """The container's learning sweeper, over these learners only (and no cycle or outbox of other runs)."""
    def ours(after, limit):
        return [i for i in sorted(learners) if after is None or i > after][:limit]

    return LearningSweeper.of(learners=ours, refresh=c.curriculum_service.tracked, cycles=lambda a, n: [],
                              reconcile=c.learning_cycle_service.reconcile, outboxes=lambda a, n: [],
                              publish=c.teaching_service.publish_pending, batch=batch)


def completed_events(c, goal_id: str) -> int:
    with c._sessions() as s:
        return s.scalar(select(func.count()).select_from(EventRow).where(
            EventRow.type == "goal.completed", EventRow.body.contains(goal_id)))


def test_concurrent_sweeps_complete_each_goal_once(replicas) -> None:  # noqa: F811
    c0, c1 = replicas
    start = utcnow()
    made = [learner_with_a_mastered_goal(c0) for _ in range(5)]
    learners = [lid for lid, _ in made]
    sweepers = [sweeper_of(replicas[i % 2], learners) for i in range(4)]
    reports, errors = race(4, lambda i: asyncio.run(sweepers[i].sweep(full=True)))
    assert errors == [], errors
    assert all(r.errors == 0 and r.done["curricula"] == len(learners) for r in reports), \
        [r.summary() for r in reports]
    for _, goal_id in made:
        goal = c1.curriculum_service.goal(goal_id)
        assert goal.status == GoalStatus.COMPLETED
        assert start - timedelta(minutes=6) <= goal.completed_at <= utcnow()  # an evidence time (D11)
        record = c1.curriculum_service.engine.record(curriculum_id_for(goal_id))
        assert record.status == curriculum_status_for(goal.status)
        assert completed_events(c1, goal_id) == 1  # transitions are reported once, whoever swept first
    before = [c1.curriculum_service.goal(g) for _, g in made]
    asyncio.run(sweeper_of(c1, learners).sweep(full=True))
    assert [c1.curriculum_service.goal(g) for _, g in made] == before
    assert all(completed_events(c1, g) == 1 for _, g in made)


def test_a_sweep_racing_the_learners_deletion_fails_nothing_and_leaves_nothing(replicas) -> None:  # noqa: F811
    c0, c1 = replicas
    for _ in range(3):
        lid, goal_id = learner_with_a_mastered_goal(c0)

        def act(i: int):
            if i == 0:
                return asyncio.run(sweeper_of(c0, [lid], batch=1).sweep(full=True))
            return c1.learner_deletion.delete(lid)

        results, errors = race(2, act)
        assert errors == [], errors
        report = next(r for r in results if hasattr(r, "errors"))
        assert report.errors == 0, report.summary()
        assert c1.learner_deletion.delete(lid).records == {}  # nothing was written after the deletion
        assert c1.learner_service._memory._repo.get(lid) is None


def test_get_next_action_writes_nothing_on_postgresql(replicas) -> None:  # noqa: F811
    c0, _ = replicas
    lid, _ = learner_with_a_mastered_goal(c0)
    bind = c0._sessions.kw["bind"]
    writes: list[str] = []

    def before(conn, cursor, statement, *args):
        if WRITE.match(statement) and not re.search(r"\b(rate_limit_\w+|stream_slots)\b", statement):
            writes.append(statement.split("\n")[0][:100])

    event.listen(bind, "before_cursor_execute", before)
    try:
        with client_for(c0, lid) as a:
            for params in ({}, {"as_of": (utcnow() + timedelta(days=90)).isoformat()}):
                r = a.get(f"/learners/{lid}/next-action", params=params)
                assert r.status_code == 200, r.text
    finally:
        event.remove(bind, "before_cursor_execute", before)
    assert writes == [], writes


def test_the_sweepers_id_queries_run_on_postgresql(replicas) -> None:  # noqa: F811
    """The three batch queries the container's sweeper uses (the race tests above stub the cycle and outbox ones)."""
    c0, _ = replicas
    lid, _ = learner_with_a_mastered_goal(c0)
    found, after = [], None
    while True:  # every learner with a curriculum, in id order, a batch at a time
        batch = c0.curriculum_service.engine.repository.learners_to_refresh(after, 500)
        found += batch
        if len(batch) < 500:
            break
        after = batch[-1]
    assert lid in found and found == sorted(found)
    for ids in (c0.learning_cycle_service.repository.active_ids(None, 50),
                c0.teaching_service.repository.with_pending_outbox(None, 50)):
        assert isinstance(ids, list) and ids == sorted(ids) and len(ids) <= 50
