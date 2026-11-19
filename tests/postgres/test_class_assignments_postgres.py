"""Class assignments on PostgreSQL (Important item 3, #62). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

SQLite serializes writers, so it cannot show what happens when they really race. Here, on two API replicas sharing
the database: racing deadline changes leave every learner's goal at the assignment's final deadline (none at a stale
one); a learner joining while the deadline changes ends at the final deadline too; a cancellation racing a change
leaves every open goal cancelled; a refresh completing a goal racing a deadline change never loses the completion;
and a learner deleted while a change is carried to the class leaves nothing of theirs behind, while everyone else
converges. Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.assignment import AssignmentConflict, CreateAssignment, UpdateAssignment
from app.schemas.common import new_id, utcnow
from app.schemas.learner import GoalStatus, LearnerProfileInput
from app.schemas.lease import LearnerDataGone
from app.services.container import build_container
from app.storage.orm import AssignmentLearnerRow, LearningGoalRow
from scripts.run_learning_cycle_demo import settings

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL assignment tests")
THREADS = 6
LEARNERS = 4


def replica(data_dir):
    return build_container(settings(data_dir, database_url=URL, log_json=False),
                           llm_providers={"mock": MockLLMProvider(default_responders())})


@pytest.fixture
def replicas(tmp_path):
    made = [replica(tmp_path / "r1"), replica(tmp_path / "r2")]
    yield made
    for c in made:
        c.close()


def race(n: int, fn) -> tuple[list, list]:
    results, errors = [], []
    start = threading.Barrier(n)

    def run(i: int) -> None:
        try:
            start.wait()
            results.append(fn(i))
        except Exception as exc:  # noqa: BLE001 - collected and asserted on
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def a_class(c, learners: int = LEARNERS):
    """A class of a fresh teacher with `learners` fresh learners on its roster; (classroom, learner ids, code)."""
    made = c.classes.create(teacher_id=new_id("t"), tenant_id=None, name="Spanish B1")
    ids = []
    for i in range(learners):
        lid = new_id("l")
        c.learner_service.upsert(lid, LearnerProfileInput(display_name=f"L{i}"))
        c.classes.join(lid, made.join_code, tenant_id=None)
        ids.append(lid)
    return c.classes.repository.get(made.class_id), ids, made.join_code


def assign(c, classroom, due=None):
    view, _ = asyncio.run(c.assignments.create(classroom, CreateAssignment(
        title="The preterite", domain="spanish", topics=["preterite"], due_at=due or utcnow() + timedelta(days=7)),
        teacher_id=classroom.teacher_id))
    return view.assignment_id


def goal_of(c, assignment_id: str, learner_id: str):
    record = c.assignments.repository.learner(assignment_id, learner_id)
    return c.curriculum_service.goal(record.goal_id)


def test_racing_deadline_changes_leave_every_goal_at_the_final_deadline(replicas) -> None:
    for _ in range(3):
        classroom, learners, _ = a_class(replicas[0])
        aid = assign(replicas[0], classroom)
        base = utcnow() + timedelta(days=10)
        dues = {i: base + timedelta(hours=i) for i in range(THREADS)}
        _, errors = race(THREADS, lambda i: asyncio.run(replicas[i % 2].assignments.update(
            classroom, aid, UpdateAssignment(due_at=dues[i], title=f"Title {i}"))))
        assert errors == []
        final = replicas[1].assignments.repository.get(aid)
        assert final.version == 1 + THREADS  # every change was applied, one after another
        assert final.due_at in dues.values()
        for lid in learners:
            goal = goal_of(replicas[0], aid, lid)
            assert (goal.target_date, goal.title) == (final.due_at, final.title)  # none left at a stale value


def test_a_learner_joining_while_the_deadline_changes_gets_the_final_deadline(replicas) -> None:
    for _ in range(3):
        classroom, _, code = a_class(replicas[0], learners=1)
        aid = assign(replicas[0], classroom)
        joiners = []
        for _ in range(THREADS // 2):
            lid = new_id("l")
            replicas[0].learner_service.upsert(lid, LearnerProfileInput(display_name="late"))
            joiners.append(lid)
        new_due = utcnow() + timedelta(days=30)

        def act(i: int):
            c = replicas[i % 2]
            if i < len(joiners):
                c.classes.join(joiners[i], code, tenant_id=None)
                return asyncio.run(c.assignments.on_join(classroom.class_id, joiners[i]))
            return asyncio.run(c.assignments.update(classroom, aid, UpdateAssignment(due_at=new_due)))

        _, errors = race(len(joiners) + 1, act)
        assert errors == []
        for lid in joiners:
            assert goal_of(replicas[1], aid, lid).target_date == new_due


def test_a_cancellation_racing_a_change_leaves_every_open_goal_cancelled(replicas) -> None:
    for _ in range(3):
        classroom, learners, _ = a_class(replicas[0])
        aid = assign(replicas[0], classroom)

        def act(i: int):
            c = replicas[i % 2]
            if i == 0:
                return asyncio.run(c.assignments.cancel(classroom, aid))
            return asyncio.run(c.assignments.update(classroom, aid, UpdateAssignment(
                due_at=utcnow() + timedelta(days=i))))

        _, errors = race(THREADS, act)
        assert all(isinstance(e, AssignmentConflict) for e in errors), errors  # a change after the cancel: 409
        assert replicas[1].assignments.repository.get(aid).status.value == "CANCELLED"
        for lid in learners:
            assert goal_of(replicas[0], aid, lid).status == GoalStatus.CANCELLED


def test_a_completion_racing_a_deadline_change_is_never_lost(replicas) -> None:
    for _ in range(5):
        classroom, learners, _ = a_class(replicas[0], learners=1)
        aid = assign(replicas[0], classroom)
        [lid] = learners
        goal_id = goal_of(replicas[0], aid, lid).goal_id
        at = utcnow()

        def complete(g):
            if g.status != GoalStatus.ACTIVE:
                return None
            return g.model_copy(update={"status": GoalStatus.COMPLETED, "completed_at": at})

        def act(i: int):
            if i == 0:
                return replicas[0].memory.update_goal(goal_id, complete)  # what a refresh's completion writes
            return asyncio.run(replicas[i % 2].assignments.update(classroom, aid, UpdateAssignment(
                due_at=utcnow() + timedelta(days=i))))

        _, errors = race(THREADS, act)
        assert errors == []
        stored = replicas[1].curriculum_service.goal(goal_id)
        assert stored.status == GoalStatus.COMPLETED and stored.completed_at == at  # never overwritten


def test_a_learner_deleted_while_a_change_is_carried_to_the_class_leaves_nothing(replicas) -> None:
    for _ in range(3):
        classroom, learners, _ = a_class(replicas[0])
        aid = assign(replicas[0], classroom)
        doomed = learners[:2]
        new_due = utcnow() + timedelta(days=20)

        def act(i: int):
            c = replicas[i % 2]
            if i < len(doomed):
                try:
                    return c.learner_deletion.delete(doomed[i])
                except LearnerDataGone:
                    return None
            if i == len(doomed):
                return asyncio.run(c.assignments.update(classroom, aid, UpdateAssignment(due_at=new_due)))
            return asyncio.run(c.assignments.update(classroom, aid, UpdateAssignment(title=f"T{i}")))

        _, errors = race(len(doomed) + 2, act)
        assert errors == []
        with replicas[0]._sessions() as s:
            for lid in doomed:
                assert s.scalars(select(AssignmentLearnerRow).where(AssignmentLearnerRow.learner_id == lid)).all() == []
                assert s.scalars(select(LearningGoalRow).where(LearningGoalRow.learner_id == lid)).all() == []
        for lid in learners[2:]:
            assert goal_of(replicas[1], aid, lid).target_date == new_due
        assert replicas[0].assignments.repository.get(aid) is not None  # the teacher's assignment stays
