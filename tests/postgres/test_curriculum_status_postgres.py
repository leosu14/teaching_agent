"""Issue #65 on PostgreSQL: a goal's curriculum status follows the goal as stored, when its writers really race.

Opt-in like the other PostgreSQL tests: skipped unless TA_TEST_POSTGRES_URL names a database the test may write to.
Two API replicas share the database.

- The issue's interleaving, forced: replica 1's refresh sees the goal PAUSED, replica 2 cancels it (goal CANCELLED,
  curriculum ARCHIVED), then replica 1 writes the curriculum status.
- Free races: a refresh that completes the goal, learner PATCHes that pause, resume and cancel it, and a teacher
  cancelling the assignment it came from start at once from a barrier. Every goal write is followed by a short pause
  before its writer goes on (`widened`), which opens the window between a goal's commit and its curriculum write
  wide enough that a writer working from its own copy of the goal loses the race in most rounds (it does on the code
  before the fix). Whatever order they commit in, the curriculum's status afterwards is the one the stored goal calls
  for (`curriculum_status_for`).

Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from datetime import timedelta

from app.schemas.common import new_id, utcnow
from app.schemas.curriculum import GoalInput, GoalUpdate, curriculum_id_for, curriculum_status_for
from app.schemas.learner import GoalStatus, LearnerProfileInput, LearningEvidence
from tests.postgres.test_class_assignments_postgres import (  # noqa: F401 - `replicas` is a fixture
    a_class,
    assign,
    goal_of,
    pytestmark,
    race,
    replicas,
)

ROUNDS = 6
WINDOW = 0.02  # seconds between a goal write's commit and its writer going on (`widened`)


def mastered(c, learner_id: str, goal_id: str) -> None:
    """Plan the goal and record enough correct evidence to master it: the next refresh completes it."""
    task = asyncio.run(c.curriculum_service.build_curriculum(goal_id, user_id="test"))
    assert task.status.value == "COMPLETED", task.errors
    start = utcnow() - timedelta(minutes=5)
    for concept in ("es.present", "es.preterite"):
        asyncio.run(c.learner_service.record_evidence(learner_id, "spanish", [LearningEvidence(
            evidence_id=LearningEvidence.id_for(learner_id, "exercise", f"m-{concept}-{i}", concept),
            learner_id=learner_id, concept_id=concept, source_type="exercise", source_ref=f"m-{concept}-{i}",
            correctness="correct", score=1.0, difficulty=0.7, timestamp=start + timedelta(seconds=i))
            for i in range(6)]))


def consistent(c, goal_id: str) -> tuple[GoalStatus, object]:
    goal = c.curriculum_service.goal(goal_id)
    record = c.curriculum_service.engine.record(curriculum_id_for(goal_id))
    assert record.status == curriculum_status_for(goal.status), (goal.status, record.status)
    return goal.status, record.status


@contextlib.contextmanager
def widened(containers):
    """Each replica's goal writes pause for WINDOW after they commit, before their writer goes on."""
    saved = []
    for c in containers:
        memory = c.curriculum_service._memory
        update_goal = memory.update_goal

        def slow(goal_id, change, _update=update_goal):
            stored = _update(goal_id, change)
            time.sleep(WINDOW)
            return stored

        saved.append((memory, update_goal))
        memory.update_goal = slow
    try:
        yield
    finally:
        for memory, update_goal in saved:
            memory.update_goal = update_goal


def on_thread(fn) -> None:
    failure = []

    def run() -> None:
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            failure.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if failure:
        raise failure[0]


def test_a_cancel_on_another_replica_after_the_refresh_saw_the_goal_paused_archives_it(replicas) -> None:  # noqa: F811
    c0, c1 = replicas
    lid = new_id("l")
    c0.learner_service.upsert(lid, LearnerProfileInput(display_name="Racer"))
    goal, _ = asyncio.run(c0.curriculum_service.create_goal(lid, GoalInput(
        title="Race", domain="spanish", target_concepts=["es.preterite"])))
    mastered(c0, lid, goal.goal_id)
    tracker = c0.curriculum_service._tracker
    complete = tracker.complete

    def paused_then_cancelled(g, version, as_of):
        on_thread(lambda: asyncio.run(c1.curriculum_service.update_goal(
            g.goal_id, GoalUpdate(status=GoalStatus.PAUSED), user_id="test")))
        seen = complete(g, version, as_of)
        assert seen.status == GoalStatus.PAUSED
        on_thread(lambda: asyncio.run(c1.curriculum_service.update_goal(
            g.goal_id, GoalUpdate(status=GoalStatus.CANCELLED), user_id="test")))
        return seen

    tracker.complete = paused_then_cancelled
    try:
        asyncio.run(c0.curriculum_service.tracked(lid, as_of=utcnow()))
    finally:
        tracker.complete = complete
    assert consistent(c1, goal.goal_id) == (GoalStatus.CANCELLED, curriculum_status_for(GoalStatus.CANCELLED))


def test_racing_refresh_and_status_changes_leave_the_curriculum_following_the_goal(replicas) -> None:  # noqa: F811
    seen = set()
    for n in range(ROUNDS):
        c0 = replicas[0]
        lid = new_id("l")
        c0.learner_service.upsert(lid, LearnerProfileInput(display_name="Racer"))
        goal, _ = asyncio.run(c0.curriculum_service.create_goal(lid, GoalInput(
            title="Race", domain="spanish", target_concepts=["es.preterite"])))
        mastered(c0, lid, goal.goal_id)
        changes = [GoalStatus.PAUSED, GoalStatus.CANCELLED, GoalStatus.ACTIVE, GoalStatus.PAUSED]
        if n % 2:
            changes.reverse()

        def act(i: int):
            c = replicas[i % 2]
            if i == 0:
                return asyncio.run(c.curriculum_service.tracked(lid, as_of=utcnow()))
            return asyncio.run(c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=changes[i - 1]),
                                                                user_id="test"))

        with widened(replicas):
            _, errors = race(len(changes) + 1, act)
        assert errors == [], errors
        seen.add(consistent(replicas[1], goal.goal_id))
    assert seen  # the outcome varies with the commit order; each one is consistent


def test_an_assignment_cancel_racing_the_learner_and_a_refresh_archives_the_curriculum(replicas) -> None:  # noqa: F811
    for _ in range(ROUNDS // 2):
        classroom, [lid], _ = a_class(replicas[0], learners=1)
        aid = assign(replicas[0], classroom)
        goal_id = goal_of(replicas[0], aid, lid).goal_id
        mastered(replicas[0], lid, goal_id)
        asyncio.run(replicas[0].curriculum_service.update_goal(goal_id, GoalUpdate(status=GoalStatus.PAUSED),
                                                               user_id="test"))

        def act(i: int):
            c = replicas[i % 2]
            if i == 0:
                return asyncio.run(c.assignments.cancel(classroom, aid))
            if i == 1:
                return asyncio.run(c.curriculum_service.tracked(lid, as_of=utcnow()))
            status = GoalStatus.ACTIVE if i % 2 else GoalStatus.PAUSED
            return asyncio.run(c.curriculum_service.update_goal(goal_id, GoalUpdate(status=status), user_id="test"))

        with widened(replicas):
            _, errors = race(6, act)
        # a learner's change after the cancel is refused (the goal is CANCELLED: D13); nothing else fails
        assert all(type(e).__name__ == "GoalLocked" for e in errors), errors
        # Which status the goal ends in depends on the commit order (and on the D13 lock, which is not #65's);
        # the curriculum follows it whichever it is.
        consistent(replicas[1], goal_id)


def test_an_assignment_cancel_racing_the_learners_deletion_does_not_deadlock(replicas) -> None:  # noqa: F811
    """The cancel's follow locks the goal row, then the curriculum row; a deletion deletes curricula, then goals. The
    follow runs under the learner's shared deletion lock (`learner_writes`), so the two take turns. Forced: the follow
    holds the goal lock and waits until the deletion has reached the curricula (or is blocked before it)."""
    import inspect

    from sqlalchemy import event

    import app.storage.repositories as repositories

    c0, c1 = replicas
    classroom, [lid], _ = a_class(c0, learners=1)
    aid = assign(c0, classroom)
    goal_id = goal_of(c0, aid, lid).goal_id
    assert asyncio.run(c0.curriculum_service.build_curriculum(goal_id, user_id="test")).status.value == "COMPLETED"
    locked, curricula_deleted = threading.Event(), threading.Event()
    lock_goal = repositories._lock_goal

    def held(s, gid):
        row = lock_goal(s, gid)
        if inspect.stack()[1].function == "follow_goal" and threading.current_thread().name == "cancel":
            locked.set()
            curricula_deleted.wait(2)  # the deletion reaches the curricula only if it is not kept out
            time.sleep(0.05)
        return row

    bind = c1.learner_deletion.repository._sessions.kw["bind"]

    def before(conn, cursor, statement, *args):
        if threading.current_thread().name == "delete" and statement.startswith("DELETE FROM learning_goals"):
            time.sleep(0.4)  # the follow is waiting on the curriculum row by now

    def after(conn, cursor, statement, *args):
        if threading.current_thread().name == "delete" and statement.startswith("DELETE FROM curricula"):
            curricula_deleted.set()

    outcome = {}

    def cancel() -> None:
        try:
            asyncio.run(c0.assignments.cancel(classroom, aid))
            outcome["cancel"] = "ok"
        except Exception as exc:  # noqa: BLE001 - asserted on
            outcome["cancel"] = repr(exc)[:300]

    def delete() -> None:
        assert locked.wait(10)
        try:
            c1.learner_deletion.delete(lid)
            outcome["delete"] = "ok"
        except Exception as exc:  # noqa: BLE001 - asserted on
            outcome["delete"] = repr(exc)[:300]

    repositories._lock_goal = held
    event.listen(bind, "before_cursor_execute", before)
    event.listen(bind, "after_cursor_execute", after)
    try:
        threads = [threading.Thread(target=cancel, name="cancel"), threading.Thread(target=delete, name="delete")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        repositories._lock_goal = lock_goal
        event.remove(bind, "before_cursor_execute", before)
        event.remove(bind, "after_cursor_execute", after)
    assert outcome == {"cancel": "ok", "delete": "ok"}, outcome
