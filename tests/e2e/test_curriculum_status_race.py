"""Issue #65: a goal's curriculum status always follows the goal as stored, whatever writer commits last.

Each test forces one interleaving deterministically: a hook runs the competing writer (through its real service
path, on another thread so it commits on its own) at the exact point where the first writer holds a copy of the goal
that is about to go stale. Before the fix, every one of these left a CANCELLED goal whose curriculum said ACTIVE,
permanently (later refreshes do not correct it)."""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta

import pytest

from app.schemas.common import utcnow
from app.schemas.curriculum import CurriculumStatus, GoalInput, GoalUpdate, curriculum_id_for
from app.schemas.learner import GoalStatus
from tests.e2e.test_class_assignments_api import master, world  # noqa: F401 - `world` is a fixture
from tests.learning_cycle_fixtures import LEARNER


def run_now(make) -> None:
    """Run the coroutine `make()` to completion on its own thread (its own transactions), and re-raise its error."""
    failure: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(make())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            failure.append(exc)

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if failure:
        raise failure[0]


def status_of(c, goal_id: str) -> CurriculumStatus:
    return c.curriculum_service.engine.record(curriculum_id_for(goal_id)).status


async def own_mastered_goal(world):  # noqa: F811 - the fixture
    """The learner's own (unassigned) goal, planned and with mastered objectives: its next refresh completes it."""
    c = world.c
    created, _ = await c.curriculum_service.create_goal(LEARNER, GoalInput(
        title="Race", description="Issue #65", domain="spanish", target_concepts=["es.preterite"], priority=2))
    await master(world, created.goal_id, utcnow(), tag="race")
    assert status_of(c, created.goal_id) == CurriculumStatus.ACTIVE
    return created


async def test_a_cancel_after_the_refresh_saw_the_goal_paused_leaves_the_curriculum_archived(world) -> None:  # noqa: F811
    """The tracker's "paused or cancelled meanwhile" branch: the refresh's `complete()` sees the goal PAUSED, a cancel
    commits (goal CANCELLED, curriculum ARCHIVED), then the refresh writes the curriculum status."""
    c = world.c
    goal = await own_mastered_goal(world)
    tracker = c.curriculum_service._tracker
    complete = tracker.complete
    raced = []

    def paused_then_cancelled(g, version, as_of):
        if g.goal_id != goal.goal_id:
            return complete(g, version, as_of)
        run_now(lambda: c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=GoalStatus.PAUSED),
                                                         user_id="test"))
        seen = complete(g, version, as_of)  # PAUSED: not completed
        assert seen.status == GoalStatus.PAUSED
        run_now(lambda: c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=GoalStatus.CANCELLED),
                                                         user_id="test"))
        assert status_of(c, goal.goal_id) == CurriculumStatus.ARCHIVED  # the cancel's own write
        raced.append(seen)
        return seen  # the stale copy the refresh goes on with

    tracker.complete = paused_then_cancelled
    try:
        tracked = await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
    finally:
        tracker.complete = complete
    assert raced and goal.goal_id not in tracked.completed_now
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.CANCELLED
    assert status_of(c, goal.goal_id) == CurriculumStatus.ARCHIVED
    # and it stays so: a later refresh does not move it
    await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=10))
    assert status_of(c, goal.goal_id) == CurriculumStatus.ARCHIVED


async def test_the_completed_branch_follows_the_stored_goal_too(world) -> None:  # noqa: F811
    """The refresh completes the goal; a status change racing it never leaves the curriculum behind the goal."""
    c = world.c
    goal = await own_mastered_goal(world)
    tracked = await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
    assert goal.goal_id in tracked.completed_now
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.COMPLETED
    assert status_of(c, goal.goal_id) == CurriculumStatus.COMPLETED


async def test_two_racing_patches_leave_the_status_of_the_last_one(world) -> None:  # noqa: F811
    """`CurriculumService.update_goal`: PATCH A (pause) commits its goal change, PATCH B (cancel) runs to the end, then
    A writes the curriculum status. A must not write it from its own copy (PAUSED -> ACTIVE)."""
    c = world.c
    goal = await own_mastered_goal(world)
    memory = c.curriculum_service._memory
    update_goal = memory.update_goal
    raced = []

    def then_cancel(goal_id, change):
        stored = update_goal(goal_id, change)
        if goal_id == goal.goal_id and not raced:
            raced.append(stored)
            run_now(lambda: c.curriculum_service.update_goal(goal_id, GoalUpdate(status=GoalStatus.CANCELLED),
                                                             user_id="test"))
            assert status_of(c, goal_id) == CurriculumStatus.ARCHIVED
        return stored

    memory.update_goal = then_cancel
    try:
        paused, _ = await c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=GoalStatus.PAUSED),
                                                           user_id="test")
    finally:
        memory.update_goal = update_goal
    assert paused.status == GoalStatus.PAUSED and raced  # A's own view
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.CANCELLED
    assert status_of(c, goal.goal_id) == CurriculumStatus.ARCHIVED


async def test_an_assignment_cancel_racing_a_resume_leaves_the_curriculum_archived(world) -> None:  # noqa: F811
    """Assignment propagation: the learner resumes their paused assigned goal; between the resume's goal write and its
    curriculum write, the teacher cancels the assignment (goal CANCELLED, curriculum ARCHIVED)."""
    c = world.c
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
    goal = world.assigned_goal(LEARNER, aid)
    await master(world, goal.goal_id, utcnow(), tag="assigned")
    await c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=GoalStatus.PAUSED), user_id="test")
    classroom = c.classes.repository.get(world.class_id)
    memory = c.curriculum_service._memory
    update_goal = memory.update_goal
    raced = []

    def then_cancel_assignment(goal_id, change):
        stored = update_goal(goal_id, change)
        if goal_id == goal.goal_id and not raced:
            raced.append(stored)
            run_now(lambda: c.assignments.cancel(classroom, aid))
            assert status_of(c, goal_id) == CurriculumStatus.ARCHIVED
        return stored

    memory.update_goal = then_cancel_assignment
    try:
        resumed, _ = await c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=GoalStatus.ACTIVE),
                                                            user_id="test")
    finally:
        memory.update_goal = update_goal
    assert resumed.status == GoalStatus.ACTIVE and raced
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.CANCELLED
    assert status_of(c, goal.goal_id) == CurriculumStatus.ARCHIVED


async def test_a_follow_lost_after_the_completion_committed_is_repaired_by_the_next_refresh(world) -> None:  # noqa: F811
    """The refresh's worker dies (or loses its lease) after `complete()` committed the goal and before the curriculum
    followed it: the retried refresh no longer sees a completion (the goal is not active), and still repairs it."""
    c = world.c
    goal = await own_mastered_goal(world)
    tracker = c.curriculum_service._tracker
    complete = tracker.complete

    def dies(g, version, as_of):
        complete(g, version, as_of)
        raise RuntimeError("the worker died here")

    tracker.complete = dies
    try:
        with pytest.raises(RuntimeError):
            await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
    finally:
        tracker.complete = complete
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.COMPLETED
    assert status_of(c, goal.goal_id) == CurriculumStatus.ACTIVE  # the follow never ran
    await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=10))
    assert status_of(c, goal.goal_id) == CurriculumStatus.COMPLETED


async def test_a_cancel_before_the_first_plan_commits_leaves_the_new_curriculum_archived(world) -> None:  # noqa: F811
    """The goal is cancelled while its first plan is being made: the cancel's follow finds no curriculum yet, then the
    commit inserts the record. The commit follows the stored goal after the insert."""
    c = world.c
    goal, _ = await c.curriculum_service.create_goal(LEARNER, GoalInput(
        title="Race", description="Issue #65", domain="spanish", target_concepts=["es.preterite"], priority=2))
    engine = c.curriculum_service.engine
    commit = engine.commit

    def cancelled_first(plan, **kw):
        if plan.goal_id == goal.goal_id and plan.base_version is None:
            run_now(lambda: c.curriculum_service.update_goal(goal.goal_id, GoalUpdate(status=GoalStatus.CANCELLED),
                                                             user_id="test"))
            assert engine.record(curriculum_id_for(goal.goal_id)) is None
        return commit(plan, **kw)

    engine.commit = cancelled_first
    try:
        await c.curriculum_service.build_curriculum(goal.goal_id, user_id="test")
    finally:
        engine.commit = commit
    assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.CANCELLED
    assert status_of(c, goal.goal_id) == CurriculumStatus.ARCHIVED


async def test_a_lost_follow_of_a_learner_with_no_active_goal_is_repaired_by_the_sweeper(world) -> None:  # noqa: F811
    """Issue #66 PR 3: no GET repairs it any more, and the learner has no ACTIVE goal left; the sweeper still visits
    every learner with a curriculum and its refresh repairs the follow."""
    c = world.c
    goal = await own_mastered_goal(world)
    for other in c.curriculum_service.goals(LEARNER):
        if other.goal_id != goal.goal_id and other.status == GoalStatus.ACTIVE:
            await c.curriculum_service.update_goal(other.goal_id, GoalUpdate(status=GoalStatus.PAUSED), user_id="test")
    tracker = c.curriculum_service._tracker
    complete = tracker.complete

    def dies(g, version, as_of):
        complete(g, version, as_of)
        raise RuntimeError("the worker died here")

    tracker.complete = dies
    try:
        with pytest.raises(RuntimeError):
            await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
    finally:
        tracker.complete = complete
    assert all(g.status != GoalStatus.ACTIVE for g in c.curriculum_service.goals(LEARNER))
    assert status_of(c, goal.goal_id) == CurriculumStatus.ACTIVE  # the follow never ran
    report = await c.task_worker.sweep_learning(full=True)
    assert report.errors == 0, report.summary()
    assert status_of(c, goal.goal_id) == CurriculumStatus.COMPLETED
