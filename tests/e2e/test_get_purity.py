"""Issue #66 PR 3: GET routes write nothing, `as_of` is a read-only what-if, and the writes the reads used to make
happen in their writers and the sweeper.

- Every GET route of the API (the security matrix's list, so a new route is covered by construction), as the learner
  and as the class teacher, including both progress streams, in a state with work pending (a teaching answer whose
  outbox a crash left unpublished, so the session's events, artifacts and finalize are owed and the cycle has not
  stored what its session did) and with a learner `as_of` far ahead. No INSERT, UPDATE or DELETE reaches any table
  but the bookkeeping ones decision A3 allows (rate limits and stream slots).
- The pending work is still there after the reads, and the sweeper (the writer that owns it now) does it, once.
- A goal whose mastery holds is completed by the evidence write that made it hold (an assessment attempt, a teacher's
  grade override), not by a GET; D11's `completed_at` is the evidence time whichever writer records it."""

from __future__ import annotations

import asyncio
import dataclasses
import re
from datetime import timedelta

import pytest
from sqlalchemy import event, func, select

from app.api.auth import TEACHER
from app.schemas.common import utcnow
from app.schemas.curriculum import curriculum_id_for
from app.schemas.learner import GoalStatus
from app.schemas.teaching import LearnerInput
from app.services.teaching import TeachingSessionService
from app.storage.orm import EventRow
from tests.auth import client_for
from tests.e2e import test_class_dashboard_api, test_security_api
from tests.e2e.test_class_assignments_api import evidence
from tests.e2e.test_security_api import TEACHER_ID, routes
from tests.learning_cycle_fixtures import LEARNER, ScriptedLearner

world = test_security_api.world
board_world = test_class_dashboard_api.board_world

BOOKKEEPING = {"rate_limit_starts", "rate_limit_slots", "stream_slots"}  # decision A3
WRITE = re.compile(r"\s*(INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM)\s+\"?(\w+)", re.I)


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing catches it."""


class Writes:
    """Every write statement on the container's database while open, as (verb, table)."""

    def __init__(self, c) -> None:
        self.bind = c._sessions.kw["bind"]
        self.seen: list[tuple[str, str]] = []

    def __enter__(self) -> Writes:
        event.listen(self.bind, "before_cursor_execute", self._before)
        return self

    def __exit__(self, *exc) -> None:
        event.remove(self.bind, "before_cursor_execute", self._before)

    def _before(self, conn, cursor, statement, *args) -> None:
        m = WRITE.match(statement)
        if m:
            self.seen.append((m.group(1).split()[0].upper(), m.group(2)))

    def domain(self) -> list[tuple[str, str]]:
        return [w for w in self.seen if w[1] not in BOOKKEEPING]


def event_count(c) -> int:
    with c._sessions() as s:
        return s.scalar(select(func.count()).select_from(EventRow))


def teacher(auth) -> dict:
    return auth.headers(user_id=TEACHER_ID, roles=(TEACHER,))


async def leave_work_pending(env, ids, monkeypatch) -> None:
    """The learner answers the session's question directly, and the process dies after the answer committed, before
    its outbox was published: its events, artifacts and finalize are owed, and the cycle has not stored the turn."""
    c = env.container
    sid = ids["session"]
    view = await c.teaching_service.view(sid)
    assert view.state.waiting_question is not None

    async def dies(self, session, item):
        raise SimulatedCrash("publish")

    with monkeypatch.context() as m:
        m.setattr(TeachingSessionService, "_publish_item", dies)
        with pytest.raises(SimulatedCrash):
            await c.teaching_service.submit(sid, LearnerInput(
                answer=ScriptedLearner().answer(view.state.waiting_question.prompt), client_turn_id="purity-1"))
    assert c.teaching_service.repository.pending_outbox(sid)


def pending(c, ids) -> tuple[int, int]:
    """(unpublished outbox items of the session, the cycle's stored version)."""
    return (len(c.teaching_service.repository.pending_outbox(ids["session"])),
            c.learning_cycle_service.repository.get(ids["cycle"]).version)


def get_routes(ids: dict) -> list[str]:
    return [p for m, p, _ in routes(ids) if m == "GET"]


async def test_no_get_route_writes_outside_the_bookkeeping_tables(world, monkeypatch) -> None:
    env, ids, auth = world
    c = env.container
    await leave_work_pending(env, ids, monkeypatch)
    before = pending(c, ids)
    ahead = (utcnow() + timedelta(days=400)).isoformat()
    paths = get_routes(ids)
    assert f"/learners/{LEARNER}/next-action" in paths and f"/learning-cycles/{ids['cycle']}/stream" in paths
    paths += [f"/learners/{LEARNER}/next-action?as_of={ahead.replace('+', '%2B')}",  # A1: a what-if
              f"/classes/{ids['class']}/assignments/{ids['assignment']}?as_of={ahead.replace('+', '%2B')}"]
    answered = {}
    # A stream on a cycle whose work is owed stays open (it is in progress) until the writer or the sweeper does it:
    # end each one after a second, with a reconnect frame, so the round goes on.
    streams = c.progress_streams
    monkeypatch.setattr(streams, "limits", dataclasses.replace(streams.limits, max_seconds=1.0))
    with client_for(c, LEARNER, auth=auth) as a, Writes(c) as writes:
        for _ in range(2):  # a second round reads what the first one would have left
            for path in paths:
                for who, headers in (("learner", None), ("teacher", teacher(auth))):
                    r = a.get(path, headers=headers)
                    assert r.status_code < 500, (who, path, r.status_code, r.text)
                    answered[(who, path.split("?")[0])] = r.status_code
    assert writes.domain() == [], writes.domain()
    # the reads did answer (not only refuse): the learner's own data, the teacher's class pages and both streams
    assert answered[("learner", f"/learners/{LEARNER}/next-action")] == 200
    assert answered[("learner", f"/learning-cycles/{ids['cycle']}")] == 200
    assert answered[("learner", f"/learning-cycles/{ids['cycle']}/stream")] == 200
    assert answered[("learner", f"/tasks/{ids['task']}/stream")] == 200
    assert answered[("learner", f"/teaching-sessions/{ids['session']}")] == 200
    assert answered[("teacher", f"/classes/{ids['class']}/dashboard")] == 200
    assert answered[("teacher", f"/classes/{ids['class']}/assignments/{ids['assignment']}")] == 200
    assert pending(c, ids) == before  # the work the reads saw is still owed

    # the sweeper owes it: it publishes the outbox and stores the cycle's observation, once
    report = await c.task_worker.sweep_learning(full=True)
    assert report.errors == 0 and report.done["outboxes"] >= 1, report.summary()
    outbox, version = pending(c, ids)
    assert outbox == 0 and version > before[1]
    events = event_count(c)
    with Writes(c) as again:
        second = await c.task_worker.sweep_learning(full=True)
    assert second.errors == 0 and second.done["outboxes"] == 0
    assert pending(c, ids) == (0, version)  # idempotent: nothing more to do
    assert not [w for w in again.domain() if w[1] in ("events", "teaching_outbox", "learning_cycles", "artifacts")]
    assert event_count(c) == events


async def test_as_of_is_a_what_if_that_stores_nothing(world) -> None:
    """A1: next-action `as_of` far ahead (the reviews are due, the curriculum may want replanning) answers for that
    time and stores no progress snapshot, transition, replan, completion or event."""
    env, ids, auth = world
    c = env.container
    stored_goals = [g.model_dump_json() for g in c.curriculum_service.goals(LEARNER)]
    versions = {g.goal_id: len(c.curriculum_service.engine.versions(curriculum_id_for(g.goal_id)))
                for g in c.curriculum_service.goals(LEARNER)}
    with client_for(c, LEARNER, auth=auth) as a, Writes(c) as writes:
        for days in (0, 30, 400):
            when = (utcnow() + timedelta(days=days)).isoformat().replace("+", "%2B")
            r = a.get(f"/learners/{LEARNER}/next-action?as_of={when}")
            assert r.status_code == 200, r.text
    assert writes.domain() == [], writes.domain()
    assert [g.model_dump_json() for g in c.curriculum_service.goals(LEARNER)] == stored_goals
    assert {g.goal_id: len(c.curriculum_service.engine.versions(curriculum_id_for(g.goal_id)))
            for g in c.curriculum_service.goals(LEARNER)} == versions


async def mastered_without_a_refresh(c, goal_id: str, start) -> None:
    """The goal planned and its concepts mastered by evidence recorded without any refresh (the learner service's
    `record_evidence`, which scripts and tests use): only a writer's refresh can complete it now."""
    goal = c.curriculum_service.goal(goal_id)
    if c.curriculum_service.curriculum(goal_id) is None:
        task = await c.curriculum_service.build_curriculum(goal_id, user_id="test")
        assert task.status.value == "COMPLETED", task.errors
    for concept in ("es.present", "es.preterite"):
        await c.learner_service.record_evidence(goal.learner_id, "spanish",
                                                evidence(goal.learner_id, concept, start, 6, f"p-{concept}"))
    assert c.curriculum_service.goal(goal_id).status == GoalStatus.ACTIVE


def assigned_goal(c, ids):
    [goal] = [g for g in c.curriculum_service.goals(LEARNER) if g.metadata.get("assignment_id") == ids["assignment"]]
    return goal


async def test_a_get_never_completes_a_goal_and_the_evidence_write_does(world) -> None:
    """GET next-action and the assignment views show the completion (computed read-only, D11's time) and store
    nothing; the next evidence write, an assessment attempt, completes the assigned goal in that same request, with
    `completed_at` the time mastery was first reached, so not late."""
    env, ids, auth = world
    c = env.container
    goal = assigned_goal(c, ids)
    start = utcnow()
    await mastered_without_a_refresh(c, goal.goal_id, start)
    with client_for(c, LEARNER, auth=auth) as a:
        with Writes(c) as writes:
            assert a.get(f"/learners/{LEARNER}/next-action").status_code == 200
            mine = a.get(f"/learners/{LEARNER}/assignments").json()
            detail = a.get(f"/classes/{ids['class']}/assignments/{ids['assignment']}", headers=teacher(auth)).json()
        assert writes.domain() == [], writes.domain()
        [entry] = [x for x in mine if x["assignment_id"] == ids["assignment"]]
        assert entry["progress"]["status"] == "COMPLETED" and entry["progress"]["completed_late"] is False
        [row] = [x for x in detail["learners"] if x["learner_id"] == LEARNER]
        assert row["status"] == "COMPLETED"
        assert c.curriculum_service.goal(goal.goal_id).status == GoalStatus.ACTIVE  # shown, not stored
        r = a.post("/assessment-items/sec-item/attempts",
                   json={"learner_id": LEARNER, "answer": "hablé", "attempt_id": "purity-attempt"})
        assert r.status_code == 201, r.text
    stored = c.curriculum_service.goal(goal.goal_id)
    assert stored.status == GoalStatus.COMPLETED  # by the attempt's own request
    assert start <= stored.completed_at <= start + timedelta(seconds=6)  # D11: the evidence time, not the write's


async def test_a_grade_override_completes_the_goal_it_makes_mastered(board_world) -> None:
    """The teacher's override of an UNCERTAIN grade is an evidence write: it completes the goal in its request."""
    env, ids, auth = board_world
    c = env.container
    goal = assigned_goal(c, ids)
    start = utcnow()
    await mastered_without_a_refresh(c, goal.goal_id, start)
    with client_for(c, LEARNER, auth=auth) as a:
        r = a.post(f"/classes/{ids['class']}/attempts/{ids['uncertain_attempt']}/override",
                   json={"outcome": "CORRECT"}, headers=teacher(auth))
        assert r.status_code == 200, r.text
    stored = c.curriculum_service.goal(goal.goal_id)
    assert stored.status == GoalStatus.COMPLETED and start <= stored.completed_at <= start + timedelta(seconds=6)


async def test_the_sweeper_completes_what_no_writer_refreshed_with_d11s_time(world) -> None:
    """Evidence recorded without a refresh, then nothing for days: the sweeper's refresh completes the goal, at the
    time mastery was reached (not late), and a second pass changes nothing."""
    env, ids, auth = world
    c = env.container
    goal = assigned_goal(c, ids)
    start = utcnow()
    await mastered_without_a_refresh(c, goal.goal_id, start)
    report = await c.task_worker.sweep_learning(full=True)
    assert report.errors == 0 and report.done["curricula"] >= 1, report.summary()
    stored = c.curriculum_service.goal(goal.goal_id)
    assert stored.status == GoalStatus.COMPLETED and start <= stored.completed_at <= start + timedelta(seconds=6)
    await c.task_worker.sweep_learning(full=True)
    assert c.curriculum_service.goal(goal.goal_id) == stored
    with c._sessions() as s:
        reported = s.scalars(select(EventRow.body).where(EventRow.type == "goal.completed")).all()
    assert len([b for b in reported if goal.goal_id in b]) == 1  # reported once, by the refresh that stored it


def test_a_get_racing_nothing_is_still_pure_after_a_restart(world) -> None:
    """A new process on the same data (nothing cached): the first reads write nothing either."""
    env, ids, auth = world
    env = env.reopen()
    c = env.container
    with client_for(c, LEARNER, auth=auth) as a, Writes(c) as writes:
        for path in get_routes(ids):
            headers = teacher(auth) if path.startswith("/classes") else None
            assert a.get(path, headers=headers).status_code < 500, path
    assert writes.domain() == [], writes.domain()
    asyncio.run(c.task_worker.sweep_learning(full=True))
