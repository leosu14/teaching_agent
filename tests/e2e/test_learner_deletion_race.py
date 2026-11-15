"""MVP item 9: a learner's data deletion racing a request that writes for them (SQLite; tests/postgres has the same
races on PostgreSQL with real row locks).

Grading an answer (`POST /assessment-items/{id}/attempts`, `POST /teaching-sessions/{id}/answers`) runs inside the
request, with no task lease or job for the deletion's busy check to see, and makes many writes: the item, the attempt
and its grade, their artifacts (files in the object store) and events, the evidence and the learner's mastery. Every
one of them checks, in its own transaction, that the learner and the lesson still exist, and locks their rows in the
order the deletion does (app/storage/repositories.py, `_guard`). Both orders are forced here:

- the deletion commits first (while the answer is being graded, or while it holds its locks and the grading's next
  write waits for them): every later write is refused, the request answers the uniform 404, nothing is left;
- the grading's write holds the locks first: the deletion waits for it to commit, then deletes it with the rest.

Either way nothing of the learner is left: no row in any erased table names any id that led back to them (scanned
column by column, not only the rows the deletion knows about), and no stored file holds their answer. A lesson
queued at the same moment is either seen by the deletion (409, nothing deleted: no row and no file) or refused after
it (404); the deletion never reports "nothing deleted" after deleting files."""

from __future__ import annotations

import sys
import threading

import pytest

from app.storage import repositories
from tests.auth import client_for
from tests.deletion_residue import db_residue, learner_ids, stored_files
from tests.e2e import test_security_api
from tests.e2e.test_learner_deletion_api import stored_objects
from tests.learning_cycle_fixtures import LEARNER

world = test_security_api.world  # the security matrix's fixture: A mid-cycle with every kind of resource

ANSWER = "hablé-race-answer"  # the answer text: no stored file may hold it after the deletion
NOT_FOUND = {"error": "NotFound", "detail": "not found"}
BLOCKED = 0.3  # how long the other side is left waiting on the locks before they are released


class Background(threading.Thread):
    """Run `call` in a thread of its own (a second request, sent while this one holds locks)."""

    def __init__(self, call) -> None:
        super().__init__(daemon=True)
        self._call, self.result, self.error = call, None, None
        self.start()

    def run(self) -> None:
        try:
            self.result = self._call()
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            self.error = exc

    def outcome(self):
        self.join(timeout=30)
        assert not self.is_alive(), "the other request never finished"
        if self.error is not None:
            raise self.error
        return self.result


def while_locked(monkeypatch, start, other) -> None:
    """Inside the deletion's transaction, once it holds its locks and has checked for work in progress: `start` the
    other request, and leave it `BLOCKED` seconds to reach its first write, which must wait for the locks."""
    real = repositories.SqlLearnerDataRepository._refuse_busy

    def refuse_busy(self, s, learner_id) -> None:
        real(self, s, learner_id)
        start()
        other().join(BLOCKED)
        assert other().is_alive(), (other().result, other().error, "the other request did not wait for the locks")

    monkeypatch.setattr(repositories.SqlLearnerDataRepository, "_refuse_busy", refuse_busy)


def residue(c, ids: set[str]) -> tuple[list, list[str]]:
    """What is left of the learner: rows naming any of their ids, and stored files holding the answer or under a key
    of one of their tasks or cycle."""
    rows = sorted(set(db_residue(c._sessions, ids)))
    files = sorted(k for k, data in stored_files(c).items()
                   if ANSWER.encode() in data or any(k.startswith(f"{i}/") for i in ids))
    return rows, files


def attempt(client, attempt_id: str = "race-attempt"):
    return client.post("/assessment-items/sec-item/attempts",
                       json={"learner_id": LEARNER, "answer": ANSWER, "attempt_id": attempt_id})


@pytest.fixture
def a_ids(world):
    env, ids, _ = world
    return learner_ids(env.container._sessions, LEARNER) | {"race-attempt"}


def test_a_deletion_while_an_answer_is_graded_leaves_nothing(world, a_ids) -> None:
    """The review's case: the deletion runs while the grader is working (between the request's writes)."""
    env, ids, auth = world
    c = env.container
    engine, receipts = c.assessment_service.engine, []
    real = engine.grade

    async def grade_after_a_deletion(*args, **kwargs):
        if not receipts:
            receipts.append(c.learner_deletion.delete(LEARNER))  # the other tab's DELETE, mid-grading
        return await real(*args, **kwargs)

    engine.grade = grade_after_a_deletion
    with client_for(c, LEARNER, auth=auth) as a:
        r = attempt(a)
        assert r.status_code == 404 and r.json() == NOT_FOUND, r.text  # every later write was refused
        assert receipts[0].records["learners"] == 1 and receipts[0].objects_deleted > 0
        assert a.get(f"/learners/{LEARNER}").status_code == 404  # the profile was not re-created
        again = a.delete(f"/learners/{LEARNER}")
        assert again.status_code == 200 and again.json()["records"] == {}, again.text  # nothing was written after
    assert residue(c, a_ids) == ([], [])
    assert not c.learner_deletion.repository.holds_data(LEARNER)


def test_a_deletion_while_a_teaching_answer_is_graded_leaves_nothing(world, a_ids) -> None:
    env, ids, auth = world
    c = env.container
    engine, receipts = c.assessment_service.engine, []
    real = engine.grade

    async def grade_after_a_deletion(*args, **kwargs):
        if not receipts:
            receipts.append(c.learner_deletion.delete(LEARNER))
        return await real(*args, **kwargs)

    engine.grade = grade_after_a_deletion
    with client_for(c, LEARNER, auth=auth) as a:
        r = a.post(f"/teaching-sessions/{ids['session']}/answers", json={"answer": ANSWER, "client_turn_id": "race1"})
        assert receipts, "the answer was not graded"
        assert r.status_code == 404 and r.json() == NOT_FOUND, r.text
        assert a.delete(f"/learners/{LEARNER}").json()["records"] == {}
    assert residue(c, a_ids) == ([], [])


def test_a_grading_write_refused_after_the_deletion_committed_while_it_waited(world, a_ids, monkeypatch) -> None:
    """The deletion holds its locks; the answer arrives, is graded, and its first write (the item) waits for them;
    once the deletion commits, the write finds the lesson gone and is refused."""
    env, ids, auth = world
    c = env.container
    grading = []
    while_locked(monkeypatch, lambda: grading.append(Background(lambda: _attempt_in_a_tab(c, auth))),
                 lambda: grading[0])
    with client_for(c, LEARNER, auth=auth) as a:
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 200 and r.json()["records"]["assessment_attempts"] == 1, r.text  # sec-attempt only
    status, body = grading[0].outcome()
    assert status == 404 and body == NOT_FOUND
    assert residue(c, a_ids) == ([], [])


def test_a_deletion_waits_for_a_grading_write_in_progress_and_then_deletes_it(world, a_ids, monkeypatch) -> None:
    """The grading's attempt-and-grade write holds the locks; the deletion arrives and waits; it then deletes the
    attempt with the rest, and the grading's next write is refused."""
    env, ids, auth = world
    c = env.container
    deletion = []
    real = repositories._guard  # (the deletion makes no guarded write: every call here is the grading's)

    def guard(s) -> None:
        if deletion:  # the grading's next write, once the deletion went ahead
            deletion[0].outcome()
        real(s)
        if not deletion and sys._getframe(1).f_code.co_name == "add_attempt":
            deletion.append(Background(lambda: c.learner_deletion.delete(LEARNER)))
            deletion[0].join(BLOCKED)
            assert deletion[0].is_alive(), "the deletion did not wait for the grading's write"

    monkeypatch.setattr(repositories, "_guard", guard)
    with client_for(c, LEARNER, auth=auth) as a:
        r = attempt(a)
        assert r.status_code == 404 and r.json() == NOT_FOUND, r.text
    receipt = deletion[0].outcome()
    assert receipt.records["assessment_attempts"] == 2  # sec-attempt and the race attempt it waited for
    assert receipt.records["assessment_grades"] == 2
    assert residue(c, a_ids) == ([], [])  # including any file the refused writes had put


def test_a_grading_that_finishes_first_is_deleted_with_the_rest(world, a_ids) -> None:
    env, ids, auth = world
    c = env.container
    with client_for(c, LEARNER, auth=auth) as a:
        assert attempt(a).status_code == 201
        assert any(ANSWER.encode() in data for data in stored_files(c).values())  # a grade file holds the answer
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 200 and r.json()["records"]["assessment_attempts"] == 2, r.text
    assert residue(c, a_ids) == ([], [])


def test_a_lesson_queued_during_the_deletion_waits_and_is_refused(world, a_ids, monkeypatch) -> None:
    env, ids, auth = world
    c = env.container
    queued = []
    while_locked(monkeypatch, lambda: queued.append(Background(lambda: _post_in_a_tab(
        c, auth, "/tasks", {"request": "Teach me Spanish", "learner_id": LEARNER}))), lambda: queued[0])
    with client_for(c, LEARNER, auth=auth) as a:
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 200 and r.json()["objects_deleted"] > 0, r.text
    status, body = queued[0].outcome()
    assert status == 404 and body == NOT_FOUND
    assert residue(c, a_ids) == ([], []) and not c.learner_deletion.repository.holds_data(LEARNER)


def test_a_lesson_queued_just_before_the_deletion_blocks_it_and_nothing_is_deleted(world) -> None:
    """The review's second case (a lesson queued between the deletion's file step and its row step) cannot happen
    any more: the check, the rows and the files are one transaction. A 409 means no file was deleted either."""
    env, ids, auth = world
    c = env.container
    objects = stored_objects(env, ids)
    deleted = []
    real = c.artifacts.delete_object
    c.artifacts.delete_object = lambda uri: (deleted.append(uri), real(uri))
    with client_for(c, LEARNER, auth=auth) as a:
        assert a.post("/tasks", json={"request": "Teach me Spanish", "learner_id": LEARNER}).status_code == 202
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 409 and r.json()["error"] == "LearnerDataBusy", r.text
        assert deleted == [] and all(c.artifacts.object_exists(u) for u in objects)
        assert a.get(f"/lessons/{ids['lesson']}/reading").status_code == 200  # the lessons still serve


def test_events_naming_the_learners_goal_or_curriculum_are_deleted(world, a_ids) -> None:
    """The review's third case: events outside any task (objective started, learning action selected) name the
    learner's goal and curriculum but not the learner."""
    env, ids, auth = world
    c = env.container
    with c._sessions() as s:
        from sqlalchemy import select

        from app.storage.orm import EventRow
        goal_events = [t for t, body in s.execute(select(EventRow.type, EventRow.body).where(EventRow.task_id.is_(None)))
                       if ids["goal"] in body and LEARNER not in body]
    assert goal_events  # the world has them
    with client_for(c, LEARNER, auth=auth) as a:
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 200, r.text
    assert residue(c, a_ids) == ([], [])


def _attempt_in_a_tab(c, auth):
    with client_for(c, LEARNER, auth=auth) as tab:
        r = attempt(tab)
        return r.status_code, r.json()


def _post_in_a_tab(c, auth, path: str, body: dict):
    with client_for(c, LEARNER, auth=auth) as tab:
        r = tab.post(path, json=body)
        return r.status_code, r.json()
