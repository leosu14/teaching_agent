"""A learner's data deletion racing the requests that write for them, on PostgreSQL with real row locks (MVP item 9).
Opt-in like the other PostgreSQL tests: skipped unless TA_TEST_POSTGRES_URL names a database the test may write to.

Two API replicas share the database. The grading runs on one, the deletion on the other, each in a thread of its own.
Both orders are forced at the transaction level (tests/e2e/test_learner_deletion_race.py does the same on SQLite):

- the grading's attempt-and-grade write holds the lesson task and learner row locks: the deletion's `FOR UPDATE`
  waits for it, then deletes the attempt with the rest, and the grading's next write is refused (404);
- the deletion holds its locks: the grading's first write waits for them, then finds the lesson gone (404);
- a lesson queued while the deletion holds its locks waits, then finds the learner gone (404).

Then everything at once: answers, a lesson request and two deletions from both replicas released together. Whatever
the order, no request fails with a deadlock or an error, and once a deletion has gone through nothing of the learner
is left: no row in any erased table names any id that led back to them, and no stored file holds their answers."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from app.api.auth import AUTHOR
from app.schemas.common import new_id
from app.storage import repositories
from tests.auth import TestAuth, client_for
from tests.learning_cycle_fixtures import ScriptedLearner, answer
from tests.deletion_residue import db_residue, learner_ids, stored_files
from tests.postgres.test_learner_deletion_postgres import placed_learner, replica, start_learning

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL deletion tests")

NOT_FOUND = {"error": "NotFound", "detail": "not found"}
BLOCKED = 0.5  # how long the other side is left waiting on the locks before they are released


class Background(threading.Thread):
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
        self.join(timeout=60)
        assert not self.is_alive(), "the other request never finished (a lock wait that never ends)"
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def two_replicas(tmp_path):
    """(replica 1, replica 2, the learner, their lesson task, a registered item on it, auth): the learner mid-cycle,
    with a lesson, a teaching session and a goal with its curriculum."""
    first, second = replica(tmp_path / "data"), replica(tmp_path / "data")
    try:
        learner = placed_learner(first, new_id("pgrace"))
        auth = TestAuth()
        with client_for(first, learner, roles=(AUTHOR,), auth=auth) as a:
            view = start_learning(a, learner)
            cycles = first.learning_cycle_service
            # through the diagnostic, so that the lesson is completed and its teaching session waits for an answer
            asyncio.run(answer(SimpleNamespace(cycles=cycles), asyncio.run(cycles.get(view["cycle_id"])),
                               ScriptedLearner(prefix="d"), until="SESSION_QUESTION"))
            [cycle] = cycles.repository.for_learner(learner)
            lesson = next(s.child_id for s in cycle.steps if s.kind.value == "LESSON")
            item = new_id("item")
            registered = a.post("/assessment-items", json={"item": {
                "assessment_item_id": item, "lesson_id": lesson, "concept_id": "es.preterite",
                "prompt": "Ayer yo ___ (hablar) con María.", "expected_answer": "hablé", "response_type": "SHORT_TEXT",
                "language": "es"}})
            assert registered.status_code == 201, registered.text
        yield first, second, learner, lesson, item, auth
    finally:
        first.close()
        second.close()


def answer_text() -> str:
    return f"respuesta-{new_id('a')}"  # unique: no stored file may hold it after the deletion


def post_attempt(c, learner: str, item: str, answer: str, auth) -> tuple[int, dict]:
    with client_for(c, learner, auth=auth) as tab:
        r = tab.post(f"/assessment-items/{item}/attempts",
                     json={"learner_id": learner, "answer": answer, "attempt_id": new_id("att")})
        return r.status_code, r.json()


def delete_learner(c, learner: str, auth) -> tuple[int, dict]:
    with client_for(c, learner, auth=auth) as tab:
        r = tab.delete(f"/learners/{learner}")
        return r.status_code, r.json()


def nothing_left(c, ids: set[str], answers: list[str]) -> None:
    assert sorted(set(db_residue(c._sessions, ids))) == []
    files = stored_files(c)
    assert not [k for k, data in files.items() if any(a.encode() in data for a in answers)]
    assert not [k for k in files if any(k.startswith(f"{i}/") for i in ids)]
    assert not c.learner_deletion.repository.holds_data(next(i for i in ids if i.startswith("pgrace")))


def test_a_deletion_waits_for_the_grading_write_that_holds_the_locks(two_replicas, monkeypatch) -> None:
    first, second, learner, lesson, item, auth = two_replicas
    ids, answer, deletion = learner_ids(first._sessions, learner), answer_text(), []
    real = repositories._guard

    def guard(s) -> None:
        if deletion:  # the grading's next write, once the deletion went ahead
            deletion[0].outcome()
        real(s)
        if not deletion and sys._getframe(1).f_code.co_name == "add_attempt":  # its locks are held now
            deletion.append(Background(lambda: delete_learner(second, learner, auth)))
            deletion[0].join(BLOCKED)
            assert deletion[0].is_alive(), "the deletion did not wait for the grading's write"

    monkeypatch.setattr(repositories, "_guard", guard)
    status, body = post_attempt(first, learner, item, answer, auth)
    assert (status, body) == (404, NOT_FOUND)
    deleted_status, receipt = deletion[0].outcome()
    assert deleted_status == 200 and receipt["records"]["assessment_attempts"] == 1, receipt  # the one it waited for
    nothing_left(first, ids | {item}, [answer])


def test_the_grading_waits_for_the_deletion_that_holds_the_locks_and_is_refused(two_replicas, monkeypatch) -> None:
    first, second, learner, lesson, item, auth = two_replicas
    ids, answer, grading = learner_ids(first._sessions, learner), answer_text(), []
    real = repositories.SqlLearnerDataRepository._refuse_busy

    def refuse_busy(self, s, learner_id) -> None:
        real(self, s, learner_id)  # the deletion holds its locks now
        grading.append(Background(lambda: post_attempt(first, learner, item, answer, auth)))
        grading[0].join(BLOCKED)
        assert grading[0].is_alive(), (grading[0].result, grading[0].error, "the grading did not wait")

    monkeypatch.setattr(repositories.SqlLearnerDataRepository, "_refuse_busy", refuse_busy)
    status, receipt = delete_learner(second, learner, auth)
    assert status == 200 and "assessment_attempts" not in receipt["records"], receipt
    assert grading[0].outcome() == (404, NOT_FOUND)
    nothing_left(first, ids | {item}, [answer])


def test_a_lesson_queued_while_the_deletion_holds_the_locks_is_refused(two_replicas, monkeypatch) -> None:
    first, second, learner, lesson, item, auth = two_replicas
    ids, queued = learner_ids(first._sessions, learner), []
    real = repositories.SqlLearnerDataRepository._refuse_busy

    def queue(c):
        with client_for(c, learner, auth=auth) as tab:
            r = tab.post("/tasks", json={"request": "Teach me Spanish", "learner_id": learner})
            return r.status_code, r.json()

    def refuse_busy(self, s, learner_id) -> None:
        real(self, s, learner_id)
        queued.append(Background(lambda: queue(first)))
        queued[0].join(BLOCKED)
        assert queued[0].is_alive(), (queued[0].result, queued[0].error, "the lesson was queued past the locks")

    monkeypatch.setattr(repositories.SqlLearnerDataRepository, "_refuse_busy", refuse_busy)
    status, receipt = delete_learner(second, learner, auth)
    assert status == 200 and receipt["objects_deleted"] > 0, receipt
    assert queued[0].outcome() == (404, NOT_FOUND)
    nothing_left(first, ids | {item}, [])


@pytest.mark.parametrize("round_", range(3))  # the order differs from round to round
def test_answers_a_lesson_and_deletions_released_together_never_deadlock_or_leave_data(two_replicas, round_) -> None:
    first, second, learner, lesson, item, auth = two_replicas
    ids = learner_ids(first._sessions, learner) | {item}
    answers = [answer_text() for _ in range(4)]

    def queue(c):
        with client_for(c, learner, auth=auth) as tab:
            r = tab.post("/tasks", json={"request": "Teach me Spanish", "learner_id": learner})
            return r.status_code, r.json()

    calls = ([lambda c=c, a=a: ("attempt", *post_attempt(c, learner, item, a, auth))
              for c, a in zip((first, second, first, second), answers)]
             + [lambda: ("queue", *queue(first))]
             + [lambda c=c: ("delete", *delete_learner(c, learner, auth)) for c in (second, first)])
    barrier = threading.Barrier(len(calls))
    threads = [Background(lambda call=call: (barrier.wait(), call())[1]) for call in calls]
    results = [t.outcome() for t in threads]
    for kind, status, body in results:  # an error or a deadlock would be a 500
        allowed = {"attempt": (201, 404, 422), "queue": (202, 404), "delete": (200, 409)}[kind]
        assert status in allowed, (kind, status, body)
        if status == 422:  # an answer that arrived after the deletion and found the lesson gone
            assert body["detail"].startswith("lesson ") and body["detail"].endswith(" not found"), body
    queued = [body["task_id"] for kind, status, body in results if kind == "queue" and status == 202]
    if not any(kind == "delete" and status == 200 for kind, status, _ in results) or queued:
        # the lesson was queued first and blocked the deletions (409): cancel it, as the learner would, and delete
        with client_for(first, learner, auth=auth) as a:
            for task_id in queued:
                assert a.post(f"/tasks/{task_id}/cancel").status_code in (200, 202, 404)
            ids |= set(queued)
            assert a.delete(f"/learners/{learner}").status_code == 200
    nothing_left(first, ids, answers)
