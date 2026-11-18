"""Teacher review on PostgreSQL (Important item 2). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

SQLite serializes writers, so it cannot show what happens when they really race. Here: racing decisions on one
lesson release (approve against reject, from many threads) leave exactly one winner and a final status that is that
winner's; racing requests for one lesson task store one release; racing overrides of one attempt store one override.
Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import os
import threading

import pytest

from app.schemas.common import new_id, utcnow
from app.schemas.review import GradeOverride, LessonRelease, ReleaseStatus
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlGradeOverrideRepository, SqlLessonReleaseRepository

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL review tests")
THREADS = 8


@pytest.fixture
def sessions():
    factory = create_db(URL)
    yield factory
    dispose(factory)


def race(n: int, fn) -> tuple[list, list]:
    results, errors = [], []
    start = threading.Barrier(n)

    def run(i: int) -> None:
        try:
            start.wait()
            results.append(fn(i))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def pending(task_id: str) -> LessonRelease:
    return LessonRelease(task_id=task_id, learner_id=new_id("l"), class_id=new_id("cls"),
                         status=ReleaseStatus.PENDING, requested_at=utcnow())


def test_racing_decisions_leave_exactly_one_winner(sessions) -> None:
    repo = SqlLessonReleaseRepository(sessions)
    for _ in range(5):
        task_id = new_id("task")
        repo.request(pending(task_id))
        wanted = {i: (ReleaseStatus.APPROVED if i % 2 else ReleaseStatus.REJECTED) for i in range(THREADS)}
        results, errors = race(THREADS, lambda i: (i, repo.decide(task_id, wanted[i], at=utcnow(), by=f"t{i}",
                                                                   note=None)))
        assert errors == []
        winners = [i for i, won in results if won]
        assert len(winners) == 1
        stored = repo.get(task_id)
        assert stored.status == wanted[winners[0]] and stored.decided_by == f"t{winners[0]}"


def test_racing_requests_store_one_release(sessions) -> None:
    repo = SqlLessonReleaseRepository(sessions)
    task_id = new_id("task")
    first = pending(task_id)
    results, errors = race(THREADS, lambda i: repo.request(first if i == 0 else first.model_copy(
        update={"class_id": f"cls_{i}"})))
    assert errors == []
    assert len({r.class_id for r in results}) == 1  # every caller got the one stored release
    assert repo.get(task_id).class_id == results[0].class_id


def test_racing_overrides_store_one(sessions) -> None:
    repo = SqlGradeOverrideRepository(sessions)
    attempt_id = new_id("att")

    def override(i: int):
        return repo.save(GradeOverride(attempt_id=attempt_id, learner_id="l", class_id="c", teacher_id=f"t{i}",
                                       outcome="CORRECT" if i % 2 else "INCORRECT", evidence_id="ev",
                                       created_at=utcnow()))

    results, errors = race(THREADS, override)
    assert errors == []
    assert sum(created for _, created in results) == 1
    assert len({stored.teacher_id for stored, _ in results}) == 1
