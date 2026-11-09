"""Classes and rosters on PostgreSQL (MVP item 4). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

SQLite serializes writers, so it cannot show what happens when they really race. Here row locks decide: concurrent
joins of one learner make one membership, a join racing a rotation either lands with the code it used before the
rotation or is refused (never after it), the roster lookup the authorization boundary makes is right under
concurrent changes, and the tenant scoping holds with NULL tenants. Every run uses fresh ids, so the database can be
reused."""

from __future__ import annotations

import os
import threading

import pytest
from sqlalchemy import text

from app.schemas.classroom import Classroom
from app.schemas.common import new_id, utcnow
from app.services.classes import ClassNotFound, hash_join_code, new_join_code
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlClassRepository

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL class tests")
THREADS = 8


@pytest.fixture
def repo():
    factory = create_db(URL)
    yield SqlClassRepository(factory)
    dispose(factory)


def make(repo: SqlClassRepository, *, teacher: str | None = None, tenant: str | None = None) -> tuple[Classroom, str]:
    classroom = Classroom(class_id=new_id("cls"), name="Class", teacher_id=teacher or new_id("t"), tenant_id=tenant,
                          created_at=utcnow())
    code = new_join_code()
    repo.create(classroom, hash_join_code(code))
    return classroom, code


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


def test_concurrent_joins_of_one_learner_make_one_membership(repo) -> None:
    classroom, code = make(repo)
    learner = new_id("l")
    results, errors = race(THREADS, lambda _: repo.join(hash_join_code(code), learner, utcnow(), None))
    assert errors == []
    assert sum(created for _, _, created in results) == 1
    assert len({m.joined_at for _, m, _ in results}) == 1  # every caller sees the one stored membership
    assert [m.learner_id for m in repo.members(classroom.class_id)] == [learner]


def test_concurrent_joins_of_many_learners_all_land(repo) -> None:
    classroom, code = make(repo)
    learners = [new_id("l") for _ in range(THREADS)]
    results, errors = race(THREADS, lambda i: repo.join(hash_join_code(code), learners[i], utcnow(), None))
    assert errors == [] and all(created for _, _, created in results)
    assert {m.learner_id for m in repo.members(classroom.class_id)} == set(learners)
    assert all(repo.teaches(classroom.teacher_id, None, learner) for learner in learners)


def test_a_join_never_lands_after_the_rotation_that_retired_its_code(repo) -> None:
    """Joins with the old code race a rotation. Each join either committed before the rotation (it is on the roster)
    or found no class (refused). After the rotation, the old code joins nobody."""
    for _ in range(5):
        classroom, code = make(repo)
        learners = [new_id("l") for _ in range(THREADS - 1)]
        fresh = new_join_code()

        def step(i: int):
            if i == 0:
                return "rotated", repo.set_join_code(classroom.class_id, hash_join_code(fresh))
            return learners[i - 1], repo.join(hash_join_code(code), learners[i - 1], utcnow(), None)

        results, errors = race(THREADS, step)
        assert errors == []
        joined = {who for who, outcome in results if who != "rotated" and outcome is not None}
        assert {m.learner_id for m in repo.members(classroom.class_id)} == joined
        assert repo.join(hash_join_code(code), new_id("late"), utcnow(), None) is None
        assert repo.join(hash_join_code(fresh), new_id("l"), utcnow(), None) is not None


def test_removal_racing_lookups_is_seen_at_once(repo) -> None:
    classroom, code = make(repo)
    learner = new_id("l")
    repo.join(hash_join_code(code), learner, utcnow(), None)
    assert repo.teaches(classroom.teacher_id, None, learner)
    results, errors = race(THREADS, lambda i: repo.remove(classroom.class_id, learner) if i == 0
                           else repo.teaches(classroom.teacher_id, None, learner))
    assert errors == []
    assert results.count(True) >= 1  # the removal itself
    assert not repo.teaches(classroom.teacher_id, None, learner)
    assert repo.remove(classroom.class_id, learner) is False


def test_tenant_scoping_holds_with_null_tenants(repo) -> None:
    teacher = new_id("t")
    plain, plain_code = make(repo, teacher=teacher)
    school, school_code = make(repo, teacher=teacher, tenant="school-a")
    learner = new_id("l")
    assert repo.join(hash_join_code(school_code), learner, utcnow(), None) is None  # NULL is not any tenant
    assert repo.join(hash_join_code(plain_code), learner, utcnow(), "school-a") is None
    repo.join(hash_join_code(school_code), learner, utcnow(), "school-a")
    assert repo.teaches(teacher, "school-a", learner)
    assert not repo.teaches(teacher, None, learner)
    assert [c.class_id for c in repo.for_teacher(teacher, None)] == [plain.class_id]
    assert [c.class_id for c in repo.for_teacher(teacher, "school-a")] == [school.class_id]


def test_join_codes_are_unique_and_stored_hashed(repo) -> None:
    classroom, code = make(repo)
    other = Classroom(class_id=new_id("cls"), name="Copy", teacher_id=new_id("t"), created_at=utcnow())
    with pytest.raises(Exception):  # the unique index: a code names at most one class
        repo.create(other, hash_join_code(code))
    with repo._sessions() as s:  # noqa: SLF001 - a test reads the raw row
        row = s.execute(text("SELECT * FROM classes WHERE class_id = :c"), {"c": classroom.class_id}).one()
    assert code not in " ".join(str(v) for v in row)
    assert repo.get(other.class_id) is None


def test_the_service_refuses_an_unknown_code_on_postgres(repo) -> None:
    from app.services.classes import ClassService

    class Memory:
        def get(self, learner_id):
            return None

    service = ClassService(repo, Memory())  # type: ignore[arg-type]
    with pytest.raises(ClassNotFound):
        service.join(new_id("l"), new_join_code(), tenant_id=None)


def test_a_teacher_reads_and_cannot_change_a_learner_over_http_on_postgres(tmp_path) -> None:
    """The whole path on PostgreSQL: create a class, join it, read as the teacher, refuse a change, remove."""
    from app.api.auth import TEACHER
    from app.config.settings import Settings
    from app.providers.llm.mock import MockLLMProvider
    from app.providers.llm.mock_responders import default_responders
    from app.services.container import build_container
    from tests.auth import client_for

    c = build_container(Settings(data_dir=tmp_path / "data", database_url=URL, log_json=False),
                        llm_providers={"mock": MockLLMProvider(default_responders())})
    learner, teacher_id = new_id("l"), new_id("t")
    try:
        with client_for(c, learner) as a:
            assert a.put(f"/learners/{learner}", json={"display_name": "A"}).status_code == 200
            teacher = a.test_auth.headers(user_id=teacher_id, roles=(TEACHER,))
            made = a.post("/classes", json={"name": "Spanish"}, headers=teacher).json()
            assert a.get(f"/learners/{learner}", headers=teacher).status_code == 404
            assert a.post(f"/learners/{learner}/classes", json={"join_code": made["join_code"]}).status_code == 201
            assert a.get(f"/learners/{learner}", headers=teacher).json()["display_name"] == "A"
            changed = a.put(f"/learners/{learner}", json={"display_name": "B"}, headers=teacher)
            assert changed.status_code == 403
            assert a.get(f"/learners/{learner}").json()["display_name"] == "A"
            assert a.delete(f"/classes/{made['class_id']}/learners/{learner}", headers=teacher).status_code == 204
            assert a.get(f"/learners/{learner}", headers=teacher).status_code == 404
    finally:
        c.close()


def test_a_join_waits_for_an_uncommitted_rotation_and_then_finds_no_class(repo) -> None:
    """Deterministic version of the race: a rotation has updated the class row but not committed. A join with the old
    code must wait for it (the row lock) and then find no class; without the lock it would read the old row and
    enroll the learner after the code was retired."""
    classroom, code = make(repo)
    learner = new_id("l")
    outcome: dict = {}
    with repo._sessions.begin() as rotation:  # noqa: SLF001 - a test holds the rotation's transaction open
        rotation.execute(text("UPDATE classes SET join_code_hash = :h WHERE class_id = :c"),
                         {"h": hash_join_code(new_join_code()), "c": classroom.class_id})
        joiner = threading.Thread(target=lambda: outcome.update(
            result=repo.join(hash_join_code(code), learner, utcnow(), None)))
        joiner.start()
        joiner.join(timeout=1.0)
        assert joiner.is_alive(), "the join did not wait for the rotation's row lock"
    joiner.join(timeout=10)
    assert outcome["result"] is None
    assert repo.member(classroom.class_id, learner) is None
