"""MVP item 9: learner data deletion over HTTP (`DELETE /learners/{id}`).

The world is the security matrix's (tests/e2e/test_security_api.py): learner A mid-cycle with every kind of resource
(goal, curriculum, lesson task with its files, teaching session, evaluation, an assessment item with an attempt and
its grade, a learning cycle, a class membership), and learner B with a profile. A deletes their data: every record and
file of theirs is gone, B's are untouched, and the spend ledger is kept. Only A's own scope may delete: A's teacher
gets 403, anyone else 404, and neither changes anything."""

from __future__ import annotations

import pytest

from app.api.auth import TEACHER
from app.schemas.privacy import RETAINED_RECORDS
from sqlalchemy import select

from app.storage.orm import SpendEntryRow, TaskRow
from app.storage.repositories import CONTENT_ADDRESSED
from tests.auth import client_for
from tests.e2e import test_security_api
from tests.e2e.test_security_api import INTRUDER, NOT_FOUND, TEACHER_ID, snapshot
from tests.learning_cycle_fixtures import LEARNER

world = test_security_api.world  # the security matrix's fixture

FORBIDDEN = {"error": "Forbidden", "detail": "a teacher may read the learners of their classes but not act for them"}


def a_reads(ids: dict) -> list[str]:
    """Routes that read A's data by id: after the deletion every one of them is the uniform 404."""
    lid = ids["learner"]
    return [f"/learners/{lid}", f"/learners/{lid}/progress", f"/goals/{ids['goal']}",
            f"/goals/{ids['goal']}/curriculum", f"/tasks/{ids['task']}", f"/tasks/{ids['evaluation']}",
            f"/tasks/{ids['task']}/artifacts", f"/tasks/{ids['task']}/events", f"/lessons/{ids['lesson']}/reading",
            f"/teaching-sessions/{ids['session']}", f"/learning-cycles/{ids['cycle']}",
            f"/learning-cycles/{ids['cycle']}/events", f"/assessment-attempts/{ids['attempt']}",
            f"/artifacts/{ids['artifact']}/content"]


def stored_objects(env, ids: dict) -> list[str]:
    """The object URIs of A's artifacts (tasks and cycle)."""
    c = env.container
    with c._sessions() as s:
        owners = list(s.scalars(select(TaskRow.task_id).where(TaskRow.learner_id == ids["learner"]))) + [ids["cycle"]]
    return sorted({a.uri for o in owners for a in c.artifacts._repo.list_for_task(o)})


def spend_rows(env) -> int:
    with env.container._sessions() as s:
        return s.query(SpendEntryRow).count()


def test_a_learner_deletes_all_of_their_data_and_only_theirs(world) -> None:
    env, ids, auth = world
    c = env.container
    repo = c.learner_deletion.repository
    objects = stored_objects(env, ids)
    assert objects and repo.holds_data(LEARNER)
    with client_for(c, INTRUDER, auth=auth) as b:
        b_before = b.get(f"/learners/{INTRUDER}").json()
    spend_before = spend_rows(env)

    with client_for(c, LEARNER, auth=auth) as a:
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 200, r.text
        receipt = r.json()
        assert receipt["learner_id"] == LEARNER and receipt["retained"] == list(RETAINED_RECORDS)
        for table in ("learners", "learning_goals", "curricula", "tasks", "artifacts", "events", "teaching_sessions",
                      "teaching_turns", "assessment_items", "assessment_attempts", "assessment_grades",
                      "learning_cycles", "class_members", "learning_evidence"):
            assert receipt["records"].get(table, 0) > 0, (table, receipt["records"])
        assert receipt["objects_deleted"] + receipt["objects_kept_shared"] == len(objects)
        assert receipt["objects_deleted"] > 0
        assert not repo.holds_data(LEARNER)  # no row of any erased table is A's any more

        for path in a_reads(ids):
            got = a.get(path)
            assert got.status_code == 404, (path, got.text)
            if not path.startswith("/learners/"):  # (A's own profile route says the profile is unknown)
                assert got.json() == NOT_FOUND, (path, got.text)
        for listing in ("goals", "learning-cycles", "classes"):  # empty, or unknown without a profile
            got = a.get(f"/learners/{LEARNER}/{listing}")
            assert got.status_code == 404 or got.json() == [], (listing, got.text)

        # A's files are gone from the object store, except content-addressed ones another artifact also holds
        for uri in objects:
            kept = c.artifacts.object_exists(uri)
            assert not kept or CONTENT_ADDRESSED in uri, uri

        # the teacher's roster no longer lists A
        teacher = auth.headers(user_id=TEACHER_ID, roles=(TEACHER,))
        roster = a.get(f"/classes/{ids['class']}", headers=teacher).json()
        assert LEARNER not in [m["learner_id"] for m in roster["learners"]]

        # a retry (a lost response) is safe: nothing left, an empty receipt
        again = a.delete(f"/learners/{LEARNER}")
        assert again.status_code == 200 and again.json()["records"] == {} and again.json()["objects_deleted"] == 0

    with client_for(c, INTRUDER, auth=auth) as b:
        assert b.get(f"/learners/{INTRUDER}").json() == b_before  # B is untouched
    assert spend_rows(env) == spend_before  # the spend ledger is kept: the caps still count what was spent


def test_the_learner_can_start_again_after_a_deletion(world) -> None:
    """The gateway still signs A in: A starts from nothing, as a new learner would."""
    env, _, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        assert a.delete(f"/learners/{LEARNER}").status_code == 200
        assert a.get(f"/learners/{LEARNER}").status_code == 404
        created = a.put(f"/learners/{LEARNER}", json={"display_name": "A again"})
        assert created.status_code == 200 and created.json()["lessons"] == [] \
            and created.json()["concepts"] == {}


def test_a_teacher_cannot_delete_their_learners_data(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        before = snapshot(a, ids)
        r = a.delete(f"/learners/{LEARNER}", headers=auth.headers(user_id=TEACHER_ID, roles=(TEACHER,)))
        assert (r.status_code, r.json()) == (403, FORBIDDEN)
        assert snapshot(a, ids) == before


def test_anyone_else_gets_the_uniform_404_and_deletes_nothing(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        before = snapshot(a, ids)
        for headers in (auth.headers(INTRUDER), auth.headers(INTRUDER, roles=(TEACHER,)),
                        auth.headers(user_id=TEACHER_ID, roles=(TEACHER,), tenant_id="other-tenant")):
            r = a.delete(f"/learners/{LEARNER}", headers=headers)
            assert (r.status_code, r.json()) == (404, NOT_FOUND)
        nobody = a.delete("/learners/does_not_exist", headers=auth.headers(INTRUDER))
        assert (nobody.status_code, nobody.json()) == (404, NOT_FOUND)  # the same answer as for A
        assert snapshot(a, ids) == before


def test_work_in_progress_blocks_the_deletion_and_nothing_is_deleted(world) -> None:
    """A queued lesson would be written by a worker after the deletion: 409, every row and file kept."""
    env, ids, auth = world
    objects = stored_objects(env, ids)
    with client_for(env.container, LEARNER, auth=auth) as a:
        queued = a.post("/tasks", json={"request": "Teach me Spanish", "learner_id": LEARNER})
        assert queued.status_code == 202, queued.text
        before = snapshot(a, ids)
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 409 and r.json()["error"] == "LearnerDataBusy", r.text
        assert "work in progress" in r.json()["detail"]
        assert snapshot(a, ids) == before
        assert all(env.container.artifacts.object_exists(u) for u in objects)
        # once the work is cancelled, the deletion goes ahead
        assert a.post(f"/tasks/{queued.json()['task_id']}/cancel").status_code in (200, 202)
        assert a.delete(f"/learners/{LEARNER}").status_code == 200
        assert not env.container.learner_deletion.repository.holds_data(LEARNER)


def test_a_failed_file_deletion_deletes_no_record_and_a_retry_finishes(world, monkeypatch) -> None:
    env, ids, auth = world
    c = env.container
    real = c.artifacts.delete_object
    calls = []

    def failing(uri: str) -> None:
        calls.append(uri)
        if len(calls) == 2:
            raise OSError("disk unavailable")
        real(uri)

    monkeypatch.setattr(c.artifacts, "delete_object", failing)
    with client_for(c, LEARNER, auth=auth) as a:
        before = a.get(f"/learners/{LEARNER}").json()
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 503 and r.json()["error"] == "ObjectDeletionFailed", r.text
        assert "disk" not in r.text  # the store's error is logged, not returned
        assert a.get(f"/learners/{LEARNER}").json() == before  # no row was deleted
        assert c.learner_deletion.repository.holds_data(LEARNER)
        retried = a.delete(f"/learners/{LEARNER}")
        assert retried.status_code == 200, retried.text
        assert not c.learner_deletion.repository.holds_data(LEARNER)
        for uri in stored_objects(env, ids):
            pytest.fail(f"an artifact of A is still stored: {uri}")
