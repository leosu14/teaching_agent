"""Important item 2 (#56): teacher lesson preview and approval before release, and overrides of UNCERTAIN grades.

Lesson approval: a class of TEACHER_ID requires approval; learner A joins and starts a learning cycle. The lesson the
cycle generates is held: every learner surface refuses it (409, nothing of the lesson), and the cycle waits BLOCKED
(failure kind APPROVAL). The teacher lists and previews it (the learner-safe reading, no answer key) and approves
it, and the cycle carries on; or rejects it, and the cycle fails. Leaving the class lifts the hold; without the
setting nothing changes.

Overrides: on the security matrix's world, the class's teacher overrides an UNCERTAIN grade. The grade stays, one
piece of evidence goes through the MasteryUpdater, the learner's views show it, and only that teacher can do it, once.
"""

from __future__ import annotations

import pytest

from app.api.auth import AUTHOR, TEACHER
from app.schemas.learning_cycle import LearnerPrompt
from tests.auth import client_for, settle
from tests.e2e import test_class_dashboard_api, test_security_api
from tests.e2e.test_security_api import INTRUDER, NOT_FOUND, SENTINEL, TEACHER_ID, assert_no_internals
from tests.learning_cycle_fixtures import CONCEPT, LEARNER, ScriptedLearner

world = test_security_api.world
board_world = test_class_dashboard_api.board_world
HELD = {"error": "LessonHeld", "detail": "this lesson is waiting for your teacher to release it"}


def teacher(auth, **kw):
    return auth.headers(user_id=kw.pop("user_id", TEACHER_ID), roles=kw.pop("roles", (TEACHER,)), **kw)


def drive_to_block(a, view: dict) -> dict:
    learner = ScriptedLearner(prefix="r")
    for _ in range(40):
        if view["status"] != "WAITING" or view["prompt"]["kind"] == "SESSION_QUESTION":
            return view
        response = learner.respond(LearnerPrompt.model_validate(view["prompt"]))
        r = a.post(f"/learning-cycles/{view['cycle_id']}/responses",
                   json=response.model_dump(mode="json", exclude_none=True))
        assert r.status_code == 200, r.text
        view = r.json()
    raise AssertionError("the cycle did not stop")


@pytest.fixture
def approval_world(cycle_env_at):
    """Learner A on the roster of a class that approves lessons, with a learning cycle held on its new lesson."""
    from tests.auth import TestAuth

    env = cycle_env_at("base")
    c = env.container
    auth = TestAuth()
    with client_for(c, LEARNER, roles=(AUTHOR,), auth=auth) as a:
        created = a.post("/classes", json={"name": "Spanish B1"}, headers=teacher(auth))
        assert created.status_code == 201, created.text
        class_id = created.json()["class_id"]
        assert created.json()["lesson_approval"] is False
        assert a.post(f"/learners/{LEARNER}/classes", json={"join_code": created.json()["join_code"]}).status_code \
            == 201
        on = a.patch(f"/classes/{class_id}", json={"lesson_approval": True}, headers=teacher(auth))
        assert on.status_code == 200 and on.json()["lesson_approval"] is True, on.text
        view = a.post(f"/learners/{LEARNER}/learning-cycles", json={"idempotency_key": "held"}).json()
        view = drive_to_block(a, view)
    steps = {s["kind"]: s["child_id"] for s in view["steps"]}
    lesson_task = steps["LESSON"]
    ids = {"class": class_id, "cycle": view["cycle_id"], "task": lesson_task,
           "lesson": c.artifacts.find(lesson_task, "lesson").artifact_id}
    yield env, ids, auth, view


def test_a_lesson_of_an_approving_class_is_held_from_the_learner(approval_world) -> None:
    env, ids, auth, view = approval_world
    assert view["status"] == "BLOCKED", view
    assert view["failure"]["kind"] == "APPROVAL" and view["failure"]["retryable"] is True
    assert view["failure"]["message"] == HELD["detail"]
    with client_for(env.container, LEARNER, roles=(AUTHOR,), auth=auth) as a:
        assert a.get(f"/tasks/{ids['task']}").json()["release"] == "PENDING"
        item = {"item": {"assessment_item_id": "held-item", "lesson_id": ids["task"], "concept_id": CONCEPT,
                         "prompt": "Ayer yo ___ (hablar).", "expected_answer": SENTINEL,
                         "response_type": "SHORT_TEXT", "language": "es"}}
        assert a.post("/assessment-items", json=item).status_code == 201  # authors may still prepare items
        refused = [
            a.get(f"/lessons/{ids['lesson']}/reading"),
            a.get(f"/lessons/{ids['task']}/reading"),
            a.post(f"/lessons/{ids['lesson']}/teaching-session", json={"idempotency_key": "early"}),
            a.post(f"/tasks/{ids['task']}/evaluation"),
            a.post("/assessment-items/held-item/attempts", json={"learner_id": LEARNER, "answer": "hablaba"}),
            a.get(f"/artifacts/{ids['lesson']}/content"),
        ]
        for r in refused:
            assert (r.status_code, r.json()) == (409, HELD), (r.request.url, r.text)
        # resuming looks again: still held, still BLOCKED (and never loops into the session)
        again = a.post(f"/learning-cycles/{ids['cycle']}/resume").json()
        assert again["status"] == "BLOCKED" and again["failure"]["kind"] == "APPROVAL"
        assert env.container.teaching_service.sessions(LEARNER) == []
        assert env.container.assessment_service.repository.attempts("held-item", LEARNER) == []


def test_the_teacher_previews_and_approves_and_the_cycle_carries_on(approval_world) -> None:
    env, ids, auth, _ = approval_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        listed = a.get(f"/classes/{ids['class']}/lessons?status=PENDING", headers=teacher(auth))
        assert listed.status_code == 200, listed.text
        [entry] = listed.json()
        assert (entry["task_id"], entry["learner_id"], entry["status"], entry["task_status"]) == \
            (ids["task"], LEARNER, "PENDING", "COMPLETED")
        assert entry["lesson_id"] == ids["lesson"]
        preview = a.get(f"/classes/{ids['class']}/lessons/{ids['task']}/preview", headers=teacher(auth))
        assert preview.status_code == 200, preview.text
        assert_no_internals(preview)
        assert preview.json()["sections"] and preview.json()["task_id"] == ids["task"]

        approved = a.post(f"/classes/{ids['class']}/lessons/{ids['task']}/approve", json={"note": "Good to go"},
                          headers=teacher(auth))
        assert approved.status_code == 200 and approved.json()["status"] == "APPROVED", approved.text
        assert approved.json()["note"] == "Good to go"
        # final and idempotent: approving again is the same; rejecting now is a conflict
        assert a.post(f"/classes/{ids['class']}/lessons/{ids['task']}/approve",
                      headers=teacher(auth)).json()["status"] == "APPROVED"
        assert a.post(f"/classes/{ids['class']}/lessons/{ids['task']}/reject",
                      headers=teacher(auth)).status_code == 409

        assert a.get(f"/tasks/{ids['task']}").json()["release"] == "APPROVED"
        assert a.get(f"/lessons/{ids['lesson']}/reading").status_code == 200
        view = a.post(f"/learning-cycles/{ids['cycle']}/resume").json()
        assert view["status"] == "WAITING" and view["prompt"]["kind"] == "SESSION_QUESTION", view


def test_a_rejected_lesson_fails_the_cycle_and_stays_held(approval_world) -> None:
    env, ids, auth, _ = approval_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        rejected = a.post(f"/classes/{ids['class']}/lessons/{ids['task']}/reject", json={"note": "Too advanced"},
                          headers=teacher(auth))
        assert rejected.status_code == 200 and rejected.json()["status"] == "REJECTED", rejected.text
        assert a.post(f"/classes/{ids['class']}/lessons/{ids['task']}/approve",
                      headers=teacher(auth)).status_code == 409
        view = a.post(f"/learning-cycles/{ids['cycle']}/resume").json()
        assert view["status"] == "FAILED", view
        assert view["failure"]["kind"] == "APPROVAL" and view["failure"]["retryable"] is False
        r = a.get(f"/lessons/{ids['lesson']}/reading")
        assert r.status_code == 409 and r.json()["detail"] == "your teacher did not release this lesson"
        assert a.get(f"/tasks/{ids['task']}").json()["release"] == "REJECTED"
        # a later cycle never reuses a rejected lesson
        assert env.container.curriculum_service.lesson_for(
            LEARNER, env.container.curriculum_service.goals(LEARNER)[0].goal_id, CONCEPT) is None


def test_only_the_classes_teacher_decides_and_previews(approval_world) -> None:
    env, ids, auth, _ = approval_world
    others = [teacher(auth, user_id="another-teacher"), teacher(auth, tenant_id="other-tenant"),
              auth.headers(user_id=TEACHER_ID), auth.headers(LEARNER), auth.headers(INTRUDER, roles=(TEACHER,))]
    paths = [("GET", f"/classes/{ids['class']}/lessons"), ("GET", f"/classes/{ids['class']}/lessons/{ids['task']}"),
             ("GET", f"/classes/{ids['class']}/lessons/{ids['task']}/preview"),
             ("POST", f"/classes/{ids['class']}/lessons/{ids['task']}/approve"),
             ("POST", f"/classes/{ids['class']}/lessons/{ids['task']}/reject"),
             ("PATCH", f"/classes/{ids['class']}")]
    with client_for(env.container, LEARNER, auth=auth) as a:
        for headers in others:
            for method, path in paths:
                r = a.request(method, path, json={} if method != "GET" else None, headers=headers)
                assert (r.status_code, r.json()) == (404, NOT_FOUND), (method, path, r.text)
        # the teacher's own class, but a lesson it never held: the same 404
        assert a.get(f"/classes/{ids['class']}/lessons/task_does_not_exist", headers=teacher(auth)).json() == NOT_FOUND
        assert a.get(f"/tasks/{ids['task']}").json()["release"] == "PENDING"  # nothing was decided


def test_leaving_the_class_lifts_the_hold(approval_world) -> None:
    env, ids, auth, _ = approval_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}").status_code == 204
        assert a.get(f"/lessons/{ids['lesson']}/reading").status_code == 200
        assert a.get(f"/tasks/{ids['task']}").json()["release"] is None
        view = a.post(f"/learning-cycles/{ids['cycle']}/resume").json()
        assert view["prompt"]["kind"] == "SESSION_QUESTION", view
        # the teacher no longer sees it
        assert a.get(f"/classes/{ids['class']}/lessons", headers=teacher(auth)).json() == []


def test_without_the_setting_nothing_is_held(world) -> None:
    """The security matrix's world: A is on the class's roster, the class does not approve lessons."""
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        assert a.get(f"/classes/{ids['class']}", headers=teacher(auth)).json()["lesson_approval"] is False
        assert a.get(f"/tasks/{ids['task']}").json()["release"] is None
        assert a.get(f"/lessons/{ids['lesson']}/reading").status_code == 200
        assert a.get(f"/classes/{ids['class']}/lessons", headers=teacher(auth)).json() == []
        # turning it on later holds nothing requested before
        a.patch(f"/classes/{ids['class']}", json={"lesson_approval": True}, headers=teacher(auth))
        assert a.get(f"/lessons/{ids['lesson']}/reading").status_code == 200
        # ... but a new lesson request is held
        task = settle(a, a.post("/tasks", json={"request": "Teach me the preterite", "learner_id": LEARNER}))
        assert task["release"] == "PENDING", task
        [entry] = a.get(f"/classes/{ids['class']}/lessons", headers=teacher(auth)).json()
        assert entry["task_id"] == task["task_id"]


# --- grade overrides -------------------------------------------------------------------------------------------


def test_the_teacher_overrides_an_uncertain_grade_once(board_world) -> None:
    env, ids, auth = board_world
    c = env.container
    attempt = ids["uncertain_attempt"]
    path = f"/classes/{ids['class']}/attempts/{attempt}/override"
    before = c.memory.get(LEARNER).concepts[CONCEPT].evidence_count
    with client_for(c, LEARNER, auth=auth) as a:
        r = a.post(path, json={"outcome": "CORRECT", "note": "Right idea, unusual wording"}, headers=teacher(auth))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["replayed"] is False and body["override"]["outcome"] == "CORRECT"
        assert len(body["evidence_ids"]) == 1 and body["mastery_changes"]
        assert SENTINEL not in r.text
        after = c.memory.get(LEARNER).concepts[CONCEPT].evidence_count
        assert after == before + 1
        [evidence] = [e for e in c.memory.evidence(LEARNER, CONCEPT) if e.evidence_id == body["evidence_ids"][0]]
        assert (evidence.source_type, evidence.correctness) == ("teacher_review", "correct")

        replay = a.post(path, json={"outcome": "CORRECT", "note": "Right idea, unusual wording"},
                        headers=teacher(auth))
        assert replay.status_code == 200 and replay.json()["replayed"] is True
        assert c.memory.get(LEARNER).concepts[CONCEPT].evidence_count == after  # recorded once
        assert a.post(path, json={"outcome": "INCORRECT"}, headers=teacher(auth)).status_code == 409

        # the grade itself is unchanged; the learner's views show the override
        assert a.get(f"/assessment-attempts/{attempt}/grade").json()["outcome"] == "UNCERTAIN"
        view = a.get(f"/assessment-attempts/{attempt}").json()
        assert view["override"]["outcome"] == "CORRECT" and view["override"]["note"] == "Right idea, unusual wording"
        assert view["retry_recommended"] is False and view["mastery_updated"] is True
        assert body["evidence_ids"][0] in view["evidence_ids"]

        # the dashboard counts it as settled
        assert a.get(f"/classes/{ids['class']}/uncertain-grades", headers=teacher(auth)).json() == []
        [settled] = a.get(f"/classes/{ids['class']}/uncertain-grades?include_resolved=true",
                          headers=teacher(auth)).json()
        assert settled["resolved"] is True and settled["override"]["outcome"] == "CORRECT"
        assert a.get(f"/classes/{ids['class']}/dashboard", headers=teacher(auth)).json()["uncertain_open"] == 0


def test_only_an_uncertain_grade_can_be_overridden(board_world) -> None:
    env, ids, auth = board_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        r = a.post(f"/classes/{ids['class']}/attempts/{ids['attempt']}/override", json={"outcome": "INCORRECT"},
                   headers=teacher(auth))
        assert r.status_code == 409, r.text
        assert a.post(f"/classes/{ids['class']}/attempts/{ids['uncertain_attempt']}/override",
                      json={"outcome": "MAYBE"}, headers=teacher(auth)).status_code == 422


def test_nobody_else_overrides(board_world) -> None:
    env, ids, auth = board_world
    c = env.container
    before = c.memory.get(LEARNER).concepts[CONCEPT].evidence_count
    path = f"/classes/{ids['class']}/attempts/{ids['uncertain_attempt']}/override"
    with client_for(c, LEARNER, auth=auth) as a:
        for headers in (teacher(auth, user_id="another-teacher"), teacher(auth, tenant_id="other-tenant"),
                        auth.headers(LEARNER), auth.headers(INTRUDER, roles=(TEACHER,)),
                        auth.headers(user_id=TEACHER_ID)):
            r = a.post(path, json={"outcome": "CORRECT"}, headers=headers)
            assert (r.status_code, r.json()) == (404, NOT_FOUND), r.text
        # the teacher's class, an attempt id that does not exist
        r = a.post(f"/classes/{ids['class']}/attempts/att_nope/override", json={"outcome": "CORRECT"},
                   headers=teacher(auth))
        assert (r.status_code, r.json()) == (404, NOT_FOUND)
        # after the learner leaves, the teacher cannot override any more
        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}").status_code == 204
        assert a.post(path, json={"outcome": "CORRECT"}, headers=teacher(auth)).json() == NOT_FOUND
    assert c.memory.get(LEARNER).concepts[CONCEPT].evidence_count == before
    assert c.assessment_service.override_of(ids["uncertain_attempt"]) is None


# --- learner data deletion -------------------------------------------------------------------------------------


def test_deleting_the_learner_erases_their_releases_and_overrides(board_world) -> None:
    from tests.deletion_residue import db_residue, learner_ids

    env, ids, auth = board_world
    c = env.container
    with client_for(c, LEARNER, auth=auth) as a:
        assert a.post(f"/classes/{ids['class']}/attempts/{ids['uncertain_attempt']}/override",
                      json={"outcome": "PARTIAL", "note": "half"}, headers=teacher(auth)).status_code == 200
        a.patch(f"/classes/{ids['class']}", json={"lesson_approval": True}, headers=teacher(auth))
        task = settle(a, a.post("/tasks", json={"request": "Teach me the preterite", "learner_id": LEARNER}))
        assert task["release"] == "PENDING"
        found = learner_ids(c._sessions, LEARNER)
        assert ids["uncertain_attempt"] in found and task["task_id"] in found
        r = a.delete(f"/learners/{LEARNER}")
        assert r.status_code == 200, r.text
        deleted = r.json()["deleted"] if "deleted" in r.json() else r.json()
        assert "lesson_releases" in str(deleted) and "grade_overrides" in str(deleted)
        assert db_residue(c._sessions, found) == []
        assert a.get(f"/classes/{ids['class']}/lessons", headers=teacher(auth)).json() == []
    assert c.assessment_service.override_of(ids["uncertain_attempt"]) is None
    assert c.lesson_releases.repository.get(task["task_id"]) is None
