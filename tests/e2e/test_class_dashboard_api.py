"""MVP item 7: the teacher dashboard over HTTP.

The world is the security matrix's (tests/e2e/test_security_api.py): learner A mid-cycle with every kind of resource,
enrolled in a class of TEACHER_ID, plus a free-text answer the grader could not decide on an item whose answer key is
the matrix's sentinel. The class's teacher reads the dashboard and the UNCERTAIN grades; the responses carry no answer
key, no grader reason and no internal field; nobody else, the same teacher without the role or in another tenant, or
after the learner left, sees anything of it; and reading changes nothing of A's."""

from __future__ import annotations

import pytest

from app.api.auth import AUTHOR, TEACHER
from tests.auth import client_for
from tests.e2e import test_security_api
from tests.e2e.test_security_api import (
    INTRUDER,
    NOT_FOUND,
    SENTINEL,
    TEACHER_ID,
    assert_no_internals,
    snapshot,
)
from tests.learning_cycle_fixtures import CONCEPT, LEARNER

GRADER_REASON_WORDS = ("uncertainty_reason", "semantic grader", "Traceback")


world = test_security_api.world  # the security matrix's fixture


@pytest.fixture
def board_world(world):
    """The matrix's world plus an UNCERTAIN grade on an item whose answer key is the sentinel."""
    env, ids, auth = world
    with client_for(env.container, LEARNER, roles=(AUTHOR,), auth=auth) as a:
        item = {"item": {"assessment_item_id": "dash-free", "lesson_id": ids["task"], "concept_id": CONCEPT,
                         "prompt": "¿Por qué se usa el pretérito aquí?", "expected_answer": SENTINEL,
                         "response_type": "FREE_TEXT", "language": "es"}}
        assert a.post("/assessment-items", json=item).status_code == 201
        unsure = a.post("/assessment-items/dash-free/attempts",
                        json={"learner_id": LEARNER, "answer": "Mi gato come pescado.", "attempt_id": "dash-att"})
        assert unsure.status_code == 201, unsure.text
        assert unsure.json()["grade"]["outcome"] == "UNCERTAIN"
    ids = {**ids, "uncertain_item": "dash-free", "uncertain_attempt": "dash-att"}
    yield env, ids, auth


def teacher(auth, **kw):
    return auth.headers(user_id=kw.pop("user_id", TEACHER_ID), roles=kw.pop("roles", (TEACHER,)), **kw)


def test_the_teacher_sees_their_class_at_a_glance(board_world) -> None:
    env, ids, auth = board_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        r = a.get(f"/classes/{ids['class']}/dashboard", headers=teacher(auth))
        assert r.status_code == 200, r.text
        assert_no_internals(r)
        board = r.json()
        assert (board["class_id"], board["name"]) == (ids["class"], "Spanish B1")
        [me] = board["learners"]
        progress = a.get(f"/learners/{LEARNER}/progress").json()  # A's own view
        assert me["learner_id"] == LEARNER and me["average_mastery"] == progress["average_mastery"]
        assert me["mastered"] == len(progress["mastered"]) and me["weak"] == len(progress["weak"])
        assert me["lessons_completed"] == progress["lessons_completed"]
        assert me["current_cycle_status"] in ("RUNNING", "WAITING", "BLOCKED") and me["cycles"] == 1
        [goal] = me["goals"]
        plan = a.get(f"/goals/{ids['goal']}/curriculum").json()["progress"]
        assert goal["goal_id"] == ids["goal"] and goal["planned"]
        assert (goal["objectives_mastered"], goal["objectives_required"]) == (plan["mastered"], plan["required"])
        assert me["uncertain_open"] == board["uncertain_open"] == 1
        assert "join_code" not in r.text and ids["join_code"] not in r.text


def test_uncertain_grades_show_the_question_and_answer_never_the_key(board_world) -> None:
    env, ids, auth = board_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        r = a.get(f"/classes/{ids['class']}/uncertain-grades", headers=teacher(auth))
        assert r.status_code == 200, r.text
        assert_no_internals(r)  # no sentinel answer key, no answer-key or internal field names
        [u] = r.json()
        assert (u["attempt_id"], u["learner_id"], u["assessment_item_id"]) == (
            ids["uncertain_attempt"], LEARNER, ids["uncertain_item"])
        assert u["prompt"] == "¿Por qué se usa el pretérito aquí?" and u["learner_answer"] == "Mi gato come pescado."
        assert u["response_type"] == "FREE_TEXT" and not u["resolved"]
        for word in GRADER_REASON_WORDS:
            assert word not in r.text
        # the internal grade keeps its reason: only the response leaves it out
        assert env.container.assessment_service.stored_grade(ids["uncertain_attempt"]).uncertainty_reason
        assert a.get(f"/classes/{ids['class']}/uncertain-grades?include_resolved=true",
                     headers=teacher(auth)).json() == [u]


@pytest.mark.parametrize("who", ["learner", "intruder", "other_teacher", "no_role", "other_tenant"])
def test_anyone_but_the_classes_teacher_gets_the_uniform_404(board_world, who) -> None:
    env, ids, auth = board_world
    headers = {
        "learner": auth.headers(LEARNER),
        "intruder": auth.headers(INTRUDER, roles=(AUTHOR,)),
        "other_teacher": teacher(auth, user_id="another-teacher"),
        "no_role": teacher(auth, roles=()),
        "other_tenant": teacher(auth, tenant_id="another-school"),
    }[who]
    with client_for(env.container, LEARNER, auth=auth) as a:
        for path in (f"/classes/{ids['class']}/dashboard", f"/classes/{ids['class']}/uncertain-grades",
                     "/classes/cls_does_not_exist/dashboard", "/classes/cls_does_not_exist/uncertain-grades"):
            r = a.get(path, headers=headers)
            assert (r.status_code, r.json()) == (404, NOT_FOUND), (who, path, r.text)


def test_the_dashboard_is_read_only_and_ends_with_the_membership(board_world) -> None:
    env, ids, auth = board_world
    with client_for(env.container, LEARNER, auth=auth) as a:
        before = snapshot(a, ids)
        for path in ("dashboard", "uncertain-grades"):
            assert a.get(f"/classes/{ids['class']}/{path}", headers=teacher(auth)).status_code == 200
            for method in ("POST", "PUT", "PATCH", "DELETE"):  # no route changes anything here
                assert a.request(method, f"/classes/{ids['class']}/{path}", headers=teacher(auth)).status_code == 405
        assert snapshot(a, ids) == before

        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}").status_code == 204  # A leaves
        board = a.get(f"/classes/{ids['class']}/dashboard", headers=teacher(auth)).json()
        assert board["learners"] == [] and board["uncertain_open"] == 0
        assert a.get(f"/classes/{ids['class']}/uncertain-grades", headers=teacher(auth)).json() == []
