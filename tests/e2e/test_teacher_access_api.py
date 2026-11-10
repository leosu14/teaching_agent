"""MVP item 4: the teacher role, classes and rosters over HTTP.

The world is the security matrix's (tests/e2e/test_security_api.py): learner A mid-cycle with every kind of resource,
enrolled in a class of TEACHER_ID. A teacher reads every learner route of a learner on their roster exactly as the
learner does (the same learner-safe views), and changes nothing: every other method is a 403 and leaves A's state as
it was. Anyone else, including that teacher without the role, in another tenant, or after the learner left the
class, gets the uniform 404."""

from __future__ import annotations

import pytest

from app.api.auth import AUTHOR, TEACHER
from app.curriculum.engine import curriculum_id_for
from app.schemas.learner import LearnerProfileInput
from tests.auth import client_for
from tests.e2e import test_security_api
from tests.e2e.test_security_api import (
    INTRUDER,
    NOT_FOUND,
    TEACHER_ID,
    assert_no_internals,
    routes,
    send,
    snapshot,
)
from tests.learning_cycle_fixtures import LEARNER

world = test_security_api.world  # the security matrix's fixture

SYSTEM = ("/providers", "/agents", "/tools", "/me", "/courses")
FORBIDDEN = {"error": "Forbidden", "detail": "a teacher may read the learners of their classes but not act for them"}


def learner_routes(ids: dict) -> list[tuple[str, str, dict | None]]:
    """The matrix without the system and class routes: every route that reads or changes A's learner data."""
    return [r for r in routes(ids) if r[1] not in SYSTEM and not r[1].startswith("/classes")
            and not r[1].startswith(f"/learners/{ids['learner']}/classes")]


def stable(body):
    """A response without the fields a read computes at request time (`as_of`, and the next action's id)."""
    if isinstance(body, dict):
        return {k: stable(v) for k, v in body.items() if k not in ("as_of", "action_id")}
    return [stable(v) for v in body] if isinstance(body, list) else body


def teacher_headers(auth, *, user_id: str = TEACHER_ID, roles=(TEACHER,), tenant_id: str | None = None):
    return auth.headers(user_id=user_id, roles=roles, tenant_id=tenant_id)


def test_a_teacher_reads_every_learner_route_as_the_learner_does(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        teacher = teacher_headers(auth)
        for method, path, _ in learner_routes(ids):
            if method != "GET":
                continue
            theirs, mine = a.get(path, headers=teacher), a.get(path)
            assert theirs.status_code == mine.status_code, (path, theirs.text)
            if theirs.headers.get("content-type", "").startswith("application/json"):
                assert stable(theirs.json()) == stable(mine.json()), path
                if theirs.status_code == 200:
                    assert_no_internals(theirs)


def test_a_teacher_cannot_choose_the_time_of_their_learners_next_action(world) -> None:
    """Selecting the next action stores the curriculum's progress as of `as_of`. The teacher's read is as of now;
    a teacher's `as_of` would write a snapshot at a time of their choosing, so it is a change (403) and stores
    nothing. The learner's own scope still chooses the time."""
    env, ids, auth = world
    lid, goal = ids["learner"], ids["goal"]
    stored = env.container.curriculum_service._engine.repository

    def snapshot_of_progress():
        found = stored.progress(curriculum_id_for(goal))
        return found.model_dump(mode="json") if found is not None else None

    with client_for(env.container, LEARNER, auth=auth) as a:
        teacher = teacher_headers(auth)
        assert a.get(f"/learners/{lid}/next-action", headers=teacher).status_code == 200
        before = snapshot_of_progress()
        for as_of in ("2099-01-01T00:00:00Z", "2000-01-01T00:00:00Z"):
            r = a.get(f"/learners/{lid}/next-action", params={"as_of": as_of}, headers=teacher)
            assert (r.status_code, r.json()) == (403, FORBIDDEN), r.text
        assert snapshot_of_progress() == before
        other = teacher_headers(auth, user_id="another-teacher")  # not their learner: still the uniform 404
        r = a.get(f"/learners/{lid}/next-action", params={"as_of": "2099-01-01T00:00:00Z"}, headers=other)
        assert (r.status_code, r.json()) == (404, NOT_FOUND)
        mine = a.get(f"/learners/{lid}/next-action", params={"as_of": "2099-01-01T00:00:00Z"})
        assert mine.status_code == 200, mine.text
        assert snapshot_of_progress()["as_of"].startswith("2099-01-01")

def test_a_teacher_changes_nothing_of_their_learner(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        before = snapshot(a, ids)
        teacher = teacher_headers(auth)
        teacher_author = teacher_headers(auth, roles=(TEACHER, AUTHOR))
        for method, path, body in learner_routes(ids):
            if method == "GET":
                continue
            for headers in (teacher, teacher_author):  # the author role does not widen a teacher's access
                r = send(a, method, path, body, headers)
                assert r.status_code == 403 and r.json()["error"] == "Forbidden", (method, path, r.status_code)
            r = send(a, method, path, body, teacher_author)  # refused by the roster rule, not by a missing role
            assert (r.status_code, r.json()) == (403, FORBIDDEN), (method, path, r.status_code, r.text)
        assert snapshot(a, ids) == before


@pytest.mark.parametrize("who", ["other-teacher", "role-revoked", "other-tenant"])
def test_anyone_but_the_classes_teacher_gets_the_uniform_404(world, who) -> None:
    """Each variant has the author role, so that registering an item is refused by scope (404), not by role."""
    env, ids, auth = world
    headers = {"other-teacher": teacher_headers(auth, user_id="another-teacher", roles=(TEACHER, AUTHOR)),
               "role-revoked": teacher_headers(auth, roles=(AUTHOR,)),  # the same user, no longer a teacher
               "other-tenant": teacher_headers(auth, roles=(TEACHER, AUTHOR), tenant_id="another-school")}[who]
    with client_for(env.container, LEARNER, auth=auth) as a:
        if who == "other-teacher":  # a teacher with a class of their own, without A on it
            assert a.post("/classes", json={"name": "Theirs"}, headers=headers).status_code == 201
        before = snapshot(a, ids)
        for method, path, body in learner_routes(ids) + [
                ("GET", f"/classes/{ids['class']}", None), ("POST", f"/classes/{ids['class']}/join-code", None),
                ("DELETE", f"/classes/{ids['class']}/learners/{LEARNER}", None),
                ("GET", f"/learners/{LEARNER}/classes", None)]:
            r = send(a, method, path, body, headers)
            assert (r.status_code, r.json()) == (404, NOT_FOUND), (who, method, path, r.status_code, r.text)
        assert snapshot(a, ids) == before


def test_the_teacher_manages_their_class_and_its_roster(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        teacher = teacher_headers(auth)
        [mine] = a.get("/classes", headers=teacher).json()
        assert mine["class_id"] == ids["class"] and [m["learner_id"] for m in mine["learners"]] == [LEARNER]
        assert "join_code" not in mine  # shown only when made
        assert a.get(f"/classes/{ids['class']}", headers=teacher).json() == mine
        # the learner's own view of the class names it, not its roster or teacher
        [enrolled] = a.get(f"/learners/{LEARNER}/classes").json()
        assert set(enrolled) == {"class_id", "name", "learner_id", "joined_at"}
        assert TEACHER_ID not in a.get(f"/learners/{LEARNER}/classes").text
        # a teacher does not see the learner's other classes
        assert a.get(f"/learners/{LEARNER}/classes", headers=teacher).json() == NOT_FOUND
        # a learner is neither a teacher nor sees a class's roster
        assert a.post("/classes", json={"name": "x"}).status_code == 403
        assert a.get("/classes").status_code == 403
        assert a.get(f"/classes/{ids['class']}").json() == NOT_FOUND


def test_join_codes_rotate_and_joining_is_idempotent(world) -> None:
    env, ids, auth = world
    c = env.container
    c.learner_service.upsert("learner-c", LearnerProfileInput(display_name="C"))
    with client_for(c, "learner-c", auth=auth) as learner_c:
        teacher = teacher_headers(auth)
        again = learner_c.post(f"/learners/{LEARNER}/classes", json={"join_code": ids["join_code"]},
                               headers=auth.headers(LEARNER))
        assert again.status_code == 200 and again.json()["class_id"] == ids["class"]  # A joins again: the same
        assert learner_c.post("/learners/learner-c/classes", json={"join_code": "not-a-code"}).json() == NOT_FOUND
        rotated = learner_c.post(f"/classes/{ids['class']}/join-code", headers=teacher)
        assert rotated.status_code == 200 and rotated.json()["join_code"] != ids["join_code"]
        assert [m["learner_id"] for m in rotated.json()["learners"]] == [LEARNER]  # the roster stays
        old = learner_c.post("/learners/learner-c/classes", json={"join_code": ids["join_code"]})
        assert (old.status_code, old.json()) == (404, NOT_FOUND)  # the old code is retired
        assert learner_c.get("/learners/learner-c", headers=teacher).json() == NOT_FOUND
        new = learner_c.post("/learners/learner-c/classes", json={"join_code": rotated.json()["join_code"]})
        assert new.status_code == 201, new.text
        assert learner_c.get("/learners/learner-c", headers=teacher).status_code == 200
        # a join code of another tenant's class is not found
        other = learner_c.post("/classes", json={"name": "Elsewhere"},
                               headers=teacher_headers(auth, tenant_id="another-school")).json()
        assert learner_c.post("/learners/learner-c/classes", json={"join_code": other["join_code"]}).json() \
            == NOT_FOUND
        # the code is never stored in the clear
        assert c.classes.repository.get(ids["class"]) is not None
        with c._sessions() as s:  # noqa: SLF001 - a test looks at the raw table
            from sqlalchemy import text
            stored = " ".join(str(v) for row in s.execute(text("SELECT * FROM classes")) for v in row)
        assert rotated.json()["join_code"] not in stored and ids["join_code"] not in stored


def test_removal_and_leaving_end_the_teachers_access(world) -> None:
    env, ids, auth = world
    with client_for(env.container, LEARNER, auth=auth) as a:
        teacher = teacher_headers(auth)
        assert a.get(f"/learners/{LEARNER}/progress", headers=teacher).status_code == 200
        # the teacher removes A
        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}", headers=teacher).status_code == 204
        assert a.get(f"/learners/{LEARNER}/progress", headers=teacher).json() == NOT_FOUND
        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}", headers=teacher).json() == NOT_FOUND
        assert a.get(f"/learners/{LEARNER}/classes").json() == []
        # A joins again with the code, then leaves on their own
        code = a.post(f"/classes/{ids['class']}/join-code", headers=teacher).json()["join_code"]
        assert a.post(f"/learners/{LEARNER}/classes", json={"join_code": code}).status_code == 201
        assert a.get(f"/learners/{LEARNER}/progress", headers=teacher).status_code == 200
        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}").status_code == 204
        assert a.get(f"/learners/{LEARNER}/progress", headers=teacher).json() == NOT_FOUND
        # another learner can neither remove A nor leave a class they are not in
        assert a.post(f"/learners/{LEARNER}/classes", json={"join_code": code}).status_code == 201
        intruder = auth.headers(INTRUDER)
        assert a.delete(f"/classes/{ids['class']}/learners/{LEARNER}", headers=intruder).json() == NOT_FOUND
        assert a.delete(f"/classes/{ids['class']}/learners/{INTRUDER}", headers=intruder).json() == NOT_FOUND
        assert a.get(f"/learners/{LEARNER}/progress", headers=teacher).status_code == 200


def test_a_teacher_with_the_learner_in_their_own_scope_keeps_full_access(world) -> None:
    """The gateway may still list a learner for a teacher (a guardian who also teaches): that scope reads and
    changes, as before; the roster adds only reads."""
    env, ids, auth = world
    with client_for(env.container, LEARNER, user_id=TEACHER_ID, roles=(TEACHER,), auth=auth) as t:
        r = t.put(f"/learners/{LEARNER}", json={"display_name": "Renamed by guardian"})
        assert r.status_code == 200, r.text
