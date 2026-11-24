"""MVP item 9: a learner's data deletion landing between the reads and the writes of any request for them (SQLite).

The second review's probes, kept as regression tests:

- every protected route of learner A (the security matrix's list), with A's deletion committing just before the
  request's 1st, 2nd or 3rd write transaction. Registering an assessment item (`POST /assessment-items`) and the next
  action (`GET /learners/{id}/next-action`, which stores curriculum progress and publishes events naming the goal) were
  the routes whose late writes survived. Whatever the route answers, it is never a server error, and after a second
  deletion no erased table holds any id that led back to A: a late write was refused, not left where no deletion can
  reach it. The profile is not re-created.
- the next action with the deletion between its goal reads and the `learning_action.selected` event, and the
  sweeper's progress refresh (the next action's old write, issue #66 PR 3) with the deletion just before it: no 500, no
  error, no event naming the deleted goal.
- a lesson queued for a learner who never sent their profile (no learner row) is accepted, as before the deletion
  feature, and is deleted with the rest of their data."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.api.auth import AUTHOR
from app.storage.orm import EventRow
from tests.auth import client_for
from tests.conftest import LEARNER as ONBOARDING
from tests.deletion_residue import db_residue, learner_ids
from tests.e2e import test_security_api
from tests.e2e.test_security_api import routes
from tests.learning_cycle_fixtures import LEARNER

world = test_security_api.world

PLACEHOLDER = {"learner": LEARNER, "goal": "G", "task": "T", "evaluation": "E", "session": "S", "item": "I",
               "attempt": "A", "lesson": "L", "cycle": "C", "artifact": "R", "class": "K", "join_code": "J",
               "assignment": "N"}
# PUT /learners/{id} creates a profile on purpose ("Start again"); DELETE /learners/{id} is the deletion itself.
ROUTES = [(m, p) for m, p, _ in routes(PLACEHOLDER)
          if not (m == "PUT" and p.startswith("/learners/")) and not (m == "DELETE" and p == f"/learners/{LEARNER}")]


def concrete(method: str, template: str, ids: dict):
    index = [r[:2] for r in routes(PLACEHOLDER)].index((method, template))
    return routes(ids)[index]


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("method,template", ROUTES)
def test_a_deletion_before_any_write_of_any_route_leaves_nothing(world, method, template, k) -> None:
    env, ids, auth = world
    c = env.container
    a_ids = learner_ids(c._sessions, LEARNER)
    m, path, body = concrete(method, template, ids)
    sessions, real_begin = c._sessions, c._sessions.begin
    state = {"n": 0, "deleting": False, "deleted": False}

    def begin(*args, **kwargs):  # every repository write is a `sessions.begin()` transaction
        if not state["deleting"] and not state["deleted"]:
            state["n"] += 1
            if state["n"] == k:
                state["deleting"] = True
                try:
                    c.learner_deletion.delete(LEARNER)  # the other tab's DELETE commits just before this write
                    state["deleted"] = True
                finally:
                    state["deleting"] = False
        return real_begin(*args, **kwargs)

    sessions.begin = begin
    try:
        with client_for(c, LEARNER, roles=(AUTHOR,), auth=auth) as a:
            status = a.request(m, path, json=body).status_code
    finally:
        sessions.begin = real_begin
    if not state["deleted"]:
        pytest.skip(f"fewer than {k} write transactions")
    assert status < 500, (m, template, status)
    c.learner_deletion.delete(LEARNER)
    assert sorted(set(db_residue(c._sessions, a_ids))) == []
    assert c.learner_service._memory._repo.get(LEARNER) is None


def _next_action_racing_the_deletion(world, obj, attr) -> int:
    """GET next-action, with the deletion committing just before `obj.attr` (a step after the route's reads)."""
    env, ids, auth = world
    c = env.container
    a_ids = learner_ids(c._sessions, LEARNER)
    real, receipts = getattr(obj, attr), []

    def after_a_deletion(*args, **kwargs):
        if not receipts:
            receipts.append(c.learner_deletion.delete(LEARNER))
        return real(*args, **kwargs)

    setattr(obj, attr, after_a_deletion)
    try:
        with client_for(c, LEARNER, auth=auth) as a:
            status = a.get(f"/learners/{LEARNER}/next-action").status_code
            assert receipts and receipts[0].records.get("learners") == 1
            assert a.get(f"/learners/{LEARNER}").status_code == 404
            assert a.delete(f"/learners/{LEARNER}").json()["records"] == {}  # nothing was written after it
    finally:
        setattr(obj, attr, real)
    assert sorted(set(db_residue(c._sessions, a_ids))) == []
    with c._sessions() as s:
        assert not [e.type for e in s.scalars(select(EventRow)) if ids["goal"] in e.body]
    return status


def test_a_deletion_during_the_next_actions_graph_load(world) -> None:
    c = world[0].container
    knowledge = c.curriculum_service._knowledge
    real, receipts = knowledge.graph, []

    async def graph_after_a_deletion(*args, **kwargs):
        if not receipts:
            receipts.append(c.learner_deletion.delete(LEARNER))
        return await real(*args, **kwargs)

    knowledge.graph = graph_after_a_deletion
    try:
        env, ids, auth = world
        a_ids = learner_ids(c._sessions, LEARNER)
        with client_for(c, LEARNER, auth=auth) as a:
            assert a.get(f"/learners/{LEARNER}/next-action").status_code < 500
            assert receipts and a.delete(f"/learners/{LEARNER}").json()["records"] == {}
    finally:
        knowledge.graph = real
    assert sorted(set(db_residue(c._sessions, a_ids))) == []


def test_a_deletion_between_the_goal_read_and_the_progress_refresh(world) -> None:
    """The progress refresh moved from GET next-action to the writers and the sweeper (issue #66 PR 3): the GET does
    not reach it (`_next_action_racing_the_deletion` would never fire), so the sweeper's refresh races the deletion:
    it skips the deleted learner (no error) and writes nothing after the deletion."""
    import asyncio

    env, ids, auth = world
    c = env.container
    engine = c.curriculum_service._tracker.engine
    a_ids = learner_ids(c._sessions, LEARNER)
    real, receipts = engine.refresh, []

    def after_a_deletion(*args, **kwargs):
        if not receipts:
            receipts.append(c.learner_deletion.delete(LEARNER))
        return real(*args, **kwargs)

    engine.refresh = after_a_deletion
    try:
        report = asyncio.run(c.task_worker.sweep_learning(full=True))
    finally:
        engine.refresh = real
    assert receipts and receipts[0].records.get("learners") == 1
    assert report is not None and report.errors == 0, report.summary()
    assert c.learner_deletion.delete(LEARNER).records == {}  # nothing was written after it
    assert sorted(set(db_residue(c._sessions, a_ids))) == []
    with c._sessions() as s:
        assert not [e.type for e in s.scalars(select(EventRow)) if ids["goal"] in e.body]


def test_a_deletion_between_the_progress_refresh_and_the_selected_event(world) -> None:
    c = world[0].container
    assert _next_action_racing_the_deletion(world, c.curriculum_service._tracker.actions, "select") in (200, 404)


def test_a_lesson_is_queued_for_a_learner_without_a_profile_and_deleted_with_their_data(container) -> None:
    new = "never-onboarded"
    with client_for(container, new, ONBOARDING["learner_id"]) as c:
        task = c.post("/tasks", json={"request": ONBOARDING["request"], "learner_id": new})
        assert task.status_code == 202, task.text  # as before: a lesson may be asked for before onboarding
        assert c.get(f"/learners/{new}").status_code == 404  # and no profile was made for them
        assert c.post(f"/tasks/{task.json()['task_id']}/cancel").status_code == 200
        deleted = c.delete(f"/learners/{new}")
        assert deleted.status_code == 200 and deleted.json()["records"]["tasks"] == 1, deleted.text
        assert c.get(f"/tasks/{task.json()['task_id']}").status_code == 404
