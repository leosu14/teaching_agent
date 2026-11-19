"""Important item 3 (#62): a teacher assigns goals, course topics and deadlines to a class.

The world: learner A (the curriculum demo learner, with their own goal) and learner B on the roster of a class of
TEACHER_ID. An assignment becomes one ordinary goal per learner; everything else (curricula, completion by mastery,
the dashboard, deletion) is the goal's. The confirmed decisions each have a test named after them (D1-D13)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.api.auth import TEACHER
from app.learner.memory import UnknownGoal
from app.schemas.assignment import LOCKED_MESSAGE
from app.schemas.common import utcnow
from app.schemas.curriculum import CurriculumStatus, curriculum_id_for
from app.schemas.learner import GoalStatus, LearnerProfileInput, LearningEvidence
from tests.auth import TestAuth, client_for
from tests.e2e.test_security_api import NOT_FOUND
from tests.learning_cycle_fixtures import LEARNER

TEACHER_ID = "teacher-of-assignments"
OTHER_TEACHER = "another-teacher"
B = "assignment-learner-b"
LOCKED = {"error": "GoalLocked", "detail": LOCKED_MESSAGE}


def teacher(auth, user_id: str = TEACHER_ID, **kw):
    return auth.headers(user_id=user_id, roles=(TEACHER,), **kw)


def iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat()


def assignment_body(due: datetime, **kw) -> dict:
    return {"title": "The preterite", "description": "Finished actions in the past.", "domain": "spanish",
            "topics": ["preterite"], "due_at": iso(due), **kw}


def evidence(learner_id: str, concept_id: str, start: datetime, count: int, tag: str) -> list[LearningEvidence]:
    """Correct exercise answers, one second apart from `start`."""
    return [LearningEvidence(
        evidence_id=LearningEvidence.id_for(learner_id, "exercise", f"{tag}-{i}", concept_id), learner_id=learner_id,
        concept_id=concept_id, source_type="exercise", source_ref=f"{tag}-{i}", correctness="correct", score=1.0,
        difficulty=0.7, timestamp=start + timedelta(seconds=i)) for i in range(count)]


class World:
    def __init__(self, env, auth, klass) -> None:
        self.env, self.auth, self.klass = env, auth, klass
        self.c = env.container

    @property
    def class_id(self) -> str:
        return self.klass["class_id"]

    def client(self, learner_id: str = LEARNER):
        return client_for(self.c, learner_id, auth=self.auth)

    def assigned_goal(self, learner_id: str, assignment_id: str):
        [goal] = [g for g in self.c.curriculum_service.goals(learner_id)
                  if g.metadata.get("assignment_id") == assignment_id]
        return goal

    def create(self, a, due: datetime, **kw) -> dict:
        r = a.post(f"/classes/{self.class_id}/assignments", json=assignment_body(due, **kw),
                   headers=teacher(self.auth))
        assert r.status_code == 201, r.text
        return r.json()

    def detail(self, a, assignment_id: str) -> dict:
        r = a.get(f"/classes/{self.class_id}/assignments/{assignment_id}", headers=teacher(self.auth))
        assert r.status_code == 200, r.text
        return r.json()

    def join(self, learner_id: str, code: str | None = None):
        with self.client(learner_id) as x:
            return x.post(f"/learners/{learner_id}/classes", json={"join_code": code or self.klass["join_code"]})


@pytest.fixture
def world(cycle_env_at):
    env = cycle_env_at("base")
    c = env.container
    c.learner_service.upsert(B, LearnerProfileInput(display_name="Bea"))
    auth = TestAuth()
    with client_for(c, LEARNER, auth=auth) as a:
        created = a.post("/classes", json={"name": "Spanish B1"}, headers=teacher(auth))
        assert created.status_code == 201, created.text
    w = World(env, auth, created.json())
    assert w.join(LEARNER).status_code == 201
    assert w.join(B).status_code == 201
    yield w


# --- creating and materializing (D1, D2, D3) ---------------------------------------------------------------------


def test_an_assignment_becomes_a_goal_for_every_learner_on_the_roster(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a:
        view = world.create(a, due)
        assert (view["status"], view["topics"], view["domain"], view["overdue"]) == \
            ("ACTIVE", ["preterite"], "spanish", False)
        assert view["totals"] == {"learners": 2, "completed": 0, "in_progress": 0, "not_started": 2, "overdue": 0,
                                  "not_assigned": 0, "cancelled": 0}
        aid = view["assignment_id"]
        for learner in (LEARNER, B):
            goal = world.assigned_goal(learner, aid)
            # D1: course topics resolve, once, to the knowledge base's concepts; no lesson entity
            assert goal.target_concepts == ["es.preterite"] and goal.target_source == "explicit"
            assert goal.target_date == due.astimezone(timezone.utc)  # D3: the same instant, in UTC
            assert goal.metadata["class_id"] == world.class_id and goal.status.value == "ACTIVE"
            assert goal.title == "The preterite" and goal.description == "Finished actions in the past."
        # learner A's own goal is untouched and not assigned
        own = [g for g in world.c.curriculum_service.goals(LEARNER) if "assignment_id" not in g.metadata]
        assert len(own) == 1 and own[0].title == "Reach B1 Spanish"

        mine = a.get(f"/learners/{LEARNER}/assignments")
        assert mine.status_code == 200, mine.text
        [entry] = mine.json()
        assert (entry["assignment_id"], entry["class_name"], entry["progress"]["status"]) == \
            (aid, "Spanish B1", "NOT_STARTED")
        assert entry["progress"]["learner_id"] == LEARNER  # the learner's own row, never the class's other learners
        assert B not in mine.text

        listed = a.get(f"/classes/{world.class_id}/assignments", headers=teacher(world.auth)).json()
        assert [x["assignment_id"] for x in listed] == [aid]
        detail = world.detail(a, aid)
        assert sorted(x["learner_id"] for x in detail["learners"]) == sorted([LEARNER, B])
        assert {x["display_name"] for x in detail["learners"]} == {"Curriculum demo learner", "Bea"}


def test_creating_again_is_idempotent_and_a_reused_key_is_a_conflict(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a:
        url = f"/classes/{world.class_id}/assignments"
        first = a.post(url, json=assignment_body(due, idempotency_key="k1"), headers=teacher(world.auth))
        again = a.post(url, json=assignment_body(due, idempotency_key="k1"), headers=teacher(world.auth))
        assert (first.status_code, again.status_code) == (201, 200)
        assert first.json()["assignment_id"] == again.json()["assignment_id"]
        other = a.post(url, json=assignment_body(due, idempotency_key="k1", topics=["subjunctive"]),
                       headers=teacher(world.auth))
        assert other.status_code == 409, other.text  # the same key for other targets
        # without a key, the same content is the same assignment (a double submit)
        x = a.post(url, json=assignment_body(due), headers=teacher(world.auth))
        y = a.post(url, json=assignment_body(due), headers=teacher(world.auth))
        assert (x.status_code, y.status_code) == (201, 200) and x.json()["assignment_id"] == y.json()["assignment_id"]
    for learner in (LEARNER, B):
        assigned = [g for g in world.c.curriculum_service.goals(learner) if "assignment_id" in g.metadata]
        assert len(assigned) == 2  # one goal per assignment, however often it was posted


def test_a_repeated_create_completes_a_materialization_that_stopped_part_way(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a:
        aid = world.create(a, due, idempotency_key="partial")["assignment_id"]
        # as if the run had stopped before B: B's record and goal are gone
        repo = world.c.assignments.repository
        with repo._sessions.begin() as s:
            from app.storage.orm import AssignmentLearnerRow, LearningGoalRow
            goal = world.assigned_goal(B, aid)
            s.query(AssignmentLearnerRow).filter_by(assignment_id=aid, learner_id=B).delete()
            s.query(LearningGoalRow).filter_by(goal_id=goal.goal_id).delete()
        pending = [x for x in world.detail(a, aid)["learners"] if x["learner_id"] == B]
        assert pending[0]["source"] == "pending" and pending[0]["goal_id"] is None
        r = a.post(f"/classes/{world.class_id}/assignments", json=assignment_body(due, idempotency_key="partial"),
                   headers=teacher(world.auth))
        assert r.status_code == 200, r.text
        assert world.assigned_goal(B, aid).goal_id == goal.goal_id
        assert repo.learner(aid, B).state.value == "ASSIGNED"


@pytest.mark.parametrize("body, detail", [
    ({"learner_ids": [B]}, "extra"),  # D2: whole class only
    ({"due_at": "2026-12-01T10:00:00"}, "timezone"),  # D3: a naive time names no instant
    ({"target_level": "A2"}, "exactly one target"),  # a level and topics
    ({"topics": []}, "exactly one target"),
    ({"topics": ["no such topic"]}, "no topics"),
    ({"domain": "klingon"}, "no course"),
    ({"topics": ["preterite", "Preterite"]}, "unique"),
])
def test_an_invalid_assignment_is_refused(world, body, detail) -> None:
    with world.client() as a:
        r = a.post(f"/classes/{world.class_id}/assignments", json={**assignment_body(utcnow()), **body},
                   headers=teacher(world.auth))
        assert r.status_code == 422, r.text
        assert detail in r.text
        assert a.get(f"/classes/{world.class_id}/assignments", headers=teacher(world.auth)).json() == []


def test_a_level_assignment_targets_the_concepts_up_to_the_level(world) -> None:
    with world.client() as a:
        view = world.create(a, utcnow() + timedelta(days=3), topics=[], target_level="A2", framework_id="cefr")
        assert view["target_level"] == "A2" and view["topics"] == []
        goal = world.assigned_goal(B, view["assignment_id"])
        assert goal.target_concepts == ["es.present"] and goal.target_level == "A2"
        assert goal.metadata["framework_id"] == "cefr"


def test_only_the_class_teacher_manages_its_assignments(world) -> None:
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        base = f"/classes/{world.class_id}/assignments"
        outsiders = [teacher(world.auth, user_id=OTHER_TEACHER), teacher(world.auth, tenant_id="other-tenant"),
                     {}]  # another teacher, the same teacher id in another tenant, the learner themselves
        for headers in outsiders:
            for method, path, body in [("GET", base, None), ("POST", base, assignment_body(utcnow())),
                                       ("GET", f"{base}/{aid}", None),
                                       ("PATCH", f"{base}/{aid}", {"title": "hijacked"}),
                                       ("POST", f"{base}/{aid}/cancel", None)]:
                r = a.request(method, path, json=body, headers=headers)
                assert (r.status_code, r.json()) == (404, NOT_FOUND), (method, path, headers, r.text)
        missing = a.get(f"{base}/asg_does_not_exist", headers=teacher(world.auth))
        assert (missing.status_code, missing.json()) == (404, NOT_FOUND)
        assert world.c.assignments.repository.get(aid).title == "The preterite"
        # a teacher reading a learner sees only their own classes' assignments
        other = a.post("/classes", json={"name": "Other"}, headers=teacher(world.auth, user_id=OTHER_TEACHER)).json()
        assert world.join(LEARNER, other["join_code"]).status_code == 201
        a.post(f"/classes/{other['class_id']}/assignments", json=assignment_body(utcnow() + timedelta(days=1),
                                                                                  title="Theirs"),
               headers=teacher(world.auth, user_id=OTHER_TEACHER))
        assert len(a.get(f"/learners/{LEARNER}/assignments").json()) == 2  # the learner sees both
        seen = a.get(f"/learners/{LEARNER}/assignments", headers=teacher(world.auth))
        assert seen.status_code == 200 and [x["assignment_id"] for x in seen.json()] == [aid]


# --- what the learner may change (D7, D13) -------------------------------------------------------------------------


def test_the_learner_may_only_change_priority_and_pause_or_resume_an_assigned_goal(world) -> None:
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        url = f"/goals/{goal.goal_id}"
        for body in ({"title": "mine now"}, {"description": "x"}, {"target_level": "B2"},
                     {"target_concepts": ["es.present"]}, {"target_date": iso(utcnow() + timedelta(days=90))},
                     {"clear_target_date": True}, {"metadata": {"assignment_id": None}}, {"status": "CANCELLED"},
                     {"priority": 1, "title": "sneaky"}):
            r = a.patch(url, json=body)
            assert (r.status_code, r.json()) == (409, LOCKED), (body, r.text)
        assert world.c.curriculum_service.goal(goal.goal_id) == goal  # nothing was written
        assert a.patch(url, json={"priority": 1}).json()["goal"]["priority"] == 1
        assert a.patch(url, json={"status": "PAUSED"}).json()["goal"]["status"] == "PAUSED"
        resumed = a.patch(url, json={"status": "ACTIVE", "priority": 2}).json()["goal"]
        assert (resumed["status"], resumed["priority"], resumed["title"]) == ("ACTIVE", 2, "The preterite")
        # the learner's own goal is not limited
        own = [g for g in world.c.curriculum_service.goals(LEARNER) if "assignment_id" not in g.metadata][0]
        assert a.patch(f"/goals/{own.goal_id}", json={"title": "My own"}).status_code == 200
        # and a learner cannot make their own goal look assigned
        made = a.post(f"/learners/{LEARNER}/goals", json={"title": "fake", "domain": "spanish",
                                                         "target_concepts": ["es.present"],
                                                         "metadata": {"assignment_id": aid, "class_id": "x"}})
        assert made.status_code == 201 and "assignment_id" not in made.json()["metadata"]
        patched = a.patch(f"/goals/{own.goal_id}", json={"metadata": {"assignment_id": aid}})
        assert "assignment_id" not in patched.json()["goal"]["metadata"]


async def test_d13_a_completed_assigned_goal_cannot_be_reopened(world) -> None:
    c = world.c
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        await master(world, goal.goal_id, utcnow())
        await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
        assert c.curriculum_service.goal(goal.goal_id).status.value == "COMPLETED"
        for body in ({"status": "ACTIVE"}, {"status": "PAUSED"}, {"priority": 5}):
            r = a.patch(f"/goals/{goal.goal_id}", json=body)
            assert (r.status_code, r.json()) == (409, LOCKED), (body, r.text)


def test_d13_a_cancelled_assigned_goal_cannot_be_resumed(world) -> None:
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        assert a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel",
                      headers=teacher(world.auth)).status_code == 200
        for body in ({"status": "ACTIVE"}, {"status": "PAUSED"}, {"priority": 5}):
            r = a.patch(f"/goals/{goal.goal_id}", json=body)
            assert (r.status_code, r.json()) == (409, LOCKED), (body, r.text)
        assert world.c.curriculum_service.goal(goal.goal_id).status.value == "CANCELLED"


# --- deadlines (D4, D5, D10) --------------------------------------------------------------------------------------


def test_d4_overdue_is_display_only(world) -> None:
    due = utcnow() - timedelta(hours=1)  # a deadline already past
    with world.client() as a:
        view = world.create(a, due)
        aid = view["assignment_id"]
        assert view["overdue"] is True and view["totals"]["overdue"] == 2
        goal = world.assigned_goal(LEARNER, aid)
        assert goal.status.value == "ACTIVE"  # not cancelled, not locked out
        assert a.patch(f"/goals/{goal.goal_id}", json={"status": "PAUSED"}).status_code == 200
        [entry] = a.get(f"/learners/{LEARNER}/assignments").json()
        assert (entry["progress"]["status"], entry["progress"]["overdue"], entry["progress"]["paused"]) == \
            ("OVERDUE", True, True)
        assert not [e for e in world.env.events if e.type in ("notification.sent", "goal.cancelled")]
        # a learning cycle on an overdue goal still starts
        assert a.patch(f"/goals/{goal.goal_id}", json={"status": "ACTIVE"}).status_code == 200
        board = a.get(f"/classes/{world.class_id}/dashboard", headers=teacher(world.auth)).json()
        row = next(x for x in board["learners"] if x["learner_id"] == LEARNER)
        assert row["assignments_overdue"] == 1
        assigned = next(g for g in row["goals"] if g["goal_id"] == goal.goal_id)
        assert assigned["assignment_id"] == aid and assigned["overdue"] is True


def test_d4_exactly_at_the_deadline_is_not_overdue(world) -> None:
    due = utcnow() + timedelta(days=1)
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
    classroom = world.c.classes.repository.get(world.class_id)
    at = world.c.assignments.detail(classroom, aid, as_of=due.astimezone(timezone.utc))
    assert not at.assignment.overdue and {x.status.value for x in at.learners} == {"NOT_STARTED"}
    after = world.c.assignments.detail(classroom, aid, as_of=due + timedelta(microseconds=1))
    assert after.assignment.overdue and {x.status.value for x in after.learners} == {"OVERDUE"}


def test_d5_a_learner_who_joins_later_gets_the_original_deadline(world) -> None:
    due = utcnow() + timedelta(days=7)
    late = "late-joiner"
    world.c.learner_service.upsert(late, LearnerProfileInput(display_name="Late"))
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        assert world.join(late).status_code == 201
        goal = world.assigned_goal(late, aid)
        assert goal.target_date == due.astimezone(timezone.utc) and goal.title == "The preterite"
        assert world.c.assignments.repository.learner(aid, late).source == "joined"
        # joining again does not make a second goal
        assert world.join(late).status_code == 200
        assert len([g for g in world.c.curriculum_service.goals(late) if "assignment_id" in g.metadata]) == 1
        # a cancelled assignment is not given to a later joiner
        cancelled = world.create(a, due, title="Cancelled one")["assignment_id"]
        a.post(f"/classes/{world.class_id}/assignments/{cancelled}/cancel", headers=teacher(world.auth))
        later = "later-joiner"
        world.c.learner_service.upsert(later, LearnerProfileInput(display_name="Later"))
        assert world.join(later).status_code == 201
        assert world.c.assignments.repository.learner(cancelled, later) is None
        assert world.c.assignments.repository.learner(aid, later) is not None


def test_d10_a_new_deadline_reaches_every_goal_without_replanning(world) -> None:
    due = utcnow() + timedelta(days=7)
    c = world.c
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        task = a.post(f"/goals/{goal.goal_id}/curriculum")
        assert task.status_code in (200, 201, 202), task.text
        versions = len(c.curriculum_service.versions(goal.goal_id))
        assert versions == 1
        calls = sum(world.env.llm.calls.values())
        new_due = due + timedelta(days=3)
        r = a.patch(f"/classes/{world.class_id}/assignments/{aid}", headers=teacher(world.auth),
                    json={"due_at": iso(new_due), "title": "Preterite, part 1", "description": "Updated."})
        assert r.status_code == 200, r.text
        assert r.json()["due_at"].startswith(new_due.astimezone(timezone.utc).isoformat()[:19])
        for learner in (LEARNER, B):
            g = world.assigned_goal(learner, aid)
            assert (g.target_date, g.title, g.description) == \
                (new_due.astimezone(timezone.utc), "Preterite, part 1", "Updated.")
            assert g.target_concepts == ["es.preterite"]
        assert len(c.curriculum_service.versions(goal.goal_id)) == versions  # no new curriculum version
        assert sum(world.env.llm.calls.values()) == calls  # and no model call
        # D6: the targets cannot be changed
        for body in ({"topics": ["subjunctive"]}, {"target_level": "B2"}, {"domain": "mathematics"}):
            refused = a.patch(f"/classes/{world.class_id}/assignments/{aid}", json=body, headers=teacher(world.auth))
            assert refused.status_code == 422, refused.text
        # a naive deadline names no instant
        naive = a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"due_at": "2027-01-01T10:00:00"},
                        headers=teacher(world.auth))
        assert naive.status_code == 422


def test_d10_a_deadline_change_leaves_a_paused_goal_paused_and_a_completed_one_alone(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        mine = world.assigned_goal(LEARNER, aid)
        assert a.patch(f"/goals/{mine.goal_id}", json={"status": "PAUSED"}).status_code == 200
        theirs = world.assigned_goal(B, aid)
        world.c.memory.update_goal(theirs.goal_id, lambda g: g.model_copy(update={"status": GoalStatus.COMPLETED}))
        new_due = due + timedelta(days=1)
        a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"due_at": iso(new_due)},
                headers=teacher(world.auth))
        paused = world.assigned_goal(LEARNER, aid)
        assert paused.status.value == "PAUSED" and paused.target_date == new_due.astimezone(timezone.utc)
        done = world.assigned_goal(B, aid)
        assert done.status.value == "COMPLETED" and done.target_date == due.astimezone(timezone.utc)


def test_cancelling_cancels_the_open_goals_and_archives_their_curricula(world) -> None:
    c = world.c
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        a.post(f"/goals/{goal.goal_id}/curriculum")
        r = a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel", headers=teacher(world.auth))
        assert r.status_code == 200 and r.json()["status"] == "CANCELLED" and r.json()["cancelled_at"]
        for learner in (LEARNER, B):
            assert world.assigned_goal(learner, aid).status.value == "CANCELLED"
        assert c.curriculum_service.engine.record(curriculum_id_for(goal.goal_id)).status == CurriculumStatus.ARCHIVED
        again = a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel", headers=teacher(world.auth))
        assert again.status_code == 200 and again.json()["cancelled_at"] == r.json()["cancelled_at"]
        changed = a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"title": "x"},
                          headers=teacher(world.auth))
        assert changed.status_code == 409
        detail = world.detail(a, aid)
        assert {x["status"] for x in detail["learners"]} == {"CANCELLED"}
        assert detail["assignment"]["totals"]["cancelled"] == 2 and detail["assignment"]["overdue"] is False


# --- leaving and rejoining (D9) ------------------------------------------------------------------------------------


def test_d9_a_learner_who_leaves_keeps_the_goal_unlocked_and_out_of_the_teachers_view(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a, world.client(B) as b:
        aid = world.create(a, due)["assignment_id"]
        goal = world.assigned_goal(B, aid)
        assert b.delete(f"/classes/{world.class_id}/learners/{B}").status_code == 204
        detail = world.detail(a, aid)
        assert [x["learner_id"] for x in detail["learners"]] == [LEARNER]
        assert detail["assignment"]["totals"]["learners"] == 1
        assert b.get(f"/learners/{B}/assignments").json() == []  # not in that class any more
        gone = a.get(f"/learners/{B}/assignments", headers=teacher(world.auth))
        assert (gone.status_code, gone.json()) == (404, NOT_FOUND)  # the teacher no longer reads B at all
        # the goal is B's, as it was, and B may now change it freely
        kept = world.c.curriculum_service.goal(goal.goal_id)
        assert kept.status.value == "ACTIVE" and kept.metadata["assignment_id"] == aid
        assert b.patch(f"/goals/{goal.goal_id}", json={"title": "My preterite"}).status_code == 200
        # the teacher's later changes and cancellation do not reach it
        a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"due_at": iso(due + timedelta(days=2))},
                headers=teacher(world.auth))
        a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel", headers=teacher(world.auth))
        after = world.c.curriculum_service.goal(goal.goal_id)
        assert (after.title, after.target_date, after.status.value) == \
            ("My preterite", due.astimezone(timezone.utc), "ACTIVE")
        assert world.assigned_goal(LEARNER, aid).status.value == "CANCELLED"


def test_d9_a_learner_who_rejoins_is_brought_back_to_the_assignment(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a, world.client(B) as b:
        aid = world.create(a, due)["assignment_id"]
        goal = world.assigned_goal(B, aid)
        b.delete(f"/classes/{world.class_id}/learners/{B}")
        new_due = due + timedelta(days=4)
        a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"due_at": iso(new_due)},
                headers=teacher(world.auth))
        assert world.c.curriculum_service.goal(goal.goal_id).target_date == due.astimezone(timezone.utc)
        assert world.join(B).status_code == 201
        back = world.c.curriculum_service.goal(goal.goal_id)
        assert back.target_date == new_due.astimezone(timezone.utc)
        assert len([g for g in world.c.curriculum_service.goals(B) if "assignment_id" in g.metadata]) == 1
        locked = b.patch(f"/goals/{goal.goal_id}", json={"title": "again"})
        assert (locked.status_code, locked.json()) == (409, LOCKED)
        assert [x["learner_id"] for x in world.detail(a, aid)["learners"] if x["learner_id"] == B] == [B]


# --- the course changed (D12) --------------------------------------------------------------------------------------


def test_d12_a_join_succeeds_when_assigned_concepts_are_gone(world) -> None:
    due = utcnow() + timedelta(days=7)
    late = "late-joiner"
    world.c.learner_service.upsert(late, LearnerProfileInput(display_name="Late"))
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        # the knowledge base no longer has one of the concepts the assignment was resolved to
        world.c.assignments.repository.update(aid, lambda x: x.model_copy(update={
            "target_concepts": [*x.target_concepts, "es.removed"]}))
        joined = world.join(late)
        assert joined.status_code == 201, joined.text
        record = world.c.assignments.repository.learner(aid, late)
        assert (record.state.value, record.goal_id) == ("NOT_ASSIGNED", None)
        assert not [g for g in world.c.curriculum_service.goals(late) if "assignment_id" in g.metadata]
        row = next(x for x in world.detail(a, aid)["learners"] if x["learner_id"] == late)
        assert (row["status"], row["goal_id"], row["source"]) == ("NOT_ASSIGNED", None, "joined")
        totals = world.detail(a, aid)["assignment"]["totals"]
        assert totals["not_assigned"] == 1 and totals["learners"] == 3
        assert world.join(late).status_code == 200  # joining again still succeeds, still not assigned
        assert world.c.assignments.repository.learner(aid, late).state.value == "NOT_ASSIGNED"


# --- completion by mastery and its time (D8, D11) -------------------------------------------------------------------


async def master(world, goal_id: str, start: datetime, tag: str = "m") -> None:
    """Plan the goal and record enough correct evidence, from `start`, to master its objectives."""
    c = world.c
    goal = c.curriculum_service.goal(goal_id)
    if c.curriculum_service.curriculum(goal_id) is None:
        task = await c.curriculum_service.build_curriculum(goal_id, user_id="test")
        assert task.status.value == "COMPLETED", task.errors
    for concept in ("es.present", "es.preterite"):
        await c.learner_service.record_evidence(goal.learner_id, "spanish",
                                                evidence(goal.learner_id, concept, start, 6, f"{tag}-{concept}"))


async def test_d8_d11_completion_is_mastery_and_completed_at_is_when_it_was_reached(world) -> None:
    c = world.c
    start = utcnow()
    due = start + timedelta(minutes=10)
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        a.post(f"/goals/{goal.goal_id}/curriculum")
        row = next(x for x in world.detail(a, aid)["learners"] if x["learner_id"] == LEARNER)
        # A's placement evidence on the prerequisite already counts as progress; nothing is complete yet
        assert row["status"] == "IN_PROGRESS" and row["planned"] is True
        assert row["completed_at"] is None and row["completed_late"] is None
        await master(world, goal.goal_id, start)
        # the refresh comes two days after the deadline: the goal was still mastered in time
        refresh = due + timedelta(days=2)
        await c.curriculum_service.tracked(LEARNER, as_of=refresh)
        stored = c.curriculum_service.goal(goal.goal_id)
        assert stored.status.value == "COMPLETED"
        assert start <= stored.completed_at <= start + timedelta(seconds=6)  # an evidence time, not the refresh
        assert stored.completed_at < due
        [entry] = a.get(f"/learners/{LEARNER}/assignments").json()
        assert entry["progress"]["status"] == "COMPLETED" and entry["progress"]["completed_late"] is False
        assert entry["progress"]["completed_at"] is not None
        # practising afterwards neither reopens it nor moves its completion time
        await c.learner_service.record_evidence(LEARNER, "spanish",
                                                evidence(LEARNER, "es.preterite", refresh, 3, "later"))
        await c.curriculum_service.tracked(LEARNER, as_of=refresh + timedelta(hours=1))
        assert c.curriculum_service.goal(goal.goal_id).completed_at == stored.completed_at


async def test_d11_practice_after_mastery_does_not_make_completion_late(world) -> None:
    """Mastered before the deadline, practised again after it, refreshed later still: completed in time."""
    c = world.c
    start = utcnow()
    due = start + timedelta(minutes=10)
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        await master(world, goal.goal_id, start)
        await c.learner_service.record_evidence(LEARNER, "spanish",
                                                evidence(LEARNER, "es.preterite", due + timedelta(hours=1), 3, "p"))
        classroom = c.classes.repository.get(world.class_id)
        before = c.assignments.detail(classroom, aid, as_of=due + timedelta(days=1))
        row = next(x for x in before.learners if x.learner_id == LEARNER)
        assert row.status.value == "COMPLETED" and row.completed_late is False  # read-only, before any refresh
        await c.curriculum_service.tracked(LEARNER, as_of=due + timedelta(days=1))
        stored = c.curriculum_service.goal(goal.goal_id)
        assert stored.completed_at < due and stored.completed_at == row.completed_at


async def test_d11_mastery_reached_after_the_deadline_is_late(world) -> None:
    c = world.c
    due = utcnow() - timedelta(minutes=1)
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        start = utcnow()
        await master(world, goal.goal_id, start)
        await c.curriculum_service.tracked(LEARNER, as_of=start + timedelta(minutes=5))
        assert c.curriculum_service.goal(goal.goal_id).completed_at > due
        [entry] = a.get(f"/learners/{LEARNER}/assignments").json()
        assert entry["progress"]["status"] == "COMPLETED" and entry["progress"]["completed_late"] is True
        assert entry["progress"]["overdue"] is False  # completed is not overdue


async def test_d8_a_paused_goal_is_not_completed_by_a_refresh(world) -> None:
    c = world.c
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=1))["assignment_id"]
        goal = world.assigned_goal(LEARNER, aid)
        await master(world, goal.goal_id, utcnow())
        a.patch(f"/goals/{goal.goal_id}", json={"status": "PAUSED"})
        await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
        stored = c.curriculum_service.goal(goal.goal_id)
        assert stored.status.value == "PAUSED" and stored.completed_at is None


# --- deletion and events --------------------------------------------------------------------------------------------


def test_deleting_a_learner_removes_their_assignment_rows_and_keeps_the_assignment(world) -> None:
    with world.client() as a, world.client(B) as b:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        goal = world.assigned_goal(B, aid)
        assert b.delete(f"/learners/{B}").status_code == 200
        repo = world.c.assignments.repository
        assert repo.learner(aid, B) is None and repo.for_learner(B) == []
        with pytest.raises(UnknownGoal):
            world.c.curriculum_service.goal(goal.goal_id)
        assert repo.get(aid) is not None
        detail = world.detail(a, aid)
        assert [x["learner_id"] for x in detail["learners"]] == [LEARNER]
        # changing and cancelling it afterwards skips the deleted learner
        assert a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"title": "after"},
                       headers=teacher(world.auth)).status_code == 200
        assert a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel",
                      headers=teacher(world.auth)).status_code == 200


def test_assignment_events_carry_ids_and_counts_never_the_teachers_text(world) -> None:
    with world.client() as a:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        a.patch(f"/classes/{world.class_id}/assignments/{aid}", json={"title": "Secret title"},
                headers=teacher(world.auth))
        a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel", headers=teacher(world.auth))
    events = [e for e in world.env.events if e.type.startswith("assignment.")]
    assert {e.type for e in events} == {"assignment.created", "assignment.materialized", "assignment.updated",
                                        "assignment.cancelled"}
    for e in events:
        text = e.model_dump_json()
        assert "preterite" not in text.lower() and "Secret" not in text and "Finished actions" not in text
    assert sum(e.type == "assignment.materialized" for e in events) == 2


# --- review findings ------------------------------------------------------------------------------------------------


def test_d9_a_goal_whose_assignment_was_cancelled_while_away_is_the_learners_after_rejoining(world) -> None:
    with world.client() as a, world.client(B) as b:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        goal = world.assigned_goal(B, aid)
        b.delete(f"/classes/{world.class_id}/learners/{B}")
        a.post(f"/classes/{world.class_id}/assignments/{aid}/cancel", headers=teacher(world.auth))
        assert world.join(B).status_code == 201
        assert world.c.curriculum_service.goal(goal.goal_id).status.value == "ACTIVE"  # the cancel never reached it
        # not locked: B may change or cancel it
        assert b.patch(f"/goals/{goal.goal_id}", json={"title": "Mine"}).status_code == 200
        assert b.patch(f"/goals/{goal.goal_id}", json={"status": "CANCELLED"}).status_code == 200
        # while A's goal, which the cancellation finished, stays locked (D13)
        mine = world.assigned_goal(LEARNER, aid)
        assert a.patch(f"/goals/{mine.goal_id}", json={"status": "ACTIVE"}).status_code == 409


def test_a_deletion_just_after_the_record_leaves_no_event_naming_the_learner(world) -> None:
    from sqlalchemy import select

    from app.storage.orm import EventRow

    late = "deleted-late-joiner"
    world.c.learner_service.upsert(late, LearnerProfileInput(display_name="Gone"))
    with world.client() as a:
        world.create(a, utcnow() + timedelta(days=7))
    repo = world.c.assignments.repository
    real = repo.record_learner

    def then_deleted(record):
        stored = real(record)
        world.c.learner_deletion.delete(late)
        return stored

    repo.record_learner = then_deleted
    try:
        world.join(late)
    finally:
        repo.record_learner = real
    assert not world.c.learner_deletion.repository.holds_data(late)
    with world.c._sessions() as s:
        assert not [e for e in s.scalars(select(EventRow)) if late in e.body]
    assert repo.for_learner(late) == []


def test_two_racing_cancels_and_retries_after_changes(world) -> None:
    due = utcnow() + timedelta(days=7)
    with world.client() as a:
        aid = world.create(a, due)["assignment_id"]
        url = f"/classes/{world.class_id}/assignments"
        a.patch(f"{url}/{aid}", json={"title": "Renamed"}, headers=teacher(world.auth))
        retry = a.post(url, json=assignment_body(due), headers=teacher(world.auth))  # the original create again
        assert retry.status_code == 200 and retry.json()["assignment_id"] == aid
        assert retry.json()["title"] == "Renamed"
        a.post(f"{url}/{aid}/cancel", headers=teacher(world.auth))
        a.post(f"{url}/{aid}/cancel", headers=teacher(world.auth))
        assert sum(e.type == "assignment.cancelled" for e in world.env.events) == 1
        # the same content after a cancel is a new assignment, and a repeat finds that one
        again = a.post(url, json=assignment_body(due), headers=teacher(world.auth))
        assert again.status_code == 201 and again.json()["assignment_id"] != aid
        assert again.json()["status"] == "ACTIVE"
        repeat = a.post(url, json=assignment_body(due), headers=teacher(world.auth))
        assert repeat.status_code == 200 and repeat.json()["assignment_id"] == again.json()["assignment_id"]


async def test_reopening_a_completed_own_goal_clears_its_completion_time(world) -> None:
    c = world.c
    with world.client() as a:
        made = a.post(f"/learners/{LEARNER}/goals", json={"title": "Own", "domain": "spanish",
                                                         "target_concepts": ["es.preterite"]}).json()
        await master(world, made["goal_id"], utcnow())
        await c.curriculum_service.tracked(LEARNER, as_of=utcnow() + timedelta(minutes=5))
        assert c.curriculum_service.goal(made["goal_id"]).completed_at is not None
        reopened = a.patch(f"/goals/{made['goal_id']}", json={"status": "ACTIVE"}).json()["goal"]
        assert reopened["status"] == "ACTIVE" and reopened["completed_at"] is None


def test_a_learners_own_key_never_takes_an_assigned_goals_id(world) -> None:
    late = "pre-empting-learner"
    world.c.learner_service.upsert(late, LearnerProfileInput(display_name="Early"))
    with world.client() as a, world.client(late) as x:
        aid = world.create(a, utcnow() + timedelta(days=7))["assignment_id"]
        own = x.post(f"/learners/{late}/goals", json={"title": "Mine", "domain": "spanish",
                                                       "target_concepts": ["es.preterite"], "idempotency_key": aid})
        assert own.status_code == 201
        assert world.join(late).status_code == 201
        assigned = world.assigned_goal(late, aid)
        assert assigned.goal_id != own.json()["goal_id"]
        assert x.patch(f"/goals/{assigned.goal_id}", json={"title": "x"}).status_code == 409
