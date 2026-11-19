"""Class assignments (Important item 3, #62): the request schemas, what a teacher's change does to a goal, and what a
learner may change on an assigned goal. The API flows are in tests/e2e/test_class_assignments_api.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.assignment import AssignmentStatus, ClassAssignment, CreateAssignment, GoalLocked, UpdateAssignment
from app.schemas.curriculum import GoalUpdate
from app.schemas.learner import GoalStatus, LearningGoal
from app.services.assignments import synced
from app.services.curriculum import check_assigned_update

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
PARIS = timezone(timedelta(hours=1))


def goal(**kw) -> LearningGoal:
    fields = {"goal_id": "g", "learner_id": "l", "title": "Old", "domain": "spanish",
              "target_concepts": ["es.preterite"], "target_date": NOW, "status": GoalStatus.ACTIVE, "created_at": NOW,
              "updated_at": NOW, "metadata": {"assignment_id": "a"}}
    return LearningGoal(**{**fields, **kw})


def assignment(**kw) -> ClassAssignment:
    return ClassAssignment(assignment_id="a", class_id="c", created_by="t", title=kw.pop("title", "New"),
                           domain="spanish", framework_id="mastery", topics=[], target_concepts=["es.preterite"],
                           due_at=kw.pop("due_at", NOW + timedelta(days=1)), created_at=NOW, updated_at=NOW, **kw)


def test_a_deadline_is_an_instant_kept_in_utc() -> None:
    made = CreateAssignment(title="t", domain="spanish", topics=["preterite"],
                            due_at=datetime(2026, 12, 1, 23, 59, tzinfo=PARIS))
    assert made.due_at == datetime(2026, 12, 1, 22, 59, tzinfo=timezone.utc) and made.due_at.tzinfo == timezone.utc
    with pytest.raises(ValidationError):
        CreateAssignment(title="t", domain="spanish", topics=["preterite"], due_at=datetime(2026, 12, 1, 23, 59))
    assert UpdateAssignment(due_at="2026-12-01T23:59:00+01:00").due_at.tzinfo == timezone.utc


@pytest.mark.parametrize("body", [{"topics": ["a"], "target_level": "B1"}, {"topics": []}, {"learner_ids": ["l"]},
                                  {"topics": ["a", " A "]}, {"topics": [" "]}])
def test_an_assignment_has_one_target_and_no_learner_list(body) -> None:
    with pytest.raises(ValidationError):
        CreateAssignment.model_validate({"title": "t", "domain": "spanish", "due_at": NOW, **body})


@pytest.mark.parametrize("field", ["topics", "target_level", "domain", "framework_id", "target_concepts"])
def test_d6_a_change_cannot_name_the_targets(field) -> None:
    with pytest.raises(ValidationError):
        UpdateAssignment.model_validate({field: ["x"] if field in ("topics", "target_concepts") else "x"})


def test_a_teachers_change_reaches_an_open_goal_only() -> None:
    new = synced(assignment(), goal(), NOW)
    assert (new.title, new.target_date, new.status) == ("New", NOW + timedelta(days=1), GoalStatus.ACTIVE)
    paused = synced(assignment(), goal(status=GoalStatus.PAUSED), NOW)
    assert paused.status == GoalStatus.PAUSED and paused.title == "New"
    for status in (GoalStatus.COMPLETED, GoalStatus.CANCELLED):  # D13: never changed again
        assert synced(assignment(), goal(status=status), NOW) is None
    same = goal(title="New", description="", target_date=NOW + timedelta(days=1))
    assert synced(assignment(), same, NOW) is None  # nothing to write
    cancelled = synced(assignment(status=AssignmentStatus.CANCELLED), goal(status=GoalStatus.PAUSED), NOW)
    assert cancelled.status == GoalStatus.CANCELLED


@pytest.mark.parametrize("body, allowed", [
    ({"priority": 1}, True), ({"status": "PAUSED"}, True), ({"status": "ACTIVE", "priority": 2}, True),
    ({"title": "x"}, False), ({"target_date": NOW}, False), ({"clear_target_date": True}, False),
    ({"metadata": {}}, False), ({"status": "CANCELLED"}, False), ({"target_level": "B2"}, False),
])
def test_d7_what_the_learner_may_change(body, allowed) -> None:
    data = GoalUpdate.model_validate(body)
    if allowed:
        check_assigned_update(goal(), data)
    else:
        with pytest.raises(GoalLocked):
            check_assigned_update(goal(), data)


@pytest.mark.parametrize("status", [GoalStatus.COMPLETED, GoalStatus.CANCELLED])
def test_d13_nothing_of_a_finished_assigned_goal_changes(status) -> None:
    for body in ({"status": "ACTIVE"}, {"status": "PAUSED"}, {"priority": 1}):
        with pytest.raises(GoalLocked):
            check_assigned_update(goal(status=status), GoalUpdate.model_validate(body))
