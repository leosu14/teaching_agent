"""MVP item 7: the teacher dashboard through ClassDashboardService and the SQL repositories (SQLite).

The class has the teaching fixture's learner (a generated lesson, then graded attempts) and a learner who joined but
has done nothing yet. The dashboard aggregates what each learner's own views show; UNCERTAIN grades are listed for
review until a later graded attempt on the same item settles them; and reading either changes nothing."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text

from app.schemas.assessment import AttemptSubmission, RegisterAssessmentItem
from app.schemas.learner import LearnerProfileInput
from scripts.run_assessment_demo import fixture
from tests.teaching_fixtures import Env

TEACHER = "teacher-1"


def register(env: Env, key: str) -> tuple[str, dict, dict]:
    entry = next(e for e in fixture()["items"] if e["item"]["assessment_item_id"] == key)
    body = {"item": {**entry["item"], "lesson_id": env.lesson_task}}
    if "rubric" in entry:
        body["rubric"] = entry["rubric"]
    item = env.container.assessment_service.register(RegisterAssessmentItem.model_validate(body))
    return item.assessment_item_id, entry["item"], entry["answers"]


def attempt(env: Env, learner: str, item_id: str, answer: str, attempt_id: str):
    return asyncio.run(env.container.assessment_service.submit_attempt(
        item_id, AttemptSubmission(learner_id=learner, answer=answer, attempt_id=attempt_id)))


@pytest.fixture
def world(teaching_env: Env):
    c = teaching_env.container
    learner = c.task_service.get(teaching_env.lesson_task).learner_id
    c.learner_service.upsert("quiet", LearnerProfileInput(display_name="Quiet Learner"))
    c.learner_service.upsert("outsider", LearnerProfileInput(display_name="Not in the class"))
    made = c.classes.create(teacher_id=TEACHER, tenant_id=None, name="Spanish B1")
    for who in (learner, "quiet"):
        c.classes.join(who, made.join_code, tenant_id=None)
    classroom = c.classes.owned(made.class_id, teacher_id=TEACHER, tenant_id=None)
    return teaching_env, learner, classroom


def test_the_dashboard_shows_each_learners_progress_as_their_own_views_do(world) -> None:
    env, learner, classroom = world
    c = env.container
    item_id, _, answers = register(env, "why_preterite")
    assert attempt(env, learner, item_id, answers["semantic"], "dash-1").grade.outcome == "CORRECT"

    board = c.class_dashboard.dashboard(classroom)
    assert (board.class_id, board.name) == (classroom.class_id, "Spanish B1")
    by_id = {o.learner_id: o for o in board.learners}
    assert set(by_id) == {learner, "quiet"}  # the roster only: never "outsider"
    mine, progress = by_id[learner], c.learner_service.progress(learner)
    assert mine.average_mastery == progress.average_mastery and mine.mastered == len(progress.mastered)
    assert mine.weak == len(progress.weak) and mine.due_for_review == len(progress.due_for_review)
    assert mine.lessons_completed == progress.lessons_completed >= 1
    assert mine.assessments_taken == progress.assessments_taken
    assert mine.display_name == c.memory.get(learner).display_name and mine.last_active_at is not None
    for g in mine.goals:  # goals and curriculum progress as GET /goals/{id}/curriculum computes them
        plan = c.curriculum_service.curriculum(g.goal_id)
        assert g.planned == (plan is not None)
        if plan is not None:
            assert (g.objectives_mastered, g.objectives_required) == (plan.progress.mastered, plan.progress.required)

    quiet = by_id["quiet"]
    assert quiet.display_name == "Quiet Learner" and quiet.goals == [] and quiet.cycles == 0
    assert quiet.current_cycle_status is None and quiet.average_mastery == 0.0 and quiet.uncertain_open == 0
    assert [o.display_name for o in board.learners] == sorted((o.display_name for o in board.learners), key=str.casefold)


def test_uncertain_grades_wait_for_review_until_a_later_graded_attempt(world) -> None:
    env, learner, classroom = world
    c = env.container
    item_id, item, answers = register(env, "why_preterite")
    unsure = attempt(env, learner, item_id, answers["uncertain"], "dash-u1")
    assert unsure.grade.outcome == "UNCERTAIN"

    [open_] = c.class_dashboard.uncertain_grades(classroom)
    assert (open_.attempt_id, open_.learner_id, open_.assessment_item_id) == ("dash-u1", learner, item_id)
    assert open_.prompt == item["prompt"] and open_.learner_answer == answers["uncertain"]
    assert open_.source == "api" and open_.attempt_number == 1 and not open_.resolved and open_.later_attempts == 0
    dumped = open_.model_dump_json()
    assert item["expected_answer"] not in dumped  # never the answer key, never the grader's reason
    assert "uncertainty_reason" not in dumped and "expected_answer" not in dumped
    assert c.assessment_service.stored_grade("dash-u1").uncertainty_reason is not None  # it exists internally
    board = c.class_dashboard.dashboard(classroom)
    assert board.uncertain_open == 1
    assert {o.learner_id: o.uncertain_open for o in board.learners} == {learner: 1, "quiet": 0}

    # an UNCERTAIN retry does not settle it; a graded one does
    attempt(env, learner, item_id, answers["uncertain"] + " ", "dash-u2")
    pending = c.class_dashboard.uncertain_grades(classroom)
    assert [u.attempt_id for u in pending] == ["dash-u2", "dash-u1"]  # newest first
    assert [u.later_attempts for u in pending] == [0, 1]
    assert attempt(env, learner, item_id, answers["semantic"], "dash-3").grade.outcome == "CORRECT"
    assert c.class_dashboard.uncertain_grades(classroom) == []
    settled = c.class_dashboard.uncertain_grades(classroom, include_resolved=True)
    assert [(u.attempt_id, u.resolved) for u in settled] == [("dash-u2", True), ("dash-u1", True)]
    assert c.class_dashboard.dashboard(classroom).uncertain_open == 0


def test_a_learner_off_the_roster_is_not_read(world) -> None:
    env, learner, classroom = world
    c = env.container
    item_id, _, answers = register(env, "why_preterite")
    attempt(env, learner, item_id, answers["uncertain"], "dash-gone")
    assert len(c.class_dashboard.uncertain_grades(classroom)) == 1
    c.classes.remove(classroom.class_id, learner)
    assert c.class_dashboard.uncertain_grades(classroom) == []
    assert [o.learner_id for o in c.class_dashboard.dashboard(classroom).learners] == ["quiet"]


def test_reading_the_dashboard_writes_nothing(world) -> None:
    env, learner, classroom = world
    c = env.container
    item_id, _, answers = register(env, "why_preterite")
    attempt(env, learner, item_id, answers["uncertain"], "dash-ro")

    def database() -> dict:
        with c._sessions() as s:  # noqa: SLF001 - the test reads every table to prove nothing changed
            names = [r[0] for r in s.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))]
            return {n: sorted(map(str, s.execute(text(f'SELECT * FROM "{n}"')).all())) for n in names}

    before = database()
    c.class_dashboard.dashboard(classroom)
    c.class_dashboard.uncertain_grades(classroom, include_resolved=True)
    assert database() == before
