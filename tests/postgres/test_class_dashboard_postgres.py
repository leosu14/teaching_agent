"""The teacher dashboard on PostgreSQL (MVP item 7). Opt-in like the other PostgreSQL tests: skipped unless
TA_TEST_POSTGRES_URL names a database the test may write to.

The dashboard reads a whole roster's attempts in one query (attempts joined with their grades, `learner_id IN (...)`,
ordered by learner, item and attempt number) and decides which UNCERTAIN grades a later graded attempt settled by
walking each learner's attempts on an item in order. PostgreSQL orders text by its collation, not by Python's code
points, so the ids here mix case and punctuation: whatever the collation, each learner's attempts on an item must
come out together and in attempt order. Two API replicas share the database: the class, the joins and the attempts
are written on one, the dashboard is read on the other (nothing is kept in process memory), over HTTP as the teacher.
Every run uses fresh ids, so the database can be reused."""

from __future__ import annotations

import os
import threading
from datetime import timedelta

import pytest

from app.api.auth import TEACHER
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.assessment import (
    AssessmentAttempt,
    AssessmentGrade,
    AssessmentItem,
    AssessmentOutcome,
    GraderType,
    ResponseType,
)
from app.schemas.common import new_id, utcnow
from app.schemas.learner import LearnerProfileInput
from app.services.container import build_container
from scripts.run_learning_cycle_demo import settings
from tests.auth import TestAuth, client_for

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL dashboard tests")
SECRET = "PG-DASHBOARD-ANSWER-KEY"


def replica(data_dir):
    return build_container(settings(data_dir, database_url=URL, log_json=False),
                           llm_providers={"mock": MockLLMProvider(default_responders())})


def store_attempt(c, item_id: str, learner_id: str, outcome: AssessmentOutcome, at, answer: str) -> str:
    attempt_id = new_id("att")
    uncertain = outcome == AssessmentOutcome.UNCERTAIN
    grade = AssessmentGrade(
        grade_id=new_id("grd"), assessment_item_id=item_id, attempt_id=attempt_id, learner_answer=answer,
        normalized_answer=answer, score=1.0 if outcome == AssessmentOutcome.CORRECT else 0.0, max_score=1.0,
        percentage=100.0 if outcome == AssessmentOutcome.CORRECT else 0.0, outcome=outcome,
        confidence=0.3 if uncertain else 1.0, grader_type=GraderType.SEMANTIC,
        feedback={"outcome": outcome.value},
        uncertainty_reason=f"grader output quoted {SECRET}" if uncertain else None, created_at=at)
    attempt = AssessmentAttempt(attempt_id=attempt_id, assessment_item_id=item_id, learner_id=learner_id,
                                attempt_number=1, learner_answer=answer, answer_hash=new_id("h"), submitted_at=at,
                                grade_id=grade.grade_id, source="api", task_id=new_id("task"))
    c.assessment_service.repository.add_attempt(attempt, grade)
    return attempt_id


def test_the_dashboard_across_two_replicas(tmp_path) -> None:
    first, second = replica(tmp_path / "data"), replica(tmp_path / "data")
    try:
        run = new_id("r").replace("_", "")
        learners = [f"Ana-{run}", f"ana_{run}", f"b.{run}", f"Zoe {run}"]  # orders differently by collation
        outsider = f"out-{run}"
        for i, lid in enumerate(learners + [outsider]):
            first.learner_service.upsert(lid, LearnerProfileInput(display_name=f"Learner {i}"))
        teacher_id = f"teacher-{run}"
        made = first.classes.create(teacher_id=teacher_id, tenant_id=None, name="PG class")
        for lid in learners:
            second.classes.join(lid, made.join_code, tenant_id=None)
        items = [f"Item-{run}", f"item_{run}"]
        for item_id in items:
            first.assessment_service.repository.save_item(AssessmentItem(
                assessment_item_id=item_id, lesson_id=new_id("lesson"), concept_id="preterite",
                prompt=f"Explain {item_id}", expected_answer=SECRET, response_type=ResponseType.FREE_TEXT))

        t0 = utcnow()
        U, C, I = AssessmentOutcome.UNCERTAIN, AssessmentOutcome.CORRECT, AssessmentOutcome.INCORRECT
        # per learner: item 0 is UNCERTAIN then settled by a graded retry; item 1 is UNCERTAIN, then UNCERTAIN again
        open_ids, settled_ids = set(), set()
        for n, lid in enumerate(learners):
            at = t0 + timedelta(seconds=10 * n)
            settled_ids.add(store_attempt(first, items[0], lid, U, at, "no sé"))
            store_attempt(second, items[0], lid, C if n % 2 else I, at + timedelta(seconds=1), "porque terminó")
            open_ids.add(store_attempt(first, items[1], lid, U, at + timedelta(seconds=2), "quizás"))
            open_ids.add(store_attempt(second, items[1], lid, U, at + timedelta(seconds=3), "quizás no"))
        store_attempt(first, items[1], outsider, U, t0, "not on the roster")

        auth = TestAuth()
        headers = auth.headers(user_id=teacher_id, roles=(TEACHER,))
        with client_for(second, auth=auth) as b:
            board = b.get(f"/classes/{made.class_id}/dashboard", headers=headers)
            assert board.status_code == 200, board.text
            body = board.json()
            assert sorted(o["learner_id"] for o in body["learners"]) == sorted(learners)
            assert {o["learner_id"]: o["uncertain_open"] for o in body["learners"]} == {lid: 2 for lid in learners}
            assert body["uncertain_open"] == 2 * len(learners)

            pending = b.get(f"/classes/{made.class_id}/uncertain-grades", headers=headers)
            assert pending.status_code == 200
            assert {u["attempt_id"] for u in pending.json()} == open_ids
            assert [u["submitted_at"] for u in pending.json()] == sorted(
                (u["submitted_at"] for u in pending.json()), reverse=True)
            later = {u["attempt_number"]: u["later_attempts"] for u in pending.json()}
            assert later == {1: 1, 2: 0}  # attempt order within each learner's item, whatever the collation
            every = b.get(f"/classes/{made.class_id}/uncertain-grades?include_resolved=true", headers=headers).json()
            assert {u["attempt_id"] for u in every} == open_ids | settled_ids
            assert all(u["resolved"] == (u["attempt_id"] in settled_ids) for u in every)
            for r in (board, pending):
                assert SECRET not in r.text and "uncertainty_reason" not in r.text and outsider not in r.text

            # a learner leaving on one replica is gone from the other's next read
            first.classes.remove(made.class_id, learners[0])
            after = b.get(f"/classes/{made.class_id}/dashboard", headers=headers).json()
            assert learners[0] not in [o["learner_id"] for o in after["learners"]]
    finally:
        first.close()
        second.close()


def test_reads_under_concurrent_attempts_see_whole_attempts(tmp_path) -> None:
    """Attempts and grades are written in one transaction and read with a join: a dashboard read while attempts are
    being stored never sees an attempt without its grade, and every read is consistent with some prefix."""
    c = replica(tmp_path / "data")
    try:
        run = new_id("r")
        lid = f"conc-{run}"
        c.learner_service.upsert(lid, LearnerProfileInput(display_name="Concurrent"))
        made = c.classes.create(teacher_id=f"t-{run}", tenant_id=None, name="Busy")
        c.classes.join(lid, made.join_code, tenant_id=None)
        classroom = c.classes.owned(made.class_id, teacher_id=f"t-{run}", tenant_id=None)
        item_id = f"conc-item-{run}"
        c.assessment_service.repository.save_item(AssessmentItem(
            assessment_item_id=item_id, lesson_id=new_id("lesson"), concept_id="preterite", prompt="Why?",
            expected_answer=SECRET, response_type=ResponseType.FREE_TEXT))
        done, seen, errors = threading.Event(), [], []

        def writer() -> None:
            try:
                for i in range(20):
                    store_attempt(c, item_id, lid, AssessmentOutcome.UNCERTAIN, utcnow(), f"answer {i}")
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=writer)
        thread.start()
        while not done.is_set():
            found = c.class_dashboard.uncertain_grades(classroom)
            seen.append(sorted(u.attempt_number for u in found))
        thread.join()
        assert not errors
        for numbers in seen:
            assert numbers == list(range(1, len(numbers) + 1))  # a prefix: no gap, no attempt without a grade
        final = c.class_dashboard.uncertain_grades(classroom)
        assert len(final) == 20 and [u.later_attempts for u in final][-1] == 19
    finally:
        c.close()
