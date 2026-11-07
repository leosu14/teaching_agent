"""Learner-memory and curriculum writes fenced at commit, and versioned against lost updates (MVP item 2).

Before this, a task's learner and curriculum writes were only pre-checked by the tool manager: a worker whose lease
was taken over between the check and the write still stored them. Now each such write runs `_require_fence` in its
own transaction (`fenced_writes`, set by the orchestrator for everything under a task's lease), and the profile and
the curriculum pointer are saved only at the version they were read at, so a concurrent writer is never overwritten.
"""

from __future__ import annotations

from collections import Counter

import pytest

from app.runtime.orchestrator.orchestrator import TASK_LEASE
from app.schemas.common import new_id, utcnow
from app.schemas.curriculum import CurriculumProgress, CurriculumRecord, CurriculumStatus, RecordConflict
from app.schemas.learner import (
    LearnerProfile,
    LearnerProfileInput,
    LearningEvent,
    LearningEvidence,
    LearningGoal,
    ProfileConflict,
)
from app.schemas.lease import LeaseLost, fenced_writes
from app.schemas.task import Task, TaskStatus
from app.storage.db import create_db, dispose
from app.storage.repositories import (
    SqlCurriculumRepository,
    SqlEvidenceRepository,
    SqlGoalRepository,
    SqlLearnerRepository,
    SqlLearningEventRepository,
    SqlTaskRepository,
)
from tests.conftest import LEARNER, add_demo_learner, evaluation_answers_for
from tests.integration.test_task_concurrency import AHEAD, Clock, finish, open_container


@pytest.fixture
def sessions(tmp_path):
    factory = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield factory
    dispose(factory)


def evidence(learner_id: str, ref: str, concept: str = "c1") -> LearningEvidence:
    return LearningEvidence(evidence_id=LearningEvidence.id_for(learner_id, "exercise", ref, concept),
                            learner_id=learner_id, concept_id=concept, source_type="exercise", source_ref=ref,
                            correctness="correct", score=1.0, difficulty=0.5, timestamp=utcnow())


def taken_over(sessions):
    """A task whose first worker's lease was taken over by a second: (stale lease, current lease)."""
    tasks = SqlTaskRepository(sessions)
    task = Task(task_id=new_id("task"), user_id="u", learner_id="l1", request="r")
    tasks.create(task)
    stale = tasks.claim(task.task_id, "worker-a", TASK_LEASE)
    current = SqlTaskRepository(sessions, Clock(AHEAD)).claim(task.task_id, "worker-b", TASK_LEASE)
    assert stale is not None and current is not None and current.fence == stale.fence + 1
    return stale, current


def writes(sessions):
    """Every fenced write, each with a probe saying whether it was stored."""
    learners, ev, events = SqlLearnerRepository(sessions), SqlEvidenceRepository(sessions), \
        SqlLearningEventRepository(sessions)
    goals, curricula = SqlGoalRepository(sessions), SqlCurriculumRepository(sessions)
    now = utcnow()
    tag = new_id("w")
    goal = LearningGoal(goal_id=f"goal_{tag}", learner_id="l1", domain="spanish", target_concepts=["c1"])
    record = CurriculumRecord(curriculum_id=f"cur_{tag}", learner_id="l1", goal_id=goal.goal_id, title="t",
                              domain="spanish", current_version=1, current_version_id="v1", created_at=now,
                              updated_at=now)
    progress = CurriculumProgress(curriculum_id=record.curriculum_id, goal_id=goal.goal_id, version=1, objectives=[],
                                  mastered=0, required=0, percent_complete=0.0,
                                  completion_rule="all_required_mastered", goal_complete=False, as_of=now)
    event = LearningEvent.create("l1", "exercise_completed", now, key=tag, subject="spanish", concept_ids=["c1"])
    item = evidence("l1", tag)
    cases = {
        "profile": (lambda: learners.save(LearnerProfile(learner_id=f"l_{tag}")),
                    lambda: learners.get(f"l_{tag}") is not None),
        "evidence": (lambda: ev.add(item), lambda: any(e.evidence_id == item.evidence_id for e in ev.for_learner("l1"))),
        "learning_event": (lambda: events.add(event),
                           lambda: any(e.event_id == event.event_id for e in events.for_learner("l1"))),
        "goal": (lambda: goals.save(goal), lambda: goals.get(goal.goal_id) is not None),
        "curriculum_record": (lambda: curricula.save_record(record.model_copy()),
                              lambda: curricula.record(record.curriculum_id) is not None),
        "curriculum_progress": (lambda: curricula.save_progress(progress),
                                lambda: curricula.progress(record.curriculum_id) is not None),
    }
    return cases


# --- the fence ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["profile", "evidence", "learning_event", "goal", "curriculum_record",
                                  "curriculum_progress"])
def test_a_stale_lease_stores_no_learner_or_curriculum_write(sessions, name) -> None:
    stale, current = taken_over(sessions)
    write, stored = writes(sessions)[name]
    with fenced_writes(stale), pytest.raises(LeaseLost):
        write()
    assert not stored()  # refused in the write's own transaction: nothing of it was committed
    with fenced_writes(current):
        write()
    assert stored()


def test_unfenced_writes_and_writes_under_a_current_lease_are_stored(sessions) -> None:
    _, current = taken_over(sessions)
    for name, (write, stored) in writes(sessions).items():
        write()
        assert stored(), name
    for name, (write, stored) in writes(sessions).items():
        with fenced_writes(current):
            write()
        assert stored(), name


def test_an_expired_lease_alone_fences_the_write(tmp_path) -> None:
    clock = Clock()
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    try:
        tasks = SqlTaskRepository(sessions, clock)
        task = Task(task_id=new_id("task"), user_id="u", learner_id="l1", request="r")
        tasks.create(task)
        lease = tasks.claim(task.task_id, "worker-a", TASK_LEASE)
        learners = SqlLearnerRepository(sessions, clock)
        clock.offset = AHEAD  # nobody took over; the lease just ran out
        with fenced_writes(lease), pytest.raises(LeaseLost):
            learners.save(LearnerProfile(learner_id="late"))
        assert learners.get("late") is None
    finally:
        dispose(sessions)


async def test_a_takeover_between_the_pre_check_and_the_write_records_the_lesson_once(tmp_path) -> None:
    """Worker A reaches `learner.record_lesson` with its lease (the tool manager's pre-check passes); B takes the task
    over before A's write commits. A's lesson record, evidence and learning events are all refused; B resumes, and
    the lesson is in learner memory exactly once."""
    data = tmp_path / "data"
    a, _ = open_container(data)
    b, _ = open_container(data, clock=Clock(AHEAD))
    try:
        lid = add_demo_learner(a)
        task = await a.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        original = a.memory.record_lesson
        intruder = {}

        def record_lesson(outcome, scope):  # runs after the pre-check, as a slow commit would
            intruder["lease"] = b.task_service._tasks.claim(task.task_id, "worker-b", TASK_LEASE)
            return original(outcome, scope)

        a.memory.record_lesson = record_lesson
        with pytest.raises(LeaseLost):
            await finish(a, task.task_id)
        assert intruder["lease"] is not None
        profile = a.memory.get(lid)
        assert [x for x in profile.lessons if x.task_id == task.task_id] == []
        assert not [e for e in a.memory.history(lid) if e.type == "lesson_completed" and e.task_id == task.task_id]
        assert a.task_service.get(task.task_id).status == TaskStatus.RUNNING  # A's last checkpoint

        b.task_service._tasks.release(intruder["lease"])
        done = await finish(b, (await b.task_service.resume(task.task_id)).task_id)
        assert done.status == TaskStatus.COMPLETED
        profile = b.memory.get(lid)
        assert len([x for x in profile.lessons if x.task_id == task.task_id]) == 1
        completed = [e for e in b.memory.history(lid) if e.type == "lesson_completed" and e.task_id == task.task_id]
        assert len(completed) == 1
        assert Counter(e.type for e in b.task_service.events(task.task_id))["task.completed"] == 1
    finally:
        a.close()
        b.close()


# --- versioned profile and curriculum pointer ---------------------------------------------------------------------


def test_a_profile_saved_from_a_stale_read_is_refused(sessions) -> None:
    learners = SqlLearnerRepository(sessions)
    learners.save(LearnerProfile(learner_id="l1", display_name="first"))
    one, two = learners.get("l1"), learners.get("l1")
    one.display_name = "one"
    learners.save(one)
    two.display_name = "two"
    with pytest.raises(ProfileConflict):
        learners.save(two)
    assert learners.get("l1").display_name == "one"
    with pytest.raises(ProfileConflict):  # a fresh profile never replaces a stored one
        learners.save(LearnerProfile(learner_id="l1"))
    again = learners.get("l1")
    again.display_name = "three"
    learners.save(again)
    again.display_name = "four"
    learners.save(again)  # the saved profile carries its new version
    assert learners.get("l1").display_name == "four"


def test_concurrent_profile_changes_are_all_kept(tmp_path) -> None:
    """A lesson's evidence is being recorded while the learner edits their profile in another process: the edit lands
    between A's read and A's save. A's save is refused, A redoes its change on the new profile, and both survive."""
    container, _ = open_container(tmp_path / "data")
    other, _ = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        repo = container.memory._repo
        save = repo.save
        raced = {"n": 0}

        def save_after_an_edit(profile):
            if not raced["n"]:
                raced["n"] += 1
                other.memory.upsert(lid, LearnerProfileInput.model_validate(
                    {**LEARNER["profile"], "display_name": "Edited elsewhere"}))
            return save(profile)

        repo.save = save_after_an_edit
        update = container.memory.record_evidence(lid, "spanish", [evidence(lid, "ex-1", "c_raced")],
                                                  [])
        assert raced["n"] == 1
        profile = other.memory.get(lid)
        assert profile.display_name == "Edited elsewhere"  # the concurrent edit was not lost
        assert "c_raced" in profile.concepts and profile.concepts["c_raced"].evidence_count == 1
        assert [c.concept_id for c in update.changes] == ["c_raced"]
    finally:
        container.close()
        other.close()


def _edit_on_first_save_with(container, other, lid, kind: str) -> dict:
    """Make another process edit the profile between the first read and save of the change that records `kind`."""
    repo = container.memory._repo
    save = repo.save
    raced = {"n": 0}

    def save_after_an_edit(profile):
        if not raced["n"] and any(a.kind == kind for a in profile.assessments):
            raced["n"] += 1
            other.memory.upsert(lid, LearnerProfileInput.model_validate(
                {**LEARNER["profile"], "display_name": "Edited elsewhere"}))
        return save(profile)

    repo.save = save_after_an_edit
    return raced


async def test_a_diagnostic_and_an_evaluation_redone_after_a_profile_race_are_kept(tmp_path) -> None:
    """The redo of an assessment's change must rebuild the evidence the first attempt already stored byte for byte
    (same time), or the immutable evidence store refuses it and the task fails instead of keeping both changes."""
    container, _ = open_container(tmp_path / "data")
    other, _ = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        raced = _edit_on_first_save_with(container, other, lid, "diagnostic")
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        task = await finish(container, task.task_id)
        assert raced["n"] == 1
        assert task.status == TaskStatus.COMPLETED, task.errors

        raced = _edit_on_first_save_with(container, other, lid, "lesson_evaluation")
        evaluation = await container.task_service.start_evaluation(task.task_id, user_id="u1")
        evaluation = await container.task_service.submit_answers(evaluation.task_id,
                                                                 evaluation_answers_for(evaluation))
        assert raced["n"] == 1
        assert evaluation.status == TaskStatus.COMPLETED, evaluation.errors
        profile = other.memory.get(lid)
        assert profile.display_name == "Edited elsewhere"
        assert [a.kind for a in profile.assessments if a.task_id in (task.task_id, evaluation.task_id)] == [
            "diagnostic", "lesson_evaluation"]
    finally:
        container.close()
        other.close()


def test_a_curriculum_pointer_saved_from_a_stale_read_is_refused(sessions) -> None:
    curricula = SqlCurriculumRepository(sessions)
    now = utcnow()
    curricula.save_record(CurriculumRecord(curriculum_id="cur", learner_id="l1", goal_id="g", title="t",
                                           domain="spanish", current_version=1, current_version_id="v1",
                                           created_at=now, updated_at=now))
    read = curricula.record("cur")
    moved = curricula.record("cur").model_copy(update={"current_version": 2, "current_version_id": "v2"})
    curricula.save_record(moved)
    with pytest.raises(RecordConflict):  # e.g. a status change from before the replan: it would move the pointer back
        curricula.save_record(read.model_copy(update={"status": CurriculumStatus.COMPLETED}))
    assert curricula.record("cur").current_version_id == "v2"


def test_a_status_change_racing_a_replan_keeps_the_new_version(tmp_path) -> None:
    from app.curriculum.engine import CurriculumEngine

    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    try:
        repo = SqlCurriculumRepository(sessions)
        now = utcnow()
        repo.save_record(CurriculumRecord(curriculum_id="cur", learner_id="l1", goal_id="g", title="t",
                                          domain="spanish", current_version=1, current_version_id="v1",
                                          created_at=now, updated_at=now))
        read = repo.record
        raced = {"n": 0}

        def record_then_replan(curriculum_id):
            found = read(curriculum_id)
            if not raced["n"]:  # a replan commits right after the status change read the pointer
                raced["n"] += 1
                repo.save_record(read(curriculum_id).model_copy(
                    update={"current_version": 2, "current_version_id": "v2"}))
            return found

        repo.record = record_then_replan
        CurriculumEngine(repo).set_status("cur", CurriculumStatus.COMPLETED)
        stored = read("cur")
        assert stored.status == CurriculumStatus.COMPLETED and stored.current_version_id == "v2"
    finally:
        dispose(sessions)


def test_an_existing_sqlite_database_gets_the_version_columns(tmp_path) -> None:
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    sessions = create_db(url)
    SqlLearnerRepository(sessions).save(LearnerProfile(learner_id="old"))
    with sessions.begin() as s:  # a database from before this change: no version columns
        s.execute(text("ALTER TABLE learners DROP COLUMN version"))
        s.execute(text("ALTER TABLE curricula DROP COLUMN version"))
    dispose(sessions)
    sessions = create_db(url)
    try:
        learners = SqlLearnerRepository(sessions)
        profile = learners.get("old")
        assert profile is not None and profile._row_version == 1
        profile.display_name = "upgraded"
        learners.save(profile)
        assert learners.get("old").display_name == "upgraded"
    finally:
        dispose(sessions)


async def test_a_lesson_still_runs_end_to_end_with_fenced_learner_writes(tmp_path) -> None:
    container, _ = open_container(tmp_path / "data")
    try:
        lid = add_demo_learner(container)
        task = await container.task_service.create_and_run(request=LEARNER["request"], learner_id=lid, user_id="u1")
        task = await finish(container, task.task_id)
        assert task.status == TaskStatus.COMPLETED
        evaluation = await container.task_service.start_evaluation(task.task_id, user_id="u1")
        evaluation = await container.task_service.submit_answers(evaluation.task_id,
                                                                 evaluation_answers_for(evaluation))
        assert evaluation.status == TaskStatus.COMPLETED
        profile = container.memory.get(lid)
        assert [x.task_id for x in profile.lessons] == [task.task_id]
        assert [a.kind for a in profile.assessments if a.task_id == evaluation.task_id] == ["lesson_evaluation"]
    finally:
        container.close()
