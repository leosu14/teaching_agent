"""The learning-cycle state machine (pure), its repositories and its request shapes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.curriculum import cycle as engine
from app.curriculum.repository import InMemoryLearningCycleRepository
from app.schemas.curriculum import LearningActionType, NextLearningAction, ObjectiveProgress, ObjectiveStatus
from app.schemas.learning_cycle import (
    CycleConflict,
    CycleFailure,
    CycleOutcome,
    CycleRequestRecord,
    CycleResponse,
    CycleResult,
    CycleStatus,
    FailureKind,
    StepKind,
    StepStatus,
)
from app.schemas.lesson import LearnerAnswer
from app.storage.db import create_db, dispose
from app.storage.repositories import SqlLearningCycleRepository
from tests.unit.test_architecture import violations_for

T0 = datetime(2026, 10, 5, 12, tzinfo=UTC)


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def action(kind: LearningActionType = LearningActionType.LEARN, **update) -> NextLearningAction:
    lesson = kind not in (LearningActionType.WAIT, LearningActionType.COMPLETE)
    fields = dict(action_id="act_1", learner_id="learner-1", action=kind, goal_id="goal_1",
                  curriculum_id="cur_1", curriculum_version=1, objective_id="obj_1" if lesson else None,
                  concept_id="es.preterite" if lesson else None, reason="selected", as_of=T0)
    return NextLearningAction(**{**fields, **update})


def progress(status: ObjectiveStatus, **update) -> ObjectiveProgress:
    fields = dict(objective_id="obj_1", concept_id="es.preterite", status=status, current_mastery=0.3,
                  target_mastery=0.8, evidence_count=1, evidence_required=3)
    return ObjectiveProgress(**{**fields, **update})


def started(kind: LearningActionType = LearningActionType.LEARN, key: str = "k1"):
    return engine.start(learner_id="learner-1", user_id="u", idempotency_key=key, action=action(kind), at=T0)


def types(cycle) -> list[str]:
    return [e.type.removeprefix("learning_cycle.") for e in cycle.pending_events]


def failure(kind: FailureKind = FailureKind.PROVIDER, retryable: bool = True, step_id: str | None = None,
            category: str | None = "ProviderError") -> CycleFailure:
    return CycleFailure(kind=kind, retryable=retryable, message="down", step_id=step_id, category=category, at=T0)


# --- policy and ids ------------------------------------------------------------------------------------------------


def test_every_action_type_has_a_step_policy() -> None:
    assert set(engine.STEP_POLICY) == set(LearningActionType)


@pytest.mark.parametrize(("kind", "steps", "reuse", "session"), [
    (LearningActionType.LEARN, [StepKind.LESSON, StepKind.TEACHING_SESSION], False, "LEARN"),
    (LearningActionType.REVIEW, [StepKind.LESSON, StepKind.TEACHING_SESSION], True, "REVIEW"),
    (LearningActionType.PRACTICE, [StepKind.LESSON, StepKind.TEACHING_SESSION], True, "PRACTICE"),
    (LearningActionType.EVALUATE, [StepKind.LESSON, StepKind.EVALUATION], True, None),
    (LearningActionType.COMPLETE, [], None, None),
    (LearningActionType.WAIT, [], None, None),
])
def test_actions_map_to_existing_workflows(kind, steps, reuse, session) -> None:
    planned = engine.plan_steps("lcyc_x", action(kind))
    assert [s.kind for s in planned] == steps
    if steps:
        assert planned[0].reuse_lesson is reuse  # LEARN always teaches a new lesson; the others may reuse one
        assert planned[-1].session_action == session
    assert all(s.status == StepStatus.PENDING and s.child_id is None for s in planned)


def test_ids_are_stable_and_never_random() -> None:
    assert engine.cycle_id_for("a", "k") == engine.cycle_id_for("a", "k")
    assert engine.cycle_id_for("a", "k") != engine.cycle_id_for("b", "k") != engine.cycle_id_for("a", "k2")
    one, two = started(), started()
    assert one.cycle_id == two.cycle_id and one.steps == two.steps
    assert [e.event_id for e in one.pending_events] == [e.event_id for e in two.pending_events]
    assert len({s.child_key for s in one.steps}) == len(one.steps)


@pytest.mark.parametrize(("kind", "status", "problem"), [
    (LearningActionType.LEARN, ObjectiveStatus.NOT_STARTED, None),
    (LearningActionType.PRACTICE, ObjectiveStatus.IN_PROGRESS, None),
    (LearningActionType.REVIEW, ObjectiveStatus.MASTERED, None),
    (LearningActionType.LEARN, ObjectiveStatus.MASTERED, "already mastered"),
    (LearningActionType.EVALUATE, ObjectiveStatus.MASTERED, "already mastered"),
    (LearningActionType.LEARN, ObjectiveStatus.BLOCKED, "blocked"),
])
def test_action_problems_are_decided_by_code(kind, status, problem) -> None:
    found = engine.action_problem(action(kind), progress(status, unmet_prerequisites=["es.present"]))
    assert (found is None) if problem is None else (problem in found)


def test_an_objective_missing_from_the_curriculum_is_a_problem() -> None:
    assert "not in the goal's current curriculum" in engine.action_problem(action(), None)
    assert engine.action_problem(action(LearningActionType.COMPLETE), None) is None


@pytest.mark.parametrize(("category", "kind", "retryable"), [
    ("ProviderError", FailureKind.PROVIDER, True),
    ("ConfigurationError", FailureKind.VALIDATION, False),
    ("BudgetExceededError", FailureKind.WORKFLOW, False),
    ("EvaluationError", FailureKind.WORKFLOW, True),
    (None, FailureKind.WORKFLOW, True),
])
def test_task_failures_are_classified(category, kind, retryable) -> None:
    assert engine.classify_task_failure(category) == (kind, retryable)


# --- transitions ---------------------------------------------------------------------------------------------------


def test_a_cycle_runs_through_its_steps_and_completes() -> None:
    c = started()
    assert c.status == CycleStatus.RUNNING and c.version == 1 and types(c) == ["started", "action_selected"]
    c = engine.published(c, {e.event_id for e in c.pending_events}, T0)
    lesson, session = c.steps
    c = engine.begin_step(c, lesson.step_id, at(1))
    with pytest.raises(engine.InvalidCycleTransition):
        engine.begin_step(c, lesson.step_id, at(1))
    c = engine.attach_child(c, lesson.step_id, "task_1", at(2))
    with pytest.raises(CycleConflict):
        engine.attach_child(c, lesson.step_id, "task_2", at(2))
    c = engine.observe(c, lesson.step_id, engine.Observation("waiting", ("diagnostic_answers", "node:1")), at(3))
    assert c.status == CycleStatus.WAITING and c.waiting.ref == "node:1"
    assert engine.observe(c, lesson.step_id, engine.Observation("waiting", ("diagnostic_answers", "node:1")),
                          at(4)) is None  # the same wait observed again changes nothing
    c = engine.receive(c, at(5))
    assert c.status == CycleStatus.RUNNING and c.waiting is None
    with pytest.raises(engine.InvalidCycleTransition):
        engine.receive(c, at(5))
    c = engine.observe(c, lesson.step_id, engine.Observation("completed"), at(6))
    assert engine.observe(c, lesson.step_id, engine.Observation("completed"), at(6)) is None
    assert c.current_step.step_id == session.step_id
    with pytest.raises(engine.InvalidCycleTransition):
        engine.complete(c, CycleOutcome(result=CycleResult.TAUGHT), at(7))
    c = engine.begin_step(c, session.step_id, at(7))
    c = engine.attach_child(c, session.step_id, "tsess_1", at(7))
    c = engine.observe(c, session.step_id, engine.Observation("completed"), at(8))
    assert c.current_step is None
    c = engine.complete(c, CycleOutcome(result=CycleResult.TAUGHT), at(9))
    assert c.status == CycleStatus.COMPLETED and c.completed_at == at(9)
    assert types(c) == ["step_started", "waiting", "response_received", "step_completed", "step_started",
                        "step_completed", "action_completed", "completed"]
    assert len({e.event_id for e in c.pending_events}) == len(c.pending_events)
    with pytest.raises(engine.InvalidCycleTransition):
        engine.cancel(c, at(10))


def test_a_cycle_without_steps_completes_with_action_completed() -> None:
    c = engine.complete(started(LearningActionType.COMPLETE),
                        CycleOutcome(result=CycleResult.GOAL_COMPLETED, goal_complete=True), at(1))
    assert types(c)[-2:] == ["action_completed", "completed"]


def test_retryable_failures_block_and_resume() -> None:
    c = started()
    step = c.steps[0].step_id
    c = engine.fail(c, failure(step_id=step), at(1))
    assert c.status == CycleStatus.BLOCKED and c.completed_at is None
    assert c.steps[0].status == StepStatus.PENDING  # the step is retried, not failed
    assert engine.fail(c, failure(step_id=step), at(2)) is None  # observed again: no second event
    c = engine.resume(c, at(3))
    assert c.status == CycleStatus.RUNNING and c.failure is None
    assert engine.resume(c, at(4)) is None
    assert types(c)[-2:] == ["failed", "resumed"]


def test_permanent_failures_are_final() -> None:
    c = started()
    step = c.steps[0].step_id
    c = engine.fail(c, failure(FailureKind.VALIDATION, False, step, None), at(1))
    assert c.status == CycleStatus.FAILED and c.steps[0].status == StepStatus.FAILED and c.completed_at == at(1)
    for transition in (engine.resume, engine.cancel):
        with pytest.raises(engine.InvalidCycleTransition):
            transition(c, at(2))
    with pytest.raises(engine.InvalidCycleTransition):
        engine.begin_step(c, c.steps[1].step_id, at(2))


def test_cancel_is_idempotent() -> None:
    c = engine.cancel(started(), at(1))
    assert c.status == CycleStatus.CANCELLED and types(c)[-1] == "cancelled"
    assert engine.cancel(c, at(2)) is None
    with pytest.raises(engine.InvalidCycleTransition):
        engine.resume(c, at(2))


def test_one_drive_lease_at_a_time_until_it_expires() -> None:
    c = started()
    held = engine.claim(c, at(0), at(10))
    assert held.lease.token.startswith("lclease_") and held.version == c.version + 1
    assert engine.claim(held, at(5), at(15)) is None  # held by another request
    observed = engine.observe(engine.attach_child(engine.begin_step(held, held.steps[0].step_id, at(1)),
                                                  held.steps[0].step_id, "t", at(1)),
                              held.steps[0].step_id, engine.Observation("running"), at(2))
    assert (observed or held).lease == held.lease  # other transitions keep the lease
    taken = engine.claim(held, at(11), at(21))  # a crashed holder's lease expired: taken over
    assert taken.lease.token != held.lease.token
    assert engine.release(taken, held.lease.token, at(12)) is None  # the old holder cannot release it
    assert engine.release(taken, taken.lease.token, at(12)).lease is None


def test_event_payloads_carry_ids_only() -> None:
    c = started()
    for event in c.pending_events:
        assert event.event_id.startswith("lcevt_")
        assert {"cycle_id", "action", "goal_id", "objective_id", "concept_id"} <= set(event.data)
        assert "learner-1" not in event.model_dump_json() and "learner_id" not in event.data


# --- repositories --------------------------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sql"])
def repo(request, tmp_path):
    if request.param == "memory":
        yield InMemoryLearningCycleRepository()
        return
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield SqlLearningCycleRepository(sessions)
    dispose(sessions)


def test_one_active_cycle_per_learner(repo) -> None:
    c = started()
    assert repo.create(c) and not repo.create(c)  # the same cycle again: not created, no error
    with pytest.raises(CycleConflict):
        repo.create(started(key="k2"))
    assert repo.active_for("learner-1").cycle_id == c.cycle_id
    repo.apply(engine.cancel(c, at(1)), expected_version=c.version)  # a terminal cycle frees the slot
    assert repo.active_for("learner-1") is None
    later = engine.start(learner_id="learner-1", user_id="u", idempotency_key="k2", action=action(), at=at(2))
    assert repo.create(later)
    assert [x.idempotency_key for x in repo.for_learner("learner-1")] == ["k2", "k1"]


def test_changes_apply_only_against_the_version_they_came_from(repo) -> None:
    c = started()
    repo.create(c)
    first = engine.claim(c, at(1), at(30))
    repo.apply(first, expected_version=1)
    with pytest.raises(CycleConflict):
        repo.apply(engine.claim(c, at(2), at(30)), expected_version=1)  # computed from a stale read
    assert repo.get(c.cycle_id).version == 2


def test_a_response_is_received_once(repo) -> None:
    c = started()
    repo.create(c)
    c = engine.observe(engine.attach_child(engine.begin_step(c, c.steps[0].step_id, T0), c.steps[0].step_id, "t",
                                           T0), c.steps[0].step_id,
                       engine.Observation("waiting", ("diagnostic_answers", "n")), T0)
    repo.apply(c, expected_version=1)
    record = CycleRequestRecord(cycle_id=c.cycle_id, client_response_id="r1", request_hash="h", step_id="s",
                                waiting_ref="n", created_at=T0)
    received = engine.receive(c, at(1))
    repo.apply(received, expected_version=c.version, request=record)
    with pytest.raises(CycleConflict):
        repo.apply(engine.claim(received, at(2), at(30)), expected_version=received.version, request=record)
    assert repo.get(c.cycle_id).version == received.version  # the whole change was refused
    assert not repo.request(c.cycle_id, "r1").applied
    repo.mark_applied(c.cycle_id, "r1")
    assert repo.request(c.cycle_id, "r1").applied
    repo.forget_request(c.cycle_id, "r1")
    assert repo.request(c.cycle_id, "r1") is None


def test_the_sql_repository_survives_a_restart(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    sessions = create_db(url)
    c = started()
    SqlLearningCycleRepository(sessions).create(c)
    dispose(sessions)
    sessions = create_db(url)
    try:
        repo = SqlLearningCycleRepository(sessions)
        assert repo.get(c.cycle_id) == c and repo.active_for("learner-1") == c
    finally:
        dispose(sessions)


# --- request shapes ------------------------------------------------------------------------------------------------


def test_a_response_is_a_sheet_or_an_answer() -> None:
    CycleResponse(client_response_id="r", answer="hablé")
    CycleResponse(client_response_id="r", answers=[LearnerAnswer(question_id="q", answer="a")])
    for bad in ({"client_response_id": "r"}, {"client_response_id": "r", "answer": "a", "answers": [
            {"question_id": "q", "answer": "a"}]}, {"client_response_id": "", "answer": "a"},
            {"client_response_id": "r", "answers": []}):
        with pytest.raises(ValidationError):
            CycleResponse.model_validate(bad)


# --- architecture --------------------------------------------------------------------------------------------------


def test_the_cycle_engine_stays_below_the_services() -> None:
    assert violations_for("curriculum", "app.services.learning_cycles", [], "x")
    assert violations_for("curriculum", "app.runtime.orchestrator", [], "x")
    assert violations_for("curriculum", "sqlalchemy", [], "x")
    assert violations_for("api", "app.curriculum.cycle", ["start"], "x")
    assert not violations_for("services", "app.curriculum.cycle", [], "x")
    assert not violations_for("services", "app.runtime.orchestrator", [], "x")


def test_the_cycle_modules_import_no_framework_or_vendor_sdk() -> None:
    from pathlib import Path

    from tests.unit.test_architecture import imports

    root = Path(__file__).resolve().parents[2]
    for rel in ("app/curriculum/cycle.py", "app/services/learning_cycles.py", "app/schemas/learning_cycle.py",
                "app/api/routes/learning_cycles.py"):
        modules = {m.split(".")[0] for m, _, _ in imports(root / rel)}
        assert not modules & {"langchain", "langgraph", "crewai", "anthropic", "openai", "httpx", "requests"}, rel
