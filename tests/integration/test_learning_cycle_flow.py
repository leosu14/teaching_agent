"""Learning cycles end to end on the mock providers: LEARN, REVIEW, EVALUATE and COMPLETE cycles; WAITING and resume
after learner input; idempotent starts and responses; a crash after each checkpoint; provider, workflow and
assessment failures; invalid and stale actions; concurrency; the artifact graph; event payloads."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import timedelta

import pytest

from app.curriculum.cycle import InvalidCycleRequest, InvalidCycleTransition
from app.schemas.common import utcnow
from app.schemas.curriculum import LearningActionType, NextLearningAction
from app.schemas.learning_cycle import (
    CycleConflict,
    CycleResponse,
    CycleStatus,
    FailureKind,
    StartLearningCycle,
    StepKind,
)
from app.schemas.lesson import LearnerAnswer
from app.schemas.task import TaskStatus
from app.schemas.teaching import TeachingSessionStatus
from app.services import learning_cycles as cycle_service
from scripts import run_curriculum_demo as curriculum_demo
from tests.learning_cycle_fixtures import CONCEPT, LEARNER, CycleEnv, ScriptedLearner, answer

LIFECYCLE = ("learning_cycle.started", "learning_cycle.action_selected", "learning_cycle.action_completed",
             "learning_cycle.completed")


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing catches it."""


def cycle_id(env: CycleEnv, key: str = "c1") -> str:
    return env.container.learning_cycle_service.repository.get(
        cycle_service.engine.cycle_id_for(LEARNER, key)).cycle_id


def lessons(env: CycleEnv) -> list:
    return [t for t in env.container.task_service.list_for_learner(LEARNER)
            if t.plan and t.plan.workflow_id == "lesson_generation"]


def evaluations(env: CycleEnv) -> list:
    return [t for t in env.container.task_service.list_for_learner(LEARNER)
            if t.plan and t.plan.workflow_id == "lesson_evaluation"]


def footprint(env: CycleEnv) -> dict:
    """Everything a duplicated side effect would change: tasks, sessions, teaching turns, grades, evidence, learning
    events, mastery, the curricula and their progress, artifacts, cycles."""
    c = env.container
    tasks = c.task_service.list_for_learner(LEARNER)
    artifacts = [a for t in tasks for a in c.task_service.artifacts(t.task_id)]
    cycles = env.cycles.repository.for_learner(LEARNER)
    artifacts += [a for cy in cycles for a in env.cycles.artifacts(cy.cycle_id)]
    sessions = c.teaching_service.sessions(LEARNER)
    profile = c.memory.get(LEARNER)
    curricula = []
    for goal in c.curriculum_service.goals(LEARNER):
        cur = c.curriculum_service.curriculum(goal.goal_id)
        curricula.append((goal.status.value, cur.version if cur else None, [
            (o.concept_id, o.status.value, o.evidence_count) for o in cur.progress.objectives] if cur else None))
    return {"tasks": sorted((t.plan.workflow_id if t.plan else "-", t.status.value) for t in tasks),
            "sessions": sorted(s.status.value for s in sessions),
            "turns": sum(len(c.teaching_service.turns(s.session_id)) for s in sessions),
            "artifacts": dict(sorted(Counter(a.type.value for a in artifacts).items())),
            "evidence": len(c.memory.evidence(LEARNER)),
            "learning_events": len(c.memory.history(LEARNER)),
            "mastery": {k: round(m.mastery, 4) for k, m in sorted(profile.concepts.items())},
            "evidence_counts": {k: m.evidence_count for k, m in sorted(profile.concepts.items())},
            "curricula": curricula, "cycles": sorted(cy.status.value for cy in cycles)}


def cycle_events(env: CycleEnv, cid: str) -> Counter:
    return Counter(e.type for e in env.cycles.events(cid))


async def objective_action(env: CycleEnv, action: LearningActionType, concept: str = CONCEPT) -> NextLearningAction:
    """A curriculum action for `concept`, as the NextActionEngine would select it (for actions the fixture learner's
    state does not reach on its own)."""
    real = await env.container.curriculum_service.next_action(LEARNER)
    goal_id = env.container.curriculum_service.goals(LEARNER)[0].goal_id
    cur = env.container.curriculum_service.curriculum(goal_id)
    objective = next(o for o in cur.objectives if o.concept_id == concept)
    return real.model_copy(update={"action": action, "goal_id": goal_id, "curriculum_id": cur.curriculum_id,
                                   "objective_description": objective.description,
                                   "curriculum_version": cur.version, "objective_id": objective.objective_id,
                                   "concept_id": concept, "action_id": f"act_test_{action.value}_{concept}"})


def select(monkeypatch, env: CycleEnv, action: NextLearningAction) -> None:
    async def fixed(_learner_id, *, as_of=None):
        return action
    monkeypatch.setattr(env.container.curriculum_service, "next_action", fixed)


async def secure(env: CycleEnv, *concepts: str, tag: str = "secured") -> None:
    for concept in concepts:
        await env.container.learner_service.record_evidence(
            LEARNER, "spanish", curriculum_demo.exercise_evidence(LEARNER, concept, 6, tag))


# --- LEARN ---------------------------------------------------------------------------------------------------------


async def test_a_learn_cycle_runs_the_lesson_and_the_session_into_mastery_and_the_next_action(cycle_env_at) -> None:
    env = cycle_env_at("done")
    cid = cycle_id(env)
    view = await env.cycles.get(cid)
    assert view.status == CycleStatus.COMPLETED and view.action.action == LearningActionType.LEARN
    assert view.action.concept_id == CONCEPT
    assert [(s.kind, s.status.value, s.reused) for s in view.steps] == [
        (StepKind.LESSON, "COMPLETED", False), (StepKind.TEACHING_SESSION, "COMPLETED", False)]

    # One lesson, generated for the cycle by the lesson workflow (found again by its step key); one session on it.
    [lesson] = lessons(env)
    assert lesson.task_id == view.steps[0].child_id and lesson.status == TaskStatus.COMPLETED
    assert lesson.metadata[cycle_service.STEP_KEY] == view.steps[0].child_key
    [session] = env.container.teaching_service.sessions(LEARNER)
    assert session.session_id == view.steps[1].child_id and session.task_id == lesson.task_id
    assert session.status == TeachingSessionStatus.COMPLETED and session.action == "LEARN"

    # The outcome is the session's own: evidence recorded by learner memory, the curriculum's next action.
    o = view.outcome
    assert o.result.value == "TAUGHT" and o.learning_evidence_ids == session.outcome.learning_evidence_ids
    stored = {e.evidence_id: e for e in env.container.memory.evidence(LEARNER)}
    assert all(stored[e].source_type == "interaction" for e in o.learning_evidence_ids)
    # Mastery is learner memory's: the cycle reports the session's own MasteryChange, which is the stored value.
    [change] = [c for c in o.mastery_changes if c["concept_id"] == CONCEPT]
    assert change == session.outcome.mastery_changes[[c["concept_id"] for c in session.outcome.mastery_changes]
                                                     .index(CONCEPT)]
    assert round(env.container.memory.get(LEARNER).concepts[CONCEPT].mastery, 6) == round(change["after"], 6)
    assert o.objective_progress == session.outcome.objective_progress
    fresh = await env.container.curriculum_service.next_action(LEARNER)
    assert (o.next_action.action, o.next_action.concept_id) == (fresh.action, fresh.concept_id)
    assert o.objective_progress["concept_id"] == CONCEPT

    # Each lifecycle event once, stored under the cycle id.
    events = cycle_events(env, cid)
    assert all(events[t] == 1 for t in LIFECYCLE) and events["learning_cycle.step_completed"] == 2
    assert events["learning_cycle.waiting"] == events["learning_cycle.response_received"]


async def test_the_artifact_graph_runs_goal_curriculum_objective_cycle_lesson_assessment_evidence(cycle_env_at) -> None:
    env = cycle_env_at("done")
    view = await env.cycles.get(cycle_id(env))
    arts = env.container.artifacts
    cycle_art = arts.get(view.artifact_ids["cycle"])
    assert cycle_art.type.value == "LEARNING_CYCLE" and cycle_art.task_id == view.cycle_id
    [objective] = [arts.get(p) for p in cycle_art.parent_ids]
    assert objective.type.value == "LEARNING_OBJECTIVE"
    assert {a.type.value for a in arts.lineage(objective.artifact_id)} >= {"CURRICULUM_VERSION", "LEARNING_GOAL"}

    lesson = arts.get(view.outcome.artifact_ids["lesson"])
    assert lesson.type.value == "LESSON" and cycle_art.artifact_id in lesson.parent_ids
    assert objective.artifact_id not in lesson.parent_ids  # Objective -> Cycle -> Lesson, not around the cycle
    action = arts.find(lesson.task_id, "learning_action")
    assert action.parent_ids == [cycle_art.artifact_id]

    lesson_task = view.steps[0].child_id
    grades = [a for a in env.container.task_service.artifacts(lesson_task) if a.type.value == "ASSESSMENT_GRADE"]
    assert grades and all(lesson.artifact_id in {x.artifact_id for x in arts.lineage(g.artifact_id)} for g in grades)
    evidence = arts.get(view.outcome.artifact_ids["learning_evidence"])
    ancestry = {a.artifact_id for a in arts.lineage(evidence.artifact_id)}
    assert {lesson.artifact_id, cycle_art.artifact_id, objective.artifact_id} <= ancestry

    outcome = arts.get(view.outcome.artifact_ids["outcome"])
    assert outcome.type.value == "LEARNING_CYCLE"
    assert {cycle_art.artifact_id, lesson.artifact_id, view.outcome.artifact_ids["learning_action"]} == \
        set(outcome.parent_ids)
    assert {a.name for a in env.cycles.artifacts(view.cycle_id)} == {"learning_cycle", "learning_cycle_outcome"}


# --- WAITING and learner input -------------------------------------------------------------------------------------


async def test_waiting_returns_and_learner_input_resumes_the_cycle(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic")
    view = await env.cycles.get(cycle_id(env))
    assert view.status == CycleStatus.WAITING and view.waiting.kind == "diagnostic_answers"
    assert view.prompt.kind == "DIAGNOSTIC_QUESTIONS" and view.prompt.questions
    assert all(set(q) <= {"question_id", "concept_id", "prompt", "kind", "choices", "difficulty"}
               for q in view.prompt.questions)  # no answer key
    assert [s.status.value for s in view.steps] == ["WAITING", "PENDING"]
    [lesson] = lessons(env)
    assert lesson.status == TaskStatus.WAITING  # the lesson waits, nothing runs in the process

    view = await answer(env, view, ScriptedLearner(prefix="d"), until="SESSION_QUESTION")
    assert view.status == CycleStatus.WAITING and view.prompt.kind == "SESSION_QUESTION"
    assert view.prompt.question is not None and view.prompt.teacher_turns
    assert "expected_answer" not in view.prompt.model_dump_json()
    assert lessons(env)[0].status == TaskStatus.COMPLETED and view.steps[1].child_id is not None


async def test_learner_input_given_through_the_session_endpoint_is_folded_in(cycle_env_at) -> None:
    from app.schemas.teaching import LearnerInput

    env = cycle_env_at("session")
    view = await env.cycles.get(cycle_id(env))
    learner = ScriptedLearner()
    sid = view.steps[1].child_id
    while (s := await env.container.teaching_service.view(sid)).status == TeachingSessionStatus.WAITING_FOR_LEARNER:
        await env.container.teaching_service.submit(sid, LearnerInput(answer=learner.answer(
            s.state.waiting_question.prompt), client_turn_id=f"direct-{s.turn_count}"))
    view = await env.cycles.get(view.cycle_id)  # a read reconciles, but does not drive the finish
    assert view.status == CycleStatus.RUNNING and all(s.status.value == "COMPLETED" for s in view.steps)
    view = await env.cycles.resume(view.cycle_id)
    assert view.status == CycleStatus.COMPLETED and view.outcome.result.value == "TAUGHT"


# --- idempotency ---------------------------------------------------------------------------------------------------


async def test_the_same_key_is_the_same_cycle_and_another_key_conflicts(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic")
    before = footprint(env)
    again = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    assert again.cycle_id == cycle_id(env) and not again.created and again.status == CycleStatus.WAITING
    assert footprint(env) == before
    with pytest.raises(CycleConflict, match="active learning cycle"):
        await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c2"))
    assert footprint(env) == before


async def test_repeated_and_conflicting_responses(cycle_env_at) -> None:
    env = cycle_env_at("session")
    cid = cycle_id(env)
    view = await env.cycles.get(cid)
    learner = ScriptedLearner(prefix="x")
    response = learner.respond(view.prompt)
    first = await env.cycles.respond(cid, response)
    after = footprint(env)
    events = cycle_events(env, cid)
    replay = await env.cycles.respond(cid, response)
    assert replay.replayed and replay.steps == first.steps and replay.prompt == first.prompt
    assert footprint(env) == after and cycle_events(env, cid) == events
    with pytest.raises(CycleConflict, match="different response"):
        await env.cycles.respond(cid, CycleResponse(client_response_id=response.client_response_id, answer="otro"))
    with pytest.raises(InvalidCycleRequest, match="session question"):
        await env.cycles.respond(cid, CycleResponse(client_response_id="sheet", answers=[
            LearnerAnswer(question_id="q", answer="a")]))
    assert footprint(env) == after


async def test_an_invalid_sheet_is_refused_and_changes_nothing(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic")
    cid = cycle_id(env)
    before = footprint(env)
    with pytest.raises(InvalidCycleRequest):
        await env.cycles.respond(cid, CycleResponse(client_response_id="bad", answers=[
            LearnerAnswer(question_id="no-such-question", answer="x")]))
    view = await env.cycles.get(cid)
    assert view.status == CycleStatus.WAITING and view.waiting.kind == "diagnostic_answers"
    assert footprint(env) == before
    view = await env.cycles.respond(cid, ScriptedLearner(prefix="bad").respond(view.prompt))  # the id is free again
    assert view.prompt.kind == "SESSION_QUESTION"


async def test_a_response_when_nothing_is_asked_is_refused(cycle_env_at) -> None:
    env = cycle_env_at("done")
    with pytest.raises(InvalidCycleTransition, match="not waiting"):
        await env.cycles.respond(cycle_id(env), CycleResponse(client_response_id="late", answer="x"))


async def test_replays_after_completion_duplicate_nothing(cycle_env_at) -> None:
    env = cycle_env_at("done")
    cid = cycle_id(env)
    before, events = footprint(env), cycle_events(env, cid)
    for _ in range(2):
        assert (await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))).cycle_id == cid
        await env.cycles.get(cid)
        await env.cycles.resume(cid)
    env = env.reopen()
    await env.cycles.get(cid)
    assert footprint(env) == before and cycle_events(env, cid) == events


# --- crashes -------------------------------------------------------------------------------------------------------


def crash_once(monkeypatch, target, name: str, when=lambda *a, **k: True):
    original = getattr(target, name)
    state = {"crashed": False}

    def wrapper(*args, **kwargs):
        if not state["crashed"] and when(*args, **kwargs):
            state["crashed"] = True
            raise SimulatedCrash(name)
        return original(*args, **kwargs)

    async def awrapper(*args, **kwargs):
        if not state["crashed"] and when(*args, **kwargs):
            state["crashed"] = True
            raise SimulatedCrash(name)
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, name, awrapper if asyncio.iscoroutinefunction(original) else wrapper)
    return state


def restart(env: CycleEnv, monkeypatch) -> CycleEnv:
    """The crashed process is gone: a new process on the same data. The crashed request still holds the drive lease
    until it expires; the new process's clock is past that."""
    monkeypatch.undo()
    env = env.reopen()
    env.cycles._clock = lambda: utcnow() + cycle_service.DRIVE_LEASE + timedelta(seconds=1)
    return env


def crashed_holding_the_lease(env: CycleEnv, cid: str) -> None:
    cycle = env.cycles.repository.get(cid)
    assert cycle.lease is not None and cycle.lease.until > utcnow()


def ran_once(env: CycleEnv, cid: str, steps: int = 2) -> None:
    """Each lifecycle event, each step event and each cycle artifact exactly once."""
    events = cycle_events(env, cid)
    assert all(events[t] == 1 for t in LIFECYCLE), events
    assert events["learning_cycle.step_started"] == events["learning_cycle.step_completed"] == steps, events
    ids = [e.event_id for e in env.cycles.events(cid)]
    assert len(ids) == len(set(ids))
    assert Counter(a.name for a in env.cycles.artifacts(cid)) == {"learning_cycle": 1, "learning_cycle_outcome": 1}
    assert not env.cycles.repository.get(cid).pending_events


async def test_a_crash_after_the_cycle_is_created_continues_with_the_same_action(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("diagnostic"))
    env = cycle_env_at("base")
    crash_once(monkeypatch, cycle_service.LearningCycleService, "_store_cycle_artifact")
    with pytest.raises(SimulatedCrash):
        await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    cid = cycle_id(env)
    stored = env.cycles.repository.get(cid)
    assert stored.status == CycleStatus.RUNNING and lessons(env) == []
    crashed_holding_the_lease(env, cid)
    env = restart(env, monkeypatch)
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    assert view.action == stored.action and view.prompt.kind == "DIAGNOSTIC_QUESTIONS"
    assert footprint(env) == baseline  # exactly the state of a run without the crash


async def test_a_crash_after_the_lesson_is_created_finds_it_by_key(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("diagnostic"))
    env = cycle_env_at("base")
    crash_once(monkeypatch, cycle_service.engine, "attach_child")
    with pytest.raises(SimulatedCrash):
        await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    [created] = lessons(env)
    assert created.status == TaskStatus.CREATED and env.cycles.repository.get(cycle_id(env)).steps[0].child_id is None
    env = restart(env, monkeypatch)
    view = await env.cycles.resume(cycle_id(env))
    assert view.steps[0].child_id == created.task_id and view.prompt.kind == "DIAGNOSTIC_QUESTIONS"
    assert footprint(env) == baseline


async def test_a_crash_while_the_lesson_runs_resumes_it_and_the_lease_stops_a_second_run(cycle_env_at,
                                                                                       monkeypatch) -> None:
    baseline = footprint(cycle_env_at("diagnostic"))
    env = cycle_env_at("base")
    tasks = env.container.task_service
    crash_once(monkeypatch, tasks, "run",  # the lesson's run (the curriculum's planning task runs before it)
               when=lambda task_id: tasks.get(task_id).plan.workflow_id == "lesson_generation")
    with pytest.raises(SimulatedCrash):
        await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    cid = cycle_id(env)
    crashed_holding_the_lease(env, cid)
    monkeypatch.undo()
    env = env.reopen()
    runs = []
    original = env.container.task_service.run

    async def counted(task_id):
        runs.append(task_id)
        return await original(task_id)

    monkeypatch.setattr(env.container.task_service, "run", counted)
    busy = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))  # a client retry, too early
    assert busy.status == CycleStatus.RUNNING and busy.busy_until is not None and busy.prompt is None
    assert runs == []  # the retry did not run the lesson a second time
    env.cycles._clock = lambda: utcnow() + cycle_service.DRIVE_LEASE + timedelta(seconds=1)
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    assert view.prompt.kind == "DIAGNOSTIC_QUESTIONS" and view.busy_until is None and len(runs) == 1
    monkeypatch.undo()
    assert footprint(env) == baseline


async def test_a_crash_after_a_response_is_received_delivers_it_once(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("done"))
    env = cycle_env_at("session")
    cid = cycle_id(env)
    view = await env.cycles.get(cid)
    sid = view.steps[1].child_id
    turns = len(env.container.teaching_service.turns(sid))
    response = ScriptedLearner(prefix="x").respond(view.prompt)
    crash_once(monkeypatch, cycle_service.LearningCycleService, "_deliver")
    with pytest.raises(SimulatedCrash):
        await env.cycles.respond(cid, response)
    assert env.cycles.repository.request(cid, response.client_response_id) is not None  # received, durable
    assert len(env.container.teaching_service.turns(sid)) == turns  # but never delivered
    env = restart(env, monkeypatch)
    assert (await env.cycles.get(cid)).status == CycleStatus.WAITING  # the session still waits for it
    replay = await env.cycles.respond(cid, response)
    assert replay.replayed and replay.prompt.kind == "SESSION_QUESTION"
    after = env.container.teaching_service.turns(sid)
    assert [t.speaker.value for t in after[turns:]].count("LEARNER") == 1
    again = await env.cycles.respond(cid, response)  # and once more: nothing new
    assert len(env.container.teaching_service.turns(sid)) == len(after) and again.steps == replay.steps
    done = await answer(env, again, ScriptedLearner(prefix="s"))
    assert done.status == CycleStatus.COMPLETED
    assert footprint(env) == baseline
    ran_once(env, cid)


async def test_a_crash_after_the_child_finished_is_reconciled(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("session"))
    env = cycle_env_at("diagnostic")
    cid = cycle_id(env)
    view = await env.cycles.get(cid)
    crash_once(monkeypatch, cycle_service.engine, "observe",
               when=lambda cycle, step_id, obs, at: obs.state == "completed")
    with pytest.raises(SimulatedCrash):
        await env.cycles.respond(cid, ScriptedLearner(prefix="d").respond(view.prompt))
    assert lessons(env)[0].status == TaskStatus.COMPLETED  # the side effect is durable
    assert env.cycles.repository.get(cid).steps[0].status.value == "RUNNING"  # the cycle had not recorded it
    env = restart(env, monkeypatch)
    view = await env.cycles.resume(cid)
    assert view.prompt.kind == "SESSION_QUESTION"
    assert footprint(env) == baseline  # one lesson, one session, the diagnostic's evidence and mastery once


async def test_a_crash_before_the_outcome_is_recorded_finishes_once(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("done"))
    env = cycle_env_at("session")
    cid = cycle_id(env)
    crash_once(monkeypatch, cycle_service.LearningCycleService, "_finish")
    with pytest.raises(SimulatedCrash):
        await answer(env, await env.cycles.get(cid), ScriptedLearner(prefix="s"))
    assert env.cycles.repository.get(cid).status == CycleStatus.RUNNING
    env = restart(env, monkeypatch)
    view = await env.cycles.resume(cid)
    assert view.status == CycleStatus.COMPLETED
    assert footprint(env) == baseline  # no evidence, mastery update, curriculum change or task twice
    ran_once(env, cid)


async def test_a_crash_after_the_outcome_artifact_completes_without_a_second_one(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("done"))
    env = cycle_env_at("session")
    cid = cycle_id(env)
    crash_once(monkeypatch, cycle_service.engine, "complete")
    with pytest.raises(SimulatedCrash):
        await answer(env, await env.cycles.get(cid), ScriptedLearner(prefix="s"))
    assert [a.name for a in env.cycles.artifacts(cid)].count("learning_cycle_outcome") == 1
    env = restart(env, monkeypatch)
    assert (await env.cycles.resume(cid)).status == CycleStatus.COMPLETED
    assert footprint(env) == baseline
    ran_once(env, cid)


async def test_a_crash_before_events_are_published_publishes_them_once(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("session")
    cid = cycle_id(env)
    original = cycle_service.LearningCycleService._publish

    async def crashing(self, cycle):
        if any(e.type == "learning_cycle.completed" for e in cycle.pending_events):
            raise SimulatedCrash("publish")
        return await original(self, cycle)

    monkeypatch.setattr(cycle_service.LearningCycleService, "_publish", crashing)
    with pytest.raises(SimulatedCrash):
        await answer(env, await env.cycles.get(cid), ScriptedLearner(prefix="s"))
    assert env.cycles.repository.get(cid).status == CycleStatus.COMPLETED
    assert cycle_events(env, cid)["learning_cycle.completed"] == 0
    env = restart(env, monkeypatch)
    for _ in range(3):
        await env.cycles.get(cid)
    types = [e.type for e in env.cycles.events(cid) if e.type.startswith("learning_cycle.")]
    assert types[-2:] == ["learning_cycle.action_completed", "learning_cycle.completed"]  # in order
    ran_once(env, cid)


async def test_a_completed_cycle_ran_each_event_and_artifact_once(cycle_env_at) -> None:
    env = cycle_env_at("done")
    cid = cycle_id(env)
    ran_once(env, cid)
    types = [e.type.removeprefix("learning_cycle.") for e in env.cycles.events(cid)
             if e.type.startswith("learning_cycle.")]
    assert types[:3] == ["started", "action_selected", "step_started"] and types[-2:] == ["action_completed",
                                                                                          "completed"]


# --- failures ------------------------------------------------------------------------------------------------------


async def test_a_provider_failure_blocks_the_cycle_and_resume_retries(cycle_env_at) -> None:
    env = cycle_env_at("session")
    cid = cycle_id(env)
    view = await env.cycles.get(cid)
    env.llm.inject("teaching_session", *[RuntimeError("provider down")] * 20)
    blocked = await env.cycles.respond(cid, ScriptedLearner(prefix="x").respond(view.prompt))
    assert blocked.status == CycleStatus.BLOCKED and blocked.failure.kind == FailureKind.PROVIDER
    assert blocked.failure.retryable and blocked.prompt is None
    assert cycle_events(env, cid)["learning_cycle.failed"] == 1
    with pytest.raises(CycleConflict):  # a blocked cycle keeps the learner's slot
        await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c2"))
    still = await env.cycles.resume(cid)  # the provider is still down
    assert still.status == CycleStatus.BLOCKED
    env.llm._faults["teaching_session"].clear()
    resumed = await env.cycles.resume(cid)
    assert resumed.status == CycleStatus.WAITING and resumed.prompt.kind == "SESSION_QUESTION"
    done = await answer(env, resumed, ScriptedLearner(prefix="y"))
    assert done.status == CycleStatus.COMPLETED


async def test_a_failed_lesson_task_blocks_and_resumes_from_its_checkpoint(cycle_env_at) -> None:
    env = cycle_env_at("base")
    env.llm.inject("teacher", *[RuntimeError("provider down")] * 40)
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1"))
    if view.status == CycleStatus.WAITING:  # the diagnostic comes before the teacher
        view = await env.cycles.respond(view.cycle_id, ScriptedLearner(prefix="d").respond(view.prompt))
    assert view.status == CycleStatus.BLOCKED and view.failure.kind in (FailureKind.PROVIDER, FailureKind.WORKFLOW)
    assert view.failure.retryable and view.failure.category
    [lesson] = lessons(env)
    assert lesson.status == TaskStatus.FAILED
    env.llm._faults["teacher"].clear()
    view = await env.cycles.resume(view.cycle_id)
    assert view.status == CycleStatus.WAITING and view.prompt.kind == "SESSION_QUESTION"
    assert len(lessons(env)) == 1 and lessons(env)[0].status == TaskStatus.COMPLETED


async def test_a_failed_assessment_blocks_the_evaluation_and_resume_completes_it(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("done")
    # The LEARN cycle mastered es.preterite (an EVALUATE there is stale); the next objective has no lesson yet.
    select(monkeypatch, env, await objective_action(env, LearningActionType.EVALUATE, "es.past_contrast"))
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="eval"))
    view = await answer(env, view, ScriptedLearner(prefix="e"), until="ASSESSMENT_QUESTIONS")
    assert [(s.kind, s.reused) for s in view.steps] == [(StepKind.LESSON, False), (StepKind.EVALUATION, False)]
    assert view.status == CycleStatus.WAITING and view.prompt.kind == "ASSESSMENT_QUESTIONS"
    assert all("answer" not in q and "expected" not in json.dumps(q) for q in view.prompt.questions)
    assert len(lessons(env)) == 2 and evaluations(env)[0].status == TaskStatus.WAITING
    env.llm.inject("learner_evaluation", *["not json {"] * 40)
    view = await env.cycles.respond(view.cycle_id, ScriptedLearner(prefix="e").respond(view.prompt))
    assert view.status == CycleStatus.BLOCKED and view.failure.kind == FailureKind.WORKFLOW
    assert view.failure.category == "EvaluationError" and view.failure.retryable
    env.llm._faults["learner_evaluation"].clear()
    before = footprint(env)["evidence"]
    view = await env.cycles.resume(view.cycle_id)
    assert view.status == CycleStatus.COMPLETED and view.outcome.result.value == "EVALUATED"
    [evaluation] = evaluations(env)
    assert view.steps[1].child_id == evaluation.task_id
    assert view.outcome.learning_evidence_ids and footprint(env)["evidence"] == before + len(
        view.outcome.learning_evidence_ids)
    assert view.outcome.next_action == evaluation.result.learning_action


# --- REVIEW and COMPLETE -------------------------------------------------------------------------------------------


async def test_a_review_cycle_reuses_the_objectives_lesson(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("done")
    [lesson] = lessons(env)
    select(monkeypatch, env, await objective_action(env, LearningActionType.REVIEW))
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="review"))
    assert view.steps[0].reused and view.steps[0].child_id == lesson.task_id
    assert view.prompt.kind == "SESSION_QUESTION" and len(lessons(env)) == 1
    before = footprint(env)
    view = await answer(env, view, ScriptedLearner(prefix="v"))
    assert view.status == CycleStatus.COMPLETED and view.outcome.result.value == "TAUGHT"
    sessions = env.container.teaching_service.sessions(LEARNER)
    review = next(s for s in sessions if s.session_id == view.steps[1].child_id)
    assert review.action == "REVIEW" and review.task_id == lesson.task_id
    assert footprint(env)["evidence"] == before["evidence"] + len(view.outcome.learning_evidence_ids)


async def test_after_completion_a_due_review_is_selected_and_executed(cycle_env_at) -> None:
    env = cycle_env_at("done")
    await secure(env, "es.present", "es.preterite", "es.past_contrast", "es.subjunctive")
    done = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="complete"))
    assert done.action.action == LearningActionType.COMPLETE and done.outcome.goal_complete
    env.cycles._clock = lambda: utcnow() + timedelta(days=120)  # months later: reviews are due
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="later"))
    assert view.action.action == LearningActionType.REVIEW
    assert view.steps[0].reused == (view.action.concept_id == CONCEPT)
    view = await answer(env, view, ScriptedLearner(prefix="l"))
    assert view.status == CycleStatus.COMPLETED and view.steps[1].session_action == "REVIEW"


async def test_a_complete_cycle_verifies_the_completion_rule_and_generates_nothing(cycle_env_at) -> None:
    env = cycle_env_at("done")
    await secure(env, "es.present", "es.preterite", "es.past_contrast", "es.subjunctive")
    before = footprint(env)
    calls = sum(env.llm.calls.values())
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="complete"))
    assert view.action.action == LearningActionType.COMPLETE and view.steps == []
    assert view.status == CycleStatus.COMPLETED and view.outcome.result.value == "GOAL_COMPLETED"
    assert view.outcome.goal_complete and view.outcome.next_action is not None
    assert view.outcome.learning_evidence_ids == [] and view.outcome.mastery_changes == []
    assert sum(env.llm.calls.values()) == calls  # no model call
    after = footprint(env)
    assert after.pop("cycles") == ["COMPLETED", "COMPLETED"] and before.pop("cycles") == ["COMPLETED"]
    assert after["artifacts"].pop("LEARNING_CYCLE") == before["artifacts"].pop("LEARNING_CYCLE") + 2
    # The goal is completed by the curriculum (its tracker, when it selected COMPLETE), not by the cycle.
    [(status_before, *rest_before)], [(status_after, *rest_after)] = before.pop("curricula"), after.pop("curricula")
    assert (status_before, status_after) == ("ACTIVE", "COMPLETED") and rest_before == rest_after
    assert after == before  # no teaching, assessment, evidence or mastery change
    events = cycle_events(env, view.cycle_id)
    assert all(events[t] == 1 for t in LIFECYCLE) and "goal.completed" not in events


async def test_complete_fails_when_the_completion_rule_does_not_hold(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("done")
    real = await env.container.curriculum_service.next_action(LEARNER)
    select(monkeypatch, env, real.model_copy(update={"action": LearningActionType.COMPLETE, "objective_id": None,
                                                     "concept_id": None}))
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="not-done"))
    assert view.status == CycleStatus.FAILED and view.failure.kind == FailureKind.VALIDATION
    assert not view.failure.retryable and "completion rule" in view.failure.message
    assert env.cycles.repository.active_for(LEARNER) is None  # a permanent failure frees the slot


async def test_a_learner_without_goals_gets_a_nothing_due_cycle(cycle_env_at) -> None:
    from app.schemas.learner import LearnerProfileInput

    env = cycle_env_at("base")
    env.container.learner_service.upsert("no-goals", LearnerProfileInput(display_name="x"))
    view = await env.cycles.start("no-goals", StartLearningCycle(idempotency_key="k"))
    assert view.action.action == LearningActionType.WAIT and view.status == CycleStatus.COMPLETED
    assert view.outcome.result.value == "NOTHING_DUE" and view.steps == []


# --- invalid actions -----------------------------------------------------------------------------------------------


async def test_an_action_on_an_already_mastered_objective_fails_without_a_lesson(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("done")
    await secure(env, CONCEPT)
    select(monkeypatch, env, await objective_action(env, LearningActionType.LEARN))
    before = footprint(env)
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="stale"))
    assert view.status == CycleStatus.FAILED and view.failure.kind == FailureKind.VALIDATION
    assert "already mastered" in view.failure.message and view.steps[0].status.value == "FAILED"
    assert footprint(env)["tasks"] == before["tasks"]
    with pytest.raises(InvalidCycleTransition):
        await env.cycles.resume(view.cycle_id)
    assert cycle_events(env, view.cycle_id)["learning_cycle.failed"] == 1


async def test_an_action_from_an_older_curriculum_version_is_refused(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("done")
    action = await objective_action(env, LearningActionType.PRACTICE)
    select(monkeypatch, env, action.model_copy(update={"curriculum_version": action.curriculum_version + 7}))
    view = await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="old"))
    assert view.status == CycleStatus.FAILED and "curriculum changed" in view.failure.message


async def test_cancel_stops_the_child_and_frees_the_slot(cycle_env_at) -> None:
    env = cycle_env_at("session")
    cid = cycle_id(env)
    view = await env.cycles.cancel(cid)
    assert view.status == CycleStatus.CANCELLED
    session = await env.container.teaching_service.get(view.steps[1].child_id)
    assert session.status == TeachingSessionStatus.CANCELLED
    assert (await env.cycles.cancel(cid)).status == CycleStatus.CANCELLED  # idempotent
    with pytest.raises(InvalidCycleTransition):
        await env.cycles.respond(cid, CycleResponse(client_response_id="z", answer="x"))
    assert env.cycles.repository.active_for(LEARNER) is None


# --- concurrency ---------------------------------------------------------------------------------------------------


def yielding(monkeypatch, target, name: str) -> None:
    """Let other requests run while this call is in progress (the mocks never yield on their own, so without this
    "concurrent" requests would simply run one after the other)."""
    original = getattr(target, name)

    async def slow(*args, **kwargs):
        for _ in range(5):
            await asyncio.sleep(0)
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, name, slow)


async def test_concurrent_starts_execute_one_action(cycle_env_at, monkeypatch) -> None:
    baseline = footprint(cycle_env_at("diagnostic"))
    env = cycle_env_at("base")
    yielding(monkeypatch, env.container.task_service, "run")
    yielding(monkeypatch, env.container.curriculum_service, "next_action")
    yielding(monkeypatch, env.container.curriculum_service, "create_lesson")
    results = await asyncio.gather(*(env.cycles.start(LEARNER, StartLearningCycle(idempotency_key=key))
                                      for key in ("c1", "other", "c1", "c1", "other")), return_exceptions=True)
    views = [r for r in results if not isinstance(r, BaseException)]
    errors = [r for r in results if isinstance(r, BaseException)]
    assert views and all(isinstance(e, CycleConflict) for e in errors), errors
    assert {v.cycle_id for v in views} == {cycle_id(env)}  # "c1" won; "other" was refused
    assert sum(v.prompt is not None for v in views) >= 1
    assert all(v.busy_until is not None for v in views if v.prompt is None)  # the others saw it busy
    monkeypatch.undo()
    assert footprint(env) == baseline  # one cycle, one curriculum, one lesson, run once


async def test_concurrent_responses_are_delivered_once(cycle_env_at, monkeypatch) -> None:
    env = cycle_env_at("session")
    cid = cycle_id(env)
    view = await env.cycles.get(cid)
    learner = ScriptedLearner(prefix="x")
    response = learner.respond(view.prompt)
    other = CycleResponse(client_response_id="y1", answer=response.answer)
    sid = view.steps[1].child_id
    turns = len(env.container.teaching_service.turns(sid))
    evidence = len(env.container.memory.evidence(LEARNER))
    yielding(monkeypatch, env.container.teaching_service, "submit")
    results = await asyncio.gather(env.cycles.respond(cid, response), env.cycles.respond(cid, response),
                                   env.cycles.respond(cid, other), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, BaseException)]
    assert ok and all(isinstance(r, CycleConflict) for r in results if isinstance(r, BaseException)), results
    new = env.container.teaching_service.turns(sid)[turns:]
    assert sum(t.speaker.value == "LEARNER" for t in new) == 1  # one answer applied, none lost silently
    assert len(env.container.memory.evidence(LEARNER)) <= evidence + 1
    replay = await env.cycles.respond(cid, response)  # the refused requests can retry: the winner replays
    assert replay.replayed or replay.prompt is not None


async def test_the_same_key_with_another_request_is_refused(cycle_env_at) -> None:
    env = cycle_env_at("diagnostic")
    before = footprint(env)
    with pytest.raises(CycleConflict, match="different request"):
        await env.cycles.start(LEARNER, StartLearningCycle(idempotency_key="c1", user_id="someone-else"))
    assert footprint(env) == before


# --- security ------------------------------------------------------------------------------------------------------


async def test_events_and_artifacts_carry_no_secret_or_learner_data(cycle_env_at, monkeypatch) -> None:
    secret = "sk-ant-test-secret-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    env = cycle_env_at("session")
    cid = cycle_id(env)
    view = await answer(env, await env.cycles.get(cid), ScriptedLearner(prefix="s"))
    assert view.status == CycleStatus.COMPLETED
    profile = env.container.memory.get(LEARNER)
    answers = {r.answer for r in ScriptedLearner().sent} | {"hablé", "cenamos"}
    for event in env.cycles.events(cid):
        assert event.type.startswith(("learning_cycle.", "artifact."))
        body = event.model_dump_json()
        assert secret not in body and LEARNER not in body and profile.display_name not in body
        assert not any(a in body for a in answers)
        assert "learner_id" not in event.data
    for artifact in env.cycles.artifacts(cid):
        content = env.container.artifacts.read(artifact.artifact_id).decode()
        assert secret not in content and profile.display_name not in content
