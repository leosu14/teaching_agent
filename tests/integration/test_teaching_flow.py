"""Interactive teaching sessions end to end on a generated curriculum lesson: the full loop into learner memory and the
curriculum, resume after a process restart and after a crash between the two phases, idempotency, concurrency,
practice sessions, the model boundary (adversarial output), privacy of the model's context, events and artifacts."""

from __future__ import annotations

import asyncio
import json
from collections import Counter

import pytest

from app.config.providers import ProviderSettings
from app.schemas.artifact import ArtifactType
from app.schemas.task import TaskStatus
from app.schemas.teaching import (
    CompletionReason,
    EvidenceType,
    LearnerInput,
    SessionConflict,
    StartTeachingSession,
    TeachingSessionStatus,
)
from app.teaching.errors import InvalidSessionTransition, InvalidTeachingRequest, TeacherUnavailable
from tests.teaching_fixtures import KEY, Env, answer_for, open_env
from scripts.run_interactive_demo import scenario

SCRIPT = scenario()["script"]
CONCEPT = "es.past_contrast"
TEACHING = ("teaching_session.", "teaching_turn.", "learner_answer.", "hint.", "misconception.", "difficulty.")
AGENT = "teaching_session"


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing catches it."""


async def start(env: Env, key: str = "k1", **kw):
    return await env.service.start(env.lesson_task, StartTeachingSession(idempotency_key=key, **kw))


async def step(env: Env, sid: str, i: int, *, client_turn_id: str | None = None):
    """Turn `i` of the interactive demo's scripted learner."""
    item = SCRIPT[i]
    cid = client_turn_id or f"c{i + 1}"
    if item["kind"] == "question":
        return await env.service.submit(sid, LearnerInput(answer=item["say"], kind="question", client_turn_id=cid))
    view = await env.service.view(sid)
    text = answer_for(view.state.waiting_question, KEY, item["say"])
    return await env.service.submit(sid, LearnerInput(answer=text, client_turn_id=cid))


def teaching_events(env: Env):
    return [e for e in env.container.task_service.events(env.lesson_task) if e.type.startswith(TEACHING)]


def transcript(env: Env, sid: str) -> list[tuple]:
    return [(t.turn_id, t.sequence, t.speaker.value, t.turn_type.value, t.content)
            for t in env.service.turns(sid)]


LEARNER = "curriculum-learner"


def concept_mastery(env: Env) -> float:
    profile = env.container.memory.get_or_create(LEARNER)
    return profile.concepts[CONCEPT].mastery if CONCEPT in profile.concepts else 0.0


async def test_a_full_session_reaches_mastery_through_learner_memory_and_the_curriculum(teaching_env: Env) -> None:
    env = teaching_env
    lesson_artifacts = {a.artifact_id for a in env.container.task_service.artifacts(env.lesson_task)}
    before = concept_mastery(env)
    started = await start(env)
    assert started.created and started.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    assert started.objective.concept_id == CONCEPT and started.difficulty == 2 and started.objective_id
    assert started.first_turn is not None and started.first_turn.turn_type.value == "EXPLANATION"
    assert started.waiting_question is not None
    sid = started.session_id
    results = [await step(env, sid, i) for i in range(len(SCRIPT))]

    final = results[-1]
    assert final.status == TeachingSessionStatus.COMPLETED
    assert final.completion_reason == CompletionReason.OBJECTIVE_DEMONSTRATED
    view = await env.service.view(sid)
    assert view.summary is not None and view.outcome is not None
    assert view.summary.difficulty_trajectory == [2, 1, 2] and view.summary.hints_used == 1
    assert view.summary.misconceptions and view.summary.recommended_next_action == "advance"

    # mastery: only through learner memory's updater, from the session's 4 graded answers
    outcome = view.outcome
    assert not outcome.practice and len(outcome.learning_evidence_ids) == 4
    after = concept_mastery(env)
    assert after > before
    change = next(c for c in outcome.mastery_changes if c["concept_id"] == CONCEPT)
    assert change["before"] == pytest.approx(before, abs=1e-3) and change["after"] == pytest.approx(after, abs=1e-3)
    stored = [e for e in env.container.memory.evidence(LEARNER) if e.source_type == "interaction"]
    assert sorted(e.evidence_id for e in stored) == sorted(outcome.learning_evidence_ids)
    by_turn = {e.metadata["turn_id"]: e for e in stored}
    hinted = next(e for e in env.service.evidence(sid)
                  if e.evidence_type == EvidenceType.ANSWER and e.answer_after_hint)
    assert by_turn[hinted.turn_id].correctness == "partial" and 0 < by_turn[hinted.turn_id].score < 1

    # objective progress and the next action come from the curriculum engine
    assert outcome.objective_progress["concept_id"] == CONCEPT
    assert outcome.objective_progress["current_mastery"] == pytest.approx(after, abs=1e-3)
    expected = await env.container.curriculum_service.next_action(LEARNER)
    assert outcome.learning_action["action"] == expected.action.value
    assert outcome.learning_action["concept_id"] == expected.concept_id
    assert view.next_action.kind == "learning_action"

    # artifacts: the lesson's lineage continues into the session, each stored once
    arts = env.container.task_service.artifacts(env.lesson_task)
    types = Counter(a.type for a in arts)
    assert types[ArtifactType.TEACHING_SESSION] == 1 and types[ArtifactType.TEACHING_SESSION_SUMMARY] == 1
    assert types[ArtifactType.TEACHING_TURN] == len(env.service.turns(sid))
    assert types[ArtifactType.INTERACTION_EVIDENCE] == len(env.service.evidence(sid))
    assert all(n == 1 for n in Counter(a.name for a in arts).values())
    evidence_art = next(a for a in arts if a.type == ArtifactType.INTERACTION_EVIDENCE)
    lineage = [a.type for a in env.container.artifacts.lineage(evidence_art.artifact_id)]
    for t in (ArtifactType.TEACHING_TURN, ArtifactType.TEACHING_SESSION, ArtifactType.LESSON,
              ArtifactType.LEARNING_OBJECTIVE, ArtifactType.CURRICULUM_VERSION, ArtifactType.LEARNING_GOAL):
        assert t in lineage
    chain = [env.container.artifacts.get(outcome.artifact_ids[k]) for k in
             ("summary", "learning_evidence", "learner_model", "learning_action")]
    assert [a.type for a in chain] == [ArtifactType.TEACHING_SESSION_SUMMARY, ArtifactType.LEARNING_EVIDENCE,
                                       ArtifactType.LEARNER_MODEL, ArtifactType.LEARNING_ACTION]
    assert all(chain[i].artifact_id in chain[i + 1].parent_ids for i in range(3))

    # events: each once, with stable ids
    events = teaching_events(env)
    counts = Counter(e.type for e in events)
    assert len({e.event_id for e in events}) == len(events)
    assert counts["teaching_session.started"] == counts["teaching_session.completed"] == 1
    assert counts["learner_answer.received"] == len(SCRIPT) and counts["difficulty.changed"] == 2
    assert counts["hint.given"] == 1 and counts["misconception.detected"] == 1
    assert counts["teaching_turn.created"] == len(env.service.turns(sid))

    # the lesson task itself is untouched
    lesson = env.container.task_service.get(env.lesson_task)
    assert lesson.status == TaskStatus.COMPLETED
    assert lesson_artifacts <= {a.artifact_id for a in arts}


async def test_resume_after_a_process_restart_continues_deterministically(teaching_env: Env, tmp_path,
                                                                         teaching_lesson) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    for i in range(3):
        await step(env, sid, i)
    turns, evidence = transcript(env, sid), [e.evidence_id for e in env.service.evidence(sid)]
    events = [e.event_id for e in teaching_events(env)]
    state = (await env.service.view(sid)).state

    env = env.reopen()  # the process is gone; a new one opens the same data directory
    view = await env.service.view(sid)
    assert view.status == TeachingSessionStatus.WAITING_FOR_LEARNER and view.state == state
    assert transcript(env, sid) == turns
    assert [e.evidence_id for e in env.service.evidence(sid)] == evidence
    assert [e.event_id for e in teaching_events(env)] == events  # nothing emitted twice by reopening
    for i in range(3, len(SCRIPT)):
        await step(env, sid, i)
    resumed = transcript(env, sid)

    # the same learner without the restart produces exactly the same session
    template, task_id = teaching_lesson
    other = open_env(tmp_path / "uninterrupted", task_id, copy_from=template)
    try:
        assert (await start(other)).session_id == sid
        for i in range(len(SCRIPT)):
            await step(other, sid, i)
        assert transcript(other, sid) == resumed
        assert len({e.event_id for e in teaching_events(env)}) == len(teaching_events(env))
        assert Counter(e.type for e in teaching_events(env)) == Counter(e.type for e in teaching_events(other))
    finally:
        other.close()


async def test_pause_refuses_answers_and_survives_a_restart(teaching_env: Env) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    await step(env, sid, 0)
    paused = await env.service.pause(sid)
    assert paused.status == TeachingSessionStatus.PAUSED and paused.next_action.kind == "resume"
    assert (await env.service.pause(sid)).status == TeachingSessionStatus.PAUSED  # idempotent
    with pytest.raises(InvalidSessionTransition):
        await step(env, sid, 1)
    env = env.reopen()
    resumed = await env.service.resume(sid)
    assert resumed.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    assert (await step(env, sid, 1)).correct
    counts = Counter(e.type for e in teaching_events(env))
    assert counts["teaching_session.paused"] == counts["teaching_session.resumed"] == 1


async def test_cancel_is_idempotent_and_keeps_the_history(teaching_env: Env) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    await step(env, sid, 0)
    turns = transcript(env, sid)
    cancelled = await env.service.cancel(sid)
    assert cancelled.status == TeachingSessionStatus.CANCELLED and cancelled.state.waiting_question is None
    assert (await env.service.cancel(sid)).status == TeachingSessionStatus.CANCELLED
    assert transcript(env, sid) == turns and env.service.evidence(sid)
    with pytest.raises(InvalidSessionTransition):
        await env.service.resume(sid)
    with pytest.raises(InvalidSessionTransition):
        await env.service.submit(sid, LearnerInput(answer="sonó", client_turn_id="late"))
    assert Counter(e.type for e in teaching_events(env))["teaching_session.cancelled"] == 1
    assert not [e for e in env.container.memory.evidence(LEARNER) if e.source_type == "interaction"]


async def test_a_crash_between_the_two_phases_loses_nothing(teaching_env: Env, monkeypatch) -> None:
    env = teaching_env
    sid = (await start(env)).session_id

    async def crash(*_args, **_kw):
        raise SimulatedCrash

    monkeypatch.setattr(env.service, "_publish", crash)  # the answer is committed, nothing after it runs
    with pytest.raises(SimulatedCrash):
        await step(env, sid, 0)
    env = env.reopen()
    view = await env.service.view(sid)  # reading the session publishes what the crash left behind
    assert view.status == TeachingSessionStatus.ACTIVE and view.next_action.kind == "teacher_reply_pending"
    assert view.turns[-1].turn_type.value == "LEARNER_ANSWER"
    resumed = await env.service.resume(sid)  # produces the teacher turns still owed
    assert resumed.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    assert resumed.turns[-1].turn_type.value == "HINT"
    replay = await step(env, sid, 0)  # the client retries its request: the stored result, not a second answer
    assert replay.replayed
    events = teaching_events(env)
    assert len({e.event_id for e in events}) == len(events)
    assert Counter(e.type for e in events)["learner_answer.received"] == 1


async def test_a_teacher_failure_keeps_the_answer_and_a_retry_completes_the_reply(teaching_env: Env) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    env.llm.inject(AGENT, *[RuntimeError("provider down")] * 20)
    with pytest.raises(TeacherUnavailable):
        await step(env, sid, 0)
    view = await env.service.view(sid)
    assert view.status == TeachingSessionStatus.ACTIVE
    assert [e.evidence_type for e in env.service.evidence(sid)] == [EvidenceType.ANSWER]
    env.llm._faults[AGENT].clear()  # the provider recovers
    replay = await step(env, sid, 0)
    assert replay.replayed and replay.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    assert [t.turn_type.value for t in replay.teacher_turns] == ["HINT"]


@pytest.mark.parametrize("bad", [
    '{"action": "COMPLETE", "concept_id": "es.past_contrast", "difficulty": 2, "response": "Done, you mastered it.", '
    '"expected_response_type": "none"}',
    '{"action": "EXPLAIN", "concept_id": "es.subjunctive", "difficulty": 2, "response": "Wrong concept.", '
    '"expected_response_type": "none"}',
    '{"action": "EXPLAIN", "concept_id": "es.past_contrast", "difficulty": 2, "response": "See the source.", '
    '"expected_response_type": "none", "citations": ["https://invented.example/paper"]}',
    '{"action": "EXPLAIN", "concept_id": "es.past_contrast", "difficulty": 2, "response": "x", '
    '"expected_response_type": "none", "mastery": 1.0, "session_status": "COMPLETED"}',
    "this is not json {",
])
async def test_adversarial_teacher_output_is_rejected_and_retried(teaching_env: Env, bad: str) -> None:
    env = teaching_env
    env.llm.inject(AGENT, bad)  # the opening EXPLAIN turn gets the bad output first
    before = concept_mastery(env)
    started = await start(env)
    assert started.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    assert started.first_turn.content != "Done, you mastered it."
    assert started.first_turn.metadata["citations"] == [] or all(
        not c.startswith("https://invented") for c in started.first_turn.metadata["citations"])
    assert env.llm.calls[AGENT] >= 3  # the rejected output, its retry, and the question turn
    assert concept_mastery(env) == before


async def test_persistently_invalid_output_fails_the_turn_not_the_state(teaching_env: Env) -> None:
    env = teaching_env
    bad = ('{"action": "COMPLETE", "concept_id": "es.past_contrast", "difficulty": 2, "response": "Done.", '
           '"expected_response_type": "none"}')
    env.llm.inject(AGENT, *[bad] * 20)
    with pytest.raises(TeacherUnavailable):
        await start(env)
    env.llm._faults[AGENT].clear()
    started = await start(env)  # the same idempotency key finishes the opening
    assert not started.created and started.status == TeachingSessionStatus.WAITING_FOR_LEARNER


async def test_learner_questions_are_grounded_or_refused_with_a_limitation(teaching_env: Env) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    grounded = (await step(env, sid, 3)).teacher_turns[0]
    assert grounded.metadata["grounded"] is True and grounded.metadata["citations"]
    lesson = env.container.artifacts.find(env.lesson_task, "lesson")
    sections = {f"lesson:{s['section_id']}" for s in json.loads(env.container.artifacts.read(lesson.artifact_id))[
        "sections"]}
    research = env.container.artifacts.find(env.lesson_task, "research_bundle")
    known = set(sections) | set(json.dumps(json.loads(env.container.artifacts.read(research.artifact_id))).split('"'))
    assert set(grounded.metadata["citations"]) <= known
    refused = (await step(env, sid, 4)).teacher_turns[0]
    assert refused.metadata["grounded"] is False and refused.metadata["citations"] == []
    assert refused.metadata["limitation"]
    view = await env.service.view(sid)
    assert view.status == TeachingSessionStatus.WAITING_FOR_LEARNER and view.state.waiting_question is not None


async def test_duplicate_answers_are_idempotent(teaching_env: Env) -> None:
    env = teaching_env
    started = await start(env)
    again = await start(env)
    assert again.session_id == started.session_id and not again.created
    with pytest.raises(SessionConflict):  # the same key for a different session
        await start(env, action="EVALUATE")
    sid = started.session_id
    first = await step(env, sid, 0)
    evidence, events = len(env.service.evidence(sid)), len(teaching_events(env))
    second = await step(env, sid, 0)
    assert second.replayed and second.learner_turn.turn_id == first.learner_turn.turn_id
    assert [t.turn_id for t in second.teacher_turns] == [t.turn_id for t in first.teacher_turns]
    assert len(env.service.evidence(sid)) == evidence and len(teaching_events(env)) == events
    with pytest.raises(SessionConflict):  # the same client_turn_id with a different answer
        await env.service.submit(sid, LearnerInput(answer="otra cosa", client_turn_id="c1"))


async def test_concurrent_answers_conflict_and_are_never_silently_lost(teaching_env: Env, monkeypatch) -> None:
    env = teaching_env
    sid = (await start(env)).session_id
    view = await env.service.view(sid)
    text = answer_for(view.state.waiting_question, KEY, "correct")
    load = env.service._load

    async def slow_load(*args, **kw):  # both requests read the same version before either writes
        session = await load(*args, **kw)
        await asyncio.sleep(0)
        return session

    monkeypatch.setattr(env.service, "_load", slow_load)
    results = await asyncio.gather(
        env.service.submit(sid, LearnerInput(answer=text, client_turn_id="a")),
        env.service.submit(sid, LearnerInput(answer=text, client_turn_id="b")), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, BaseException)]
    failed = [r for r in results if isinstance(r, BaseException)]
    assert len(ok) == 1 and len(failed) == 1 and isinstance(failed[0], (SessionConflict, InvalidSessionTransition))
    answers = [t for t in env.service.turns(sid) if t.turn_type.value == "LEARNER_ANSWER"]
    assert len(answers) == 1  # the loser was told (409), not applied on top
    # the same request sent twice at once is applied once and both callers get its result
    view = await env.service.view(sid)
    text = answer_for(view.state.waiting_question, KEY, "correct")
    same = await asyncio.gather(*[env.service.submit(sid, LearnerInput(answer=text, client_turn_id="dup"))
                                  for _ in range(2)])
    assert same[0].learner_turn.turn_id == same[1].learner_turn.turn_id
    assert sorted(r.replayed for r in same) == [False, True]


async def test_a_stale_writer_gets_a_conflict(teaching_env: Env) -> None:
    from app.teaching import engine

    env = teaching_env
    sid = (await start(env)).session_id
    stale = env.service.repository.get(sid)
    await step(env, sid, 0)
    change = engine.record_answer(stale, "sonó", stale.updated_at).change
    with pytest.raises(SessionConflict):
        env.service.repository.apply(change, expected_version=stale.version)


async def test_practice_sessions_record_practice_evidence_only(teaching_env: Env) -> None:
    env = teaching_env
    before = concept_mastery(env)
    started = await start(env, key="practice", practice=True)
    assert started.waiting_question is not None and started.first_turn.turn_type.value == "QUESTION"
    sid = started.session_id
    view = await env.service.view(sid)
    await env.service.submit(sid, LearnerInput(answer=answer_for(view.state.waiting_question, KEY, "correct"),
                                                client_turn_id="p1"))
    done = await env.service.submit(sid, LearnerInput(answer="stop", kind="stop", client_turn_id="p2"))
    assert done.status == TeachingSessionStatus.COMPLETED
    assert done.completion_reason == CompletionReason.LEARNER_STOPPED
    assert done.summary.practice and done.outcome.practice and not done.outcome.learning_evidence_ids
    assert all(e.practice for e in env.service.evidence(sid))
    assert concept_mastery(env) == before
    assert not [e for e in env.container.memory.evidence(LEARNER) if e.source_type == "interaction"]


async def test_invalid_requests(teaching_env: Env) -> None:
    env = teaching_env
    with pytest.raises(InvalidTeachingRequest):
        await start(env, key="bad", objective_id="obj_not_in_this_lesson")
    from app.storage.repositories import NotFound

    with pytest.raises(NotFound):
        await env.service.start("art_missing", StartTeachingSession())
    from app.teaching.errors import TeachingSessionNotFound

    with pytest.raises(TeachingSessionNotFound):
        await env.service.view("tsess_missing")


async def test_the_model_sees_minimal_context_and_nothing_secret_is_stored(teaching_lesson, tmp_path) -> None:
    secret = "sk-teaching-secret-0123456789"
    template, task_id = teaching_lesson
    env = open_env(tmp_path / "data", task_id, copy_from=template,
                   providers=ProviderSettings(openai_api_key=secret, llm_routes={}))
    try:
        sid = (await start(env)).session_id
        for i in range(len(SCRIPT)):
            await step(env, sid, i)
        session = env.service.repository.get(sid)
        goal_id = session.metadata["goal_id"]
        requests = [r for r in env.llm.requests if r.agent_id == AGENT]
        assert requests
        for r in requests:
            text = r.system + "".join(m.content for m in r.messages) + str(r.input_payload)
            for private in (LEARNER, sid, task_id, goal_id, secret, "display_name", "learner_id", "session_id"):
                assert private not in text, private
        for e in teaching_events(env):
            dumped = e.model_dump_json()
            assert secret not in dumped and "expected_answer" not in dumped and "content" not in e.data
        stored = b"".join(p.read_bytes() for p in env.data_dir.rglob("*") if p.is_file())
        assert secret.encode() not in stored
    finally:
        env.close()
