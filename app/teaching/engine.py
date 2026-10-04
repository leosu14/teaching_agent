"""Session transitions: pure functions from (session, input) to the change to store.

A change (SessionChange) is everything one transition produces: the new session (its version incremented), new
turns, new evidence and the outbox items (events, artifacts, the completion step) published after the change is
stored. Ids are derived from the session id and the turn sequence, so the same transition always produces the same
records, and a repeated or concurrent request is caught by the store (optimistic lock, unique sequence) instead of
being applied twice.

Learner input is handled in two steps. `record_*` grades the input and decides, by the policy, which teacher turns
are owed (status ACTIVE). `apply_teacher` + `settle` then store the teacher's phrasing of those turns (status
WAITING_FOR_LEARNER, or COMPLETED when the policy decided the session ends). A failed model call therefore never loses
the learner's input: the session stays ACTIVE and the owed turns are generated again on resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from app.schemas.artifact import ArtifactType
from app.schemas.events import EventType
from app.schemas.learner import stable_id
from app.schemas.teaching import (
    QUESTION_ACTIONS,
    AnswerOutcome,
    CompletionReason,
    DifficultyChange,
    EvidenceType,
    InteractionEvidence,
    MisconceptionRecord,
    OutboxItem,
    PendingQuestion,
    PlannedTurn,
    SessionChange,
    SessionMode,
    SessionState,
    Speaker,
    TeacherTurnOutput,
    TeachingAction,
    TeachingConfig,
    TeachingSession,
    TeachingSessionStatus,
    TeachingTurn,
    TurnType,
)
from app.teaching.errors import InvalidSessionTransition
from app.teaching.grading import grade, normalize
from app.teaching.policy import RETEACH_STRATEGY, DifficultyDecision, TeachingPolicy
from app.teaching.summary import build_summary

Status = TeachingSessionStatus


@dataclass
class AnswerTransition:
    change: SessionChange
    correct: bool
    decision: DifficultyDecision
    difficulty_change: DifficultyChange | None


# --- ids and outbox items ------------------------------------------------------------------------------------------


def session_id_for(learner_id: str, lesson_id: str, key: str) -> str:
    return stable_id("tsess", learner_id, lesson_id, key)


def turn_id_for(session_id: str, sequence: int) -> str:
    return stable_id("tturn", session_id, str(sequence))


def evidence_id_for(session_id: str, turn_id: str, kind: EvidenceType, index: int = 0) -> str:
    return stable_id("tevid", session_id, turn_id, kind.value, str(index))


def session_artifact(session_id: str) -> str:
    return f"teaching_session.{session_id}"


def turn_artifact(turn_id: str) -> str:
    return f"teaching_turn.{turn_id}"


def evidence_artifact(evidence_id: str) -> str:
    return f"interaction_evidence.{evidence_id}"


def summary_artifact(session_id: str) -> str:
    return f"teaching_session_summary.{session_id}"


def _item(session: TeachingSession, kind: str, key: str, payload: dict) -> OutboxItem:
    return OutboxItem(item_id=stable_id("tout", session.session_id, str(session.version), kind, key),
                      session_id=session.session_id, kind=kind, payload=payload)


def event_item(session: TeachingSession, type: str, key: str = "", **data: object) -> OutboxItem:
    """An event with stable ids and minimal metadata: never answer text, prompts, provider details or secrets."""
    return _item(session, "event", f"{type}:{key}",
                 {"type": type, "data": {"session_id": session.session_id, **data}})


def artifact_item(session: TeachingSession, *, name: str, type: ArtifactType, content: dict,
                  parent_names: list[str] = (), parent_ids: list[str] = (), metadata: dict | None = None) -> OutboxItem:
    return _item(session, "artifact", name, {
        "name": name, "type": type.value, "content": json.dumps(content, ensure_ascii=False, indent=2, default=str),
        "parent_names": list(parent_names), "parent_ids": list(parent_ids), "metadata": metadata or {}})


def _turn_items(session: TeachingSession, turn: TeachingTurn) -> list[OutboxItem]:
    return [
        artifact_item(session, name=turn_artifact(turn.turn_id), type=ArtifactType.TEACHING_TURN,
                      content=turn.model_dump(mode="json"), parent_names=[session_artifact(session.session_id)],
                      metadata={"session_id": session.session_id, "sequence": turn.sequence,
                                "speaker": turn.speaker.value, "turn_type": turn.turn_type.value}),
        event_item(session, EventType.TEACHING_TURN_CREATED, turn.turn_id, turn_id=turn.turn_id,
                   sequence=turn.sequence, speaker=turn.speaker.value, turn_type=turn.turn_type.value,
                   action=turn.metadata.get("action")),
    ]


def _evidence_item(session: TeachingSession, ev: InteractionEvidence) -> OutboxItem:
    return artifact_item(session, name=evidence_artifact(ev.evidence_id), type=ArtifactType.INTERACTION_EVIDENCE,
                         content=ev.model_dump(mode="json"), parent_names=[turn_artifact(ev.turn_id)],
                         metadata={"session_id": session.session_id, "evidence_type": ev.evidence_type.value,
                                   "concept_id": ev.concept_id, "correct": ev.correct, "practice": ev.practice})


def _advance(session: TeachingSession, at: datetime) -> TeachingSession:
    s = session.model_copy(deep=True)
    s.version += 1
    s.updated_at = at
    return s


# --- start ---------------------------------------------------------------------------------------------------------


def start(*, session_id: str, learner_id: str, task_id: str, lesson_id: str, objective_id: str | None,
          objective_description: str, concept_id: str, concept_name: str, action: str, mode: SessionMode,
          config: TeachingConfig, at: datetime, metadata: dict) -> SessionChange:
    policy = TeachingPolicy(config)
    state = SessionState(objective_id=objective_id, objective_description=objective_description,
                         concept_id=concept_id, concept_name=concept_name, difficulty=config.start_difficulty,
                         strategy=policy.strategy(config, config.start_difficulty), turn_budget=config.max_turns,
                         planned=policy.opening(action, mode))
    session = TeachingSession(session_id=session_id, learner_id=learner_id, task_id=task_id, lesson_id=lesson_id,
                              objective_id=objective_id, action=action, mode=mode, status=Status.ACTIVE,
                              started_at=at, updated_at=at, config=config, state=state, metadata=metadata)
    record = {"session_id": session_id, "lesson_id": lesson_id, "objective_id": objective_id,
              "concept_id": concept_id, "objective": objective_description, "action": action, "mode": mode,
              "config": config.model_dump(), "started_at": at.isoformat()}
    return SessionChange(session=session, outbox=[
        artifact_item(session, name=session_artifact(session_id), type=ArtifactType.TEACHING_SESSION, content=record,
                      parent_ids=[lesson_id], metadata={"session_id": session_id, "action": action, "mode": mode,
                                                        "concept_id": concept_id, "objective_id": objective_id}),
        event_item(session, EventType.TEACHING_SESSION_STARTED, lesson_id=lesson_id, objective_id=objective_id,
                   concept_id=concept_id, action=action, mode=mode, difficulty=state.difficulty),
    ])


# --- learner input -------------------------------------------------------------------------------------------------


def _require_waiting(session: TeachingSession) -> None:
    if session.status == Status.PAUSED:
        raise InvalidSessionTransition(f"session {session.session_id} is paused: resume it before answering")
    if session.status != Status.WAITING_FOR_LEARNER:
        raise InvalidSessionTransition(f"session {session.session_id} is {session.status.value}: it does not accept "
                                       "learner input")


def _learner_turn(session: TeachingSession, turn_type: TurnType, content: str, at: datetime,
                  metadata: dict) -> TeachingTurn:
    sequence = session.turn_count + 1
    session.turn_count = sequence
    return TeachingTurn(turn_id=turn_id_for(session.session_id, sequence), session_id=session.session_id,
                        sequence=sequence, speaker=Speaker.LEARNER, turn_type=turn_type, content=content,
                        created_at=at, metadata=metadata)


def record_answer(session: TeachingSession, answer: str, at: datetime, *,
                  client_turn_id: str | None = None) -> AnswerTransition:
    _require_waiting(session)
    question = session.state.pending_question
    if question is None:
        raise InvalidSessionTransition(f"session {session.session_id} has no open question to answer")
    s = _advance(session, at)
    st, q, policy = s.state, s.state.pending_question, TeachingPolicy(s.config)
    assert q is not None
    correct = grade(q, answer)
    hinted = q.hint_level > 0
    turn = _learner_turn(s, TurnType.LEARNER_ANSWER, answer, at, {
        "question_turn_id": q.turn_id, "correct": correct, "hint_level": q.hint_level, "difficulty": q.difficulty,
        **({"client_turn_id": client_turn_id} if client_turn_id else {})})
    st.questions_answered += 1
    if correct:
        st.correct_answers += 1
        st.assisted_correct += hinted
    else:
        st.incorrect_answers += 1
    decision = policy.difficulty.after_answer(st.difficulty, st.consecutive_successes, st.consecutive_failures,
                                              correct=correct, hinted=hinted)
    change = None
    if decision.change != "keep":
        change = DifficultyChange(turn_id=turn.turn_id, before=st.difficulty, after=decision.difficulty,
                                  reason=decision.reason)
        st.difficulty_history.append(change)
    st.difficulty, st.consecutive_successes, st.consecutive_failures = (decision.difficulty, decision.successes,
                                                                        decision.failures)
    st.strategy = policy.strategy(s.config, st.difficulty)
    q.attempts += 1
    before = sum(1 for m in st.misconceptions if m.concept_id == q.concept_id)
    planned, keep_open = policy.after_answer(st, q, correct=correct, answer_turn_id=turn.turn_id, mode=s.mode,
                                             misconceptions_before=before)
    reason = policy.completion(st, correct=correct, hinted=hinted, question_difficulty=q.difficulty,
                               question_open=keep_open, turn_count=s.turn_count)
    if reason is not None:
        planned, keep_open = [PlannedTurn(action=TeachingAction.SUMMARIZE)], False
        st.completion_reason = reason
    if any(p.action == TeachingAction.RETEACH for p in planned):
        st.strategy = RETEACH_STRATEGY
    st.pending_question = q if keep_open else None
    st.last_question = q.model_copy()
    st.planned = planned
    st.last_answer = AnswerOutcome(turn_id=turn.turn_id, answer=answer, correct=correct, hint_level=q.hint_level)
    evidence = InteractionEvidence(
        evidence_id=evidence_id_for(s.session_id, turn.turn_id, EvidenceType.ANSWER), session_id=s.session_id,
        turn_id=turn.turn_id, learner_id=s.learner_id, concept_id=q.concept_id, evidence_type=EvidenceType.ANSWER,
        correct=correct, hint_level=q.hint_level, hints_used=q.hint_level, answer_after_hint=hinted,
        difficulty=q.difficulty, practice=s.mode == "practice", created_at=at,
        metadata={"question_turn_id": q.turn_id, "attempt": q.attempts, "question_action": q.action.value})
    st.evidence_ids.append(evidence.evidence_id)
    s.status = Status.ACTIVE
    outbox = [*_turn_items(s, turn), _evidence_item(s, evidence),
              event_item(s, EventType.LEARNER_ANSWER_RECEIVED, turn.turn_id, turn_id=turn.turn_id, kind="answer",
                         sequence=turn.sequence, concept_id=q.concept_id, correct=correct, hint_level=q.hint_level,
                         difficulty=q.difficulty, evidence_id=evidence.evidence_id)]
    if change is not None:
        outbox.append(event_item(s, EventType.DIFFICULTY_CHANGED, turn.turn_id, turn_id=turn.turn_id,
                                 before=change.before, after=change.after, reason=change.reason))
    return AnswerTransition(change=SessionChange(session=s, turns=[turn], evidence=[evidence], outbox=outbox),
                            correct=correct, decision=decision, difficulty_change=change)


def record_question(session: TeachingSession, question: str, at: datetime, *,
                    client_turn_id: str | None = None) -> SessionChange:
    """A learner question: answered from the lesson and research material; the open question stays open."""
    _require_waiting(session)
    s = _advance(session, at)
    st = s.state
    turn = _learner_turn(s, TurnType.LEARNER_QUESTION, question, at,
                         {"client_turn_id": client_turn_id} if client_turn_id else {})
    st.learner_questions += 1
    evidence = InteractionEvidence(
        evidence_id=evidence_id_for(s.session_id, turn.turn_id, EvidenceType.LEARNER_QUESTION),
        session_id=s.session_id, turn_id=turn.turn_id, learner_id=s.learner_id, concept_id=st.concept_id,
        evidence_type=EvidenceType.LEARNER_QUESTION, difficulty=st.difficulty, practice=s.mode == "practice",
        created_at=at)
    st.evidence_ids.append(evidence.evidence_id)
    st.planned = [PlannedTurn(action=TeachingAction.EXPLAIN, purpose="answer_question", responds_to=turn.turn_id)]
    if s.turn_count + 3 > s.config.max_turns:
        st.completion_reason = CompletionReason.TURN_BUDGET_REACHED
        st.pending_question = None
        st.planned.append(PlannedTurn(action=TeachingAction.SUMMARIZE))
    s.status = Status.ACTIVE
    return SessionChange(session=s, turns=[turn], evidence=[evidence], outbox=[
        *_turn_items(s, turn), _evidence_item(s, evidence),
        event_item(s, EventType.LEARNER_ANSWER_RECEIVED, turn.turn_id, turn_id=turn.turn_id, kind="question",
                   sequence=turn.sequence, concept_id=st.concept_id, evidence_id=evidence.evidence_id)])


def record_stop(session: TeachingSession, text: str, at: datetime, *,
                client_turn_id: str | None = None) -> SessionChange:
    """The learner asks to stop: the session ends (LEARNER_STOPPED) after the teacher's summary."""
    _require_waiting(session)
    s = _advance(session, at)
    turn = _learner_turn(s, TurnType.LEARNER_ANSWER, text, at,
                         {"intent": "stop", **({"client_turn_id": client_turn_id} if client_turn_id else {})})
    s.state.pending_question = None
    s.state.completion_reason = CompletionReason.LEARNER_STOPPED
    s.state.planned = [PlannedTurn(action=TeachingAction.SUMMARIZE)]
    s.status = Status.ACTIVE
    return SessionChange(session=s, turns=[turn], outbox=[
        *_turn_items(s, turn),
        event_item(s, EventType.LEARNER_ANSWER_RECEIVED, turn.turn_id, turn_id=turn.turn_id, kind="stop",
                   sequence=turn.sequence)])


# --- teacher turns -------------------------------------------------------------------------------------------------


def turn_type_for(item: PlannedTurn, last_answer: AnswerOutcome | None) -> TurnType:
    if item.action in (TeachingAction.EXPLAIN, TeachingAction.RETEACH):
        return TurnType.EXPLANATION
    if item.action in (TeachingAction.ASK, TeachingAction.PRACTICE):
        return TurnType.QUESTION
    if item.action == TeachingAction.CHECK:
        return TurnType.ASSESSMENT
    if item.action == TeachingAction.HINT:
        return TurnType.HINT
    if item.action == TeachingAction.SUMMARIZE:
        return TurnType.SUMMARY
    if item.correction:
        return TurnType.CORRECTION
    if last_answer is not None and last_answer.correct and last_answer.hint_level > 0:
        return TurnType.ENCOURAGEMENT
    return TurnType.FEEDBACK


def check_output(session: TeachingSession, item: PlannedTurn, out: TeacherTurnOutput) -> None:
    """The deterministic invariants of a teacher turn, enforced again at the state boundary (the agent validates the
    same and more): the turn phrases exactly the planned action, for the session's concept, at its difficulty."""
    expected = TeachingAction.EXPLAIN if item.purpose == "answer_question" else item.action
    if out.action != expected:
        raise ValueError(f"the teacher turn must be {expected.value}, not {out.action.value}")
    if out.concept_id != session.state.concept_id:
        raise ValueError(f"the teacher turn must be about {session.state.concept_id}")
    if (item.action in QUESTION_ACTIONS) != (out.question is not None):
        raise ValueError("a question turn carries exactly one question; no other turn carries one")


def apply_teacher(session: TeachingSession, item: PlannedTurn, out: TeacherTurnOutput, at: datetime) -> SessionChange:
    """Store one planned teacher turn on `session` (mutated in place: a working copy inside one transition)."""
    check_output(session, item, out)
    st = session.state
    if not st.planned or st.planned[0] != item:
        raise ValueError("teacher turns are applied in the planned order")
    sequence = session.turn_count + 1
    turn_id = turn_id_for(session.session_id, sequence)
    metadata: dict = {"action": item.action.value, "concept_id": st.concept_id, "difficulty": st.difficulty,
                      "citations": list(out.citations)}
    if item.responds_to:
        metadata["responds_to"] = item.responds_to
    evidence: list[InteractionEvidence] = []
    outbox: list[OutboxItem] = []
    if item.purpose == "answer_question":
        metadata.update(grounded=bool(out.grounded), limitation=out.limitation)
    if item.action in QUESTION_ACTIONS:
        assert out.question is not None
        q = out.question
        st.pending_question = PendingQuestion(
            turn_id=turn_id, action=item.action, concept_id=st.concept_id, difficulty=st.difficulty, kind=q.kind,
            prompt=q.prompt, choices=list(q.choices), expected_answer=q.expected_answer,
            accepted_answers=list(q.accepted_answers))
        st.questions_asked += 1
        st.difficulty_trajectory.append(st.difficulty)
        metadata.update(kind=q.kind, prompt=q.prompt, choices=list(q.choices), question_number=st.questions_asked)
    if item.action == TeachingAction.HINT:
        pending = st.pending_question
        assert pending is not None
        pending.hint_level = item.hint_level
        st.hints_used += 1
        metadata["hint_level"] = item.hint_level
        hint = InteractionEvidence(
            evidence_id=evidence_id_for(session.session_id, turn_id, EvidenceType.HINT), session_id=session.session_id,
            turn_id=turn_id, learner_id=session.learner_id, concept_id=st.concept_id, evidence_type=EvidenceType.HINT,
            hint_level=item.hint_level, hints_used=st.hints_used, difficulty=pending.difficulty,
            practice=session.mode == "practice", created_at=at, metadata={"question_turn_id": pending.turn_id})
        evidence.append(hint)
        outbox.append(event_item(session, EventType.HINT_GIVEN, turn_id, turn_id=turn_id, hint_level=item.hint_level,
                                 question_turn_id=pending.turn_id, evidence_id=hint.evidence_id))
    answer = st.last_answer
    if (item.responds_to and answer is not None and answer.turn_id == item.responds_to and not answer.correct):
        seen: set[str] = set()
        for i, candidate in enumerate(out.misconceptions):
            label = normalize(candidate.misconception)
            if candidate.confidence < session.config.misconception_min_confidence or label in seen:
                continue
            seen.add(label)
            ev = InteractionEvidence(
                evidence_id=evidence_id_for(session.session_id, answer.turn_id, EvidenceType.MISCONCEPTION, i),
                session_id=session.session_id, turn_id=answer.turn_id, learner_id=session.learner_id,
                concept_id=candidate.concept_id, evidence_type=EvidenceType.MISCONCEPTION, correct=False,
                difficulty=st.difficulty, practice=session.mode == "practice", misconception=candidate.misconception,
                confidence=candidate.confidence, created_at=at, metadata={"proposed_with": turn_id})
            evidence.append(ev)
            st.misconceptions.append(MisconceptionRecord(evidence_id=ev.evidence_id, concept_id=ev.concept_id,
                                                         misconception=candidate.misconception,
                                                         confidence=candidate.confidence))
            outbox.append(event_item(session, EventType.MISCONCEPTION_DETECTED, ev.evidence_id,
                                     turn_id=answer.turn_id, concept_id=ev.concept_id, evidence_id=ev.evidence_id,
                                     confidence=candidate.confidence))
    st.evidence_ids += [e.evidence_id for e in evidence]
    session.turn_count = sequence
    turn = TeachingTurn(turn_id=turn_id, session_id=session.session_id, sequence=sequence, speaker=Speaker.TEACHER,
                        turn_type=turn_type_for(item, answer), content=out.response, created_at=at, metadata=metadata)
    st.planned = st.planned[1:]
    return SessionChange(session=session, turns=[turn], evidence=evidence,
                  outbox=[*_turn_items(session, turn), *(_evidence_item(session, e) for e in evidence), *outbox])


def settle(session: TeachingSession, at: datetime, narrative: str = "") -> list[OutboxItem]:
    """After every owed teacher turn is stored: wait for the learner, or complete the session (the policy decided
    that when it set the completion reason; a model never completes a session)."""
    st = session.state
    if st.planned:
        raise ValueError("teacher turns are still owed")
    if st.completion_reason is None:
        session.status = Status.WAITING_FOR_LEARNER
        return []
    session.status = Status.COMPLETED
    session.completed_at = at
    session.summary = build_summary(session, narrative)
    summary = session.summary
    return [
        artifact_item(session, name=summary_artifact(session.session_id), type=ArtifactType.TEACHING_SESSION_SUMMARY,
                      content=summary.model_dump(mode="json"),
                      parent_names=[session_artifact(session.session_id),
                                    *(evidence_artifact(e) for e in st.evidence_ids)],
                      metadata={"session_id": session.session_id, "completion_reason": summary.completion_reason.value,
                                "recommended_next_action": summary.recommended_next_action,
                                "practice": summary.practice}),
        _item(session, "finalize", "finalize", {}),
    ]


# --- lifecycle -----------------------------------------------------------------------------------------------------


def pause(session: TeachingSession, at: datetime) -> SessionChange | None:
    """None: already paused (idempotent, nothing to store)."""
    if session.status == Status.PAUSED:
        return None
    if session.terminal:
        raise InvalidSessionTransition(f"session {session.session_id} is {session.status.value}: it cannot be paused")
    s = _advance(session, at)
    s.paused_from, s.status = s.status, Status.PAUSED
    return SessionChange(session=s, outbox=[event_item(s, EventType.TEACHING_SESSION_PAUSED,
                                                       turn_count=s.turn_count)])


def resume(session: TeachingSession, at: datetime) -> SessionChange | None:
    """None: not paused (idempotent). Resuming returns to where the session was: waiting for the learner, or owing
    teacher turns (the caller generates them)."""
    if session.terminal:
        raise InvalidSessionTransition(f"session {session.session_id} is {session.status.value}: it cannot be resumed")
    if session.status != Status.PAUSED:
        return None
    s = _advance(session, at)
    s.status, s.paused_from = s.paused_from or Status.WAITING_FOR_LEARNER, None
    return SessionChange(session=s, outbox=[event_item(s, EventType.TEACHING_SESSION_RESUMED, turn_count=s.turn_count,
                                                status=s.status.value)])


def cancel(session: TeachingSession, at: datetime) -> SessionChange | None:
    """None: already cancelled (idempotent). The history (turns, evidence, artifacts) is kept."""
    if session.status == Status.CANCELLED:
        return None
    if session.terminal:
        raise InvalidSessionTransition(f"session {session.session_id} is {session.status.value}: it cannot be "
                                       "cancelled")
    s = _advance(session, at)
    s.status, s.paused_from = Status.CANCELLED, None
    s.state.planned = []
    s.state.pending_question = None
    return SessionChange(session=s, outbox=[event_item(s, EventType.TEACHING_SESSION_CANCELLED,
                                                       turn_count=s.turn_count)])


def fail(session: TeachingSession, at: datetime, reason: str) -> SessionChange:
    s = _advance(session, at)
    s.status = Status.FAILED
    s.state.planned = []
    s.metadata = {**s.metadata, "failure": reason}
    return SessionChange(session=s, outbox=[event_item(s, EventType.TEACHING_SESSION_FAILED, reason=reason)])
