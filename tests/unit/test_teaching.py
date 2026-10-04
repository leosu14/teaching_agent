"""The interactive teaching core: difficulty controller, policy, grading, the session engine's transitions, the
in-memory repository, the summary, and the teacher agent's validation of adversarial model output."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.agents.base import OutputRejected
from app.agents.teaching.agent import TeachingSessionAgent
from app.schemas.teaching import (
    QUESTION_ACTIONS,
    CompletionReason,
    EvidenceType,
    GroundedAnswer,
    GroundingItem,
    MisconceptionCandidate,
    ObjectiveBrief,
    PendingQuestion,
    PlannedTurn,
    QuestionBrief,
    SectionBrief,
    SessionChange,
    SessionConflict,
    SessionState,
    StateBrief,
    TeacherTurnOutput,
    TeachingAction,
    TeachingConfig,
    TeachingQuestion,
    TeachingSession,
    TeachingSessionStatus,
    TeachingTurnInput,
    TurnType,
)
from app.teaching import engine
from app.teaching.errors import InvalidSessionTransition
from app.teaching.grading import grade
from app.teaching.policy import DifficultyController, TeachingPolicy
from app.teaching.repository import InMemoryTeachingRepository
from tests.unit.helpers import NOW

CONCEPT = "c.past"
QUESTION = TeachingQuestion(kind="short_answer", prompt="Yesterday I ___ home. (go)", expected_answer="went",
                            accepted_answers=["went back"])


# --- difficulty controller -----------------------------------------------------------------------------------------


def test_three_correct_answers_increase_difficulty_and_restart_the_streak() -> None:
    c = DifficultyController(TeachingConfig())
    d1 = c.after_answer(2, 0, 0, correct=True, hinted=False)
    d2 = c.after_answer(d1.difficulty, d1.successes, d1.failures, correct=True, hinted=False)
    d3 = c.after_answer(d2.difficulty, d2.successes, d2.failures, correct=True, hinted=False)
    assert [d.change for d in (d1, d2, d3)] == ["keep", "keep", "increase"]
    assert (d3.difficulty, d3.successes, d3.failures) == (3, 0, 0)


def test_two_incorrect_answers_decrease_difficulty() -> None:
    c = DifficultyController(TeachingConfig())
    d1 = c.after_answer(2, 0, 0, correct=False, hinted=False)
    d2 = c.after_answer(d1.difficulty, d1.successes, d1.failures, correct=False, hinted=False)
    assert (d1.change, d1.difficulty, d1.failures) == ("keep", 2, 1)
    assert (d2.change, d2.difficulty, d2.failures) == ("decrease", 1, 0)


def test_correct_after_hint_keeps_difficulty_but_counts_in_the_streak() -> None:
    c = DifficultyController(TeachingConfig(increase_after_successes=1))
    d = c.after_answer(2, 0, 1, correct=True, hinted=True)
    assert (d.change, d.difficulty, d.successes, d.failures) == ("keep", 2, 1, 0)
    assert "hint" in d.reason


def test_difficulty_is_clamped_to_the_configured_range() -> None:
    c = DifficultyController(TeachingConfig(increase_after_successes=1, decrease_after_failures=1))
    assert c.after_answer(3, 0, 0, correct=True, hinted=False).change == "keep"
    assert c.after_answer(1, 0, 0, correct=False, hinted=False).change == "keep"


def test_an_incorrect_answer_resets_the_success_streak() -> None:
    c = DifficultyController(TeachingConfig())
    d = c.after_answer(2, 2, 0, correct=False, hinted=False)
    assert (d.successes, d.failures) == (0, 1)
    d = c.after_answer(2, 0, 1, correct=True, hinted=False)
    assert (d.successes, d.failures) == (1, 0)


@pytest.mark.parametrize(("increase", "decrease", "answers", "expected"), [
    (1, 1, [True], 3),
    (2, 1, [True, True], 3),
    (2, 1, [False], 1),
    (4, 3, [False, False], 2),
    (4, 3, [False, False, False], 1),
])
def test_thresholds_are_configurable(increase, decrease, answers, expected) -> None:
    c = DifficultyController(TeachingConfig(increase_after_successes=increase, decrease_after_failures=decrease))
    difficulty, s, f = 2, 0, 0
    for correct in answers:
        d = c.after_answer(difficulty, s, f, correct=correct, hinted=False)
        difficulty, s, f = d.difficulty, d.successes, d.failures
    assert difficulty == expected


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TeachingConfig(min_difficulty=3, max_difficulty=2)
    with pytest.raises(ValidationError):
        TeachingConfig(start_difficulty=5)
    assert TeachingConfig().fingerprint() == TeachingConfig().fingerprint()
    assert TeachingConfig().fingerprint() != TeachingConfig(max_questions=3).fingerprint()


# --- policy ----------------------------------------------------------------------------------------------------------


def _question(hint_level: int = 0) -> PendingQuestion:
    return PendingQuestion(turn_id="q1", action=TeachingAction.ASK, concept_id=CONCEPT, difficulty=2,
                           kind="short_answer", prompt=QUESTION.prompt, expected_answer="went",
                           hint_level=hint_level)


def _state(**kw) -> SessionState:
    base = {"objective_description": "Use the past", "concept_id": CONCEPT, "concept_name": "Past", "difficulty": 2,
            "strategy": "independent practice", "turn_budget": 40}
    return SessionState(**{**base, **kw})


@pytest.mark.parametrize(("action", "mode", "expected"), [
    ("LEARN", "assessed", [TeachingAction.EXPLAIN, TeachingAction.ASK]),
    ("REVIEW", "assessed", [TeachingAction.EXPLAIN, TeachingAction.ASK]),
    ("PRACTICE", "assessed", [TeachingAction.PRACTICE]),
    ("LEARN", "practice", [TeachingAction.PRACTICE]),
    ("EVALUATE", "assessed", [TeachingAction.CHECK]),
])
def test_opening_turns(action, mode, expected) -> None:
    assert [p.action for p in TeachingPolicy(TeachingConfig()).opening(action, mode)] == expected


def test_after_answer_correct_gives_feedback_then_a_new_question() -> None:
    planned, keep_open = TeachingPolicy(TeachingConfig()).after_answer(
        _state(), _question(), correct=True, answer_turn_id="a1", mode="assessed", misconceptions_before=0)
    assert [p.action for p in planned] == [TeachingAction.FEEDBACK, TeachingAction.ASK] and not keep_open


def test_after_answer_incorrect_hints_at_the_next_level_and_keeps_the_question_open() -> None:
    policy = TeachingPolicy(TeachingConfig())
    for level in (0, 1, 2):
        planned, keep_open = policy.after_answer(_state(), _question(level), correct=False, answer_turn_id="a1",
                                                 mode="assessed", misconceptions_before=0)
        assert [(p.action, p.hint_level) for p in planned] == [(TeachingAction.HINT, level + 1)] and keep_open


def test_after_answer_hints_exhausted_or_disabled_corrects_and_moves_on() -> None:
    planned, keep_open = TeachingPolicy(TeachingConfig()).after_answer(
        _state(), _question(3), correct=False, answer_turn_id="a1", mode="assessed", misconceptions_before=0)
    assert [p.action for p in planned] == [TeachingAction.FEEDBACK, TeachingAction.ASK]
    assert planned[0].correction and not keep_open
    planned, _ = TeachingPolicy(TeachingConfig(max_hint_level=0)).after_answer(
        _state(), _question(), correct=False, answer_turn_id="a1", mode="assessed", misconceptions_before=0)
    assert planned[0].action == TeachingAction.FEEDBACK and planned[0].correction


def test_after_answer_incorrect_with_a_repeated_misconception_reteaches() -> None:
    planned, keep_open = TeachingPolicy(TeachingConfig()).after_answer(
        _state(), _question(), correct=False, answer_turn_id="a1", mode="assessed", misconceptions_before=1)
    assert [p.action for p in planned] == [TeachingAction.RETEACH, TeachingAction.ASK] and not keep_open


def test_next_question_is_a_check_when_one_more_correct_answer_demonstrates_the_objective() -> None:
    policy = TeachingPolicy(TeachingConfig(demonstration_correct=3))
    assert policy.question_action(_state(correct_answers=1), "assessed") == TeachingAction.ASK
    assert policy.question_action(_state(correct_answers=2), "assessed") == TeachingAction.CHECK
    assert policy.question_action(_state(correct_answers=2, difficulty=1), "assessed") == TeachingAction.ASK
    assert policy.question_action(_state(correct_answers=2), "practice") == TeachingAction.PRACTICE


@pytest.mark.parametrize(("state", "kw", "expected"), [
    ({"correct_answers": 3}, {"correct": True, "hinted": False, "question_difficulty": 2},
     CompletionReason.OBJECTIVE_DEMONSTRATED),
    ({"correct_answers": 3}, {"correct": True, "hinted": True, "question_difficulty": 2}, None),
    ({"correct_answers": 3}, {"correct": True, "hinted": False, "question_difficulty": 1}, None),
    ({"incorrect_answers": 5}, {"correct": False, "hinted": False, "question_difficulty": 2},
     CompletionReason.REPEATED_FAILURE),
    ({"questions_asked": 10, "correct_answers": 1}, {"correct": True, "hinted": False, "question_difficulty": 2},
     CompletionReason.QUESTION_LIMIT_REACHED),
])
def test_completion_rules(state, kw, expected) -> None:
    policy = TeachingPolicy(TeachingConfig())
    assert policy.completion(_state(**state), question_open=False, turn_count=5, **kw) == expected


def test_completion_waits_for_an_open_question_and_respects_the_turn_budget() -> None:
    policy = TeachingPolicy(TeachingConfig(max_turns=10))
    kw = {"correct": False, "hinted": False, "question_difficulty": 2}
    assert policy.completion(_state(questions_asked=10), question_open=True, turn_count=2, **kw) is None
    assert policy.completion(_state(), question_open=True, turn_count=8, **kw) == CompletionReason.TURN_BUDGET_REACHED


# --- grading -------------------------------------------------------------------------------------------------------


def test_grading_is_deterministic_and_normalised() -> None:
    q = _question()
    assert grade(q, "went") and grade(q, "  Went! ") and not grade(q, "goed") and not grade(q, "")
    accented = q.model_copy(update={"expected_answer": "sonó"})
    assert grade(accented, "Sonó.") and not grade(accented, "sono")  # accents are part of the answer


def test_multiple_choice_accepts_labels_only_when_they_are_not_choices() -> None:
    q = _question().model_copy(update={"kind": "multiple_choice", "choices": ["go", "went", "gone"],
                                       "expected_answer": "went"})
    assert grade(q, "b") and grade(q, "2") and grade(q, "went") and not grade(q, "a")
    numbers = q.model_copy(update={"choices": ["1", "2", "3"], "expected_answer": "2"})
    # "1" is a choice, so it is not read as the label of the first one; the letter "b" still names the second
    assert grade(numbers, "2") and not grade(numbers, "1") and grade(numbers, "b") and not grade(numbers, "c")


# --- engine --------------------------------------------------------------------------------------------------------


def new_session(config: TeachingConfig | None = None, action: str = "LEARN", mode: str = "assessed"
                ) -> TeachingSession:
    change = engine.start(session_id="tsess_1", learner_id="l1", task_id="t1", lesson_id="art_lesson",
                          objective_id="obj_1", objective_description="Use the past", concept_id=CONCEPT,
                          concept_name="Past", action=action, mode=mode,
                          config=config or TeachingConfig(increase_after_successes=2, decrease_after_failures=1),
                          at=NOW, metadata={})
    return change.session


def output(session: TeachingSession, item: PlannedTurn, *, question: TeachingQuestion = QUESTION,
           misconceptions=(), action: TeachingAction | None = None) -> TeacherTurnOutput:
    asks = item.action in QUESTION_ACTIONS
    return TeacherTurnOutput(
        action=action or (TeachingAction.EXPLAIN if item.purpose == "answer_question" else item.action),
        concept_id=CONCEPT, difficulty=session.state.difficulty, response=f"{item.action.value} turn",
        expected_response_type="free_text" if asks else "none", question=question if asks else None,
        hint_level=item.hint_level if item.action == TeachingAction.HINT else 0,
        misconceptions=list(misconceptions))


def teach(session: TeachingSession, **kw) -> tuple[TeachingSession, list]:
    """Produce every owed teacher turn with valid output, then settle: what the service's second phase does."""
    s = session.model_copy(deep=True)
    s.version += 1
    turns, evidence, outbox = [], [], []
    while s.state.planned:
        part = engine.apply_teacher(s, s.state.planned[0], output(s, s.state.planned[0], **kw), NOW)
        turns += part.turns
        evidence += part.evidence
        outbox += part.outbox
    outbox += engine.settle(s, NOW, "narrative")
    return s, [turns, evidence, outbox]


def test_start_plans_the_opening_and_then_waits_with_a_question() -> None:
    s = new_session()
    assert s.status == TeachingSessionStatus.ACTIVE and s.state.difficulty == 2
    assert [p.action for p in s.state.planned] == [TeachingAction.EXPLAIN, TeachingAction.ASK]
    s, (turns, _, outbox) = teach(s)
    assert s.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    assert [t.turn_type for t in turns] == [TurnType.EXPLANATION, TurnType.QUESTION]
    assert [t.sequence for t in turns] == [1, 2]
    assert s.state.pending_question is not None and s.state.pending_question.expected_answer == "went"
    assert s.state.questions_asked == 1 and s.state.difficulty_trajectory == [2]
    assert {o.kind for o in outbox} == {"artifact", "event"}


def test_answers_drive_difficulty_hints_misconceptions_and_completion() -> None:
    s, _ = teach(new_session())
    wrong = engine.record_answer(s, "goed", NOW + timedelta(seconds=1))
    s = wrong.change.session
    assert not wrong.correct and wrong.difficulty_change is not None
    assert (wrong.difficulty_change.before, wrong.difficulty_change.after) == (2, 1)
    assert s.status == TeachingSessionStatus.ACTIVE
    assert [(p.action, p.hint_level) for p in s.state.planned] == [(TeachingAction.HINT, 1)]
    candidates = [MisconceptionCandidate(concept_id=CONCEPT, misconception="adds -ed to irregular verbs",
                                         confidence=0.8),
                  MisconceptionCandidate(concept_id=CONCEPT, misconception="Adds -ed to irregular verbs!",
                                         confidence=0.9),
                  MisconceptionCandidate(concept_id=CONCEPT, misconception="a guess", confidence=0.2)]
    s, (turns, evidence, _) = teach(s, misconceptions=candidates)
    assert [t.turn_type for t in turns] == [TurnType.HINT] and turns[0].metadata["hint_level"] == 1
    kinds = [e.evidence_type for e in evidence]
    assert kinds == [EvidenceType.HINT, EvidenceType.MISCONCEPTION]  # duplicate and low-confidence dropped
    assert s.state.pending_question.hint_level == 1 and s.state.hints_used == 1

    hinted = engine.record_answer(s, "went", NOW + timedelta(seconds=2))
    ev = hinted.change.evidence[0]
    assert hinted.correct and hinted.difficulty_change is None
    assert (ev.hint_level, ev.answer_after_hint, ev.correct) == (1, True, True)
    s, (turns, _, _) = teach(hinted.change.session)
    assert turns[0].turn_type == TurnType.ENCOURAGEMENT and s.state.difficulty == 1

    up = engine.record_answer(s, "went back", NOW + timedelta(seconds=3))
    assert up.correct and up.difficulty_change is not None and up.difficulty_change.after == 2
    s, (turns, _, _) = teach(up.change.session)
    assert turns[-1].turn_type == TurnType.ASSESSMENT  # one more correct answer demonstrates the objective

    final = engine.record_answer(s, "went", NOW + timedelta(seconds=4))
    s = final.change.session
    assert s.state.completion_reason == CompletionReason.OBJECTIVE_DEMONSTRATED
    assert [p.action for p in s.state.planned] == [TeachingAction.SUMMARIZE]
    s, (turns, _, outbox) = teach(s)
    assert s.status == TeachingSessionStatus.COMPLETED and turns[-1].turn_type == TurnType.SUMMARY
    assert s.summary is not None and s.summary.difficulty_trajectory == [2, 1, 2]
    assert (s.summary.correct_answers, s.summary.incorrect_answers, s.summary.hints_used) == (3, 1, 1)
    assert s.summary.misconceptions == ["adds -ed to irregular verbs"]
    assert s.summary.recommended_next_action == "advance"
    assert [o.kind for o in outbox][-2:] == ["artifact", "finalize"]


def test_a_repeated_misconception_leads_to_reteaching() -> None:
    s, _ = teach(new_session(TeachingConfig(decrease_after_failures=5)))
    s, _ = teach(engine.record_answer(s, "goed", NOW).change.session,
                 misconceptions=[MisconceptionCandidate(concept_id=CONCEPT, misconception="regularises",
                                                        confidence=0.9)])
    again = engine.record_answer(s, "goed", NOW).change.session
    assert [p.action for p in again.state.planned] == [TeachingAction.RETEACH, TeachingAction.ASK]
    assert again.state.pending_question is None and again.state.strategy.startswith("reteach")


def test_misconceptions_are_not_recorded_for_a_correct_answer() -> None:
    s, _ = teach(new_session())
    s = engine.record_answer(s, "went", NOW).change.session
    _, (_, evidence, _) = teach(s, misconceptions=[MisconceptionCandidate(
        concept_id=CONCEPT, misconception="none really", confidence=0.9)])
    assert not [e for e in evidence if e.evidence_type == EvidenceType.MISCONCEPTION]


def test_learner_input_is_refused_unless_the_session_waits_for_it() -> None:
    s = new_session()
    with pytest.raises(InvalidSessionTransition):
        engine.record_answer(s, "went", NOW)  # ACTIVE: the teacher owes turns
    s, _ = teach(s)
    paused = engine.pause(s, NOW).session
    with pytest.raises(InvalidSessionTransition, match="paused"):
        engine.record_answer(paused, "went", NOW)
    with pytest.raises(InvalidSessionTransition):
        engine.record_question(paused, "why?", NOW)


def test_pause_resume_cancel_are_idempotent_and_terminal_states_are_final() -> None:
    s, _ = teach(new_session())
    paused = engine.pause(s, NOW).session
    assert engine.pause(paused, NOW) is None
    resumed = engine.resume(paused, NOW).session
    assert resumed.status == TeachingSessionStatus.WAITING_FOR_LEARNER and engine.resume(resumed, NOW) is None
    cancelled = engine.cancel(resumed, NOW).session
    assert cancelled.status == TeachingSessionStatus.CANCELLED and engine.cancel(cancelled, NOW) is None
    for transition in (engine.pause, engine.resume):
        with pytest.raises(InvalidSessionTransition):
            transition(cancelled, NOW)
    assert cancelled.turn_count == s.turn_count  # the history is kept


def test_a_learner_question_keeps_the_open_question_and_stop_completes() -> None:
    s, _ = teach(new_session())
    asked = engine.record_question(s, "Why is it irregular?", NOW)
    assert asked.session.state.pending_question is not None
    assert asked.evidence[0].evidence_type == EvidenceType.LEARNER_QUESTION
    s, (turns, _, _) = teach(asked.session)
    assert turns[0].metadata["responds_to"] == asked.turns[0].turn_id
    assert s.status == TeachingSessionStatus.WAITING_FOR_LEARNER
    stopped = engine.record_stop(s, "stop", NOW).session
    s, _ = teach(stopped)
    assert s.status == TeachingSessionStatus.COMPLETED
    assert s.summary.completion_reason == CompletionReason.LEARNER_STOPPED
    assert s.summary.recommended_next_action == "practice"


def test_practice_mode_marks_its_evidence() -> None:
    s, _ = teach(new_session(mode="practice"))
    assert s.state.pending_question.action == TeachingAction.PRACTICE
    ev = engine.record_answer(s, "went", NOW).change.evidence[0]
    assert ev.practice


def test_the_state_boundary_rejects_a_turn_that_is_not_the_planned_one() -> None:
    s = new_session()
    item = s.state.planned[0]
    for bad in (output(s, item, action=TeachingAction.COMPLETE), output(s, item, action=TeachingAction.ASK)):
        with pytest.raises(ValueError):
            engine.apply_teacher(s.model_copy(deep=True), item, bad, NOW)
    wrong_concept = output(s, item).model_copy(update={"concept_id": "c.other"})
    with pytest.raises(ValueError):
        engine.apply_teacher(s.model_copy(deep=True), item, wrong_concept, NOW)


def test_ids_are_stable() -> None:
    assert engine.session_id_for("l", "lesson", "k") == engine.session_id_for("l", "lesson", "k")
    assert engine.session_id_for("l", "lesson", "k") != engine.session_id_for("l", "lesson", "k2")
    assert engine.turn_id_for("s", 1) != engine.turn_id_for("s", 2)


# --- in-memory repository ------------------------------------------------------------------------------------------


def test_repository_uses_optimistic_locking() -> None:
    repo = InMemoryTeachingRepository()
    start = engine.start(session_id="tsess_r", learner_id="l1", task_id="t1", lesson_id="art", objective_id=None,
                         objective_description="d", concept_id=CONCEPT, concept_name="Past", action="LEARN",
                         mode="assessed", config=TeachingConfig(), at=NOW, metadata={})
    assert repo.create(start) and not repo.create(start)
    s = repo.get("tsess_r")
    taught, (turns, evidence, outbox) = teach(s)
    repo.apply(SessionChange(session=taught, turns=turns, evidence=evidence, outbox=outbox),
               expected_version=s.version)
    with pytest.raises(SessionConflict):  # a stale writer
        repo.apply(SessionChange(session=taught, turns=turns), expected_version=s.version)
    assert [t.sequence for t in repo.turns("tsess_r")] == [1, 2]
    pending = repo.pending_outbox("tsess_r")
    assert pending
    repo.mark_published(pending[0].item_id)
    assert len(repo.pending_outbox("tsess_r")) == len(pending) - 1


# --- the teacher agent's validation (adversarial model output) ------------------------------------------------------


def turn_input(action: TeachingAction = TeachingAction.HINT, **kw) -> TeachingTurnInput:
    base = dict(
        stage="turn", action=action, subject="english", level="A2", language="en",
        objective=ObjectiveBrief(concept_id=CONCEPT, concept_name="Past", description="Use the past"),
        lesson_title="The past",
        sections=[SectionBrief(ref="lesson:sec_1", concept_id=CONCEPT, heading="Past", explanation="...",
                               citations=["cit_1"])],
        state=StateBrief(difficulty=2, min_difficulty=1, max_difficulty=3, strategy="independent practice",
                         questions_asked=1, correct_answers=0, incorrect_answers=1, consecutive_successes=0,
                         consecutive_failures=1, hints_used=0, mastery=0.3),
        question=QuestionBrief(kind="short_answer", prompt=QUESTION.prompt, expected_answer="went", difficulty=2),
        learner_answer="goed", answer_correct=False, hint_level=1 if action == TeachingAction.HINT else 0,
        difficulty=2)
    return TeachingTurnInput(**{**base, **kw})


def good(source: TeachingTurnInput, **kw) -> TeacherTurnOutput:
    asks = source.action in QUESTION_ACTIONS
    base = dict(action=source.action, concept_id=CONCEPT, difficulty=2, response="Think about irregular verbs.",
                expected_response_type="free_text" if asks else "none", question=QUESTION if asks else None,
                hint_level=source.hint_level if source.action == TeachingAction.HINT else 0)
    return TeacherTurnOutput(**{**base, **kw})


AGENT = TeachingSessionAgent()


def test_a_valid_turn_passes() -> None:
    AGENT.check(good(turn_input()), turn_input())
    ask = turn_input(TeachingAction.ASK, learner_answer=None, answer_correct=None, question=None)
    AGENT.check(good(ask), ask)


@pytest.mark.parametrize(("change", "message"), [
    ({"action": TeachingAction.ASK}, "must be HINT"),
    ({"action": TeachingAction.COMPLETE}, "cannot complete the session"),
    ({"concept_id": "c.other"}, "concept"),
    ({"difficulty": 3}, "difficulty"),
    ({"citations": ["https://invented.example/source"]}, "do not exist"),
    ({"response": "The answer is went."}, "reveal"),
    ({"hint_level": 3}, "hint_level"),
    ({"grounded": True}, "learner questions only"),
    ({"question": QUESTION, "expected_response_type": "free_text"}, "exactly one question"),
    ({"misconceptions": [MisconceptionCandidate(concept_id="c.other", misconception="confuses", confidence=0.9)]},
     "misconceptions must be about"),
])
def test_adversarial_turns_are_rejected(change, message) -> None:
    source = turn_input()
    with pytest.raises(OutputRejected, match=message):
        AGENT.check(good(source).model_copy(update=change), source)


def test_a_question_cannot_contain_its_answer_and_misconceptions_need_a_wrong_answer() -> None:
    ask = turn_input(TeachingAction.ASK, learner_answer=None, answer_correct=None, question=None)
    leaky = TeachingQuestion(kind="short_answer", prompt="Yesterday I went home: write went.",
                             expected_answer="went")
    with pytest.raises(OutputRejected, match="its own answer"):
        AGENT.check(good(ask, question=leaky), ask)
    feedback = turn_input(TeachingAction.FEEDBACK, answer_correct=True, learner_answer="went")
    with pytest.raises(OutputRejected, match="incorrect answer"):
        AGENT.check(good(feedback, misconceptions=[MisconceptionCandidate(
            concept_id=CONCEPT, misconception="none", confidence=0.9)]), feedback)


def test_a_revealing_hint_is_allowed_only_when_configured() -> None:
    source = turn_input(reveal_answer=True)
    AGENT.check(good(source, response="The answer is went."), source)


def test_grounded_answers_cite_only_the_material() -> None:
    source = turn_input(TeachingAction.EXPLAIN, stage="answer", learner_question="why?", sources=[
        GroundingItem(ref="lesson:sec_1", kind="lesson", title="Past", text="...", citations=["cit_1"])])
    AGENT.check(GroundedAnswer(response="Because ...", citations=["lesson:sec_1", "cit_1"], grounded=True), source)
    with pytest.raises(OutputRejected, match="not in the grounding material"):
        AGENT.check(GroundedAnswer(response="Because ...", citations=["doi:10.1/fake"], grounded=True), source)


def test_the_output_schemas_have_no_way_to_change_mastery_or_complete_anything() -> None:
    data = good(turn_input()).model_dump(mode="json")
    for field, value in (("mastery", 1.0), ("session_status", "COMPLETED"), ("objective_completed", True)):
        with pytest.raises(ValidationError):
            TeacherTurnOutput.model_validate({**data, field: value})
    with pytest.raises(ValidationError):
        GroundedAnswer(response="x", grounded=True)  # grounded without citations
    with pytest.raises(ValidationError):
        GroundedAnswer(response="x", grounded=False)  # ungrounded without a stated limitation
    with pytest.raises(ValidationError):
        TeachingQuestion(kind="multiple_choice", prompt="?", choices=["a", "b"], expected_answer="c")
