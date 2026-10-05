"""The assessment service end to end on a generated curriculum lesson: deterministic and semantic grading, attempts,
idempotency, uncertainty, mastery through learner memory, artifacts and events, the model boundary, cost, privacy,
crash recovery, and the two integrations (interactive teaching sessions, the evaluation workflow)."""

from __future__ import annotations

from collections import Counter

import pytest

from app.config.providers import ProviderSettings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders, evaluate_learner
from app.schemas.assessment import (
    AttemptConflict,
    AttemptExists,
    AttemptSubmission,
    ItemConflict,
    RegisterAssessmentItem,
)
from app.schemas.evaluation import AssessmentQuestion, EvaluationInput, LearnerEvaluationReport
from app.schemas.learner import LearnerProfileInput
from app.schemas.task import TaskStatus
from app.schemas.teaching import (
    QUESTION_ACTIONS,
    LearnerInput,
    StartTeachingSession,
    TeacherTurnOutput,
    TeachingQuestion,
    TeachingSessionStatus,
    TeachingTurnInput,
)
from app.services.assessment import AssessmentItemNotFound, AttemptNotFound, InvalidAssessmentRequest
from app.services.container import build_container
from scripts.run_assessment_demo import register
from tests.conftest import evaluation_answers_for, start_evaluation
from tests.teaching_fixtures import Env, open_env

CONCEPT = "es.past_contrast"
GRADER = "semantic_grader"
EVENTS = ("assessment.", "misconception.")


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing catches it."""


def setup(env: Env) -> tuple[dict, str]:
    learner = env.container.task_service.get(env.lesson_task).learner_id
    return register(env.container, env.lesson_task, None), learner


async def submit(env: Env, items: dict, key: str, say: str, learner: str, attempt_id: str | None = None,
                 answer: str | None = None):
    entry = items[key]
    return await env.container.assessment_service.submit_attempt(
        entry["item"].assessment_item_id,
        AttemptSubmission(learner_id=learner, answer=answer or entry["answers"][say], attempt_id=attempt_id))


def mastery(env: Env, learner: str) -> float:
    return env.container.memory.get(learner).concepts[CONCEPT].mastery


def assessment_events(env: Env):
    return [e for e in env.container.task_service.events(env.lesson_task) if e.type.startswith(EVENTS)]


async def test_deterministic_grading_calls_no_model_and_semantic_grading_one(teaching_env: Env) -> None:
    env = teaching_env
    items, learner = setup(env)
    exact = await submit(env, items, "conjugate", "exact", learner)
    wrong = await submit(env, items, "conjugate", "known", learner)
    rubric = await submit(env, items, "explain_story", "rubric", learner)
    assert [r.grade.grader_type.value for r in (exact, wrong, rubric)] == ["EXACT", "RULE", "RUBRIC"]
    assert env.llm.calls[GRADER] == 0
    semantic = await submit(env, items, "why_preterite", "semantic", learner)
    assert (semantic.grade.outcome.value, semantic.grade.grader_type.value) == ("CORRECT", "SEMANTIC")
    assert env.llm.calls[GRADER] == 1
    usage = env.container.assessment_service.stored_grade(semantic.attempt.attempt_id).grader
    assert (usage.llm_calls, usage.provider, usage.model) == (1, "mock", "mock-standard")
    assert usage.input_tokens > 0 and usage.output_tokens > 0 and usage.estimated_cost_usd > 0
    graded = [e for e in env.events if e.type == "assessment.graded" and e.data["attempt_id"] ==
              semantic.attempt.attempt_id]
    assert graded[0].data["llm_calls"] == 1 and graded[0].data["model"] == "mock-standard"


async def test_attempts_are_idempotent_numbered_and_never_overwritten(teaching_env: Env) -> None:
    env = teaching_env
    items, learner = setup(env)
    first = await submit(env, items, "why_preterite", "partial", learner, attempt_id="att-1")
    evidence = len(env.container.memory.evidence(learner))
    calls = env.llm.calls[GRADER]
    again = await submit(env, items, "why_preterite", "partial", learner, attempt_id="att-1")
    assert again.replayed and again.grade == first.grade and again.attempt == first.attempt
    assert env.llm.calls[GRADER] == calls and len(env.container.memory.evidence(learner)) == evidence
    with pytest.raises(AttemptConflict):
        await submit(env, items, "why_preterite", "semantic", learner, attempt_id="att-1")
    second = await submit(env, items, "why_preterite", "semantic", learner, attempt_id="att-2")
    third = await submit(env, items, "why_preterite", "misconception", learner)
    assert [r.attempt.attempt_number for r in (first, second, third)] == [1, 2, 3]
    history = env.container.assessment_service.attempts(items["why_preterite"]["item"].assessment_item_id, learner)
    assert [(a.attempt_id, a.outcome.value) for a in history][:2] == [("att-1", "PARTIAL"), ("att-2", "CORRECT")]
    assert all(a.completed and a.submitted_at and a.grade_id for a in history)
    assert len({a.grade_id for a in history}) == 3
    started = [e for e in assessment_events(env) if e.type == "assessment.started"]
    assert len(started) == 3 and len({e.event_id for e in assessment_events(env)}) == len(assessment_events(env))


async def test_uncertain_is_kept_never_counts_and_recommends_a_retry(teaching_env: Env) -> None:
    env = teaching_env
    items, learner = setup(env)
    await submit(env, items, "conjugate", "exact", learner)
    before, evidence = mastery(env, learner), len(env.container.memory.evidence(learner))
    r = await submit(env, items, "why_preterite", "uncertain", learner)
    assert r.grade.outcome.value == "UNCERTAIN" and r.attempt.retry_recommended
    assert not r.attempt.mastery_updated and r.attempt.evidence_ids == [] and r.mastery_changes == []
    assert mastery(env, learner) == before and len(env.container.memory.evidence(learner)) == evidence
    assert r.grade.feedback.next_hint and r.grade.score == 0.0 and r.grade.misconceptions == []
    stored = env.container.assessment_service.stored_grade(r.attempt.attempt_id)
    assert stored.uncertainty_reason and "confidence" in stored.uncertainty_reason
    assert [e.type for e in assessment_events(env)][-3:] == ["assessment.started", "assessment.uncertain",
                                                             "assessment.completed"]
    retry = await submit(env, items, "why_preterite", "semantic", learner)
    assert retry.attempt.attempt_number == 2 and retry.grade.outcome.value == "CORRECT"
    assert env.container.assessment_service.attempt(r.attempt.attempt_id).outcome.value == "UNCERTAIN"


async def test_graded_answers_update_mastery_through_learner_memory(teaching_env: Env) -> None:
    env = teaching_env
    items, learner = setup(env)
    before = mastery(env, learner)
    r = await submit(env, items, "why_preterite", "semantic", learner)
    change = next(c for c in r.mastery_changes if c["concept_id"] == CONCEPT)
    assert change["before"] == pytest.approx(before) and change["after"] > before == pytest.approx(before)
    evidence = [e for e in env.container.memory.evidence(learner) if e.evidence_id in r.attempt.evidence_ids]
    assert [(e.source_type, e.correctness, e.score) for e in evidence] == [("assessment", "correct", 1.0)]
    partial = await submit(env, items, "why_preterite", "partial", learner)
    evidence = [e for e in env.container.memory.evidence(learner) if e.evidence_id in partial.attempt.evidence_ids]
    assert [(e.correctness, e.score) for e in evidence] == [("partial", 0.4)]
    wrong = await submit(env, items, "why_preterite", "misconception", learner)
    evidence = [e for e in env.container.memory.evidence(learner) if e.evidence_id in wrong.attempt.evidence_ids]
    assert evidence[0].correctness == "incorrect"
    assert [m["type"] for m in evidence[0].metadata["misconceptions"]] == ["preterite_as_habitual"]
    assert wrong.learning_action and wrong.learning_action["action"]
    detected = [e for e in assessment_events(env) if e.type == "misconception.detected"]
    assert [e.data["misconception_type"] for e in detected] == ["preterite_as_habitual"]


async def test_artifact_lineage(teaching_env: Env) -> None:
    env = teaching_env
    items, learner = setup(env)
    r = await submit(env, items, "why_preterite", "semantic", learner)
    arts = env.container.task_service.artifacts(env.lesson_task)
    by_type = Counter(a.type.value for a in arts)
    assert by_type["ASSESSMENT_ITEM"] == 3 and by_type["ASSESSMENT_RUBRIC"] == 2
    assert by_type["ASSESSMENT_GRADE"] == 1 and by_type["ASSESSMENT_FEEDBACK"] == 1
    grade = next(a for a in arts if a.type.value == "ASSESSMENT_GRADE")
    up = {a.type.value for a in env.container.artifacts.lineage(grade.artifact_id)}
    assert {"LESSON", "ASSESSMENT_ITEM", "ASSESSMENT_RUBRIC"} <= up
    evidence = next(a for a in arts if a.type.value == "LEARNING_EVIDENCE"
                    and a.metadata.get("attempt_id") == r.attempt.attempt_id)
    assert grade.artifact_id in evidence.parent_ids
    feedback = next(a for a in arts if a.type.value == "ASSESSMENT_FEEDBACK")
    assert grade.artifact_id in feedback.parent_ids
    names = Counter(a.name for a in arts)
    assert all(n == 1 for n in names.values())  # registering again stores nothing twice
    register(env.container, env.lesson_task, None)
    assert len(env.container.task_service.artifacts(env.lesson_task)) == len(arts)


@pytest.mark.parametrize("bad", [
    "{not json",
    '{"score": 1.7, "criterion_results": [], "outcome": "CORRECT", "confidence": 0.9}',
    '{"score": 1.0, "outcome": "CORRECT", "confidence": 0.99, "mastery": 1.0, "objective_status": "MASTERED", '
    '"criterion_results": [{"criterion_id": "completed", "score": 1}, {"criterion_id": "bounded", "score": 1}]}',
    '{"score": 1.0, "outcome": "CORRECT", "confidence": 0.99, "citations": ["https://invented.example"], '
    '"criterion_results": [{"criterion_id": "completed", "score": 1}, {"criterion_id": "bounded", "score": 1}]}',
])
async def test_invalid_grader_output_is_retried_and_never_reaches_the_grade(teaching_env: Env, bad: str) -> None:
    env = teaching_env
    items, learner = setup(env)
    env.llm.inject(GRADER, bad)  # one bad output: the retry is valid
    retried = await submit(env, items, "why_preterite", "semantic", learner)
    assert retried.grade.outcome.value == "CORRECT" and env.llm.calls[GRADER] == 2
    assert env.container.assessment_service.stored_grade(retried.attempt.attempt_id).grader.llm_calls == 2
    env.llm.inject(GRADER, bad, bad, bad)  # every attempt bad: UNCERTAIN, never INCORRECT
    before = mastery(env, learner)
    failed = await submit(env, items, "why_preterite", "semantic", learner)
    assert failed.grade.outcome.value == "UNCERTAIN" and not failed.attempt.mastery_updated
    assert mastery(env, learner) == before
    assert "no valid grade" in env.container.assessment_service.stored_grade(
        failed.attempt.attempt_id).uncertainty_reason


async def test_the_grader_sees_minimal_context_and_nothing_secret_is_stored(teaching_lesson, tmp_path) -> None:
    secret = "sk-assessment-secret-0123456789"
    template, task_id = teaching_lesson
    env = open_env(tmp_path / "data", task_id, copy_from=template,
                   providers=ProviderSettings(openai_api_key=secret, llm_routes={}))
    try:
        items, learner = setup(env)
        for say in ("semantic", "partial", "misconception", "uncertain"):
            await submit(env, items, "why_preterite", say, learner)
        goal = env.container.memory.goals(learner)[0].goal_id
        requests = [r for r in env.llm.requests if r.agent_id == GRADER]
        assert len(requests) == 4
        for r in requests:
            rubric_id = r.input_payload["rubric"]["rubric_id"]  # the item's own rubric (named after its lesson here)
            text = (r.system + "".join(m.content for m in r.messages) + str(r.input_payload)).replace(rubric_id, "")
            for private in (learner, task_id, goal, secret, "learner_id", "display_name", "es.subjunctive"):
                assert private not in text, private
            assert "mastery" not in str(r.input_payload)  # nothing about the learner's state
            assert set(r.input_payload) == {"language", "item", "rubric", "learner_answer", "lesson_context",
                                            "research_evidence"}
        for e in assessment_events(env):
            dumped = e.model_dump_json()
            assert secret not in dumped and "expected_answer" not in dumped and "learner_answer" not in e.data
        stored = b"".join(p.read_bytes() for p in env.data_dir.rglob("*") if p.is_file())
        assert secret.encode() not in stored
    finally:
        env.close()


async def test_a_crash_after_grading_resumes_without_grading_again(teaching_env: Env, monkeypatch) -> None:
    env = teaching_env
    items, learner = setup(env)
    service = env.container.assessment_service
    original = service._publish

    def crash(*args, **kwargs):
        raise SimulatedCrash()

    monkeypatch.setattr(service, "_publish", crash)
    with pytest.raises(SimulatedCrash):
        await submit(env, items, "why_preterite", "semantic", learner, attempt_id="crashy")
    stored = service.attempt("crashy")
    assert not stored.completed and not stored.mastery_updated  # graded and kept, not yet published
    monkeypatch.setattr(service, "_publish", original)
    env2 = env.reopen()
    service = env2.container.assessment_service
    done = await submit(env2, items, "why_preterite", "semantic", learner, attempt_id="crashy")
    assert done.replayed and done.attempt.completed and done.attempt.mastery_updated
    assert env.llm.calls[GRADER] + env2.llm.calls[GRADER] == 1  # graded once, across the restart
    again = await submit(env2, items, "why_preterite", "semantic", learner, attempt_id="crashy")
    assert again.attempt.evidence_ids == done.attempt.evidence_ids
    events = [e for e in env2.container.task_service.events(env.lesson_task) if e.type.startswith(EVENTS)]
    assert Counter(e.type for e in events)["assessment.completed"] == 1


async def test_invalid_requests(teaching_env: Env) -> None:
    env = teaching_env
    items, learner = setup(env)
    service = env.container.assessment_service
    item = items["why_preterite"]["item"]
    elsewhere = item.model_copy(update={"assessment_item_id": "x", "concept_id": "es.subjunctive", "rubric_id": None})
    with pytest.raises(InvalidAssessmentRequest, match="does not teach"):
        service.register(RegisterAssessmentItem(item=elsewhere))
    with pytest.raises(InvalidAssessmentRequest, match="not registered"):
        service.register(RegisterAssessmentItem(item=elsewhere.model_copy(
            update={"concept_id": CONCEPT, "rubric_id": "missing"})))
    with pytest.raises(ItemConflict):
        service.register(RegisterAssessmentItem(item=item.model_copy(update={"expected_answer": "Something else."})))
    env.container.learner_service.upsert("someone-else", LearnerProfileInput())
    with pytest.raises(InvalidAssessmentRequest, match="another learner"):
        await submit(env, items, "why_preterite", "semantic", "someone-else")
    with pytest.raises(AssessmentItemNotFound):
        await service.submit_attempt("nope", AttemptSubmission(learner_id=learner, answer="x"))
    with pytest.raises(AttemptNotFound):
        service.attempt("nope")


# --- interactive teaching sessions --------------------------------------------------------------------------------

FREE_PROMPT = "¿Por qué usamos el pretérito en «Ayer terminé el libro»?"
FREE_EXPECTED = "la acción terminó en un momento concreto"


def free_text_teacher(request) -> dict:
    """The mock teacher, asking one free-text question instead of its short-answer ones."""
    out = TeacherTurnOutput.model_validate(default_responders()["teaching_session"](request))
    p = TeachingTurnInput.model_validate(request.input_payload)
    if p.stage == "turn" and p.action in QUESTION_ACTIONS:
        out = out.model_copy(update={
            "question": TeachingQuestion(kind="free_text", prompt=FREE_PROMPT, expected_answer=FREE_EXPECTED),
            "expected_response_type": "free_text", "response": FREE_PROMPT})
    return out.model_dump(mode="json")


async def test_a_teaching_session_grades_free_text_through_the_assessment_service(teaching_lesson, tmp_path) -> None:
    template, task_id = teaching_lesson
    env = open_env(tmp_path / "data", task_id, copy_from=template,
                   responders={**default_responders(), "teaching_session": free_text_teacher})
    try:
        service = env.container.teaching_service
        sid = (await service.start(task_id, StartTeachingSession(idempotency_key="free"))).session_id
        view = await service.view(sid)
        assert view.state.waiting_question.kind == "free_text"

        unclear = await service.submit(sid, LearnerInput(answer="Mi gato come pescado.", client_turn_id="u1"))
        assert unclear.answer_outcome == "UNCERTAIN" and unclear.correct is None
        assert unclear.assessment.outcome.value == "UNCERTAIN" and unclear.difficulty_change is None
        assert unclear.waiting_question is not None and unclear.waiting_question.prompt == FREE_PROMPT
        assert FREE_EXPECTED not in unclear.teacher_turns[0].content  # asked again, not revealed
        assert all(e.correct is None for e in unclear.evidence)

        partial = await service.submit(sid, LearnerInput(answer="Porque la acción terminó.", client_turn_id="u2"))
        assert partial.answer_outcome == "PARTIAL" and 0 < partial.assessment.score < 1
        assert partial.assessment.grader_type.value == "SEMANTIC" and partial.teacher_turns[0].content.startswith(
            "Partly right")

        right = await service.submit(sid, LearnerInput(
            answer="Porque la acción terminó en un momento concreto del pasado.", client_turn_id="u3"))
        assert right.answer_outcome == "CORRECT" and right.correct is True

        replay = await service.submit(sid, LearnerInput(answer="Mi gato come pescado.", client_turn_id="u1"))
        assert replay.replayed and replay.answer_outcome == "UNCERTAIN"
        assert env.llm.calls[GRADER] == 3  # one grade per answer, none for the replay

        state = (await service.view(sid)).state
        assert (state.uncertain_answers, state.partial_answers, state.correct_answers) == (1, 1, 1)
        attempts = env.container.assessment_service.repository
        session = service.repository.get(sid)
        learner_turns = [t for t in service.turns(sid) if t.metadata.get("attempt_id")]
        assert len(learner_turns) == 3
        assert all(attempts.attempt(t.metadata["attempt_id"]).source == "teaching_session" for t in learner_turns)
        assert session.status in (TeachingSessionStatus.WAITING_FOR_LEARNER, TeachingSessionStatus.COMPLETED)
    finally:
        env.close()


# --- the evaluation workflow --------------------------------------------------------------------------------------


def with_free_text(request) -> dict:
    """The mock evaluator, adding one free-text question per assessment."""
    step = evaluate_learner(request)
    p = EvaluationInput.model_validate(request.input_payload)
    if p.stage == "assess":
        section = p.lesson.sections[0]
        step["assessment"]["questions"].append(AssessmentQuestion(
            question_id="ft_1", concept_id=section.concept_id, objective=step["assessment"]["objectives"][0],
            kind="free_text", prompt="Explain in your own words why the preterite tells the match result.",
            difficulty=0.6, expected_answer="the match finished at a specific moment in the past",
        ).model_dump(mode="json"))
        step["assessment"]["total_points"] += 1
    return step


@pytest.mark.parametrize(("answer", "outcome", "counted"), [
    ("Because the match finished at a specific moment in the past.", "CORRECT", True),
    ("My cat eats fish on sundays.", "UNCERTAIN", False),
])
async def test_the_evaluation_workflow_grades_through_the_assessment_service(settings, answer, outcome,
                                                                            counted) -> None:
    llm = MockLLMProvider({**default_responders(), "learner_evaluation": with_free_text})
    container = build_container(settings, llm_providers={"mock": llm})
    try:
        _, task = await start_evaluation(container)
        questions = task.waiting.prompt["questions"]
        assert any(q["kind"] == "free_text" for q in questions)
        body = {"answers": [a for a in evaluation_answers_for(
            task.model_copy(update={"waiting": task.waiting.model_copy(update={"prompt": {
                **task.waiting.prompt, "questions": [q for q in questions if q["kind"] != "free_text"]}})}))
            ["answers"]] + [{"question_id": "ft_1", "answer": answer}]}
        done = await container.task_service.submit_answers(task.task_id, body)
        assert done.status == TaskStatus.COMPLETED, done.errors
        assert llm.calls[GRADER] == 1  # only the free-text answer reached the semantic grader
        art = container.artifacts.find(task.task_id, "learner_evaluation")
        report = LearnerEvaluationReport.model_validate_json(container.artifacts.read(art.artifact_id))
        grades = {g.question_id: g for g in report.grades}
        assert grades["ft_1"].outcome == outcome and grades["ft_1"].grader_type == "SEMANTIC"
        assert all(g.grader_type in ("EXACT", "RULE") for q, g in grades.items() if q != "ft_1")
        evaluation = next(e for e in report.evaluations if e.question_id == "ft_1")
        assert evaluation.correct == (outcome == "CORRECT")
        types = {a.type.value for a in container.task_service.artifacts(task.task_id)}
        assert {"ASSESSMENT_ITEM", "ASSESSMENT_GRADE", "ASSESSMENT_FEEDBACK"} <= types
        evidence = [e for e in container.memory.evidence(task.learner_id) if e.metadata.get("question_id") == "ft_1"
                    or e.source_ref.endswith("ft_1")]
        assert bool(evidence) == counted
        # resubmitting is refused; the stored grades are what the evaluation used
        assert done.result.score is not None
    finally:
        container.close()


async def test_the_evaluator_cannot_overrule_the_grades(settings) -> None:
    flipped = []

    def contrarian(request) -> dict:
        step = evaluate_learner(request)
        if EvaluationInput.model_validate(request.input_payload).stage == "evaluate" and not flipped:
            flipped.append(True)  # first answer only: a model that regrades is rejected, then retried
            first = step["result"]["evaluations"][0]
            first["correct"] = not first["correct"]
        return step

    llm = MockLLMProvider({**default_responders(), "learner_evaluation": contrarian})
    container = build_container(settings, llm_providers={"mock": llm})
    try:
        _, task = await start_evaluation(container)
        calls = llm.calls["learner_evaluation"]
        done = await container.task_service.submit_answers(task.task_id, evaluation_answers_for(task))
        assert done.status == TaskStatus.COMPLETED, done.errors
        assert flipped and llm.calls["learner_evaluation"] == calls + 2
        art = container.artifacts.find(task.task_id, "learner_evaluation")
        report = LearnerEvaluationReport.model_validate_json(container.artifacts.read(art.artifact_id))
        grades = {g.question_id: g for g in report.grades}
        assert all(e.correct == (grades[e.question_id].outcome == "CORRECT") for e in report.evaluations)
    finally:
        container.close()


def test_the_sql_store_numbers_attempts_and_refuses_duplicates(teaching_env: Env) -> None:
    from tests.unit.test_assessment import attempt, short_item

    repo = teaching_env.container.assessment_service.repository
    assert repo.save_item(short_item()) and not repo.save_item(short_item())
    with pytest.raises(ItemConflict):
        repo.save_item(short_item(expected_answer="hablaste"))
    assert [repo.add_attempt(*attempt(f"s{i}")).attempt_number for i in range(3)] == [1, 2, 3]
    with pytest.raises(AttemptExists):
        repo.add_attempt(*attempt("s1", "other"))
    assert repo.attempt("s1").learner_answer == "x" and repo.grade("g-s1").learner_answer == "x"
    done = repo.complete("s0", None, repo.attempt("s0").submitted_at)
    assert done.completed_at is not None
    assert [a.attempt_id for a in repo.attempts("conj", "l1")] == ["s0", "s1", "s2"]
