"""Post-lesson evaluation: assessment -> WAITING -> answers -> grading -> memory -> recommendation -> artifact."""

from __future__ import annotations

import pytest

from app.agents.base import OutputRejected
from app.agents.evaluator.agent import LearnerEvaluationAgent
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders, evaluate_learner
from app.runtime.orchestrator.orchestrator import InvalidInput
from app.runtime.tasks.state_machine import InvalidTransition
from app.schemas.artifact import ArtifactType
from app.schemas.evaluation import AssessmentPlan, AssessmentQuestion, EvaluationInput, EvaluationStep, LearnerEvaluationReport
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import EVALUATION_ANSWERS, add_demo_learner, evaluation_answers_for, run_lesson, start_evaluation

OPINIONS = "es.football.opinions"
PRETERITE = "es.football.preterite_match"


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing in the runtime catches it."""


def crash_after(node_id: str):
    def observer(task, finished: str) -> None:
        if finished == node_id and task.plan.workflow_id == "lesson_evaluation":
            raise SimulatedCrash(node_id)
    return observer


def answers_key(opinions: tuple[str, str], preterite: tuple[str, str]) -> dict:
    """Answers per concept as (short_answer, multiple_choice)."""
    return {"answers": {cid: {"short_answer": sa, "multiple_choice": mc}
                        for cid, (sa, mc) in ((OPINIONS, opinions), (PRETERITE, preterite))}}


def report(container, task) -> LearnerEvaluationReport:
    artifact = container.artifacts.find(task.task_id, "learner_evaluation")
    assert artifact is not None
    return LearnerEvaluationReport.model_validate_json(container.artifacts.read(artifact.artifact_id))


async def test_assessment_is_generated_and_task_waits(container) -> None:
    lesson, task = await start_evaluation(container)
    assert task.status == TaskStatus.WAITING
    assert task.waiting.node_id == "answers" and task.waiting.kind == "assessment_answers"
    sheet = task.waiting.prompt
    assert {q["concept_id"] for q in sheet["questions"]} == {OPINIONS, PRETERITE}
    assert all("expected_answer" not in q for q in sheet["questions"])  # answer keys never reach the learner
    assert task.plan.inputs == {"lesson_task_id": lesson.task_id}
    assert task.workflow.node_states["evaluate"].status.value == "PENDING"

    types = [e.type for e in container.task_service.events(task.task_id)]
    assert types.index("assessment.created") < types.index("assessment.waiting") < types.index("task.waiting")
    assert "evaluation.started" not in types
    assert container.task_service.artifacts(task.task_id) == []


async def test_question_count_follows_learner_mastery(container, mock_llm) -> None:
    await start_evaluation(container)
    request = next(r for r in mock_llm.requests if r.agent_id == "learner_evaluation")
    weak = EvaluationInput.model_validate(request.input_payload)
    assert len(evaluate_learner(request)["assessment"]["questions"]) == 4  # both concepts weak: 2 questions each

    strong = weak.model_copy(deep=True)
    for c in strong.snapshot.concept_mastery:
        c.mastery = 0.9
    fewer = EvaluationStep.model_validate(evaluate_learner(request.model_copy(
        update={"input_payload": strong.model_dump(mode="json")})))
    assert [q.kind for q in fewer.assessment.questions] == ["short_answer", "short_answer"]


async def test_answers_complete_the_evaluation(container) -> None:
    lesson, task = await start_evaluation(container)
    before = container.learner_service.get(task.learner_id).concepts
    done = await container.task_service.submit_answers(task.task_id, evaluation_answers_for(task))
    assert done.status == TaskStatus.COMPLETED, done.errors
    assert done.waiting is None

    # Grading, remaining gaps and the recommendation.
    result = done.result
    assert result.score == 0.5
    assert result.remaining_gaps == [OPINIONS]
    assert result.recommendation.action == "reteach"
    assert result.recommendation.focus_concepts == [OPINIONS]
    assert "reteaches" in result.recommendation.suggested_request

    # Mastery went through learner memory.
    after = container.learner_service.get(task.learner_id).concepts
    changes = {c.concept_id: c for c in result.mastery_changes}
    assert changes[OPINIONS].after < changes[OPINIONS].before == pytest.approx(before[OPINIONS].mastery)
    assert changes[PRETERITE].after > changes[PRETERITE].before
    assert after[PRETERITE].mastery == changes[PRETERITE].after
    progress = container.learner_service.progress(task.learner_id)
    assert progress.assessments_taken == 2  # the diagnostic and the evaluation

    # RUNNING -> WAITING -> RUNNING -> COMPLETED, with every evaluation event, in order.
    types = [e.type for e in container.task_service.events(task.task_id)]
    status_events = [t for t in types if t in ("task.started", "task.waiting", "task.resumed", "task.completed")]
    assert status_events == ["task.started", "task.waiting", "task.resumed", "task.completed"]
    expected = ["assessment.created", "assessment.waiting", "assessment.submitted", "evaluation.started",
                "evaluation.completed", "recommendation.created", "learner.mastery_updated"]
    assert [t for t in types if t in expected] == expected
    mastery_event = next(e for e in container.task_service.events(task.task_id) if e.type == "learner.mastery_updated")
    assert {c["concept_id"] for c in mastery_event.data["changes"]} == {OPINIONS, PRETERITE}

    # The lesson task is untouched by its evaluation.
    assert container.task_service.get(lesson.task_id).status == TaskStatus.COMPLETED


async def test_evaluation_artifact(container) -> None:
    lesson, task = await start_evaluation(container)
    done = await container.task_service.submit_answers(task.task_id, evaluation_answers_for(task))
    [summary] = done.result.artifacts
    assert summary.type == ArtifactType.LEARNER_EVALUATION and summary.name == "learner_evaluation"
    lesson_artifact = container.artifacts.find(lesson.task_id, "lesson")
    assert summary.parent_ids == [lesson_artifact.artifact_id]

    r = report(container, done)
    assert (r.task_id, r.learner_id) == (done.task_id, done.learner_id)
    assert r.lesson.task_id == lesson.task_id and r.lesson.artifact_id == lesson_artifact.artifact_id
    assert len(r.questions) == len(r.answers) == len(r.evaluations) == 4
    assert r.score == 0.5 and r.points_earned == 2 and r.points_possible == 4
    assert {c.concept_id for c in r.mastery_changes} == {OPINIONS, PRETERITE}
    assert r.remaining_gaps == [OPINIONS] and r.mastered == [PRETERITE]
    assert r.recommendation == done.result.recommendation
    assert r.assessment_created_at <= r.answers_submitted_at <= r.evaluated_at


@pytest.mark.parametrize(("key", "action", "partial", "gaps"), [
    (answers_key(("gusta", "Me gustan los partidos de fútbol."),
                 ("ganamos el partido", "El equipo ganó el partido tres a uno.")), "review", [OPINIONS], []),
    (answers_key(("gustan", "Me gustan los partidos de fútbol."),
                 ("ganamos el partido", "El equipo ganó el partido tres a uno.")), "advance", [], []),
    (answers_key(("gusta", "El equipo ganó el partido tres a uno."),
                 ("ganó", "Me gustan los partidos de fútbol.")), "reteach", [], [OPINIONS, PRETERITE]),
])
async def test_gaps_and_recommendations(container, key, action, partial, gaps) -> None:
    _, task = await start_evaluation(container)
    done = await container.task_service.submit_answers(task.task_id, evaluation_answers_for(task, key))
    assert done.status == TaskStatus.COMPLETED, done.errors
    r = report(container, done)
    assert (r.recommendation.action, r.partial, sorted(r.remaining_gaps)) == (action, partial, gaps)
    if action == "advance":
        assert r.score == 1 and OPINIONS not in r.recommendation.focus_concepts


@pytest.mark.parametrize(("body", "message"), [
    ({"answers": [{"question_id": "nope", "answer": "x"}]}, "unknown questions"),
    ({"answers": [{"question_id": f"sa_{OPINIONS}", "answer": "gustan"}]}, "missing"),
    ({"answers": []}, "at least 1"),
    ({"answers": "gustan"}, "list"),
])
async def test_invalid_answers_are_rejected_and_task_keeps_waiting(container, body, message) -> None:
    _, task = await start_evaluation(container)
    with pytest.raises(InvalidInput, match=message):
        await container.task_service.submit_answers(task.task_id, body)
    still = container.task_service.get(task.task_id)
    assert still.status == TaskStatus.WAITING and still.waiting.node_id == "answers"
    done = await container.task_service.submit_answers(task.task_id, evaluation_answers_for(task))
    assert done.status == TaskStatus.COMPLETED


async def test_multiple_choice_answer_must_be_a_choice(container) -> None:
    _, task = await start_evaluation(container)
    body = evaluation_answers_for(task)
    body["answers"] = [a | {"answer": "something else"} if a["question_id"].startswith("mc_") else a
                       for a in body["answers"]]
    with pytest.raises(InvalidInput, match="one of its choices"):
        await container.task_service.submit_answers(task.task_id, body)


async def test_only_completed_lessons_can_be_evaluated(container) -> None:
    learner_id = add_demo_learner(container)
    waiting = await container.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                          learner_id=learner_id, user_id="u1")
    assert waiting.status == TaskStatus.WAITING
    with pytest.raises(InvalidTransition, match="not a completed lesson"):
        await container.task_service.start_evaluation(waiting.task_id, user_id="u1")
    _, evaluation = await start_evaluation(container)
    with pytest.raises(InvalidTransition):
        await container.task_service.start_evaluation(evaluation.task_id, user_id="u1")  # not a lesson
    with pytest.raises(InvalidTransition, match="not waiting"):
        await container.task_service.submit_answers(
            (await run_lesson(container)).task_id, evaluation_answers_for(evaluation))


async def test_answers_arrive_in_a_new_process(settings) -> None:
    first = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    _, task = await start_evaluation(first)
    body = evaluation_answers_for(task)
    first.close()

    llm = MockLLMProvider(default_responders())
    second = build_container(settings, llm_providers={"mock": llm})
    waiting = second.task_service.get(task.task_id)
    assert waiting.status == TaskStatus.WAITING
    done = await second.task_service.submit_answers(task.task_id, body)
    assert done.status == TaskStatus.COMPLETED, done.errors
    assert llm.calls["learner_evaluation"] == 1  # only grading; the assessment was not regenerated
    assert done.result.remaining_gaps == [OPINIONS]
    second.close()


async def test_crash_after_grading_resumes_without_regrading(settings) -> None:
    first = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())},
                            observers=[crash_after("evaluate")])
    _, task = await start_evaluation(first)
    with pytest.raises(SimulatedCrash):
        await first.task_service.submit_answers(task.task_id, evaluation_answers_for(task))
    first.close()

    llm = MockLLMProvider(default_responders())
    second = build_container(settings, llm_providers={"mock": llm})
    crashed = second.task_service.get(task.task_id)
    assert crashed.status == TaskStatus.RUNNING
    assert crashed.workflow.node_states["evaluate"].status.value == "COMPLETED"
    assert crashed.workflow.node_states["update_mastery"].status.value == "PENDING"

    done = await second.task_service.resume(task.task_id)
    assert done.status == TaskStatus.COMPLETED, done.errors
    assert llm.calls["learner_evaluation"] == 0
    events = second.task_service.events(task.task_id)
    assert sum(1 for e in events if e.type == "node.started" and e.node_id == "evaluate") == 1
    assert sum(1 for e in events if e.type == "learner.mastery_updated") == 1
    profile = second.learner_service.get(task.learner_id)
    assert sum(1 for a in profile.assessments if a.kind == "lesson_evaluation") == 1
    second.close()


async def test_failed_evaluation_is_inspectable_and_resumable(container, mock_llm) -> None:
    _, task = await start_evaluation(container)
    # Enough broken output to exhaust the agent's correction attempts on every node attempt (3 x 2).
    mock_llm.inject("learner_evaluation", *["not json"] * 6)
    failed = await container.task_service.submit_answers(task.task_id, evaluation_answers_for(task))
    assert failed.status == TaskStatus.FAILED
    assert failed.errors and failed.errors[-1].node_id == "evaluate"
    assert "no valid output" in failed.errors[-1].message
    assert failed.workflow.node_states["answers"].status.value == "COMPLETED"  # the learner's answers are kept
    assert container.task_service.get(task.task_id).status == TaskStatus.FAILED
    assert "task.failed" in [e.type for e in container.task_service.events(task.task_id)]
    assert container.learner_service.progress(task.learner_id).assessments_taken == 1  # memory untouched

    resumed = await container.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    assert resumed.result.remaining_gaps == [OPINIONS]


async def test_agent_rejects_assessments_that_skip_taught_concepts(container, mock_llm) -> None:
    await start_evaluation(container)
    request = next(r for r in mock_llm.requests if r.agent_id == "learner_evaluation")
    source = EvaluationInput.model_validate(request.input_payload)
    agent = LearnerEvaluationAgent()

    def plan(*concepts: str) -> EvaluationStep:
        questions = [AssessmentQuestion(question_id=f"q{i}", concept_id=c, objective="o", kind="short_answer",
                                        prompt="p", difficulty=0.5, expected_answer="x") for i, c in enumerate(concepts)]
        return EvaluationStep(stage="assess", assessment=AssessmentPlan(
            title="t", level="A2", objectives=["o"], questions=questions, total_points=len(questions), rationale="r"))

    agent.check(plan(OPINIONS, PRETERITE), source)
    with pytest.raises(OutputRejected, match="missing"):
        agent.check(plan(OPINIONS), source)
    with pytest.raises(OutputRejected, match="did not teach"):
        agent.check(plan(OPINIONS, PRETERITE, "es.food.menu"), source)


def test_fixture_answers_cover_both_question_kinds() -> None:
    assert set(EVALUATION_ANSWERS["answers"]) == {OPINIONS, PRETERITE}
