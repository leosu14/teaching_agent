"""The adaptive lesson end to end: placement -> diagnostic evidence -> learner model -> gaps -> pedagogical plan ->
lesson -> evaluation evidence -> mastery update -> a different next recommendation. Plus resume, idempotency,
privacy and failure handling: a failed step never corrupts or invents learner state."""

from __future__ import annotations

import json
from typing import get_args

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.evaluation import AssessmentSheet
from app.schemas.learner import LearnerProfileInput, LearningGoal
from app.schemas.lesson import (
    DiagnosticAnswers,
    DiagnosticQuestionSheet,
    LearnerAnswer,
    LessonContent,
    SectionPurpose,
)
from app.schemas.pedagogy import KnowledgeGapSet, LearnerModel, PedagogicalPlan
from app.schemas.task import TaskStatus
from app.services.container import build_container
from scripts.run_adaptive_demo import FIXTURES, placement_evidence, run_adaptive_demo
from tests.integration.test_review_loop import always_reject

FIXTURE = json.loads((FIXTURES / "learner.json").read_text(encoding="utf-8"))
DIAGNOSTIC = json.loads((FIXTURES / "answers.json").read_text(encoding="utf-8"))
EVALUATION = json.loads((FIXTURES / "evaluation_answers.json").read_text(encoding="utf-8"))
LEARNER = FIXTURE["learner_id"]
OPINIONS, PRETERITE, IMPERFECT = "es.conv.opinions", "es.conv.preterite", "es.conv.imperfect"
STORY, SUBJUNCTIVE = "es.conv.preterite_imperfect", "es.conv.subjunctive_opinion"


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing in the runtime catches it."""


def crash_after(node_id: str):
    def observer(task, finished: str) -> None:
        if finished == node_id:
            raise SimulatedCrash(node_id)
    return observer


@pytest.fixture
def adaptive_settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", corpus_dir=FIXTURES, log_json=False)


def make(settings: Settings, responders=None, observers=()):
    llm = MockLLMProvider(responders or default_responders())
    return build_container(settings, llm_providers={"mock": llm}, observers=observers), llm


async def placed_learner(container) -> LearningGoal:
    container.learner_service.upsert(LEARNER, LearnerProfileInput.model_validate(FIXTURE["profile"]))
    goal = container.learner_service.set_goal(LearningGoal.model_validate({**FIXTURE["goal"], "learner_id": LEARNER}))
    await container.learner_service.record_evidence(LEARNER, "spanish", placement_evidence(LEARNER, FIXTURE["placement"]))
    return goal


def diagnostic_answers(task) -> DiagnosticAnswers:
    sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
    key = DIAGNOSTIC["rounds"][sheet.round_number - 1]
    return DiagnosticAnswers(answers=[LearnerAnswer(question_id=q.question_id, answer=key.get(q.concept_id, ""))
                                      for q in sheet.questions])


def evaluation_answers(task) -> dict:
    sheet = AssessmentSheet.model_validate(task.waiting.prompt)
    return {"answers": [{"question_id": q.question_id,
                         "answer": EVALUATION["answers"].get(q.concept_id, {}).get(q.kind, "")}
                        for q in sheet.questions]}


async def lesson(container):
    task = await container.task_service.create_and_run(request=FIXTURE["request"], learner_id=LEARNER, user_id="u1")
    while task.status == TaskStatus.WAITING:
        task = await container.task_service.submit_assessment(task.task_id, diagnostic_answers(task))
    return task


def content(container, task, name: str) -> bytes:
    artifact = next(a for a in container.task_service.artifacts(task.task_id) if a.name == name)
    return container.artifacts.read(artifact.artifact_id)


def mastery(container) -> dict[str, float]:
    return {cid: c.mastery for cid, c in container.learner_service.get(LEARNER).concepts.items()}


async def test_the_demo_recommendation_changes_with_mastery(adaptive_settings) -> None:
    container, _ = make(adaptive_settings)
    lines: list[str] = []
    try:
        task, changed = await run_adaptive_demo(container, out=lines.append)
    finally:
        container.close()
    assert task.status == TaskStatus.COMPLETED and changed
    text = "\n".join(lines)
    assert "es.conv.opinions                  0.20 -> 0.62" in text
    assert "es.conv.preterite                 0.52 -> 0.77" in text
    assert "es.conv.imperfect                 0.78 -> 0.78" in text  # not taught, not assessed: unchanged


async def test_adaptive_lesson_is_planned_from_the_learner_model(adaptive_settings) -> None:
    container, llm = make(adaptive_settings)
    goal = await placed_learner(container)
    task = await lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors

    # The diagnostic only probed what memory does not already know with confidence.
    last = [r for r in llm.requests if r.agent_id == "knowledge_diagnostic"][-1]
    asked = [i["question"]["concept_id"] for rnd in last.input_payload["rounds"] for i in rnd["items"]]
    assert asked == [STORY, SUBJUNCTIVE, STORY, SUBJUNCTIVE]  # two rounds; the placed concepts were never asked

    model = LearnerModel.model_validate_json(content(container, task, "learner_model"))
    gaps = KnowledgeGapSet.model_validate_json(content(container, task, "knowledge_gaps"))
    plan = PedagogicalPlan.model_validate_json(content(container, task, "pedagogical_plan"))
    lesson_ = LessonContent.model_validate_json(content(container, task, "lesson"))
    assert model.learner_id == gaps.learner_id == plan.learner_id == LEARNER
    assert gaps.goal_id == plan.goal_id == goal.goal_id and plan.gap_set_id == gaps.gap_set_id
    assert model.mastery_of(STORY) > 0 and model.has_evidence(SUBJUNCTIVE)  # diagnostic evidence was recorded
    assert plan.target_concepts == [OPINIONS, PRETERITE]
    assert "Disagreeing with the subjunctive" in plan.rationale  # deferred: opinions are not secure yet

    # The lesson follows the plan: its objectives, a purpose and objective per section, only plan concepts.
    assert [(o.objective_id, o.concept_id) for o in lesson_.objectives] == \
        [(o.objective_id, o.concept_id) for o in plan.lesson_objectives]
    assert {s.concept_id for s in lesson_.sections} <= set(plan.concept_ids())
    assert IMPERFECT not in {s.concept_id for s in lesson_.sections}  # not in the plan: not taught
    for s in lesson_.sections:
        assert s.objective_ids and s.purpose in get_args(SectionPurpose)
    container.close()


async def test_resume_after_the_gap_analysis_does_not_repeat_learner_updates(adaptive_settings) -> None:
    first, _ = make(adaptive_settings, observers=[crash_after("knowledge_gaps")])
    await placed_learner(first)
    task = await first.task_service.create_and_run(request=FIXTURE["request"], learner_id=LEARNER, user_id="u1")
    task = await first.task_service.submit_assessment(task.task_id, diagnostic_answers(task))
    with pytest.raises(SimulatedCrash):
        await first.task_service.submit_assessment(task.task_id, diagnostic_answers(task))
    evidence_before = first.memory.evidence(LEARNER)
    first.close()

    second, _ = make(adaptive_settings)
    crashed = second.task_service.get(task.task_id)
    assert crashed.workflow.node_states["knowledge_gaps"].status.value == "COMPLETED"
    assert crashed.workflow.node_states["pedagogical_plan"].status.value == "PENDING"
    resumed = await second.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    started = [e.node_id for e in second.task_service.events(task.task_id) if e.type == "node.started"]
    assert started.count("record_diagnostic") == 1 and started.count("knowledge_gaps") == 1
    assert started.count("pedagogical_plan") == 1
    # Lesson exposure is not evidence: the evidence is exactly what the diagnostic recorded before the crash.
    assert second.memory.evidence(LEARNER) == evidence_before

    # Re-recording the same diagnostic is a no-op (idempotent per task).
    from app.observability.scope import ExecutionScope, UsageLedger
    from app.schemas.lesson import DiagnosticOutcome, DiagnosticStep
    step = DiagnosticStep.model_validate(resumed.workflow.node_states["diagnostic"].output)
    outcome = DiagnosticOutcome(task_id=task.task_id, learner_id=LEARNER, request=resumed.plan.lesson_request,
                                diagnostic=step.result, concepts=step.concepts)
    again = second.memory.record_diagnostic(outcome, ExecutionScope(events=second.events, usage=UsageLedger()))
    assert second.memory.evidence(LEARNER) == evidence_before and again.evidence
    second.close()


async def test_providers_never_see_learner_identity(adaptive_settings) -> None:
    container, llm = make(adaptive_settings)
    await placed_learner(container)
    task = await lesson(container)
    evaluation = await container.task_service.start_evaluation(task.task_id, user_id="u1")
    done = await container.task_service.submit_answers(evaluation.task_id, evaluation_answers(evaluation))
    assert done.status == TaskStatus.COMPLETED, done.errors
    name = FIXTURE["profile"]["display_name"]
    assert llm.requests
    for request in llm.requests:
        sent = json.dumps(request.input_payload) + request.system + "".join(m.content for m in request.messages)
        assert LEARNER not in sent and name not in sent, request.agent_id
    lesson_text = content(container, task, "lesson").decode("utf-8")
    assert LEARNER not in lesson_text and name not in lesson_text
    container.close()


async def test_a_failed_lesson_records_no_lesson_and_invents_no_mastery(adaptive_settings) -> None:
    container, _ = make(adaptive_settings, responders={**default_responders(), "content_reviewer": always_reject})
    await placed_learner(container)
    placed = mastery(container)
    task = await lesson(container)
    assert task.status == TaskStatus.FAILED and task.errors[-1].node_id == "teach_review"
    after = mastery(container)
    for cid in (OPINIONS, PRETERITE, IMPERFECT):
        assert after[cid] == pytest.approx(placed[cid])  # only the diagnostic's own concepts changed
    profile = container.learner_service.get(LEARNER)
    assert profile.lessons == [] and all(c.exposures == 0 for c in profile.concepts.values())
    sources = {e.source_type for e in container.memory.evidence(LEARNER)}
    assert sources == {"exercise", "manual", "diagnostic"}
    container.close()


async def test_an_llm_failure_in_planning_leaves_state_intact(adaptive_settings) -> None:
    container, llm = make(adaptive_settings)
    await placed_learner(container)
    llm.inject("curriculum_planner", *["not json"] * 6)  # every correction attempt of every node attempt
    task = await lesson(container)
    assert task.status == TaskStatus.FAILED and task.errors[-1].node_id == "plan"
    # The deterministic plan was computed and stored before the model was asked to word it.
    assert task.workflow.node_states["pedagogical_plan"].status.value == "COMPLETED"
    assert not [a for a in container.task_service.artifacts(task.task_id) if a.name == "lesson"]
    assert container.learner_service.get(LEARNER).lessons == []
    container.close()


async def test_a_failed_evaluation_changes_no_mastery(adaptive_settings) -> None:
    container, llm = make(adaptive_settings)
    await placed_learner(container)
    task = await lesson(container)
    evaluation = await container.task_service.start_evaluation(task.task_id, user_id="u1")
    before, evidence_before = mastery(container), container.memory.evidence(LEARNER)
    llm.inject("learner_evaluation", *["not json"] * 6)
    failed = await container.task_service.submit_answers(evaluation.task_id, evaluation_answers(evaluation))
    assert failed.status == TaskStatus.FAILED and failed.errors[-1].node_id == "evaluate"
    assert mastery(container) == before and container.memory.evidence(LEARNER) == evidence_before
    resumed = await container.task_service.resume(evaluation.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    assert resumed.result.next_recommendation is not None and resumed.result.feedback is not None
    assert mastery(container)[OPINIONS] > before[OPINIONS]
    container.close()
