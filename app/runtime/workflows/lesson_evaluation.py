"""Post-lesson evaluation workflow template.

completed lesson -> learner snapshot -> assessment -> WAITING for answers -> evaluation
-> learning evidence and deterministic mastery update -> updated learner model -> next-learning recommendation
(the pedagogical engine re-plans from the new state) -> feedback -> evaluation artifacts.
"""

from __future__ import annotations

from app.runtime.orchestrator.planner import ExpectedCall, WorkflowTemplate
from app.runtime.workflow.engine import WorkflowDefinition
from app.runtime.workflow.nodes import AgentNode, HumanApprovalNode, StateView, ToolNode, TransformNode
from app.schemas.artifact import ArtifactBatch, ArtifactDraft, ArtifactType, StoredArtifacts
from app.schemas.evaluation import (
    AssessmentPlan,
    AssessmentResponse,
    AssessmentSheet,
    EvaluationInput,
    EvaluationOutcome,
    EvaluationStep,
    LearnerEvaluationReport,
    LearnerEvaluationResult,
    LessonReference,
    SheetQuestion,
)
from app.schemas.events import EventType
from app.schemas.learner import LearnerSnapshot, LearningGoal, MasteryUpdate
from app.schemas.lesson import ConceptRef, LessonContent, LessonPlan, LessonRequest
from app.schemas.pedagogy import (
    ConceptSet,
    EvaluationFeedback,
    FeedbackRequest,
    LearnerModel,
    NextLearningRecommendation,
    PedagogicalPlan,
    RecommendationRequest,
)
from app.schemas.task import ArtifactSummary, TaskResult
from app.tools.artifacts.tools import ReadArtifactsOutput

WORKFLOW_ID = "lesson_evaluation"
LESSON_TASK = "lesson_task_id"


def _request(v: StateView) -> LessonRequest:
    assert v.task.plan is not None
    return v.task.plan.lesson_request


def _lesson_task(v: StateView) -> str:
    assert v.task.plan is not None
    return v.task.plan.inputs[LESSON_TASK]


def _loaded(v: StateView) -> ReadArtifactsOutput:
    return v.output("load_lesson", ReadArtifactsOutput)


def _lesson(v: StateView) -> LessonContent:
    return LessonContent.model_validate_json(_loaded(v).content("lesson"))


def _plan(v: StateView) -> LessonPlan:
    return LessonPlan.model_validate_json(_loaded(v).content("lesson_plan"))


def _pedagogical_plan(v: StateView) -> PedagogicalPlan:
    return PedagogicalPlan.model_validate_json(_loaded(v).content("pedagogical_plan"))


def _concepts(v: StateView) -> ConceptSet:
    return v.output("knowledge_graph", ConceptSet)


def _model(v: StateView) -> LearnerModel:
    return v.output("learner_model", LearnerModel)


def _recommendation(v: StateView) -> NextLearningRecommendation:
    return v.output("next_recommendation", NextLearningRecommendation)


def _assessed(v: StateView) -> list[str]:
    return list(dict.fromkeys(e.concept_id for e in _result(v).evaluations))


def _assessment(v: StateView) -> AssessmentPlan:
    step = v.output("assess", EvaluationStep)
    assert step.assessment is not None
    return step.assessment


def _result(v: StateView) -> LearnerEvaluationResult:
    step = v.output("evaluate", EvaluationStep)
    assert step.result is not None
    return step.result


def _evaluation_input(stage: str):
    def build(v: StateView) -> EvaluationInput:
        return EvaluationInput(
            stage=stage, request=_request(v), lesson=_lesson(v), plan=_plan(v),
            snapshot=v.output("learner_snapshot", LearnerSnapshot).for_provider(),
            assessment=_assessment(v) if stage == "evaluate" else None,
            response=v.output("answers", AssessmentResponse) if stage == "evaluate" else None,
        )
    return build


def _sheet(v: StateView) -> AssessmentSheet:
    plan = _assessment(v)
    return AssessmentSheet(title=plan.title, questions=[
        SheetQuestion(question_id=q.question_id, concept_id=q.concept_id, kind=q.kind, prompt=q.prompt,
                      choices=q.choices)
        for q in plan.questions
    ])


def _validate_answers(v: StateView, response) -> None:
    questions = {q.question_id: q for q in _assessment(v).questions}
    given = {a.question_id: a.answer for a in response.answers}
    unknown = sorted(set(given) - set(questions))
    if unknown:
        raise ValueError(f"answers reference unknown questions {unknown}")
    missing = sorted(set(questions) - set(given))
    if missing:
        raise ValueError(f"every question needs an answer; missing {missing}")
    for qid, answer in given.items():
        q = questions[qid]
        if q.kind == "multiple_choice" and answer not in q.choices:
            raise ValueError(f"answer to {qid} must be one of its choices")


def _outcome(v: StateView) -> EvaluationOutcome:
    req = _request(v)
    return EvaluationOutcome(
        task_id=v.task.task_id, learner_id=v.task.learner_id, lesson_task_id=_lesson_task(v),
        subject=req.subject, framework_id=req.framework_id,
        concepts=[ConceptRef(concept_id=s.concept_id, name=s.heading) for s in _lesson(v).sections],
        evaluations=_result(v).evaluations,
    )


def _package(v: StateView) -> ArtifactBatch:
    lesson_artifact = _loaded(v).artifact("lesson")
    result = _result(v)
    update = v.output("update_mastery", MasteryUpdate)
    report = LearnerEvaluationReport(
        task_id=v.task.task_id, learner_id=v.task.learner_id,
        lesson=LessonReference(task_id=_lesson_task(v), artifact_id=lesson_artifact.artifact_id,
                               title=_lesson(v).title),
        questions=_assessment(v).questions, answers=v.output("answers", AssessmentResponse).answers,
        score=result.score, points_earned=result.points_earned, points_possible=result.points_possible,
        evaluations=result.evaluations, concepts=result.concepts, mastery_changes=update.changes,
        mastered=result.mastered, partial=result.partial, remaining_gaps=result.gaps,
        recommendation=result.recommendation, feedback=v.output("feedback", EvaluationFeedback),
        next_recommendation=_recommendation(v),
        assessment_created_at=v.finished_at("assess"), answers_submitted_at=v.finished_at("answers"),
        evaluated_at=v.finished_at("evaluate"),
    )
    model, rec = _model(v), _recommendation(v)
    drafts = [ArtifactDraft(
        key="evaluation", name="learner_evaluation", type=ArtifactType.LEARNER_EVALUATION,
        media_type="application/json", content=report.model_dump_json(indent=2),
        parent_ids=[lesson_artifact.artifact_id],
        metadata={"score": result.score, "remaining_gaps": result.gaps, "action": result.recommendation.action},
    )]
    # Evaluation -> LearningEvidence -> updated LearnerModel -> NextLearningRecommendation
    if update.evidence:
        drafts.append(ArtifactDraft(
            key="learning_evidence", name="learning_evidence", type=ArtifactType.LEARNING_EVIDENCE,
            media_type="application/json",
            content="[\n" + ",\n".join(e.model_dump_json(indent=2) for e in update.evidence) + "\n]",
            parent_keys=["evaluation"],
            metadata={"source": "evaluation", "items": len(update.evidence),
                      "concepts": sorted({e.concept_id for e in update.evidence})}))
    drafts += [
        ArtifactDraft(key="learner_model", name="learner_model", type=ArtifactType.LEARNER_MODEL,
                      media_type="application/json", content=model.model_dump_json(indent=2),
                      parent_keys=["learning_evidence" if update.evidence else "evaluation"],
                      metadata={"domain": model.domain, "mastered": model.mastered_concepts,
                                "developing": model.developing_concepts, "weak": model.weak_concepts,
                                "evidence": model.evidence_count}),
        ArtifactDraft(key="next_recommendation", name="next_recommendation",
                      type=ArtifactType.LEARNING_RECOMMENDATION, media_type="application/json",
                      content=rec.model_dump_json(indent=2), parent_keys=["learner_model"],
                      metadata={"concepts": rec.recommended_concepts, "prerequisite_review": rec.prerequisite_review,
                                "goal_achieved": rec.goal_achieved, "plan_id": rec.plan_id}),
    ]
    return ArtifactBatch(drafts=drafts)


def _summarize(v: StateView) -> TaskResult:
    stored = v.output("store_artifacts", StoredArtifacts)
    update = v.output("update_mastery", MasteryUpdate)
    result = _result(v)
    return TaskResult(
        title=f"Evaluation: {_lesson(v).title}",
        artifacts=[ArtifactSummary(artifact_id=a.artifact_id, type=a.type, name=a.name, version=a.version,
                                   uri=a.uri, parent_ids=a.parent_ids) for a in stored.artifacts],
        mastery_changes=update.changes,
        estimated_level=update.estimated_level,
        score=result.score,
        remaining_gaps=result.gaps,
        recommendation=result.recommendation,
        next_recommendation=_recommendation(v),
        feedback=v.output("feedback", EvaluationFeedback),
    )


def build_evaluation_workflow(request: LessonRequest) -> WorkflowDefinition:
    nodes = (
        ToolNode(id="load_lesson", tool="artifact.read", permissions=frozenset({"artifact:read"}),
                 build_input=lambda v: {"task_id": _lesson_task(v),
                                        "names": ["lesson", "lesson_plan", "pedagogical_plan"]}),
        ToolNode(id="learner_snapshot", tool="learner.snapshot", permissions=frozenset({"learner:read"}),
                 depends_on=("load_lesson",),
                 build_input=lambda v: {"learner_id": v.task.learner_id, "subject": _request(v).subject,
                                        "framework_id": _request(v).framework_id,
                                        "target_level": _request(v).target_level}),
        AgentNode(id="assess", agent="learner_evaluation", depends_on=("learner_snapshot",),
                  build_input=_evaluation_input("assess")),
        HumanApprovalNode(id="answers", depends_on=("assess",), wait_kind="assessment_answers",
                          build_request=_sheet, response_model=AssessmentResponse,
                          validate_input=_validate_answers, wait_event=EventType.ASSESSMENT_WAITING,
                          submitted_event=EventType.ASSESSMENT_SUBMITTED),
        AgentNode(id="evaluate", agent="learner_evaluation", depends_on=("answers",),
                  build_input=_evaluation_input("evaluate")),
        ToolNode(id="update_mastery", tool="learner.record_evaluation", permissions=frozenset({"learner:write"}),
                 depends_on=("evaluate",), build_input=_outcome),
        ToolNode(id="knowledge_graph", tool="knowledge.concepts", permissions=frozenset({"knowledge:read"}),
                 depends_on=("update_mastery",), build_input=lambda v: {"domain": _request(v).subject}),
        ToolNode(id="load_goal", tool="learning_goal.resolve", permissions=frozenset({"learner:read"}),
                 depends_on=("knowledge_graph",),
                 build_input=lambda v: {"learner_id": v.task.learner_id, "domain": _request(v).subject,
                                        "topic": _request(v).topic, "target_level": _request(v).target_level,
                                        "goal_id": _pedagogical_plan(v).goal_id}),
        ToolNode(id="learner_model", tool="learner.model", permissions=frozenset({"learner:read"}),
                 depends_on=("load_goal",),
                 build_input=lambda v: {"learner_id": v.task.learner_id, "domain": _request(v).subject,
                                        "framework_id": _request(v).framework_id,
                                        "target_level": _request(v).target_level,
                                        "concept_ids": [c.concept_id for c in _concepts(v).concepts]}),
        ToolNode(id="next_recommendation", tool="pedagogy.recommend", permissions=frozenset({"learner:read"}),
                 depends_on=("learner_model",),
                 build_input=lambda v: RecommendationRequest(model=_model(v), goal=v.output("load_goal", LearningGoal),
                                                             concepts=_concepts(v).concepts,
                                                             available_minutes=_pedagogical_plan(v).available_minutes)),
        ToolNode(id="feedback", tool="pedagogy.evaluation_feedback", permissions=frozenset({"learner:read"}),
                 depends_on=("next_recommendation",),
                 build_input=lambda v: FeedbackRequest(changes=v.output("update_mastery", MasteryUpdate).changes,
                                                       assessed_concepts=_assessed(v), model=_model(v),
                                                       recommendation=_recommendation(v))),
        TransformNode(id="package_artifacts", depends_on=("feedback",), fn=_package),
        ToolNode(id="store_artifacts", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("package_artifacts",), build_input=lambda v: v.output("package_artifacts", ArtifactBatch)),
    )
    return WorkflowDefinition(id=WORKFLOW_ID, nodes=nodes, summarize=_summarize,
                              description="Assess a completed lesson, grade answers, record evidence, update mastery "
                                          "deterministically and recommend the next lesson from the new state.")


def evaluation_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        id=WORKFLOW_ID,
        description="Post-lesson assessment, grading, memory update and next-learning recommendation.",
        provides=frozenset({"lesson.evaluation", "learner.model", "pedagogy.recommendation"}),
        build=build_evaluation_workflow,
        expected_calls=(
            ExpectedCall("learner_evaluation", 8000, 1500),
            ExpectedCall("learner_evaluation", 9000, 2000),
        ),
    )
