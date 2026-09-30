"""Evaluation schemas, the evaluation agent's checks, and learner memory's evaluation update."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.evaluation import (
    AssessmentPlan,
    AssessmentQuestion,
    AssessmentResponse,
    ConceptOutcome,
    EvaluationOutcome,
    EvaluationStep,
    LearnerEvaluationResult,
    LearningRecommendation,
)
from app.schemas.learner import AnswerEvaluation
from app.schemas.lesson import ConceptRef, LearnerAnswer
from tests.unit.helpers import scope, service


def question(qid: str, cid: str = "c1", **kw) -> AssessmentQuestion:
    data = dict(question_id=qid, concept_id=cid, objective="o", kind="short_answer", prompt="p", difficulty=0.5,
                expected_answer="x")
    data.update(kw)
    return AssessmentQuestion(**data)


def recommendation(**kw) -> LearningRecommendation:
    data = dict(action="review", focus_concepts=["c1"], rationale="r", suggested_request="Review c1.")
    data.update(kw)
    return LearningRecommendation(**data)


def evaluation(qid: str, cid: str, correct: bool) -> AnswerEvaluation:
    return AnswerEvaluation(question_id=qid, concept_id=cid, answer="a", expected="x", correct=correct,
                            difficulty=0.5, feedback="")


def test_assessment_plan_validation() -> None:
    plan = AssessmentPlan(title="t", level="A2", objectives=["o"], questions=[question("q1"), question("q2")],
                          total_points=2, rationale="r")
    assert len(plan.questions) == 2
    with pytest.raises(ValidationError, match="unique"):
        AssessmentPlan(title="t", level="A2", objectives=["o"], questions=[question("q1"), question("q1")],
                       total_points=2, rationale="r")
    with pytest.raises(ValidationError, match="total_points"):
        AssessmentPlan(title="t", level="A2", objectives=["o"], questions=[question("q1")], total_points=3,
                       rationale="r")
    with pytest.raises(ValidationError, match="one of the choices"):
        question("q1", kind="multiple_choice", choices=["a", "b"], expected_answer="c")


def test_response_rejects_duplicate_answers() -> None:
    with pytest.raises(ValidationError, match="only once"):
        AssessmentResponse(answers=[LearnerAnswer(question_id="q1", answer="a"),
                                    LearnerAnswer(question_id="q1", answer="b")])
    with pytest.raises(ValidationError):
        AssessmentResponse(answers=[])


def test_result_status_lists_must_match_concepts() -> None:
    concepts = [ConceptOutcome(concept_id="c1", name="C1", questions=1, points_earned=0, points_possible=1, score=0,
                               status="gap")]
    ok = LearnerEvaluationResult(score=0, points_earned=0, points_possible=1, evaluations=[evaluation("q1", "c1", False)],
                                 concepts=concepts, mastered=[], partial=[], gaps=["c1"],
                                 recommendation=recommendation(action="reteach"))
    assert ok.gaps == ["c1"]
    with pytest.raises(ValidationError, match="must match"):
        LearnerEvaluationResult(score=0, points_earned=0, points_possible=1,
                                evaluations=[evaluation("q1", "c1", False)], concepts=concepts, mastered=["c1"],
                                partial=[], gaps=[], recommendation=recommendation())


def test_step_carries_exactly_the_stage_output() -> None:
    with pytest.raises(ValidationError, match="assessment"):
        EvaluationStep(stage="assess")
    with pytest.raises(ValidationError, match="result"):
        EvaluationStep(stage="evaluate", assessment=AssessmentPlan(
            title="t", level="A2", objectives=["o"], questions=[question("q1")], total_points=1, rationale="r"))


def test_record_evaluation_updates_mastery_once() -> None:
    memory, _ = service()
    run_scope, seen = scope("task_eval")
    outcome = EvaluationOutcome(
        task_id="task_eval", learner_id="l1", lesson_task_id="task_lesson", subject="math", framework_id="mastery",
        concepts=[ConceptRef(concept_id="fractions", name="Fractions"), ConceptRef(concept_id="decimals", name="Decimals")],
        evaluations=[evaluation("q1", "fractions", True), evaluation("q2", "fractions", True),
                     evaluation("q3", "decimals", False)],
    )
    update = memory.record_evaluation(outcome, run_scope)
    changes = {c.concept_id: c for c in update.changes}
    assert changes["fractions"].after > changes["fractions"].before
    assert changes["decimals"].after < changes["decimals"].before
    assert "missed on assessment" in changes["decimals"].reason

    profile = memory.get_or_create("l1")
    assert [a.kind for a in profile.assessments] == ["lesson_evaluation"]
    assert [m.question_id for m in profile.mistakes] == ["q3"]
    assert [e.type for e in seen] == ["learner.mastery_updated", "learner.updated"]

    again = memory.record_evaluation(outcome, run_scope)  # a resumed task re-running the node changes nothing
    assert again.changes == update.changes
    assert len(memory.get_or_create("l1").assessments) == 1
    assert memory.get_or_create("l1").concepts["fractions"].mastery == changes["fractions"].after
    assert len(seen) == 2
