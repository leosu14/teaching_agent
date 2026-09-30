"""Post-lesson learner evaluation: assessment, responses, results and the next-learning recommendation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import Schema
from app.schemas.learner import AnswerEvaluation, LearnerSnapshot, MasteryChange
from app.schemas.lesson import ConceptRef, LearnerAnswer, LessonContent, LessonPlan, LessonRequest

ConceptStatus = Literal["mastered", "partial", "gap"]


class AssessmentQuestion(Schema):
    question_id: str
    concept_id: str
    objective: str
    kind: Literal["short_answer", "multiple_choice", "translation", "problem"]
    prompt: str
    choices: list[str] = Field(default_factory=list)
    difficulty: float = Field(ge=0, le=1)
    expected_answer: str = Field(min_length=1)
    accepted_answers: list[str] = Field(default_factory=list)
    points: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _choices_contain_answer(self) -> AssessmentQuestion:
        if self.kind == "multiple_choice" and self.expected_answer not in self.choices:
            raise ValueError(f"question {self.question_id}: expected answer must be one of the choices")
        return self


class AssessmentPlan(Schema):
    title: str
    level: str
    objectives: list[str] = Field(min_length=1)
    questions: list[AssessmentQuestion] = Field(min_length=1)
    total_points: float = Field(gt=0)
    rationale: str

    @model_validator(mode="after")
    def _consistent(self) -> AssessmentPlan:
        ids = [q.question_id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        if abs(sum(q.points for q in self.questions) - self.total_points) > 1e-9:
            raise ValueError("total_points must equal the sum of question points")
        return self


class SheetQuestion(Schema):
    question_id: str
    concept_id: str
    kind: str
    prompt: str
    choices: list[str] = Field(default_factory=list)


class AssessmentSheet(Schema):
    """What the learner sees while the task is WAITING. Answer keys are never included."""

    title: str
    questions: list[SheetQuestion]


class AssessmentResponse(Schema):
    answers: list[LearnerAnswer] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> AssessmentResponse:
        ids = [a.question_id for a in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("each question may be answered only once")
        return self


class ConceptOutcome(Schema):
    concept_id: str
    name: str
    questions: int = Field(ge=1)
    points_earned: float = Field(ge=0)
    points_possible: float = Field(gt=0)
    score: float = Field(ge=0, le=1)
    status: ConceptStatus


class LearningRecommendation(Schema):
    action: Literal["advance", "review", "reteach"]
    focus_concepts: list[str]
    review_concepts: list[str] = Field(default_factory=list)
    rationale: str
    suggested_request: str = Field(min_length=1)


class LearnerEvaluationResult(Schema):
    score: float = Field(ge=0, le=1)
    points_earned: float = Field(ge=0)
    points_possible: float = Field(gt=0)
    evaluations: list[AnswerEvaluation] = Field(min_length=1)
    concepts: list[ConceptOutcome] = Field(min_length=1)
    mastered: list[str]
    partial: list[str]
    gaps: list[str]
    recommendation: LearningRecommendation

    @model_validator(mode="after")
    def _consistent(self) -> LearnerEvaluationResult:
        by_status: dict[str, list[str]] = {"mastered": [], "partial": [], "gap": []}
        for c in self.concepts:
            by_status[c.status].append(c.concept_id)
        if (sorted(self.mastered), sorted(self.partial), sorted(self.gaps)) != (
            sorted(by_status["mastered"]), sorted(by_status["partial"]), sorted(by_status["gap"])
        ):
            raise ValueError("mastered/partial/gaps must match the concept outcomes")
        if self.points_earned > self.points_possible:
            raise ValueError("points_earned cannot exceed points_possible")
        return self


class EvaluationInput(Schema):
    stage: Literal["assess", "evaluate"]
    request: LessonRequest
    lesson: LessonContent
    plan: LessonPlan
    snapshot: LearnerSnapshot
    assessment: AssessmentPlan | None = None
    response: AssessmentResponse | None = None

    @model_validator(mode="after")
    def _stage_inputs(self) -> EvaluationInput:
        if self.stage == "evaluate" and (self.assessment is None or self.response is None):
            raise ValueError("the evaluate stage needs the assessment and the learner's response")
        return self


class EvaluationStep(Schema):
    stage: Literal["assess", "evaluate"]
    assessment: AssessmentPlan | None = None
    result: LearnerEvaluationResult | None = None

    @model_validator(mode="after")
    def _one_output(self) -> EvaluationStep:
        if self.stage == "assess" and (self.assessment is None or self.result is not None):
            raise ValueError("stage 'assess' must return an assessment and no result")
        if self.stage == "evaluate" and (self.result is None or self.assessment is not None):
            raise ValueError("stage 'evaluate' must return a result and no assessment")
        return self


class EvaluationOutcome(Schema):
    """What learner memory records from an evaluation."""

    task_id: str
    learner_id: str
    lesson_task_id: str
    subject: str
    framework_id: str
    concepts: list[ConceptRef]
    evaluations: list[AnswerEvaluation] = Field(min_length=1)


class LessonReference(Schema):
    task_id: str
    artifact_id: str
    title: str


class LearnerEvaluationReport(Schema):
    """Content of the LEARNER_EVALUATION artifact."""

    task_id: str
    learner_id: str
    lesson: LessonReference
    questions: list[AssessmentQuestion]
    answers: list[LearnerAnswer]
    score: float
    points_earned: float
    points_possible: float
    evaluations: list[AnswerEvaluation]
    concepts: list[ConceptOutcome]
    mastery_changes: list[MasteryChange]
    mastered: list[str]
    partial: list[str]
    remaining_gaps: list[str]
    recommendation: LearningRecommendation
    assessment_created_at: datetime | None
    answers_submitted_at: datetime | None
    evaluated_at: datetime | None
