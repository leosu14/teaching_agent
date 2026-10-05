"""Semantic assessment: assessment items, rubrics, attempts and immutable grades.

Grading is layered: deterministic normalisation and matching first, then deterministic rules and rubric scoring, and a
semantic grader (a model) only when the answer is free text that deterministic grading cannot decide. Whatever grades
an answer, code computes the final score from the criterion results, classifies the outcome against configured
thresholds and builds the grade. A model only proposes a candidate (`SemanticGradeCandidate`), which is validated
before anything is stored; it has no field for mastery, objective, goal or curriculum state.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from app.schemas.common import Schema

# --- enums ---------------------------------------------------------------------------------------------------------


class ResponseType(str, Enum):
    SHORT_TEXT = "SHORT_TEXT"  # a word or phrase: deterministic unless the item has a rubric
    FREE_TEXT = "FREE_TEXT"  # a sentence or explanation: graded against a rubric, semantically when needed
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    TRUE_FALSE = "TRUE_FALSE"


class AssessmentOutcome(str, Enum):
    CORRECT = "CORRECT"
    PARTIAL = "PARTIAL"
    INCORRECT = "INCORRECT"
    UNCERTAIN = "UNCERTAIN"  # not decided: never evidence, never a mastery change, the learner may try again


class GraderType(str, Enum):
    EXACT = "EXACT"  # the normalised answer was compared with the expected and acceptable answers (or choices)
    RULE = "RULE"  # a deterministic rule decided: an empty answer, a known wrong answer, an invalid choice
    RUBRIC = "RUBRIC"  # a deterministic rubric (criteria scored from their indicators)
    SEMANTIC = "SEMANTIC"  # a semantic grader proposed criterion scores; code aggregated and classified them


AttemptSource = Literal["api", "teaching_session", "evaluation"]
CLOSED_TYPES = frozenset({ResponseType.MULTIPLE_CHOICE, ResponseType.TRUE_FALSE})


def _hash(model: Schema) -> str:
    return hashlib.sha256(json.dumps(model.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


# --- configuration -------------------------------------------------------------------------------------------------


class NormalizationConfig(Schema):
    """Deterministic, surface-level normalisation. It never changes what an answer means: "no" and "sí" stay apart."""

    unicode_form: Literal["NFC", "NFKC"] = "NFKC"
    case_sensitive: bool = False
    strip_punctuation: bool = True  # "¿Qué pasó?" -> "qué pasó"
    collapse_whitespace: bool = True
    fold_accents: bool = False  # "qué pasó" -> "que paso": only for configured languages (accents can carry meaning)


class AssessmentConfig(Schema):
    """Every threshold of the deterministic grade. A model's own outcome or score never overrides them."""

    correct_threshold: float = Field(default=0.8, gt=0, le=1)  # a rubric's passing_threshold takes precedence
    partial_threshold: float = Field(default=0.4, ge=0, lt=1)
    accept_confidence: float = Field(default=0.85, gt=0, le=1)  # at or above: the semantic grade stands
    min_confidence: float = Field(default=0.6, ge=0, le=1)  # below: UNCERTAIN
    # Between the two: "partial" caps a CORRECT grade at PARTIAL and makes an INCORRECT one UNCERTAIN (not enough
    # confidence to fail the learner); "uncertain" makes every grade in the band UNCERTAIN.
    mid_confidence: Literal["partial", "uncertain"] = "partial"
    contradiction_tolerance: float = Field(default=0.2, ge=0, le=1)  # model score vs its own criteria
    misconception_min_confidence: float = Field(default=0.5, ge=0, le=1)  # weaker candidates are not recorded
    semantic_enabled: bool = True
    accent_insensitive_languages: list[str] = Field(default_factory=list)  # e.g. ["es"]: "que paso" == "qué pasó"
    max_context_passages: int = Field(default=4, ge=0, le=20)  # lesson sections / research findings sent per grade

    @model_validator(mode="after")
    def _ordered(self) -> AssessmentConfig:
        if self.partial_threshold >= self.correct_threshold:
            raise ValueError("partial_threshold must be below correct_threshold")
        if self.min_confidence > self.accept_confidence:
            raise ValueError("min_confidence must not exceed accept_confidence")
        return self

    def normalization(self, language: str) -> NormalizationConfig:
        base = language.split("-")[0].lower()
        return NormalizationConfig(fold_accents=base in {x.split("-")[0].lower()
                                                         for x in self.accent_insensitive_languages})


# --- items and rubrics ---------------------------------------------------------------------------------------------


class KnownError(Schema):
    """A deterministic rule: this answer is wrong, and (optionally) why. Matched after normalisation."""

    answer: str = Field(min_length=1, max_length=500)
    misconception_type: str | None = Field(default=None, max_length=80)
    description: str | None = Field(default=None, max_length=300)


class MisconceptionPattern(Schema):
    """A misconception the item is known to provoke. The semantic grader may only report these or others about the
    item's concepts; `cues` are reference phrasings that show it."""

    type: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    concept_id: str | None = None
    cues: list[str] = Field(default_factory=list)


class AssessmentItem(Schema):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_item_id: str = Field(min_length=1, max_length=128)
    lesson_id: str | None = None  # the LESSON artifact the item assesses
    objective_id: str | None = None
    concept_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=2000)
    expected_answer: str = Field(min_length=1, max_length=1000)  # the expected answer, or its expected meaning
    acceptable_answers: list[str] = Field(default_factory=list)
    response_type: ResponseType
    rubric_id: str | None = None
    difficulty: float = Field(default=0.5, ge=0, le=1)
    language: str = Field(default="en", min_length=2, max_length=16)  # BCP 47 language of the answer
    choices: list[str] = Field(default_factory=list)
    max_score: float = Field(default=1.0, gt=0)
    known_errors: list[KnownError] = Field(default_factory=list)
    misconceptions: list[MisconceptionPattern] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> AssessmentItem:
        if self.response_type == ResponseType.MULTIPLE_CHOICE:
            if len(self.choices) < 2 or len(set(self.choices)) != len(self.choices):
                raise ValueError("a multiple-choice item needs at least two distinct choices")
            if self.expected_answer not in self.choices:
                raise ValueError("the expected answer of a multiple-choice item must be one of its choices")
        elif self.choices:
            raise ValueError("only a multiple-choice item has choices")
        if self.response_type == ResponseType.TRUE_FALSE and self.expected_answer.strip().lower() not in (
                "true", "false"):
            raise ValueError("a true/false item expects 'true' or 'false'")
        return self

    @property
    def semantic(self) -> bool:
        """May the item go to the semantic grader? Free text always; a short text answer only with a rubric."""
        return self.response_type == ResponseType.FREE_TEXT or (
            self.response_type == ResponseType.SHORT_TEXT and self.rubric_id is not None)

    def content_hash(self) -> str:
        return _hash(self)


class ScoreScale(Schema):
    """The partial-credit scale. Criterion scores are snapped to the nearest point (ties go down)."""

    points: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    labels: list[str] = Field(default_factory=lambda: ["incorrect", "minimal", "partial", "mostly correct",
                                                       "correct"])

    @model_validator(mode="after")
    def _valid(self) -> ScoreScale:
        if sorted(set(self.points)) != self.points or len(self.points) < 2:
            raise ValueError("scale points must be strictly increasing (at least two)")
        if self.points[0] != 0.0 or self.points[-1] != 1.0:
            raise ValueError("a scale runs from 0.0 to 1.0")
        if self.labels and len(self.labels) != len(self.points):
            raise ValueError("one label per scale point")
        return self

    def snap(self, value: float) -> float:
        return min(self.points, key=lambda p: (abs(p - value), p))


class RubricCriterion(Schema):
    criterion_id: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)
    weight: float = Field(gt=0, le=1)
    required: bool = False  # a required criterion that is not met caps the grade at PARTIAL
    concept_id: str | None = None
    indicators: list[str] = Field(default_factory=list)  # reference phrasings that show the criterion is met


class AssessmentRubric(Schema):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rubric_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    criteria: list[RubricCriterion] = Field(min_length=1)
    scale: ScoreScale = Field(default_factory=ScoreScale)
    passing_threshold: float = Field(default=0.8, gt=0, le=1)  # the CORRECT threshold for items using the rubric
    language: str = Field(default="en", min_length=2, max_length=16)
    deterministic: bool = False  # True: criteria are scored from their indicators by code, never by a model

    @model_validator(mode="after")
    def _valid(self) -> AssessmentRubric:
        ids = [c.criterion_id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        total = sum(c.weight for c in self.criteria)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"criterion weights must sum to 1.0, not {round(total, 6)}")
        if self.deterministic and any(not c.indicators for c in self.criteria):
            raise ValueError("every criterion of a deterministic rubric needs indicators")
        return self

    def content_hash(self) -> str:
        return _hash(self)


# --- grades --------------------------------------------------------------------------------------------------------


class CriterionResult(Schema):
    criterion_id: str
    score: float = Field(ge=0, le=1)  # on the rubric's scale
    weight: float = Field(gt=0, le=1)
    weighted_score: float = Field(ge=0, le=1)
    met: bool  # score >= 0.5
    rationale: str = ""


class Misconception(Schema):
    concept_id: str
    type: str
    description: str
    confidence: float = Field(ge=0, le=1)
    source: Literal["rule", "semantic"]


class AssessmentFeedback(Schema):
    outcome: AssessmentOutcome
    strengths: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    explanation: str = ""
    next_hint: str = ""
    citations: list[str] = Field(default_factory=list)  # refs of the lesson/research context only


class GraderUsage(Schema):
    """What grading cost. No prompts, no raw provider payloads, no configuration."""

    llm_calls: int = 0
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None


class AssessmentGrade(Schema):
    """Immutable. The final outcome, score and criterion results are computed by code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grade_id: str
    assessment_item_id: str
    attempt_id: str
    learner_answer: str
    normalized_answer: str
    score: float = Field(ge=0)  # points out of max_score
    max_score: float = Field(gt=0)
    percentage: float = Field(ge=0, le=100)
    outcome: AssessmentOutcome
    criterion_results: list[CriterionResult] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    grader_type: GraderType
    rubric_id: str | None = None
    matched: str | None = None  # the acceptable answer, choice or rule that decided a deterministic grade
    misconceptions: list[Misconception] = Field(default_factory=list)
    feedback: AssessmentFeedback
    uncertainty_reason: str | None = None
    notes: list[str] = Field(default_factory=list)  # deterministic adjustments, e.g. "capped at PARTIAL: ..."
    grader: GraderUsage = Field(default_factory=GraderUsage)
    created_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> AssessmentGrade:
        if self.score > self.max_score + 1e-9:
            raise ValueError("score cannot exceed max_score")
        if (self.outcome == AssessmentOutcome.UNCERTAIN) != (self.uncertainty_reason is not None):
            raise ValueError("an UNCERTAIN grade (and only one) states its reason")
        if self.feedback.outcome != self.outcome:
            raise ValueError("the feedback must carry the grade's outcome")
        return self

    @property
    def fraction(self) -> float:
        return round(self.score / self.max_score, 4)

    @property
    def counts(self) -> bool:
        """Does the grade count as evidence? UNCERTAIN never does."""
        return self.outcome != AssessmentOutcome.UNCERTAIN


class AttemptOutcome(Schema):
    """What an API attempt changed downstream, recorded once when the attempt completes."""

    evidence_ids: list[str] = Field(default_factory=list)
    mastery_changes: list[dict] = Field(default_factory=list)  # MasteryChange dumps from learner memory
    objective_progress: dict | None = None
    learning_action: dict | None = None
    artifact_ids: dict[str, str] = Field(default_factory=dict)


class AssessmentAttempt(Schema):
    attempt_id: str
    assessment_item_id: str
    learner_id: str
    attempt_number: int = Field(ge=1)  # per learner and item, in submission order
    learner_answer: str
    answer_hash: str  # what makes a repeated attempt a replay (same) or a conflict (different)
    submitted_at: datetime
    grade_id: str
    source: AttemptSource
    source_ref: str | None = None  # the teaching turn or evaluation question
    task_id: str | None = None  # where its artifacts and events belong
    completed_at: datetime | None = None  # its artifacts, events (and evidence for API attempts) are published
    outcome: AttemptOutcome | None = None


class AttemptConflict(Exception):
    """The attempt id was already used for a different answer or item (409). Nothing was stored."""


class ItemConflict(ValueError):
    """An item or rubric id is already stored with different content (409): items and rubrics are immutable."""


class AttemptExists(Exception):
    """The attempt id is already stored (the store refused a second copy); the caller replays or reports a conflict."""


# --- the semantic grader boundary ----------------------------------------------------------------------------------


class ContextPassage(Schema):
    ref: str  # "lesson:<section id>" or a research citation id: what a citation may name
    kind: Literal["lesson", "research"]
    title: str
    text: str


class SemanticItemBrief(Schema):
    prompt: str
    expected_answer: str
    acceptable_answers: list[str] = Field(default_factory=list)
    response_type: ResponseType
    concept_id: str
    difficulty: float = Field(ge=0, le=1)
    misconceptions: list[MisconceptionPattern] = Field(default_factory=list)


class SemanticGradingRequest(Schema):
    """Everything the grader sees, and nothing more: the question, the expected answer, the rubric, the answer and the
    relevant lesson section(s) and research evidence. No learner, task or session ids, no history, no other goals."""

    language: str
    item: SemanticItemBrief
    rubric: AssessmentRubric
    learner_answer: str = Field(min_length=1, max_length=4000)
    lesson_context: list[ContextPassage] = Field(default_factory=list)
    research_evidence: list[ContextPassage] = Field(default_factory=list)

    def allowed_citations(self) -> set[str]:
        return {p.ref for p in [*self.lesson_context, *self.research_evidence]}

    def known_concepts(self) -> set[str]:
        return ({self.item.concept_id} | {c.concept_id for c in self.rubric.criteria if c.concept_id}
                | {m.concept_id for m in self.item.misconceptions if m.concept_id})


class CandidateCriterion(Schema):
    criterion_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    rationale: str = Field(default="", max_length=500)


class CandidateMisconception(Schema):
    concept_id: str = Field(min_length=1)
    type: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1)


class CandidateFeedback(Schema):
    strengths: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    explanation: str = Field(default="", max_length=1000)
    next_hint: str = Field(default="", max_length=500)


class SemanticGradeCandidate(Schema):
    """The grader's proposal. Validated (schema, then the rules in app/assessment/validation.py) before use: the
    score and outcome are advisory; code aggregates the criterion scores and classifies the outcome."""

    score: float = Field(ge=0, le=1)
    criterion_results: list[CandidateCriterion]
    outcome: AssessmentOutcome
    confidence: float = Field(ge=0, le=1)
    misconceptions: list[CandidateMisconception] = Field(default_factory=list)
    feedback: CandidateFeedback = Field(default_factory=CandidateFeedback)
    citations: list[str] = Field(default_factory=list)
    insufficient_context: bool = False  # the material does not let the grader decide: the grade is UNCERTAIN


class SemanticGraderResult(Schema):
    candidate: SemanticGradeCandidate | None = None
    error: str | None = None  # why there is no candidate (invalid output after retries, providers down)
    usage: GraderUsage = Field(default_factory=GraderUsage)


# --- service inputs and views --------------------------------------------------------------------------------------


class GradingContext(Schema):
    lesson_context: list[ContextPassage] = Field(default_factory=list)
    research_evidence: list[ContextPassage] = Field(default_factory=list)


class AssessRequest(Schema):
    """One answer to grade through the AssessmentService (the teaching session and the evaluation workflow)."""

    item: AssessmentItem
    rubric: AssessmentRubric | None = None
    learner_id: str
    answer: str = Field(max_length=4000)  # an empty answer is graded INCORRECT by rule
    attempt_id: str = Field(min_length=1, max_length=128)
    task_id: str
    source: AttemptSource
    source_ref: str | None = None


class AssessmentResult(Schema):
    attempt: AssessmentAttempt
    grade: AssessmentGrade
    replayed: bool
    artifact_ids: dict[str, str] = Field(default_factory=dict)


class BatchAssessmentRequest(Schema):
    requests: list[AssessRequest] = Field(min_length=1)


class BatchAssessmentResult(Schema):
    results: list[AssessmentResult]


class RegisterAssessmentItem(Schema):
    """POST /assessment-items: an item on a lesson, with its rubric when it has one."""

    item: AssessmentItem
    rubric: AssessmentRubric | None = None

    @model_validator(mode="after")
    def _rubric(self) -> RegisterAssessmentItem:
        if self.rubric is not None and self.item.rubric_id != self.rubric.rubric_id:
            raise ValueError("the item's rubric_id must name the rubric sent with it")
        if self.item.lesson_id is None:
            raise ValueError("an item assesses a lesson: lesson_id is required")
        return self


class AttemptSubmission(Schema):
    learner_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(min_length=1, max_length=4000)
    attempt_id: str | None = Field(default=None, min_length=1, max_length=128)  # idempotency key of the attempt


class PublicGrade(Schema):
    """A grade as the learner sees it: no provider internals."""

    grade_id: str
    assessment_item_id: str
    attempt_id: str
    outcome: AssessmentOutcome
    score: float
    max_score: float
    percentage: float
    confidence: float
    grader_type: GraderType
    criterion_results: list[CriterionResult]
    misconceptions: list[Misconception]
    feedback: AssessmentFeedback
    uncertainty_reason: str | None
    notes: list[str]
    created_at: datetime

    @classmethod
    def of(cls, grade: AssessmentGrade) -> PublicGrade:
        return cls(**grade.model_dump(include=set(cls.model_fields)))


class AttemptView(Schema):
    attempt_id: str
    assessment_item_id: str
    learner_id: str
    attempt_number: int
    learner_answer: str
    submitted_at: datetime
    grade_id: str
    source: AttemptSource
    outcome: AssessmentOutcome
    completed: bool
    mastery_updated: bool
    evidence_ids: list[str]
    retry_recommended: bool  # UNCERTAIN: the answer could not be graded, another attempt is welcome


class AttemptResult(Schema):
    attempt: AttemptView
    grade: PublicGrade
    replayed: bool
    mastery_changes: list[dict] = Field(default_factory=list)
    objective_progress: dict | None = None
    learning_action: dict | None = None
