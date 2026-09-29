"""Schemas for the lesson-generation learning loop: every agent input and output is defined here."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import Schema
from app.schemas.learner import AnswerEvaluation, LearnerSnapshot, LearnerSummary

# --- Request interpretation -------------------------------------------------------------


class LessonRequest(Schema):
    raw_request: str
    subject: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    framework_id: str = Field(min_length=1)
    target_level: str | None = None
    language_of_instruction: str = "en"
    capabilities: list[str] = Field(min_length=1)


class InterpretRequest(Schema):
    request: str = Field(min_length=1)
    learner_id: str


class InterpreterInput(Schema):
    request: str
    learner: LearnerSummary


# --- Knowledge concepts ------------------------------------------------------------------


class ConceptRef(Schema):
    concept_id: str
    name: str
    level: str | None = None
    prerequisites: list[str] = Field(default_factory=list)
    description: str = ""


class ProbeSpec(Schema):
    """Knowledge-base hints a diagnostic can draw on. The agent decides how to use them."""

    prompt: str
    answer: str
    accepted: list[str] = Field(default_factory=list)
    difficulty: float = Field(default=0.5, ge=0, le=1)


class ConceptEntry(Schema):
    concept: ConceptRef
    probes: list[ProbeSpec] = Field(default_factory=list)
    source_id: str


# --- Diagnostic --------------------------------------------------------------------------


class DiagnosticQuestion(Schema):
    question_id: str
    concept_id: str
    prompt: str
    difficulty: float = Field(ge=0, le=1)


class DiagnosticItem(Schema):
    question: DiagnosticQuestion
    expected_answer: str
    accepted_answers: list[str] = Field(default_factory=list)


class LearnerAnswer(Schema):
    question_id: str
    answer: str


class DiagnosticAnswers(Schema):
    answers: list[LearnerAnswer] = Field(min_length=1)


class DiagnosticRound(Schema):
    items: list[DiagnosticItem]
    answers: list[LearnerAnswer]


class ConceptEstimate(Schema):
    concept_id: str
    mastery: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class DiagnosticResult(Schema):
    source: Literal["assessment", "memory"]
    estimated_level: str
    concept_mastery: list[ConceptEstimate]
    known: list[str]
    gaps: list[str]
    starting_point: str
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    rationale: str = ""


class DiagnosticInput(Schema):
    request: LessonRequest
    snapshot: LearnerSnapshot
    concepts: list[ConceptEntry] = Field(default_factory=list)
    rounds: list[DiagnosticRound] = Field(default_factory=list)
    round_number: int = Field(default=1, ge=1)
    max_rounds: int = Field(default=2, ge=1)
    memory_confidence_threshold: float = Field(default=0.6, ge=0, le=1)


class DiagnosticStep(Schema):
    status: Literal["ask", "complete"]
    concepts: list[ConceptRef]
    items: list[DiagnosticItem] = Field(default_factory=list)
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    result: DiagnosticResult | None = None

    @model_validator(mode="after")
    def _consistent(self) -> DiagnosticStep:
        if self.status == "ask" and not self.items:
            raise ValueError("status 'ask' requires at least one item")
        if self.status == "complete" and self.result is None:
            raise ValueError("status 'complete' requires a result")
        return self


class DiagnosticQuestionSheet(Schema):
    """What the learner sees while the task is WAITING. Answer keys are never included."""

    round_number: int
    questions: list[DiagnosticQuestion]


# --- Research ----------------------------------------------------------------------------


class SourceCandidate(Schema):
    source_id: str
    url: str
    title: str
    publisher: str
    snippet: str
    retrieved_via: Literal["web", "knowledge_base"]
    metadata: dict = Field(default_factory=dict)


class Source(Schema):
    source_id: str
    url: str
    title: str
    publisher: str
    retrieved_via: Literal["web", "knowledge_base"]
    reliability: float = Field(ge=0, le=1)
    reliable: bool
    reason: str = ""


class Fact(Schema):
    fact_id: str
    concept_id: str
    statement: str
    example: str | None = None
    practice_prompt: str | None = None
    practice_answer: str | None = None
    source_ids: list[str] = Field(min_length=1)


class ResearchRequest(Schema):
    request: LessonRequest
    diagnostic: DiagnosticResult
    concepts: list[ConceptRef] = Field(min_length=1)


class ResearchInput(Schema):
    query: str
    request: LessonRequest
    diagnostic: DiagnosticResult
    concepts: list[ConceptRef]
    candidates: list[SourceCandidate] = Field(default_factory=list)


class ResearchBundle(Schema):
    query: str
    sources: list[Source]
    facts: list[Fact]
    context_summary: str

    @model_validator(mode="after")
    def _facts_cite_reliable_sources(self) -> ResearchBundle:
        reliable = {s.source_id for s in self.sources if s.reliable}
        for fact in self.facts:
            missing = [sid for sid in fact.source_ids if sid not in reliable]
            if missing:
                raise ValueError(f"fact {fact.fact_id} cites unknown or unreliable sources {missing}")
        return self


# --- Curriculum plan ---------------------------------------------------------------------


class PlannedConcept(Schema):
    concept_id: str
    name: str
    rationale: str
    strategy: str
    examples: list[str] = Field(default_factory=list)


class PlanStep(Schema):
    step_id: str
    concept_id: str | None
    activity: Literal["warm_up", "review", "explain", "practice", "assess"]
    minutes: int = Field(ge=1)


class PlannedExercise(Schema):
    exercise_id: str
    concept_id: str
    kind: Literal["short_answer", "multiple_choice", "translation", "coding", "problem"]
    prompt: str


class PlannerInput(Schema):
    request: LessonRequest
    snapshot: LearnerSnapshot
    diagnostic: DiagnosticResult
    research: ResearchBundle
    concepts: list[ConceptRef]


class LessonPlan(Schema):
    title: str
    level: str
    objectives: list[str] = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    concepts: list[PlannedConcept] = Field(min_length=1)
    review_concepts: list[str] = Field(default_factory=list)
    sequence: list[PlanStep] = Field(min_length=1)
    estimated_minutes: int = Field(ge=1)
    teaching_strategy: str
    exercises: list[PlannedExercise] = Field(default_factory=list)
    assessment: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _coherent(self) -> LessonPlan:
        if sum(step.minutes for step in self.sequence) != self.estimated_minutes:
            raise ValueError("estimated_minutes must equal the sum of sequence minutes")
        known = {c.concept_id for c in self.concepts} | set(self.review_concepts)
        for step in self.sequence:
            if step.concept_id is not None and step.concept_id not in known:
                raise ValueError(f"sequence step {step.step_id} references unknown concept {step.concept_id}")
        for exercise in self.exercises:
            if exercise.concept_id not in known:
                raise ValueError(f"exercise {exercise.exercise_id} references unknown concept")
        return self


# --- Teaching content --------------------------------------------------------------------


class LessonSection(Schema):
    section_id: str
    concept_id: str
    heading: str
    explanation: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)
    analogy: str | None = None
    narration: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class Exercise(Schema):
    exercise_id: str
    concept_id: str
    kind: str
    prompt: str
    answer: str
    explanation: str = ""


class CheckQuestion(Schema):
    question_id: str
    concept_id: str
    prompt: str
    answer: str


class LessonContent(Schema):
    title: str
    level: str
    introduction: str
    sections: list[LessonSection] = Field(min_length=1)
    exercises: list[Exercise] = Field(default_factory=list)
    check_questions: list[CheckQuestion] = Field(default_factory=list)
    summary: str

    @field_validator("sections")
    @classmethod
    def _unique_sections(cls, sections: list[LessonSection]) -> list[LessonSection]:
        ids = [s.section_id for s in sections]
        if len(ids) != len(set(ids)):
            raise ValueError("section ids must be unique")
        return sections


# --- Review ------------------------------------------------------------------------------


class Verdict(str, Enum):
    APPROVED = "APPROVED"
    REVISION_REQUIRED = "REVISION_REQUIRED"


class ReviewCriterion(str, Enum):
    FACTUAL_CORRECTNESS = "factual_correctness"
    PEDAGOGICAL_QUALITY = "pedagogical_quality"
    LEVEL_APPROPRIATENESS = "level_appropriateness"
    STRUCTURE = "structure"
    COMPLETENESS = "completeness"
    HALLUCINATION_RISK = "hallucination_risk"
    SOURCE_QUALITY = "source_quality"
    NARRATION_QUALITY = "narration_quality"
    EXERCISE_QUALITY = "exercise_quality"


class ReviewIssue(Schema):
    issue_id: str
    criterion: ReviewCriterion
    severity: Literal["minor", "major", "critical"]
    location: str
    problem: str
    suggested_fix: str


class ReviewResult(Schema):
    verdict: Verdict
    scores: dict[ReviewCriterion, float]
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str

    @model_validator(mode="after")
    def _verdict_matches_issues(self) -> ReviewResult:
        blocking = [i for i in self.issues if i.severity != "minor"]
        if self.verdict == Verdict.APPROVED and blocking:
            raise ValueError("APPROVED review cannot contain major or critical issues")
        if self.verdict == Verdict.REVISION_REQUIRED and not blocking:
            raise ValueError("REVISION_REQUIRED review must name at least one major or critical issue")
        for score in self.scores.values():
            if not 0 <= score <= 1:
                raise ValueError("scores must be between 0 and 1")
        return self


class RevisionContext(Schema):
    revision_number: int = Field(ge=1)
    issues: list[ReviewIssue]
    previous: LessonContent


class TeacherInput(Schema):
    request: LessonRequest
    plan: LessonPlan
    research: ResearchBundle
    snapshot: LearnerSnapshot
    revision: RevisionContext | None = None


class ReviewerInput(Schema):
    request: LessonRequest
    plan: LessonPlan
    research: ResearchBundle
    content: LessonContent
    revision_number: int = 0


# --- Slides ------------------------------------------------------------------------------

MAX_BULLETS_PER_SLIDE = 5
MAX_WORDS_PER_BULLET = 20


class VisualSpec(Schema):
    kind: Literal["none", "image", "diagram", "equation", "code"]
    description: str = ""


class Slide(Schema):
    slide_id: str
    kind: Literal["title", "objectives", "explanation", "example", "exercise", "summary", "diagram", "code", "equation"]
    heading: str
    bullets: list[str] = Field(default_factory=list, max_length=MAX_BULLETS_PER_SLIDE)
    visual: VisualSpec = Field(default_factory=lambda: VisualSpec(kind="none"))
    narration_section_id: str | None = None
    speaker_notes: str = ""

    @field_validator("bullets")
    @classmethod
    def _short_bullets(cls, bullets: list[str]) -> list[str]:
        for bullet in bullets:
            if len(bullet.split()) > MAX_WORDS_PER_BULLET:
                raise ValueError(f"bullet exceeds {MAX_WORDS_PER_BULLET} words: {bullet[:40]}...")
        return bullets


class SlideInput(Schema):
    lesson: LessonContent
    plan: LessonPlan


class SlideDeckPlan(Schema):
    title: str
    slides: list[Slide] = Field(min_length=2)


# --- Learner memory update -----------------------------------------------------------------


class LessonOutcome(Schema):
    task_id: str
    learner_id: str
    request: LessonRequest
    diagnostic: DiagnosticResult
    concepts: list[ConceptRef]
    lesson_title: str
    taught_concept_ids: list[str]
    artifact_ids: list[str] = Field(default_factory=list)
