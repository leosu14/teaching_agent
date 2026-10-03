"""Schemas for the lesson-generation learning loop: every agent input and output is defined here."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import Schema
from app.schemas.concepts import ConceptRef
from app.schemas.pedagogy import AdaptiveQuestioningPolicy, KnowledgeGap, LearnerContext, LearningObjective, PlanBrief
from app.schemas.learner import AnswerEvaluation, LearnerSnapshot, LearnerSummary
from app.schemas.research import Citation, ResearchBundle
from app.schemas.visual import ImageOrigin, VisualType

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
    questioning: AdaptiveQuestioningPolicy = Field(default_factory=AdaptiveQuestioningPolicy)

    def confident_concepts(self) -> set[str]:
        """Concepts memory already covers with enough evidence and confidence: not worth asking about again."""
        return {c.concept_id for c in self.snapshot.concept_mastery
                if c.evidence_count > 0 and c.confidence >= self.memory_confidence_threshold}

    def questions_per_concept(self) -> dict[str, int]:
        asked: dict[str, int] = {}
        for r in self.rounds:
            for item in r.items:
                asked[item.question.concept_id] = asked.get(item.question.concept_id, 0) + 1
        return asked


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


class ResearchRequest(Schema):
    """What the lesson needs researched. The research schemas themselves are domain-independent."""

    request: LessonRequest
    diagnostic: DiagnosticResult
    concepts: list[ConceptRef] = Field(min_length=1)
    max_results_per_query: int = Field(default=5, ge=1, le=50)
    max_sources: int = Field(default=6, ge=1)
    min_reliability: float = Field(default=0.5, ge=0, le=1)


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
    """The curriculum planner words the deterministic pedagogical plan; it never chooses the concepts."""

    request: LessonRequest
    learner: LearnerContext
    pedagogical_plan: PlanBrief
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


class SectionVisual(Schema):
    """A section's reference to an IMAGE_ASSET artifact. Visual attribution lives on the artifact, separate
    from the section's textual citations."""

    visual_id: str
    artifact_id: str
    asset_id: str
    visual_type: VisualType
    origin: ImageOrigin
    purpose: str
    description: str


SectionPurpose = Literal["explanation", "example", "guided_practice", "free_practice", "review", "assessment"]


class LessonSection(Schema):
    section_id: str
    concept_id: str
    purpose: SectionPurpose = "explanation"  # the section's pedagogical purpose
    objective_ids: list[str] = Field(default_factory=list)  # the lesson objectives this section serves
    heading: str
    explanation: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)
    analogy: str | None = None
    narration: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)  # citation ids from the research bundle
    # IMAGE_ASSET artifacts illustrating this section. Attached by the workflow after review, never model-written.
    visuals: list[SectionVisual] = Field(default_factory=list)


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
    # Explicit objectives (from the pedagogical plan); sections reference them by id.
    objectives: list[LearningObjective] = Field(default_factory=list)
    introduction: str
    sections: list[LessonSection] = Field(min_length=1)
    exercises: list[Exercise] = Field(default_factory=list)
    check_questions: list[CheckQuestion] = Field(default_factory=list)
    summary: str
    # Resolved from the research bundle by the workflow for every citation id the sections use.
    references: list[Citation] = Field(default_factory=list)

    @field_validator("sections")
    @classmethod
    def _unique_sections(cls, sections: list[LessonSection]) -> list[LessonSection]:
        ids = [s.section_id for s in sections]
        if len(ids) != len(set(ids)):
            raise ValueError("section ids must be unique")
        return sections

    @model_validator(mode="after")
    def _objectives_resolve(self) -> LessonContent:
        known = {o.objective_id for o in self.objectives}
        if len(known) != len(self.objectives):
            raise ValueError("objective ids must be unique")
        for s in self.sections:
            unknown = sorted(set(s.objective_ids) - known)
            if unknown:
                raise ValueError(f"section {s.section_id} references unknown objectives {unknown}")
        return self


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
    """What the teacher model sees: the plan and its pedagogical structure, the minimum learner context and the
    gaps being taught. No learner id, history or raw answers."""

    request: LessonRequest
    plan: LessonPlan
    research: ResearchBundle
    pedagogical_plan: PlanBrief
    learner: LearnerContext
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    revision: RevisionContext | None = None


class ReviewerInput(Schema):
    request: LessonRequest
    plan: LessonPlan
    research: ResearchBundle
    content: LessonContent
    revision_number: int = 0


# --- Visuals -----------------------------------------------------------------------------


class VisualPlanningInput(Schema):
    """What the model sees when planning visuals for an approved lesson."""

    plan: LessonPlan
    lesson: LessonContent
    research: ResearchBundle
    max_visuals: int = Field(default=6, ge=0)
    # The task's image budget (None: no limit). A visual counts against each source it may use, fallback included.
    max_generated_images: int | None = Field(default=None, ge=0)
    max_searched_images: int | None = Field(default=None, ge=0)


class VisualRequest(Schema):
    """Input of the VisualAgent: an approved lesson and where its artifacts hang in the artifact graph."""

    plan: LessonPlan
    lesson: LessonContent
    research: ResearchBundle
    language: str | None = None
    max_visuals: int = Field(default=6, ge=0)
    max_candidates: int = Field(default=3, ge=1)  # searched candidates tried per visual before giving up
    max_generated_images: int | None = Field(default=None, ge=0)
    max_searched_images: int | None = Field(default=None, ge=0)
    parent_artifact_ids: list[str] = Field(default_factory=list)  # parents of the VISUAL_PLAN artifact

    def planning_input(self) -> VisualPlanningInput:
        return VisualPlanningInput(plan=self.plan, lesson=self.lesson, research=self.research,
                                   max_visuals=self.max_visuals, max_generated_images=self.max_generated_images,
                                   max_searched_images=self.max_searched_images)


# --- Learner memory update -----------------------------------------------------------------


class DiagnosticOutcome(Schema):
    """What learner memory records from a finished diagnostic: its graded answers become LearningEvidence."""

    task_id: str
    learner_id: str
    request: LessonRequest
    diagnostic: DiagnosticResult
    concepts: list[ConceptRef]


class LessonOutcome(Schema):
    task_id: str
    learner_id: str
    request: LessonRequest
    diagnostic: DiagnosticResult
    concepts: list[ConceptRef]
    lesson_title: str
    taught_concept_ids: list[str]
    artifact_ids: list[str] = Field(default_factory=list)
