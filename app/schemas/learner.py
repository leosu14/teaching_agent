"""Learner model schemas. Domain-independent: subjects and levels are data, never code."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import Schema, utcnow


# What a model sees instead of a learner id: providers never receive internal learner identifiers.
ANONYMOUS_LEARNER = "learner"


class AnswerEvaluation(Schema):
    question_id: str
    concept_id: str
    answer: str
    expected: str
    correct: bool
    difficulty: float = Field(ge=0, le=1)
    feedback: str = ""
    # Set when the answer was graded by the assessment layer: the validated outcome (CORRECT, PARTIAL, INCORRECT,
    # UNCERTAIN), its 0-1 score and the grade. An UNCERTAIN answer is never evidence.
    outcome: Literal["CORRECT", "PARTIAL", "INCORRECT", "UNCERTAIN"] | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    grade_id: str | None = None

    @property
    def counts(self) -> bool:
        return self.outcome != "UNCERTAIN"


class ConceptMastery(Schema):
    """The canonical mastery state of one concept: a normalised 0-1 estimate and the evidence behind it. Labels such
    as "weak" or "mastered" are derived from it for presentation, never stored as the state. Only the deterministic
    MasteryUpdater (app/learner/mastery.py) writes it, from LearningEvidence."""

    concept_id: str
    name: str
    subject: str
    mastery: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence_count: int = Field(default=0, ge=0)
    correct_count: int = Field(default=0, ge=0)
    incorrect_count: int = Field(default=0, ge=0)
    incorrect_streak: int = Field(default=0, ge=0)  # consecutive incorrect evidence, most recent last
    exposures: int = Field(default=0, ge=0)
    # Older profiles stored this as `last_evidence_at`; both names are accepted.
    last_assessed_at: datetime | None = Field(default=None,
                                              validation_alias=AliasChoices("last_assessed_at", "last_evidence_at"))
    last_updated_at: datetime | None = None
    next_review_at: datetime | None = None


class SubjectState(Schema):
    subject: str
    framework_id: str
    target_level: str | None = None
    estimated_level: str | None = None
    objectives: list[str] = Field(default_factory=list)


class MasteryChange(Schema):
    concept_id: str
    before: float
    after: float
    reason: str


class AssessmentRecord(Schema):
    task_id: str
    subject: str
    at: datetime = Field(default_factory=utcnow)
    estimated_level: str | None = None
    kind: str = "diagnostic"
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)
    mastery_changes: list[MasteryChange] = Field(default_factory=list)


class MistakeRecord(Schema):
    task_id: str
    concept_id: str
    question_id: str
    answer: str
    expected: str
    at: datetime = Field(default_factory=utcnow)


class LessonRecord(Schema):
    task_id: str
    subject: str
    topic: str
    title: str
    concept_ids: list[str]
    artifact_ids: list[str] = Field(default_factory=list)
    mastery_changes: list[MasteryChange] = Field(default_factory=list)
    at: datetime = Field(default_factory=utcnow)


class InteractionRecord(Schema):
    kind: str
    task_id: str | None = None
    detail: str = ""
    at: datetime = Field(default_factory=utcnow)


class LearnerPreferences(Schema):
    language_of_instruction: str = "en"
    session_minutes: int = Field(default=30, ge=5, le=240)
    explanation_style: str = "examples-first"
    modalities: list[str] = Field(default_factory=lambda: ["text", "slides"])


class LearnerProfile(Schema):
    learner_id: str
    display_name: str = ""
    subjects: dict[str, SubjectState] = Field(default_factory=dict)
    concepts: dict[str, ConceptMastery] = Field(default_factory=dict)
    assessments: list[AssessmentRecord] = Field(default_factory=list)
    mistakes: list[MistakeRecord] = Field(default_factory=list)
    lessons: list[LessonRecord] = Field(default_factory=list)
    interactions: list[InteractionRecord] = Field(default_factory=list)
    preferences: LearnerPreferences = Field(default_factory=LearnerPreferences)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class LearnerProfileInput(Schema):
    """What a client may set directly; history fields are only written by the learning loop."""

    display_name: str = ""
    subjects: list[SubjectState] = Field(default_factory=list)
    preferences: LearnerPreferences = Field(default_factory=LearnerPreferences)


class LearnerSummary(Schema):
    learner_id: str
    display_name: str
    subjects: list[SubjectState]
    preferences: LearnerPreferences
    lesson_count: int


class LearnerSnapshot(Schema):
    """Answers: what does the learner know, probably not know, need next, need to review."""

    learner_id: str
    subject: str
    framework_id: str
    framework_levels: list[str]
    target_level: str | None
    estimated_level: str | None
    concept_mastery: list[ConceptMastery]
    known: list[str]
    mastered: list[str]
    weak: list[str]
    likely_unknown: list[str]
    due_for_review: list[str]
    recommended_next: list[str]
    recent_topics: list[str]
    recent_mistakes: list[MistakeRecord]
    preferences: LearnerPreferences

    def for_provider(self) -> LearnerSnapshot:
        """The snapshot as a model may see it: the learner's state without their internal identifier."""
        return self.model_copy(update={"learner_id": ANONYMOUS_LEARNER})

    def has_evidence_for(self, concept_ids: list[str], min_confidence: float) -> bool:
        by_id = {c.concept_id: c for c in self.concept_mastery}
        return bool(concept_ids) and all(
            cid in by_id and by_id[cid].evidence_count > 0 and by_id[cid].confidence >= min_confidence
            for cid in concept_ids
        )


class MasteryUpdate(Schema):
    learner_id: str
    subject: str
    estimated_level: str | None
    changes: list[MasteryChange]
    evidence: list[LearningEvidence] = Field(default_factory=list)  # the evidence this update applied


class LearnerProgress(Schema):
    learner_id: str
    subjects: list[SubjectState]
    lessons_completed: int
    assessments_taken: int
    mastered: list[str]
    weak: list[str]
    due_for_review: list[str]
    average_mastery: float


# --- Evidence, learning events and goals -----------------------------------------------------------------------------

EvidenceSource = Literal["diagnostic", "lesson", "evaluation", "exercise", "manual", "interaction", "assessment"]
Correctness = Literal["correct", "partial", "incorrect"]


def graded_correctness(outcome: str, score: float) -> tuple[Correctness, float]:
    """A validated assessment outcome and its 0-1 score as evidence: CORRECT keeps at least 0.5, PARTIAL lies strictly
    between 0 and 1, INCORRECT stays below 0.5. UNCERTAIN is never evidence (the caller does not record it)."""
    if outcome == "UNCERTAIN":
        raise ValueError("an UNCERTAIN grade is not evidence")
    if outcome == "CORRECT":
        return "correct", round(max(0.5, min(1.0, score)), 4)
    if outcome == "PARTIAL":
        return "partial", round(min(0.95, max(0.05, score)), 4)
    return "incorrect", round(max(0.0, min(0.49, score)), 4)


def stable_id(prefix: str, *parts: str) -> str:
    """A deterministic id: the same parts always give the same id, so recording is idempotent."""
    return f"{prefix}_{hashlib.sha256(chr(31).join(parts).encode('utf-8')).hexdigest()[:20]}"


class EvidenceConflict(ValueError):
    """Evidence is immutable: the same id was recorded before with different content."""


class LearningEvidence(Schema):
    """One observation of a learner on one concept. Immutable once recorded; mastery is derived from it.

    Assessment-agnostic: a diagnostic answer, an exercise, a post-lesson evaluation or a manual placement all
    become the same record. `score` is the normalised result (0-1); `difficulty` weights how informative it is."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    source_type: EvidenceSource
    source_ref: str = Field(min_length=1)  # e.g. "<task id>/<question id>"
    correctness: Correctness
    score: float = Field(ge=0, le=1)
    difficulty: float = Field(default=0.5, ge=0, le=1)
    timestamp: datetime
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _score_matches_correctness(self) -> LearningEvidence:
        if self.correctness == "correct" and self.score < 0.5:
            raise ValueError("correct evidence needs a score of at least 0.5")
        if self.correctness == "incorrect" and self.score >= 0.5:
            raise ValueError("incorrect evidence needs a score below 0.5")
        if self.correctness == "partial" and not 0 < self.score < 1:
            raise ValueError("partial evidence needs a score strictly between 0 and 1")
        return self

    @staticmethod
    def id_for(learner_id: str, source_type: str, source_ref: str, concept_id: str) -> str:
        return stable_id("ev", learner_id, source_type, source_ref, concept_id)

    @classmethod
    def from_answer(cls, learner_id: str, source_type: EvidenceSource, task_id: str, evaluation: AnswerEvaluation,
                    at: datetime) -> LearningEvidence:
        """A graded answer (diagnostic or evaluation) as evidence. The grading is the agent's interpretation; the
        number it becomes and every mastery update after it are computed by code."""
        ref = f"{task_id}/{evaluation.question_id}"
        metadata = {"task_id": task_id, "question_id": evaluation.question_id}
        if evaluation.outcome is not None:  # graded by the assessment layer: partial credit
            correctness, score = graded_correctness(evaluation.outcome,
                                                    evaluation.score if evaluation.score is not None
                                                    else float(evaluation.correct))
            metadata.update(outcome=evaluation.outcome, grade_id=evaluation.grade_id)
        else:
            correctness, score = ("correct", 1.0) if evaluation.correct else ("incorrect", 0.0)
        return cls(evidence_id=cls.id_for(learner_id, source_type, ref, evaluation.concept_id), learner_id=learner_id,
                   concept_id=evaluation.concept_id, source_type=source_type, source_ref=ref,
                   correctness=correctness, score=score, difficulty=evaluation.difficulty, timestamp=at,
                   metadata=metadata)


LearningEventType = Literal["diagnostic_completed", "lesson_completed", "evaluation_completed", "concept_mastered",
                            "concept_reviewed", "exercise_completed"]


class LearningEvent(Schema):
    """An entry of the learner's learning history. History is append-only: the current state is derived from
    events and evidence, never the only thing kept."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    learner_id: str
    type: LearningEventType
    at: datetime
    subject: str | None = None
    task_id: str | None = None
    concept_ids: list[str] = Field(default_factory=list)
    data: dict = Field(default_factory=dict)

    @classmethod
    def create(cls, learner_id: str, type: LearningEventType, at: datetime, *, key: str, subject: str | None = None,
               task_id: str | None = None, concept_ids: list[str] | None = None, data: dict | None = None
               ) -> LearningEvent:
        return cls(event_id=stable_id("lev", learner_id, type, key), learner_id=learner_id, type=type, at=at,
                   subject=subject, task_id=task_id, concept_ids=concept_ids or [], data=data or {})


class GoalStatus(str, Enum):
    """ACTIVE goals are planned and taught; PAUSED ones keep their curriculum but get no next action; COMPLETED is
    set only by the deterministic completion rule (app/curriculum/progress.py); CANCELLED goals are kept as history."""

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


# Statuses stored before goals had a lifecycle of their own.
_LEGACY_GOAL_STATUS = {"active": "ACTIVE", "paused": "PAUSED", "achieved": "COMPLETED", "abandoned": "CANCELLED",
                       "completed": "COMPLETED", "cancelled": "CANCELLED"}
GoalTargetSource = Literal["explicit", "level"]


class LearningGoal(Schema):
    """What the learner is working towards: lessons are planned against a goal, not just a topic.

    Domain-independent: a language level ("Reach B2 Spanish"), an exam syllabus, a professional skill or any other
    subject the knowledge base covers. `target_concepts` are always real knowledge-base concepts; a goal created from
    a target level only gets them resolved from the knowledge base (`target_source="level"`), so a level change
    re-resolves them. `priority` 1 is the highest."""

    goal_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    title: str = ""
    description: str = ""
    domain: str = Field(min_length=1)
    target_level: str | None = None
    target_concepts: list[str] = Field(min_length=1)
    target_source: GoalTargetSource = "explicit"
    # Older goals stored this as `deadline`; both names are accepted.
    target_date: datetime | None = Field(default=None, validation_alias=AliasChoices("target_date", "deadline"))
    priority: int = Field(default=3, ge=1, le=5)  # 1 = highest
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _legacy_status(cls, value):
        return _LEGACY_GOAL_STATUS.get(value, value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _unique_targets(self) -> LearningGoal:
        if len(set(self.target_concepts)) != len(self.target_concepts):
            raise ValueError("target concepts must be unique")
        return self

    @property
    def is_active(self) -> bool:
        return self.status == GoalStatus.ACTIVE


class ReviewIntervals(Schema):
    """Base review intervals (days) by mastery: below `weak_below` -> weak, below `strong_from` -> medium."""

    weak_below: float = Field(default=0.5, ge=0, le=1)
    strong_from: float = Field(default=0.8, ge=0, le=1)
    weak_days: float = Field(default=1, gt=0)
    medium_days: float = Field(default=3, gt=0)
    strong_days: float = Field(default=7, gt=0)
    retention_multiplier: float = Field(default=2.0, ge=1)  # a success after a long gap stretches the interval
    max_days: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> ReviewIntervals:
        if self.weak_below > self.strong_from:
            raise ValueError("weak_below must not exceed strong_from")
        return self


class MasteryConfig(Schema):
    """Parameters of the deterministic mastery update. Successes on hard items and failures on easy ones move the
    estimate more; the step shrinks as evidence accumulates so the estimate stabilises."""

    prior_mastery: float = Field(default=0.3, ge=0, le=1)
    base_rate: float = Field(default=0.35, gt=0, le=1)
    informativeness_rate: float = Field(default=0.3, ge=0, le=1)
    rate_decay: float = Field(default=0.15, ge=0)
    confidence_base: float = Field(default=0.6, gt=0, lt=1)  # confidence = 1 - base ** evidence_count
    manual_weight: float = Field(default=1.0, ge=0, le=1)  # how far a manual placement moves mastery to its score
    review: ReviewIntervals = Field(default_factory=ReviewIntervals)

    @model_validator(mode="after")
    def _bounded_rate(self) -> MasteryConfig:
        if self.base_rate + self.informativeness_rate > 1:
            raise ValueError("base_rate + informativeness_rate must not exceed 1 (mastery would overshoot)")
        return self
