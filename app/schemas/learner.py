"""Learner model schemas. Domain-independent: subjects and levels are data, never code."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import Schema, utcnow


class AnswerEvaluation(Schema):
    question_id: str
    concept_id: str
    answer: str
    expected: str
    correct: bool
    difficulty: float = Field(ge=0, le=1)
    feedback: str = ""


class ConceptMastery(Schema):
    concept_id: str
    name: str
    subject: str
    mastery: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence_count: int = 0
    exposures: int = 0
    last_evidence_at: datetime | None = None
    next_review_at: datetime | None = None


class SubjectState(Schema):
    subject: str
    framework_id: str
    target_level: str | None = None
    estimated_level: str | None = None
    objectives: list[str] = Field(default_factory=list)


class AssessmentRecord(Schema):
    task_id: str
    subject: str
    at: datetime = Field(default_factory=utcnow)
    estimated_level: str | None = None
    evaluations: list[AnswerEvaluation] = Field(default_factory=list)


class MistakeRecord(Schema):
    task_id: str
    concept_id: str
    question_id: str
    answer: str
    expected: str
    at: datetime = Field(default_factory=utcnow)


class MasteryChange(Schema):
    concept_id: str
    before: float
    after: float
    reason: str


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


class LearnerProgress(Schema):
    learner_id: str
    subjects: list[SubjectState]
    lessons_completed: int
    assessments_taken: int
    mastered: list[str]
    weak: list[str]
    due_for_review: list[str]
    average_mastery: float
