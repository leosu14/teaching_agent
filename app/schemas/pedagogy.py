"""Adaptive pedagogy schemas: learner model, knowledge gaps, pedagogical plans, objectives, activities,
recommendations and the configuration of every deterministic decision.

Subject-independent: concepts, levels and activity types are data. Subject-specific behaviour belongs in the
knowledge base (concepts) and in a PedagogicalStrategy, never here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import Schema
from app.schemas.concepts import Concept
from app.schemas.learner import (
    ConceptMastery,
    LearnerPreferences,
    LearningEvent,
    LearningGoal,
    MasteryChange,
    MasteryConfig,
)

# --- Configuration ---------------------------------------------------------------------------------------------------

DifficultyBand = Literal["foundational", "guided", "independent", "consolidation"]
BANDS: tuple[DifficultyBand, ...] = ("foundational", "guided", "independent", "consolidation")


class DifficultyBands(Schema):
    """Mastery thresholds of the difficulty bands: [0, guided) foundational, [guided, independent) guided,
    [independent, consolidation) independent, [consolidation, 1] consolidation."""

    guided: float = Field(default=0.3, gt=0, lt=1)
    independent: float = Field(default=0.6, gt=0, lt=1)
    consolidation: float = Field(default=0.8, gt=0, lt=1)

    @model_validator(mode="after")
    def _ascending(self) -> DifficultyBands:
        if not self.guided < self.independent < self.consolidation:
            raise ValueError("band thresholds must be strictly ascending: guided < independent < consolidation")
        return self

    def band_for(self, mastery: float) -> DifficultyBand:
        if mastery >= self.consolidation:
            return "consolidation"
        if mastery >= self.independent:
            return "independent"
        if mastery >= self.guided:
            return "guided"
        return "foundational"


class PriorityWeights(Schema):
    """Weights of the gap-priority factors (each factor is 0-1; the priority is their weighted mean)."""

    deficit: float = Field(default=0.4, ge=0)
    prerequisite_importance: float = Field(default=0.15, ge=0)
    goal_relevance: float = Field(default=0.15, ge=0)
    recent_errors: float = Field(default=0.15, ge=0)
    recency: float = Field(default=0.05, ge=0)
    repeated_failure: float = Field(default=0.1, ge=0)

    @model_validator(mode="after")
    def _positive(self) -> PriorityWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one priority weight must be positive")
        return self


class ActivityMinutes(Schema):
    """Time the planner allocates per planned activity kind."""

    explanation: int = Field(default=5, ge=1)
    practice: int = Field(default=5, ge=1)
    review: int = Field(default=3, ge=1)
    assessment: int = Field(default=2, ge=1)


class PlannerConfig(Schema):
    max_target_concepts: int = Field(default=2, ge=1)
    max_prerequisite_concepts: int = Field(default=2, ge=0)
    max_review_concepts: int = Field(default=1, ge=0)
    default_minutes: int = Field(default=30, ge=1)
    minutes: ActivityMinutes = Field(default_factory=ActivityMinutes)


class AdaptiveQuestioningPolicy(Schema):
    """Adaptive follow-up for diagnostics: a missed concept gets a targeted follow-up, an answered one is not asked
    again, and asking stops at the question budget or when nothing is left to follow up."""

    max_questions: int = Field(default=12, ge=1)
    max_follow_ups_per_concept: int = Field(default=1, ge=0)
    skip_confident_concepts: bool = True  # do not probe concepts that memory already covers with confidence

    def first_round(self, concept_ids: list[str], confident: set[str]) -> list[str]:
        """Concepts worth probing in the first round (in the given order), within the question budget."""
        wanted = [c for c in concept_ids if not (self.skip_confident_concepts and c in confident)]
        return wanted[:self.max_questions]

    def follow_ups(self, asked: dict[str, int], missed: list[str], total_asked: int) -> list[str]:
        """Concepts that get a follow-up now: missed in the latest round, follow-up budget left, within the
        question budget. `asked` counts the questions already asked per concept."""
        remaining = self.max_questions - total_asked
        eligible = [c for c in dict.fromkeys(missed) if asked.get(c, 0) - 1 < self.max_follow_ups_per_concept]
        return eligible[:max(0, remaining)]

    def should_stop(self, asked: dict[str, int], missed: list[str], total_asked: int) -> bool:
        return not self.follow_ups(asked, missed, total_asked)


class PedagogyConfig(Schema):
    """Every threshold and weight the adaptive engine uses, in one place. Fingerprinted into plans so that the same
    learner state, goal, concepts and configuration always produce the same plan."""

    bands: DifficultyBands = Field(default_factory=DifficultyBands)
    mastery_target: float = Field(default=0.8, gt=0, le=1)  # a concept at or above this is mastered
    prerequisite_threshold: float = Field(default=0.6, gt=0, le=1)  # a prerequisite below this is unmet
    weights: PriorityWeights = Field(default_factory=PriorityWeights)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    questioning: AdaptiveQuestioningPolicy = Field(default_factory=AdaptiveQuestioningPolicy)
    mastery: MasteryConfig = Field(default_factory=MasteryConfig)
    error_window: int = Field(default=10, ge=1)  # most recent incorrect evidence considered "recent errors"
    error_saturation: int = Field(default=3, ge=1)  # this many recent errors on a concept is the maximum signal
    repeated_failure_streak: int = Field(default=2, ge=1)  # consecutive incorrect answers that count as repeated
    recency_horizon_days: float = Field(default=30, gt=0)  # unpractised this long is the maximum recency signal
    history_limit: int = Field(default=20, ge=1)  # learning events kept on the learner model
    strategies: dict[str, str] = Field(default_factory=dict)  # domain -> strategy id; others use the default

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:16]


# --- Learner model ---------------------------------------------------------------------------------------------------


class LearnerError(Schema):
    """A recent incorrect answer, by reference: the evidence id, never the learner's raw answer."""

    concept_id: str
    evidence_id: str
    source_type: str
    at: datetime


class EvaluationSummary(Schema):
    task_id: str
    kind: str
    at: datetime
    questions: int = Field(ge=0)
    correct: int = Field(ge=0)
    concept_ids: list[str]


class LearnerModel(Schema):
    """The learner's current state in one domain, built deterministically from mastery, evidence and history.

    `concepts` is the canonical numeric state; the mastered/developing/weak/unknown lists are derived from it with
    the configured bands, for presentation and quick lookup."""

    learner_id: str
    domain: str
    framework_id: str
    level: str | None = None
    target_level: str | None = None
    goals: list[LearningGoal] = Field(default_factory=list)
    concepts: list[ConceptMastery] = Field(default_factory=list)
    mastered_concepts: list[str] = Field(default_factory=list)
    developing_concepts: list[str] = Field(default_factory=list)
    weak_concepts: list[str] = Field(default_factory=list)
    unknown_concepts: list[str] = Field(default_factory=list)
    due_for_review: list[str] = Field(default_factory=list)
    recent_errors: list[LearnerError] = Field(default_factory=list)
    recent_evaluations: list[EvaluationSummary] = Field(default_factory=list)
    learning_history: list[LearningEvent] = Field(default_factory=list)
    preferences: LearnerPreferences = Field(default_factory=LearnerPreferences)
    evidence_count: int = Field(default=0, ge=0)
    as_of: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _consistent(self) -> LearnerModel:
        ids = [c.concept_id for c in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("concept states must be unique")
        groups = [set(self.mastered_concepts), set(self.developing_concepts), set(self.weak_concepts),
                  set(self.unknown_concepts)]
        if sum(len(g) for g in groups) != len(set().union(*groups)):
            raise ValueError("mastered, developing, weak and unknown concepts must not overlap")
        if any(g.learner_id != self.learner_id for g in self.goals):
            raise ValueError("goals must belong to the learner")
        return self

    def state(self, concept_id: str) -> ConceptMastery | None:
        return next((c for c in self.concepts if c.concept_id == concept_id), None)

    def mastery_of(self, concept_id: str) -> float:
        """Estimated mastery; a concept without evidence counts as 0 (nothing is known about it yet)."""
        state = self.state(concept_id)
        return state.mastery if state is not None and state.evidence_count else 0.0

    def has_evidence(self, concept_id: str) -> bool:
        state = self.state(concept_id)
        return state is not None and state.evidence_count > 0

    def errors_on(self, concept_id: str) -> int:
        return sum(1 for e in self.recent_errors if e.concept_id == concept_id)

    def lessons_on(self, concept_id: str) -> int:
        return sum(1 for e in self.learning_history if e.type == "lesson_completed" and concept_id in e.concept_ids)


class ConceptContext(Schema):
    concept_id: str
    name: str
    band: DifficultyBand | Literal["unknown"]
    recent_errors: int = Field(ge=0)


class LearnerContext(Schema):
    """The minimum a model needs about the learner to write a lesson: no learner id, no history, no answers."""

    domain: str
    level: str | None
    target_level: str | None
    levels: list[str] = Field(default_factory=list)  # the level framework's levels
    language_of_instruction: str
    explanation_style: str
    concepts: list[ConceptContext] = Field(default_factory=list)


def learner_context(model: LearnerModel, plan: PlanBrief, levels: list[str]) -> LearnerContext:
    """The provider-facing projection of the learner model for one lesson: only the plan's concepts, as difficulty
    bands and error counts. Nothing identifies the learner and no history or answers leave the system."""
    return LearnerContext(
        domain=model.domain, level=model.level, target_level=model.target_level, levels=levels,
        language_of_instruction=model.preferences.language_of_instruction,
        explanation_style=model.preferences.explanation_style,
        concepts=[ConceptContext(concept_id=t.concept_id, name=t.name,
                                 band=t.band if model.has_evidence(t.concept_id) else "unknown",
                                 recent_errors=model.errors_on(t.concept_id)) for t in plan.treatments])


# --- Knowledge gaps ----------------------------------------------------------------------------------------------------

GapAction = Literal["introduce", "reteach", "reinforce", "prerequisite_first"]


class PriorityFactors(Schema):
    deficit: float = Field(ge=0, le=1)
    prerequisite_importance: float = Field(ge=0, le=1)
    goal_relevance: float = Field(ge=0, le=1)
    recent_errors: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    repeated_failure: float = Field(ge=0, le=1)


class KnowledgeGap(Schema):
    """A concept in the goal's scope that is below the mastery target. `priority` is internal pedagogical urgency
    for this learner, never a learner score or a ranking between learners."""

    concept: Concept
    mastery: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    band: DifficultyBand
    assessed: bool
    priority: float = Field(ge=0, le=1)
    factors: PriorityFactors
    prerequisites: list[str] = Field(default_factory=list)
    unmet_prerequisites: list[str] = Field(default_factory=list)
    reason: str
    recommended_action: GapAction

    @model_validator(mode="after")
    def _unmet_are_prerequisites(self) -> KnowledgeGap:
        if not set(self.unmet_prerequisites) <= set(self.prerequisites):
            raise ValueError("unmet prerequisites must be prerequisites of the concept")
        if (self.recommended_action == "prerequisite_first") != bool(self.unmet_prerequisites):
            raise ValueError("prerequisite_first is the action exactly when a prerequisite is unmet")
        return self


class KnowledgeGapSet(Schema):
    gap_set_id: str
    learner_id: str
    goal_id: str
    domain: str
    gaps: list[KnowledgeGap] = Field(default_factory=list)
    mastered: list[str] = Field(default_factory=list)  # in scope and at or above the target
    due_for_review: list[str] = Field(default_factory=list)  # mastered but due for spaced review
    scope: list[str] = Field(default_factory=list)  # goal targets and their prerequisites, prerequisites first
    config_fingerprint: str
    explanation: str | None = None  # optional model-written explanation; never used for decisions

    @model_validator(mode="after")
    def _ordered(self) -> KnowledgeGapSet:
        ids = [g.concept.concept_id for g in self.gaps]
        if len(ids) != len(set(ids)):
            raise ValueError("a concept appears in at most one gap")
        if [g.priority for g in self.gaps] != sorted((g.priority for g in self.gaps), reverse=True):
            raise ValueError("gaps must be ordered by priority, highest first")
        if not set(ids) <= set(self.scope) or not set(self.due_for_review) <= set(self.mastered):
            raise ValueError("gaps lie in the scope, and only mastered concepts can be due for review")
        return self

    def gap(self, concept_id: str) -> KnowledgeGap | None:
        return next((g for g in self.gaps if g.concept.concept_id == concept_id), None)

    def for_concepts(self, concept_ids: list[str]) -> list[KnowledgeGap]:
        wanted = set(concept_ids)
        return [g for g in self.gaps if g.concept.concept_id in wanted]


# --- Objectives, activities and the pedagogical plan ------------------------------------------------------------

CORE_ACTIVITY_TYPES = ("explanation", "multiple_choice", "fill_blank", "translation", "speaking", "writing",
                       "listening", "reading", "free_response")
ACTIVITY_TYPE_PATTERN = r"^[a-z][a-z0-9_]*$"  # extensible: a strategy may add its own types


class LearningObjective(Schema):
    objective_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target_mastery: float = Field(ge=0, le=1)
    assessment_method: str = Field(pattern=ACTIVITY_TYPE_PATTERN)


class LearningActivity(Schema):
    activity_id: str = Field(min_length=1)
    type: str = Field(pattern=ACTIVITY_TYPE_PATTERN)
    concept_ids: list[str] = Field(min_length=1)
    difficulty: DifficultyBand
    estimated_minutes: int = Field(ge=1)
    instructions: str = Field(min_length=1)
    expected_response: str = Field(min_length=1)
    assessment_target: str | None = None  # the objective this activity provides evidence for


ConceptRole = Literal["target", "prerequisite", "review"]
TeachingMode = Literal["introduce", "reinforce", "review"]
Phase = Literal["prerequisite_review", "instruction", "practice", "spaced_review", "assessment"]


class ConceptTreatment(Schema):
    """How the plan treats one concept: why it is in the lesson and whether it is introduced or reinforced."""

    concept_id: str
    name: str
    role: ConceptRole
    mode: TeachingMode
    band: DifficultyBand
    mastery: float = Field(ge=0, le=1)
    prerequisites: list[str] = Field(default_factory=list)  # prerequisites taught or reviewed in this lesson


class SequenceStep(Schema):
    order: int = Field(ge=1)
    activity_id: str
    phase: Phase
    minutes: int = Field(ge=1)


class PlanBrief(Schema):
    """The structural plan without learner or goal identifiers: what a model may see when writing the lesson."""

    plan_id: str
    domain: str
    level: str | None = None
    strategy_id: str
    target_concepts: list[str] = Field(min_length=1)
    prerequisite_concepts: list[str] = Field(default_factory=list)
    review_concepts: list[str] = Field(default_factory=list)  # spaced review explicitly requested
    treatments: list[ConceptTreatment] = Field(min_length=1)
    lesson_objectives: list[LearningObjective] = Field(min_length=1)
    activities: list[LearningActivity] = Field(min_length=1)
    sequencing: list[SequenceStep] = Field(min_length=1)
    estimated_duration: int = Field(ge=1)  # minutes
    available_minutes: int = Field(ge=1)
    rationale: str = Field(min_length=1)

    def concept_ids(self) -> list[str]:
        """Every concept the lesson covers, in teaching order (prerequisites, targets, reviews)."""
        return [*self.prerequisite_concepts, *self.target_concepts, *self.review_concepts]

    def treatment(self, concept_id: str) -> ConceptTreatment:
        return next(t for t in self.treatments if t.concept_id == concept_id)

    def objectives_for(self, concept_id: str) -> list[LearningObjective]:
        return [o for o in self.lesson_objectives if o.concept_id == concept_id]

    @model_validator(mode="after")
    def _valid_plan(self) -> PlanBrief:
        groups = [self.prerequisite_concepts, self.target_concepts, self.review_concepts]
        concepts = self.concept_ids()
        if len(concepts) != len(set(concepts)):
            raise ValueError("a concept has exactly one role in the plan")
        known = set(concepts)
        if [t.concept_id for t in self.treatments] != concepts:
            raise ValueError("treatments must list every plan concept once, in teaching order")
        roles = {c: role for g, role in zip(groups, ("prerequisite", "target", "review")) for c in g}
        for t in self.treatments:
            if t.role != roles[t.concept_id]:
                raise ValueError(f"concept {t.concept_id} has role {t.role}, expected {roles[t.concept_id]}")
        objective_ids = [o.objective_id for o in self.lesson_objectives]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("objective ids must be unique")
        if any(o.concept_id not in known for o in self.lesson_objectives):
            raise ValueError("objectives must reference plan concepts")
        if any(not self.objectives_for(c) for c in self.target_concepts):
            raise ValueError("every target concept needs a learning objective")
        activity_ids = [a.activity_id for a in self.activities]
        if len(activity_ids) != len(set(activity_ids)):
            raise ValueError("activity ids must be unique")
        for a in self.activities:
            if not set(a.concept_ids) <= known:
                raise ValueError(f"activity {a.activity_id} references concepts outside the plan")
            if a.assessment_target is not None and a.assessment_target not in objective_ids:
                raise ValueError(f"activity {a.activity_id} assesses an unknown objective")
        if sorted(s.activity_id for s in self.sequencing) != sorted(activity_ids):
            raise ValueError("the sequence must schedule every activity exactly once")
        if [s.order for s in self.sequencing] != list(range(1, len(self.sequencing) + 1)):
            raise ValueError("sequence order must be 1..n")
        minutes = {a.activity_id: a.estimated_minutes for a in self.activities}
        if any(s.minutes != minutes[s.activity_id] for s in self.sequencing):
            raise ValueError("sequence minutes must match the activities")
        if sum(s.minutes for s in self.sequencing) != self.estimated_duration:
            raise ValueError("estimated_duration must equal the sum of the sequence")
        if self.estimated_duration > self.available_minutes:
            raise ValueError("the plan does not fit the available time")
        # Prerequisites first: a concept's first activity comes after every activity on a prerequisite it relies on.
        by_activity = {a.activity_id: a for a in self.activities}
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        for s in self.sequencing:
            for c in by_activity[s.activity_id].concept_ids:
                first.setdefault(c, s.order)
                last[c] = s.order
        for t in self.treatments:
            for pre in t.prerequisites:
                if pre not in known:
                    raise ValueError(f"{t.concept_id} relies on {pre}, which the plan does not cover")
                if t.concept_id in first and pre in last and first[t.concept_id] < first[pre]:
                    raise ValueError(f"{t.concept_id} is taught before its prerequisite {pre}")
        return self

    def structural_hash(self) -> str:
        body = self.model_dump(mode="json", exclude={"plan_id"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


class PedagogicalPlan(PlanBrief):
    learner_id: str
    goal_id: str
    gap_set_id: str
    config_fingerprint: str

    def brief(self) -> PlanBrief:
        return PlanBrief.model_validate(self.model_dump(
            exclude={"learner_id", "goal_id", "gap_set_id", "config_fingerprint"}))


# --- Recommendation and feedback -------------------------------------------------------------------------------------


class NextLearningRecommendation(Schema):
    """What to learn next, computed from the updated learner state (never model-written)."""

    learner_id: str
    goal_id: str
    recommended_concepts: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    prerequisite_review: list[str] = Field(default_factory=list)
    review_concepts: list[str] = Field(default_factory=list)
    suggested_activity_types: list[str] = Field(default_factory=list)
    estimated_duration: int = Field(ge=0)  # minutes
    goal_achieved: bool = False
    plan_id: str | None = None
    gap_set_id: str | None = None

    @field_validator("suggested_activity_types")
    @classmethod
    def _types(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("activity types must be unique")
        return value


class EvaluationFeedback(Schema):
    """What the evaluation changed, from the deterministic mastery state: mastered, still weak, changed, next."""

    mastered: list[str] = Field(default_factory=list)
    developing: list[str] = Field(default_factory=list)
    still_weak: list[str] = Field(default_factory=list)
    changes: list[MasteryChange] = Field(default_factory=list)
    review_next: list[str] = Field(default_factory=list)
    summary: str


# --- Tool inputs -----------------------------------------------------------------------------------------------------


class ConceptSet(Schema):
    domain: str
    concepts: list[Concept]


class GapAnalysisRequest(Schema):
    model: LearnerModel
    goal: LearningGoal
    concepts: list[Concept]


class PlanningRequest(Schema):
    model: LearnerModel
    gaps: KnowledgeGapSet
    goal: LearningGoal
    concepts: list[Concept]
    available_minutes: int | None = Field(default=None, ge=1)  # None: the learner's session length
    lesson_history: list[LearningEvent] | None = None  # None: the learner model's history


class RecommendationRequest(Schema):
    model: LearnerModel
    goal: LearningGoal
    concepts: list[Concept]
    available_minutes: int | None = Field(default=None, ge=1)


class FeedbackRequest(Schema):
    changes: list[MasteryChange]
    assessed_concepts: list[str]
    model: LearnerModel
    recommendation: NextLearningRecommendation
