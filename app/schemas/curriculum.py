"""Curriculum schemas: a learner's long-term goal turned into a versioned curriculum of objectives, the objectives'
progress, review schedules, the next learning action and the configuration of every deterministic decision.

Domain-independent: concepts, levels and domains are data. A curriculum objective always references a real
knowledge-base concept; the learner model (mastery, evidence) stays the one in `app/schemas/learner.py`, and objective
progress is derived from it, never stored as a second learner model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import Schema
from app.schemas.learner import GoalStatus, LearningGoal, stable_id

# --- Enumerations ------------------------------------------------------------------------------------------------


class ObjectiveStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    MASTERED = "MASTERED"
    BLOCKED = "BLOCKED"  # a prerequisite is below the prerequisite threshold


class LearningActionType(str, Enum):
    LEARN = "LEARN"  # first instruction or reteaching
    REVIEW = "REVIEW"  # a mastered objective is due for retrieval practice
    PRACTICE = "PRACTICE"  # close to the target: controlled and applied exercises
    EVALUATE = "EVALUATE"  # at the target but without enough evidence: assess
    WAIT = "WAIT"  # nothing is due now
    COMPLETE = "COMPLETE"  # every required objective is mastered: the goal is complete


LESSON_ACTIONS = (LearningActionType.LEARN, LearningActionType.REVIEW, LearningActionType.PRACTICE,
                  LearningActionType.EVALUATE)


class CurriculumStatus(str, Enum):
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"  # the goal is complete
    ARCHIVED = "ARCHIVED"  # the goal was cancelled; the history is kept


class ReplanReason(str, Enum):
    INITIAL = "initial"
    REBUILD = "rebuild"  # an explicit rebuild request
    GOAL_CHANGED = "goal_changed"  # target level, target concepts or domain changed
    TARGET_DATE_CHANGED = "target_date_changed"
    REPEATED_FAILURE = "repeated_failure"  # an objective failed repeatedly: remediate its prerequisites
    PREREQUISITE_REGRESSION = "prerequisite_regression"  # a prerequisite assumed mastered is not
    EARLY_MASTERY = "early_mastery"  # an objective was mastered before the objectives ordered ahead of it


ObjectiveRole = Literal["target", "prerequisite"]  # a goal concept, or a prerequisite the goal relies on
ObjectiveMode = Literal["learn", "maintain"]  # maintain: mastered when planned; kept by spaced review
CompletionRule = Literal["all_required_mastered", "targets_mastered"]
WarningCode = Literal["deadline_infeasible", "deadline_passed", "unresolved_concept", "rejected_concept",
                      "ordering_rejected", "description_missing"]


# --- Configuration -----------------------------------------------------------------------------------------------


class PriorityModelWeights(Schema):
    """Weights of the next-action priority score (each factor is 0-1). Documented in docs/architecture.md."""

    deficit: float = Field(default=0.30, ge=0)
    objective_priority: float = Field(default=0.15, ge=0)
    goal_priority: float = Field(default=0.15, ge=0)
    review_urgency: float = Field(default=0.10, ge=0)
    deadline_pressure: float = Field(default=0.15, ge=0)
    recent_failure: float = Field(default=0.10, ge=0)
    recency: float = Field(default=0.05, ge=0)

    @model_validator(mode="after")
    def _positive(self) -> PriorityModelWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one priority weight must be positive")
        return self


class FeasibilityConfig(Schema):
    """A deliberately simple capacity estimate for target dates: each remaining objective needs
    ceil(deficit / mastery_gain_per_session) sessions, at `sessions_per_week` sessions a week."""

    sessions_per_week: float = Field(default=3, gt=0)
    mastery_gain_per_session: float = Field(default=0.25, gt=0, le=1)


class CurriculumConfig(Schema):
    """Every threshold and weight the curriculum engine uses. Fingerprinted into curriculum versions."""

    mastery_target: float = Field(default=0.8, gt=0, le=1)  # an objective's default target mastery
    prerequisite_threshold: float = Field(default=0.6, gt=0, le=1)  # below this a prerequisite blocks
    practice_from: float = Field(default=0.6, ge=0, le=1)  # at or above (and below target): PRACTICE, not LEARN
    evidence_required: int = Field(default=2, ge=1)  # evidence items before an objective can count as mastered
    remediation_evidence_bonus: int = Field(default=1, ge=0)
    target_priority: int = Field(default=2, ge=1, le=5)  # objective priority of a goal concept (1 = highest)
    prerequisite_priority: int = Field(default=3, ge=1, le=5)
    remediation_priority: int = Field(default=1, ge=1, le=5)
    repeated_failure_streak: int = Field(default=2, ge=1)
    completion_rule: CompletionRule = "all_required_mastered"
    weights: PriorityModelWeights = Field(default_factory=PriorityModelWeights)
    deadline_horizon_days: float = Field(default=90, gt=0)  # pressure grows linearly over the last N days
    recency_horizon_days: float = Field(default=30, gt=0)
    feasibility: FeasibilityConfig = Field(default_factory=FeasibilityConfig)
    max_objectives: int = Field(default=60, ge=1)
    max_candidates_reported: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> CurriculumConfig:
        if self.prerequisite_threshold > self.mastery_target:
            raise ValueError("prerequisite_threshold must not exceed mastery_target")
        if self.practice_from > self.mastery_target:
            raise ValueError("practice_from must not exceed mastery_target")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:16]


# --- Goals ---------------------------------------------------------------------------------------------------------


class GoalInput(Schema):
    """POST /learners/{learner_id}/goals. Without target concepts, the knowledge base's concepts up to the target
    level are the targets. The same `idempotency_key` never creates a second goal."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    domain: str = Field(min_length=1, max_length=128)
    target_level: str | None = Field(default=None, max_length=32)
    target_concepts: list[str] = Field(default_factory=list, max_length=200)
    target_date: datetime | None = None
    priority: int = Field(default=3, ge=1, le=5)
    metadata: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def _has_target(self) -> GoalInput:
        if not self.target_level and not self.target_concepts:
            raise ValueError("a goal needs a target level or target concepts")
        if len(set(self.target_concepts)) != len(self.target_concepts):
            raise ValueError("target concepts must be unique")
        return self


class GoalUpdate(Schema):
    """PATCH /goals/{goal_id}. A goal is COMPLETED only by the deterministic completion rule, never by request."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    target_level: str | None = Field(default=None, max_length=32)
    target_concepts: list[str] | None = Field(default=None, max_length=200)
    target_date: datetime | None = None
    clear_target_date: bool = False
    priority: int | None = Field(default=None, ge=1, le=5)
    status: GoalStatus | None = None
    metadata: dict | None = None

    @field_validator("status")
    @classmethod
    def _not_completed(cls, value: GoalStatus | None) -> GoalStatus | None:
        if value == GoalStatus.COMPLETED:
            raise ValueError("a goal is completed by the completion rule, not by request")
        return value


class GoalSpec(Schema):
    """What a curriculum version was planned against: the goal's definition at that time."""

    title: str
    domain: str
    target_level: str | None = None
    target_concepts: list[str]
    target_source: str = "explicit"
    target_date: datetime | None = None
    priority: int = Field(ge=1, le=5)

    @classmethod
    def of(cls, goal: LearningGoal) -> GoalSpec:
        return cls(title=goal.title or goal.description or goal.goal_id, domain=goal.domain,
                   target_level=goal.target_level, target_concepts=list(goal.target_concepts),
                   target_source=goal.target_source, target_date=goal.target_date, priority=goal.priority)


# --- Objectives and curricula ---------------------------------------------------------------------------------------


class CurriculumObjective(Schema):
    """The bridge between a long-term goal and the concept/mastery system: one knowledge-base concept with a target
    mastery, its prerequisites (other objectives' concepts) and the evidence it needs.

    (The per-lesson `LearningObjective` in app/schemas/pedagogy.py is what one lesson states; a curriculum
    objective spans many lessons.) `current_mastery` and `status` are progress: in a stored curriculum version they
    are the values at planning time, in the current curriculum view they are recomputed from the learner model."""

    objective_id: str = Field(min_length=1)
    goal_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=400)
    order: int = Field(ge=1)
    role: ObjectiveRole
    mode: ObjectiveMode = "learn"
    required: bool = True
    target_mastery: float = Field(gt=0, le=1)
    current_mastery: float = Field(default=0.0, ge=0, le=1)
    priority: int = Field(ge=1, le=5)  # 1 = highest
    status: ObjectiveStatus = ObjectiveStatus.NOT_STARTED
    prerequisites: list[str] = Field(default_factory=list)  # concept ids of objectives in the same curriculum
    evidence_required: int = Field(ge=1)
    remediated: bool = False  # planned after repeated failure: top priority, more evidence, prerequisites revisited
    remediation: list[str] = Field(default_factory=list)  # prerequisites to revisit first

    def structure(self) -> dict:
        """The fields that define the learning path (what versioning compares): no wording, no progress."""
        return self.model_dump(mode="json", include=set(STRUCTURAL_OBJECTIVE_FIELDS))

    @staticmethod
    def id_for(goal_id: str, concept_id: str) -> str:
        return stable_id("obj", goal_id, concept_id)


STRUCTURAL_OBJECTIVE_FIELDS = ("objective_id", "goal_id", "concept_id", "order", "role", "mode", "required",
                               "target_mastery", "priority", "prerequisites", "evidence_required", "remediated",
                               "remediation")


class CurriculumWarning(Schema):
    code: WarningCode
    message: str
    details: dict = Field(default_factory=dict)


class ProposedObjective(Schema):
    concept_id: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=400)
    rationale: str = Field(default="", max_length=600)


class CurriculumProposal(Schema):
    """What the learning-path planner model may propose: objectives (by concept id) with their wording, a suggested
    order and its pedagogical reasoning. Nothing else: no mastery, no status, no completion (unknown fields are
    rejected). Every proposal is validated by code before anything is stored."""

    objectives: list[ProposedObjective] = Field(min_length=1)
    suggested_order: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=2000)


class BriefConcept(Schema):
    concept_id: str
    name: str
    level: str | None = None
    description: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    role: ObjectiveRole
    state: Literal["mastered", "developing", "new"]  # a band, never the learner's numbers or history


class CurriculumBrief(Schema):
    """The minimum a model needs to word a curriculum: no learner id, goal id, evidence, notes or task ids."""

    domain: str
    goal_title: str
    goal_description: str = ""
    target_level: str | None = None
    language_of_instruction: str = "en"
    concepts: list[BriefConcept] = Field(min_length=1)


class ProposalReview(Schema):
    """What validation made of the model's proposal."""

    accepted: list[str] = Field(default_factory=list)  # concept ids whose wording was used
    unresolved: list[str] = Field(default_factory=list)  # not in the knowledge base: rejected
    rejected: dict[str, str] = Field(default_factory=dict)  # known concept -> why it was not used
    missing: list[str] = Field(default_factory=list)  # in scope but not worded: template wording used
    ordering_accepted: bool = True
    ordering_issues: list[str] = Field(default_factory=list)


class CurriculumDraft(Schema):
    """The deterministic curriculum before wording: scope, order, targets, priorities, evidence, warnings."""

    curriculum_id: str
    learner_id: str
    goal: LearningGoal
    objectives: list[CurriculumObjective] = Field(min_length=1)
    brief: CurriculumBrief
    warnings: list[CurriculumWarning] = Field(default_factory=list)
    config_fingerprint: str


class CurriculumPlan(Schema):
    """A validated curriculum ready to be stored as a version. `changed` is False when its content equals the
    current version's (then nothing new is stored)."""

    curriculum_id: str
    learner_id: str
    goal_id: str
    title: str
    goal: GoalSpec
    objectives: list[CurriculumObjective] = Field(min_length=1)
    content_hash: str
    version: int = Field(ge=1)
    version_id: str
    base_version: int | None = None
    changed: bool = True
    reasons: list[ReplanReason] = Field(default_factory=list)
    rationale: str = ""
    proposal_review: ProposalReview = Field(default_factory=ProposalReview)
    warnings: list[CurriculumWarning] = Field(default_factory=list)
    config_fingerprint: str


class CurriculumVersion(Schema):
    """An immutable curriculum definition. Never edited: a materially different path is a new version."""

    version_id: str
    curriculum_id: str
    learner_id: str
    goal_id: str
    version: int = Field(ge=1)
    parent_version: int | None = None
    content_hash: str
    goal: GoalSpec
    objectives: list[CurriculumObjective] = Field(min_length=1)
    reasons: list[ReplanReason] = Field(default_factory=list)
    rationale: str = ""
    proposal_review: ProposalReview = Field(default_factory=ProposalReview)
    warnings: list[CurriculumWarning] = Field(default_factory=list)
    config_fingerprint: str
    created_at: datetime
    task_id: str | None = None  # the planning (or evaluation) task that created it
    artifact_ids: dict[str, str] = Field(default_factory=dict)  # "version" and objective ids -> artifact ids

    @model_validator(mode="after")
    def _consistent(self) -> CurriculumVersion:
        problems = curriculum_consistency(self.goal_id, self.objectives)
        if problems:
            raise ValueError("; ".join(problems))
        if self.content_hash != content_hash(self.goal, self.objectives, self.config_fingerprint):
            raise ValueError("content_hash does not match the version's content")
        return self

    def objective(self, objective_id: str) -> CurriculumObjective:
        return next(o for o in self.objectives if o.objective_id == objective_id)

    def by_concept(self, concept_id: str) -> CurriculumObjective | None:
        return next((o for o in self.objectives if o.concept_id == concept_id), None)


class CurriculumRecord(Schema):
    """The current pointer of a goal's curriculum (stored); the versions themselves are immutable."""

    curriculum_id: str
    learner_id: str
    goal_id: str
    title: str
    domain: str
    current_version: int = Field(ge=1)
    current_version_id: str
    status: CurriculumStatus = CurriculumStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


# --- Progress, review and next action -------------------------------------------------------------------------------


class ReviewSchedule(Schema):
    concept_id: str
    next_review_at: datetime | None = None
    last_reviewed_at: datetime | None = None
    review_interval_days: float | None = None
    review_count: int = Field(default=0, ge=0)
    due: bool = False


class ObjectiveProgress(Schema):
    objective_id: str
    concept_id: str
    status: ObjectiveStatus
    current_mastery: float = Field(ge=0, le=1)
    target_mastery: float = Field(gt=0, le=1)
    evidence_count: int = Field(ge=0)
    evidence_required: int = Field(ge=1)
    incorrect_streak: int = Field(default=0, ge=0)
    unmet_prerequisites: list[str] = Field(default_factory=list)
    review: ReviewSchedule | None = None


class ReplanTrigger(Schema):
    reason: ReplanReason
    objective_id: str | None = None
    concept_id: str | None = None
    detail: str


class CurriculumProgress(Schema):
    """Progress of one curriculum version, derived from the learner model by code. Deterministic for the same
    learner state, version and `as_of`."""

    curriculum_id: str
    goal_id: str
    version: int
    objectives: list[ObjectiveProgress]
    mastered: int = Field(ge=0)
    required: int = Field(ge=0)
    percent_complete: float = Field(ge=0, le=100)
    completion_rule: CompletionRule
    goal_complete: bool
    triggers: list[ReplanTrigger] = Field(default_factory=list)
    as_of: datetime

    def of(self, objective_id: str) -> ObjectiveProgress:
        return next(p for p in self.objectives if p.objective_id == objective_id)

    def statuses(self) -> dict[str, ObjectiveStatus]:
        return {p.objective_id: p.status for p in self.objectives}


class Curriculum(Schema):
    """The current curriculum of a goal: the current version's objectives with their progress recomputed now."""

    curriculum_id: str
    learner_id: str
    goal_id: str
    title: str
    objectives: list[CurriculumObjective]
    version: int
    version_id: str
    versions: int = Field(ge=1)
    status: CurriculumStatus
    created_at: datetime
    updated_at: datetime
    progress: CurriculumProgress
    warnings: list[CurriculumWarning] = Field(default_factory=list)


class PriorityFactors(Schema):
    deficit: float = Field(ge=0, le=1)
    objective_priority: float = Field(ge=0, le=1)
    goal_priority: float = Field(ge=0, le=1)
    review_urgency: float = Field(ge=0, le=1)
    deadline_pressure: float = Field(ge=0, le=1)
    recent_failure: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)


class PriorityScore(Schema):
    """score = prerequisite_ready x sum(weight x factor) / sum(weights); see app/curriculum/priority.py."""

    factors: PriorityFactors
    prerequisite_ready: bool
    score: float = Field(ge=0, le=1)


class ActionCandidate(Schema):
    goal_id: str
    objective_id: str
    concept_id: str
    action: LearningActionType
    eligible: bool
    score: float = Field(ge=0, le=1)
    reason: str


class NextLearningAction(Schema):
    """What the learner should do next, selected by code from the curricula of their goals. The reason is
    deterministic text built from the inputs, never model-written."""

    action_id: str
    learner_id: str
    action: LearningActionType
    goal_id: str | None = None
    curriculum_id: str | None = None
    curriculum_version: int | None = None
    objective_id: str | None = None
    concept_id: str | None = None
    objective_description: str | None = None
    priority: PriorityScore | None = None
    reason: str = Field(min_length=1)
    candidates: list[ActionCandidate] = Field(default_factory=list)
    next_review_at: datetime | None = None  # WAIT: when something is next due
    as_of: datetime

    @model_validator(mode="after")
    def _shape(self) -> NextLearningAction:
        if self.action in LESSON_ACTIONS and (self.objective_id is None or self.goal_id is None):
            raise ValueError(f"{self.action.value} needs a goal and an objective")
        if self.action == LearningActionType.COMPLETE and self.goal_id is None:
            raise ValueError("COMPLETE names the goal that is complete")
        return self


class ProgressUpdate(Schema):
    """What one deterministic progress refresh changed for one curriculum (transitions, completion, replanning)."""

    curriculum_id: str
    goal_id: str
    progress: CurriculumProgress
    started: list[str] = Field(default_factory=list)  # objective ids that left NOT_STARTED / BLOCKED
    mastered: list[str] = Field(default_factory=list)  # objective ids that became MASTERED
    goal_completed: bool = False
    replanned: CurriculumVersion | None = None


class ProgressRefresh(Schema):
    """Progress of every curriculum of a learner in one domain after a mastery update."""

    updates: list[ProgressUpdate] = Field(default_factory=list)


# --- Integrity ----------------------------------------------------------------------------------------------------


class VersionConflict(ValueError):
    """A curriculum version id is already stored with different content: history is never rewritten."""



def content_hash(goal: GoalSpec, objectives: list[CurriculumObjective], config_fingerprint: str) -> str:
    """The identity of a learning path: goal definition (not its priority), objective structure, configuration.
    Model-written wording, progress and timestamps are not part of it."""
    body = {
        "goal": goal.model_dump(mode="json", include={"domain", "target_level", "target_concepts", "target_date"}),
        "objectives": [o.structure() for o in objectives],
        "config": config_fingerprint,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def version_id_for(curriculum_id: str, version: int, digest: str) -> str:
    return stable_id("curv", curriculum_id, str(version), digest)


def curriculum_id_for(goal_id: str) -> str:
    return stable_id("cur", goal_id)


def curriculum_consistency(goal_id: str, objectives: list[CurriculumObjective]) -> list[str]:
    """Structural invariants every stored curriculum satisfies (the full validation, against the knowledge base,
    is app/curriculum/validation.py)."""
    problems: list[str] = []
    ids = [o.objective_id for o in objectives]
    concepts = [o.concept_id for o in objectives]
    if len(set(ids)) != len(ids) or len(set(concepts)) != len(concepts):
        problems.append("objectives and their concepts must be unique")
    if [o.order for o in objectives] != list(range(1, len(objectives) + 1)):
        problems.append("objective order must be 1..n in list order")
    position = {o.concept_id: o.order for o in objectives}
    for o in objectives:
        if o.goal_id != goal_id:
            problems.append(f"objective {o.objective_id} belongs to another goal")
        if o.objective_id != CurriculumObjective.id_for(goal_id, o.concept_id):
            problems.append(f"objective {o.objective_id} does not have the deterministic id of its concept")
        for pre in o.prerequisites:
            if pre not in position:
                problems.append(f"{o.concept_id} requires {pre}, which is not an objective of the curriculum")
            elif position[pre] >= o.order:
                problems.append(f"{o.concept_id} is ordered before its prerequisite {pre}")
        if not set(o.remediation) <= set(o.prerequisites):
            problems.append(f"{o.concept_id} remediates concepts that are not its prerequisites")
    return problems
