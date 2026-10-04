"""Curriculum tools: the deterministic curriculum engine as workflow steps.

draft      goal + concept graph + learner model -> the curriculum's structure (scope, order, targets, warnings)
finalize   draft + the model's proposal (optional) -> a validated plan; unknown concepts and bad orders are rejected
save       plan + artifact ids -> the stored version (idempotent: a resumed task never stores a second version)
track      after a mastery update: progress of the learner's curricula, transitions, completion, replanning, and the
           next learning action (nothing for a learner without curricula)
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.artifacts.service import ArtifactService
from app.curriculum.engine import CurriculumConflict, CurriculumEngine
from app.curriculum.tracker import CurriculumTracker
from app.curriculum.validation import CurriculumValidationError
from app.learner.memory import UnknownGoal
from app.observability.scope import ExecutionScope
from app.pedagogy.graph import ConceptGraph
from app.schemas.common import Schema
from app.schemas.concepts import Concept
from app.schemas.curriculum import (
    CurriculumDraft,
    CurriculumPlan,
    CurriculumProposal,
    CurriculumVersion,
    NextLearningAction,
    ProgressUpdate,
    ReplanReason,
)
from app.schemas.learner import LearningGoal
from app.schemas.pedagogy import LearnerModel
from app.tools.base import Tool, ToolError
from app.tools.curriculum.artifacts import VERSION_KEY, artifact_ids, curriculum_drafts


class DraftRequest(Schema):
    goal: LearningGoal
    model: LearnerModel
    concepts: list[Concept]
    as_of: datetime  # fixed when the planning task is created, so a resumed task drafts the same curriculum
    language: str = "en"


class FinalizeRequest(Schema):
    draft: CurriculumDraft
    concepts: list[Concept]
    proposal: CurriculumProposal | None = None
    reasons: list[ReplanReason] = Field(default_factory=list)


class FinalizedPlan(Schema):
    plan: CurriculumPlan
    previous_artifact_id: str | None = None  # the current version's CURRICULUM_VERSION artifact, if any


class SaveRequest(Schema):
    plan: CurriculumPlan
    domain: str
    artifact_ids: dict[str, str] = Field(default_factory=dict)


class SaveResult(Schema):
    version: CurriculumVersion
    created: bool


class TrackRequest(Schema):
    learner_id: str
    domain: str
    concepts: list[Concept]
    as_of: datetime | None = None  # None: now


class TrackResult(Schema):
    updates: list[ProgressUpdate] = Field(default_factory=list)
    action: NextLearningAction | None = None  # None: the learner has no curriculum in this domain


class CurriculumDraftTool(Tool[DraftRequest, CurriculumDraft]):
    name = "curriculum.draft"
    description = "Deterministic curriculum structure for a goal: scope, prerequisite order, targets, warnings."
    input_model = DraftRequest
    output_model = CurriculumDraft
    permissions = frozenset({"learner:read"})

    def __init__(self, engine: CurriculumEngine) -> None:
        self._engine = engine

    async def run(self, data: DraftRequest, scope: ExecutionScope) -> CurriculumDraft:
        try:
            return self._engine.planner.draft(data.goal, ConceptGraph(data.concepts), data.model, as_of=data.as_of,
                                              language=data.language)
        except CurriculumValidationError as exc:
            raise ToolError(str(exc)) from exc


class CurriculumFinalizeTool(Tool[FinalizeRequest, FinalizedPlan]):
    name = "curriculum.finalize"
    description = "Validate a curriculum (and the model's wording) and decide whether it is a new version."
    input_model = FinalizeRequest
    output_model = FinalizedPlan
    permissions = frozenset({"learner:read"})

    def __init__(self, engine: CurriculumEngine) -> None:
        self._engine = engine

    async def run(self, data: FinalizeRequest, scope: ExecutionScope) -> FinalizedPlan:
        current = self._engine.current(data.draft.curriculum_id)
        try:
            plan = self._engine.planner.finalize(data.draft, ConceptGraph(data.concepts), proposal=data.proposal,
                                                 current=current, reasons=data.reasons or None)
        except CurriculumValidationError as exc:
            raise ToolError(str(exc)) from exc
        previous = current.artifact_ids.get(VERSION_KEY) if current is not None else None
        return FinalizedPlan(plan=plan, previous_artifact_id=previous)


class CurriculumSaveTool(Tool[SaveRequest, SaveResult]):
    name = "curriculum.save"
    description = "Store a validated curriculum as the goal's current version (idempotent)."
    input_model = SaveRequest
    output_model = SaveResult
    permissions = frozenset({"learner:write"})

    def __init__(self, engine: CurriculumEngine) -> None:
        self._engine = engine

    async def run(self, data: SaveRequest, scope: ExecutionScope) -> SaveResult:
        try:
            version, created = self._engine.commit(data.plan, domain=data.domain, task_id=scope.task_id,
                                                   artifact_ids=data.artifact_ids, scope=scope)
        except CurriculumConflict as exc:
            raise ToolError(str(exc)) from exc
        return SaveResult(version=version, created=created)


class CurriculumTrackTool(Tool[TrackRequest, TrackResult]):
    name = "curriculum.track"
    description = ("After a mastery update: curriculum progress, objective transitions, goal completion, "
                   "deterministic replanning and the next learning action.")
    input_model = TrackRequest
    output_model = TrackResult
    permissions = frozenset({"learner:write"})

    def __init__(self, tracker: CurriculumTracker, artifacts: ArtifactService) -> None:
        self._tracker = tracker
        self._artifacts = artifacts

    async def run(self, data: TrackRequest, scope: ExecutionScope) -> TrackResult:
        as_of = data.as_of or self._tracker.engine.now()
        graph = ConceptGraph(data.concepts)

        def store(plan: CurriculumPlan, previous: CurriculumVersion) -> dict[str, str]:
            if scope.task_id is None:
                return {}
            goal = self._tracker.memory.goal(plan.goal_id)
            drafts = curriculum_drafts(plan, goal, previous.artifact_ids.get(VERSION_KEY))
            stored = self._artifacts.store_batch(scope.task_id, drafts, scope)
            return artifact_ids(plan, stored)

        try:
            tracked = self._tracker.tracked(data.learner_id, as_of, graphs={data.domain: graph}, domain=data.domain,
                                            task_id=scope.task_id, scope=scope, artifacts=store)
        except (CurriculumValidationError, CurriculumConflict, UnknownGoal) as exc:
            raise ToolError(str(exc)) from exc
        if not tracked.entries:
            return TrackResult()
        return TrackResult(updates=tracked.updates, action=self._tracker.next_action(data.learner_id, tracked, scope))
