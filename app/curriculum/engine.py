"""CurriculumEngine: the only code that changes curriculum state (versions, the current pointer, progress snapshots).

Every change is deterministic and idempotent:

- `commit` stores a validated plan as a new immutable version (only when its content differs from the current
  version) and moves the pointer; re-committing the same plan (a resumed task) changes nothing and emits nothing;
- `refresh` recomputes a curriculum's progress from the learner model, reports objectives that started or were
  mastered since the last snapshot (once), applies the completion rule and, when a replanning trigger fires,
  replans deterministically (a new version only when the path changes; wording is carried over).

The goal's own status lives with the learner's goals: `refresh` reports `goal_completed`, the caller stores it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

from app.curriculum.planner import CurriculumPlanner
from app.curriculum.progress import compute_progress
from app.curriculum.repository import CurriculumRepository
from app.curriculum.review import ReviewPolicy
from app.observability.scope import ExecutionScope
from app.pedagogy.graph import ConceptGraph
from app.schemas.common import utcnow
from app.schemas.curriculum import (
    CurriculumConfig,
    CurriculumPlan,
    CurriculumProgress,
    CurriculumRecord,
    CurriculumStatus,
    CurriculumVersion,
    ObjectiveStatus,
    ProgressUpdate,
    ReplanReason,
    curriculum_id_for,
)
from app.schemas.events import EventType
from app.schemas.learner import LearningGoal
from app.schemas.pedagogy import LearnerModel


# Stores a replanned version's artifacts (plan, previous version) and returns their ids by key.
ArtifactWriter = Callable[[CurriculumPlan, CurriculumVersion], dict[str, str]]


class CurriculumConflict(ValueError):
    """The curriculum moved on while a plan was being made (its base version is no longer current)."""


class CurriculumEngine:
    def __init__(self, repository: CurriculumRepository, config: CurriculumConfig | None = None, *,
                 review_policy: ReviewPolicy | None = None, clock: Callable[[], datetime] = utcnow) -> None:
        self.repository = repository
        self.config = config or CurriculumConfig()
        self.planner = CurriculumPlanner(self.config)
        self.review_policy = review_policy
        self._clock = clock

    # --- reads ------------------------------------------------------------------------------------------------

    def now(self) -> datetime:
        return self._clock()

    def record(self, curriculum_id: str) -> CurriculumRecord | None:
        return self.repository.record(curriculum_id)

    def current(self, curriculum_id: str) -> CurriculumVersion | None:
        record = self.repository.record(curriculum_id)
        return self.repository.version(record.current_version_id) if record is not None else None

    def versions(self, curriculum_id: str) -> list[CurriculumVersion]:
        return self.repository.versions(curriculum_id)

    def progress(self, version: CurriculumVersion, model: LearnerModel, as_of: datetime,
                 review_counts: Mapping[str, int] | None = None) -> CurriculumProgress:
        return compute_progress(version, model, self.config, as_of, review_counts, self.review_policy)

    # --- writes -----------------------------------------------------------------------------------------------

    def commit(self, plan: CurriculumPlan, *, domain: str, task_id: str | None, artifact_ids: dict[str, str],
               scope: ExecutionScope | None = None) -> tuple[CurriculumVersion, bool]:
        """Store the plan as the current version. Returns (version, created)."""
        record = self.repository.record(plan.curriculum_id)
        current = self.repository.version(record.current_version_id) if record is not None else None
        if not plan.changed:
            assert current is not None and current.version_id == plan.version_id
            return current, False
        if current is not None and current.version_id == plan.version_id:
            return current, False  # already committed (a resumed task)
        if (current.version if current is not None else None) != plan.base_version:
            raise CurriculumConflict(f"curriculum {plan.curriculum_id} is at version "
                                     f"{current.version if current else None}, the plan was made on "
                                     f"{plan.base_version}; rebuild it")
        now = self._clock()
        version = CurriculumVersion(
            version_id=plan.version_id, curriculum_id=plan.curriculum_id, learner_id=plan.learner_id,
            goal_id=plan.goal_id, version=plan.version, parent_version=plan.base_version,
            content_hash=plan.content_hash, goal=plan.goal, objectives=plan.objectives, reasons=plan.reasons,
            rationale=plan.rationale, proposal_review=plan.proposal_review, warnings=plan.warnings,
            config_fingerprint=plan.config_fingerprint, created_at=now, task_id=task_id, artifact_ids=artifact_ids)
        created = self.repository.add_version(version)
        stored = self.repository.version(version.version_id)
        assert stored is not None
        self.repository.save_record(CurriculumRecord(
            curriculum_id=plan.curriculum_id, learner_id=plan.learner_id, goal_id=plan.goal_id, title=plan.title,
            domain=domain, current_version=stored.version, current_version_id=stored.version_id,
            status=record.status if record is not None else CurriculumStatus.ACTIVE,
            created_at=record.created_at if record is not None else now, updated_at=now))
        if created and scope is not None:
            event = EventType.CURRICULUM_CREATED if plan.base_version is None else EventType.CURRICULUM_REPLANNED
            scope.emit(event, goal_id=plan.goal_id, curriculum_id=plan.curriculum_id, version=stored.version,
                       version_id=stored.version_id, objectives=[o.concept_id for o in stored.objectives],
                       reasons=[r.value for r in stored.reasons], warnings=[w.code for w in stored.warnings])
        return stored, created

    def set_status(self, curriculum_id: str, status: CurriculumStatus) -> None:
        record = self.repository.record(curriculum_id)
        if record is not None and record.status != status:
            self.repository.save_record(record.model_copy(update={"status": status, "updated_at": self._clock()}))

    def refresh(self, goal: LearningGoal, model: LearnerModel, as_of: datetime, *,
                review_counts: Mapping[str, int] | None = None, graph: ConceptGraph | None = None,
                task_id: str | None = None, scope: ExecutionScope | None = None,
                artifacts: ArtifactWriter | None = None) -> ProgressUpdate | None:
        """Progress of the goal's current curriculum, with transitions reported once. With `graph`, a fired
        replanning trigger replans the curriculum (the learner model is read, never written); `artifacts` stores a
        replanned version's artifacts and returns their ids (before the version is committed)."""
        curriculum_id = curriculum_id_for(goal.goal_id)
        version = self.current(curriculum_id)
        if version is None:
            return None
        progress = self.progress(version, model, as_of, review_counts)
        replanned = None
        if progress.triggers and graph is not None and goal.is_active:
            replanned = self._replan(goal, version, model, graph, progress, as_of, task_id, scope, artifacts)
            if replanned is not None:
                version = replanned
                progress = self.progress(version, model, as_of, review_counts)
        previous = self.repository.progress(curriculum_id)
        # Without a snapshot yet, what the curriculum was planned as already mastered is not news.
        before = previous.statuses() if previous is not None else {
            o.objective_id: ObjectiveStatus.MASTERED for o in version.objectives if o.mode == "maintain"}
        started, mastered = [], []
        for p in progress.objectives:
            old = before.get(p.objective_id, ObjectiveStatus.NOT_STARTED)
            if p.status == ObjectiveStatus.IN_PROGRESS and old in (ObjectiveStatus.NOT_STARTED,
                                                                   ObjectiveStatus.BLOCKED):
                started.append(p.objective_id)
            if p.status == ObjectiveStatus.MASTERED and old != ObjectiveStatus.MASTERED:
                mastered.append(p.objective_id)
        completed = progress.goal_complete and goal.is_active
        self.repository.save_progress(progress)
        if completed:
            self.set_status(curriculum_id, CurriculumStatus.COMPLETED)
        if scope is not None:
            for oid in started:
                o = version.objective(oid)
                scope.emit(EventType.OBJECTIVE_STARTED, goal_id=goal.goal_id, curriculum_id=curriculum_id,
                           objective_id=oid, concept_id=o.concept_id)
            for oid in mastered:
                o, p = version.objective(oid), progress.of(oid)
                scope.emit(EventType.OBJECTIVE_MASTERED, goal_id=goal.goal_id, curriculum_id=curriculum_id,
                           objective_id=oid, concept_id=o.concept_id, mastery=p.current_mastery,
                           evidence=p.evidence_count)
            if completed:
                scope.emit(EventType.GOAL_COMPLETED, goal_id=goal.goal_id, curriculum_id=curriculum_id,
                           version=version.version, rule=self.config.completion_rule,
                           objectives=progress.required)
        return ProgressUpdate(curriculum_id=curriculum_id, goal_id=goal.goal_id, progress=progress,
                              started=started, mastered=mastered, goal_completed=completed, replanned=replanned)

    def _replan(self, goal: LearningGoal, version: CurriculumVersion, model: LearnerModel, graph: ConceptGraph,
                progress: CurriculumProgress, as_of: datetime, task_id: str | None,
                scope: ExecutionScope | None, artifacts: ArtifactWriter | None) -> CurriculumVersion | None:
        reasons = list(dict.fromkeys(t.reason for t in progress.triggers))
        draft = self.planner.draft(goal, graph, model, as_of=as_of)
        before = {o.objective_id: o for o in version.objectives}
        # Wording is carried over where the objective's treatment did not change; changed ones get the template.
        carried = {o.objective_id: before[o.objective_id].description for o in draft.objectives
                   if o.objective_id in before and (o.mode, o.remediated) == (before[o.objective_id].mode,
                                                                              before[o.objective_id].remediated)}
        plan = self.planner.finalize(draft, graph, carried=carried, current=version,
                                     reasons=reasons or [ReplanReason.REBUILD])
        if not plan.changed:
            return None
        ids = artifacts(plan, version) if artifacts is not None else {}
        stored, _ = self.commit(plan, domain=goal.domain, task_id=task_id, artifact_ids=ids, scope=scope)
        return stored
