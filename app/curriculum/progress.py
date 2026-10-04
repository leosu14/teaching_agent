"""Objective progress, curriculum progress, goal completion and replanning triggers: all derived from the learner
model by code. Mastery is read, never written; the same learner state, version and `as_of` give the same progress.

Objective status (in this order):

    MASTERED     mastery >= target_mastery and evidence_count >= evidence_required
    BLOCKED      a prerequisite objective's concept is below the prerequisite threshold
    IN_PROGRESS  the concept has evidence or was taught (an exposure)
    NOT_STARTED  otherwise

Completion rules (`CurriculumConfig.completion_rule`):

    all_required_mastered  every required objective is MASTERED (default)
    targets_mastered       every objective whose concept is a goal target is MASTERED

Replanning triggers (a trigger makes a new version only if the replanned path differs):

    repeated_failure         an unmastered, not yet remediated objective has an incorrect streak >= the threshold
    prerequisite_regression  an objective planned as already mastered (maintain) that other objectives rely on is
                             no longer mastered
    early_mastery            an objective planned to be learned is mastered before one of its prerequisites
Goal and target-date changes trigger replanning where the goal changes (app/services/curriculum.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from app.curriculum.review import MasteryReviewPolicy, ReviewPolicy
from app.schemas.curriculum import (
    CompletionRule,
    CurriculumConfig,
    CurriculumObjective,
    CurriculumProgress,
    CurriculumVersion,
    ObjectiveProgress,
    ObjectiveStatus,
    ReplanReason,
    ReplanTrigger,
)
from app.schemas.pedagogy import LearnerModel


def objective_status(objective: CurriculumObjective, model: LearnerModel,
                     config: CurriculumConfig) -> tuple[ObjectiveStatus, list[str]]:
    """(status, unmet prerequisites) of one objective in the learner's current state."""
    state = model.state(objective.concept_id)
    mastery = model.mastery_of(objective.concept_id)
    evidence = state.evidence_count if state is not None else 0
    unmet = [p for p in objective.prerequisites if model.mastery_of(p) < config.prerequisite_threshold]
    if mastery >= objective.target_mastery and evidence >= objective.evidence_required:
        return ObjectiveStatus.MASTERED, []
    if unmet:
        return ObjectiveStatus.BLOCKED, unmet
    if evidence or (state is not None and state.exposures):
        return ObjectiveStatus.IN_PROGRESS, []
    return ObjectiveStatus.NOT_STARTED, []


def goal_complete(objectives: list[CurriculumObjective], statuses: Mapping[str, ObjectiveStatus],
                  rule: CompletionRule) -> bool:
    if rule == "targets_mastered":
        relevant = [o for o in objectives if o.role == "target"]
    else:
        relevant = [o for o in objectives if o.required]
    return bool(relevant) and all(statuses[o.objective_id] == ObjectiveStatus.MASTERED for o in relevant)


def compute_progress(version: CurriculumVersion, model: LearnerModel, config: CurriculumConfig, as_of: datetime,
                     review_counts: Mapping[str, int] | None = None,
                     review_policy: ReviewPolicy | None = None) -> CurriculumProgress:
    policy = review_policy or MasteryReviewPolicy()
    counts = review_counts or {}
    items: list[ObjectiveProgress] = []
    for o in version.objectives:
        status, unmet = objective_status(o, model, config)
        state = model.state(o.concept_id)
        items.append(ObjectiveProgress(
            objective_id=o.objective_id, concept_id=o.concept_id, status=status,
            current_mastery=model.mastery_of(o.concept_id), target_mastery=o.target_mastery,
            evidence_count=state.evidence_count if state is not None else 0, evidence_required=o.evidence_required,
            incorrect_streak=state.incorrect_streak if state is not None else 0, unmet_prerequisites=unmet,
            review=policy.schedule(o.concept_id, state if state is not None and state.evidence_count else None,
                                   counts.get(o.concept_id, 0), as_of)))
    statuses = {p.objective_id: p.status for p in items}
    relevant = [o for o in version.objectives if (o.role == "target" if config.completion_rule == "targets_mastered"
                                                  else o.required)]
    mastered = sum(1 for o in relevant if statuses[o.objective_id] == ObjectiveStatus.MASTERED)
    progress = CurriculumProgress(
        curriculum_id=version.curriculum_id, goal_id=version.goal_id, version=version.version, objectives=items,
        mastered=mastered, required=len(relevant),
        percent_complete=round(100 * mastered / len(relevant), 2) if relevant else 0.0,
        completion_rule=config.completion_rule,
        goal_complete=goal_complete(version.objectives, statuses, config.completion_rule), as_of=as_of)
    return progress.model_copy(update={"triggers": replan_triggers(version, progress, config)})


def replan_triggers(version: CurriculumVersion, progress: CurriculumProgress,
                    config: CurriculumConfig) -> list[ReplanTrigger]:
    """Deterministic replanning triggers from the learner state (none once the goal is complete)."""
    if progress.goal_complete:
        return []
    statuses = progress.statuses()
    by_concept = {o.concept_id: o for o in version.objectives}
    relied_on = {pre for o in version.objectives for pre in o.prerequisites}
    triggers: list[ReplanTrigger] = []
    for o in version.objectives:
        p = progress.of(o.objective_id)
        if (p.status != ObjectiveStatus.MASTERED and not o.remediated
                and p.incorrect_streak >= config.repeated_failure_streak):
            triggers.append(ReplanTrigger(
                reason=ReplanReason.REPEATED_FAILURE, objective_id=o.objective_id, concept_id=o.concept_id,
                detail=f"{o.name}: {p.incorrect_streak} incorrect answers in a row"))
        if o.mode == "maintain" and p.status != ObjectiveStatus.MASTERED and o.concept_id in relied_on:
            triggers.append(ReplanTrigger(
                reason=ReplanReason.PREREQUISITE_REGRESSION, objective_id=o.objective_id, concept_id=o.concept_id,
                detail=f"{o.name} was planned as mastered but is at {p.current_mastery:.2f}"))
        if o.mode == "learn" and p.status == ObjectiveStatus.MASTERED:
            pending = [pre for pre in o.prerequisites
                       if statuses[by_concept[pre].objective_id] != ObjectiveStatus.MASTERED]
            if pending:
                triggers.append(ReplanTrigger(
                    reason=ReplanReason.EARLY_MASTERY, objective_id=o.objective_id, concept_id=o.concept_id,
                    detail=f"{o.name} is mastered before its prerequisites {', '.join(pending)}"))
    return triggers
