"""NextActionEngine: the learner's curricula + learner models -> one NextLearningAction, deterministically.

Per objective of every curriculum (goals in priority order):

    MASTERED, review due                 -> REVIEW candidate
    MASTERED, not due                    -> no candidate
    BLOCKED (a prerequisite not ready)   -> not eligible (reported, never selected)
    mastery >= target, evidence missing  -> EVALUATE
    incorrect streak >= threshold        -> LEARN (reteach)
    mastery >= practice_from             -> PRACTICE
    otherwise                            -> LEARN

Active goals contribute every candidate; completed goals only REVIEW candidates (mastered concepts keep coming
back for review); paused and cancelled goals none. The eligible candidate with the highest priority score
(app/curriculum/priority.py) wins; ties go to the higher-priority goal, then the earlier objective, then ids.
An active goal whose completion rule holds yields COMPLETE first. With nothing eligible the action is WAIT, with the
next review date when there is one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from app.curriculum.priority import priority_factors, priority_score
from app.schemas.curriculum import (
    ActionCandidate,
    CurriculumConfig,
    CurriculumObjective,
    CurriculumProgress,
    CurriculumVersion,
    LearningActionType,
    NextLearningAction,
    ObjectiveProgress,
    ObjectiveStatus,
    PriorityScore,
)
from app.schemas.learner import GoalStatus, LearningGoal, stable_id
from app.schemas.pedagogy import LearnerModel


@dataclass(frozen=True)
class GoalCurriculum:
    """One goal with its current curriculum version and that version's progress."""

    goal: LearningGoal
    version: CurriculumVersion
    progress: CurriculumProgress


@dataclass(frozen=True)
class _Scored:
    entry: GoalCurriculum
    objective: CurriculumObjective
    action: LearningActionType
    score: PriorityScore
    reason: str


def classify(objective: CurriculumObjective, progress: ObjectiveProgress,
             config: CurriculumConfig) -> LearningActionType | None:
    """The action an objective calls for now (None: nothing to do on it)."""
    if progress.status == ObjectiveStatus.MASTERED:
        return LearningActionType.REVIEW if progress.review is not None and progress.review.due else None
    if progress.current_mastery >= objective.target_mastery:
        return LearningActionType.EVALUATE
    if progress.incorrect_streak >= config.repeated_failure_streak:
        return LearningActionType.LEARN
    if progress.current_mastery >= config.practice_from:
        return LearningActionType.PRACTICE
    return LearningActionType.LEARN


class NextActionEngine:
    def __init__(self, config: CurriculumConfig | None = None) -> None:
        self.config = config or CurriculumConfig()

    def select(self, learner_id: str, entries: list[GoalCurriculum], models: Mapping[str, LearnerModel],
               as_of: datetime, *, completed_now: frozenset[str] = frozenset()) -> NextLearningAction:
        """`models` maps a domain to the learner model in it. `completed_now`: goals whose completion rule just
        fired (their COMPLETE is reported even though the goal is no longer active)."""
        ordered = sorted(entries, key=lambda e: (e.goal.priority, e.goal.goal_id))
        for e in ordered:
            if e.goal.goal_id in completed_now or (e.goal.status == GoalStatus.ACTIVE and e.progress.goal_complete):
                return self._action(learner_id, LearningActionType.COMPLETE, as_of, entry=e, reason=(
                    f"Every {'required' if self.config.completion_rule == 'all_required_mastered' else 'target'} "
                    f"objective of {e.version.goal.title!r} is mastered "
                    f"({e.progress.mastered}/{e.progress.required}): the goal is complete."))
        scored: list[_Scored] = []
        blocked: list[ActionCandidate] = []
        for e in ordered:
            if e.goal.status not in (GoalStatus.ACTIVE, GoalStatus.COMPLETED):
                continue
            model = models[e.goal.domain]
            for o in e.version.objectives:
                p = e.progress.of(o.objective_id)
                if p.status == ObjectiveStatus.BLOCKED:
                    if e.goal.status == GoalStatus.ACTIVE:
                        blocked.append(ActionCandidate(
                            goal_id=e.goal.goal_id, objective_id=o.objective_id, concept_id=o.concept_id,
                            action=classify(o, p.model_copy(update={"status": ObjectiveStatus.IN_PROGRESS}),
                                            self.config) or LearningActionType.LEARN,
                            eligible=False, score=0.0,
                            reason=f"blocked: needs {', '.join(p.unmet_prerequisites)} at "
                                   f"{self.config.prerequisite_threshold:.2f} first"))
                    continue
                action = classify(o, p, self.config)
                if action is None or (e.goal.status == GoalStatus.COMPLETED and action != LearningActionType.REVIEW):
                    continue
                factors = priority_factors(o, p, model.state(o.concept_id), e.goal.priority, e.goal.target_date,
                                           as_of, self.config, reviewing=action == LearningActionType.REVIEW)
                score = priority_score(factors, True, self.config)
                scored.append(_Scored(e, o, action, score, self._reason(e, o, p, action, score)))
        scored.sort(key=lambda s: (-s.score.score, s.entry.goal.priority, s.entry.goal.goal_id, s.objective.order,
                                   s.objective.objective_id))
        candidates = [ActionCandidate(goal_id=s.entry.goal.goal_id, objective_id=s.objective.objective_id,
                                      concept_id=s.objective.concept_id, action=s.action, eligible=True,
                                      score=s.score.score, reason=s.reason) for s in scored]
        report = (candidates + blocked)[:self.config.max_candidates_reported]
        if scored:
            best = scored[0]
            return self._action(learner_id, best.action, as_of, entry=best.entry, objective=best.objective,
                                score=best.score, reason=best.reason, candidates=report)
        upcoming = sorted(p.review.next_review_at for e in ordered for p in e.progress.objectives
                          if p.review is not None and p.review.next_review_at is not None
                          and p.status == ObjectiveStatus.MASTERED and e.goal.status in (GoalStatus.ACTIVE,
                                                                                          GoalStatus.COMPLETED))
        active = [e for e in ordered if e.goal.status == GoalStatus.ACTIVE]
        if not active:
            reason = "No active goal has a curriculum." if not entries else "No active goal: nothing to learn now."
        elif blocked:
            reason = "Every remaining objective waits on a prerequisite that is not ready."
        else:
            reason = "Nothing is due now."
        if upcoming:
            reason += f" Next review due {upcoming[0].isoformat()}."
        return self._action(learner_id, LearningActionType.WAIT, as_of, reason=reason, candidates=report,
                            next_review_at=upcoming[0] if upcoming else None)

    def _reason(self, e: GoalCurriculum, o: CurriculumObjective, p: ObjectiveProgress, action: LearningActionType,
                score: PriorityScore) -> str:
        f = score.factors
        what = {
            LearningActionType.LEARN: (f"mastery {p.current_mastery:.2f} is below the practice threshold "
                                       f"{self.config.practice_from:.2f}" if p.incorrect_streak
                                       < self.config.repeated_failure_streak
                                       else f"{p.incorrect_streak} incorrect answers in a row: reteach"),
            LearningActionType.PRACTICE: (f"mastery {p.current_mastery:.2f} is close to the target "
                                          f"{o.target_mastery:.2f}: apply it"),
            LearningActionType.EVALUATE: (f"mastery {p.current_mastery:.2f} reaches the target but only "
                                          f"{p.evidence_count}/{o.evidence_required} pieces of evidence: assess it"),
            LearningActionType.REVIEW: "mastered and due for review",
        }[action]
        prereqs = f"prerequisites ready ({', '.join(o.prerequisites)})" if o.prerequisites else "no prerequisites"
        return (f"{action.value} {o.name} for goal {e.version.goal.title!r}: {what}; {prereqs}; priority "
                f"{score.score:.3f} (deficit {f.deficit:.2f}, objective {f.objective_priority:.2f}, goal "
                f"{f.goal_priority:.2f}, review {f.review_urgency:.2f}, deadline {f.deadline_pressure:.2f}, "
                f"failure {f.recent_failure:.2f}, recency {f.recency:.2f}).")

    @staticmethod
    def _action(learner_id: str, action: LearningActionType, as_of: datetime, *, entry: GoalCurriculum | None = None,
                objective: CurriculumObjective | None = None, score: PriorityScore | None = None, reason: str,
                candidates: list[ActionCandidate] | None = None,
                next_review_at: datetime | None = None) -> NextLearningAction:
        goal_id = entry.goal.goal_id if entry else None
        return NextLearningAction(
            action_id=stable_id("act", learner_id, action.value, goal_id or "",
                                objective.objective_id if objective else "", as_of.isoformat()),
            learner_id=learner_id, action=action, goal_id=goal_id,
            curriculum_id=entry.version.curriculum_id if entry else None,
            curriculum_version=entry.version.version if entry else None,
            objective_id=objective.objective_id if objective else None,
            concept_id=objective.concept_id if objective else None,
            objective_description=objective.description if objective else None, priority=score, reason=reason,
            candidates=candidates or [], next_review_at=next_review_at, as_of=as_of)
