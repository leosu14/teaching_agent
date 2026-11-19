"""CurriculumTracker: a learner's goals and curricula against their learner memory.

It reads the learner model (never writes it), refreshes the progress of every goal that has a curriculum (the
engine reports transitions once, applies the completion rule and replans on a trigger), stores a goal the
completion rule completed, and selects the next learning action across all goals. Goals without a curriculum are
ignored: a learner without curricula keeps the adaptive lesson loop exactly as before.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.curriculum.engine import ArtifactWriter, CurriculumEngine
from app.curriculum.next_action import GoalCurriculum, NextActionEngine
from app.learner.memory import LearnerMemoryService, UnknownGoal
from app.observability.scope import ExecutionScope
from app.pedagogy.graph import ConceptGraph
from app.schemas.curriculum import (
    CurriculumConfig,
    CurriculumObjective,
    CurriculumStatus,
    CurriculumVersion,
    NextLearningAction,
    ProgressUpdate,
    curriculum_id_for,
)
from app.schemas.events import EventType
from app.schemas.learner import GoalStatus, LearningEvidence, LearningGoal
from app.schemas.lease import LearnerDataGone
from app.schemas.pedagogy import LearnerModel

DEFAULT_FRAMEWORK = "mastery"  # the generic level scale, for a subject the learner's profile does not list
FRAMEWORK_KEY = "framework_id"  # goal metadata key: the level framework a goal's target level belongs to


def _mastered(state, objective: CurriculumObjective) -> bool:
    """`objective_status`'s MASTERED test (app/curriculum/progress.py) on one concept state."""
    return state.mastery >= objective.target_mastery and state.evidence_count >= objective.evidence_required


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


@dataclass
class TrackedGoals:
    """The learner's goals that have a curriculum, refreshed at `as_of`."""

    as_of: datetime
    entries: list[GoalCurriculum] = field(default_factory=list)
    models: dict[str, LearnerModel] = field(default_factory=dict)  # by domain
    updates: list[ProgressUpdate] = field(default_factory=list)
    completed_now: frozenset[str] = frozenset()


def completion_concepts(version: CurriculumVersion, config: CurriculumConfig) -> set[str]:
    """The concepts whose objectives the completion rule counts (app/curriculum/progress.py `goal_complete`)."""
    return {o.concept_id for o in counted_objectives(version, config)}


def counted_objectives(version: CurriculumVersion, config: CurriculumConfig) -> list[CurriculumObjective]:
    if config.completion_rule == "targets_mastered":
        return [o for o in version.objectives if o.role == "target"]
    return [o for o in version.objectives if o.required]


def completion_time(evidence: Iterable[LearningEvidence], concepts: set[str], as_of: datetime) -> datetime:
    """The timestamp of the latest evidence on the counted concepts recorded up to `as_of` (`as_of` without any): the
    fallback of `CurriculumTracker.completion_time` when the replay cannot tell."""
    times = [_aware(e.timestamp) for e in evidence if e.concept_id in concepts and _aware(e.timestamp) <= as_of]
    return max(times, default=as_of)


class CurriculumTracker:
    def __init__(self, memory: LearnerMemoryService, engine: CurriculumEngine,
                 actions: NextActionEngine | None = None) -> None:
        self.memory = memory
        self.engine = engine
        self.actions = actions or NextActionEngine(engine.config)

    def framework_for(self, goal: LearningGoal) -> str:
        return self.memory.framework_for(goal.learner_id, goal.domain,
                                         goal.metadata.get(FRAMEWORK_KEY) or DEFAULT_FRAMEWORK)

    def model(self, goal: LearningGoal, universe: list[str], as_of: datetime) -> LearnerModel:
        return self.memory.learner_model(goal.learner_id, goal.domain, self.framework_for(goal), goal.target_level,
                                         universe, as_of=as_of)

    def completion_time(self, goal: LearningGoal, version: CurriculumVersion, as_of: datetime) -> datetime:
        """When the learner first reached the mastery that completes the goal (D11): the timestamp of the earliest
        piece of evidence after which the completion rule held. Mastery is replayed forward from the learner's
        evidence and exposures (the MasteryUpdater's own rebuild), so this is the learner's moment: never the time a
        refresh noticed it (a refresh long after the deadline does not make a goal mastered in time late), and never
        moved forward by practice after it (nor by a dip and a regain before the refresh). Never after `as_of`. When
        the replay never shows the rule holding (a state not rebuilt from evidence alone), the latest evidence's time
        up to `as_of`."""
        objectives = counted_objectives(version, self.engine.config)
        concepts = {o.concept_id for o in objectives}
        if objectives:
            for at, states in self.memory.mastery_history(goal.learner_id, concepts, as_of):
                if all(_mastered(states[o.concept_id], o) for o in objectives):
                    return at
        return _aware(completion_time(self.memory.evidence(goal.learner_id), concepts, as_of))

    def complete(self, goal: LearningGoal, version: CurriculumVersion, as_of: datetime) -> LearningGoal:
        """Store a goal the completion rule (mastery, D8) completed: COMPLETED, and `completed_at` set once, to when
        the learner reached that mastery (`completion_time`, D11), never to the time this refresh noticed. Only those
        fields change, in one atomic update of the stored goal, so a concurrent change of another field (a teacher's
        new deadline) is not lost. A goal no longer ACTIVE meanwhile (paused, cancelled) is left alone."""
        at = self.completion_time(goal, version, as_of)

        def done(current: LearningGoal) -> LearningGoal | None:
            if current.status != GoalStatus.ACTIVE:
                return None
            return current.model_copy(update={"status": GoalStatus.COMPLETED, "updated_at": as_of,
                                              "completed_at": current.completed_at or at})

        try:
            return self.memory.update_goal(goal.goal_id, done)
        except UnknownGoal:  # deleted since it was read: with the learner's data
            raise LearnerDataGone() from None

    @staticmethod
    def review_counts(model: LearnerModel, evidence_required: Mapping[str, int]) -> dict[str, int]:
        """Reviews of a concept: the pieces of evidence recorded beyond what mastering it required."""
        counts = {}
        for cid, required in evidence_required.items():
            state = model.state(cid)
            counts[cid] = max(0, (state.evidence_count if state else 0) - required)
        return counts

    def tracked(self, learner_id: str, as_of: datetime, *, graphs: Mapping[str, ConceptGraph],
                domain: str | None = None, replan: bool = True, task_id: str | None = None,
                scope: ExecutionScope | None = None, artifacts: ArtifactWriter | None = None) -> TrackedGoals:
        """Refresh every goal of the learner (in `domain`, if given) that has a curriculum. `graphs` maps a domain
        to its concept graph (replanning needs it; with `replan=False` no trigger replans)."""
        result = TrackedGoals(as_of=as_of)
        completed: set[str] = set()
        for goal in self.memory.goals(learner_id, domain):
            if self.engine.current(curriculum_id_for(goal.goal_id)) is None:
                continue
            graph = graphs.get(goal.domain)
            model = result.models.get(goal.domain)
            if model is None:
                model = self.model(goal, graph.ids if graph is not None else [], as_of)
                result.models[goal.domain] = model
            version = self.engine.current(curriculum_id_for(goal.goal_id))
            if version is None:  # curricula are deleted only with their learner's data: deleted since the goal read
                raise LearnerDataGone()
            counts = self.review_counts(model, {o.concept_id: o.evidence_required for o in version.objectives})
            update = self.engine.refresh(goal, model, as_of, review_counts=counts,
                                         graph=graph if replan else None, task_id=task_id, scope=scope,
                                         artifacts=artifacts)
            if update is None:  # its curriculum was deleted since it was read: with the learner's data
                raise LearnerDataGone()
            if update.goal_completed:
                goal = self.complete(goal, version, as_of)
                if goal.status == GoalStatus.COMPLETED:
                    completed.add(goal.goal_id)
                else:  # paused or cancelled meanwhile: not completed, and its curriculum follows the goal
                    self.engine.set_status(curriculum_id_for(goal.goal_id), CurriculumStatus.ARCHIVED
                                           if goal.status == GoalStatus.CANCELLED else CurriculumStatus.ACTIVE)
            current = self.engine.current(curriculum_id_for(goal.goal_id))
            if current is None:  # as above
                raise LearnerDataGone()
            result.updates.append(update)
            result.entries.append(GoalCurriculum(goal=goal, version=current, progress=update.progress))
        result.completed_now = frozenset(completed)
        return result

    def next_action(self, learner_id: str, tracked: TrackedGoals,
                    scope: ExecutionScope | None = None) -> NextLearningAction:
        action = self.actions.select(learner_id, tracked.entries, tracked.models, tracked.as_of,
                                     completed_now=tracked.completed_now)
        if scope is not None:
            scope.emit(EventType.LEARNING_ACTION_SELECTED, action_id=action.action_id, action=action.action.value,
                       goal_id=action.goal_id, objective_id=action.objective_id, concept_id=action.concept_id,
                       curriculum_version=action.curriculum_version,
                       score=action.priority.score if action.priority else None)
        return action
