"""CurriculumTracker: a learner's goals and curricula against their learner memory.

It reads the learner model (never writes it), refreshes the progress of every goal that has a curriculum (the
engine reports transitions once, applies the completion rule and replans on a trigger), stores a goal the
completion rule completed, and selects the next learning action across all goals. Goals without a curriculum are
ignored: a learner without curricula keeps the adaptive lesson loop exactly as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from app.curriculum.engine import ArtifactWriter, CurriculumEngine
from app.curriculum.next_action import GoalCurriculum, NextActionEngine
from app.learner.memory import LearnerMemoryService
from app.observability.scope import ExecutionScope
from app.pedagogy.graph import ConceptGraph
from app.schemas.curriculum import NextLearningAction, ProgressUpdate, curriculum_id_for
from app.schemas.events import EventType
from app.schemas.learner import GoalStatus, LearningGoal
from app.schemas.pedagogy import LearnerModel

DEFAULT_FRAMEWORK = "mastery"  # the generic level scale, for a subject the learner's profile does not list
FRAMEWORK_KEY = "framework_id"  # goal metadata key: the level framework a goal's target level belongs to


@dataclass
class TrackedGoals:
    """The learner's goals that have a curriculum, refreshed at `as_of`."""

    as_of: datetime
    entries: list[GoalCurriculum] = field(default_factory=list)
    models: dict[str, LearnerModel] = field(default_factory=dict)  # by domain
    updates: list[ProgressUpdate] = field(default_factory=list)
    completed_now: frozenset[str] = frozenset()


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
            assert version is not None
            counts = self.review_counts(model, {o.concept_id: o.evidence_required for o in version.objectives})
            update = self.engine.refresh(goal, model, as_of, review_counts=counts,
                                         graph=graph if replan else None, task_id=task_id, scope=scope,
                                         artifacts=artifacts)
            assert update is not None
            if update.goal_completed:
                goal = self.memory.save_goal(goal.model_copy(update={"status": GoalStatus.COMPLETED,
                                                                     "updated_at": as_of}))
                completed.add(goal.goal_id)
            current = self.engine.current(curriculum_id_for(goal.goal_id))
            assert current is not None
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
