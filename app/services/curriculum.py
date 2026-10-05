"""Curriculum application service: learning goals, their curricula, the next learning action and lessons started
from it. The API and the demos use it; the curriculum state itself is only changed by the CurriculumEngine.

- A goal is created once per idempotency key (without one, per its content); its target concepts come from the
  request or, for a target level, from the knowledge base's concepts up to that level.
- A curriculum is planned by a `curriculum_planning` task. Building the same goal again reuses the finished task
  (or resumes an unfinished one) and stores no second version when nothing changed.
- Changing a goal's definition or target date replans its curriculum as a new version; history is kept.
- The next learning action is computed from every goal's curriculum and the learner model, deterministically.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.curriculum.engine import CurriculumEngine
from app.curriculum.planner import targets_for_level
from app.curriculum.tracker import DEFAULT_FRAMEWORK, FRAMEWORK_KEY, CurriculumTracker, TrackedGoals
from app.curriculum.validation import CurriculumValidationError
from app.learner.memory import LearnerMemoryService, UnknownGoal
from app.observability.scope import ExecutionScope, UsageLedger
from app.pedagogy.graph import ConceptGraph
from app.pedagogy.knowledge import KnowledgeBase
from app.runtime.workflows.curriculum_planning import AS_OF_KEY, CAPABILITY, GOAL_INPUT, REASON_INPUT
from app.runtime.workflows.curriculum_planning import WORKFLOW_ID as CURRICULUM_WORKFLOW
from app.runtime.workflows.lesson_generation import FOCUS_KEY, GOAL_KEY
from app.observability.events import EventBus
from app.schemas.common import utcnow
from app.schemas.curriculum import (
    LESSON_ACTIONS,
    Curriculum,
    CurriculumStatus,
    CurriculumVersion,
    GoalInput,
    GoalSpec,
    GoalUpdate,
    NextLearningAction,
    ReplanReason,
    curriculum_id_for,
)
from app.schemas.events import EventType
from app.schemas.learner import GoalStatus, LearningGoal, stable_id
from app.schemas.lesson import LessonRequest
from app.schemas.pedagogy import LessonFocus
from app.schemas.task import Task, TaskStatus
from app.services.tasks import TaskService

BUILD_KEY = "curriculum_build"  # Task.metadata: the idempotency key of a curriculum build
LESSON_CAPABILITIES = ["lesson.text", "lesson.review", "slides.plan"]


class InvalidGoal(ValueError):
    """The goal cannot be planned: unknown concepts, an unknown level, or a conflicting idempotency key."""


class CurriculumService:
    def __init__(self, memory: LearnerMemoryService, knowledge: KnowledgeBase, engine: CurriculumEngine,
                 tasks: TaskService, events: EventBus, clock: Callable[[], datetime] = utcnow) -> None:
        self._memory = memory
        self._knowledge = knowledge
        self._engine = engine
        self._tracker = CurriculumTracker(memory, engine)
        self._tasks = tasks
        self._events = events
        self._clock = clock

    @property
    def engine(self) -> CurriculumEngine:
        return self._engine

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(events=self._events, usage=UsageLedger())

    # --- goals --------------------------------------------------------------------------------------------------

    async def create_goal(self, learner_id: str, data: GoalInput) -> tuple[LearningGoal, bool]:
        """Returns (goal, created). The same idempotency key (or, without one, the same content) returns the stored
        goal instead of creating a second one."""
        fields = data.model_dump(mode="json", exclude={"idempotency_key", "metadata"})
        key = data.idempotency_key or stable_id("goalreq", *(str(fields[k]) for k in sorted(fields)))
        goal_id = stable_id("goal", learner_id, key)
        try:
            existing = self._memory.goal(goal_id)
        except UnknownGoal:
            existing = None
        framework = data.metadata.get(FRAMEWORK_KEY) or self._memory.framework_for(learner_id, data.domain,
                                                                                    DEFAULT_FRAMEWORK)
        targets, source = await self._targets(data.domain, data.target_concepts, data.target_level, framework)
        if existing is not None:
            same = (existing.domain, existing.target_level, existing.target_concepts) == \
                   (data.domain, data.target_level, targets)
            if not same:
                raise InvalidGoal(f"idempotency key {data.idempotency_key!r} was used for a different goal")
            return existing, False
        now = self._clock()
        goal = LearningGoal(
            goal_id=goal_id, learner_id=learner_id, title=data.title, description=data.description,
            domain=data.domain, target_level=data.target_level, target_concepts=targets, target_source=source,
            target_date=data.target_date, priority=data.priority, status=GoalStatus.ACTIVE, created_at=now,
            updated_at=now, metadata={**data.metadata, FRAMEWORK_KEY: framework})
        self._memory.save_goal(goal)
        self._scope().emit(EventType.GOAL_CREATED, learner_id=learner_id, goal_id=goal_id, domain=goal.domain,
                           target_level=goal.target_level, targets=len(targets), priority=goal.priority)
        return goal, True

    async def _targets(self, domain: str, concepts: list[str], level: str | None,
                       framework_id: str) -> tuple[list[str], str]:
        graph = await self._knowledge.graph(domain)
        if concepts:
            unknown = [c for c in concepts if c not in graph]
            if unknown:
                raise InvalidGoal(f"the knowledge base does not know these {domain} concepts: {unknown}")
            return list(concepts), "explicit"
        assert level is not None
        levels = self._memory.frameworks.get(framework_id).levels
        try:
            targets = targets_for_level(graph.concepts(), level, levels)
        except CurriculumValidationError as exc:
            raise InvalidGoal(str(exc)) from exc
        if not targets:
            raise InvalidGoal(f"the knowledge base has no {domain} concepts up to level {level}")
        return targets, "level"

    def goal(self, goal_id: str) -> LearningGoal:
        return self._memory.goal(goal_id)

    def goals(self, learner_id: str) -> list[LearningGoal]:
        return self._memory.goals(learner_id)

    async def update_goal(self, goal_id: str, data: GoalUpdate, *, user_id: str) -> tuple[LearningGoal, Task | None]:
        """Apply the change; a changed definition or target date replans an existing curriculum (a new version)."""
        goal = self._memory.goal(goal_id)
        if goal.status == GoalStatus.COMPLETED and data.status is None:
            raise InvalidGoal(f"goal {goal_id} is completed")
        update = data.model_dump(exclude_unset=True, exclude={"clear_target_date"})
        if data.clear_target_date:
            update["target_date"] = None
        if "target_concepts" in update or "target_level" in update:
            framework = goal.metadata.get(FRAMEWORK_KEY) or DEFAULT_FRAMEWORK
            level = update.get("target_level", goal.target_level)
            concepts = update.get("target_concepts") or ([] if "target_level" in update else goal.target_concepts)
            update["target_concepts"], update["target_source"] = await self._targets(goal.domain, concepts, level,
                                                                                      framework)
        if "metadata" in update:
            update["metadata"] = {**update["metadata"], FRAMEWORK_KEY: goal.metadata.get(FRAMEWORK_KEY)}
        changed = goal.model_copy(update={**update, "updated_at": self._clock()})
        changed = LearningGoal.model_validate(changed.model_dump())
        self._memory.save_goal(changed)
        if changed.status != goal.status:
            self._engine.set_status(curriculum_id_for(goal_id), CurriculumStatus.ARCHIVED
                                    if changed.status == GoalStatus.CANCELLED else CurriculumStatus.ACTIVE)
        self._scope().emit(EventType.GOAL_UPDATED, learner_id=goal.learner_id, goal_id=goal_id,
                           fields=sorted(update), status=changed.status.value)
        before, after = GoalSpec.of(goal), GoalSpec.of(changed)
        task = None
        if changed.is_active and self._engine.current(curriculum_id_for(goal_id)) is not None:
            definition = ("domain", "target_level", "target_concepts")
            if any(getattr(before, f) != getattr(after, f) for f in definition):
                task = await self.build_curriculum(goal_id, user_id=user_id, reason=ReplanReason.GOAL_CHANGED)
            elif before.target_date != after.target_date:
                task = await self.build_curriculum(goal_id, user_id=user_id, reason=ReplanReason.TARGET_DATE_CHANGED)
        return changed, task

    # --- curricula ----------------------------------------------------------------------------------------------

    async def build_curriculum(self, goal_id: str, *, user_id: str, reason: ReplanReason | None = None) -> Task:
        """Run (or reuse) the planning task for the goal's curriculum. Idempotent: the same goal, definition,
        current version and reason reuse the same task; an unfinished one is resumed, never duplicated."""
        goal = self._memory.goal(goal_id)
        if not goal.is_active:
            raise InvalidGoal(f"goal {goal_id} is {goal.status.value}: only an active goal is planned")
        current = self._engine.current(curriculum_id_for(goal_id))
        why = reason or (ReplanReason.INITIAL if current is None else ReplanReason.REBUILD)
        spec = GoalSpec.of(goal).model_dump_json()
        key = stable_id("build", goal_id, spec, current.version_id if current else "", why.value,
                        self._engine.config.fingerprint())
        for task in self._tasks.list_for_learner(goal.learner_id):
            if task.metadata.get(BUILD_KEY) != key:
                continue
            if task.status == TaskStatus.COMPLETED:
                return task
            if task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                continue
            return await self._tasks.resume(task.task_id) if task.status != TaskStatus.CREATED \
                else await self._tasks.run(task.task_id)
        framework = goal.metadata.get(FRAMEWORK_KEY) or DEFAULT_FRAMEWORK
        request = LessonRequest(raw_request=f"Plan the curriculum for {goal.title or goal.description}",
                                subject=goal.domain, topic=goal.title or goal.domain, framework_id=framework,
                                target_level=goal.target_level,
                                language_of_instruction=self._memory.get_or_create(goal.learner_id)
                                .preferences.language_of_instruction,
                                capabilities=[CAPABILITY])
        task = self._tasks.create_planned(
            request=request, learner_id=goal.learner_id, user_id=user_id, workflow_id=CURRICULUM_WORKFLOW,
            inputs={GOAL_INPUT: goal_id, REASON_INPUT: why.value},
            metadata={BUILD_KEY: key, AS_OF_KEY: self._clock().isoformat()})
        return await self._tasks.run(task.task_id)

    def curriculum(self, goal_id: str, *, as_of: datetime | None = None) -> Curriculum | None:
        """The goal's current curriculum with its progress now (read-only: nothing is recorded)."""
        goal = self._memory.goal(goal_id)
        version = self._engine.current(curriculum_id_for(goal_id))
        record = self._engine.record(curriculum_id_for(goal_id))
        if version is None or record is None:
            return None
        when = as_of or self._clock()
        model = self._tracker.model(goal, [], when)
        counts = self._tracker.review_counts(model, {o.concept_id: o.evidence_required for o in version.objectives})
        progress = self._engine.progress(version, model, when, counts)
        objectives = [o.model_copy(update={"current_mastery": progress.of(o.objective_id).current_mastery,
                                           "status": progress.of(o.objective_id).status}) for o in version.objectives]
        return Curriculum(curriculum_id=record.curriculum_id, learner_id=record.learner_id, goal_id=record.goal_id,
                          title=record.title, objectives=objectives, version=version.version,
                          version_id=version.version_id, versions=len(self._engine.versions(record.curriculum_id)),
                          status=record.status, created_at=record.created_at, updated_at=record.updated_at,
                          progress=progress, warnings=version.warnings)

    def versions(self, goal_id: str) -> list[CurriculumVersion]:
        self._memory.goal(goal_id)
        return self._engine.versions(curriculum_id_for(goal_id))

    # --- next learning action -----------------------------------------------------------------------------------

    async def tracked(self, learner_id: str, *, as_of: datetime | None = None) -> TrackedGoals:
        """Refresh every curriculum of the learner (transitions and completion are recorded once)."""
        when = as_of or self._clock()
        domains = sorted({g.domain for g in self._memory.goals(learner_id)})
        graphs = {d: await self._knowledge.graph(d) for d in domains}
        # Replanning needs a task (its artifacts belong to one): evaluations and explicit rebuilds replan.
        return self._tracker.tracked(learner_id, when, graphs=graphs, replan=False, scope=self._scope())

    async def next_action(self, learner_id: str, *, as_of: datetime | None = None) -> NextLearningAction:
        self._memory.get(learner_id)  # unknown learners are rejected
        tracked = await self.tracked(learner_id, as_of=as_of)
        return self._tracker.next_action(learner_id, tracked, self._scope())

    async def start_lesson(self, action: NextLearningAction, *, user_id: str) -> Task:
        """Start the lesson a LEARN / REVIEW / PRACTICE / EVALUATE action asks for, planned around its objective."""
        task = await self.create_lesson(action, user_id=user_id)
        return await self._tasks.run(task.task_id)

    async def create_lesson(self, action: NextLearningAction, *, user_id: str, metadata: dict | None = None,
                            cycle_artifact_id: str | None = None) -> Task:
        """The lesson task for the action, created but not run. `metadata` is added to the task's (a learning cycle's
        step key); `cycle_artifact_id` makes the lesson's LEARNING_ACTION derive from that LEARNING_CYCLE artifact."""
        if action.action not in LESSON_ACTIONS:
            raise InvalidGoal(f"a {action.action.value} action does not start a lesson")
        assert action.goal_id and action.objective_id and action.concept_id and action.curriculum_id
        goal = self._memory.goal(action.goal_id)
        version = self._engine.current(action.curriculum_id)
        if version is None or version.version != action.curriculum_version:
            raise InvalidGoal("the curriculum changed since this action was selected; ask for the next action again")
        objective = version.objective(action.objective_id)
        graph: ConceptGraph = await self._knowledge.graph(goal.domain)
        concept = graph.concept(objective.concept_id)
        focus = LessonFocus(action=action.action.value, concept_id=objective.concept_id,
                            objective_id=objective.objective_id, description=objective.description,
                            goal_id=goal.goal_id, curriculum_id=version.curriculum_id,
                            curriculum_version=version.version, action_id=action.action_id,
                            objective_artifact_id=version.artifact_ids.get(objective.objective_id),
                            cycle_artifact_id=cycle_artifact_id)
        profile = self._memory.get_or_create(goal.learner_id)
        request = LessonRequest(
            raw_request=f"{action.action.value.title()} {concept.name} ({goal.title or goal.domain})",
            subject=goal.domain, topic=concept.topic or concept.name,
            framework_id=goal.metadata.get(FRAMEWORK_KEY) or DEFAULT_FRAMEWORK, target_level=goal.target_level,
            language_of_instruction=profile.preferences.language_of_instruction, capabilities=LESSON_CAPABILITIES)
        return self._tasks.create_lesson(lesson_request=request, learner_id=goal.learner_id, user_id=user_id,
                                         metadata={**(metadata or {}), GOAL_KEY: goal.goal_id,
                                                   FOCUS_KEY: focus.model_dump(mode="json")})

    def lesson_for(self, learner_id: str, goal_id: str, concept_id: str) -> Task | None:
        """The learner's newest completed lesson for a goal's concept (started from one of its actions), if any."""
        found = [t for t in self._tasks.list_for_learner(learner_id)
                 if t.status == TaskStatus.COMPLETED and t.metadata.get(GOAL_KEY) == goal_id
                 and (t.metadata.get(FOCUS_KEY) or {}).get("concept_id") == concept_id]
        return max(found, key=lambda t: (t.created_at, t.task_id), default=None)
