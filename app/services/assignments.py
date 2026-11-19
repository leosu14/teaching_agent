"""Class assignments (Important item 3, issue #62): a teacher assigns a goal, or course topics, with a deadline to a
whole class, and follows each learner's progress on it.

An assignment is materialized, for each learner on the roster, as an ordinary `LearningGoal` made by
`CurriculumService.create_goal` (`target_date = due_at`, `metadata.assignment_id`, its targets resolved once, when the
assignment was created). Everything a goal already has (curricula, next actions, learning cycles, completion by
mastery, the dashboard, learner deletion) works on it unchanged. See app/schemas/assignment.py for the decisions.

Writes, and the order of their locks:

- Creating materializes for every learner on the roster; a learner who joins later gets it then (`on_join`, D5).
  Both are idempotent (the goal id and the learner record are keyed by the assignment), so repeating the request, or
  joining again, completes a run that stopped part way.
- A teacher's change (title, description, `due_at`) or cancellation is written to the assignment under its row lock,
  then carried to each current member's goal by `SqlAssignmentRepository.sync_goal`, which takes the assignment row
  lock again and writes the assignment's values *as they are then*. So concurrent changes, a materialization racing a
  change, and a cancellation racing a change all leave every goal at the assignment's latest values. No curriculum is
  rebuilt and no model is called (D10): the deadline is read when the next actions are ranked.
- Every write for a learner runs under `learner_writes`: a learner deleted meanwhile is skipped (LearnerDataGone),
  never half-written.

Reads are read-only, as the class dashboard is: progress is computed now, nothing is refreshed or recorded."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from app.curriculum.engine import curriculum_id_for
from app.curriculum.tracker import DEFAULT_FRAMEWORK, FRAMEWORK_KEY
from app.learner.memory import LearnerMemoryService, UnknownGoal, UnknownLearner
from app.schemas.assignment import (
    ASSIGNMENT_KEY,
    CLASS_KEY,
    AssignedTopic,
    AssignmentConflict,
    AssignmentDetail,
    AssignmentLearner,
    AssignmentNotFound,
    AssignmentStatus,
    AssignmentTotals,
    AssignmentView,
    ClassAssignment,
    CreateAssignment,
    InvalidAssignment,
    LearnerAssignment,
    LearnerAssignmentProgress,
    LearnerAssignmentStatus,
    MaterializedState,
    TopicProgress,
    UpdateAssignment,
)
from app.schemas.classroom import Classroom
from app.schemas.common import utcnow
from app.schemas.curriculum import CurriculumStatus, GoalInput, ObjectiveStatus
from app.schemas.events import EventType
from app.schemas.lease import LearnerDataGone, learner_writes
from app.schemas.learner import GoalStatus, LearningGoal, stable_id
from app.services.curriculum import CurriculumService, InvalidGoal
from app.storage.repositories import SqlAssignmentRepository, SqlClassRepository

TARGETS_MAX = 200  # GoalInput.target_concepts


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def synced(assignment: ClassAssignment, goal: LearningGoal, now: datetime) -> LearningGoal | None:
    """The learner's goal with what the assignment controls now: its title, description and deadline, or CANCELLED
    once the assignment is. None when nothing changes. A goal already COMPLETED or CANCELLED is never changed (D13)."""
    if goal.status not in (GoalStatus.ACTIVE, GoalStatus.PAUSED):
        return None
    if not assignment.is_active:
        return goal.model_copy(update={"status": GoalStatus.CANCELLED, "updated_at": now})
    wanted = {"title": assignment.title, "description": assignment.description, "target_date": assignment.due_at}
    if all(getattr(goal, k) == v for k, v in wanted.items()):
        return None
    return goal.model_copy(update={**wanted, "updated_at": now})


class AssignmentService:
    def __init__(self, repository: SqlAssignmentRepository, classes: SqlClassRepository, memory: LearnerMemoryService,
                 curriculum: CurriculumService, courses: Iterable = (), events=None,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self.repository = repository
        self._classes = classes
        self._memory = memory
        self._curriculum = curriculum
        self._courses = list(courses)
        self._events = events
        self._clock = clock

    # --- the teacher's side ---------------------------------------------------------------------------------------

    async def create(self, classroom: Classroom, data: CreateAssignment, *,
                     teacher_id: str) -> tuple[AssignmentView, bool]:
        """Create the assignment and give it to every learner on the roster. Returns (view, created). The same
        idempotency key (or, without one, the same content) returns the stored assignment, and completes its
        materialization; a key used for different content is AssignmentConflict (409)."""
        framework = data.framework_id or self._framework(data.domain)
        if framework not in self._memory.frameworks.ids():
            raise InvalidAssignment(f"unknown level framework {framework!r}")
        topics, targets = await self._resolve(data, framework)
        content = data.model_dump(mode="json", exclude={"idempotency_key"})
        key = data.idempotency_key or stable_id("asgreq", *(f"{k}={content[k]}" for k in sorted(content)))
        assignment_id = stable_id("asg", classroom.class_id, key)
        now = self._clock()
        assignment = ClassAssignment(
            assignment_id=assignment_id, class_id=classroom.class_id, created_by=teacher_id, title=data.title,
            description=data.description, domain=data.domain, framework_id=framework, target_level=data.target_level,
            topics=topics, target_concepts=targets, due_at=data.due_at, created_at=now, updated_at=now)
        stored, created = self.repository.create(assignment)
        if not created and not self._same(stored, assignment):
            raise AssignmentConflict("this idempotency key was used for a different assignment")
        while not created and not stored.is_active and data.idempotency_key is None:
            # The same content as a cancelled assignment: a new one (D6: cancel and assign again), not the old one.
            # Its id follows from the cancelled one's, so a repeated request finds it again.
            assignment = assignment.model_copy(update={"assignment_id": stable_id("asg", stored.assignment_id,
                                                                                    "after-cancel")})
            stored, created = self.repository.create(assignment)
        if created:
            self._emit(EventType.ASSIGNMENT_CREATED, stored, targets=len(targets), topics=len(topics))
        if stored.is_active:
            for member in self._classes.members(classroom.class_id):
                await self._materialize(stored, member.learner_id, "created")
        return self._summary(stored, self._clock()), created

    def for_class(self, classroom: Classroom, *, as_of: datetime | None = None) -> list[AssignmentView]:
        when = as_of or self._clock()
        members = self._classes.members(classroom.class_id)
        return [self._summary(a, when, members) for a in self.repository.for_class(classroom.class_id)]

    def _summary(self, assignment: ClassAssignment, when: datetime, members=None) -> AssignmentView:
        """The assignment with its totals over the learners on the roster now."""
        members = self._classes.members(assignment.class_id) if members is None else members
        totals = AssignmentTotals(learners=len(members))
        for m in members:
            _count(totals, self._progress(assignment, self.repository.learner(assignment.assignment_id, m.learner_id),
                                          m.learner_id, when))
        return self.view(assignment, when, totals)

    def get(self, classroom: Classroom, assignment_id: str) -> ClassAssignment:
        found = self.repository.get(assignment_id)
        if found is None or found.class_id != classroom.class_id:
            raise AssignmentNotFound()
        return found

    def detail(self, classroom: Classroom, assignment_id: str, *, as_of: datetime | None = None) -> AssignmentDetail:
        """The assignment and each learner on the roster now (D9: a learner who left is not shown)."""
        assignment = self.get(classroom, assignment_id)
        when = as_of or self._clock()
        members = self._classes.members(classroom.class_id)
        learners = [self._progress(assignment, self.repository.learner(assignment_id, m.learner_id), m.learner_id,
                                   when) for m in members]
        totals = AssignmentTotals(learners=len(learners))
        for p in learners:
            _count(totals, p)
        return AssignmentDetail(assignment=self.view(assignment, when, totals), as_of=when, learners=learners)

    async def update(self, classroom: Classroom, assignment_id: str, data: UpdateAssignment) -> AssignmentView:
        """Change the title, description or deadline, and carry the change to every current member's goal, without
        replanning (D10). The targets cannot change (D6). A cancelled assignment cannot change (409)."""
        self.get(classroom, assignment_id)
        fields = data.model_dump(exclude_unset=True)
        fields = {k: v for k, v in fields.items() if v is not None}
        now = self._clock()

        def change(current: ClassAssignment) -> ClassAssignment | None:
            if not current.is_active:
                raise AssignmentConflict("a cancelled assignment cannot be changed")
            if all(getattr(current, k) == v for k, v in fields.items()):
                return None
            return ClassAssignment.model_validate({**current.model_dump(), **fields, "updated_at": now})

        stored = self.repository.update(assignment_id, change)
        if stored is None:
            raise AssignmentNotFound()
        if fields:
            self._emit(EventType.ASSIGNMENT_UPDATED, stored, fields=sorted(fields))
        self._propagate(stored)
        return self._summary(stored, self._clock())

    async def cancel(self, classroom: Classroom, assignment_id: str) -> AssignmentView:
        """Cancel the assignment (idempotent): every current member's goal that is not COMPLETED becomes CANCELLED
        and its curriculum is archived. A learner who already left keeps theirs (D9)."""
        self.get(classroom, assignment_id)
        now = self._clock()
        made = []

        def change(current: ClassAssignment) -> ClassAssignment | None:
            if not current.is_active:
                return None
            made.append(True)  # decided under the row lock: of two racing cancels, one makes the change
            return current.model_copy(update={"status": AssignmentStatus.CANCELLED, "cancelled_at": now,
                                              "updated_at": now})

        stored = self.repository.update(assignment_id, change)
        if stored is None:
            raise AssignmentNotFound()
        if made:
            self._emit(EventType.ASSIGNMENT_CANCELLED, stored)
        self._propagate(stored)  # again on a repeated cancel: completes a run that stopped part way
        return self._summary(stored, self._clock())

    # --- the learner's side ---------------------------------------------------------------------------------------

    async def on_join(self, class_id: str, learner_id: str) -> None:
        """A learner joined (or joined again): they get every active assignment of the class, with its original
        deadline (D5). One whose concepts are not all in the knowledge base any more is NOT_ASSIGNED for them, and the
        join still succeeds (D12). A learner joining again after leaving has their goals brought back to the
        assignments' current title, description and deadline."""
        for assignment in self.repository.for_class(class_id):
            if assignment.is_active:
                await self._materialize(assignment, learner_id, "joined")

    def for_learner(self, learner_id: str, *, tenant_id: str | None,
                    as_of: datetime | None = None) -> list[LearnerAssignment]:
        """The learner's assignments in the classes they are in now (of the caller's tenant)."""
        when = as_of or self._clock()
        classes = {c.class_id: c for c, _ in self._classes.enrollments(learner_id) if c.tenant_id == tenant_id}
        out = []
        for assignment, record in self.repository.for_learner(learner_id):
            classroom = classes.get(assignment.class_id)
            if classroom is None:
                continue
            out.append(LearnerAssignment(
                assignment_id=assignment.assignment_id, class_id=classroom.class_id, class_name=classroom.name,
                title=assignment.title, description=assignment.description, domain=assignment.domain,
                target_level=assignment.target_level, topics=[t.topic for t in assignment.topics],
                due_at=assignment.due_at, assignment_status=assignment.status,
                progress=self._progress(assignment, record, learner_id, when)))
        return out

    def locks(self, goal: LearningGoal) -> bool:
        """Whether the learner's changes to this goal are limited (D7, D13): it was materialized from an assignment,
        for this learner, and they are on its class roster. While the assignment is active they may change its
        priority and pause or resume it; once the goal is COMPLETED or CANCELLED (a cancelled assignment cancels the
        goals of its roster), nothing. An open goal of a cancelled assignment is not locked.
        A learner who left keeps the goal, unlocked (D9). A goal whose metadata merely names an assignment is not
        locked."""
        assignment_id = goal.metadata.get(ASSIGNMENT_KEY)
        if not assignment_id:
            return False
        assignment = self.repository.get(assignment_id)
        if assignment is None:
            return False
        record = self.repository.learner(assignment_id, goal.learner_id)
        if record is None or record.goal_id != goal.goal_id:
            return False
        if not assignment.is_active and goal.status in (GoalStatus.ACTIVE, GoalStatus.PAUSED):
            # Cancelled while the learner was away (D9): the cancellation never reached this goal, so it is theirs
            # now, whole, even after they rejoin. A goal the cancellation (or mastery) finished stays locked (D13).
            return False
        return self._classes.member(assignment.class_id, goal.learner_id) is not None

    # --- materializing and propagating ------------------------------------------------------------------------------

    async def _materialize(self, assignment: ClassAssignment, learner_id: str, source: str) -> None:
        """Give the learner the assignment, idempotently. A learner deleted meanwhile, or no longer on the roster (a
        leave racing the create), is skipped."""
        try:
            if self._classes.member(assignment.class_id, learner_id) is None:
                return
            record = self.repository.learner(assignment.assignment_id, learner_id)
            if record is not None and record.state == MaterializedState.ASSIGNED:
                self._sync(assignment.assignment_id, record)  # a rejoin: back to the assignment's current values
                return
            goal = await self._goal_for(assignment, learner_id)
            state = MaterializedState.ASSIGNED if goal is not None else MaterializedState.NOT_ASSIGNED
            with learner_writes(learner_id):
                record = self.repository.record_learner(AssignmentLearner(
                    assignment_id=assignment.assignment_id, learner_id=learner_id,
                    goal_id=goal.goal_id if goal is not None else None, state=state, assigned_at=self._clock(),
                    source=source))
            if record.state == MaterializedState.ASSIGNED:
                # A change or cancellation committed since `assignment` was read may have missed this goal: its record
                # did not exist yet. Read the assignment again, under its lock, and write what it says now.
                self._sync(assignment.assignment_id, record)
            if self._events is not None:
                with learner_writes(learner_id):  # the event names the learner: refused once they are deleted
                    self._events.emit(EventType.ASSIGNMENT_MATERIALIZED, assignment_id=assignment.assignment_id,
                                      class_id=assignment.class_id, learner_id=learner_id, state=record.state.value,
                                      source=record.source)
        except (LearnerDataGone, UnknownLearner):
            return

    async def _goal_for(self, assignment: ClassAssignment, learner_id: str) -> LearningGoal | None:
        """The learner's goal for the assignment, made unless it exists; None when the knowledge base no longer has
        all of its concepts (D12)."""
        graph = await self._curriculum.graph(assignment.domain)
        if any(c not in graph for c in assignment.target_concepts):
            return None
        data = GoalInput(
            title=assignment.title, description=assignment.description, domain=assignment.domain,
            target_level=assignment.target_level, target_concepts=assignment.target_concepts,
            target_date=assignment.due_at, priority=3, idempotency_key=assignment.assignment_id,
            metadata={ASSIGNMENT_KEY: assignment.assignment_id, CLASS_KEY: assignment.class_id,
                      FRAMEWORK_KEY: assignment.framework_id})
        try:
            goal, _ = await self._curriculum.create_goal(learner_id, data)
        except InvalidGoal:
            # The goal exists but no longer matches (the learner changed it while it was not locked), or a concept
            # went missing between the check and the write: what the learner has is kept.
            goal_id = stable_id("goal", learner_id, assignment.assignment_id)
            try:
                return self._memory.goal(goal_id)
            except UnknownGoal:
                return None
        return goal

    def _propagate(self, assignment: ClassAssignment) -> None:
        """Carry the assignment's current values to the goal of every learner on the roster now. Learners who left
        keep their goal as it is (D9); a learner deleted meanwhile is skipped."""
        members = {m.learner_id for m in self._classes.members(assignment.class_id)}
        for record in self.repository.learners(assignment.assignment_id):
            if record.learner_id in members and record.state == MaterializedState.ASSIGNED:
                self._sync(assignment.assignment_id, record)

    def _sync(self, assignment_id: str, record: AssignmentLearner) -> None:
        if record.goal_id is None:
            return
        now = self._clock()
        try:
            with learner_writes(record.learner_id):
                goal = self.repository.sync_goal(assignment_id, record.goal_id, lambda a, g: synced(a, g, now))
        except LearnerDataGone:
            return
        if goal is not None and goal.status == GoalStatus.CANCELLED:
            self._curriculum.engine.set_status(curriculum_id_for(goal.goal_id), CurriculumStatus.ARCHIVED)

    # --- targets ----------------------------------------------------------------------------------------------------

    def _framework(self, domain: str) -> str:
        """The level framework of the course of this subject, or the generic one."""
        for course in self._courses:
            if getattr(course, "subject", None) == domain:
                return course.framework
        return DEFAULT_FRAMEWORK

    async def _resolve(self, data: CreateAssignment, framework: str) -> tuple[list[AssignedTopic], list[str]]:
        """The assigned topics and the concepts they stand for, or the concepts up to the level: resolved once, now
        (D1, D6). InvalidAssignment for an unknown course, topic or level."""
        graph = await self._curriculum.graph(data.domain)
        if len(graph) == 0:
            raise InvalidAssignment(f"the knowledge base has no course for {data.domain!r}")
        if data.target_level:
            try:
                targets, _ = await self._curriculum.resolve_targets(data.domain, [], data.target_level, framework)
            except InvalidGoal as exc:
                raise InvalidAssignment(str(exc)) from exc
            topics: list[AssignedTopic] = []
        else:
            by_topic: dict[str, list[str]] = {}
            for concept in graph.concepts():
                if concept.topic:
                    by_topic.setdefault(concept.topic.casefold(), []).append(concept.concept_id)
            names = {c.topic.casefold(): c.topic for c in graph.concepts() if c.topic}
            unknown = [t for t in data.topics if t.casefold() not in by_topic]
            if unknown:
                raise InvalidAssignment(f"the {data.domain} course has no topics {unknown}")
            topics = [AssignedTopic(topic=names[t.casefold()], concept_ids=graph.order(by_topic[t.casefold()]))
                      for t in data.topics]
            targets = list(dict.fromkeys(c for t in topics for c in t.concept_ids))
        if len(targets) > TARGETS_MAX:
            raise InvalidAssignment(f"the assignment covers {len(targets)} concepts; at most {TARGETS_MAX} can be "
                                    "assigned at once")
        return topics, targets

    @staticmethod
    def _same(stored: ClassAssignment, new: ClassAssignment) -> bool:
        """The same assignment as far as a repeated create can tell: the same targets. Its title, description and
        deadline may have been changed since (a retry of the original create then returns it as it is now)."""
        keys = ("domain", "framework_id", "target_level")
        return all(getattr(stored, k) == getattr(new, k) for k in keys) and \
            [t.topic for t in stored.topics] == [t.topic for t in new.topics]

    # --- views ------------------------------------------------------------------------------------------------------

    def view(self, assignment: ClassAssignment, as_of: datetime | None = None,
             totals: AssignmentTotals | None = None) -> AssignmentView:
        when = as_of or self._clock()
        return AssignmentView(
            assignment_id=assignment.assignment_id, class_id=assignment.class_id, title=assignment.title,
            description=assignment.description, domain=assignment.domain, framework_id=assignment.framework_id,
            target_level=assignment.target_level, topics=[t.topic for t in assignment.topics],
            due_at=assignment.due_at, status=assignment.status, created_at=assignment.created_at,
            updated_at=assignment.updated_at, cancelled_at=assignment.cancelled_at,
            overdue=assignment.is_active and when > assignment.due_at, totals=totals or AssignmentTotals())

    def _progress(self, assignment: ClassAssignment, record: AssignmentLearner | None, learner_id: str,
                  as_of: datetime) -> LearnerAssignmentProgress:
        """One learner's status on the assignment, derived now (section 4 of #62)."""
        try:
            profile = self._memory.get(learner_id)
            name = profile.display_name or learner_id
        except UnknownLearner:
            name = learner_id
        topics = [TopicProgress(topic=t.topic, mastered=0, total=len(t.concept_ids)) for t in assignment.topics]
        base = dict(learner_id=learner_id, display_name=name, topics=topics,
                    assigned_at=record.assigned_at if record else assignment.created_at,
                    source=record.source if record else "pending")
        past_due = as_of > assignment.due_at  # exactly at due_at is not overdue
        goal = None
        if record is not None and record.goal_id is not None:
            try:
                goal = self._memory.goal(record.goal_id)
            except UnknownGoal:
                goal = None
        if goal is None:
            if not assignment.is_active:
                status = LearnerAssignmentStatus.CANCELLED
            elif record is not None:
                status = LearnerAssignmentStatus.NOT_ASSIGNED
            else:  # on the roster, not materialized yet (a join or a create still running, or interrupted)
                status = LearnerAssignmentStatus.OVERDUE if past_due else LearnerAssignmentStatus.NOT_STARTED
            return LearnerAssignmentProgress(status=status, overdue=status == LearnerAssignmentStatus.OVERDUE,
                                             **base)
        plan = self._curriculum.curriculum(goal.goal_id, as_of=as_of)
        mastered: set[str] = set()
        started = False
        completed_at = goal.completed_at if goal.status == GoalStatus.COMPLETED else None
        rule_holds = False
        if plan is not None:
            mastered = {o.concept_id for o in plan.objectives if o.status == ObjectiveStatus.MASTERED}
            started = any(p.evidence_count > 0 or p.status in (ObjectiveStatus.IN_PROGRESS, ObjectiveStatus.MASTERED)
                          for p in plan.progress.objectives)
            rule_holds = plan.progress.goal_complete
            if rule_holds and goal.status == GoalStatus.ACTIVE:  # not recorded yet: when it was reached (D11)
                completed_at = self._curriculum.completion_time(goal, as_of)
        topics = [t.model_copy(update={"mastered": sum(c in mastered for c in a.concept_ids)})
                  for t, a in zip(topics, assignment.topics)]
        if goal.status == GoalStatus.COMPLETED:
            status = LearnerAssignmentStatus.COMPLETED
        elif goal.status == GoalStatus.CANCELLED or not assignment.is_active:
            status = LearnerAssignmentStatus.CANCELLED
        elif rule_holds and goal.status == GoalStatus.ACTIVE:  # a paused goal is not completed while paused
            status = LearnerAssignmentStatus.COMPLETED
        elif past_due:
            status = LearnerAssignmentStatus.OVERDUE
        elif started:
            status = LearnerAssignmentStatus.IN_PROGRESS
        else:
            status = LearnerAssignmentStatus.NOT_STARTED
        completed = status == LearnerAssignmentStatus.COMPLETED
        if completed_at is not None:
            completed_at = _aware(completed_at)
        return LearnerAssignmentProgress(
            **{**base, "topics": topics}, goal_id=goal.goal_id, status=status, paused=goal.status == GoalStatus.PAUSED,
            overdue=status == LearnerAssignmentStatus.OVERDUE, planned=plan is not None,
            percent_complete=plan.progress.percent_complete if plan else 0.0,
            objectives_mastered=plan.progress.mastered if plan else 0,
            objectives_required=plan.progress.required if plan else 0,
            completed_at=completed_at if completed else None,
            completed_late=(completed_at > assignment.due_at) if completed and completed_at is not None else None)

    def _emit(self, type: str, assignment: ClassAssignment, **data: object) -> None:
        """Ids, the status and counts only: never the title or description."""
        if self._events is not None:
            self._events.emit(type, assignment_id=assignment.assignment_id, class_id=assignment.class_id,
                              status=assignment.status.value, **data)


def _count(totals: AssignmentTotals, p: LearnerAssignmentProgress) -> None:
    field = {LearnerAssignmentStatus.COMPLETED: "completed", LearnerAssignmentStatus.IN_PROGRESS: "in_progress",
             LearnerAssignmentStatus.NOT_STARTED: "not_started", LearnerAssignmentStatus.OVERDUE: "overdue",
             LearnerAssignmentStatus.NOT_ASSIGNED: "not_assigned",
             LearnerAssignmentStatus.CANCELLED: "cancelled"}[p.status]
    setattr(totals, field, getattr(totals, field) + 1)
