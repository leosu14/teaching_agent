"""The teacher dashboard (MVP item 7): a class's roster with each learner's progress, and the UNCERTAIN grades of
those learners for the teacher to review.

Read-only, and built only from stored records: learner profiles (mastery), goals and their curricula (progress
computed now, nothing recorded, as `GET /goals/{id}/curriculum` does), the learning cycles as stored (no
reconciliation: `GET /learning-cycles/{id}` reconciles, this does not) and the assessment attempts with their grades.
It never calls a model and never writes, so a teacher's dashboard cannot change a learner's state.

The caller has already established that the class is the principal's (`taught_class`): its roster is exactly the set
of learners the teacher may read (the rule of `Access.check`), so nothing outside it is read."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from app.assessment.repository import AssessmentRepository
from app.curriculum.repository import LearningCycleRepository
from app.learner.memory import LearnerMemoryService, UnknownLearner
from app.schemas.assessment import AssessmentItem, AssessmentOutcome
from app.schemas.classroom import Classroom, ClassMember
from app.schemas.common import utcnow
from app.schemas.dashboard import ClassDashboard, GoalOverview, LearnerOverview, UncertainGrade
from app.schemas.learner import GoalStatus, LearnerProfile
from app.schemas.learning_cycle import ACTIVE_CYCLE_STATUSES
from app.services.classes import ClassService
from app.services.curriculum import CurriculumService


class ClassDashboardService:
    def __init__(self, *, classes: ClassService, memory: LearnerMemoryService, curriculum: CurriculumService,
                 cycles: LearningCycleRepository, assessment: AssessmentRepository,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._classes = classes
        self._memory = memory
        self._curriculum = curriculum
        self._cycles = cycles
        self._assessment = assessment
        self._clock = clock

    def dashboard(self, classroom: Classroom) -> ClassDashboard:
        members = self._classes.repository.members(classroom.class_id)
        profiles = {m.learner_id: self._profile(m.learner_id) for m in members}
        uncertain = self._uncertain(members, profiles)
        open_by_learner: dict[str, int] = {}
        for u in uncertain:
            if not u.resolved:
                open_by_learner[u.learner_id] = open_by_learner.get(u.learner_id, 0) + 1
        learners = [self._overview(m, profiles[m.learner_id], open_by_learner.get(m.learner_id, 0)) for m in members]
        learners.sort(key=lambda o: (o.display_name.casefold(), o.learner_id))
        return ClassDashboard(class_id=classroom.class_id, name=classroom.name, created_at=classroom.created_at,
                              as_of=self._clock(), learners=learners, uncertain_open=sum(open_by_learner.values()))

    def uncertain_grades(self, classroom: Classroom, *, include_resolved: bool = False) -> list[UncertainGrade]:
        """The roster's UNCERTAIN grades, newest first; by default only those no later graded attempt settled."""
        members = self._classes.repository.members(classroom.class_id)
        profiles = {m.learner_id: self._profile(m.learner_id) for m in members}
        found = self._uncertain(members, profiles)
        return [u for u in found if include_resolved or not u.resolved]

    # --- parts ---------------------------------------------------------------------------------------------------

    def _profile(self, learner_id: str) -> LearnerProfile | None:
        try:
            return self._memory.get(learner_id)
        except UnknownLearner:  # joined, then the profile went away: still on the roster, nothing to show
            return None

    def _overview(self, member: ClassMember, profile: LearnerProfile | None, uncertain_open: int) -> LearnerOverview:
        learner_id = member.learner_id
        if profile is None:
            return LearnerOverview(learner_id=learner_id, display_name=learner_id, joined_at=member.joined_at,
                                   average_mastery=0.0, mastered=0, weak=0, due_for_review=0, lessons_completed=0,
                                   assessments_taken=0, cycles=0, uncertain_open=uncertain_open)
        progress = self._memory.progress(learner_id)
        cycles = self._cycles.for_learner(learner_id)  # newest first
        current = next((c for c in cycles if c.status in ACTIVE_CYCLE_STATUSES), None)
        latest = cycles[0] if cycles else None
        return LearnerOverview(
            learner_id=learner_id, display_name=profile.display_name or learner_id, joined_at=member.joined_at,
            average_mastery=progress.average_mastery, mastered=len(progress.mastered), weak=len(progress.weak),
            due_for_review=len(progress.due_for_review), lessons_completed=progress.lessons_completed,
            assessments_taken=progress.assessments_taken, goals=self._goals(learner_id), cycles=len(cycles),
            current_cycle_status=current.status.value if current else None,
            last_cycle_status=latest.status.value if latest else None,
            last_active_at=max(latest.updated_at, profile.updated_at) if latest else profile.updated_at,
            uncertain_open=uncertain_open)

    def _goals(self, learner_id: str) -> list[GoalOverview]:
        out = []
        for goal in self._curriculum.goals(learner_id):
            if goal.status == GoalStatus.CANCELLED:
                continue
            plan = self._curriculum.curriculum(goal.goal_id)  # read-only: progress now, nothing recorded
            base = GoalOverview(goal_id=goal.goal_id, title=goal.title, domain=goal.domain,
                                target_level=goal.target_level, target_date=goal.target_date,
                                status=goal.status.value, planned=plan is not None)
            if plan is not None:
                p = plan.progress
                base = base.model_copy(update={"objectives_mastered": p.mastered, "objectives_required": p.required,
                                               "percent_complete": p.percent_complete,
                                               "goal_complete": p.goal_complete})
            out.append(base)
        return out

    def _uncertain(self, members: list[ClassMember],
                   profiles: dict[str, LearnerProfile | None]) -> list[UncertainGrade]:
        rows = self._assessment.graded_attempts(m.learner_id for m in members)
        items: dict[str, AssessmentItem | None] = {}
        found: list[UncertainGrade] = []
        for i, (attempt, grade) in enumerate(rows):
            if grade.outcome != AssessmentOutcome.UNCERTAIN:
                continue
            later = []  # rows are ordered by learner, item and attempt number: the same item's later attempts follow
            for a, g in rows[i + 1:]:
                if a.learner_id != attempt.learner_id or a.assessment_item_id != attempt.assessment_item_id:
                    break
                later.append(g)
            item_id = attempt.assessment_item_id
            if item_id not in items:
                items[item_id] = self._assessment.item(item_id)
            item = items[item_id]
            if item is None:
                continue
            profile = profiles.get(attempt.learner_id)
            concept = profile.concepts.get(item.concept_id) if profile else None
            found.append(UncertainGrade(
                attempt_id=attempt.attempt_id, learner_id=attempt.learner_id,
                display_name=(profile.display_name if profile else "") or attempt.learner_id,
                assessment_item_id=item_id, concept_id=item.concept_id,
                concept_name=concept.name if concept else item.concept_id, prompt=item.prompt,
                response_type=item.response_type, choices=list(item.choices), learner_answer=attempt.learner_answer,
                source=attempt.source, attempt_number=attempt.attempt_number, submitted_at=attempt.submitted_at,
                grader_type=grade.grader_type, confidence=grade.confidence, later_attempts=len(later),
                resolved=any(g.counts for g in later)))
        found.sort(key=lambda u: (u.submitted_at, u.attempt_id), reverse=True)
        return found
