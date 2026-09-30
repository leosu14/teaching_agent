"""Long-term learner memory. The only way anything reads or writes learner history."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from app.learner.frameworks import FrameworkRegistry
from app.learner.mastery import (
    KNOWN,
    MASTERED,
    PRIOR_MASTERY,
    UNKNOWN,
    WEAK,
    apply_evidence,
    apply_exposure,
)
from app.observability.scope import ExecutionScope
from app.schemas.common import utcnow
from app.schemas.events import EventType
from app.schemas.learner import (
    AssessmentRecord,
    ConceptMastery,
    InteractionRecord,
    LearnerProfile,
    LearnerProfileInput,
    LearnerProgress,
    LearnerSnapshot,
    LearnerSummary,
    LessonRecord,
    MasteryChange,
    MasteryUpdate,
    MistakeRecord,
    SubjectState,
)
from app.schemas.evaluation import EvaluationOutcome
from app.schemas.lesson import ConceptRef, LessonOutcome


class LearnerRepository(Protocol):
    def get(self, learner_id: str) -> LearnerProfile | None: ...

    def save(self, profile: LearnerProfile) -> None: ...


class UnknownLearner(KeyError):
    pass


class LearnerMemoryService:
    def __init__(
        self,
        repository: LearnerRepository,
        frameworks: FrameworkRegistry,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._repo = repository
        self._frameworks = frameworks
        self._clock = clock

    # --- profile ---------------------------------------------------------------------------

    def get(self, learner_id: str) -> LearnerProfile:
        profile = self._repo.get(learner_id)
        if profile is None:
            raise UnknownLearner(learner_id)
        return profile

    def get_or_create(self, learner_id: str) -> LearnerProfile:
        profile = self._repo.get(learner_id)
        if profile is None:
            profile = LearnerProfile(learner_id=learner_id, created_at=self._clock(), updated_at=self._clock())
            self._repo.save(profile)
        return profile

    def upsert(self, learner_id: str, data: LearnerProfileInput) -> LearnerProfile:
        profile = self.get_or_create(learner_id)
        for state in data.subjects:
            self._frameworks.get(state.framework_id)  # reject unknown frameworks early
            if state.target_level:
                self._frameworks.get(state.framework_id).index(state.target_level)
            existing = profile.subjects.get(state.subject)
            if existing and not state.estimated_level:
                state = state.model_copy(update={"estimated_level": existing.estimated_level})
            profile.subjects[state.subject] = state
        profile.display_name = data.display_name or profile.display_name
        profile.preferences = data.preferences
        profile.updated_at = self._clock()
        self._repo.save(profile)
        return profile

    def summary(self, learner_id: str) -> LearnerSummary:
        profile = self.get_or_create(learner_id)
        return LearnerSummary(
            learner_id=profile.learner_id,
            display_name=profile.display_name,
            subjects=list(profile.subjects.values()),
            preferences=profile.preferences,
            lesson_count=len(profile.lessons),
        )

    # --- queries ---------------------------------------------------------------------------

    def snapshot(self, learner_id: str, subject: str, framework_id: str, target_level: str | None) -> LearnerSnapshot:
        profile = self.get_or_create(learner_id)
        framework = self._frameworks.get(framework_id)
        state = profile.subjects.get(subject)
        now = self._clock()
        concepts = sorted((c for c in profile.concepts.values() if c.subject == subject), key=lambda c: c.concept_id)
        seen = [c for c in concepts if c.evidence_count or c.exposures]
        weak = sorted((c for c in seen if c.mastery < WEAK), key=lambda c: (c.mastery, c.concept_id))
        likely_unknown = [c.concept_id for c in concepts if c.mastery < UNKNOWN]
        return LearnerSnapshot(
            learner_id=learner_id,
            subject=subject,
            framework_id=framework.framework_id,
            framework_levels=framework.levels,
            target_level=target_level or (state.target_level if state else None),
            estimated_level=state.estimated_level if state else None,
            concept_mastery=concepts,
            known=[c.concept_id for c in concepts if c.evidence_count and c.mastery >= KNOWN],
            mastered=[c.concept_id for c in concepts if c.mastery >= MASTERED],
            weak=[c.concept_id for c in weak],
            likely_unknown=likely_unknown,
            due_for_review=[c.concept_id for c in concepts if c.next_review_at and c.next_review_at <= now],
            recommended_next=list(dict.fromkeys([c.concept_id for c in weak] + likely_unknown)),
            recent_topics=[lesson.topic for lesson in profile.lessons if lesson.subject == subject][-5:],
            recent_mistakes=[m for m in profile.mistakes if m.concept_id in {c.concept_id for c in concepts}][-10:],
            preferences=profile.preferences,
        )

    def progress(self, learner_id: str) -> LearnerProgress:
        profile = self.get(learner_id)
        now = self._clock()
        concepts = list(profile.concepts.values())
        return LearnerProgress(
            learner_id=learner_id,
            subjects=list(profile.subjects.values()),
            lessons_completed=len(profile.lessons),
            assessments_taken=len(profile.assessments),
            mastered=sorted(c.concept_id for c in concepts if c.mastery >= MASTERED),
            weak=sorted(c.concept_id for c in concepts if (c.evidence_count or c.exposures) and c.mastery < WEAK),
            due_for_review=sorted(c.concept_id for c in concepts if c.next_review_at and c.next_review_at <= now),
            average_mastery=round(sum(c.mastery for c in concepts) / len(concepts), 4) if concepts else 0.0,
        )

    # --- updates ---------------------------------------------------------------------------

    def record_lesson(self, outcome: LessonOutcome, scope: ExecutionScope) -> MasteryUpdate:
        """Apply a finished lesson to long-term memory. Idempotent per task, so a resumed task never double-counts."""
        profile = self.get_or_create(outcome.learner_id)
        request = outcome.request
        previous = next((lesson for lesson in profile.lessons if lesson.task_id == outcome.task_id), None)
        state = profile.subjects.get(request.subject)
        if previous is not None:
            return MasteryUpdate(learner_id=profile.learner_id, subject=request.subject,
                                 estimated_level=state.estimated_level if state else None,
                                 changes=previous.mastery_changes)

        now = self._clock()
        framework = self._frameworks.get(request.framework_id)
        framework.index(outcome.diagnostic.estimated_level)
        if state is None:
            state = SubjectState(subject=request.subject, framework_id=request.framework_id,
                                 target_level=request.target_level)
            profile.subjects[request.subject] = state
        state.estimated_level = outcome.diagnostic.estimated_level

        tracker = _ChangeTracker(profile, request.subject, outcome.concepts)
        concept, reasons = tracker.concept, tracker.reasons

        diagnostic = outcome.diagnostic
        if diagnostic.source == "assessment":
            profile.assessments.append(AssessmentRecord(task_id=outcome.task_id, subject=request.subject, at=now,
                                                        estimated_level=diagnostic.estimated_level,
                                                        evaluations=diagnostic.evaluations))
            for ev in diagnostic.evaluations:
                apply_evidence(concept(ev.concept_id), correct=ev.correct, difficulty=ev.difficulty, now=now)
                reasons.setdefault(ev.concept_id, []).append("correct answer" if ev.correct else "wrong answer")
                if not ev.correct:
                    profile.mistakes.append(MistakeRecord(task_id=outcome.task_id, concept_id=ev.concept_id,
                                                          question_id=ev.question_id, answer=ev.answer,
                                                          expected=ev.expected, at=now))
        for cid in outcome.taught_concept_ids:
            apply_exposure(concept(cid), now=now)
            reasons.setdefault(cid, []).append("taught in lesson")

        changes = tracker.changes()
        profile.lessons.append(LessonRecord(task_id=outcome.task_id, subject=request.subject, topic=request.topic,
                                            title=outcome.lesson_title, concept_ids=outcome.taught_concept_ids,
                                            artifact_ids=outcome.artifact_ids, mastery_changes=changes, at=now))
        profile.interactions.append(InteractionRecord(kind="lesson_completed", task_id=outcome.task_id,
                                                      detail=outcome.lesson_title, at=now))
        profile.updated_at = now
        self._repo.save(profile)
        scope.emit(EventType.LEARNER_UPDATED, learner_id=profile.learner_id, subject=request.subject,
                   estimated_level=state.estimated_level, changes=len(changes))
        return MasteryUpdate(learner_id=profile.learner_id, subject=request.subject,
                             estimated_level=state.estimated_level, changes=changes)

    def record_evaluation(self, outcome: EvaluationOutcome, scope: ExecutionScope) -> MasteryUpdate:
        """Apply post-lesson assessment evidence. Idempotent per task, like `record_lesson`."""
        profile = self.get_or_create(outcome.learner_id)
        state = profile.subjects.get(outcome.subject)
        level = state.estimated_level if state else None
        previous = next((a for a in profile.assessments if a.task_id == outcome.task_id), None)
        if previous is not None:
            return MasteryUpdate(learner_id=profile.learner_id, subject=outcome.subject, estimated_level=level,
                                 changes=previous.mastery_changes)

        self._frameworks.get(outcome.framework_id)
        now = self._clock()
        tracker = _ChangeTracker(profile, outcome.subject, outcome.concepts)
        for ev in outcome.evaluations:
            apply_evidence(tracker.concept(ev.concept_id), correct=ev.correct, difficulty=ev.difficulty, now=now)
            tracker.reasons.setdefault(ev.concept_id, []).append(
                "correct on assessment" if ev.correct else "missed on assessment")
            if not ev.correct:
                profile.mistakes.append(MistakeRecord(task_id=outcome.task_id, concept_id=ev.concept_id,
                                                      question_id=ev.question_id, answer=ev.answer,
                                                      expected=ev.expected, at=now))
        changes = tracker.changes()
        profile.assessments.append(AssessmentRecord(task_id=outcome.task_id, subject=outcome.subject, at=now,
                                                    estimated_level=level, kind="lesson_evaluation",
                                                    evaluations=outcome.evaluations, mastery_changes=changes))
        profile.interactions.append(InteractionRecord(kind="lesson_evaluated", task_id=outcome.task_id,
                                                      detail=f"lesson task {outcome.lesson_task_id}", at=now))
        profile.updated_at = now
        self._repo.save(profile)
        scope.emit(EventType.LEARNER_MASTERY_UPDATED, learner_id=profile.learner_id, subject=outcome.subject,
                   changes=[c.model_dump() for c in changes])
        scope.emit(EventType.LEARNER_UPDATED, learner_id=profile.learner_id, subject=outcome.subject,
                   estimated_level=level, changes=len(changes))
        return MasteryUpdate(learner_id=profile.learner_id, subject=outcome.subject, estimated_level=level,
                             changes=changes)


class _ChangeTracker:
    """Creates concept entries on first use and records each concept's mastery before the update."""

    def __init__(self, profile: LearnerProfile, subject: str, concepts: list[ConceptRef]) -> None:
        self._profile = profile
        self._subject = subject
        self._names = {c.concept_id: c.name for c in concepts}
        self._before: dict[str, float] = {}
        self.reasons: dict[str, list[str]] = {}

    def concept(self, cid: str) -> ConceptMastery:
        if cid not in self._profile.concepts:
            self._profile.concepts[cid] = ConceptMastery(
                concept_id=cid, name=self._names.get(cid, cid), subject=self._subject, mastery=PRIOR_MASTERY
            )
        self._before.setdefault(cid, self._profile.concepts[cid].mastery)
        return self._profile.concepts[cid]

    def changes(self) -> list[MasteryChange]:
        return [
            MasteryChange(concept_id=cid, before=round(b, 4), after=self._profile.concepts[cid].mastery,
                          reason=", ".join(self.reasons[cid]))
            for cid, b in self._before.items()
        ]
