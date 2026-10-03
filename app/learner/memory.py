"""Long-term learner memory. The only way anything reads or writes learner history.

Mastery is never written directly: every update records LearningEvidence (immutable, idempotent by id) and the
affected concepts are rebuilt from their full evidence and exposure history by the deterministic MasteryUpdater.
So the current state is always reconstructible, and a resumed or repeated task never applies evidence twice.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from app.learner.frameworks import FrameworkRegistry
from app.learner.history import (
    EvidenceRepository,
    GoalRepository,
    InMemoryEvidenceRepository,
    InMemoryGoalRepository,
    InMemoryLearningEventRepository,
    LearningEventRepository,
)
from app.learner.mastery import KNOWN, MASTERED, UNKNOWN, WEAK, MasteryUpdater
from app.learner.model import build_learner_model
from app.observability.scope import ExecutionScope
from app.schemas.common import utcnow
from app.schemas.events import EventType
from app.schemas.evaluation import EvaluationOutcome
from app.schemas.learner import (
    AnswerEvaluation,
    AssessmentRecord,
    ConceptMastery,
    InteractionRecord,
    LearnerProfile,
    LearnerProfileInput,
    LearnerProgress,
    LearnerSnapshot,
    LearnerSummary,
    LearningEvent,
    LearningEvidence,
    LearningGoal,
    LessonRecord,
    MasteryChange,
    MasteryUpdate,
    MistakeRecord,
    SubjectState,
    stable_id,
)
from app.schemas.lesson import ConceptRef, DiagnosticOutcome, LessonOutcome
from app.schemas.pedagogy import LearnerModel, PedagogyConfig


class LearnerRepository(Protocol):
    def get(self, learner_id: str) -> LearnerProfile | None: ...

    def save(self, profile: LearnerProfile) -> None: ...


class UnknownLearner(KeyError):
    pass


class UnknownGoal(KeyError):
    pass


class LearnerMemoryService:
    def __init__(
        self,
        repository: LearnerRepository,
        frameworks: FrameworkRegistry,
        clock: Callable[[], datetime] = utcnow,
        *,
        evidence: EvidenceRepository | None = None,
        events: LearningEventRepository | None = None,
        goals: GoalRepository | None = None,
        config: PedagogyConfig | None = None,
    ) -> None:
        self._repo = repository
        self._frameworks = frameworks
        self._clock = clock
        self._evidence = evidence if evidence is not None else InMemoryEvidenceRepository()
        self._events = events if events is not None else InMemoryLearningEventRepository()
        self._goals = goals if goals is not None else InMemoryGoalRepository()
        self.config = config or PedagogyConfig()
        self.updater = MasteryUpdater(self.config.mastery)

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

    # --- goals -----------------------------------------------------------------------------

    def save_goal(self, goal: LearningGoal) -> LearningGoal:
        self.get_or_create(goal.learner_id)
        self._goals.save(goal)
        return goal

    def goal(self, goal_id: str) -> LearningGoal:
        goal = self._goals.get(goal_id)
        if goal is None:
            raise UnknownGoal(goal_id)
        return goal

    def goals(self, learner_id: str, domain: str | None = None) -> list[LearningGoal]:
        return sorted((g for g in self._goals.for_learner(learner_id) if domain is None or g.domain == domain),
                      key=lambda g: (g.priority, g.deadline is None, g.deadline, g.goal_id))

    def resolve_goal(self, learner_id: str, domain: str, topic: str, topic_concepts: list[str],
                     target_level: str | None, goal_id: str | None = None) -> LearningGoal:
        """The goal a lesson is planned against: the named one; else the learner's highest-priority active goal in
        the domain that the topic serves; else an implicit goal for the topic (stored, so it is stable)."""
        if goal_id is not None:
            goal = self.goal(goal_id)
            if goal.learner_id != learner_id:
                raise UnknownGoal(goal_id)
            return goal
        wanted = set(topic_concepts)
        for goal in self.goals(learner_id, domain):
            if goal.status == "active" and wanted & set(goal.target_concepts):
                return goal
        if not topic_concepts:
            raise ValueError(f"no concepts known for {domain}/{topic}; cannot derive a learning goal")
        implicit = LearningGoal(
            goal_id=stable_id("goal", learner_id, domain, topic, target_level or ""), learner_id=learner_id,
            domain=domain, target_level=target_level, target_concepts=list(dict.fromkeys(topic_concepts)),
            description=f"Learn {topic}" + (f" at level {target_level}" if target_level else ""))
        existing = self._goals.get(implicit.goal_id)
        return existing if existing is not None else self.save_goal(implicit)

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

    def learner_model(self, learner_id: str, subject: str, framework_id: str, target_level: str | None = None,
                      universe: list[str] | None = None) -> LearnerModel:
        self._frameworks.get(framework_id)
        profile = self.get_or_create(learner_id)
        return build_learner_model(
            profile, domain=subject, framework_id=framework_id, target_level=target_level,
            evidence=self._evidence.for_learner(learner_id), events=self._events.for_learner(learner_id),
            goals=self._goals.for_learner(learner_id), universe=universe or [], config=self.config, now=self._clock())

    def evidence(self, learner_id: str, concept_id: str | None = None) -> list[LearningEvidence]:
        return self._evidence.for_learner(learner_id, concept_id)

    def history(self, learner_id: str) -> list[LearningEvent]:
        return sorted(self._events.for_learner(learner_id), key=lambda e: (e.at, e.event_id))

    def rebuild(self, learner_id: str, subject: str) -> dict[str, ConceptMastery]:
        """Every concept state of a subject recomputed from evidence and exposures alone (no stored state used)."""
        profile = self.get_or_create(learner_id)
        ids = {c.concept_id for c in profile.concepts.values() if c.subject == subject}
        names = {cid: profile.concepts[cid].name for cid in ids}
        return {cid: self._replay(learner_id, cid, names[cid], subject) for cid in sorted(ids)}

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

    def record_evidence(self, learner_id: str, subject: str, evidence: list[LearningEvidence],
                        concepts: list[ConceptRef], scope: ExecutionScope | None = None) -> MasteryUpdate:
        """Record evidence from any source (exercise, manual placement, ...) and update mastery from it."""
        if any(e.learner_id != learner_id for e in evidence):
            raise ValueError("evidence must belong to the learner it is recorded for")
        profile = self.get_or_create(learner_id)
        tracker = _ChangeTracker(profile, subject, concepts, self.config.mastery.prior_mastery)
        now = self._clock()
        for e in evidence:
            if e.source_type == "exercise":
                self._events.add(LearningEvent.create(learner_id, "exercise_completed", e.timestamp,
                                                      key=e.source_ref, subject=subject, concept_ids=[e.concept_id]))
        applied = self._apply(profile, tracker, evidence, key=f"evidence:{evidence[0].evidence_id}" if evidence else "")
        profile.updated_at = now
        self._repo.save(profile)
        changes = tracker.changes()
        if scope is not None:
            scope.emit(EventType.LEARNER_EVIDENCE_RECORDED, learner_id=learner_id, subject=subject,
                       evidence=len(applied), changes=[c.model_dump() for c in changes])
        state = profile.subjects.get(subject)
        return MasteryUpdate(learner_id=learner_id, subject=subject, estimated_level=state.estimated_level if state
                             else None, changes=changes, evidence=evidence)

    def record_diagnostic(self, outcome: DiagnosticOutcome, scope: ExecutionScope) -> MasteryUpdate:
        """Turn a finished diagnostic into evidence and mastery, before the lesson is planned. Idempotent per task.
        The diagnostic agent graded the answers; it never touches mastery."""
        profile = self.get_or_create(outcome.learner_id)
        request = outcome.request
        framework = self._frameworks.get(request.framework_id)
        framework.index(outcome.diagnostic.estimated_level)
        previous = self._assessment(profile, outcome.task_id, "diagnostic")
        if previous is not None:
            return self._replayed_update(profile, request.subject, previous)
        state = self._subject_state(profile, request.subject, request.framework_id, request.target_level)
        state.estimated_level = outcome.diagnostic.estimated_level
        tracker = _ChangeTracker(profile, request.subject, outcome.concepts, self.config.mastery.prior_mastery)
        evidence = self._record_assessment(profile, tracker, outcome.task_id, request.subject, "diagnostic",
                                           outcome.diagnostic.estimated_level, outcome.diagnostic.evaluations,
                                           source=outcome.diagnostic.source)
        now = self._clock()
        profile.updated_at = now
        self._repo.save(profile)
        changes = tracker.changes()
        if evidence:
            scope.emit(EventType.LEARNER_EVIDENCE_RECORDED, learner_id=profile.learner_id, subject=request.subject,
                       evidence=len(evidence), changes=[c.model_dump() for c in changes])
        return MasteryUpdate(learner_id=profile.learner_id, subject=request.subject,
                             estimated_level=state.estimated_level, changes=changes, evidence=evidence)

    def record_lesson(self, outcome: LessonOutcome, scope: ExecutionScope) -> MasteryUpdate:
        """Apply a finished lesson to long-term memory. Idempotent per task, so a resumed task never double-counts.

        The diagnostic's evidence is recorded here unless `record_diagnostic` already did; being taught a concept
        counts as an exposure (a review is scheduled) but never as evidence of mastery."""
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
        state = self._subject_state(profile, request.subject, request.framework_id, request.target_level)
        state.estimated_level = outcome.diagnostic.estimated_level

        tracker = _ChangeTracker(profile, request.subject, outcome.concepts, self.config.mastery.prior_mastery)
        evidence: list[LearningEvidence] = []
        if self._assessment(profile, outcome.task_id, "diagnostic") is None:
            evidence = self._record_assessment(profile, tracker, outcome.task_id, request.subject, "diagnostic",
                                               outcome.diagnostic.estimated_level, outcome.diagnostic.evaluations,
                                               source=outcome.diagnostic.source)
        self._events.add(LearningEvent.create(profile.learner_id, "lesson_completed", now, key=outcome.task_id,
                                              subject=request.subject, task_id=outcome.task_id,
                                              concept_ids=outcome.taught_concept_ids,
                                              data={"title": outcome.lesson_title}))
        for cid in outcome.taught_concept_ids:
            tracker.touch(cid)
            tracker.reasons.setdefault(cid, []).append("taught in lesson")
        self._recompute(profile, tracker, outcome.taught_concept_ids)

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
                             estimated_level=state.estimated_level, changes=changes, evidence=evidence)

    def record_evaluation(self, outcome: EvaluationOutcome, scope: ExecutionScope) -> MasteryUpdate:
        """Apply post-lesson assessment evidence. Idempotent per task, like `record_lesson`."""
        profile = self.get_or_create(outcome.learner_id)
        state = profile.subjects.get(outcome.subject)
        level = state.estimated_level if state else None
        previous = self._assessment(profile, outcome.task_id, "lesson_evaluation")
        if previous is not None:
            return self._replayed_update(profile, outcome.subject, previous)

        self._frameworks.get(outcome.framework_id)
        now = self._clock()
        tracker = _ChangeTracker(profile, outcome.subject, outcome.concepts, self.config.mastery.prior_mastery)
        evidence = self._record_assessment(profile, tracker, outcome.task_id, outcome.subject, "lesson_evaluation",
                                           level, outcome.evaluations, source="assessment",
                                           detail=f"lesson task {outcome.lesson_task_id}")
        changes = tracker.changes()
        profile.updated_at = now
        self._repo.save(profile)
        scope.emit(EventType.LEARNER_MASTERY_UPDATED, learner_id=profile.learner_id, subject=outcome.subject,
                   changes=[c.model_dump() for c in changes], evidence=len(evidence))
        scope.emit(EventType.LEARNER_UPDATED, learner_id=profile.learner_id, subject=outcome.subject,
                   estimated_level=level, changes=len(changes))
        return MasteryUpdate(learner_id=profile.learner_id, subject=outcome.subject, estimated_level=level,
                             changes=changes, evidence=evidence)

    # --- internals -------------------------------------------------------------------------

    def _subject_state(self, profile: LearnerProfile, subject: str, framework_id: str,
                       target_level: str | None) -> SubjectState:
        state = profile.subjects.get(subject)
        if state is None:
            state = SubjectState(subject=subject, framework_id=framework_id, target_level=target_level)
            profile.subjects[subject] = state
        return state

    @staticmethod
    def _assessment(profile: LearnerProfile, task_id: str, kind: str) -> AssessmentRecord | None:
        return next((a for a in profile.assessments if a.task_id == task_id and a.kind == kind), None)

    def _replayed_update(self, profile: LearnerProfile, subject: str, record: AssessmentRecord) -> MasteryUpdate:
        """The result of an assessment that was already recorded: same changes, same evidence."""
        kind = "diagnostic" if record.kind == "diagnostic" else "evaluation"
        ids = {LearningEvidence.id_for(profile.learner_id, kind, f"{record.task_id}/{e.question_id}", e.concept_id)
               for e in record.evaluations}
        state = profile.subjects.get(subject)
        return MasteryUpdate(learner_id=profile.learner_id, subject=subject,
                             estimated_level=state.estimated_level if state else None, changes=record.mastery_changes,
                             evidence=[e for e in self._evidence.for_learner(profile.learner_id)
                                       if e.evidence_id in ids])

    def _record_assessment(self, profile: LearnerProfile, tracker: _ChangeTracker, task_id: str, subject: str,
                           kind: str, level: str | None, evaluations: list[AnswerEvaluation], *, source: str,
                           detail: str = "") -> list[LearningEvidence]:
        """Graded answers -> evidence -> mastery, plus the assessment record, mistakes and learning events."""
        if source != "assessment":  # concluded from memory: nothing was asked, nothing is evidence
            return []
        now = self._clock()
        evidence_kind = "diagnostic" if kind == "diagnostic" else "evaluation"
        evidence = [LearningEvidence.from_answer(profile.learner_id, evidence_kind, task_id, ev, now)
                    for ev in evaluations]
        reasons = ("correct answer", "wrong answer") if kind == "diagnostic" else (
            "correct on assessment", "missed on assessment")
        for ev in evaluations:
            tracker.reasons.setdefault(ev.concept_id, []).append(reasons[0] if ev.correct else reasons[1])
            if not ev.correct:
                profile.mistakes.append(MistakeRecord(task_id=task_id, concept_id=ev.concept_id,
                                                      question_id=ev.question_id, answer=ev.answer,
                                                      expected=ev.expected, at=now))
        event_type = "diagnostic_completed" if kind == "diagnostic" else "evaluation_completed"
        self._events.add(LearningEvent.create(
            profile.learner_id, event_type, now, key=task_id, subject=subject, task_id=task_id,
            concept_ids=sorted({ev.concept_id for ev in evaluations}),
            data={"questions": len(evaluations), "correct": sum(ev.correct for ev in evaluations)}))
        self._apply(profile, tracker, evidence, key=task_id)
        profile.assessments.append(AssessmentRecord(task_id=task_id, subject=subject, at=now, estimated_level=level,
                                                    kind=kind, evaluations=evaluations,
                                                    mastery_changes=tracker.changes()))
        if kind != "diagnostic":
            profile.interactions.append(InteractionRecord(kind="lesson_evaluated", task_id=task_id, detail=detail,
                                                          at=now))
        return evidence

    def _apply(self, profile: LearnerProfile, tracker: _ChangeTracker, evidence: list[LearningEvidence],
               key: str) -> list[LearningEvidence]:
        """Store the evidence (immutable; identical re-records are no-ops), then rebuild every touched concept from
        its full history, so a crash between the two steps is repaired by simply recording again."""
        new = [e for e in evidence if self._evidence.add(e)]
        touched = list(dict.fromkeys(e.concept_id for e in evidence))
        for cid in touched:
            tracker.touch(cid)
        before = {cid: profile.concepts[cid].model_copy() for cid in touched}
        self._recompute(profile, tracker, touched)
        target = self.config.mastery_target
        for cid in touched:
            old, new_state = before[cid], profile.concepts[cid]
            mine = [e for e in new if e.concept_id == cid]
            if not mine:
                continue
            if old.mastery < target <= new_state.mastery:
                self._events.add(LearningEvent.create(profile.learner_id, "concept_mastered", mine[-1].timestamp,
                                                      key=f"{key}:{cid}", subject=tracker.subject,
                                                      concept_ids=[cid], data={"mastery": new_state.mastery}))
            if old.next_review_at is not None and old.next_review_at <= mine[0].timestamp:
                self._events.add(LearningEvent.create(profile.learner_id, "concept_reviewed", mine[0].timestamp,
                                                      key=f"{key}:{cid}", subject=tracker.subject,
                                                      concept_ids=[cid], data={"due": old.next_review_at.isoformat()}))
        return new

    def _recompute(self, profile: LearnerProfile, tracker: _ChangeTracker, concept_ids: list[str]) -> None:
        for cid in concept_ids:
            concept = profile.concepts[cid]
            profile.concepts[cid] = self._replay(profile.learner_id, cid, concept.name, concept.subject)

    def _replay(self, learner_id: str, concept_id: str, name: str, subject: str) -> ConceptMastery:
        exposures = [e.at for e in self.history(learner_id)
                     if e.type == "lesson_completed" and concept_id in e.concept_ids]
        seed = self.updater.initial(concept_id, name, subject)
        return self.updater.replay(seed, self._evidence.for_learner(learner_id, concept_id), exposures)


class _ChangeTracker:
    """Creates concept entries on first use and records each concept's mastery before the update."""

    def __init__(self, profile: LearnerProfile, subject: str, concepts: list[ConceptRef], prior: float) -> None:
        self._profile = profile
        self.subject = subject
        self._prior = prior
        self._names = {c.concept_id: c.name for c in concepts}
        self._before: dict[str, float] = {}
        self.reasons: dict[str, list[str]] = {}

    def touch(self, cid: str) -> ConceptMastery:
        if cid not in self._profile.concepts:
            self._profile.concepts[cid] = ConceptMastery(
                concept_id=cid, name=self._names.get(cid, cid), subject=self.subject,
                mastery=self._prior,
            )
        self._before.setdefault(cid, self._profile.concepts[cid].mastery)
        return self._profile.concepts[cid]

    def changes(self) -> list[MasteryChange]:
        return [
            MasteryChange(concept_id=cid, before=round(b, 4), after=self._profile.concepts[cid].mastery,
                          reason=", ".join(self.reasons.get(cid, [])) or "recorded evidence")
            for cid, b in self._before.items()
        ]
