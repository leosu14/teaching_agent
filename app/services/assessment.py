"""The AssessmentService: the one place learner answers are graded.

Every caller grades through `assess`: the assessment API (`/assessment-items/{id}/attempts`), interactive teaching
sessions and the evaluation workflow (through the `assessment.grade` tool). It

- resolves the item (and its rubric) and the attempt (idempotent per attempt id; a different answer under the same
  attempt id is a conflict);
- grades with the AssessmentEngine (`app/assessment/engine.py`): normalisation, acceptable answers, deterministic
  rules, rubric grading, and the semantic grader only for free text that deterministic grading cannot decide;
- stores the attempt and its immutable grade together, then publishes the ASSESSMENT_* artifacts and events.

The service never updates mastery itself. An API attempt's validated grade becomes LearningEvidence recorded through
learner memory's existing MasteryUpdater (UNCERTAIN grades never); a teaching session hands the grade to its own
deterministic engine (its evidence reaches the updater when the session completes); the evaluation workflow records
its grades through its existing `update_mastery` node. Objective progress and the next action come from the
curriculum engine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.assessment.engine import AssessmentEngine
from app.assessment.errors import (
    AssessmentItemNotFound,
    AttemptConflict,
    AttemptExists,
    AttemptNotFound,
    InvalidAssessmentRequest,
    ItemConflict,
)
from app.assessment.repository import AssessmentRepository
from app.learner.memory import LearnerMemoryService
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.pedagogy.knowledge import KnowledgeBase
from app.runtime.interaction.grader import AgentSemanticGrader
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.assessment import (
    AssessmentAttempt,
    AssessmentConfig,
    AssessmentGrade,
    AssessmentItem,
    AssessmentOutcome,
    AssessmentResult,
    AssessmentRubric,
    AssessRequest,
    AttemptOutcome,
    AttemptResult,
    AttemptSubmission,
    AttemptView,
    BatchAssessmentRequest,
    BatchAssessmentResult,
    ContextPassage,
    GradingContext,
    PublicGrade,
    RegisterAssessmentItem,
)
from app.schemas.common import new_id, utcnow
from app.schemas.events import EventType
from app.schemas.learner import LearningEvidence, graded_correctness, stable_id
from app.services.curriculum import CurriculumService
from app.services.lessons import LessonMaterial, LessonNotReady, load_lesson
from app.services.tasks import TaskService
from app.storage.repositories import NotFound

__all__ = ["AssessmentItemNotFound", "AssessmentService", "AttemptConflict", "AttemptNotFound",
           "InvalidAssessmentRequest", "ItemConflict"]

NODE_ID = "assessment"
PROVIDER = "teaching-agent"


def item_artifact(item_id: str) -> str:
    return f"assessment_item.{item_id}"


def rubric_artifact(rubric_id: str) -> str:
    return f"assessment_rubric.{rubric_id}"


def grade_artifact(grade_id: str) -> str:
    return f"assessment_grade.{grade_id}"


def feedback_artifact(grade_id: str) -> str:
    return f"assessment_feedback.{grade_id}"


def grade_id_for(attempt_id: str, item_id: str) -> str:
    return stable_id("agrade", attempt_id, item_id)


class AssessmentService:
    def __init__(self, repository: AssessmentRepository, grader: AgentSemanticGrader | None, *,
                 artifacts: ArtifactService, memory: LearnerMemoryService, knowledge: KnowledgeBase,
                 curriculum: CurriculumService, tasks: TaskService, events: EventBus,
                 config: AssessmentConfig | None = None, clock: Callable[[], datetime] = utcnow) -> None:
        self._repo = repository
        self._grader = grader
        self._artifacts = artifacts
        self._memory = memory
        self._knowledge = knowledge
        self._curriculum = curriculum
        self._tasks = tasks
        self._events = events
        self.config = config or AssessmentConfig()
        self.engine = AssessmentEngine(self.config)
        self._clock = clock

    @property
    def repository(self) -> AssessmentRepository:
        return self._repo

    # --- items ---------------------------------------------------------------------------------------------------

    def register(self, data: RegisterAssessmentItem) -> AssessmentItem:
        """Store an item on a lesson (with its rubric). Items and rubrics are immutable: registering the same one
        again is a no-op, a different one under the same id a conflict."""
        item = data.item
        assert item.lesson_id is not None
        m = self._material(item.lesson_id)
        if not any(s.concept_id == item.concept_id for s in m.lesson.sections):
            raise InvalidAssessmentRequest(f"the lesson does not teach concept {item.concept_id}")
        rubric = data.rubric
        if rubric is None and item.rubric_id is not None:
            rubric = self._repo.rubric(item.rubric_id)
            if rubric is None:
                raise InvalidAssessmentRequest(f"rubric {item.rubric_id} is not registered")
        if rubric is not None:
            self._repo.save_rubric(rubric)
        self._repo.save_item(item.model_copy(update={"lesson_id": m.artifact.artifact_id}))
        stored = self.item(item.assessment_item_id)
        scope = self._scope(m.task.task_id, None)
        self._store_definitions(stored, rubric, m.task.task_id, scope)
        return stored

    def item(self, item_id: str) -> AssessmentItem:
        found = self._repo.item(item_id)
        if found is None:
            raise AssessmentItemNotFound(f"assessment item {item_id} not found")
        return found

    # --- grading -------------------------------------------------------------------------------------------------

    async def assess(self, request: AssessRequest, *, scope: ExecutionScope | None = None,
                     context: GradingContext | None = None) -> AssessmentResult:
        """Grade one answer, once. The same attempt id with the same answer returns the stored grade (nothing is
        graded or stored again); with a different answer or item it is a conflict."""
        item, digest = request.item, self._digest(request)
        existing = self._repo.attempt(request.attempt_id)
        if existing is not None:
            return self._replay(existing, digest, scope)
        self._repo.save_item(item)
        rubric = request.rubric
        if rubric is None and item.rubric_id is not None:
            rubric = self._repo.rubric(item.rubric_id)
            if rubric is None:
                raise InvalidAssessmentRequest(f"rubric {item.rubric_id} is not registered")
        if rubric is not None:
            self._repo.save_rubric(rubric)
        scope = self._scope(request.task_id, scope)
        grade_id = grade_id_for(request.attempt_id, item.assessment_item_id)
        scope.events.emit(EventType.ASSESSMENT_STARTED, task_id=scope.task_id, node_id=scope.node_id,
                          event_id=stable_id("aevt", request.attempt_id, "started"), attempt_id=request.attempt_id,
                          assessment_item_id=item.assessment_item_id, concept_id=item.concept_id,
                          response_type=item.response_type.value, source=request.source)
        if context is None:
            context = self._context(item)
        grader = self._grader.bind(task_id=request.task_id, scope=scope) if self._grader is not None else None
        now = self._clock()
        grade = await self.engine.grade(item, rubric, request.answer, grade_id=grade_id,
                                        attempt_id=request.attempt_id, at=now, context=context, grader=grader)
        attempt = AssessmentAttempt(
            attempt_id=request.attempt_id, assessment_item_id=item.assessment_item_id, learner_id=request.learner_id,
            attempt_number=1, learner_answer=request.answer, answer_hash=digest, submitted_at=now, grade_id=grade_id,
            source=request.source, source_ref=request.source_ref, task_id=request.task_id)
        try:
            stored = self._repo.add_attempt(attempt, grade)
        except AttemptExists:  # the same attempt, submitted twice at once: graded and stored once
            current = self._repo.attempt(request.attempt_id)
            assert current is not None
            return self._replay(current, digest, scope)
        rubric = rubric or self.engine.rubric_for(item, None)
        artifact_ids = self._publish(stored, grade, item, rubric if grade.rubric_id else None, scope)
        if request.source != "api":  # an API attempt completes after its evidence is recorded
            stored = self._complete(stored, grade, AttemptOutcome(artifact_ids=artifact_ids), scope,
                                    mastery_updated=False)
        return AssessmentResult(attempt=stored, grade=grade, replayed=False, artifact_ids=artifact_ids)

    async def assess_batch(self, request: BatchAssessmentRequest, scope: ExecutionScope) -> BatchAssessmentResult:
        """The evaluation workflow's grading step: every answer through `assess`, in order."""
        return BatchAssessmentResult(results=[await self.assess(r, scope=scope) for r in request.requests])

    def _replay(self, existing: AssessmentAttempt, digest: str, scope: ExecutionScope | None) -> AssessmentResult:
        if existing.answer_hash != digest:
            raise AttemptConflict(f"attempt {existing.attempt_id} was already submitted with a different answer")
        grade = self._repo.grade(existing.grade_id)
        assert grade is not None
        artifact_ids = existing.outcome.artifact_ids if existing.outcome else {}
        if existing.completed_at is None:  # a crash before it was published: finish it (nothing is stored twice)
            item = self.item(existing.assessment_item_id)
            scope = self._scope(existing.task_id, scope)
            rubric = (self._repo.rubric(grade.rubric_id) or self.engine.rubric_for(item, None)) if grade.rubric_id \
                else None
            artifact_ids = self._publish(existing, grade, item, rubric, scope)
            if existing.source != "api":
                existing = self._complete(existing, grade, AttemptOutcome(artifact_ids=artifact_ids), scope,
                                          mastery_updated=False)
        return AssessmentResult(attempt=existing, grade=grade, replayed=True, artifact_ids=artifact_ids)

    @staticmethod
    def _digest(request: AssessRequest) -> str:
        return hashlib.sha256("\x1f".join([request.item.assessment_item_id, request.learner_id,
                                            request.answer]).encode("utf-8")).hexdigest()

    # --- the assessment API --------------------------------------------------------------------------------------

    async def submit_attempt(self, item_id: str, data: AttemptSubmission) -> AttemptResult:
        """Grade an answer to a registered item; a graded (not UNCERTAIN) answer becomes LearningEvidence recorded by
        learner memory's deterministic updater, then the objective's progress and the next learning action are read
        from the curriculum. Idempotent per attempt id."""
        item = self.item(item_id)
        self._memory.get(data.learner_id)  # unknown learners are rejected (404)
        m = self._material(item.lesson_id) if item.lesson_id else None
        if m is None:
            raise InvalidAssessmentRequest(f"item {item_id} has no lesson")
        if m.task.learner_id != data.learner_id:
            raise InvalidAssessmentRequest("the item's lesson belongs to another learner")
        result = await self.assess(AssessRequest(
            item=item, learner_id=data.learner_id, answer=data.answer, attempt_id=data.attempt_id or new_id("att"),
            task_id=m.task.task_id, source="api"))
        attempt = result.attempt
        if attempt.completed_at is None:  # new, or a crash left it unfinished: record its evidence (idempotent)
            outcome = await self._record(attempt, result.grade, item, m, result.artifact_ids)
            attempt = self._complete(attempt, result.grade, outcome, self._scope(m.task.task_id, None),
                                     mastery_updated=bool(outcome.evidence_ids))
        return self._attempt_result(attempt, result.grade, replayed=result.replayed)

    def attempt(self, attempt_id: str) -> AttemptView:
        return self._view(self._attempt(attempt_id))

    def grade_for(self, attempt_id: str) -> PublicGrade:
        grade = self._repo.grade(self._attempt(attempt_id).grade_id)
        assert grade is not None
        return PublicGrade.of(grade)

    def stored_grade(self, attempt_id: str) -> AssessmentGrade:
        grade = self._repo.grade(self._attempt(attempt_id).grade_id)
        assert grade is not None
        return grade

    def attempts(self, item_id: str, learner_id: str) -> list[AttemptView]:
        self.item(item_id)
        return [self._view(a) for a in self._repo.attempts(item_id, learner_id)]

    def _attempt(self, attempt_id: str) -> AssessmentAttempt:
        found = self._repo.attempt(attempt_id)
        if found is None:
            raise AttemptNotFound(f"assessment attempt {attempt_id} not found")
        return found

    def _attempt_result(self, attempt: AssessmentAttempt, grade: AssessmentGrade, *, replayed: bool) -> AttemptResult:
        outcome = attempt.outcome or AttemptOutcome()
        return AttemptResult(attempt=self._view(attempt), grade=PublicGrade.of(grade), replayed=replayed,
                             mastery_changes=outcome.mastery_changes, objective_progress=outcome.objective_progress,
                             learning_action=outcome.learning_action)

    def _view(self, a: AssessmentAttempt) -> AttemptView:
        grade = self._repo.grade(a.grade_id)
        assert grade is not None
        evidence = a.outcome.evidence_ids if a.outcome else []
        return AttemptView(attempt_id=a.attempt_id, assessment_item_id=a.assessment_item_id, learner_id=a.learner_id,
                           attempt_number=a.attempt_number, learner_answer=a.learner_answer,
                           submitted_at=a.submitted_at, grade_id=a.grade_id, source=a.source, outcome=grade.outcome,
                           completed=a.completed_at is not None, mastery_updated=bool(evidence),
                           evidence_ids=evidence, retry_recommended=grade.outcome == AssessmentOutcome.UNCERTAIN)

    async def _record(self, attempt: AssessmentAttempt, grade: AssessmentGrade, item: AssessmentItem,
                      m: LessonMaterial, artifact_ids: dict[str, str]) -> AttemptOutcome:
        """Grade -> evidence -> the existing mastery updater -> objective progress -> next learning action."""
        scope = self._scope(m.task.task_id, None)
        domain = m.request.subject
        artifact_ids = dict(artifact_ids) or self._artifact_ids(m.task.task_id, item, grade)
        evidence: list[LearningEvidence] = []
        changes: list[dict] = []
        if grade.counts:
            correctness, score = graded_correctness(grade.outcome.value, grade.fraction)
            ref = f"{item.assessment_item_id}/{attempt.attempt_id}"
            evidence.append(LearningEvidence(
                evidence_id=LearningEvidence.id_for(attempt.learner_id, "assessment", ref, item.concept_id),
                learner_id=attempt.learner_id, concept_id=item.concept_id, source_type="assessment", source_ref=ref,
                correctness=correctness, score=score, difficulty=item.difficulty, timestamp=attempt.submitted_at,
                metadata={"attempt_id": attempt.attempt_id, "assessment_item_id": item.assessment_item_id,
                          "grade_id": grade.grade_id, "outcome": grade.outcome.value,
                          "grader_type": grade.grader_type.value,
                          "misconceptions": [mc.model_dump(mode="json") for mc in grade.misconceptions]}))
            graph = await self._knowledge.graph(domain)
            update = self._memory.record_evidence(attempt.learner_id, domain, evidence,
                                                  [c.ref() for c in graph.concepts()], scope)
            changes = [c.model_dump(mode="json") for c in update.changes]
            parent = artifact_ids.get("grade")
            stored = self._artifacts.store(
                task_id=m.task.task_id, name=f"learning_evidence.{attempt.attempt_id}",
                type=ArtifactType.LEARNING_EVIDENCE, media_type="application/json",
                content=("[\n" + ",\n".join(e.model_dump_json(indent=2) for e in evidence) + "\n]").encode("utf-8"),
                provider=PROVIDER, parent_ids=[parent] if parent else [],
                metadata={"source": "assessment", "items": len(evidence), "concepts": [item.concept_id],
                          "attempt_id": attempt.attempt_id, "misconceptions": len(grade.misconceptions)},
                scope=scope)
            artifact_ids["learning_evidence"] = stored.artifact_id
        progress = self._objective_progress(attempt.learner_id, item.objective_id)
        action = await self._curriculum.next_action(attempt.learner_id)
        return AttemptOutcome(evidence_ids=[e.evidence_id for e in evidence], mastery_changes=changes,
                              objective_progress=progress, learning_action=action.model_dump(mode="json"),
                              artifact_ids=artifact_ids)

    def _objective_progress(self, learner_id: str, objective_id: str | None) -> dict | None:
        if objective_id is None:
            return None
        for goal in self._memory.goals(learner_id):
            curriculum = self._curriculum.curriculum(goal.goal_id)
            if curriculum is not None and any(o.objective_id == objective_id for o in curriculum.objectives):
                return curriculum.progress.of(objective_id).model_dump(mode="json")
        return None

    # --- context -------------------------------------------------------------------------------------------------

    def _context(self, item: AssessmentItem) -> GradingContext:
        """Only what grading this item needs: the lesson sections on its concept and the research findings they
        cite. Nothing about the learner."""
        if item.lesson_id is None:
            return GradingContext()
        try:
            m = self._material(item.lesson_id)
        except (NotFound, InvalidAssessmentRequest):
            return GradingContext()
        return grading_context(m, item.concept_id, self.config.max_context_passages)

    def _material(self, lesson_id: str) -> LessonMaterial:
        try:
            return load_lesson(self._artifacts, self._tasks, lesson_id)
        except LessonNotReady as exc:
            raise InvalidAssessmentRequest(str(exc)) from None
        except (NotFound, KeyError):
            raise InvalidAssessmentRequest(f"lesson {lesson_id} not found") from None

    # --- publishing ----------------------------------------------------------------------------------------------

    def _scope(self, task_id: str | None, scope: ExecutionScope | None) -> ExecutionScope:
        if scope is not None:
            return scope
        return ExecutionScope(events=self._events, usage=UsageLedger(), task_id=task_id, node_id=NODE_ID)

    def _store(self, task_id: str, name: str, type: ArtifactType, content: dict, parents: list[str], metadata: dict,
               scope: ExecutionScope) -> Artifact:
        return self._artifacts.store(task_id=task_id, name=name, type=type, media_type="application/json",
                                     content=json.dumps(content, ensure_ascii=False, indent=2,
                                                        default=str).encode("utf-8"),
                                     provider=PROVIDER, parent_ids=parents, metadata=metadata, scope=scope)

    def _store_definitions(self, item: AssessmentItem, rubric: AssessmentRubric | None, task_id: str,
                           scope: ExecutionScope) -> tuple[Artifact, Artifact | None]:
        """LESSON -> ASSESSMENT_ITEM, and the ASSESSMENT_RUBRIC it is graded with (content-deduplicated)."""
        lesson = self._lesson_artifact(item.lesson_id)
        stored_item = self._store(task_id, item_artifact(item.assessment_item_id), ArtifactType.ASSESSMENT_ITEM,
                                  item.model_dump(mode="json"), [lesson] if lesson else [],
                                  {"concept_id": item.concept_id, "objective_id": item.objective_id,
                                   "response_type": item.response_type.value, "rubric_id": item.rubric_id}, scope)
        stored_rubric = None
        if rubric is not None:
            stored_rubric = self._store(task_id, rubric_artifact(rubric.rubric_id), ArtifactType.ASSESSMENT_RUBRIC,
                                        rubric.model_dump(mode="json"), [],
                                        {"criteria": len(rubric.criteria),
                                         "passing_threshold": rubric.passing_threshold,
                                         "deterministic": rubric.deterministic}, scope)
        return stored_item, stored_rubric

    def _lesson_artifact(self, lesson_id: str | None) -> str | None:
        if lesson_id is None:
            return None
        try:
            found = self._artifacts.get(lesson_id)
        except (NotFound, KeyError):
            return None
        return found.artifact_id if found.type == ArtifactType.LESSON else None

    def _publish(self, attempt: AssessmentAttempt, grade: AssessmentGrade, item: AssessmentItem,
                 rubric: AssessmentRubric | None, scope: ExecutionScope) -> dict[str, str]:
        """ASSESSMENT_ITEM (+ ASSESSMENT_RUBRIC) -> ASSESSMENT_GRADE -> ASSESSMENT_FEEDBACK, then the events. Stable
        ids and content-addressed artifacts: publishing again (after a crash) stores and emits nothing twice."""
        task_id = attempt.task_id or scope.task_id
        assert task_id is not None
        stored_item, stored_rubric = self._store_definitions(item, rubric, task_id, scope)
        parents = [stored_item.artifact_id, *([stored_rubric.artifact_id] if stored_rubric else [])]
        stored_grade = self._store(task_id, grade_artifact(grade.grade_id), ArtifactType.ASSESSMENT_GRADE,
                                   grade.model_dump(mode="json"), parents,
                                   {"attempt_id": attempt.attempt_id, "assessment_item_id": item.assessment_item_id,
                                    "outcome": grade.outcome.value, "score": grade.score,
                                    "grader_type": grade.grader_type.value, "concept_id": item.concept_id,
                                    "attempt_number": attempt.attempt_number}, scope)
        stored_feedback = self._store(task_id, feedback_artifact(grade.grade_id), ArtifactType.ASSESSMENT_FEEDBACK,
                                      grade.feedback.model_dump(mode="json"), [stored_grade.artifact_id],
                                      {"grade_id": grade.grade_id, "outcome": grade.outcome.value}, scope)
        data = {"attempt_id": attempt.attempt_id, "assessment_item_id": item.assessment_item_id,
                "grade_id": grade.grade_id, "concept_id": item.concept_id, "attempt_number": attempt.attempt_number,
                "source": attempt.source}
        summary = {**data, "outcome": grade.outcome.value, "score": grade.score, "max_score": grade.max_score,
                   "percentage": grade.percentage, "confidence": grade.confidence,
                   "grader_type": grade.grader_type.value, "llm_calls": grade.grader.llm_calls,
                   **({"provider": grade.grader.provider, "model": grade.grader.model,
                       "input_tokens": grade.grader.input_tokens, "output_tokens": grade.grader.output_tokens,
                       "estimated_cost_usd": grade.grader.estimated_cost_usd} if grade.grader.llm_calls else {})}
        if grade.outcome == AssessmentOutcome.UNCERTAIN:
            self._emit(scope, EventType.ASSESSMENT_UNCERTAIN, attempt, "uncertain", **summary,
                       reason=grade.uncertainty_reason)
        else:
            self._emit(scope, EventType.ASSESSMENT_GRADED, attempt, "graded", **summary)
        if attempt.source != "teaching_session":  # a session records them as its own evidence and events
            for i, mc in enumerate(grade.misconceptions):
                self._emit(scope, EventType.MISCONCEPTION_DETECTED, attempt, f"misconception:{i}", **data,
                           misconception_type=mc.type, misconception_concept_id=mc.concept_id,
                           confidence=mc.confidence, misconception_source=mc.source)
        return {"item": stored_item.artifact_id, **({"rubric": stored_rubric.artifact_id} if stored_rubric else {}),
                "grade": stored_grade.artifact_id, "feedback": stored_feedback.artifact_id}

    def _artifact_ids(self, task_id: str, item: AssessmentItem, grade: AssessmentGrade) -> dict[str, str]:
        found = {"item": self._artifacts.find(task_id, item_artifact(item.assessment_item_id)),
                 "grade": self._artifacts.find(task_id, grade_artifact(grade.grade_id)),
                 "feedback": self._artifacts.find(task_id, feedback_artifact(grade.grade_id))}
        return {k: v.artifact_id for k, v in found.items() if v is not None}

    def _complete(self, attempt: AssessmentAttempt, grade: AssessmentGrade, outcome: AttemptOutcome,
                  scope: ExecutionScope, *, mastery_updated: bool) -> AssessmentAttempt:
        done = self._repo.complete(attempt.attempt_id, outcome, self._clock())
        self._emit(scope, EventType.ASSESSMENT_COMPLETED, done, "completed", attempt_id=done.attempt_id,
                   assessment_item_id=done.assessment_item_id, grade_id=done.grade_id, outcome=grade.outcome.value,
                   source=done.source, mastery_updated=mastery_updated,
                   evidence=len(done.outcome.evidence_ids) if done.outcome else 0,
                   next_action=(done.outcome.learning_action or {}).get("action") if done.outcome else None)
        return done

    @staticmethod
    def _emit(scope: ExecutionScope, type: str, attempt: AssessmentAttempt, key: str, **data: object) -> None:
        """Ids, outcome, scores and grader type only: never the answer, prompts, provider payloads or secrets."""
        scope.events.emit(type, task_id=attempt.task_id or scope.task_id, node_id=scope.node_id,
                          event_id=stable_id("aevt", attempt.attempt_id, key), **data)


def grading_context(m: LessonMaterial, concept_id: str, limit: int) -> GradingContext:
    sections = [s for s in m.lesson.sections if s.concept_id == concept_id][:limit]
    lesson = [ContextPassage(ref=f"lesson:{s.section_id}", kind="lesson", title=s.heading,
                             text=" ".join([s.explanation, *s.examples[:4]])) for s in sections]
    research: list[ContextPassage] = []
    if m.research is not None:
        cited = {c for s in sections for c in s.citations}
        for f in m.research.key_findings:
            ids = m.research.citation_ids_for(f)
            if ids and cited & set(ids) and len(research) < limit:
                research.append(ContextPassage(ref=ids[0], kind="research", title=f.statement[:80],
                                               text=" ".join(filter(None, [f.statement, f.example]))))
    return GradingContext(lesson_context=lesson, research_evidence=research)
