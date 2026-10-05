"""Interactive teaching sessions: the application service behind the /lessons/{id}/teaching-session and
/teaching-sessions endpoints. Opt-in: lessons and their evaluation work exactly as before without a session.

The service orchestrates; the existing systems stay authoritative:
- every state transition is computed by the deterministic engine (`app/teaching/`), never by a model;
- the teacher's turns are phrased by the TeachingSessionAgent through the runtime (validated structured output);
- mastery changes only through learner memory's MasteryUpdater, once, when the session completes (assessed sessions);
- objective progress and the next learning action come from the curriculum engine, as after an evaluation.

Persistence is the session store: each change is one atomic, version-checked write (optimistic lock), so concurrent
requests cannot corrupt a session and a repeated request cannot store anything twice. Events and artifacts are written
to an outbox in the same transaction and published after it, with stable ids: a restart publishes exactly what a
crash left unpublished, once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from app.artifacts.service import ArtifactService
from app.learner.memory import LearnerMemoryService
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger
from app.pedagogy.knowledge import KnowledgeBase
from app.runtime.interaction.teacher import TeacherTurnFailed, TeachingRuntime
from app.runtime.workflows.lesson_generation import WORKFLOW_ID as LESSON_WORKFLOW
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.common import new_id, utcnow
from app.schemas.events import EventType
from app.schemas.learner import LearningEvidence, stable_id
from app.schemas.lesson import LessonContent, LessonRequest
from app.schemas.pedagogy import LessonFocus, PedagogicalPlan
from app.schemas.research import ResearchBundle
from app.schemas.task import Task, TaskStatus
from app.schemas.teaching import (
    AnswerResult,
    EvidenceBrief,
    EvidenceType,
    GroundingItem,
    InteractionEvidence,
    LearnerInput,
    ObjectiveBrief,
    OutboxItem,
    PlannedTurn,
    PracticeItem,
    PublicQuestion,
    PublicState,
    QuestionBrief,
    SectionBrief,
    SessionChange,
    SessionConflict,
    SessionNextStep,
    SessionOutcome,
    SessionStarted,
    Speaker,
    StartTeachingSession,
    StateBrief,
    TeachingConfig,
    TeachingRequestRecord,
    TeachingSession,
    TeachingSessionStatus,
    TeachingSessionView,
    TeachingTurn,
    TeachingTurnInput,
    TurnBrief,
)
from app.services.curriculum import CurriculumService
from app.services.tasks import TaskService
from app.storage.repositories import NotFound
from app.teaching import engine
from app.teaching.errors import (
    InvalidSessionTransition,
    InvalidTeachingRequest,
    TeacherUnavailable,
    TeachingSessionNotFound,
)
from app.teaching.repository import TeachingRepository

__all__ = ["InvalidSessionTransition", "InvalidTeachingRequest", "SessionConflict", "TeacherUnavailable",
           "TeachingSessionNotFound", "TeachingSessionService"]

Status = TeachingSessionStatus
GOAL_KEY, CURRICULUM_KEY = "goal_id", "curriculum_id"
CONTROL_ATTEMPTS = 3  # pause / resume / cancel re-read the session after a concurrent change


@dataclass(frozen=True)
class LessonMaterial:
    task: Task
    artifact: Artifact
    lesson: LessonContent
    request: LessonRequest
    research: ResearchBundle | None
    focus: LessonFocus | None


class TeachingSessionService:
    def __init__(self, repository: TeachingRepository, runtime: TeachingRuntime, *, artifacts: ArtifactService,
                 memory: LearnerMemoryService, knowledge: KnowledgeBase, curriculum: CurriculumService,
                 tasks: TaskService, events: EventBus, config: TeachingConfig | None = None,
                 clock: Callable[[], datetime] = utcnow) -> None:
        self._repo = repository
        self._runtime = runtime
        self._artifacts = artifacts
        self._memory = memory
        self._knowledge = knowledge
        self._curriculum = curriculum
        self._tasks = tasks
        self._events = events
        self.config = config or TeachingConfig()
        self._clock = clock

    @property
    def repository(self) -> TeachingRepository:
        return self._repo

    # --- start ---------------------------------------------------------------------------------------------------

    async def start(self, lesson_id: str, data: StartTeachingSession) -> SessionStarted:
        """Create a session on a lesson and produce the teacher's opening turns. The same idempotency key returns the
        same session (finishing its opening if a failure interrupted it); without a key every call starts one."""
        m = self._lesson(lesson_id)
        objective_id, concept_id, description, action = self._objective(m, data)
        mode = "practice" if data.practice else "assessed"
        learner_id = m.task.learner_id
        key = data.idempotency_key or new_id("tskey")
        session_id = engine.session_id_for(learner_id, m.artifact.artifact_id, key)
        existing = await self._load(session_id, missing_ok=True)
        if existing is None:
            graph = await self._knowledge.graph(m.request.subject)
            section = next(s for s in m.lesson.sections if s.concept_id == concept_id)
            name = graph.concept(concept_id).name if concept_id in graph else section.heading
            metadata = {"idempotency_key": key, "domain": m.request.subject, "framework_id": m.request.framework_id,
                        "target_level": m.request.target_level, "language": m.request.language_of_instruction,
                        "config_fingerprint": self.config.fingerprint()}
            if m.focus is not None:
                metadata.update({GOAL_KEY: m.focus.goal_id, CURRICULUM_KEY: m.focus.curriculum_id})
            change = engine.start(session_id=session_id, learner_id=learner_id, task_id=m.task.task_id,
                                  lesson_id=m.artifact.artifact_id, objective_id=objective_id,
                                  objective_description=description, concept_id=concept_id, concept_name=name,
                                  action=action, mode=mode, config=self.config, at=self._clock(), metadata=metadata)
            created = self._repo.create(change)
            session = await self._load(session_id)
        else:
            created = False
            session = existing
        if not created and (session.objective_id, session.action, session.mode) != (objective_id, action, mode):
            raise SessionConflict(f"idempotency key {key!r} already started a different session on this lesson")
        session = await self._drive(session)
        opening = []
        for turn in self._repo.turns(session_id):
            if turn.speaker == Speaker.LEARNER:
                break
            opening.append(turn)
        return SessionStarted(session_id=session_id, status=session.status, created=created,
                              first_turn=opening[0] if opening else None, teacher_turns=opening,
                              objective=ObjectiveBrief(concept_id=session.state.concept_id,
                                                       concept_name=session.state.concept_name,
                                                       description=session.state.objective_description),
                              objective_id=session.objective_id, difficulty=session.state.difficulty,
                              waiting_question=self._public_question(session))

    def _lesson(self, lesson_id: str) -> LessonMaterial:
        """A LESSON artifact id, or the id of the completed lesson task that produced it."""
        try:
            artifact = self._artifacts.get(lesson_id)
        except (NotFound, KeyError):
            artifact = None
        if artifact is not None:
            if artifact.type != ArtifactType.LESSON:
                raise NotFound(f"artifact {lesson_id} is not a lesson")
            task = self._tasks.get(artifact.task_id)
        else:
            try:
                task = self._tasks.get(lesson_id)
            except NotFound:
                raise NotFound(f"lesson {lesson_id} not found") from None
            if task.plan is None or task.plan.workflow_id != LESSON_WORKFLOW:
                raise NotFound(f"task {lesson_id} is not a lesson")
            if task.status != TaskStatus.COMPLETED:
                raise InvalidSessionTransition(f"lesson task {lesson_id} is {task.status.value}, not completed")
            artifact = self._artifacts.find(task.task_id, "lesson")
            if artifact is None:
                raise NotFound(f"lesson task {lesson_id} has no lesson")
        assert task.plan is not None
        research_artifact = self._artifacts.find(task.task_id, "research_bundle")
        plan_artifact = self._artifacts.find(task.task_id, "pedagogical_plan")
        plan = (PedagogicalPlan.model_validate_json(self._artifacts.read(plan_artifact.artifact_id))
                if plan_artifact is not None else None)
        return LessonMaterial(
            task=task, artifact=artifact,
            lesson=LessonContent.model_validate_json(self._artifacts.read(artifact.artifact_id)),
            request=task.plan.lesson_request,
            research=(ResearchBundle.model_validate_json(self._artifacts.read(research_artifact.artifact_id))
                      if research_artifact is not None else None),
            focus=plan.focus if plan is not None else None)

    @staticmethod
    def _objective(m: LessonMaterial, data: StartTeachingSession) -> tuple[str | None, str, str, str]:
        """(objective id, concept, description, action): as requested, else the lesson's curriculum focus, else its
        first objective. The concept must be one the lesson teaches."""
        focus, lesson = m.focus, m.lesson
        if data.objective_id is not None:
            if focus is not None and data.objective_id == focus.objective_id:
                found = (focus.objective_id, focus.concept_id, focus.description)
            else:
                o = next((o for o in lesson.objectives if o.objective_id == data.objective_id), None)
                if o is None:
                    raise InvalidTeachingRequest(f"objective {data.objective_id} is not an objective of this lesson")
                found = (o.objective_id, o.concept_id, o.description)
        elif focus is not None:
            found = (focus.objective_id, focus.concept_id, focus.description)
        elif lesson.objectives:
            o = lesson.objectives[0]
            found = (o.objective_id, o.concept_id, o.description)
        else:
            s = lesson.sections[0]
            found = (None, s.concept_id, f"Learn {s.heading}")
        if not any(s.concept_id == found[1] for s in lesson.sections):
            raise InvalidTeachingRequest(f"the lesson has no section on concept {found[1]}")
        action = data.action or (focus.action if focus is not None else "LEARN")
        return found[0], found[1], found[2], action

    # --- learner input -------------------------------------------------------------------------------------------

    async def submit(self, session_id: str, data: LearnerInput) -> AnswerResult:
        """An answer, a learner question or a request to stop. Idempotent per client_turn_id: the same id with the
        same input returns the stored result; with a different input it is a conflict."""
        session = await self._load(session_id)
        digest = hashlib.sha256(f"{data.kind}\x1f{data.answer}".encode()).hexdigest()
        if data.client_turn_id:
            record = self._repo.request(session_id, data.client_turn_id)
            if record is not None:
                return await self._replay(record, digest)
        now = self._clock()
        if data.kind == "answer":
            change = engine.record_answer(session, data.answer, now, client_turn_id=data.client_turn_id).change
        elif data.kind == "question":
            change = engine.record_question(session, data.answer, now, client_turn_id=data.client_turn_id)
        else:
            change = engine.record_stop(session, data.answer, now, client_turn_id=data.client_turn_id)
        learner_turn = change.turns[0]
        record = TeachingRequestRecord(session_id=session_id,
                                       client_turn_id=data.client_turn_id or learner_turn.turn_id,
                                       request_hash=digest, turn_ids=[learner_turn.turn_id], created_at=now)
        try:
            self._repo.apply(change, expected_version=session.version, request=record)
        except SessionConflict:
            if data.client_turn_id and (stored := self._repo.request(session_id, data.client_turn_id)) is not None:
                return await self._replay(stored, digest)  # the same answer, sent twice at once: applied once
            raise
        await self._publish(change.session)
        driven = await self._drive(change.session)
        return self._result(driven, learner_turn.turn_id, replayed=False)

    async def _replay(self, record: TeachingRequestRecord, digest: str) -> AnswerResult:
        if record.request_hash != digest:
            raise SessionConflict(f"client_turn_id {record.client_turn_id} was already used for a different input")
        session = await self._drive(await self._load(record.session_id))  # finish a reply a failure interrupted
        return self._result(session, record.turn_ids[0], replayed=True)

    def _result(self, session: TeachingSession, learner_turn_id: str, *, replayed: bool) -> AnswerResult:
        turns = self._repo.turns(session.session_id)
        learner = next(t for t in turns if t.turn_id == learner_turn_id)
        later = [t for t in turns if t.sequence > learner.sequence]
        reply = []
        for t in later:
            if t.speaker == Speaker.LEARNER:
                break
            reply.append(t)
        ids = {learner.turn_id, *(t.turn_id for t in reply)}
        evidence = [e for e in self._repo.evidence(session.session_id) if e.turn_id in ids]
        change = next((c for c in session.state.difficulty_history if c.turn_id == learner.turn_id), None)
        return AnswerResult(session_id=session.session_id, status=session.status, replayed=replayed,
                            learner_turn=learner, teacher_turns=reply, evidence=evidence,
                            correct=learner.metadata.get("correct"), difficulty=session.state.difficulty,
                            difficulty_change=change, completion_reason=session.state.completion_reason,
                            waiting_question=self._public_question(session), summary=session.summary,
                            outcome=session.outcome)

    # --- lifecycle -----------------------------------------------------------------------------------------------

    async def pause(self, session_id: str) -> TeachingSessionView:
        """Answers are refused while paused. Idempotent."""
        await self._control(session_id, engine.pause)
        return await self.view(session_id)

    async def resume(self, session_id: str) -> TeachingSessionView:
        """Continue from the stored state: publish what a crash left unpublished and produce any teacher turns still
        owed. Idempotent (resuming a session that is not paused only does that recovery)."""
        session = await self._control(session_id, engine.resume)
        if session.status == Status.ACTIVE:
            await self._drive(session)
        return await self.view(session_id)

    async def cancel(self, session_id: str) -> TeachingSessionView:
        """Stop the session for good; its turns, evidence and artifacts are kept. Idempotent."""
        await self._control(session_id, engine.cancel)
        return await self.view(session_id)

    async def _control(self, session_id: str, transition: Callable[[TeachingSession, datetime],
                                                               SessionChange | None]) -> TeachingSession:
        for attempt in range(CONTROL_ATTEMPTS):
            session = await self._load(session_id)
            change = transition(session, self._clock())
            if change is None:
                return session
            try:
                self._repo.apply(change, expected_version=session.version)
            except SessionConflict:
                if attempt == CONTROL_ATTEMPTS - 1:
                    raise
                continue
            await self._publish(change.session)
            return change.session
        raise AssertionError("unreachable")

    # --- reads ---------------------------------------------------------------------------------------------------

    async def get(self, session_id: str) -> TeachingSession:
        return await self._load(session_id)

    def sessions(self, learner_id: str) -> list[TeachingSession]:
        return self._repo.for_learner(learner_id)

    def turns(self, session_id: str) -> list[TeachingTurn]:
        return self._repo.turns(session_id)

    def evidence(self, session_id: str) -> list[InteractionEvidence]:
        return self._repo.evidence(session_id)

    async def view(self, session_id: str) -> TeachingSessionView:
        session = await self._load(session_id)
        st = session.state
        public = PublicState(
            objective_id=st.objective_id, objective_description=st.objective_description, concept_id=st.concept_id,
            concept_name=st.concept_name, difficulty=st.difficulty, strategy=st.strategy,
            questions_asked=st.questions_asked, questions_answered=st.questions_answered,
            correct_answers=st.correct_answers, incorrect_answers=st.incorrect_answers,
            consecutive_successes=st.consecutive_successes, consecutive_failures=st.consecutive_failures,
            hints_used=st.hints_used, misconceptions=list(dict.fromkeys(m.misconception for m in st.misconceptions)),
            difficulty_trajectory=st.difficulty_trajectory, difficulty_changes=st.difficulty_history,
            turn_budget=st.turn_budget, completion_reason=st.completion_reason,
            waiting_question=self._public_question(session))
        return TeachingSessionView(
            session_id=session.session_id, learner_id=session.learner_id, task_id=session.task_id,
            lesson_id=session.lesson_id, objective_id=session.objective_id, action=session.action, mode=session.mode,
            status=session.status, turn_count=session.turn_count, started_at=session.started_at,
            updated_at=session.updated_at, completed_at=session.completed_at, state=public,
            turns=self._repo.turns(session_id), objective_progress=self._objective_progress(session),
            next_action=self._next_step(session), summary=session.summary, outcome=session.outcome)

    def _objective_progress(self, session: TeachingSession) -> dict | None:
        if session.outcome is not None and session.outcome.objective_progress is not None:
            return session.outcome.objective_progress
        goal_id = session.metadata.get(GOAL_KEY)
        if goal_id and session.objective_id:
            try:
                curriculum = self._curriculum.curriculum(goal_id)
            except KeyError:
                curriculum = None
            if curriculum is not None and any(o.objective_id == session.objective_id for o in curriculum.objectives):
                return curriculum.progress.of(session.objective_id).model_dump(mode="json")
        concept = self._memory.get_or_create(session.learner_id).concepts.get(session.state.concept_id)
        return {"concept_id": session.state.concept_id, "current_mastery": concept.mastery if concept else None,
                "evidence_count": concept.evidence_count if concept else 0}

    @staticmethod
    def _next_step(session: TeachingSession) -> SessionNextStep:
        if session.status == Status.WAITING_FOR_LEARNER:
            q = session.state.pending_question
            return SessionNextStep(kind="answer_question", prompt=q.prompt if q else None)
        if session.status == Status.PAUSED:
            return SessionNextStep(kind="resume")
        if session.status == Status.ACTIVE:
            return SessionNextStep(kind="teacher_reply_pending")
        if session.status == Status.COMPLETED and session.outcome is not None:
            return SessionNextStep(kind="learning_action", learning_action=session.outcome.learning_action)
        return SessionNextStep(kind="none")

    @staticmethod
    def _public_question(session: TeachingSession) -> PublicQuestion | None:
        q = session.state.pending_question
        if q is None or session.status in (Status.COMPLETED, Status.CANCELLED, Status.FAILED):
            return None
        return PublicQuestion(turn_id=q.turn_id, kind=q.kind, prompt=q.prompt, choices=q.choices,
                              difficulty=q.difficulty, hint_level=q.hint_level)

    async def _load(self, session_id: str, *, missing_ok: bool = False) -> TeachingSession:
        session = self._repo.get(session_id)
        if session is None:
            if missing_ok:
                return None  # type: ignore[return-value]
            raise TeachingSessionNotFound(f"teaching session {session_id} not found")
        if self._repo.pending_outbox(session_id):
            await self._publish(session)  # what a crash left unpublished
            session = self._repo.get(session_id) or session
        return session

    # --- teacher turns -------------------------------------------------------------------------------------------

    async def _drive(self, session: TeachingSession) -> TeachingSession:
        """Produce every teacher turn the policy owes (status ACTIVE) and store them in one change. A failure stores
        nothing: the session stays ACTIVE with the learner's input recorded, and the next call tries again."""
        if session.status != Status.ACTIVE:
            return session
        m = self._lesson(session.lesson_id)
        now = self._clock()
        work = session.model_copy(deep=True)
        work.version += 1
        work.updated_at = now
        history = self._repo.turns(session.session_id)
        evidence_so_far = self._repo.evidence(session.session_id)
        change = SessionChange(session=work)
        narrative = ""
        while work.state.planned:
            item = work.state.planned[0]
            data = self._context(work, item, m, [*history, *change.turns], [*evidence_so_far, *change.evidence])
            try:
                output, _usage = await self._runtime.teacher_turn(data, task_id=session.task_id)
                part = engine.apply_teacher(work, item, output, now)
            except (TeacherTurnFailed, ValueError) as exc:
                raise TeacherUnavailable(f"the teacher's {item.action.value} turn could not be produced; what the "
                                         f"learner sent is stored and resuming the session retries ({exc})") from exc
            change.turns += part.turns
            change.evidence += part.evidence
            change.outbox += part.outbox
            if item.action.value == "SUMMARIZE":
                narrative = output.response
        change.outbox += engine.settle(work, now, narrative)
        change.session = work
        try:
            self._repo.apply(change, expected_version=session.version)
        except SessionConflict:
            current = await self._load(session.session_id)
            if current.status != Status.ACTIVE:
                return current  # another request produced these turns (or paused / cancelled the session) first
            raise
        await self._publish(work)
        return self._repo.get(session.session_id) or work

    def _context(self, session: TeachingSession, item: PlannedTurn, m: LessonMaterial, turns: list[TeachingTurn],
                 evidence: list[InteractionEvidence]) -> TeachingTurnInput:
        """The structured context window the model sees: the objective, the lesson material on its concept, the
        session state, the last few turns and evidence. No learner, session or goal ids, no other sessions."""
        st, cfg = session.state, session.config
        concept = st.concept_id
        sections = [SectionBrief(ref=f"lesson:{s.section_id}", concept_id=s.concept_id, heading=s.heading,
                                 explanation=s.explanation, examples=s.examples[:4], analogy=s.analogy,
                                 citations=s.citations)
                    for s in m.lesson.sections if s.concept_id == concept]
        practice = [PracticeItem(prompt=e.prompt, answer=_answer(e.answer)) for e in m.lesson.exercises
                    if e.concept_id == concept and _answer(e.answer)]
        practice += [PracticeItem(prompt=q.prompt, answer=_answer(q.answer)) for q in m.lesson.check_questions
                     if q.concept_id == concept and _answer(q.answer)]
        question = st.pending_question or (st.last_question if item.responds_to else None)
        answer = st.last_answer if item.responds_to and st.last_answer and \
            st.last_answer.turn_id == item.responds_to else None
        learner_question = None
        sources: list[GroundingItem] = []
        if item.purpose == "answer_question":
            learner_question = next(t.content for t in turns if t.turn_id == item.responds_to)
            sources = self._sources(m)
        profile = self._memory.get_or_create(session.learner_id)
        mastery = profile.concepts[concept].mastery if concept in profile.concepts else 0.0
        return TeachingTurnInput(
            stage="answer" if item.purpose == "answer_question" else "turn",
            action=item.action, subject=m.request.subject, topic=m.request.topic, level=m.lesson.level,
            language=m.request.language_of_instruction,
            objective=ObjectiveBrief(concept_id=concept, concept_name=st.concept_name,
                                     description=st.objective_description),
            lesson_title=m.lesson.title, sections=sections, practice=practice,
            state=StateBrief(difficulty=st.difficulty, min_difficulty=cfg.min_difficulty,
                             max_difficulty=cfg.max_difficulty, strategy=st.strategy,
                             questions_asked=st.questions_asked, correct_answers=st.correct_answers,
                             incorrect_answers=st.incorrect_answers, consecutive_successes=st.consecutive_successes,
                             consecutive_failures=st.consecutive_failures, hints_used=st.hints_used,
                             mastery=round(mastery, 2)),
            recent_turns=[TurnBrief(speaker=t.speaker, turn_type=t.turn_type, content=t.content)
                          for t in turns[-cfg.recent_turns:]],
            evidence=[EvidenceBrief(evidence_type=e.evidence_type, correct=e.correct, hint_level=e.hint_level,
                                    difficulty=e.difficulty)
                      for e in evidence if e.evidence_type in (EvidenceType.ANSWER, EvidenceType.HINT)][-6:],
            question=QuestionBrief(kind=question.kind, prompt=question.prompt, choices=question.choices,
                                   expected_answer=question.expected_answer,
                                   accepted_answers=question.accepted_answers, difficulty=question.difficulty)
            if question is not None else None,
            learner_answer=answer.answer if answer else None, answer_correct=answer.correct if answer else None,
            correction=item.correction,
            hint_level=item.hint_level if item.action.value == "HINT" else (answer.hint_level if answer else 0),
            reveal_answer=cfg.reveal_answer_in_hints, difficulty=st.difficulty,
            learner_question=learner_question, sources=sources)

    @staticmethod
    def _sources(m: LessonMaterial) -> list[GroundingItem]:
        """What a learner question may be answered from: the lesson's sections and its research findings (the
        knowledge base is added by the grounding tool). Citation ids are the research bundle's, never invented."""
        items = [GroundingItem(ref=f"lesson:{s.section_id}", kind="lesson", title=s.heading,
                               text=" ".join([s.explanation, *s.examples]), citations=s.citations)
                 for s in m.lesson.sections]
        if m.research is not None:
            for f in m.research.key_findings:
                ids = m.research.citation_ids_for(f)
                items.append(GroundingItem(ref=ids[0], kind="research", title=f.statement[:80],
                                           text=" ".join(filter(None, [f.statement, f.example])), citations=ids))
        return items

    # --- publishing (outbox) -------------------------------------------------------------------------------------

    def _scope(self, session: TeachingSession) -> ExecutionScope:
        return ExecutionScope(events=self._events, usage=UsageLedger(), task_id=session.task_id,
                              node_id="teaching_session")

    async def _publish(self, session: TeachingSession) -> None:
        for item in self._repo.pending_outbox(session.session_id):
            await self._publish_item(session, item)
            self._repo.mark_published(item.item_id)

    async def _publish_item(self, session: TeachingSession, item: OutboxItem) -> None:
        scope = self._scope(session)
        if item.kind == "event":
            self._events.emit(item.payload["type"], task_id=session.task_id, node_id="teaching_session",
                              event_id=item.item_id, **item.payload["data"])
        elif item.kind == "artifact":
            self._store_artifact(session, item.payload, scope)
        else:
            await self._finalize(session.session_id, scope)

    def _store_artifact(self, session: TeachingSession, p: dict, scope: ExecutionScope) -> Artifact:
        parents = list(p["parent_ids"])
        for name in p["parent_names"]:
            parent = self._artifacts.find(session.task_id, name)
            if parent is None:
                raise RuntimeError(f"artifact {name} must be stored before its children")
            parents.append(parent.artifact_id)
        return self._artifacts.store(task_id=session.task_id, name=p["name"], type=ArtifactType(p["type"]),
                                     media_type="application/json", content=p["content"].encode("utf-8"),
                                     provider="teaching-agent", parent_ids=parents, metadata=p["metadata"],
                                     scope=scope)

    # --- completion ----------------------------------------------------------------------------------------------

    async def _finalize(self, session_id: str, scope: ExecutionScope) -> None:
        """The completed session reaches the existing systems, once: its graded answers become LearningEvidence and
        mastery is updated by learner memory's deterministic updater (not for practice sessions); the curriculum
        engine then records objective progress and selects the next learning action."""
        session = self._repo.get(session_id)
        assert session is not None and session.status == Status.COMPLETED and session.summary is not None
        if session.outcome is None:
            outcome = await self._integrate(session, scope)
            done = session.model_copy(deep=True)
            done.version += 1
            done.outcome = outcome
            try:
                self._repo.apply(SessionChange(session=done), expected_version=session.version)
            except SessionConflict:
                stored = self._repo.get(session_id)
                if stored is None or stored.outcome is None:
                    raise
                done = stored
            session = done
        outcome = session.outcome
        assert outcome is not None
        action = outcome.learning_action or {}
        self._events.emit(EventType.TEACHING_SESSION_COMPLETED, task_id=session.task_id, node_id="teaching_session",
                          event_id=stable_id("tevt", session_id, "completed"), session_id=session_id,
                          completion_reason=session.summary.completion_reason.value, practice=outcome.practice,
                          evidence=len(session.state.evidence_ids),
                          learning_evidence=len(outcome.learning_evidence_ids),
                          mastery_updated=bool(outcome.learning_evidence_ids),
                          objective_id=session.objective_id, next_action=action.get("action"),
                          summary_artifact_id=outcome.artifact_ids.get("summary"))

    async def _integrate(self, session: TeachingSession, scope: ExecutionScope) -> SessionOutcome:
        domain = session.metadata["domain"]
        graph = await self._knowledge.graph(domain)
        practice = session.mode == "practice"
        summary = self._artifacts.find(session.task_id, engine.summary_artifact(session.session_id))
        artifact_ids = {"session": self._artifact_id(session, engine.session_artifact(session.session_id)),
                        "summary": summary.artifact_id if summary else None}
        answers = [e for e in self._repo.evidence(session.session_id)
                   if e.evidence_type == EvidenceType.ANSWER and e.correct is not None]
        learning = [] if practice else [self._learning_evidence(session, e) for e in answers]
        changes = []
        if learning:
            update = self._memory.record_evidence(session.learner_id, domain, learning,
                                                  [c.ref() for c in graph.concepts()], scope)
            changes = [c.model_dump(mode="json") for c in update.changes]
        action = await self._curriculum.next_action(session.learner_id)
        progress = None
        goal_id = session.metadata.get(GOAL_KEY)
        if goal_id and session.objective_id:
            curriculum = self._curriculum.curriculum(goal_id)
            if curriculum is not None and any(o.objective_id == session.objective_id for o in curriculum.objectives):
                progress = curriculum.progress.of(session.objective_id).model_dump(mode="json")
        model = self._memory.learner_model(session.learner_id, domain, session.metadata["framework_id"],
                                           session.metadata.get("target_level"), graph.ids)
        parent = artifact_ids["summary"]
        drafts: list[tuple[str, str, ArtifactType, str, dict]] = []
        if learning:
            drafts.append(("learning_evidence", f"learning_evidence.{session.session_id}",
                           ArtifactType.LEARNING_EVIDENCE,
                           "[\n" + ",\n".join(e.model_dump_json(indent=2) for e in learning) + "\n]",
                           {"source": "teaching_session", "items": len(learning),
                            "concepts": sorted({e.concept_id for e in learning})}))
        drafts.append(("learner_model", f"learner_model.{session.session_id}", ArtifactType.LEARNER_MODEL,
                       model.model_dump_json(indent=2), {"domain": model.domain, "mastered": model.mastered_concepts,
                                                         "weak": model.weak_concepts}))
        drafts.append(("learning_action", f"learning_action.{session.session_id}", ArtifactType.LEARNING_ACTION,
                       action.model_dump_json(indent=2), {"action": action.action.value,
                                                          "concept_id": action.concept_id}))
        for key, name, type, content, metadata in drafts:
            stored = self._artifacts.store(task_id=session.task_id, name=name, type=type,
                                           media_type="application/json", content=content.encode("utf-8"),
                                           provider="teaching-agent", parent_ids=[parent] if parent else [],
                                           metadata={"session_id": session.session_id, **metadata}, scope=scope)
            artifact_ids[key] = parent = stored.artifact_id
        return SessionOutcome(practice=practice, learning_evidence_ids=[e.evidence_id for e in learning],
                              mastery_changes=changes, objective_progress=progress,
                              learning_action=action.model_dump(mode="json"),
                              artifact_ids={k: v for k, v in artifact_ids.items() if v})

    def _artifact_id(self, session: TeachingSession, name: str) -> str | None:
        found = self._artifacts.find(session.task_id, name)
        return found.artifact_id if found else None

    @staticmethod
    def _learning_evidence(session: TeachingSession, e: InteractionEvidence) -> LearningEvidence:
        """A graded session answer as LearningEvidence: correct unaided 1.0, correct after a hint partial credit,
        incorrect 0.0; the session difficulty weights it like any other evidence."""
        if e.correct and e.hint_level == 0:
            correctness, score = "correct", 1.0
        elif e.correct:
            correctness = "partial"
            score = round(min(0.95, max(0.05, 1 - session.config.hint_penalty * e.hint_level)), 4)
        else:
            correctness, score = "incorrect", 0.0
        ref = f"{session.session_id}/{e.turn_id}"
        return LearningEvidence(
            evidence_id=LearningEvidence.id_for(session.learner_id, "interaction", ref, e.concept_id),
            learner_id=session.learner_id, concept_id=e.concept_id, source_type="interaction", source_ref=ref,
            correctness=correctness, score=score, difficulty=session.config.evidence_difficulty(e.difficulty),
            timestamp=e.created_at,
            metadata={"session_id": session.session_id, "turn_id": e.turn_id, "interaction_evidence_id": e.evidence_id,
                      "hint_level": e.hint_level})


def _answer(text: str) -> str:
    """A lesson exercise's answer as an expected answer: without the sentence's closing punctuation."""
    return text.strip().rstrip(".!")
