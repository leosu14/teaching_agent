"""Interactive teaching sessions: a persistent, resumable multi-turn conversation built on a lesson.

A session is orchestration on top of the existing learning system, never a second copy of it: mastery stays with the
deterministic MasteryUpdater (through learner memory), objective progress and the next action with the curriculum
engine. The session keeps only what the conversation needs: its turns, a deterministic state and the interaction
evidence the learner produced. Every state change is computed by code (`app/teaching/`); a model only phrases the
teacher's turns and proposes candidate misconceptions, which are validated before anything is stored.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import Schema

# --- enums ---------------------------------------------------------------------------------------------------------


class TeachingSessionStatus(str, Enum):
    ACTIVE = "ACTIVE"  # the teacher owes the next turn(s): stored after a learner turn, before the teacher's reply
    WAITING_FOR_LEARNER = "WAITING_FOR_LEARNER"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


TERMINAL_STATUSES = frozenset({TeachingSessionStatus.COMPLETED, TeachingSessionStatus.CANCELLED,
                               TeachingSessionStatus.FAILED})


class Speaker(str, Enum):
    TEACHER = "TEACHER"
    LEARNER = "LEARNER"
    SYSTEM = "SYSTEM"


class TurnType(str, Enum):
    EXPLANATION = "EXPLANATION"
    QUESTION = "QUESTION"
    HINT = "HINT"
    FEEDBACK = "FEEDBACK"
    CORRECTION = "CORRECTION"
    ENCOURAGEMENT = "ENCOURAGEMENT"
    LEARNER_ANSWER = "LEARNER_ANSWER"
    LEARNER_QUESTION = "LEARNER_QUESTION"
    SUMMARY = "SUMMARY"
    ASSESSMENT = "ASSESSMENT"


class TeachingAction(str, Enum):
    EXPLAIN = "EXPLAIN"
    ASK = "ASK"
    HINT = "HINT"
    FEEDBACK = "FEEDBACK"
    RETEACH = "RETEACH"
    PRACTICE = "PRACTICE"
    CHECK = "CHECK"
    SUMMARIZE = "SUMMARIZE"
    COMPLETE = "COMPLETE"


QUESTION_ACTIONS = frozenset({TeachingAction.ASK, TeachingAction.PRACTICE, TeachingAction.CHECK})


class CompletionReason(str, Enum):
    OBJECTIVE_DEMONSTRATED = "OBJECTIVE_DEMONSTRATED"
    TURN_BUDGET_REACHED = "TURN_BUDGET_REACHED"
    REPEATED_FAILURE = "REPEATED_FAILURE"  # the learner needs a new lesson, not more questions
    LEARNER_STOPPED = "LEARNER_STOPPED"
    QUESTION_LIMIT_REACHED = "QUESTION_LIMIT_REACHED"  # the teacher's configured question budget is spent


class EvidenceType(str, Enum):
    ANSWER = "ANSWER"  # a graded answer (correctness, hint usage, difficulty)
    HINT = "HINT"  # a hint was given
    MISCONCEPTION = "MISCONCEPTION"  # a validated candidate misconception (never a mastery change by itself)
    LEARNER_QUESTION = "LEARNER_QUESTION"  # the learner asked a question


SessionMode = Literal["assessed", "practice"]
QuestionKind = Literal["short_answer", "multiple_choice"]
LearnerInputKind = Literal["answer", "question", "stop"]


# --- configuration -------------------------------------------------------------------------------------------------


class TeachingConfig(Schema):
    """Every threshold of the deterministic policy. A session keeps the configuration it started with, so a resumed
    session (even after a configuration change) continues exactly as it would have."""

    min_difficulty: int = Field(default=1, ge=1)
    max_difficulty: int = Field(default=3, ge=1)
    start_difficulty: int = Field(default=2, ge=1)
    increase_after_successes: int = Field(default=3, ge=1)  # consecutive correct answers before a harder question
    decrease_after_failures: int = Field(default=2, ge=1)  # consecutive incorrect answers before an easier one
    hints_enabled: bool = True
    max_hint_level: int = Field(default=3, ge=0, le=3)
    reveal_answer_in_hints: bool = False  # a hint never contains the expected answer unless this is set
    hint_penalty: float = Field(default=0.15, ge=0, lt=0.5)  # score lost per hint level on a hinted correct answer
    misconception_min_confidence: float = Field(default=0.5, ge=0, le=1)  # weaker candidates are not recorded
    # An incorrect answer when at least this many misconceptions were already recorded on the concept in this
    # session (the misconception repeats): reteach instead of hinting.
    reteach_after_misconceptions: int = Field(default=1, ge=1)
    demonstration_correct: int = Field(default=3, ge=1)  # correct answers that demonstrate the objective...
    demonstration_min_difficulty: int = Field(default=2, ge=1)  # ...the last one unaided and at least this hard
    max_incorrect: int = Field(default=5, ge=1)  # incorrect answers before the session ends with REPEATED_FAILURE
    max_questions: int = Field(default=10, ge=1)  # the teacher's question budget
    max_turns: int = Field(default=40, ge=4)  # turn budget of the whole session
    recent_turns: int = Field(default=6, ge=1, le=20)  # turns the model sees (a context window, not the history)

    @model_validator(mode="after")
    def _ordered(self) -> TeachingConfig:
        if not self.min_difficulty <= self.start_difficulty <= self.max_difficulty:
            raise ValueError("start_difficulty must lie between min_difficulty and max_difficulty")
        if not self.min_difficulty <= self.demonstration_min_difficulty <= self.max_difficulty:
            raise ValueError("demonstration_min_difficulty must lie between min_difficulty and max_difficulty")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:12]

    def evidence_difficulty(self, level: int) -> float:
        """A session difficulty level as the 0-1 difficulty of LearningEvidence (0.2 easiest, 0.8 hardest)."""
        span = self.max_difficulty - self.min_difficulty
        return round(0.5 if span == 0 else 0.2 + 0.6 * (level - self.min_difficulty) / span, 4)


# --- persistent records ----------------------------------------------------------------------------------------------


class PendingQuestion(Schema):
    """The question the learner is answering. Internal: it carries the answer key and never leaves the service."""

    turn_id: str
    action: TeachingAction
    concept_id: str
    difficulty: int
    kind: QuestionKind
    prompt: str
    choices: list[str] = Field(default_factory=list)
    expected_answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    hint_level: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)


class PlannedTurn(Schema):
    """A teacher turn the policy decided on and the teacher still owes. `responds_to` names the learner turn it
    answers (the first turn after an answer or a learner question)."""

    action: TeachingAction
    purpose: Literal["teach", "answer_question"] = "teach"
    responds_to: str | None = None
    correction: bool = False  # FEEDBACK that reveals the expected answer (the question is closed)
    hint_level: int = Field(default=0, ge=0, le=3)  # HINT: the level this hint is written at


class DifficultyChange(Schema):
    turn_id: str  # the learner answer that caused it
    before: int
    after: int
    reason: str


class MisconceptionRecord(Schema):
    evidence_id: str
    concept_id: str
    misconception: str
    confidence: float = Field(ge=0, le=1)


class AnswerOutcome(Schema):
    """The last graded answer: what the next teacher turn responds to."""

    turn_id: str
    answer: str
    correct: bool
    hint_level: int = 0


class SessionState(Schema):
    """The deterministic session state. Serializable, and the only thing a transition reads: identical state and
    learner input always give an identical transition."""

    objective_id: str | None = None
    objective_description: str
    concept_id: str
    concept_name: str
    difficulty: int
    strategy: str
    questions_asked: int = 0
    questions_answered: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    assisted_correct: int = 0  # correct after a hint
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    hints_used: int = 0
    learner_questions: int = 0
    misconceptions: list[MisconceptionRecord] = Field(default_factory=list)
    difficulty_history: list[DifficultyChange] = Field(default_factory=list)
    difficulty_trajectory: list[int] = Field(default_factory=list)  # difficulty of every question asked, in order
    evidence_ids: list[str] = Field(default_factory=list)
    pending_question: PendingQuestion | None = None
    planned: list[PlannedTurn] = Field(default_factory=list)
    last_answer: AnswerOutcome | None = None
    last_question: PendingQuestion | None = None  # the question the last answer was to (feedback refers to it)
    turn_budget: int
    completion_reason: CompletionReason | None = None


class SessionOutcome(Schema):
    """What the completed session changed in the existing systems (filled once, by the finalize step)."""

    practice: bool
    learning_evidence_ids: list[str] = Field(default_factory=list)
    mastery_changes: list[dict] = Field(default_factory=list)  # MasteryChange dumps, from learner memory
    objective_progress: dict | None = None  # ObjectiveProgress dump after the update (curriculum sessions)
    learning_action: dict | None = None  # NextLearningAction dump: the next action after the session
    artifact_ids: dict[str, str] = Field(default_factory=dict)


class TeachingSessionSummary(Schema):
    """Deterministic fields come from the session state; `narrative` is the teacher's wording only."""

    session_id: str
    lesson_id: str
    objective_id: str | None
    concepts_practiced: list[str]
    strengths: list[str]
    difficulties: list[str]
    misconceptions: list[str]
    hints_used: int
    questions_asked: int
    correct_answers: int
    incorrect_answers: int
    difficulty_trajectory: list[int]
    evidence_ids: list[str]
    recommended_next_action: Literal["advance", "practice", "reteach", "new_lesson"]
    completion_reason: CompletionReason
    practice: bool
    narrative: str = ""


class TeachingSession(Schema):
    session_id: str
    learner_id: str
    task_id: str  # the lesson task: the session's artifacts and events belong to it
    lesson_id: str  # the LESSON artifact
    objective_id: str | None = None
    action: str  # LEARN / REVIEW / PRACTICE / EVALUATE
    mode: SessionMode = "assessed"
    status: TeachingSessionStatus
    turn_count: int = Field(default=0, ge=0)
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    paused_from: TeachingSessionStatus | None = None
    version: int = Field(default=1, ge=1)  # optimistic lock: every stored change increments it
    config: TeachingConfig
    state: SessionState
    summary: TeachingSessionSummary | None = None
    outcome: SessionOutcome | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class TeachingTurn(Schema):
    turn_id: str
    session_id: str
    sequence: int = Field(ge=1)
    speaker: Speaker
    turn_type: TurnType
    content: str
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


class InteractionEvidence(Schema):
    """One structured observation from the session. Session-local: only ANSWER evidence of an assessed session ever
    reaches the mastery updater, and only through learner memory when the session completes."""

    evidence_id: str
    session_id: str
    turn_id: str  # where it was observed: the learner turn (answer, question, misconception) or the hint turn
    learner_id: str
    concept_id: str
    evidence_type: EvidenceType
    correct: bool | None = None
    hint_level: int = Field(default=0, ge=0)
    hints_used: int = Field(default=0, ge=0)
    answer_after_hint: bool = False
    difficulty: int
    practice: bool = False  # practice evidence never updates mastery
    misconception: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime
    metadata: dict = Field(default_factory=dict)


class OutboxItem(Schema):
    """Work to publish after a state change was stored (an event, an artifact, the completion step). Stored in the
    same transaction as the change and published once: a restart publishes what a crash left behind."""

    item_id: str
    session_id: str
    kind: Literal["event", "artifact", "finalize"]
    payload: dict = Field(default_factory=dict)
    published: bool = False


class SessionConflict(Exception):
    """Another request changed the session first (optimistic lock) or reused a client_turn_id for a different
    request (409). Nothing of this request was stored: it can be sent again."""


class SessionChange(Schema):
    """Everything one transition stores, atomically: the new session (its version incremented), new turns, new
    evidence and the outbox items to publish once it is stored."""

    session: TeachingSession
    turns: list[TeachingTurn] = Field(default_factory=list)
    evidence: list[InteractionEvidence] = Field(default_factory=list)
    outbox: list[OutboxItem] = Field(default_factory=list)


class TeachingRequestRecord(Schema):
    """An accepted learner request (by client_turn_id): a repeat returns the stored result instead of re-applying."""

    session_id: str
    client_turn_id: str
    request_hash: str
    turn_ids: list[str]
    created_at: datetime


# --- the model boundary --------------------------------------------------------------------------------------------


class ObjectiveBrief(Schema):
    concept_id: str
    concept_name: str
    description: str


class SectionBrief(Schema):
    ref: str  # "lesson:<section id>": what a teacher turn may cite
    concept_id: str
    heading: str
    explanation: str
    examples: list[str] = Field(default_factory=list)
    analogy: str | None = None
    citations: list[str] = Field(default_factory=list)


class PracticeItem(Schema):
    prompt: str
    answer: str
    accepted: list[str] = Field(default_factory=list)


class TurnBrief(Schema):
    speaker: Speaker
    turn_type: TurnType
    content: str


class QuestionBrief(Schema):
    kind: QuestionKind
    prompt: str
    choices: list[str] = Field(default_factory=list)
    expected_answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    difficulty: int


class StateBrief(Schema):
    difficulty: int
    min_difficulty: int
    max_difficulty: int
    strategy: str
    questions_asked: int
    correct_answers: int
    incorrect_answers: int
    consecutive_successes: int
    consecutive_failures: int
    hints_used: int
    mastery: float = Field(ge=0, le=1)  # the concept's current mastery (read-only context)


class EvidenceBrief(Schema):
    evidence_type: EvidenceType
    correct: bool | None = None
    hint_level: int = 0
    difficulty: int


class GroundingItem(Schema):
    ref: str  # "lesson:<section>", a research citation id, or "kb:<doc>"
    kind: Literal["lesson", "research", "knowledge_base"]
    title: str
    text: str
    citations: list[str] = Field(default_factory=list)  # research citation ids the item already carries


class TeachingTurnInput(Schema):
    """What the model sees: structured, minimal context. No learner id, session id, history beyond the recent
    turns, other goals or sessions, or metadata."""

    stage: Literal["turn", "answer"]
    action: TeachingAction
    subject: str
    topic: str | None = None  # the lesson's topic: where the knowledge base's practice probes are looked up
    level: str
    language: str
    objective: ObjectiveBrief
    lesson_title: str
    sections: list[SectionBrief] = Field(default_factory=list)
    practice: list[PracticeItem] = Field(default_factory=list)
    state: StateBrief
    recent_turns: list[TurnBrief] = Field(default_factory=list)
    evidence: list[EvidenceBrief] = Field(default_factory=list)
    question: QuestionBrief | None = None  # the question being answered or hinted at
    learner_answer: str | None = None
    answer_correct: bool | None = None
    correction: bool = False  # a FEEDBACK turn that may reveal the expected answer
    hint_level: int = Field(default=0, ge=0, le=3)
    reveal_answer: bool = False
    difficulty: int  # the difficulty the turn must be written at
    learner_question: str | None = None
    sources: list[GroundingItem] = Field(default_factory=list)  # lesson and research material a question may use

    @model_validator(mode="after")
    def _stage(self) -> TeachingTurnInput:
        if self.stage == "answer" and not self.learner_question:
            raise ValueError("the answer stage needs the learner's question")
        return self


class TeachingQuestion(Schema):
    kind: QuestionKind
    prompt: str = Field(min_length=1, max_length=1000)
    choices: list[str] = Field(default_factory=list)
    expected_answer: str = Field(min_length=1, max_length=300)
    accepted_answers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _choices(self) -> TeachingQuestion:
        if self.kind == "multiple_choice":
            if len(self.choices) < 2 or len(set(self.choices)) != len(self.choices):
                raise ValueError("a multiple-choice question needs at least two distinct choices")
            if self.expected_answer not in self.choices:
                raise ValueError("the expected answer must be one of the choices")
        elif self.choices:
            raise ValueError("only a multiple-choice question has choices")
        return self


class MisconceptionCandidate(Schema):
    concept_id: str = Field(min_length=1)
    misconception: str = Field(min_length=3, max_length=200)
    confidence: float = Field(ge=0, le=1)


class TeacherTurnOutput(Schema):
    """The model's structured teacher turn. It can only phrase the action the policy chose: no field can change
    mastery, objective status or the session's status (unknown fields are rejected)."""

    action: TeachingAction
    concept_id: str
    difficulty: int
    response: str = Field(min_length=1, max_length=4000)
    expected_response_type: Literal["free_text", "multiple_choice", "none"]
    question: TeachingQuestion | None = None
    hint_level: int = Field(default=0, ge=0, le=3)
    citations: list[str] = Field(default_factory=list)
    misconceptions: list[MisconceptionCandidate] = Field(default_factory=list)
    grounded: bool | None = None  # answer stage only
    limitation: str | None = None  # answer stage only: why the material cannot answer the question


class GroundedAnswer(Schema):
    """The model's answer to a learner question, from the grounding material only."""

    response: str = Field(min_length=1, max_length=4000)
    citations: list[str] = Field(default_factory=list)
    grounded: bool
    limitation: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GroundedAnswer:
        if self.grounded and not self.citations:
            raise ValueError("a grounded answer cites the material it uses")
        if not self.grounded and not self.limitation:
            raise ValueError("an ungrounded answer states its limitation")
        return self


class GroundingRequest(Schema):
    question: str = Field(min_length=1)
    subject: str
    candidates: list[GroundingItem] = Field(default_factory=list)
    max_items: int = Field(default=4, ge=1, le=20)


class GroundingPack(Schema):
    items: list[GroundingItem] = Field(default_factory=list)

    def allowed_citations(self) -> set[str]:
        return {i.ref for i in self.items} | {c for i in self.items for c in i.citations}


# --- service inputs and views --------------------------------------------------------------------------------------


class StartTeachingSession(Schema):
    objective_id: str | None = None
    action: Literal["LEARN", "REVIEW", "PRACTICE", "EVALUATE"] | None = None
    practice: bool = False  # practice sessions record practice evidence only: mastery is not updated
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class LearnerInput(Schema):
    answer: str = Field(min_length=1, max_length=2000)
    client_turn_id: str | None = Field(default=None, min_length=1, max_length=128)
    kind: LearnerInputKind = "answer"


class PublicQuestion(Schema):
    """The waiting question as the learner sees it: never the answer key."""

    turn_id: str
    kind: QuestionKind
    prompt: str
    choices: list[str] = Field(default_factory=list)
    difficulty: int
    hint_level: int


class PublicState(Schema):
    objective_id: str | None
    objective_description: str
    concept_id: str
    concept_name: str
    difficulty: int
    strategy: str
    questions_asked: int
    questions_answered: int
    correct_answers: int
    incorrect_answers: int
    consecutive_successes: int
    consecutive_failures: int
    hints_used: int
    misconceptions: list[str]
    difficulty_trajectory: list[int]
    difficulty_changes: list[DifficultyChange]
    turn_budget: int
    completion_reason: CompletionReason | None
    waiting_question: PublicQuestion | None


class SessionNextStep(Schema):
    kind: Literal["answer_question", "resume", "teacher_reply_pending", "learning_action", "none"]
    prompt: str | None = None
    learning_action: dict | None = None


class TeachingSessionView(Schema):
    session_id: str
    learner_id: str
    task_id: str
    lesson_id: str
    objective_id: str | None
    action: str
    mode: SessionMode
    status: TeachingSessionStatus
    turn_count: int
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    state: PublicState
    turns: list[TeachingTurn]
    objective_progress: dict | None = None
    next_action: SessionNextStep
    summary: TeachingSessionSummary | None = None
    outcome: SessionOutcome | None = None


class SessionStarted(Schema):
    session_id: str
    status: TeachingSessionStatus
    created: bool
    first_turn: TeachingTurn | None
    teacher_turns: list[TeachingTurn]
    objective: ObjectiveBrief
    objective_id: str | None
    difficulty: int
    waiting_question: PublicQuestion | None


class AnswerResult(Schema):
    session_id: str
    status: TeachingSessionStatus
    replayed: bool  # the same client_turn_id was already applied: nothing changed
    learner_turn: TeachingTurn
    teacher_turns: list[TeachingTurn]
    evidence: list[InteractionEvidence]
    correct: bool | None = None
    difficulty: int
    difficulty_change: DifficultyChange | None = None
    completion_reason: CompletionReason | None = None
    waiting_question: PublicQuestion | None = None
    summary: TeachingSessionSummary | None = None
    outcome: SessionOutcome | None = None
