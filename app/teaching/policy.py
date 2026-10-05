"""The deterministic teaching policy: what the teacher does next, how difficulty adapts and when a session ends.

Everything here is a pure function of the session state, the configuration and the graded learner input. A model
phrases the turns these functions choose; it never chooses them.

Difficulty (`DifficultyController`):
  - correct after a hint: keep the difficulty (it still counts as a success in the streak);
  - `increase_after_successes` consecutive correct answers: one level harder (the streak restarts);
  - `decrease_after_failures` consecutive incorrect answers: one level easier (the streak restarts).

Answers are graded by the AssessmentService (CORRECT, PARTIAL, INCORRECT or UNCERTAIN); the policy reads the grade:
  - partially correct: keep the difficulty (the success streak restarts; it is not a failure);
  - uncertain: nothing changes (no difficulty change, no count), the learner is asked to answer again.

Next turns after an answer (`TeachingPolicy.after_answer`):
  - the session's completion condition holds: SUMMARIZE (then COMPLETE);
  - uncertain: FEEDBACK asking for a clearer answer (the same question stays open);
  - correct: FEEDBACK, then the next question;
  - partially correct: FEEDBACK on what is missing, then the next question;
  - incorrect and the misconception repeats (`reteach_after_misconceptions`): RETEACH, then a new question;
  - incorrect with a hint level left: HINT at the next level (the same question stays open);
  - incorrect without hints left: FEEDBACK that corrects (reveals the answer), then a new question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.schemas.teaching import (
    CompletionReason,
    PendingQuestion,
    PlannedTurn,
    SessionMode,
    SessionState,
    TeachingAction,
    TeachingConfig,
)

STRATEGIES = {"easier": "guided practice with worked examples", "middle": "independent practice",
              "harder": "challenge: apply the rule without cues"}
RETEACH_STRATEGY = "reteach with a different explanation"


@dataclass(frozen=True)
class DifficultyDecision:
    difficulty: int
    successes: int
    failures: int
    change: Literal["increase", "decrease", "keep"]
    reason: str


class DifficultyController:
    def __init__(self, config: TeachingConfig) -> None:
        self.config = config

    def after_answer(self, difficulty: int, successes: int, failures: int, *, correct: bool,
                     hinted: bool) -> DifficultyDecision:
        c = self.config
        if correct:
            successes, failures = successes + 1, 0
            if hinted:
                return DifficultyDecision(difficulty, successes, failures, "keep", "correct after a hint: keep")
            if successes >= c.increase_after_successes and difficulty < c.max_difficulty:
                return DifficultyDecision(difficulty + 1, 0, 0, "increase",
                                          f"{successes} consecutive correct answer(s): harder")
            return DifficultyDecision(difficulty, successes, failures, "keep", "correct")
        successes, failures = 0, failures + 1
        if failures >= c.decrease_after_failures and difficulty > c.min_difficulty:
            return DifficultyDecision(difficulty - 1, 0, 0, "decrease",
                                      f"{failures} consecutive incorrect answer(s): easier")
        return DifficultyDecision(difficulty, successes, failures, "keep", "incorrect")

    @staticmethod
    def after_partial(difficulty: int, failures: int) -> DifficultyDecision:
        """A partially correct answer is neither a success nor a failure: keep the level, restart the success
        streak, leave the failure streak as it is."""
        return DifficultyDecision(difficulty, 0, failures, "keep", "partially correct: keep")


class TeachingPolicy:
    def __init__(self, config: TeachingConfig) -> None:
        self.config = config
        self.difficulty = DifficultyController(config)

    @staticmethod
    def strategy(config: TeachingConfig, difficulty: int, *, reteach: bool = False) -> str:
        if reteach:
            return RETEACH_STRATEGY
        if difficulty <= config.min_difficulty and config.min_difficulty < config.max_difficulty:
            return STRATEGIES["easier"]
        if difficulty >= config.max_difficulty and config.min_difficulty < config.max_difficulty:
            return STRATEGIES["harder"]
        return STRATEGIES["middle"]

    def opening(self, action: str, mode: SessionMode) -> list[PlannedTurn]:
        """LEARN and REVIEW explain first; PRACTICE and EVALUATE go straight to a question."""
        if action == "EVALUATE":
            return [PlannedTurn(action=TeachingAction.CHECK)]
        if action == "PRACTICE" or mode == "practice":
            return [PlannedTurn(action=TeachingAction.PRACTICE)]
        return [PlannedTurn(action=TeachingAction.EXPLAIN), PlannedTurn(action=TeachingAction.ASK)]

    def question_action(self, state: SessionState, mode: SessionMode) -> TeachingAction:
        """PRACTICE in a practice session; CHECK when one more correct answer could demonstrate the objective;
        otherwise ASK."""
        if mode == "practice":
            return TeachingAction.PRACTICE
        if (state.correct_answers + 1 >= self.config.demonstration_correct
                and state.difficulty >= self.config.demonstration_min_difficulty):
            return TeachingAction.CHECK
        return TeachingAction.ASK

    def completion(self, state: SessionState, *, correct: bool, hinted: bool, question_difficulty: int,
                   question_open: bool, turn_count: int) -> CompletionReason | None:
        """Checked after every graded answer (state already updated). The first rule that holds ends the session."""
        c = self.config
        if (correct and not hinted and state.correct_answers >= c.demonstration_correct
                and question_difficulty >= c.demonstration_min_difficulty):
            return CompletionReason.OBJECTIVE_DEMONSTRATED
        if state.incorrect_answers >= c.max_incorrect:
            return CompletionReason.REPEATED_FAILURE
        if state.questions_asked >= c.max_questions and not question_open:
            return CompletionReason.QUESTION_LIMIT_REACHED
        if turn_count + 3 > c.max_turns:  # room for the teacher's reply and the summary
            return CompletionReason.TURN_BUDGET_REACHED
        return None

    def after_answer(self, state: SessionState, question: PendingQuestion, *, correct: bool, answer_turn_id: str,
                     mode: SessionMode, misconceptions_before: int, outcome: str | None = None
                     ) -> tuple[list[PlannedTurn], bool]:
        """The teacher turns owed after a graded answer, and whether the question stays open (a hint follows, or the
        answer could not be graded and is asked for again)."""
        c = self.config
        if outcome == "UNCERTAIN":
            return [PlannedTurn(action=TeachingAction.FEEDBACK, responds_to=answer_turn_id, clarify=True)], True
        if outcome == "PARTIAL":
            return [PlannedTurn(action=TeachingAction.FEEDBACK, responds_to=answer_turn_id),
                    PlannedTurn(action=self.question_action(state, mode))], False
        if correct:
            return [PlannedTurn(action=TeachingAction.FEEDBACK, responds_to=answer_turn_id),
                    PlannedTurn(action=self.question_action(state, mode))], False
        if misconceptions_before >= c.reteach_after_misconceptions:
            return [PlannedTurn(action=TeachingAction.RETEACH, responds_to=answer_turn_id),
                    PlannedTurn(action=self.question_action(state, mode))], False
        if c.hints_enabled and question.hint_level < c.max_hint_level:
            return [PlannedTurn(action=TeachingAction.HINT, responds_to=answer_turn_id,
                                hint_level=question.hint_level + 1)], True
        return [PlannedTurn(action=TeachingAction.FEEDBACK, responds_to=answer_turn_id, correction=True),
                PlannedTurn(action=self.question_action(state, mode))], False
