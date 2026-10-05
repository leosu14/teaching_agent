"""Answer grading and answer-leak checks. A session answer is graded by the AssessmentService (its validated grade
reaches the engine as an AnswerAssessment); without one, the engine grades a closed question with the same
deterministic matching the assessment layer uses. No model decides a transition: the same grade of the same answer
always gives the same transition."""

from __future__ import annotations

from app.assessment.matching import match_answer
from app.schemas.assessment import ResponseType
from app.schemas.teaching import AnswerAssessment, PendingQuestion
from app.utils.text import normalize, reveals

__all__ = ["RESPONSE_TYPES", "accepted", "deterministic", "grade", "normalize", "reveals"]

RESPONSE_TYPES = {"short_answer": ResponseType.SHORT_TEXT, "multiple_choice": ResponseType.MULTIPLE_CHOICE,
                  "free_text": ResponseType.FREE_TEXT}


def accepted(question: PendingQuestion) -> set[str]:
    return {a for a in (normalize(question.expected_answer), *(normalize(x) for x in question.accepted_answers)) if a}


def grade(question: PendingQuestion, answer: str) -> bool:
    """Correct when the normalised answer equals the expected answer or an accepted alternative. A multiple-choice
    answer may also give the choice's letter or number (when it is not itself one of the choices)."""
    found = match_answer(answer, expected=question.expected_answer, acceptable=question.accepted_answers,
                         response_type=RESPONSE_TYPES[question.kind], choices=question.choices)
    return found.kind == "accepted"


def deterministic(question: PendingQuestion, answer: str) -> AnswerAssessment:
    correct = grade(question, answer)
    return AnswerAssessment(outcome="CORRECT" if correct else "INCORRECT", score=1.0 if correct else 0.0,
                            grader_type="EXACT")
