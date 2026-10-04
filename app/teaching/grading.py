"""Deterministic answer grading and answer-leak checks. No model grades a session answer: the classification of an
answer (and so every state transition after it) is the same for the same answer to the same question."""

from __future__ import annotations

from app.schemas.teaching import PendingQuestion
from app.utils.text import normalize, reveals

__all__ = ["accepted", "grade", "normalize", "reveals"]


def accepted(question: PendingQuestion) -> set[str]:
    return {a for a in (normalize(question.expected_answer), *(normalize(x) for x in question.accepted_answers)) if a}


def grade(question: PendingQuestion, answer: str) -> bool:
    """Correct when the normalised answer equals the expected answer or an accepted alternative. A multiple-choice
    answer may also give the choice's letter or number (when it is not itself one of the choices)."""
    given = normalize(answer)
    if not given:
        return False
    if question.kind == "multiple_choice" and given not in {normalize(c) for c in question.choices}:
        labels = {**{chr(ord("a") + i): c for i, c in enumerate(question.choices)},
                  **{str(i + 1): c for i, c in enumerate(question.choices)}}
        if given in labels:
            given = normalize(labels[given])
    return given in accepted(question)
