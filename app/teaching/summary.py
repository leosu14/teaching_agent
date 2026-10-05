"""The session summary, built from the structured session state. Only `narrative` is model-written."""

from __future__ import annotations

from typing import Literal

from app.schemas.teaching import CompletionReason, TeachingSession, TeachingSessionSummary


def recommended_next(session: TeachingSession) -> Literal["advance", "practice", "reteach", "new_lesson"]:
    st = session.state
    if st.completion_reason == CompletionReason.OBJECTIVE_DEMONSTRATED:
        return "advance"
    if st.completion_reason == CompletionReason.REPEATED_FAILURE:
        return "new_lesson"
    return "practice" if st.correct_answers >= st.incorrect_answers else "reteach"


def build_summary(session: TeachingSession, narrative: str = "") -> TeachingSessionSummary:
    st = session.state
    if st.completion_reason is None:
        raise ValueError("only a session with a completion reason has a summary")
    unaided = st.correct_answers - st.assisted_correct
    strengths = []
    if st.correct_answers:
        strengths.append(f"answered {st.correct_answers} of {st.questions_answered} attempt(s) correctly")
    if unaided:
        strengths.append(f"{unaided} correct answer(s) without a hint")
    if st.difficulty_trajectory and max(st.difficulty_trajectory) > min(st.difficulty_trajectory):
        strengths.append(f"worked up to difficulty {max(st.difficulty_trajectory)}")
    difficulties = []
    if st.incorrect_answers:
        difficulties.append(f"{st.incorrect_answers} incorrect answer(s)")
    if st.hints_used:
        difficulties.append(f"needed {st.hints_used} hint(s)")
    difficulties += [f"difficulty lowered {c.before} -> {c.after}" for c in st.difficulty_history if c.after < c.before]
    return TeachingSessionSummary(
        session_id=session.session_id, lesson_id=session.lesson_id, objective_id=session.objective_id,
        concepts_practiced=[st.concept_id], strengths=strengths, difficulties=difficulties,
        misconceptions=list(dict.fromkeys(m.misconception for m in st.misconceptions)), hints_used=st.hints_used,
        questions_asked=st.questions_asked, correct_answers=st.correct_answers, incorrect_answers=st.incorrect_answers,
        difficulty_trajectory=list(st.difficulty_trajectory), evidence_ids=list(st.evidence_ids),
        recommended_next_action=recommended_next(session), completion_reason=st.completion_reason,
        practice=session.mode == "practice", narrative=narrative)
