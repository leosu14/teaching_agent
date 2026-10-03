from __future__ import annotations

from datetime import timedelta

import pytest

from app.learner.frameworks import CEFR, MASTERY_SCALE, UnknownFramework, default_frameworks
from app.learner.mastery import MasteryUpdater
from app.learner.memory import UnknownLearner
from app.schemas.learner import AnswerEvaluation, ConceptMastery, LearnerProfileInput, LearningEvidence, SubjectState
from app.schemas.lesson import ConceptEstimate, ConceptRef, DiagnosticResult, LessonOutcome, LessonRequest
from tests.unit.helpers import NOW, scope, service

def test_level_frameworks_are_pluggable_and_generic() -> None:
    assert CEFR.level_for(CEFR.score_for("B1")) == "B1"
    assert MASTERY_SCALE.level_for(0.99) == "expert" and MASTERY_SCALE.level_for(0.0) == "novice"
    with pytest.raises(ValueError):
        CEFR.index("Z9")
    with pytest.raises(UnknownFramework):
        default_frameworks().get("elo")


def _evidence(correct: bool, difficulty: float, ref: str) -> LearningEvidence:
    return LearningEvidence(evidence_id=ref, learner_id="l", concept_id="c", source_type="exercise", source_ref=ref,
                            correctness="correct" if correct else "incorrect", score=1.0 if correct else 0.0,
                            difficulty=difficulty, timestamp=NOW)


def test_evidence_updates_mastery_and_exposure_only_schedules_review() -> None:
    updater = MasteryUpdater()
    c = ConceptMastery(concept_id="c", name="c", subject="s", mastery=0.3)
    c = updater.apply(c, _evidence(True, 0.8, "e1"))
    assert c.mastery > 0.6 and c.evidence_count == 1 and c.confidence == pytest.approx(0.4)
    high = c.mastery
    c = updater.apply(c, _evidence(False, 0.2, "e2"))
    assert c.mastery < high and c.next_review_at == NOW + timedelta(days=1)  # a miss is reviewed soon
    before = c.mastery
    c = updater.expose(c, NOW)  # being taught is not evidence: mastery is unchanged, a review is scheduled
    assert c.mastery == before and c.exposures == 1 and c.next_review_at is not None


def outcome(task_id: str = "t1") -> LessonOutcome:
    req = LessonRequest(raw_request="r", subject="physics", topic="forces", framework_id="mastery",
                        target_level="beginner", capabilities=["lesson.text"])
    evals = [
        AnswerEvaluation(question_id="q1", concept_id="newton1", answer="inertia", expected="inertia", correct=True, difficulty=0.5),
        AnswerEvaluation(question_id="q2", concept_id="newton2", answer="f=m", expected="f=ma", correct=False, difficulty=0.5),
    ]
    diag = DiagnosticResult(source="assessment", estimated_level="beginner", known=["newton1"], gaps=["newton2"],
                            starting_point="newton2", evaluations=evals,
                            concept_mastery=[ConceptEstimate(concept_id="newton1", mastery=0.75, confidence=0.55),
                                             ConceptEstimate(concept_id="newton2", mastery=0.2, confidence=0.55)])
    return LessonOutcome(task_id=task_id, learner_id="l1", request=req, diagnostic=diag,
                         concepts=[ConceptRef(concept_id="newton1", name="First law"),
                                   ConceptRef(concept_id="newton2", name="Second law")],
                         lesson_title="Forces", taught_concept_ids=["newton2"])


def test_record_lesson_updates_memory_and_is_idempotent() -> None:
    memory, clock = service()
    sc, events = scope()
    update = memory.record_lesson(outcome(), sc)
    assert update.estimated_level == "beginner"
    changes = {c.concept_id: c for c in update.changes}
    assert changes["newton1"].after > changes["newton1"].before
    assert "wrong answer" in changes["newton2"].reason and "taught in lesson" in changes["newton2"].reason
    assert [e.type for e in events] == ["learner.updated"]

    again = memory.record_lesson(outcome(), sc)  # same task replayed after a crash
    assert again.changes == update.changes
    profile = memory.get("l1")
    assert len(profile.lessons) == 1 and len(profile.assessments) == 1 and len(profile.mistakes) == 1


def test_snapshot_answers_what_the_learner_knows_and_needs() -> None:
    memory, clock = service()
    memory.record_lesson(outcome(), scope()[0])
    snap = memory.snapshot("l1", "physics", "mastery", None)
    assert snap.framework_levels == MASTERY_SCALE.levels and snap.estimated_level == "beginner"
    assert "newton2" in snap.weak and snap.recommended_next[0] == "newton2"
    assert snap.due_for_review == []
    clock.now = NOW + timedelta(days=2)
    assert "newton2" in memory.snapshot("l1", "physics", "mastery", None).due_for_review
    assert snap.recent_topics == ["forces"] and snap.recent_mistakes[0].concept_id == "newton2"


def test_profile_upsert_validates_framework_and_level() -> None:
    memory, _ = service()
    memory.upsert("l2", LearnerProfileInput(subjects=[SubjectState(subject="german", framework_id="cefr", target_level="B1")]))
    assert memory.get("l2").subjects["german"].target_level == "B1"
    with pytest.raises(UnknownFramework):
        memory.upsert("l2", LearnerProfileInput(subjects=[SubjectState(subject="chess", framework_id="elo")]))
    with pytest.raises(ValueError):
        memory.upsert("l2", LearnerProfileInput(subjects=[SubjectState(subject="german", framework_id="cefr", target_level="Z")]))
    with pytest.raises(UnknownLearner):
        memory.progress("missing")
