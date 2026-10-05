"""The grading pipeline. One implementation, used by every caller through the AssessmentService.

  1. normalisation                (deterministic)
  2. acceptable-answer matching   (deterministic: expected / acceptable answers, choices, true/false)
  3. deterministic rules          (an empty answer, an invalid choice, a known wrong answer, a closed question)
  4. rubric grading               (a deterministic rubric scores its criteria from their indicators)
  5. semantic grading             (only for free text the steps above cannot decide)

then, for 4 and 5: the rubric engine aggregates the criterion scores, the classifier decides the outcome against the
configured thresholds, and the grade is built. No model is called when deterministic grading decides, and a model's
output reaches the grade only after `validate_candidate`. A grade that cannot be decided is UNCERTAIN, never
INCORRECT.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.assessment import classification, rubric as rubrics
from app.assessment.matching import Match, match_answer
from app.assessment.validation import CandidateRejected, validate_candidate
from app.schemas.assessment import (
    CLOSED_TYPES,
    AssessmentConfig,
    AssessmentFeedback,
    AssessmentGrade,
    AssessmentItem,
    AssessmentOutcome,
    AssessmentRubric,
    CriterionResult,
    GraderType,
    GradingContext,
    Misconception,
    SemanticGradeCandidate,
    SemanticGraderResult,
    SemanticGradingRequest,
    SemanticItemBrief,
)

Outcome = AssessmentOutcome
RETRY_HINT = "Please answer again in your own words, as a complete sentence."


class SemanticGrader(Protocol):
    """Proposes a candidate grade for one free-text answer. Implementations call a model; they never persist."""

    async def grade(self, request: SemanticGradingRequest) -> SemanticGraderResult: ...


def build_request(item: AssessmentItem, rubric: AssessmentRubric, answer: str, context: GradingContext,
                  config: AssessmentConfig) -> SemanticGradingRequest:
    """The minimum the grader needs: the item, its rubric, the answer and the relevant lesson and research passages."""
    limit = config.max_context_passages
    return SemanticGradingRequest(
        language=item.language,
        item=SemanticItemBrief(prompt=item.prompt, expected_answer=item.expected_answer,
                               acceptable_answers=list(item.acceptable_answers), response_type=item.response_type,
                               concept_id=item.concept_id, difficulty=item.difficulty,
                               misconceptions=list(item.misconceptions)),
        rubric=rubric, learner_answer=answer, lesson_context=context.lesson_context[:limit],
        research_evidence=context.research_evidence[:limit])


class AssessmentEngine:
    def __init__(self, config: AssessmentConfig | None = None) -> None:
        self.config = config or AssessmentConfig()

    def rubric_for(self, item: AssessmentItem, rubric: AssessmentRubric | None) -> AssessmentRubric | None:
        if rubric is not None:
            return rubric
        return rubrics.default_rubric(item, self.config) if item.semantic else None

    def match(self, item: AssessmentItem, answer: str) -> Match:
        return match_answer(answer, expected=item.expected_answer, acceptable=item.acceptable_answers,
                            response_type=item.response_type, choices=item.choices, known_errors=item.known_errors,
                            language=item.language, config=self.config.normalization(item.language))

    async def grade(self, item: AssessmentItem, rubric: AssessmentRubric | None, answer: str, *, grade_id: str,
                    attempt_id: str, at: datetime, context: GradingContext | None = None,
                    grader: SemanticGrader | None = None) -> AssessmentGrade:
        if rubric is not None and item.rubric_id not in (None, rubric.rubric_id):
            raise ValueError(f"item {item.assessment_item_id} uses rubric {item.rubric_id}, not {rubric.rubric_id}")
        rubric = self.rubric_for(item, rubric)
        base = {"grade_id": grade_id, "assessment_item_id": item.assessment_item_id, "attempt_id": attempt_id,
                "learner_answer": answer, "max_score": item.max_score, "created_at": at,
                "rubric_id": rubric.rubric_id if rubric else None}
        found = self.match(item, answer)
        base["normalized_answer"] = found.normalized
        # 2. acceptable-answer matching
        if found.kind == "accepted":
            marks = rubrics.full_marks(rubric, "matches an acceptable answer") if rubric else None
            return self._grade(base, Outcome.CORRECT, 1.0, 1.0, GraderType.EXACT, matched=found.matched,
                               results=marks.results if marks else [],
                               feedback=AssessmentFeedback(outcome=Outcome.CORRECT,
                                                           strengths=["matches the expected answer"]))
        # 3. deterministic rules
        if found.kind in ("empty", "invalid", "known_error") or item.response_type in CLOSED_TYPES \
                or not item.semantic:
            return self._rule_grade(item, base, found)
        assert rubric is not None
        # 4. a deterministic rubric
        if rubric.deterministic:
            scores, rationales = rubrics.indicator_scores(rubric, answer, self.config.normalization(item.language))
            agg = rubrics.aggregate(rubric, scores, rationales)
            return self._classified(base, rubric, agg, 1.0, GraderType.RUBRIC, [], self._rubric_feedback(agg))
        # 5. semantic grading
        if grader is None or not self.config.semantic_enabled:
            return self._uncertain(base, "semantic grading is not available for this free-text answer")
        request = build_request(item, rubric, answer, context or GradingContext(), self.config)
        result = await grader.grade(request)
        base["grader"] = result.usage
        if result.candidate is None:
            return self._uncertain(base, f"the semantic grader produced no valid grade ({result.error})")
        try:
            candidate = validate_candidate(result.candidate, request, self.config)
        except CandidateRejected as exc:
            return self._uncertain(base, f"the semantic grader's output was rejected ({exc})")
        if candidate.insufficient_context:
            return self._uncertain(base, "the lesson and research context do not let the answer be graded",
                                   explanation=candidate.feedback.explanation)
        agg = rubrics.aggregate(rubric, {r.criterion_id: r.score for r in candidate.criterion_results},
                                {r.criterion_id: r.rationale for r in candidate.criterion_results})
        return self._classified(base, rubric, agg, candidate.confidence, GraderType.SEMANTIC, candidate.citations,
                                None, candidate=candidate, concept=item.concept_id)

    # --- grade builders --------------------------------------------------------------------------------------------

    def _rule_grade(self, item: AssessmentItem, base: dict, found: Match) -> AssessmentGrade:
        if found.kind == "empty":
            errors, grader = ["the answer is empty"], GraderType.RULE
        elif found.kind == "invalid":
            what = "one of the choices" if item.response_type.value == "MULTIPLE_CHOICE" else "true or false"
            errors, grader = [f"the answer is not {what}"], GraderType.RULE
        elif found.kind == "known_error":
            errors, grader = ["a known incorrect answer"], GraderType.RULE
        else:
            errors, grader = ["does not match the expected answer"], GraderType.EXACT
        misconceptions = []
        explanation = ""
        if found.known_error is not None and found.known_error.misconception_type:
            e = found.known_error
            misconceptions.append(Misconception(concept_id=item.concept_id, type=e.misconception_type,
                                                description=e.description or e.misconception_type, confidence=1.0,
                                                source="rule"))
            explanation = e.description or ""
        return self._grade(base, Outcome.INCORRECT, 0.0, 1.0, grader, matched=found.matched,
                           misconceptions=misconceptions,
                           feedback=AssessmentFeedback(outcome=Outcome.INCORRECT, errors=errors,
                                                       explanation=explanation))

    def _classified(self, base: dict, rubric: AssessmentRubric, agg: rubrics.Aggregate, confidence: float,
                    grader: GraderType, citations: list[str], feedback: AssessmentFeedback | None, *,
                    candidate: SemanticGradeCandidate | None = None, concept: str | None = None) -> AssessmentGrade:
        decided = classification.classify(agg.fraction, confidence, correct_threshold=rubric.passing_threshold,
                                          config=self.config, scale=rubric.scale,
                                          required_unmet=bool(agg.required_unmet))
        notes = [] if decided.reason is None or decided.outcome == Outcome.UNCERTAIN else [decided.reason]
        if decided.outcome == Outcome.UNCERTAIN:
            return self._uncertain(base, decided.reason or "not decided", results=agg.results,
                                   fraction=decided.fraction, confidence=confidence)
        misconceptions = []
        if candidate is not None:
            if decided.outcome in (Outcome.INCORRECT, Outcome.PARTIAL):
                seen: set[tuple[str, str]] = set()
                for m in candidate.misconceptions:
                    key = (m.concept_id, m.type.strip().lower())
                    if m.confidence < self.config.misconception_min_confidence or key in seen:
                        continue
                    seen.add(key)
                    misconceptions.append(Misconception(concept_id=m.concept_id, type=m.type,
                                                        description=m.description, confidence=m.confidence,
                                                        source="semantic"))
            f = candidate.feedback
            feedback = AssessmentFeedback(outcome=decided.outcome, strengths=f.strengths, errors=f.errors,
                                          explanation=f.explanation, next_hint=f.next_hint, citations=citations)
        assert feedback is not None
        feedback = feedback.model_copy(update={"outcome": decided.outcome})
        return self._grade(base, decided.outcome, decided.fraction, confidence, grader, results=agg.results,
                           misconceptions=misconceptions, feedback=feedback, notes=notes)

    @staticmethod
    def _rubric_feedback(agg: rubrics.Aggregate) -> AssessmentFeedback:
        return AssessmentFeedback(outcome=Outcome.INCORRECT,  # replaced by the classified outcome
                                  strengths=[f"meets {r.criterion_id}" for r in agg.results if r.met],
                                  errors=[f"does not meet {r.criterion_id}" for r in agg.results if not r.met])

    def _uncertain(self, base: dict, reason: str, *, results: list[CriterionResult] | None = None,
                   fraction: float = 0.0, confidence: float = 0.0, explanation: str = "") -> AssessmentGrade:
        feedback = AssessmentFeedback(
            outcome=Outcome.UNCERTAIN, next_hint=RETRY_HINT,
            explanation=explanation or "Your answer could not be graded reliably, so it does not count yet.")
        return self._grade(base, Outcome.UNCERTAIN, fraction, confidence, GraderType.SEMANTIC, results=results or [],
                           feedback=feedback, uncertainty_reason=reason)

    @staticmethod
    def _grade(base: dict, outcome: AssessmentOutcome, fraction: float, confidence: float, grader: GraderType, *,
               feedback: AssessmentFeedback, matched: str | None = None, results: list[CriterionResult] = (),
               misconceptions: list[Misconception] = (), notes: list[str] = (),
               uncertainty_reason: str | None = None) -> AssessmentGrade:
        fraction = round(max(0.0, min(1.0, fraction)), 4)
        return AssessmentGrade(**base, score=round(fraction * base["max_score"], 4),
                               percentage=round(fraction * 100, 2), outcome=outcome,
                               criterion_results=list(results), confidence=round(confidence, 4), grader_type=grader,
                               matched=matched, misconceptions=list(misconceptions), feedback=feedback,
                               notes=list(notes), uncertainty_reason=uncertainty_reason)
