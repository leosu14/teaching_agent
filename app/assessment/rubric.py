"""The rubric engine: deterministic scoring and aggregation.

`final = Σ criterion_score × criterion_weight` over the rubric's criteria, each score snapped to the rubric's scale.
Weights are validated to sum to 1.0 when the rubric is built (an invalid rubric is rejected). A grader (or the
indicators of a deterministic rubric) only proposes criterion scores; this module computes the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.assessment.normalization import normalize_answer
from app.schemas.assessment import (
    AssessmentConfig,
    AssessmentItem,
    AssessmentRubric,
    CriterionResult,
    NormalizationConfig,
    RubricCriterion,
)

MET = 0.5  # a criterion scoring at least this is met


@dataclass(frozen=True)
class Aggregate:
    fraction: float  # 0-1
    results: list[CriterionResult]
    required_unmet: list[str]  # required criteria that are not met: the grade cannot be CORRECT


def default_rubric(item: AssessmentItem, config: AssessmentConfig) -> AssessmentRubric:
    """The rubric of a free-text item that names none: one required criterion, the expected meaning."""
    return AssessmentRubric(
        rubric_id=f"default:{item.assessment_item_id}", name="Expected meaning",
        description="The answer conveys the expected answer's meaning.",
        criteria=[RubricCriterion(criterion_id="meaning", weight=1.0, required=True, concept_id=item.concept_id,
                                  description=f"Conveys the meaning of: {item.expected_answer}")],
        passing_threshold=config.correct_threshold, language=item.language)


def aggregate(rubric: AssessmentRubric, scores: Mapping[str, float], rationales: Mapping[str, str] | None = None
              ) -> Aggregate:
    """`scores` must hold exactly the rubric's criteria (validated before)."""
    results = []
    for c in rubric.criteria:
        score = rubric.scale.snap(scores[c.criterion_id])
        results.append(CriterionResult(criterion_id=c.criterion_id, score=score, weight=c.weight,
                                       weighted_score=round(score * c.weight, 6), met=score >= MET,
                                       rationale=(rationales or {}).get(c.criterion_id, "")))
    fraction = round(min(1.0, sum(r.weighted_score for r in results)), 4)
    unmet = [c.criterion_id for c, r in zip(rubric.criteria, results, strict=True) if c.required and not r.met]
    return Aggregate(fraction=fraction, results=results, required_unmet=unmet)


def full_marks(rubric: AssessmentRubric, rationale: str) -> Aggregate:
    return aggregate(rubric, {c.criterion_id: 1.0 for c in rubric.criteria},
                     {c.criterion_id: rationale for c in rubric.criteria})


def indicator_scores(rubric: AssessmentRubric, answer: str, config: NormalizationConfig | None = None
                     ) -> tuple[dict[str, float], dict[str, str]]:
    """A deterministic rubric: a criterion is met (1.0) when the answer contains one of its indicators as whole
    words, otherwise 0.0."""
    haystack = f" {normalize_answer(answer, config)} "
    scores, rationales = {}, {}
    for c in rubric.criteria:
        hit = next((i for i in c.indicators if (n := normalize_answer(i, config)) and f" {n} " in haystack), None)
        scores[c.criterion_id] = 1.0 if hit else 0.0
        rationales[c.criterion_id] = f"contains '{hit}'" if hit else "no indicator found"
    return scores, rationales
