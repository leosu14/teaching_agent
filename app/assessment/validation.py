"""Validation of a semantic grader's output: the boundary between a model and the grade.

Every rule is explicit and names itself in the rejection. A rejected candidate is never repaired into a grade: the
grader is asked again, and if it still fails the grade is UNCERTAIN (never INCORRECT).

  malformed_json        the output is not a JSON object
  state_mutation        the output carries a learner-state field (mastery, objective / goal / curriculum status, ...)
  schema                schema violation: a score outside 0-1, an unknown outcome, a missing or unknown field
  unknown_criterion     a criterion result the rubric does not have
  duplicate_criterion   a criterion scored twice
  missing_criterion     a rubric criterion without a result
  fabricated_citation   a citation that is not a ref of the lesson or research context the grader was given
  unknown_concept       a misconception about a concept the item and rubric do not assess
  contradictory_scores  the overall score disagrees with the grader's own criterion scores
  contradictory_outcome the proposed outcome contradicts the grader's own criterion scores

Safe normalisations (also explicit): criterion scores are snapped to the rubric's scale, the overall score and
outcome are replaced by the deterministic aggregate and classification, misconceptions below the configured
confidence or on a grade that is not INCORRECT / PARTIAL are dropped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from pydantic import ValidationError

from app.schemas.assessment import (
    AssessmentConfig,
    AssessmentOutcome,
    SemanticGradeCandidate,
    SemanticGradingRequest,
)

FORBIDDEN_FIELDS = frozenset({
    "mastery", "mastery_level", "mastery_update", "concept_mastery", "mastered", "objective_status",
    "objective_completed", "objective_mastered", "goal_status", "goal_completed", "goal_completion", "complete_goal",
    "curriculum", "curriculum_status", "curriculum_update", "learner_state", "learner_model", "evidence",
    "next_action", "status",
})


class CandidateRejected(ValueError):
    def __init__(self, rule: str, detail: str) -> None:
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


def _keys(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, inner in value.items():
            yield str(key)
            yield from _keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _keys(inner)


def parse_candidate(raw: str | dict) -> SemanticGradeCandidate:
    """Raw grader output (text or a parsed object) -> a schema-valid candidate, or CandidateRejected."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CandidateRejected("malformed_json", f"not JSON ({exc.msg})") from None
    if not isinstance(raw, dict):
        raise CandidateRejected("malformed_json", "the output must be a JSON object")
    reject_state_fields(raw)
    try:
        return SemanticGradeCandidate.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors())
        raise CandidateRejected("schema", problems) from None


def reject_state_fields(raw: dict) -> None:
    found = sorted({k for k in _keys(raw) if k.lower() in FORBIDDEN_FIELDS})
    if found:
        raise CandidateRejected("state_mutation", f"a grader cannot set learner state ({', '.join(found)})")


def raw_aggregate(candidate: SemanticGradeCandidate, request: SemanticGradingRequest) -> float:
    weights = {c.criterion_id: c.weight for c in request.rubric.criteria}
    return sum(r.score * weights.get(r.criterion_id, 0.0) for r in candidate.criterion_results)


def check_candidate(candidate: SemanticGradeCandidate, request: SemanticGradingRequest,
                    config: AssessmentConfig) -> None:
    """The rules that need the request: criteria, citations, concepts, consistency. Raises CandidateRejected."""
    rubric = request.rubric
    known = [c.criterion_id for c in rubric.criteria]
    given = [r.criterion_id for r in candidate.criterion_results]
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise CandidateRejected("unknown_criterion", f"the rubric has no criterion {unknown}")
    duplicated = sorted({g for g in given if given.count(g) > 1})
    if duplicated:
        raise CandidateRejected("duplicate_criterion", f"criteria scored more than once: {duplicated}")
    missing = [k for k in known if k not in given]
    if missing:
        raise CandidateRejected("missing_criterion", f"every rubric criterion needs a result; missing {missing}")
    invented = sorted(set(candidate.citations) - request.allowed_citations())
    if invented:
        raise CandidateRejected("fabricated_citation", f"{invented} are not refs of the lesson or research context")
    stray = sorted({m.concept_id for m in candidate.misconceptions} - request.known_concepts())
    if stray:
        raise CandidateRejected("unknown_concept", f"misconceptions about concepts the item does not assess: {stray}")
    aggregate = raw_aggregate(candidate, request)
    if abs(candidate.score - aggregate) > config.contradiction_tolerance:
        raise CandidateRejected("contradictory_scores", f"score {candidate.score:.2f} disagrees with the weighted "
                                                        f"criterion scores ({aggregate:.2f})")
    if ((candidate.outcome == AssessmentOutcome.CORRECT and aggregate < config.partial_threshold)
            or (candidate.outcome == AssessmentOutcome.INCORRECT and aggregate >= rubric.passing_threshold)):
        raise CandidateRejected("contradictory_outcome", f"outcome {candidate.outcome.value} contradicts the "
                                                         f"criterion scores ({aggregate:.2f})")


def validate_candidate(raw: str | dict | SemanticGradeCandidate, request: SemanticGradingRequest,
                       config: AssessmentConfig) -> SemanticGradeCandidate:
    """Every rule, in order. The same function guards the agent's output and the engine's input."""
    if isinstance(raw, SemanticGradeCandidate):  # re-validated: an instance may have been built without validation
        raw = raw.model_dump(mode="json", warnings=False)
    candidate = parse_candidate(raw)
    check_candidate(candidate, request, config)
    return candidate
