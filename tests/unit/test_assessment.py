"""The assessment core: normalisation, matching, rubrics, classification, the validation of a semantic grader's
output (adversarial), the grading pipeline and the in-memory store. Pure: no container, no provider."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.agents.assessment.agent import SemanticGraderAgent
from app.agents.base import OutputRejected
from app.assessment import classification, rubric as rubrics
from app.assessment.engine import RETRY_HINT, AssessmentEngine, build_request
from app.assessment.errors import AttemptExists, ItemConflict
from app.assessment.matching import match_answer
from app.assessment.normalization import fold_accents, normalize_answer
from app.assessment.repository import InMemoryAssessmentRepository
from app.assessment.validation import FORBIDDEN_FIELDS, CandidateRejected, validate_candidate
from app.schemas.assessment import (
    AssessmentAttempt,
    AssessmentConfig,
    AssessmentGrade,
    AssessmentItem,
    AssessmentOutcome,
    AssessmentRubric,
    AttemptOutcome,
    ContextPassage,
    GraderType,
    GraderUsage,
    GradingContext,
    KnownError,
    MisconceptionPattern,
    NormalizationConfig,
    ResponseType,
    ScoreScale,
    SemanticGradeCandidate,
    SemanticGraderResult,
    SemanticGradingRequest,
)

AT = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
CONCEPT = "es.past_contrast"
Out = AssessmentOutcome


def free_item(**kw) -> AssessmentItem:
    data = {"assessment_item_id": "why", "lesson_id": "lesson-1", "concept_id": CONCEPT,
            "prompt": "¿Por qué usamos el pretérito en esta frase?", "expected_answer": "The action is completed.",
            "response_type": ResponseType.FREE_TEXT, "rubric_id": "r1", "language": "es",
            "misconceptions": [MisconceptionPattern(type="habitual", description="treats it as habitual",
                                                    cues=["habitual"])]}
    return AssessmentItem(**{**data, **kw})


def short_item(**kw) -> AssessmentItem:
    data = {"assessment_item_id": "conj", "concept_id": CONCEPT, "prompt": "Ayer yo ___ (hablar).",
            "expected_answer": "hablé", "response_type": ResponseType.SHORT_TEXT, "language": "es",
            "known_errors": [KnownError(answer="hablaba", misconception_type="imperfect_for_completed",
                                        description="imperfect for a completed action")]}
    return AssessmentItem(**{**data, **kw})


def two_criteria(**kw) -> AssessmentRubric:
    data = {"rubric_id": "r1", "name": "why", "criteria": [
        {"criterion_id": "completed", "description": "the action is completed", "weight": 0.6, "required": True,
         "indicators": ["terminó"]},
        {"criterion_id": "bounded", "description": "a bounded moment", "weight": 0.4, "indicators": ["ayer"]}]}
    return AssessmentRubric.model_validate({**data, **kw})


CONTEXT = GradingContext(lesson_context=[ContextPassage(ref="lesson:s1", kind="lesson", title="Contrast",
                                                        text="The preterite tells completed events.")])


def candidate(**kw) -> dict:
    data = {"score": 1.0, "outcome": "CORRECT", "confidence": 0.95,
            "criterion_results": [{"criterion_id": "completed", "score": 1.0, "rationale": "says it ended"},
                                  {"criterion_id": "bounded", "score": 1.0, "rationale": "says when"}],
            "misconceptions": [], "feedback": {"strengths": ["clear"], "explanation": "Completed events."},
            "citations": ["lesson:s1"]}
    return {**data, **kw}


class FakeGrader:
    """A SemanticGrader returning scripted output (a dict is validated as a candidate, a string is an error)."""

    def __init__(self, *outputs) -> None:
        self.outputs = list(outputs)
        self.requests: list[SemanticGradingRequest] = []

    async def grade(self, request: SemanticGradingRequest) -> SemanticGraderResult:
        self.requests.append(request)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        usage = GraderUsage(llm_calls=1, provider="mock", model="mock-standard", input_tokens=10, output_tokens=5,
                            estimated_cost_usd=0.0001)
        if isinstance(out, str):
            return SemanticGraderResult(error=out, usage=usage)
        return SemanticGraderResult(candidate=SemanticGradeCandidate.model_validate(out), usage=usage)


async def grade(item, rubric, answer, grader=None, config=None, context=CONTEXT) -> AssessmentGrade:
    return await AssessmentEngine(config).grade(item, rubric, answer, grade_id="g1", attempt_id="a1", at=AT,
                                                context=context, grader=grader)


# --- normalisation and matching ---------------------------------------------------------------------------------


def test_normalisation_is_surface_only() -> None:
    assert normalize_answer("  ¿Qué   PASÓ? ") == "qué pasó"
    folded = NormalizationConfig(fold_accents=True)
    assert normalize_answer("¿Qué pasó?", folded) == "que paso"
    assert fold_accents("año") == "año" and fold_accents("canción") == "cancion"  # ñ is a letter, not an accent
    assert normalize_answer("no") != normalize_answer("sí")


@pytest.mark.parametrize(("answer", "kind"), [
    ("Hablé.", "accepted"), (" HABLÉ ", "accepted"), ("hable", "no_match"), ("hablaba", "known_error"),
    ("", "empty"), ("   ", "empty"), ("hablo", "no_match")])
def test_short_text_matching(answer: str, kind: str) -> None:
    item = short_item()
    found = match_answer(answer, expected=item.expected_answer, acceptable=item.acceptable_answers,
                         response_type=item.response_type, known_errors=item.known_errors, language="es")
    assert found.kind == kind


def test_accent_folding_only_for_configured_languages() -> None:
    config = AssessmentConfig(accent_insensitive_languages=["es"])
    assert AssessmentEngine(config).match(short_item(), "hable").kind == "accepted"
    assert AssessmentEngine(config).match(short_item(language="pt"), "hable").kind == "no_match"


@pytest.mark.parametrize(("answer", "kind"), [("b", "accepted"), ("2", "accepted"), ("Sonaba", "no_match"),
                                              ("Sonó", "accepted"), ("d", "invalid"), ("llovió", "invalid")])
def test_multiple_choice_matching(answer: str, kind: str) -> None:
    found = match_answer(answer, expected="sonó", response_type=ResponseType.MULTIPLE_CHOICE,
                         choices=["sonaba", "sonó", "suena"], language="es")
    assert found.kind == kind


@pytest.mark.parametrize(("answer", "language", "kind"), [
    ("verdadero", "es", "accepted"), ("Cierto", "es", "accepted"), ("true", "es", "accepted"),
    ("falso", "es", "no_match"), ("vrai", "fr", "accepted"), ("quizás", "es", "invalid")])
def test_true_false_matching_by_language(answer: str, language: str, kind: str) -> None:
    assert match_answer(answer, expected="true", response_type=ResponseType.TRUE_FALSE,
                        language=language).kind == kind


# --- items and rubrics ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    {"criteria": [{"criterion_id": "a", "description": "a", "weight": 0.5},
                  {"criterion_id": "b", "description": "b", "weight": 0.4}]},  # weights sum to 0.9
    {"criteria": [{"criterion_id": "a", "description": "a", "weight": 0.5},
                  {"criterion_id": "a", "description": "b", "weight": 0.5}]},  # duplicate criterion
    {"criteria": []},
    {"criteria": [{"criterion_id": "a", "description": "a", "weight": 1.0}], "deterministic": True},  # no indicators
    {"criteria": [{"criterion_id": "a", "description": "a", "weight": 1.2}]},
])
def test_invalid_rubrics_are_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        AssessmentRubric.model_validate({"rubric_id": "r", "name": "r", **bad})


def test_invalid_items_are_rejected() -> None:
    with pytest.raises(ValidationError):
        short_item(response_type=ResponseType.MULTIPLE_CHOICE, choices=["a", "b"])  # expected not a choice
    with pytest.raises(ValidationError):
        short_item(choices=["a", "b"])  # choices on a short answer
    with pytest.raises(ValidationError):
        short_item(response_type=ResponseType.TRUE_FALSE)  # expects true or false
    with pytest.raises(ValidationError):
        ScoreScale(points=[0.0, 0.5, 0.4, 1.0])


def test_rubric_aggregation_is_the_weighted_sum_on_the_scale() -> None:
    r = two_criteria()
    agg = rubrics.aggregate(r, {"completed": 1.0, "bounded": 0.0})
    assert agg.fraction == 0.6 and not agg.required_unmet
    agg = rubrics.aggregate(r, {"completed": 0.0, "bounded": 1.0})
    assert agg.fraction == 0.4 and agg.required_unmet == ["completed"]
    agg = rubrics.aggregate(r, {"completed": 0.6, "bounded": 0.9})  # snapped to 0.5 and 1.0
    assert [c.score for c in agg.results] == [0.5, 1.0] and agg.fraction == 0.7
    assert [c.weighted_score for c in agg.results] == [0.3, 0.4]


def test_default_rubric_is_one_required_meaning_criterion() -> None:
    r = rubrics.default_rubric(free_item(rubric_id=None), AssessmentConfig())
    assert [(c.criterion_id, c.weight, c.required) for c in r.criteria] == [("meaning", 1.0, True)]


# --- classification -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("fraction", "confidence", "outcome"), [
    (1.0, 0.95, Out.CORRECT), (0.8, 0.85, Out.CORRECT), (0.6, 0.9, Out.PARTIAL), (0.4, 0.9, Out.PARTIAL),
    (0.2, 0.9, Out.INCORRECT), (1.0, 0.7, Out.PARTIAL), (0.0, 0.7, Out.UNCERTAIN), (0.5, 0.7, Out.PARTIAL),
    (1.0, 0.59, Out.UNCERTAIN), (0.0, 0.1, Out.UNCERTAIN)])
def test_classification_thresholds(fraction: float, confidence: float, outcome: AssessmentOutcome) -> None:
    c = classification.classify(fraction, confidence, correct_threshold=0.8, config=AssessmentConfig(),
                                scale=ScoreScale())
    assert c.outcome == outcome
    if outcome == Out.PARTIAL and fraction >= 0.8:
        assert c.fraction == 0.75  # capped below the correct threshold, so the score agrees with the outcome


def test_classification_thresholds_are_configurable() -> None:
    strict = AssessmentConfig(mid_confidence="uncertain", accept_confidence=0.9, min_confidence=0.5)
    assert classification.classify(1.0, 0.85, correct_threshold=0.8, config=strict,
                                   scale=ScoreScale()).outcome == Out.UNCERTAIN
    assert classification.classify(1.0, 0.95, correct_threshold=0.8, config=strict,
                                   scale=ScoreScale()).outcome == Out.CORRECT
    capped = classification.classify(1.0, 0.95, correct_threshold=0.8, config=AssessmentConfig(),
                                     scale=ScoreScale(), required_unmet=True)
    assert capped.outcome == Out.PARTIAL and capped.fraction == 0.75
    with pytest.raises(ValidationError):
        AssessmentConfig(min_confidence=0.9, accept_confidence=0.8)


# --- adversarial model output -------------------------------------------------------------------------------------


def request_for(item=None, rubric=None, answer="Porque terminó ayer.") -> SemanticGradingRequest:
    return build_request(item or free_item(), rubric or two_criteria(), answer, CONTEXT, AssessmentConfig())


@pytest.mark.parametrize(("raw", "rule"), [
    (candidate(score=1.4), "schema"),
    (candidate(score=-0.1), "schema"),
    (candidate(outcome="EXCELLENT"), "schema"),
    (candidate(criterion_results=[{"criterion_id": "completed", "score": 1.5}, {"criterion_id": "bounded",
                                                                               "score": 1.0}]), "schema"),
    (candidate(confidence=2), "schema"),
    (candidate(criterion_results=[{"criterion_id": "completed", "score": 1.0}]), "missing_criterion"),
    (candidate(criterion_results=[{"criterion_id": "completed", "score": 1.0}, {"criterion_id": "bounded",
                                                                               "score": 1.0},
                                  {"criterion_id": "style", "score": 1.0}]), "unknown_criterion"),
    (candidate(criterion_results=[{"criterion_id": "completed", "score": 1.0}, {"criterion_id": "completed",
                                                                               "score": 1.0}]),
     "duplicate_criterion"),
    (candidate(citations=["https://example.com/made-up"]), "fabricated_citation"),
    (candidate(misconceptions=[{"concept_id": "math.algebra", "type": "x", "description": "x",
                                "confidence": 0.9}]), "unknown_concept"),
    (candidate(mastery=0.99), "state_mutation"),
    (candidate(objective_status="MASTERED"), "state_mutation"),
    (candidate(goal_completed=True), "state_mutation"),
    (candidate(feedback={"explanation": "ok", "mastery_update": {"es.past_contrast": 1}}), "state_mutation"),
    ("{not json", "malformed_json"),
    ("[1, 2]", "malformed_json"),
    (candidate(score=0.1), "contradictory_scores"),
    (candidate(score=0.0, outcome="INCORRECT", criterion_results=[{"criterion_id": "completed", "score": 0.0},
                                                                  {"criterion_id": "bounded", "score": 1.0}],
               confidence=0.9) | {"score": 0.9}, "contradictory_scores"),
    (candidate(score=0.0, outcome="CORRECT", criterion_results=[{"criterion_id": "completed", "score": 0.0},
                                                                {"criterion_id": "bounded", "score": 0.0}]),
     "contradictory_outcome"),
    (candidate(outcome="INCORRECT"), "contradictory_outcome"),
])
def test_adversarial_grader_output_is_rejected_by_a_named_rule(raw, rule: str) -> None:
    payload = raw if isinstance(raw, str) else json.dumps(raw)
    with pytest.raises(CandidateRejected) as err:
        validate_candidate(payload, request_for(), AssessmentConfig())
    assert err.value.rule == rule


def test_every_learner_state_field_is_forbidden() -> None:
    for name in ("mastery", "objective_status", "goal_completed", "curriculum_status", "next_action"):
        assert name in FORBIDDEN_FIELDS


def test_the_agent_applies_the_same_validation() -> None:
    agent = SemanticGraderAgent()
    request = request_for()
    agent.check(SemanticGradeCandidate.model_validate(candidate()), request)
    with pytest.raises(OutputRejected, match="fabricated_citation"):
        agent.check(SemanticGradeCandidate.model_validate(candidate(citations=["lesson:nope"])), request)
    with pytest.raises(OutputRejected, match="contradictory_scores"):
        agent.check(SemanticGradeCandidate.model_validate(candidate(score=0.2)), request)


@pytest.mark.parametrize("raw", [candidate(score=1.4), candidate(confidence=7), "{oops",
                                 candidate(citations=["made-up"]), candidate(outcome="INCORRECT")])
async def test_the_model_cannot_bypass_validation_in_the_engine(raw) -> None:
    """Even a grader implementation that skips the agent's checks cannot reach the grade: the engine validates."""

    class Raw:
        async def grade(self, request):
            return SemanticGraderResult.model_construct(
                candidate=SemanticGradeCandidate.model_construct(**raw) if isinstance(raw, dict) else None,
                error=None if isinstance(raw, dict) else "unparseable", usage=GraderUsage(llm_calls=1))

    g = await grade(free_item(), two_criteria(), "Porque terminó ayer.", grader=Raw())
    assert g.outcome == Out.UNCERTAIN and g.uncertainty_reason and g.feedback.next_hint == RETRY_HINT
    assert g.score == 0.0 and not g.misconceptions


# --- the pipeline -------------------------------------------------------------------------------------------------


async def test_exact_match_is_tried_first_and_calls_no_model() -> None:
    grader = FakeGrader(candidate())
    g = await grade(short_item(), None, "Hablé", grader=grader)
    assert (g.outcome, g.grader_type, g.score, g.matched) == (Out.CORRECT, GraderType.EXACT, 1.0, "hablé")
    g = await grade(free_item(acceptable_answers=["Because the action is finished."]), two_criteria(),
                    "because the action is finished", grader=grader)
    assert g.outcome == Out.CORRECT and g.grader_type == GraderType.EXACT
    assert all(c.score == 1.0 for c in g.criterion_results)
    assert grader.requests == [] and g.grader.llm_calls == 0


async def test_closed_questions_never_reach_the_semantic_grader() -> None:
    grader = FakeGrader(candidate())
    g = await grade(short_item(), None, "hablo", grader=grader)
    assert (g.outcome, g.grader_type) == (Out.INCORRECT, GraderType.EXACT)
    g = await grade(short_item(), None, "hablaba", grader=grader)
    assert (g.outcome, g.grader_type) == (Out.INCORRECT, GraderType.RULE)
    assert [(m.type, m.source, m.confidence) for m in g.misconceptions] == [("imperfect_for_completed", "rule", 1.0)]
    g = await grade(free_item(), two_criteria(), "   ", grader=grader)
    assert (g.outcome, g.grader_type, g.feedback.errors) == (Out.INCORRECT, GraderType.RULE, ["the answer is empty"])
    assert grader.requests == []


async def test_semantic_equivalence_gets_credit() -> None:
    grader = FakeGrader(candidate())
    g = await grade(free_item(), two_criteria(), "Porque la acción ya terminó.", grader=grader)
    assert (g.outcome, g.grader_type, g.score, g.confidence) == (Out.CORRECT, GraderType.SEMANTIC, 1.0, 0.95)
    assert g.feedback.citations == ["lesson:s1"] and g.grader.llm_calls == 1
    assert len(grader.requests) == 1


async def test_partial_credit_and_a_required_criterion() -> None:
    partial = candidate(score=0.4, outcome="PARTIAL", criterion_results=[
        {"criterion_id": "completed", "score": 0.0}, {"criterion_id": "bounded", "score": 1.0}])
    g = await grade(free_item(), two_criteria(), "Porque pasó ayer.", grader=FakeGrader(partial))
    assert (g.outcome, g.score) == (Out.PARTIAL, 0.4)
    assert [c.met for c in g.criterion_results] == [False, True]
    # A rubric where the unmet required criterion weighs little: the score passes but the outcome is capped.
    rubric = two_criteria(criteria=[
        {"criterion_id": "completed", "description": "completed", "weight": 0.2, "required": True},
        {"criterion_id": "bounded", "description": "bounded", "weight": 0.8}])
    capped = candidate(score=0.8, outcome="CORRECT", criterion_results=[
        {"criterion_id": "completed", "score": 0.0}, {"criterion_id": "bounded", "score": 1.0}])
    g = await grade(free_item(), rubric, "Porque pasó ayer.", grader=FakeGrader(capped))
    assert (g.outcome, g.score) == (Out.PARTIAL, 0.75) and g.notes == ["a required criterion is not met"]


async def test_the_outcome_and_score_are_recomputed_not_taken_from_the_model() -> None:
    # The model says 0.9 and CORRECT; its own criteria give 1.0 x 0.6 + 0.75 x 0.4 = 0.9 -> CORRECT at 0.9.
    raw = candidate(score=0.9, outcome="PARTIAL", criterion_results=[
        {"criterion_id": "completed", "score": 1.0}, {"criterion_id": "bounded", "score": 0.75}])
    g = await grade(free_item(), two_criteria(), "Porque terminó.", grader=FakeGrader(raw))
    assert (g.outcome, g.score) == (Out.CORRECT, 0.9)


@pytest.mark.parametrize(("confidence", "outcome"), [(0.95, Out.CORRECT), (0.7, Out.PARTIAL), (0.5, Out.UNCERTAIN)])
async def test_confidence_bands(confidence: float, outcome: AssessmentOutcome) -> None:
    g = await grade(free_item(), two_criteria(), "Porque terminó ayer.",
                    grader=FakeGrader(candidate(confidence=confidence)))
    assert g.outcome == outcome


async def test_an_incorrect_grade_needs_confidence_too() -> None:
    wrong = candidate(score=0.0, outcome="INCORRECT", confidence=0.7, criterion_results=[
        {"criterion_id": "completed", "score": 0.0}, {"criterion_id": "bounded", "score": 0.0}])
    g = await grade(free_item(), two_criteria(), "No sé.", grader=FakeGrader(wrong))
    assert g.outcome == Out.UNCERTAIN  # never failed on a guess


@pytest.mark.parametrize(("grader", "config", "context", "reason"), [
    (None, None, CONTEXT, "not available"),
    (FakeGrader(candidate()), AssessmentConfig(semantic_enabled=False), CONTEXT, "not available"),
    (FakeGrader("OutputRejected: invalid after retries"), None, CONTEXT, "no valid grade"),
    (FakeGrader(candidate(insufficient_context=True)), None, CONTEXT, "context"),
    (FakeGrader(candidate(citations=[])), None, GradingContext(), None),
])
async def test_ungradable_answers_are_uncertain_never_incorrect(grader, config, context, reason) -> None:
    g = await grade(free_item(), two_criteria(), "Porque terminó ayer.", grader=grader, config=config,
                    context=context)
    if reason is None:  # no context but a confident, consistent grade: it stands (and cites nothing)
        assert g.outcome == Out.CORRECT and g.feedback.citations == []
        return
    assert g.outcome == Out.UNCERTAIN and reason in g.uncertainty_reason
    assert g.feedback.next_hint == RETRY_HINT and not g.misconceptions and g.score == 0.0


async def test_misconceptions_only_on_wrong_grades_and_above_the_confidence_floor() -> None:
    mis = [{"concept_id": CONCEPT, "type": "habitual", "description": "treats it as habitual", "confidence": 0.8},
           {"concept_id": CONCEPT, "type": "Habitual ", "description": "duplicate", "confidence": 0.9},
           {"concept_id": CONCEPT, "type": "weak", "description": "a guess", "confidence": 0.3}]
    wrong = candidate(score=0.0, outcome="INCORRECT", misconceptions=mis, criterion_results=[
        {"criterion_id": "completed", "score": 0.0}, {"criterion_id": "bounded", "score": 0.0}])
    g = await grade(free_item(), two_criteria(), "Porque es habitual.", grader=FakeGrader(wrong))
    assert g.outcome == Out.INCORRECT
    assert [(m.type, m.source) for m in g.misconceptions] == [("habitual", "semantic")]
    g = await grade(free_item(), two_criteria(), "Porque terminó ayer.",
                    grader=FakeGrader(candidate(misconceptions=mis)))
    assert g.outcome == Out.CORRECT and g.misconceptions == []


async def test_a_deterministic_rubric_is_scored_by_code() -> None:
    rubric = two_criteria(deterministic=True)
    grader = FakeGrader(candidate())
    g = await grade(free_item(), rubric, "Porque terminó ayer.", grader=grader)
    assert (g.outcome, g.grader_type, g.score, g.confidence) == (Out.CORRECT, GraderType.RUBRIC, 1.0, 1.0)
    g = await grade(free_item(), rubric, "Porque fue ayer.", grader=grader)
    assert (g.outcome, g.score) == (Out.PARTIAL, 0.4) and g.feedback.errors == ["does not meet completed"]
    assert grader.requests == []


async def test_the_grader_gets_the_minimum_context() -> None:
    grader = FakeGrader(candidate())
    many = GradingContext(lesson_context=[ContextPassage(ref=f"lesson:s{i}", kind="lesson", title="t", text="x")
                                          for i in range(10)])
    await grade(free_item(metadata={"learner_note": "private"}), two_criteria(), "Porque terminó ayer.",
                grader=grader, context=many, config=AssessmentConfig(max_context_passages=2))
    sent = grader.requests[0].model_dump()
    assert len(sent["lesson_context"]) == 2
    assert set(sent) == {"language", "item", "rubric", "learner_answer", "lesson_context", "research_evidence"}
    assert "private" not in json.dumps(sent) and "lesson-1" not in json.dumps(sent)


async def test_identical_inputs_give_identical_grades() -> None:
    a = await grade(free_item(), two_criteria(), "Porque terminó ayer.", grader=FakeGrader(candidate()))
    b = await grade(free_item(), two_criteria(), "Porque terminó ayer.", grader=FakeGrader(candidate()))
    assert a == b


def test_grades_are_immutable_and_uncertain_needs_a_reason() -> None:
    g = AssessmentGrade(grade_id="g", assessment_item_id="i", attempt_id="a", learner_answer="x",
                        normalized_answer="x", score=1.0, max_score=1.0, percentage=100.0, outcome=Out.CORRECT,
                        confidence=1.0, grader_type=GraderType.EXACT, feedback={"outcome": "CORRECT"}, created_at=AT)
    with pytest.raises(ValidationError):
        g.score = 0.0
    with pytest.raises(ValidationError):
        AssessmentGrade.model_validate({**g.model_dump(), "outcome": "UNCERTAIN",
                                        "feedback": {"outcome": "UNCERTAIN"}})


# --- the store ----------------------------------------------------------------------------------------------------


def attempt(attempt_id: str, answer: str = "x") -> tuple[AssessmentAttempt, AssessmentGrade]:
    g = AssessmentGrade(grade_id=f"g-{attempt_id}", assessment_item_id="conj", attempt_id=attempt_id,
                        learner_answer=answer, normalized_answer=answer, score=0.0, max_score=1.0, percentage=0.0,
                        outcome=Out.INCORRECT, confidence=1.0, grader_type=GraderType.EXACT,
                        feedback={"outcome": "INCORRECT"}, created_at=AT)
    a = AssessmentAttempt(attempt_id=attempt_id, assessment_item_id="conj", learner_id="l1", attempt_number=1,
                          learner_answer=answer, answer_hash="h", submitted_at=AT, grade_id=g.grade_id, source="api",
                          task_id="t1")
    return a, g


def test_the_store_numbers_attempts_and_never_overwrites() -> None:
    repo = InMemoryAssessmentRepository()
    assert repo.save_item(short_item()) is True and repo.save_item(short_item()) is False
    with pytest.raises(ItemConflict):
        repo.save_item(short_item(expected_answer="hablaste"))
    assert [repo.add_attempt(*attempt(f"a{i}")).attempt_number for i in range(3)] == [1, 2, 3]
    with pytest.raises(AttemptExists):
        repo.add_attempt(*attempt("a1", "other"))
    assert repo.attempt("a1").learner_answer == "x"
    first = repo.complete("a0", AttemptOutcome(evidence_ids=["e1"]), AT)
    second = repo.complete("a0", AttemptOutcome(evidence_ids=["e2"]), AT)
    assert first.outcome.evidence_ids == second.outcome.evidence_ids == ["e1"]
    assert [a.attempt_id for a in repo.attempts("conj", "l1")] == ["a0", "a1", "a2"]
