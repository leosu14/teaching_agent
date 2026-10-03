"""The adaptive pedagogical engine: concept graph, evidence and mastery, learner model, gaps, planning, recommendation.

Everything here is deterministic code: the same learner state, goal, concepts and configuration always give the
same result, and no model is involved.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.learner.history import InMemoryEvidenceRepository
from app.learner.mastery import IntervalReviewScheduler, MasteryUpdater
from app.pedagogy.gaps import KnowledgeGapAnalyzer
from app.pedagogy.graph import ConceptGraph, ConceptGraphError
from app.pedagogy.planner import PedagogicalPlanner, PlanningError
from app.pedagogy.recommend import NextLessonRecommender, evaluation_feedback
from app.pedagogy.strategy import GenericStrategy, PlannedActivity, StrategyRegistry, UnknownStrategy
from app.schemas.concepts import Concept
from app.schemas.learner import (
    ConceptMastery,
    EvidenceConflict,
    LearnerProfileInput,
    LearningEvidence,
    LearningGoal,
    MasteryChange,
)
from app.schemas.pedagogy import (
    AdaptiveQuestioningPolicy,
    DifficultyBands,
    LearningActivity,
    PedagogyConfig,
    PlannerConfig,
    learner_context,
)
from tests.unit.helpers import NOW, service

LEARNER = "learner-1"
DOMAIN = "spanish"
PRETERITE, IMPERFECT, OPINIONS = "es.conv.preterite", "es.conv.imperfect", "es.conv.opinions"
STORY, SUBJUNCTIVE = "es.conv.preterite_imperfect", "es.conv.subjunctive_opinion"
ALL = [OPINIONS, PRETERITE, IMPERFECT, STORY, SUBJUNCTIVE]


def concept(cid: str, *prerequisites: str, domain: str = DOMAIN) -> Concept:
    return Concept(concept_id=cid, name=cid.split(".")[-1].replace("_", " "), domain=domain, level="B1",
                   prerequisites=list(prerequisites), topic="conversation")


def spanish() -> ConceptGraph:
    return ConceptGraph([concept(SUBJUNCTIVE, OPINIONS), concept(STORY, PRETERITE, IMPERFECT), concept(OPINIONS),
                         concept(PRETERITE), concept(IMPERFECT)])


def evidence(cid: str, correctness: str, *, score: float | None = None, difficulty: float = 0.5,
             source: str = "exercise", ref: str = "r", at=NOW, learner: str = LEARNER) -> LearningEvidence:
    if score is None:
        score = {"correct": 1.0, "incorrect": 0.0, "partial": 0.5}[correctness]
    return LearningEvidence(evidence_id=LearningEvidence.id_for(learner, source, ref, cid), learner_id=learner,
                            concept_id=cid, source_type=source, source_ref=ref, correctness=correctness, score=score,
                            difficulty=difficulty, timestamp=at)


def placement(cid: str, mastery: float, at=NOW) -> list[LearningEvidence]:
    """Two practice results, then a calibrated manual placement: mastery == `mastery`, confidence > 0.6."""
    return [evidence(cid, "correct", ref="p1", at=at - timedelta(minutes=2)),
            evidence(cid, "incorrect", ref="p2", at=at - timedelta(minutes=1)),
            evidence(cid, "partial", score=mastery, source="manual", ref="placement", at=at)]


def goal(*targets: str, gid: str = "goal-1") -> LearningGoal:
    return LearningGoal(goal_id=gid, learner_id=LEARNER, domain=DOMAIN, target_level="B1",
                        target_concepts=list(targets or ALL), description="Improve conversational Spanish")


def learner(masteries: dict[str, float], *, graph: ConceptGraph | None = None, session_minutes: int = 30):
    graph = graph or spanish()
    memory, clock = service()
    memory.upsert(LEARNER, LearnerProfileInput.model_validate({
        "display_name": "Ana Example",
        "subjects": [{"subject": DOMAIN, "framework_id": "cefr", "target_level": "B1", "estimated_level": "B1"}],
        "preferences": {"session_minutes": session_minutes}}))
    items = [e for cid, m in masteries.items() for e in placement(cid, m)]
    memory.record_evidence(LEARNER, DOMAIN, items, [c.ref() for c in graph.concepts()])
    return memory, clock, graph


def model_of(memory, graph):
    return memory.learner_model(LEARNER, DOMAIN, "cefr", "B1", graph.ids)


DEMO = {PRETERITE: 0.52, OPINIONS: 0.20, IMPERFECT: 0.78}


# --- concept graph -------------------------------------------------------------------------------------------------


def test_concept_graph_orders_prerequisites_first_and_computes_closure() -> None:
    g = spanish()
    order = g.ids
    for cid in g.ids:
        assert all(order.index(p) < order.index(cid) for p in g.prerequisites(cid))
    assert set(g.closure([STORY])) == {STORY, PRETERITE, IMPERFECT}
    assert g.dependents(OPINIONS) == [SUBJUNCTIVE]
    assert g.order([SUBJUNCTIVE, OPINIONS]) == [OPINIONS, SUBJUNCTIVE]


@pytest.mark.parametrize("concepts", [
    [concept("a", "missing")],
    [concept("a", "a")],
    [concept("a", "b"), concept("b", "a")],
])
def test_concept_graph_rejects_invalid_prerequisites(concepts) -> None:
    with pytest.raises(ConceptGraphError):
        ConceptGraph(concepts)


# --- evidence and mastery ------------------------------------------------------------------------------------------


def test_evidence_is_immutable_and_never_silently_rewritten() -> None:
    e = evidence(PRETERITE, "correct")
    with pytest.raises(ValidationError):
        e.score = 0.0  # frozen
    store = InMemoryEvidenceRepository()
    assert store.add(e) is True
    assert store.add(e) is False  # identical re-record: idempotent
    with pytest.raises(EvidenceConflict):
        store.add(e.model_copy(update={"score": 0.9}))
    with pytest.raises(ValidationError):
        evidence(PRETERITE, "correct", score=0.1)  # score must agree with correctness


def test_mastery_update_is_deterministic_bounded_and_does_not_touch_its_input() -> None:
    updater = MasteryUpdater()
    seed = updater.initial(PRETERITE, "preterite", DOMAIN)
    frozen = seed.model_dump()
    history = [evidence(PRETERITE, c, difficulty=d, ref=f"r{i}", at=NOW + timedelta(hours=i))
               for i, (c, d) in enumerate([("correct", 0.9), ("correct", 0.1), ("incorrect", 0.1), ("correct", 0.5)]
                                          * 10)]
    state = seed
    for e in history:
        nxt = updater.apply(state, e)
        assert 0.0 <= nxt.mastery <= 1.0 and 0.0 <= nxt.confidence <= 1.0
        state = nxt
    assert seed.model_dump() == frozen
    assert updater.replay(seed, history) == state  # rebuilt from evidence alone
    assert updater.replay(seed, list(reversed(history))) == state  # replay orders by time, not arrival
    assert state.evidence_count == len(history) and state.correct_count + state.incorrect_count == len(history)


def test_correct_answers_raise_mastery_and_errors_lower_it() -> None:
    updater = MasteryUpdater()
    seed = updater.initial(PRETERITE, "preterite", DOMAIN)
    up = updater.apply(seed, evidence(PRETERITE, "correct"))
    down = updater.apply(seed, evidence(PRETERITE, "incorrect"))
    assert down.mastery < seed.mastery < up.mastery
    hard = updater.apply(seed, evidence(PRETERITE, "correct", difficulty=0.9))
    easy = updater.apply(seed, evidence(PRETERITE, "correct", difficulty=0.1))
    assert hard.mastery > easy.mastery  # succeeding on a hard item says more


def test_a_manual_placement_sets_mastery_and_exposure_does_not() -> None:
    updater = MasteryUpdater()
    state = updater.initial(OPINIONS, "opinions", DOMAIN)
    for e in placement(OPINIONS, 0.2):
        state = updater.apply(state, e)
    assert state.mastery == pytest.approx(0.2) and state.confidence > 0.6
    exposed = updater.expose(state, NOW + timedelta(days=1))
    assert exposed.mastery == state.mastery and exposed.exposures == state.exposures + 1
    assert exposed.next_review_at is not None and exposed.next_review_at > NOW


def test_spaced_review_schedule() -> None:
    scheduler = IntervalReviewScheduler()
    updater = MasteryUpdater(scheduler=scheduler)
    strong = ConceptMastery(concept_id=IMPERFECT, name="imperfect", subject=DOMAIN, mastery=0.9, evidence_count=3,
                            last_assessed_at=NOW - timedelta(days=30))
    missed = updater.apply(strong, evidence(IMPERFECT, "incorrect", at=NOW))
    assert missed.next_review_at == NOW + timedelta(days=scheduler.intervals.weak_days)
    retained = updater.apply(strong, evidence(IMPERFECT, "correct", at=NOW))
    soon = updater.apply(strong.model_copy(update={"last_assessed_at": NOW - timedelta(hours=1)}),
                         evidence(IMPERFECT, "correct", at=NOW))
    assert retained.next_review_at > soon.next_review_at > missed.next_review_at  # retained over a gap: stretched


# --- learner model -------------------------------------------------------------------------------------------------


def test_learner_model_categories_and_rebuild_from_history() -> None:
    memory, _, graph = learner(DEMO)
    model = model_of(memory, graph)
    assert model.learner_id == LEARNER and model.domain == DOMAIN
    assert model.mastery_of(PRETERITE) == pytest.approx(0.52)
    assert model.mastery_of(OPINIONS) == pytest.approx(0.20)
    assert model.mastery_of(IMPERFECT) == pytest.approx(0.78)
    assert set(model.unknown_concepts) == {STORY, SUBJUNCTIVE}
    assert model.weak_concepts == [OPINIONS] and set(model.developing_concepts) == {PRETERITE, IMPERFECT}
    assert model.mastered_concepts == []
    assert model.evidence_count == 9
    rebuilt = memory.rebuild(LEARNER, DOMAIN)
    assert {cid: rebuilt[cid] for cid in DEMO} == {cid: model.state(cid) for cid in DEMO}


def test_learner_context_for_providers_carries_no_identity() -> None:
    memory, _, graph = learner(DEMO)
    model = model_of(memory, graph)
    plan = PedagogicalPlanner().plan(model, KnowledgeGapAnalyzer().analyze(model, graph, goal()), goal(), graph)
    context = learner_context(model, plan.brief(), ["A1", "A2", "B1", "B2"]).model_dump_json()
    assert LEARNER not in context and "Ana Example" not in context
    assert LEARNER not in plan.brief().model_dump_json()  # the brief the teacher receives is anonymous too


# --- knowledge gaps ------------------------------------------------------------------------------------------------


def test_gaps_are_prioritised_deterministically_and_respect_prerequisites() -> None:
    memory, _, graph = learner(DEMO)
    model = model_of(memory, graph)
    analyzer = KnowledgeGapAnalyzer()
    gaps = analyzer.analyze(model, graph, goal())
    assert gaps == analyzer.analyze(model, graph, goal())  # same input, same output (and id)
    priorities = [g.priority for g in gaps.gaps]
    assert priorities == sorted(priorities, reverse=True)
    by_id = {g.concept.concept_id: g for g in gaps.gaps}
    assert by_id[SUBJUNCTIVE].recommended_action == "prerequisite_first"
    assert by_id[SUBJUNCTIVE].unmet_prerequisites == [OPINIONS]
    assert by_id[STORY].unmet_prerequisites == [PRETERITE]  # imperfect (0.78) is secure, preterite (0.52) is not
    assert by_id[IMPERFECT].recommended_action == "reinforce"
    assert all(g.learner_id == LEARNER for g in [gaps]) and gaps.goal_id == "goal-1"


def test_mastered_concepts_are_not_gaps() -> None:
    memory, _, graph = learner({**DEMO, IMPERFECT: 0.9})
    gaps = KnowledgeGapAnalyzer().analyze(model_of(memory, graph), graph, goal())
    assert IMPERFECT in gaps.mastered and gaps.gap(IMPERFECT) is None


def test_configurable_difficulty_bands() -> None:
    assert DifficultyBands().band_for(0.25) == "foundational"
    custom = DifficultyBands(guided=0.2, independent=0.4, consolidation=0.7)
    assert custom.band_for(0.25) == "guided" and custom.band_for(0.75) == "consolidation"
    with pytest.raises(ValidationError):
        DifficultyBands(guided=0.6, independent=0.3, consolidation=0.8)
    memory, _, graph = learner(DEMO)
    model = model_of(memory, graph)
    default = KnowledgeGapAnalyzer().analyze(model, graph, goal()).gap(OPINIONS)
    shifted = KnowledgeGapAnalyzer(PedagogyConfig(bands=custom)).analyze(model, graph, goal()).gap(OPINIONS)
    assert (default.band, shifted.band) == ("foundational", "guided")


# --- pedagogical planning ------------------------------------------------------------------------------------------


def plan_for(masteries, *, minutes=None, config=None, goal_=None, history=None):
    memory, clock, graph = learner(masteries)
    model = model_of(memory, graph)
    g = goal_ or goal()
    gaps = KnowledgeGapAnalyzer(config).analyze(model, graph, g)
    return PedagogicalPlanner(config).plan(model, gaps, g, graph, minutes, history), model, graph


def test_plan_invariants() -> None:
    plan, model, graph = plan_for(DEMO)
    concepts = set(plan.concept_ids())
    assert concepts <= set(graph.ids)
    assert plan.learner_id == model.learner_id
    assert all(o.concept_id in concepts for o in plan.lesson_objectives)
    assert all(set(a.concept_ids) <= concepts and a.estimated_minutes > 0 for a in plan.activities)
    assert plan.estimated_duration == sum(s.minutes for s in plan.sequencing) <= plan.available_minutes
    order = plan.concept_ids()
    for cid in order:
        assert all(order.index(p) < order.index(cid) for p in graph.prerequisites(cid) if p in order)
    # Storytelling is taught with a review of the preterite (0.52: guided, secure enough to review in passing);
    # the subjunctive waits until opinions (0.20: foundational) are secure, so opinions are taught first.
    assert plan.target_concepts == [OPINIONS, STORY] and plan.prerequisite_concepts == [PRETERITE]
    assert "Deferred until prerequisites are secure: subjunctive opinion (needs opinions)" in plan.rationale
    assert plan == plan_for(DEMO)[0]  # deterministic, including the plan id


def test_secure_enough_prerequisites_are_reviewed_in_the_same_lesson() -> None:
    plan, _, _ = plan_for({PRETERITE: 0.5, IMPERFECT: 0.9, OPINIONS: 0.95, SUBJUNCTIVE: 0.9},
                          goal_=goal(STORY, IMPERFECT, PRETERITE))
    assert plan.target_concepts == [STORY] and plan.prerequisite_concepts == [PRETERITE]
    assert plan.treatment(PRETERITE).mode == "review"
    assert [s.phase for s in plan.sequencing] == ["prerequisite_review", "instruction", "practice", "assessment"]


def test_mastered_concepts_are_only_reviewed_when_due() -> None:
    masteries = {**DEMO, IMPERFECT: 0.95}
    plan, model, _ = plan_for(masteries)
    assert IMPERFECT not in plan.target_concepts and IMPERFECT not in plan.review_concepts  # not due yet
    memory, clock, graph = learner(masteries)
    clock.now = NOW + timedelta(days=60)  # well past every review date
    model = model_of(memory, graph)
    assert IMPERFECT in model.due_for_review
    gaps = KnowledgeGapAnalyzer().analyze(model, graph, goal())
    plan = PedagogicalPlanner().plan(model, gaps, goal(), graph)
    assert IMPERFECT in plan.review_concepts and IMPERFECT not in plan.target_concepts
    assert plan.treatment(IMPERFECT).mode == "review"


def test_time_budget_shapes_the_plan_and_impossible_plans_are_refused() -> None:
    short, _, _ = plan_for(DEMO, minutes=12)
    assert len(short.target_concepts) == 1 and short.estimated_duration <= 12
    with pytest.raises(PlanningError):
        plan_for(DEMO, minutes=3)
    one, _, _ = plan_for(DEMO, config=PedagogyConfig(planner=PlannerConfig(max_target_concepts=1)))
    assert len(one.target_concepts) == 1


def test_plan_brief_validator_rejects_impossible_plans() -> None:
    plan, _, _ = plan_for(DEMO)
    data = plan.model_dump()
    with pytest.raises(ValidationError):
        type(plan).model_validate({**data, "available_minutes": plan.estimated_duration - 1})
    bad_activity = {**data["activities"][0], "concept_ids": ["es.unknown"]}
    with pytest.raises(ValidationError):
        type(plan).model_validate({**data, "activities": [bad_activity, *data["activities"][1:]]})
    with pytest.raises(ValidationError):
        type(plan).model_validate({**data, "target_concepts": [*plan.target_concepts, SUBJUNCTIVE]})
    with pytest.raises(ValidationError):
        LearningActivity.model_validate({**data["activities"][0], "estimated_minutes": -1})


def test_reinforce_rather_than_reintroduce_what_was_taught() -> None:
    from app.schemas.learner import LearningEvent
    taught = [LearningEvent.create(LEARNER, "lesson_completed", NOW, key="t1", subject=DOMAIN,
                                   concept_ids=[OPINIONS])]
    plan, _, _ = plan_for(DEMO, history=taught)
    assert plan.treatment(OPINIONS).mode == "reinforce"
    fresh, _, _ = plan_for(DEMO, history=[])
    assert fresh.treatment(OPINIONS).mode == "introduce"


def test_activity_types_are_extensible_and_strategies_pluggable() -> None:
    LearningActivity(activity_id="a1", type="role_play", concept_ids=[OPINIONS], difficulty="guided",
                     estimated_minutes=4, instructions="Debate a film.", expected_response="an opinion")
    with pytest.raises(ValidationError):
        LearningActivity(activity_id="a1", type="Role Play!", concept_ids=[OPINIONS], difficulty="guided",
                         estimated_minutes=4, instructions="x", expected_response="y")

    class RolePlay(GenericStrategy):
        strategy_id = "role_play"

        def practice_type(self, band):
            return "role_play"

    registry = StrategyRegistry()
    registry.register(RolePlay())
    with pytest.raises(UnknownStrategy):
        registry.get("nope")
    config = PedagogyConfig(strategies={DOMAIN: "role_play"})
    memory, _, graph = learner(DEMO)
    model = model_of(memory, graph)
    gaps = KnowledgeGapAnalyzer(config).analyze(model, graph, goal())
    plan = PedagogicalPlanner(config, registry).plan(model, gaps, goal(), graph)
    assert plan.strategy_id == "role_play" and "role_play" in {a.type for a in plan.activities}
    assert isinstance(PlannedActivity("practice", plan.activities[0]), PlannedActivity)


def test_the_engine_is_subject_independent() -> None:
    math = ConceptGraph([concept("math.fractions", domain="math"),
                         concept("math.ratios", "math.fractions", domain="math")])
    memory, _ = service()
    memory.record_evidence(LEARNER, "math", placement("math.fractions", 0.4), [c.ref() for c in math.concepts()])
    model = memory.learner_model(LEARNER, "math", "mastery", None, math.ids)
    g = LearningGoal(goal_id="g-math", learner_id=LEARNER, domain="math", target_concepts=["math.ratios"])
    rec = NextLessonRecommender().recommend(model, math, g)
    # Fractions (0.4) are below the prerequisite threshold but secure enough to review before ratios.
    assert rec.recommended_concepts == ["math.ratios"] and rec.prerequisite_review == ["math.fractions"]
    assert not rec.goal_achieved


# --- adaptive questioning ------------------------------------------------------------------------------------------


def test_adaptive_questioning_policy() -> None:
    policy = AdaptiveQuestioningPolicy(max_questions=3, max_follow_ups_per_concept=1)
    assert policy.first_round(ALL, confident={PRETERITE, OPINIONS}) == [IMPERFECT, STORY, SUBJUNCTIVE]
    assert AdaptiveQuestioningPolicy(max_questions=2).first_round(ALL, set()) == ALL[:2]
    asked = {STORY: 1, SUBJUNCTIVE: 1}
    assert policy.follow_ups(asked, [STORY, SUBJUNCTIVE], total_asked=2) == [STORY]  # question budget
    assert policy.follow_ups({STORY: 2}, [STORY], total_asked=2) == []  # follow-up budget spent
    assert policy.should_stop({STORY: 1}, [], total_asked=1)  # nothing missed: stop


# --- recommendation and feedback -----------------------------------------------------------------------------------


def test_the_recommendation_follows_the_mastery_state() -> None:
    recommender = NextLessonRecommender()
    memory, _, graph = learner(DEMO)
    first = recommender.recommend(model_of(memory, graph), graph, goal())
    later = [e for cid, m in {OPINIONS: 0.7, PRETERITE: 0.85}.items() for e in placement(cid, m, NOW + timedelta(1))]
    later = [e.model_copy(update={"evidence_id": e.evidence_id + "-2", "source_ref": e.source_ref + "-2"})
             for e in later]
    memory.record_evidence(LEARNER, DOMAIN, later, [c.ref() for c in graph.concepts()])
    second = recommender.recommend(model_of(memory, graph), graph, goal())
    assert first.recommended_concepts != second.recommended_concepts
    assert OPINIONS in first.recommended_concepts and SUBJUNCTIVE in second.recommended_concepts
    done = {cid: 0.95 for cid in ALL}
    memory2, _, _ = learner(done)
    finished = recommender.recommend(model_of(memory2, graph), graph, goal())
    assert finished.goal_achieved and finished.recommended_concepts == []


def test_evaluation_feedback_is_read_from_state() -> None:
    memory, _, graph = learner({**DEMO, IMPERFECT: 0.9})
    model = model_of(memory, graph)
    rec = NextLessonRecommender().recommend(model, graph, goal())
    changes = [MasteryChange(concept_id=OPINIONS, before=0.1, after=0.2, reason="evaluation")]
    fb = evaluation_feedback(changes, [OPINIONS, PRETERITE, IMPERFECT], model, rec, PedagogyConfig())
    assert fb.mastered == [IMPERFECT] and fb.still_weak == [OPINIONS, PRETERITE] and fb.changes == changes
    assert fb.review_next == list(dict.fromkeys([*rec.prerequisite_review, *rec.recommended_concepts,
                                                 *rec.review_concepts]))
