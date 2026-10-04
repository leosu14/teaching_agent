"""The curriculum engine: goals, planning, validation, versioning, priority, next learning action, review, progress,
completion and replanning. Deterministic code only; the model's part (wording) is checked at its boundary."""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from app.agents.base import OutputRejected
from app.agents.curriculum.agent import LearningPathPlannerAgent
from app.curriculum.engine import CurriculumConflict, CurriculumEngine
from app.curriculum.next_action import GoalCurriculum, NextActionEngine
from app.curriculum.planner import CurriculumPlanner, targets_for_level
from app.curriculum.priority import priority_factors, priority_score
from app.curriculum.progress import compute_progress
from app.curriculum.repository import InMemoryCurriculumRepository
from app.curriculum.review import MasteryReviewPolicy
from app.curriculum.tracker import CurriculumTracker
from app.curriculum.validation import (
    CurriculumValidationError,
    curriculum_problems,
    find_cycle,
    review_proposal,
    version_problems,
)
from app.pedagogy.gaps import KnowledgeGapAnalyzer
from app.pedagogy.graph import ConceptGraph
from app.pedagogy.planner import PedagogicalPlanner
from app.providers.llm.base import LLMMessage, LLMRequest
from app.providers.llm.mock_responders import plan_learning_path
from app.schemas.concepts import Concept
from app.schemas.curriculum import (
    CurriculumConfig,
    CurriculumProposal,
    CurriculumVersion,
    GoalInput,
    GoalUpdate,
    LearningActionType,
    NextLearningAction,
    ObjectiveStatus,
    ProposedObjective,
    ReplanReason,
    VersionConflict,
    curriculum_id_for,
)
from app.schemas.events import EventType
from app.schemas.learner import GoalStatus, LearnerProfileInput, LearningEvidence, LearningGoal
from app.schemas.pedagogy import LessonFocus
from tests.unit.helpers import NOW, scope, service

LEARNER = "learner-1"
DOMAIN = "spanish"
A, B, C, D, E = "es.present", "es.preterite", "es.past_contrast", "es.subjunctive", "es.argument"
LEVELS = {A: "A2", B: "B1", C: "B1", D: "B1", E: "B2"}


def concept(cid: str, *prerequisites: str) -> Concept:
    return Concept(concept_id=cid, name=cid.split(".")[-1].replace("_", " "), domain=DOMAIN, level=LEVELS.get(cid),
                   prerequisites=list(prerequisites), topic=cid)


def graph() -> ConceptGraph:
    """A <- B <- {C, D} <- E"""
    return ConceptGraph([concept(E, C, D), concept(D, B), concept(C, B), concept(B, A), concept(A)])


def evidence(cid: str, correctness: str, *, score: float | None = None, source: str = "exercise", ref: str = "r",
             at=NOW, difficulty: float = 0.5) -> LearningEvidence:
    if score is None:
        score = {"correct": 1.0, "incorrect": 0.0, "partial": 0.5}[correctness]
    return LearningEvidence(evidence_id=LearningEvidence.id_for(LEARNER, source, ref, cid), learner_id=LEARNER,
                            concept_id=cid, source_type=source, source_ref=ref, correctness=correctness, score=score,
                            difficulty=difficulty, timestamp=at)


def placement(cid: str, mastery: float, at=NOW) -> list[LearningEvidence]:
    """Two practice results, then a calibrated placement: mastery == `mastery`, 3 pieces of evidence."""
    return [evidence(cid, "correct", ref="p1", at=at - timedelta(minutes=2)),
            evidence(cid, "incorrect", ref="p2", at=at - timedelta(minutes=1)),
            evidence(cid, "partial", score=mastery, source="manual", ref="placement", at=at)]


def learner(masteries: dict[str, float], g: ConceptGraph | None = None):
    g = g or graph()
    memory, clock = service()
    memory.upsert(LEARNER, LearnerProfileInput.model_validate({
        "subjects": [{"subject": DOMAIN, "framework_id": "cefr", "target_level": "B1"}]}))
    items = [e for cid, m in masteries.items() for e in placement(cid, m)]
    if items:
        memory.record_evidence(LEARNER, DOMAIN, items, [c.ref() for c in g.concepts()])
    return memory, clock, g


def model_of(memory, g, as_of=NOW):
    return memory.learner_model(LEARNER, DOMAIN, "cefr", "B1", g.ids, as_of=as_of)


def goal(*targets: str, gid: str = "goal-1", priority: int = 3, target_date=None, domain: str = DOMAIN,
         status: GoalStatus = GoalStatus.ACTIVE) -> LearningGoal:
    return LearningGoal(goal_id=gid, learner_id=LEARNER, title="Reach B1 Spanish", domain=domain, target_level="B1",
                        target_concepts=list(targets or (A, B, C, D)), priority=priority, target_date=target_date,
                        status=status)


SPEC = {A: 0.90, B: 0.35, C: 0.10, D: 0.20}


def committed(engine: CurriculumEngine, g_: LearningGoal, model, g: ConceptGraph, proposal=None):
    planner = engine.planner
    draft = planner.draft(g_, g, model, as_of=NOW)
    plan = planner.finalize(draft, g, proposal=proposal, current=engine.current(draft.curriculum_id))
    version, _ = engine.commit(plan, domain=g_.domain, task_id=None, artifact_ids={})
    return version


def entry(engine: CurriculumEngine, g_: LearningGoal, model, as_of=NOW) -> GoalCurriculum:
    version = engine.current(curriculum_id_for(g_.goal_id))
    assert version is not None
    return GoalCurriculum(goal=g_, version=version, progress=engine.progress(version, model, as_of))


# --- goals ---------------------------------------------------------------------------------------------------------


def test_goal_schema_statuses_legacy_values_and_inputs() -> None:
    g = LearningGoal.model_validate({"goal_id": "g", "learner_id": "l", "domain": "maths", "target_concepts": ["x"],
                                     "status": "achieved", "deadline": "2026-12-01T00:00:00Z"})
    assert g.status == GoalStatus.COMPLETED and g.target_date is not None and not g.is_active
    assert LearningGoal.model_validate({"goal_id": "g", "learner_id": "l", "domain": "d", "target_concepts": ["x"],
                                        "status": "abandoned"}).status == GoalStatus.CANCELLED
    with pytest.raises(ValidationError):
        GoalInput(title="t", domain="d")  # neither a level nor concepts
    with pytest.raises(ValidationError):
        GoalUpdate(status=GoalStatus.COMPLETED)  # only the completion rule completes a goal
    with pytest.raises(ValidationError):
        LearningGoal(goal_id="g", learner_id="l", domain="d", target_concepts=["x", "x"])


def test_level_targets_come_from_the_knowledge_base_and_framework() -> None:
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    assert targets_for_level(graph().concepts(), "B1", levels) == [A, B, C, D]
    assert targets_for_level(graph().concepts(), "B2", levels) == [A, B, C, D, E]
    with pytest.raises(CurriculumValidationError):
        targets_for_level(graph().concepts(), "Z9", levels)


# --- planning and validation ---------------------------------------------------------------------------------------


def test_draft_scope_order_modes_and_stable_ids() -> None:
    memory, _, g = learner(SPEC)
    planner = CurriculumPlanner()
    draft = planner.draft(goal(C), g, model_of(memory, g), as_of=NOW)
    assert [o.concept_id for o in draft.objectives] == [A, B, C]  # the goal's closure, prerequisites first
    by = {o.concept_id: o for o in draft.objectives}
    assert by[A].mode == "maintain" and by[A].status == ObjectiveStatus.MASTERED  # 0.90 with 3 evidence
    assert by[B].mode == "learn" and by[B].role == "prerequisite" and by[C].role == "target"
    assert by[C].status == ObjectiveStatus.BLOCKED and by[C].prerequisites == [B]
    again = planner.draft(goal(C), g, model_of(memory, g), as_of=NOW)
    assert again == draft
    assert {o.objective_id for o in draft.objectives} == {o.objective_id for o in again.objectives}
    # The brief a model sees carries no learner or goal identifiers and no numbers.
    brief = draft.brief.model_dump_json()
    assert LEARNER not in brief and "goal-1" not in brief and "0.35" not in brief


def test_draft_rejects_unknown_concepts_and_foreign_goals() -> None:
    memory, _, g = learner(SPEC)
    with pytest.raises(CurriculumValidationError, match="does not know"):
        CurriculumPlanner().draft(goal("es.unknown"), g, model_of(memory, g), as_of=NOW)
    other = goal(C).model_copy(update={"learner_id": "someone-else"})
    with pytest.raises(CurriculumValidationError, match="does not belong"):
        CurriculumPlanner().draft(other, g, model_of(memory, g), as_of=NOW)


def test_validation_catches_cycles_wrong_prerequisites_targets_and_order() -> None:
    memory, _, g = learner(SPEC)
    cfg = CurriculumConfig()
    draft = CurriculumPlanner(cfg).draft(goal(C), g, model_of(memory, g), as_of=NOW)
    objectives = draft.objectives
    assert find_cycle({"x": ["y"], "y": ["x"]}) is not None and find_cycle({"x": ["y"], "y": []}) is None

    def problems(objs, goal_=goal(C), learner_id=LEARNER):
        return curriculum_problems(learner_id=learner_id, goal=goal_, objectives=objs, graph=g, config=cfg)

    assert problems(objectives) == []
    cyclic = [objectives[0].model_copy(update={"prerequisites": [C]}), *objectives[1:]]
    assert any("circular" in p for p in problems(cyclic))
    invented = [*objectives[:2], objectives[2].model_copy(update={"prerequisites": [A]})]  # not the KB's
    assert any("not the knowledge base's" in p for p in problems(invented))
    assert any("without an objective" in p for p in problems(objectives[:2]))
    reordered = [objectives[2].model_copy(update={"order": 1}), objectives[0].model_copy(update={"order": 2}),
                 objectives[1].model_copy(update={"order": 3})]
    assert problems(reordered)  # a concept before its prerequisites
    assert any("between" in p for p in problems([*objectives[:2],
                                                 objectives[2].model_copy(update={"target_mastery": 0.3})]))
    assert any("belong" in p for p in problems(objectives, learner_id="someone-else"))
    foreign = [*objectives[:2], objectives[2].model_copy(update={"goal_id": "goal-2"})]
    assert problems(foreign)


def test_model_proposal_is_validated_unknown_concepts_rejected_order_ignored() -> None:
    memory, _, g = learner(SPEC)
    planner = CurriculumPlanner()
    draft = planner.draft(goal(C), g, model_of(memory, g), as_of=NOW)
    proposal = CurriculumProposal(
        objectives=[ProposedObjective(concept_id=C, description="Tell stories with both past tenses"),
                    ProposedObjective(concept_id="es.invented", description="Something the KB does not have"),
                    ProposedObjective(concept_id=E, description="Outside the goal's scope")],
        suggested_order=[C, B, A], explanation="Stories first.")
    plan = planner.finalize(draft, g, proposal=proposal)
    assert plan.proposal_review.unresolved == ["es.invented"]
    assert E in plan.proposal_review.rejected
    assert not plan.proposal_review.ordering_accepted
    assert [o.concept_id for o in plan.objectives] == [A, B, C]  # the structure is never the model's
    assert next(o for o in plan.objectives if o.concept_id == C).description == "Tell stories with both past tenses"
    assert {w.code for w in plan.warnings} >= {"unresolved_concept", "rejected_concept"}
    descriptions, review, _ = review_proposal(None, draft.objectives, set(g.ids))
    assert descriptions == {} and review.missing == [A, B, C]


@pytest.mark.parametrize("field", ["mastery", "current_mastery", "status", "goal_complete", "prerequisites"])
def test_llm_cannot_set_mastery_status_completion_or_prerequisites(field: str) -> None:
    body = {"objectives": [{"concept_id": C, "description": "x", field: 1.0}], "explanation": "e"}
    with pytest.raises(ValidationError):
        CurriculumProposal.model_validate(body)
    with pytest.raises(ValidationError):
        CurriculumProposal.model_validate({**body, "objectives": [{"concept_id": C, "description": "x"}],
                                           field: True})


def test_learning_path_agent_rejects_concepts_outside_the_brief() -> None:
    memory, _, g = learner(SPEC)
    brief = CurriculumPlanner().draft(goal(C), g, model_of(memory, g), as_of=NOW).brief
    agent = LearningPathPlannerAgent()
    with pytest.raises(OutputRejected):
        agent.check(CurriculumProposal(objectives=[ProposedObjective(concept_id="es.invented", description="x")],
                                       explanation="e"), brief)
    payload = plan_learning_path(LLMRequest(agent_id="learning_path_planner", model="m", system="",
                                            messages=[LLMMessage(role="user", content="word it")],
                                            max_output_tokens=100, input_payload=brief.model_dump(mode="json")))
    agent.check(CurriculumProposal.model_validate(payload), brief)  # the mock's wording passes


def test_identical_inputs_give_identical_versions_and_wording_is_not_material() -> None:
    memory, _, g = learner(SPEC)
    planner = CurriculumPlanner()
    model = model_of(memory, g)
    a = planner.finalize(planner.draft(goal(C), g, model, as_of=NOW), g)
    b = planner.finalize(planner.draft(goal(C), g, model, as_of=NOW + timedelta(hours=3)), g)
    assert a.version_id == b.version_id and a.content_hash == b.content_hash and a.objectives == b.objectives
    worded = planner.finalize(planner.draft(goal(C), g, model, as_of=NOW), g, proposal=CurriculumProposal(
        objectives=[ProposedObjective(concept_id=C, description="Different wording")], explanation="e"))
    assert worded.content_hash == a.content_hash  # wording alone is never a new version


# --- versioning ----------------------------------------------------------------------------------------------------


def test_commit_is_idempotent_versions_are_immutable_and_conflicts_detected() -> None:
    memory, _, g = learner(SPEC)
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    s, events = scope()
    planner = engine.planner
    plan = planner.finalize(planner.draft(goal(C), g, model, as_of=NOW), g)
    v1, created = engine.commit(plan, domain=DOMAIN, task_id="t1", artifact_ids={}, scope=s)
    again, created_again = engine.commit(plan, domain=DOMAIN, task_id="t1", artifact_ids={}, scope=s)
    assert created and not created_again and again == v1 and len(engine.versions(plan.curriculum_id)) == 1
    assert [e.type for e in events].count(EventType.CURRICULUM_CREATED) == 1
    same = planner.finalize(planner.draft(goal(C), g, model, as_of=NOW), g, current=v1)
    assert not same.changed and same.version_id == v1.version_id
    # A material change (the goal now targets D too) is a new version; version 1 stays as it was.
    changed = planner.finalize(planner.draft(goal(C, D), g, model, as_of=NOW), g, current=v1,
                               reasons=[ReplanReason.GOAL_CHANGED])
    v2, _ = engine.commit(changed, domain=DOMAIN, task_id="t2", artifact_ids={}, scope=s)
    assert (v2.version, v2.parent_version) == (2, 1) and v2.version_id != v1.version_id
    assert engine.repository.version(v1.version_id) == v1
    assert version_problems(engine.versions(plan.curriculum_id)) == []
    assert EventType.CURRICULUM_REPLANNED in [e.type for e in events]
    with pytest.raises(CurriculumConflict):  # a plan made on version 1 cannot overwrite version 2
        engine.commit(planner.finalize(planner.draft(goal(B, C), g, model, as_of=NOW), g, current=v1),
                      domain=DOMAIN, task_id="t3", artifact_ids={})
    tampered = v1.model_copy(update={"rationale": "rewritten history"})
    with pytest.raises(VersionConflict):
        engine.repository.add_version(tampered)
    with pytest.raises(ValidationError):  # a version whose content does not match its hash cannot exist
        CurriculumVersion.model_validate({**v1.model_dump(), "objectives": [o.model_dump() for o in v2.objectives]})


def test_sql_repository_round_trip(tmp_path) -> None:
    from app.storage.db import create_db
    from app.storage.repositories import SqlCurriculumRepository

    memory, _, g = learner(SPEC)
    repo = SqlCurriculumRepository(create_db(f"sqlite:///{tmp_path / 'c.db'}"))
    engine = CurriculumEngine(repo)
    v1 = committed(engine, goal(C), model_of(memory, g), g)
    assert repo.version(v1.version_id) == v1 and repo.versions(v1.curriculum_id) == [v1]
    assert repo.record(v1.curriculum_id).current_version_id == v1.version_id
    assert repo.add_version(v1) is False
    with pytest.raises(VersionConflict):
        repo.add_version(v1.model_copy(update={"rationale": "changed"}))
    progress = engine.progress(v1, model_of(memory, g), NOW)
    repo.save_progress(progress)
    assert repo.progress(v1.curriculum_id) == progress
    assert [r.curriculum_id for r in repo.records_for_learner(LEARNER)] == [v1.curriculum_id]


# --- next learning action ------------------------------------------------------------------------------------------


def test_spec_example_learn_the_weakest_ready_objective() -> None:
    """past 0.92, contrastive 0.61, subjunctive 0.21 -> LEARN subjunctive (deterministically)."""
    g = ConceptGraph([concept(A), concept(C, A), concept(D, A)])
    memory, _, g = learner({A: 0.92, C: 0.61, D: 0.21}, g)
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    committed(engine, goal(A, C, D), model, g)
    actions = NextActionEngine()
    first = actions.select(LEARNER, [entry(engine, goal(A, C, D), model)], {DOMAIN: model}, NOW)
    assert (first.action, first.concept_id) == (LearningActionType.LEARN, D)
    assert first.priority is not None and first.priority.score > 0 and "LEARN" in first.reason
    by = {c.concept_id: c for c in first.candidates}
    assert by[C].action == LearningActionType.PRACTICE  # 0.61: practice, not learn
    again = actions.select(LEARNER, [entry(engine, goal(A, C, D), model)], {DOMAIN: model}, NOW)
    assert again == first


def test_blocked_objectives_are_never_selected() -> None:
    memory, _, g = learner(SPEC)
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    committed(engine, goal(), model, g)
    action = NextActionEngine().select(LEARNER, [entry(engine, goal(), model)], {DOMAIN: model}, NOW)
    assert (action.action, action.concept_id) == (LearningActionType.LEARN, B)
    blocked = [c for c in action.candidates if not c.eligible]
    assert {c.concept_id for c in blocked} == {C, D} and all("blocked" in c.reason for c in blocked)


def test_evaluate_practice_review_wait_and_complete() -> None:
    memory, clock, g = learner({A: 0.95, B: 0.85})
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    committed(engine, goal(A, B), model, g)
    actions = NextActionEngine()
    # Both mastered with enough evidence: the goal is complete.
    done = actions.select(LEARNER, [entry(engine, goal(A, B), model)], {DOMAIN: model}, NOW)
    assert done.action == LearningActionType.COMPLETE and done.goal_id == "goal-1"
    # A completed goal later contributes only reviews, once they are due; before that, WAIT with the next date.
    completed = goal(A, B, status=GoalStatus.COMPLETED)
    wait = actions.select(LEARNER, [entry(engine, completed, model)], {DOMAIN: model}, NOW)
    assert wait.action == LearningActionType.WAIT and wait.next_review_at is not None
    later = wait.next_review_at + timedelta(days=1)
    late_model = model_of(memory, g, as_of=later)
    review = actions.select(LEARNER, [entry(engine, completed, late_model, later)], {DOMAIN: late_model}, later)
    assert review.action == LearningActionType.REVIEW
    # Mastery at target but too little evidence: EVALUATE. Between practice_from and target: PRACTICE.
    cfg = CurriculumConfig(evidence_required=5)
    strict = CurriculumEngine(InMemoryCurriculumRepository(), cfg)
    committed(strict, goal(A, B), model, g)
    evaluate = NextActionEngine(cfg).select(LEARNER, [entry(strict, goal(A, B), model)], {DOMAIN: model}, NOW)
    assert evaluate.action == LearningActionType.EVALUATE
    memory2, _, g2 = learner({A: 0.95, B: 0.7})
    model2 = model_of(memory2, g2)
    e2 = CurriculumEngine(InMemoryCurriculumRepository())
    committed(e2, goal(A, B), model2, g2)
    practice = NextActionEngine().select(LEARNER, [entry(e2, goal(A, B), model2)], {DOMAIN: model2}, NOW)
    assert (practice.action, practice.concept_id) == (LearningActionType.PRACTICE, B)
    # Paused and cancelled goals contribute nothing; no curricula at all: WAIT.
    paused = goal(A, B, status=GoalStatus.PAUSED)
    assert actions.select(LEARNER, [entry(e2, paused, model2)], {DOMAIN: model2}, NOW).action == \
        LearningActionType.WAIT
    assert actions.select(LEARNER, [], {}, NOW).action == LearningActionType.WAIT


def test_lesson_actions_need_an_objective() -> None:
    with pytest.raises(ValidationError):
        NextLearningAction(action_id="a", learner_id=LEARNER, action=LearningActionType.LEARN, reason="r", as_of=NOW)
    with pytest.raises(ValidationError):
        NextLearningAction(action_id="a", learner_id=LEARNER, action=LearningActionType.COMPLETE, reason="r",
                           as_of=NOW)


def test_multiple_goals_priority_and_deterministic_ties() -> None:
    memory, _, g = learner(SPEC)
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    first, second = goal(B, gid="goal-a", priority=2), goal(B, gid="goal-b", priority=1)
    for gl in (first, second):
        committed(engine, gl, model, g)
    action = NextActionEngine().select(LEARNER, [entry(engine, first, model), entry(engine, second, model)],
                                      {DOMAIN: model}, NOW)
    assert action.goal_id == "goal-b"  # the higher-priority goal wins
    same = goal(B, gid="goal-a2", priority=1)
    committed(engine, same, model, g)
    tie = NextActionEngine().select(LEARNER, [entry(engine, same, model), entry(engine, second, model)],
                                    {DOMAIN: model}, NOW)
    assert tie.goal_id == "goal-a2"  # equal scores and priorities: by goal id


def test_priority_factors_are_documented_and_bounded() -> None:
    memory, _, g = learner(SPEC)
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    version = committed(engine, goal(), model, g)
    progress = engine.progress(version, model, NOW)
    obj = version.by_concept(B)
    p = progress.of(obj.objective_id)
    cfg = CurriculumConfig()
    far = priority_factors(obj, p, model.state(B), 3, NOW + timedelta(days=200), NOW, cfg)
    near = priority_factors(obj, p, model.state(B), 3, NOW + timedelta(days=5), NOW, cfg)
    passed = priority_factors(obj, p, model.state(B), 3, NOW - timedelta(days=1), NOW, cfg)
    assert far.deadline_pressure == 0 and 0 < near.deadline_pressure < 1 and passed.deadline_pressure == 1
    assert near.deficit == round((0.8 - 0.35) / 0.8, 4)
    assert priority_score(near, True, cfg).score > priority_score(far, True, cfg).score
    assert priority_score(near, False, cfg).score == 0  # prerequisites not ready: never selected
    high = priority_factors(obj, p, model.state(B), 1, None, NOW, cfg)
    low = priority_factors(obj, p, model.state(B), 5, None, NOW, cfg)
    assert high.goal_priority == 1 and low.goal_priority == 0


def test_review_schedule_from_the_learner_model() -> None:
    memory, _, g = learner({A: 0.9})
    state = model_of(memory, g).state(A)
    policy = MasteryReviewPolicy()
    schedule = policy.schedule(A, state, 2, NOW)
    assert schedule.next_review_at == state.next_review_at and schedule.review_count == 2
    assert not schedule.due and policy.schedule(A, state, 0, state.next_review_at).due
    assert policy.schedule(A, None, 0, NOW).next_review_at is None


def test_target_date_feasibility_is_a_warning_never_a_fix() -> None:
    memory, _, g = learner(SPEC)
    planner = CurriculumPlanner()
    model = model_of(memory, g)
    tight = planner.draft(goal(target_date=NOW + timedelta(days=3)), g, model, as_of=NOW)
    passed = planner.draft(goal(target_date=NOW - timedelta(days=1)), g, model, as_of=NOW)
    roomy = planner.draft(goal(target_date=NOW + timedelta(days=365)), g, model, as_of=NOW)
    assert [w.code for w in tight.warnings] == ["deadline_infeasible"]
    assert tight.warnings[0].details["sessions_needed"] > 0
    assert [w.code for w in passed.warnings] == ["deadline_passed"] and roomy.warnings == []
    assert [o.concept_id for o in tight.objectives] == [o.concept_id for o in roomy.objectives]


# --- progress, completion, replanning ------------------------------------------------------------------------------


def test_refresh_reports_transitions_once_and_completes_the_goal_once() -> None:
    memory, clock, g = learner({A: 0.9, B: 0.35})
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    gl = goal(A, B)
    committed(engine, gl, model_of(memory, g), g)
    s, events = scope()
    first = engine.refresh(gl, model_of(memory, g), NOW, scope=s)
    assert first is not None and B in [engine.current(first.curriculum_id).objective(o).concept_id
                                       for o in first.started]
    assert not first.goal_completed and engine.refresh(gl, model_of(memory, g), NOW, scope=s).started == []
    memory.record_evidence(LEARNER, DOMAIN, [evidence(B, "correct", ref=f"x{i}", difficulty=0.9,
                                                      at=NOW + timedelta(minutes=i)) for i in range(6)],
                           [c.ref() for c in g.concepts()])
    done = engine.refresh(gl, model_of(memory, g), NOW, scope=s)
    assert done.goal_completed and len(done.mastered) == 1
    assert [e.type for e in events].count(EventType.GOAL_COMPLETED) == 1
    assert [e.type for e in events].count(EventType.OBJECTIVE_MASTERED) == 1
    assert engine.record(curriculum_id_for(gl.goal_id)).status.value == "COMPLETED"
    completed = gl.model_copy(update={"status": GoalStatus.COMPLETED})
    assert not engine.refresh(completed, model_of(memory, g), NOW, scope=s).goal_completed


def test_repeated_failure_replans_a_new_version_and_keeps_history() -> None:
    memory, _, g = learner({A: 0.9, B: 0.65})
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    gl = goal(A, B, C)
    v1 = committed(engine, gl, model_of(memory, g), g)
    memory.record_evidence(LEARNER, DOMAIN, [evidence(C, "incorrect", ref=f"f{i}", at=NOW + timedelta(minutes=i))
                                             for i in range(3)], [c.ref() for c in g.concepts()])
    s, events = scope()
    update = engine.refresh(gl, model_of(memory, g), NOW, graph=g, scope=s)
    assert update.replanned is not None and update.replanned.version == 2
    assert ReplanReason.REPEATED_FAILURE in update.replanned.reasons
    remediated = update.replanned.by_concept(C)
    assert remediated.remediated and remediated.priority == 1 and remediated.evidence_required == 3
    assert engine.repository.version(v1.version_id) == v1
    assert EventType.CURRICULUM_REPLANNED in [e.type for e in events]
    # The same state again: no further version (remediation already planned).
    assert engine.refresh(gl, model_of(memory, g), NOW, graph=g, scope=s).replanned is None
    assert len(engine.versions(v1.curriculum_id)) == 2


def test_progress_and_completion_rules() -> None:
    memory, _, g = learner({A: 0.9, B: 0.9, C: 0.2})
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    model = model_of(memory, g)
    version = committed(engine, goal(B), model, g)  # target B, prerequisite A
    progress = compute_progress(version, model, CurriculumConfig(), NOW)
    assert progress.goal_complete and progress.percent_complete == 100
    targets_only = compute_progress(version, model, CurriculumConfig(completion_rule="targets_mastered"), NOW)
    assert targets_only.required == 1 and targets_only.goal_complete


def test_tracker_ignores_goals_without_curricula_and_stores_completion() -> None:
    memory, _, g = learner({A: 0.95, B: 0.95})
    engine = CurriculumEngine(InMemoryCurriculumRepository())
    tracker = CurriculumTracker(memory, engine)
    implicit = memory.save_goal(goal(A, gid="implicit"))  # e.g. a goal resolved for a plain lesson
    tracked = tracker.tracked(LEARNER, NOW, graphs={DOMAIN: g})
    assert tracked.entries == [] and tracker.next_action(LEARNER, tracked).action == LearningActionType.WAIT
    gl = memory.save_goal(goal(A, B, gid="goal-2"))
    committed(engine, gl, model_of(memory, g), g)
    tracked = tracker.tracked(LEARNER, NOW, graphs={DOMAIN: g})
    assert tracked.completed_now == {"goal-2"} and memory.goal("goal-2").status == GoalStatus.COMPLETED
    assert tracker.next_action(LEARNER, tracked).action == LearningActionType.COMPLETE
    assert memory.goal(implicit.goal_id).status == GoalStatus.ACTIVE


# --- lessons from an objective -------------------------------------------------------------------------------------


def focus(action: str, concept_id: str = B) -> LessonFocus:
    return LessonFocus(action=action, concept_id=concept_id, objective_id=f"obj-{concept_id}",
                       description="Use it accurately", goal_id="goal-1", curriculum_id="cur-1",
                       curriculum_version=1, action_id="act-1")


@pytest.mark.parametrize("action,mode,extra", [("LEARN", "reinforce", None), ("REVIEW", "review", None),
                                               ("PRACTICE", "reinforce", "act_es.preterite_apply"),
                                               ("EVALUATE", "review", "act_es.preterite_transfer")])
def test_pedagogical_plan_follows_the_curriculum_action(action, mode, extra) -> None:
    memory, _, g = learner(SPEC)
    model = model_of(memory, g)
    gl = goal()
    gaps = KnowledgeGapAnalyzer(memory.config).analyze(model, g, gl)
    planner = PedagogicalPlanner(memory.config)
    plan = planner.plan(model, gaps, gl, g, focus=focus(action))
    assert plan.target_concepts == [B] and plan.focus is not None and plan.focus.action == action
    assert plan.treatment(B).mode == mode or (action == "LEARN" and plan.treatment(B).mode == "introduce")
    ids = [a.activity_id for a in plan.activities]
    assert f"act_{B}_check" in ids  # every focus lesson checks the objective
    if extra:
        assert extra in ids
    assert plan.rationale.startswith(f"Curriculum action {action}")
    brief = plan.brief()
    assert brief.focus is not None and brief.focus.goal_id == "goal" and brief.focus.action_id == "action"
    assert plan == planner.plan(model, gaps, gl, g, focus=focus(action))  # deterministic


def test_plans_without_focus_are_unchanged() -> None:
    memory, _, g = learner(SPEC)
    model = model_of(memory, g)
    gl = goal()
    gaps = KnowledgeGapAnalyzer(memory.config).analyze(model, g, gl)
    plan = PedagogicalPlanner(memory.config).plan(model, gaps, gl, g)
    assert plan.focus is None and "focus" not in plan.brief().model_dump(exclude_none=True)
    assert not plan.rationale.startswith("Curriculum action")
