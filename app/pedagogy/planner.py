"""PedagogicalPlanner: learner model + knowledge gaps + goal + time -> a structured, validated PedagogicalPlan.

Deterministic: identical learner state, goal, concepts and configuration always give the same plan (and plan id).
It decides the target concepts, prerequisite review, introduction vs reinforcement, practice and assessment types
and the time allocation. Models only word the lesson afterwards.

With a `LessonFocus` (a lesson started from a curriculum's next learning action) the target is the focus objective's
concept instead of the top gaps, treated as the action asks: LEARN teaches it (introduce, or reteach after earlier
lessons), REVIEW retrieves it (review mode, then a check), PRACTICE reinforces it with an extra applied exercise,
EVALUATE retrieves it and assesses it twice. Prerequisite review and spaced review are added as usual.
"""

from __future__ import annotations

from app.pedagogy.graph import ConceptGraph
from app.pedagogy.strategy import PedagogicalStrategy, PlannedActivity, StrategyRegistry
from app.schemas.learner import LearningEvent, LearningGoal
from app.schemas.pedagogy import (
    ConceptRole,
    ConceptTreatment,
    KnowledgeGap,
    KnowledgeGapSet,
    LearnerModel,
    LearningActivity,
    LearningObjective,
    LessonFocus,
    PedagogicalPlan,
    PedagogyConfig,
    SequenceStep,
    TeachingMode,
)


class PlanningError(ValueError):
    """No valid plan exists (nothing to teach, or nothing fits the available time)."""


class PedagogicalPlanner:
    def __init__(self, config: PedagogyConfig | None = None, strategies: StrategyRegistry | None = None) -> None:
        self.config = config or PedagogyConfig()
        self.strategies = strategies or StrategyRegistry()

    def plan(self, model: LearnerModel, gaps: KnowledgeGapSet, goal: LearningGoal, graph: ConceptGraph,
             available_minutes: int | None = None, lesson_history: list[LearningEvent] | None = None,
             focus: LessonFocus | None = None) -> PedagogicalPlan:
        if gaps.learner_id != model.learner_id or gaps.goal_id != goal.goal_id:
            raise PlanningError("the gap set must be for this learner and goal")
        cfg = self.config
        available = available_minutes or model.preferences.session_minutes or cfg.planner.default_minutes
        strategy = self.strategies.for_domain(model.domain, cfg)
        history = model.learning_history if lesson_history is None else lesson_history
        taught_before = {c for e in history if e.type == "lesson_completed" for c in e.concept_ids}

        selection = _Selection(self, strategy, model, gaps, graph, taught_before, available)
        if focus is not None:
            return self._focused(model, gaps, goal, graph, strategy, selection, available, focus)
        for gap in gaps.gaps:
            if len(selection.targets) >= cfg.planner.max_target_concepts:
                break
            selection.consider(gap)
        if not selection.targets:
            if not gaps.gaps:
                raise PlanningError(f"goal {goal.goal_id} has no knowledge gaps left to plan for")
            raise PlanningError(f"no gap fits the available {available} minutes")
        for cid in gaps.due_for_review[:cfg.planner.max_review_concepts]:
            selection.add_review(cid)
        return self._build(model, gaps, goal, graph, strategy, selection, available)

    def treatment(self, gap: KnowledgeGap, role: ConceptRole, taught_before: set[str],
                  prerequisites: list[str]) -> ConceptTreatment:
        cid = gap.concept.concept_id
        if role != "target":
            mode: TeachingMode = "review"
        elif gap.recommended_action == "reinforce" or cid in taught_before:
            mode = "reinforce"
        else:
            mode = "introduce"
        return ConceptTreatment(concept_id=cid, name=gap.concept.name, role=role, mode=mode, band=gap.band,
                                mastery=gap.mastery, prerequisites=prerequisites)

    def _focused(self, model: LearnerModel, gaps: KnowledgeGapSet, goal: LearningGoal, graph: ConceptGraph,
                 strategy: PedagogicalStrategy, sel: _Selection, available: int,
                 focus: LessonFocus) -> PedagogicalPlan:
        cid = focus.concept_id
        if cid not in graph:
            raise PlanningError(f"the focus concept {cid} is not in the knowledge base")
        gap = gaps.gap(cid)
        mastery = model.mastery_of(cid)
        reviews = [p for p in (gap.unmet_prerequisites if gap else []) if not sel.chosen(p)]
        in_plan = [p for p in graph.prerequisites(cid) if p in reviews]
        if focus.action == "LEARN" and gap is not None:
            target = self.treatment(gap, "target", sel.taught_before, in_plan)
        else:
            mode: TeachingMode = {"LEARN": "introduce", "PRACTICE": "reinforce"}.get(focus.action, "review")
            target = ConceptTreatment(concept_id=cid, name=graph.concept(cid).name, role="target", mode=mode,
                                      band=self.config.bands.band_for(mastery), mastery=mastery,
                                      prerequisites=in_plan)
        review_treatments = []
        for p in reviews:
            state = model.state(p)
            pm = state.mastery if state else 0.0
            review_treatments.append(ConceptTreatment(concept_id=p, name=graph.concept(p).name, role="prerequisite",
                                                      mode="review", band=self.config.bands.band_for(pm), mastery=pm))
        extra = self._focus_minutes(focus, target)
        if not sel.admit(target, review_treatments, extra):
            raise PlanningError(f"the focus objective does not fit the available {available} minutes")
        for r in gaps.due_for_review[:self.config.planner.max_review_concepts]:
            if r != cid:
                sel.add_review(r)
        return self._build(model, gaps, goal, graph, strategy, sel, available, focus)

    def _focus_minutes(self, focus: LessonFocus, target: ConceptTreatment) -> int:
        """Minutes the strategy's estimate leaves out: the extra activity, and the check of a target in review mode."""
        m = self.config.planner.minutes
        return ({"PRACTICE": m.practice, "EVALUATE": m.assessment}.get(focus.action, 0)
                + (m.assessment if target.mode == "review" else 0))

    def _focus_steps(self, focus: LessonFocus, objective: LearningObjective, treatment: ConceptTreatment,
                     name: str) -> list[PlannedActivity]:
        """The extra activity a PRACTICE or EVALUATE action asks for (none for LEARN and REVIEW)."""
        m = self.config.planner.minutes
        cid = focus.concept_id
        if focus.action == "PRACTICE":
            return [PlannedActivity("practice", LearningActivity(
                activity_id=f"act_{cid}_apply", type="free_response", concept_ids=[cid], difficulty=treatment.band,
                estimated_minutes=m.practice, instructions=f"Apply {name} in a new, realistic context.",
                expected_response="an original answer that applies the concept",
                assessment_target=objective.objective_id))]
        if focus.action == "EVALUATE":
            return [PlannedActivity("assessment", LearningActivity(
                activity_id=f"act_{cid}_transfer", type="free_response", concept_ids=[cid], difficulty=treatment.band,
                estimated_minutes=m.assessment, instructions=f"Check transfer: use {name} without support.",
                expected_response="an original answer that applies the concept",
                assessment_target=objective.objective_id))]
        return []

    def _build(self, model: LearnerModel, gaps: KnowledgeGapSet, goal: LearningGoal, graph: ConceptGraph,
               strategy: PedagogicalStrategy, sel: _Selection, available: int,
               focus: LessonFocus | None = None) -> PedagogicalPlan:
        cfg = self.config
        prerequisites = graph.order(sel.prerequisites)
        targets = graph.order(sel.targets)
        reviews = list(sel.reviews)
        treatments = [sel.treatments[c] for c in [*prerequisites, *targets, *reviews]]
        objectives: dict[str, LearningObjective] = {}
        steps: list[PlannedActivity] = []
        for t in treatments:
            concept = graph.concept(t.concept_id)
            objectives[t.concept_id] = strategy.objective(concept, t, cfg)
            steps += strategy.activities(concept, t, objectives[t.concept_id], cfg)
        if focus is not None:
            steps += self._focus_steps(focus, objectives[focus.concept_id], sel.treatments[focus.concept_id],
                                       graph.concept(focus.concept_id).name)
        for cid in targets:
            steps.append(strategy.assessment(objectives[cid], sel.treatments[cid].band, cfg))
        phase_rank = {"prerequisite_review": 0, "instruction": 1, "practice": 1, "spaced_review": 2, "assessment": 3}
        steps = sorted(steps, key=lambda p: phase_rank[p.phase])  # stable: keeps concept order within a phase
        activities = [p.activity for p in steps]
        sequencing = [SequenceStep(order=i, activity_id=p.activity.activity_id, phase=p.phase,
                                   minutes=p.activity.estimated_minutes) for i, p in enumerate(steps, start=1)]
        plan = PedagogicalPlan(
            plan_id="pending", domain=model.domain, level=model.level or goal.target_level,
            strategy_id=strategy.strategy_id, target_concepts=targets, prerequisite_concepts=prerequisites,
            review_concepts=reviews, treatments=treatments, lesson_objectives=list(objectives.values()),
            activities=activities, sequencing=sequencing, estimated_duration=sum(s.minutes for s in sequencing),
            available_minutes=available, rationale=self._rationale(gaps, sel, graph, focus), focus=focus,
            learner_id=model.learner_id, goal_id=goal.goal_id, gap_set_id=gaps.gap_set_id,
            config_fingerprint=cfg.fingerprint())
        return plan.model_copy(update={"plan_id": f"pp_{plan.brief().structural_hash()[:16]}"})

    @staticmethod
    def _rationale(gaps: KnowledgeGapSet, sel: _Selection, graph: ConceptGraph,
                   focus: LessonFocus | None = None) -> str:
        def name(cid: str) -> str:
            return graph.concept(cid).name
        parts = []
        for cid in graph.order(sel.targets):
            t = sel.treatments[cid]
            parts.append(f"{t.mode} {name(cid)} ({t.band}, mastery {t.mastery:.2f})")
        text = "Targets: " + "; ".join(parts) + "."
        if focus is not None:
            text = f"Curriculum action {focus.action} for the objective: {focus.description}. " + text
        if sel.prerequisites:
            text += " Prerequisite review first: " + ", ".join(name(c) for c in graph.order(sel.prerequisites)) + "."
        if sel.deferred:
            text += " Deferred until prerequisites are secure: " + ", ".join(
                f"{name(c)} (needs {', '.join(name(p) for p in blockers)})" for c, blockers in sel.deferred.items()) + "."
        if sel.reviews:
            text += " Spaced review: " + ", ".join(name(c) for c in sel.reviews) + "."
        return text


class _Selection:
    """Greedy, deterministic selection of the lesson's concepts by gap priority within the time available."""

    def __init__(self, planner: PedagogicalPlanner, strategy: PedagogicalStrategy, model: LearnerModel,
                 gaps: KnowledgeGapSet, graph: ConceptGraph, taught_before: set[str], available: int) -> None:
        self.planner, self.strategy, self.model, self.gaps, self.graph = planner, strategy, model, gaps, graph
        self.config = planner.config
        self.taught_before = taught_before
        self.available = available
        self.targets: list[str] = []
        self.prerequisites: list[str] = []
        self.reviews: list[str] = []
        self.treatments: dict[str, ConceptTreatment] = {}
        self.deferred: dict[str, list[str]] = {}
        self.used = 0

    def chosen(self, cid: str) -> bool:
        return cid in self.treatments

    def consider(self, gap: KnowledgeGap) -> None:
        cid = gap.concept.concept_id
        if self.chosen(cid):
            return
        if gap.recommended_action != "prerequisite_first":
            self._add_target(gap, [])
            return
        unmet = gap.unmet_prerequisites
        reviewable = all(self.model.mastery_of(p) >= self.config.bands.guided
                         and not (self.gaps.gap(p) and self.gaps.gap(p).unmet_prerequisites) for p in unmet)
        new_reviews = [p for p in unmet if not self.chosen(p)]
        if reviewable and len(self.prerequisites) + len(new_reviews) <= self.config.planner.max_prerequisite_concepts:
            if self._add_target(gap, new_reviews):
                return
        # The prerequisites are too weak to review in passing: learn the deepest blocking one first instead.
        self.deferred[cid] = unmet
        blocker = self._blocker(cid)
        if blocker is not None and not self.chosen(blocker) and \
                len(self.targets) < self.config.planner.max_target_concepts:
            blocker_gap = self.gaps.gap(blocker)
            assert blocker_gap is not None
            self._add_target(blocker_gap, [])

    def _blocker(self, cid: str) -> str | None:
        gap = self.gaps.gap(cid)
        if gap is None:
            return None
        for pre in self.graph.order(gap.unmet_prerequisites):
            pre_gap = self.gaps.gap(pre)
            if pre_gap is None:
                continue
            if not pre_gap.unmet_prerequisites:
                return pre
            deeper = self._blocker(pre)
            if deeper is not None:
                return deeper
        return None

    def _add_target(self, gap: KnowledgeGap, reviews: list[str]) -> bool:
        cid = gap.concept.concept_id
        in_plan = [p for p in self.graph.prerequisites(cid) if self.chosen(p) or p in reviews]
        target = self.planner.treatment(gap, "target", self.taught_before, in_plan)
        review_treatments = []
        for p in reviews:
            pre_gap = self.gaps.gap(p)
            assert pre_gap is not None
            review_treatments.append(self.planner.treatment(pre_gap, "prerequisite", self.taught_before, []))
        return self.admit(target, review_treatments)

    def admit(self, target: ConceptTreatment, review_treatments: list[ConceptTreatment], extra: int = 0) -> bool:
        """Add a target with the prerequisites it reviews if they fit the time left (plus `extra` minutes)."""
        cost = extra + self.strategy.minutes(target, self.config) + sum(
            self.strategy.minutes(t, self.config) for t in review_treatments)
        if self.used + cost > self.available:
            return False
        self.used += cost
        for t in review_treatments:
            self.treatments[t.concept_id] = t
            self.prerequisites.append(t.concept_id)
        self.treatments[target.concept_id] = target
        self.targets.append(target.concept_id)
        return True

    def add_review(self, cid: str) -> None:
        if self.chosen(cid):
            return
        state = self.model.state(cid)
        mastery = state.mastery if state else 0.0
        t = ConceptTreatment(concept_id=cid, name=self.graph.concept(cid).name, role="review", mode="review",
                             band=self.config.bands.band_for(mastery), mastery=mastery)
        cost = self.strategy.minutes(t, self.config)
        if self.used + cost <= self.available:
            self.used += cost
            self.treatments[cid] = t
            self.reviews.append(cid)
