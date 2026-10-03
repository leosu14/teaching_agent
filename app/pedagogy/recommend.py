"""Next-learning recommendation and evaluation feedback, both computed from the updated learner state."""

from __future__ import annotations

from app.pedagogy.gaps import KnowledgeGapAnalyzer
from app.pedagogy.graph import ConceptGraph
from app.pedagogy.planner import PedagogicalPlanner
from app.pedagogy.strategy import StrategyRegistry
from app.schemas.learner import LearningGoal, MasteryChange
from app.schemas.pedagogy import (
    EvaluationFeedback,
    LearnerModel,
    NextLearningRecommendation,
    PedagogyConfig,
)


class NextLessonRecommender:
    """Analyses the gaps of the updated learner model and plans the next lesson structurally; the recommendation is
    that plan's concepts, activity types and duration. A changed mastery state therefore changes the recommendation.
    """

    def __init__(self, config: PedagogyConfig | None = None, strategies: StrategyRegistry | None = None) -> None:
        self.config = config or PedagogyConfig()
        self.analyzer = KnowledgeGapAnalyzer(self.config)
        self.planner = PedagogicalPlanner(self.config, strategies)

    def recommend(self, model: LearnerModel, graph: ConceptGraph, goal: LearningGoal,
                  available_minutes: int | None = None) -> NextLearningRecommendation:
        gaps = self.analyzer.analyze(model, graph, goal)
        if not gaps.gaps:
            reviews = gaps.due_for_review[:self.config.planner.max_review_concepts]
            reason = "Every goal concept is at or above the mastery target."
            if reviews:
                reason += " Keep them fresh with a spaced review of " + ", ".join(
                    graph.concept(c).name for c in reviews) + "."
            return NextLearningRecommendation(
                learner_id=model.learner_id, goal_id=goal.goal_id, recommended_concepts=[], reason=reason,
                review_concepts=reviews, estimated_duration=len(reviews) * self.config.planner.minutes.review,
                suggested_activity_types=[], goal_achieved=True, gap_set_id=gaps.gap_set_id)
        plan = self.planner.plan(model, gaps, goal, graph, available_minutes)
        reasons = [g.reason for c in plan.target_concepts if (g := gaps.gap(c)) is not None]
        deferred = [g.concept.name for g in gaps.gaps if g.recommended_action == "prerequisite_first"
                    and g.concept.concept_id not in plan.concept_ids()]
        reason = " ".join(reasons)
        if deferred:
            reason += f" Later: {', '.join(deferred)} once their prerequisites are secure."
        return NextLearningRecommendation(
            learner_id=model.learner_id, goal_id=goal.goal_id, recommended_concepts=plan.target_concepts,
            reason=reason, prerequisite_review=plan.prerequisite_concepts, review_concepts=plan.review_concepts,
            suggested_activity_types=list(dict.fromkeys(a.type for a in plan.activities)),
            estimated_duration=plan.estimated_duration, plan_id=plan.plan_id, gap_set_id=gaps.gap_set_id)


def evaluation_feedback(changes: list[MasteryChange], assessed: list[str], model: LearnerModel,
                        recommendation: NextLearningRecommendation, config: PedagogyConfig) -> EvaluationFeedback:
    """What was mastered, what remains weak, what changed and what to review next: read from the deterministic
    mastery state and the recorded changes, never from a model's claims."""
    mastered, developing, weak = [], [], []
    for cid in dict.fromkeys(assessed):
        m = model.mastery_of(cid)
        (mastered if m >= config.mastery_target else developing if m >= config.bands.independent else weak).append(cid)
    changed = [c for c in changes if c.concept_id in set(assessed)]
    review_next = list(dict.fromkeys([*recommendation.prerequisite_review, *recommendation.recommended_concepts,
                                      *recommendation.review_concepts]))
    moves = ", ".join(f"{c.concept_id} {c.before:.2f} -> {c.after:.2f}" for c in changed) or "no change"
    summary = (f"Mastered: {', '.join(mastered) or 'none'}. Developing: {', '.join(developing) or 'none'}. "
               f"Still weak: {', '.join(weak) or 'none'}. Changes: {moves}. "
               f"Next: {', '.join(review_next) or 'nothing pending for this goal'}.")
    return EvaluationFeedback(mastered=mastered, developing=developing, still_weak=weak, changes=changed,
                              review_next=review_next, summary=summary)
