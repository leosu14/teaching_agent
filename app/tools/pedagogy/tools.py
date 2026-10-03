"""Adaptive-pedagogy tools: thin adapters that run the deterministic engine (app/pedagogy) for workflows."""

from __future__ import annotations

from app.observability.scope import ExecutionScope
from app.pedagogy.gaps import GapAnalysisError, KnowledgeGapAnalyzer
from app.pedagogy.graph import ConceptGraph
from app.pedagogy.planner import PedagogicalPlanner, PlanningError
from app.pedagogy.recommend import NextLessonRecommender, evaluation_feedback
from app.pedagogy.strategy import StrategyRegistry
from app.schemas.events import EventType
from app.schemas.pedagogy import (
    EvaluationFeedback,
    FeedbackRequest,
    GapAnalysisRequest,
    KnowledgeGapSet,
    NextLearningRecommendation,
    PedagogicalPlan,
    PedagogyConfig,
    PlanningRequest,
    RecommendationRequest,
)
from app.tools.base import Tool, ToolError


class GapAnalysisTool(Tool[GapAnalysisRequest, KnowledgeGapSet]):
    name = "pedagogy.analyze_gaps"
    description = "Deterministic knowledge-gap detection and prioritisation for a learner model against a goal."
    input_model = GapAnalysisRequest
    output_model = KnowledgeGapSet
    permissions = frozenset({"learner:read"})

    def __init__(self, config: PedagogyConfig) -> None:
        self._analyzer = KnowledgeGapAnalyzer(config)

    async def run(self, data: GapAnalysisRequest, scope: ExecutionScope) -> KnowledgeGapSet:
        try:
            gaps = self._analyzer.analyze(data.model, ConceptGraph(data.concepts), data.goal)
        except GapAnalysisError as exc:
            raise ToolError(str(exc)) from exc
        scope.emit(EventType.KNOWLEDGE_GAPS_ANALYZED, gap_set_id=gaps.gap_set_id, goal_id=gaps.goal_id,
                   gaps=[g.concept.concept_id for g in gaps.gaps], due_for_review=gaps.due_for_review)
        return gaps


class PlanningTool(Tool[PlanningRequest, PedagogicalPlan]):
    name = "pedagogy.plan"
    description = "Deterministic pedagogical plan: targets, prerequisite review, objectives, activities, timing."
    input_model = PlanningRequest
    output_model = PedagogicalPlan
    permissions = frozenset({"learner:read"})

    def __init__(self, config: PedagogyConfig, strategies: StrategyRegistry) -> None:
        self._planner = PedagogicalPlanner(config, strategies)

    async def run(self, data: PlanningRequest, scope: ExecutionScope) -> PedagogicalPlan:
        try:
            plan = self._planner.plan(data.model, data.gaps, data.goal, ConceptGraph(data.concepts),
                                      data.available_minutes, data.lesson_history)
        except PlanningError as exc:
            raise ToolError(str(exc)) from exc
        scope.emit(EventType.PEDAGOGICAL_PLAN_CREATED, plan_id=plan.plan_id, targets=plan.target_concepts,
                   prerequisites=plan.prerequisite_concepts, reviews=plan.review_concepts,
                   minutes=plan.estimated_duration)
        return plan


class RecommendationTool(Tool[RecommendationRequest, NextLearningRecommendation]):
    name = "pedagogy.recommend"
    description = "The next-learning recommendation computed from the (updated) learner model."
    input_model = RecommendationRequest
    output_model = NextLearningRecommendation
    permissions = frozenset({"learner:read"})

    def __init__(self, config: PedagogyConfig, strategies: StrategyRegistry) -> None:
        self._recommender = NextLessonRecommender(config, strategies)

    async def run(self, data: RecommendationRequest, scope: ExecutionScope) -> NextLearningRecommendation:
        try:
            rec = self._recommender.recommend(data.model, ConceptGraph(data.concepts), data.goal,
                                              data.available_minutes)
        except (GapAnalysisError, PlanningError) as exc:
            raise ToolError(str(exc)) from exc
        scope.emit(EventType.NEXT_RECOMMENDATION_CREATED, goal_id=rec.goal_id, concepts=rec.recommended_concepts,
                   prerequisite_review=rec.prerequisite_review, goal_achieved=rec.goal_achieved)
        return rec


class FeedbackTool(Tool[FeedbackRequest, EvaluationFeedback]):
    name = "pedagogy.evaluation_feedback"
    description = "What an evaluation changed: mastered, still weak, mastery changes, what to review next."
    input_model = FeedbackRequest
    output_model = EvaluationFeedback
    permissions = frozenset({"learner:read"})

    def __init__(self, config: PedagogyConfig) -> None:
        self._config = config

    async def run(self, data: FeedbackRequest, scope: ExecutionScope) -> EvaluationFeedback:
        return evaluation_feedback(data.changes, data.assessed_concepts, data.model, data.recommendation,
                                   self._config)
