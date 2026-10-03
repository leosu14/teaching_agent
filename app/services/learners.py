"""Learner application service."""

from __future__ import annotations

from app.learner.memory import LearnerMemoryService
from app.pedagogy.knowledge import KnowledgeBase
from app.pedagogy.recommend import NextLessonRecommender
from app.pedagogy.strategy import StrategyRegistry
from app.schemas.learner import (
    LearnerProfile,
    LearnerProfileInput,
    LearnerProgress,
    LearningEvidence,
    LearningGoal,
    MasteryUpdate,
)
from app.schemas.pedagogy import LearnerModel, NextLearningRecommendation, PedagogyConfig


class LearnerService:
    def __init__(self, memory: LearnerMemoryService, knowledge: KnowledgeBase | None = None,
                 pedagogy: PedagogyConfig | None = None, strategies: StrategyRegistry | None = None) -> None:
        self._memory = memory
        self._knowledge = knowledge
        self._recommender = NextLessonRecommender(pedagogy or memory.config, strategies)

    def upsert(self, learner_id: str, data: LearnerProfileInput) -> LearnerProfile:
        return self._memory.upsert(learner_id, data)

    def get(self, learner_id: str) -> LearnerProfile:
        return self._memory.get(learner_id)

    def progress(self, learner_id: str) -> LearnerProgress:
        return self._memory.progress(learner_id)

    # --- adaptive pedagogy ------------------------------------------------------------------------

    def set_goal(self, goal: LearningGoal) -> LearningGoal:
        return self._memory.save_goal(goal)

    def goals(self, learner_id: str, domain: str | None = None) -> list[LearningGoal]:
        return self._memory.goals(learner_id, domain)

    async def record_evidence(self, learner_id: str, domain: str, evidence: list[LearningEvidence]) -> MasteryUpdate:
        """Evidence from outside a workflow (an exercise, a manual placement); mastery is updated from it by code."""
        graph = await self._graph(domain)
        return self._memory.record_evidence(learner_id, domain, evidence, [c.ref() for c in graph.concepts()])

    async def model(self, learner_id: str, domain: str, framework_id: str,
                    target_level: str | None = None) -> LearnerModel:
        graph = await self._graph(domain)
        return self._memory.learner_model(learner_id, domain, framework_id, target_level, graph.ids)

    async def recommend(self, learner_id: str, goal_id: str, framework_id: str,
                        available_minutes: int | None = None) -> NextLearningRecommendation:
        goal = self._memory.goal(goal_id)
        if goal.learner_id != learner_id:
            raise ValueError(f"goal {goal_id} belongs to another learner")
        graph = await self._graph(goal.domain)
        model = self._memory.learner_model(learner_id, goal.domain, framework_id, goal.target_level, graph.ids)
        return self._recommender.recommend(model, graph, goal, available_minutes)

    async def _graph(self, domain: str):
        if self._knowledge is None:
            raise RuntimeError("LearnerService was built without a knowledge base")
        return await self._knowledge.graph(domain)
