"""CurriculumPlannerAgent: builds a lesson plan optimised for this learner."""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import LessonPlan, PlannerInput


class CurriculumPlannerAgent(Agent[PlannerInput, LessonPlan]):
    spec = AgentSpec(
        id="curriculum_planner",
        name="Curriculum Planner",
        description="Plans objectives, concept sequence, timing, exercises, assessment and remediation.",
        input_model=PlannerInput,
        output_model=LessonPlan,
        tier=ModelTier.REASONING,
    )
    instructions = "Plan the next lesson for this learner."

    def check(self, output: LessonPlan, source: PlannerInput) -> None:
        allowed = {c.concept_id for c in source.concepts}
        extra = [c.concept_id for c in output.concepts if c.concept_id not in allowed]
        if extra:
            raise OutputRejected(f"plan uses concepts outside the topic: {extra}")
        if output.level not in source.snapshot.framework_levels:
            raise OutputRejected(f"level must be one of {source.snapshot.framework_levels}")
