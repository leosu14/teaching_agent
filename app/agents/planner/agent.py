"""CurriculumPlannerAgent: turns the deterministic pedagogical plan into a worded lesson plan for this learner."""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import LessonPlan, PlannerInput


class CurriculumPlannerAgent(Agent[PlannerInput, LessonPlan]):
    spec = AgentSpec(
        id="curriculum_planner",
        name="Curriculum Planner",
        description=("Words the deterministic pedagogical plan as a lesson plan: objectives, sequence, exercises, "
                     "assessment and remediation."),
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
        if source.learner.levels and output.level not in source.learner.levels:
            raise OutputRejected(f"level must be one of {source.learner.levels}")
        # The concepts, reviews and time come from the deterministic pedagogical plan; the model only words them.
        brief = source.pedagogical_plan
        teach = {*brief.prerequisite_concepts, *brief.target_concepts}
        if {c.concept_id for c in output.concepts} != teach:
            raise OutputRejected(f"plan exactly the pedagogical plan's concepts {sorted(teach)}")
        if set(output.review_concepts) != set(brief.review_concepts):
            raise OutputRejected(f"review exactly the concepts the pedagogical plan reviews {brief.review_concepts}")
        if output.estimated_minutes > brief.available_minutes:
            raise OutputRejected(f"the lesson must fit {brief.available_minutes} minutes")
