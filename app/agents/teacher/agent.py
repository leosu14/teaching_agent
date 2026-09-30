"""TeacherAgent: writes structured lesson content from the plan and research."""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import LessonContent, TeacherInput


class TeacherAgent(Agent[TeacherInput, LessonContent]):
    spec = AgentSpec(
        id="teacher",
        name="Teacher",
        description="Explains concepts with examples and analogies, writes narration, exercises and questions.",
        input_model=TeacherInput,
        output_model=LessonContent,
        tier=ModelTier.REASONING,
        max_output_tokens=8000,
    )
    instructions = "Write the lesson content. If a revision is requested, fix every listed issue."

    def check(self, output: LessonContent, source: TeacherInput) -> None:
        planned = {c.concept_id for c in source.plan.concepts}
        stray = [s.section_id for s in output.sections if s.concept_id not in planned]
        if stray:
            raise OutputRejected(f"sections {stray} teach concepts that are not in the plan")
        known = {c.citation_id for c in source.research.citations}
        invented = sorted({c for s in output.sections for c in s.citations} - known)
        if invented:
            raise OutputRejected(f"citations {invented} do not exist in the research bundle")
