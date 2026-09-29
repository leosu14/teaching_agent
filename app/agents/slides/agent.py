"""SlideGenerationAgent: plans a slide deck synchronised with the lesson narration."""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import SlideDeckPlan, SlideInput


class SlideGenerationAgent(Agent[SlideInput, SlideDeckPlan]):
    spec = AgentSpec(
        id="slide_generation",
        name="Slide Generation",
        description="Turns lesson content into a structured slide plan with visual hierarchy and narration links.",
        input_model=SlideInput,
        output_model=SlideDeckPlan,
        tier=ModelTier.STANDARD,
    )
    instructions = "Plan the slides for this lesson."

    def check(self, output: SlideDeckPlan, source: SlideInput) -> None:
        sections = {s.section_id for s in source.lesson.sections}
        referenced = {s.narration_section_id for s in output.slides if s.narration_section_id}
        if referenced - sections:
            raise OutputRejected(f"slides reference unknown sections {sorted(referenced - sections)}")
        if sections - referenced:
            raise OutputRejected(f"sections {sorted(sections - referenced)} have no slide")
