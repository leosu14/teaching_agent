"""ContentReviewAgent: structured quality review of generated lesson content."""

from __future__ import annotations

from app.agents.base import Agent, AgentSpec
from app.schemas.common import ModelTier
from app.schemas.lesson import ReviewerInput, ReviewResult


class ContentReviewAgent(Agent[ReviewerInput, ReviewResult]):
    spec = AgentSpec(
        id="content_reviewer",
        name="Content Reviewer",
        description=(
            "Evaluates factual correctness, pedagogy, level fit, structure, completeness, hallucination risk, "
            "source quality, narration and exercises; approves or requests revision."
        ),
        input_model=ReviewerInput,
        output_model=ReviewResult,
        tier=ModelTier.REASONING,
    )
    instructions = "Review the lesson content against the plan and research."
