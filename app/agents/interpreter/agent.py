"""RequestInterpreterAgent: turns a free-text request into a structured LessonRequest."""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentSpec
from app.schemas.common import ModelTier
from app.schemas.learner import ANONYMOUS_LEARNER, LearnerSummary
from app.schemas.lesson import InterpreterInput, InterpretRequest, LessonRequest


class RequestInterpreterAgent(Agent[InterpretRequest, LessonRequest]):
    spec = AgentSpec(
        id="request_interpreter",
        name="Request Interpreter",
        description="Classifies a learning request into subject, topic, level framework, level and capabilities.",
        input_model=InterpretRequest,
        output_model=LessonRequest,
        tier=ModelTier.CHEAP,
        tools=("learner.summary",),
        permissions=frozenset({"learner:read"}),
        max_output_tokens=800,
    )
    instructions = "Interpret the learner's request using their profile."

    async def run(self, data: InterpretRequest, ctx: AgentContext) -> LessonRequest:
        summary = await self.use_tool("learner.summary", {"learner_id": data.learner_id}, ctx)
        assert isinstance(summary, LearnerSummary)
        # The model needs the learner's subjects, levels and preferences, not who they are.
        payload = InterpreterInput(request=data.request, learner=summary.model_copy(
            update={"learner_id": ANONYMOUS_LEARNER, "display_name": ""}))
        return await self.generate(payload, ctx, source=payload)
