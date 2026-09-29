"""KnowledgeDiagnosticAgent: adaptive diagnosis of what the learner knows about a topic."""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentError, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import DiagnosticInput, DiagnosticStep
from app.tools.rag.retrieve import ConceptMapOutput


class KnowledgeDiagnosticAgent(Agent[DiagnosticInput, DiagnosticStep]):
    spec = AgentSpec(
        id="knowledge_diagnostic",
        name="Knowledge Diagnostic",
        description=(
            "Identifies the topic's concepts and prerequisites, decides whether memory is enough, asks adaptive "
            "diagnostic questions, grades answers and estimates mastery, gaps and a starting point."
        ),
        input_model=DiagnosticInput,
        output_model=DiagnosticStep,
        tier=ModelTier.STANDARD,
        tools=("rag.concept_map",),
        permissions=frozenset({"knowledge:read"}),
    )
    instructions = "Run the next step of the diagnostic for this learner."

    async def run(self, data: DiagnosticInput, ctx: AgentContext) -> DiagnosticStep:
        if not data.concepts:
            found = await self.use_tool("rag.concept_map", {
                "subject": data.request.subject, "topic": data.request.topic, "level": data.request.target_level,
            }, ctx)
            assert isinstance(found, ConceptMapOutput)
            if not found.concepts:
                raise AgentError(f"no concepts known for {data.request.subject}/{data.request.topic}")
            data = data.model_copy(update={"concepts": found.concepts})
        return await self.generate(data, ctx, source=data)

    def check(self, output: DiagnosticStep, source: DiagnosticInput) -> None:
        concept_ids = {e.concept.concept_id for e in source.concepts}
        if output.status == "ask":
            if len(source.rounds) >= source.max_rounds:
                raise OutputRejected("the question budget is used up; conclude with status 'complete'")
            ids = [i.question.question_id for i in output.items]
            if len(ids) != len(set(ids)):
                raise OutputRejected("question ids must be unique")
            asked = {i.question.question_id for r in source.rounds for i in r.items}
            if asked & set(ids):
                raise OutputRejected("question ids must not repeat earlier rounds")
            if any(i.question.concept_id not in concept_ids for i in output.items):
                raise OutputRejected("questions must target the topic's concepts")
        else:
            result = output.result
            assert result is not None
            if result.estimated_level not in source.snapshot.framework_levels:
                raise OutputRejected(f"estimated_level must be one of {source.snapshot.framework_levels}")
            if {c.concept_id for c in result.concept_mastery} != concept_ids:
                raise OutputRejected("concept_mastery must cover exactly the topic's concepts")
            if result.starting_point not in concept_ids:
                raise OutputRejected("starting_point must be one of the topic's concepts")
