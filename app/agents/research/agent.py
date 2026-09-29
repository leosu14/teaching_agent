"""KnowledgeResearchAgent: gathers, ranks and extracts traceable knowledge for a lesson."""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentSpec, OutputRejected
from app.schemas.common import ModelTier
from app.schemas.lesson import ResearchBundle, ResearchInput, ResearchRequest
from app.tools.rag.retrieve import RetrieveOutput
from app.tools.web.search import WebSearchOutput


class KnowledgeResearchAgent(Agent[ResearchRequest, ResearchBundle]):
    spec = AgentSpec(
        id="knowledge_research",
        name="Knowledge Research",
        description="Searches web and knowledge base, rates source reliability and extracts cited facts.",
        input_model=ResearchRequest,
        output_model=ResearchBundle,
        tier=ModelTier.STANDARD,
        tools=("search.web", "rag.retrieve"),
        permissions=frozenset({"network", "knowledge:read"}),
    )
    instructions = "Build a research bundle for this lesson from the candidate sources."

    async def run(self, data: ResearchRequest, ctx: AgentContext) -> ResearchBundle:
        req = data.request
        query = " ".join(filter(None, [req.topic, req.subject, data.diagnostic.estimated_level]))
        web = await self.use_tool("search.web", {"query": query, "max_results": 8}, ctx)
        kb = await self.use_tool("rag.retrieve", {
            "query": query, "k": 5, "filters": {"kind": "reference", "subject": req.subject},
        }, ctx)
        assert isinstance(web, WebSearchOutput) and isinstance(kb, RetrieveOutput)
        payload = ResearchInput(query=query, request=req, diagnostic=data.diagnostic, concepts=data.concepts,
                                candidates=[*web.candidates, *kb.passages])
        return await self.generate(payload, ctx, source=payload)

    def check(self, output: ResearchBundle, source: ResearchInput) -> None:
        candidates = {c.source_id for c in source.candidates}
        unknown = [s.source_id for s in output.sources if s.source_id not in candidates]
        if unknown:
            raise OutputRejected(f"sources {unknown} were not among the retrieved candidates")
        if not output.facts:
            raise OutputRejected("no facts extracted; extract facts from the reliable sources")
