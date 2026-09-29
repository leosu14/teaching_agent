"""Catalogue of agents, tools, providers and workflows for introspection endpoints."""

from __future__ import annotations

from app.agents.registry import AgentRegistry
from app.learner.frameworks import FrameworkRegistry
from app.providers.llm.router import ModelRouter
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.schemas.catalog import AgentInfo, ProviderInfo, ToolInfo
from app.schemas.common import ModelTier
from app.tools.registry import ToolRegistry




class CatalogService:
    def __init__(self, agents: AgentRegistry, tools: ToolRegistry, router: ModelRouter, planner: WorkflowPlanner,
                 frameworks: FrameworkRegistry) -> None:
        self._agents = agents
        self._tools = tools
        self._router = router
        self._planner = planner
        self._frameworks = frameworks

    def agents(self) -> list[AgentInfo]:
        return self._agents.describe()

    def tools(self) -> list[ToolInfo]:
        return self._tools.describe()

    def providers(self) -> ProviderInfo:
        return ProviderInfo(
            llm_providers=self._router.provider_names,
            tiers={t.value: [f"{x.provider}/{x.model}" for x in self._router.targets(t)] for t in ModelTier},
            level_frameworks=self._frameworks.ids(),
            workflows=[t.id for t in self._planner.templates()],
        )
