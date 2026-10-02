"""Catalogue of agents, tools, providers and workflows for introspection endpoints."""

from __future__ import annotations

from app.agents.registry import AgentRegistry
from app.learner.frameworks import FrameworkRegistry
from app.providers.core.registry import ProviderRegistry
from app.providers.core.selector import ProviderSelector
from app.providers.llm.router import ModelRouter
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.schemas.catalog import AgentInfo, ProviderInfo, ToolInfo
from app.schemas.common import ModelTier
from app.tools.registry import ToolRegistry




class CatalogService:
    def __init__(self, agents: AgentRegistry, tools: ToolRegistry, router: ModelRouter, planner: WorkflowPlanner,
                 frameworks: FrameworkRegistry, providers: ProviderRegistry | None = None,
                 selector: ProviderSelector | None = None) -> None:
        self._agents = agents
        self._providers = providers
        self._selector = selector
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
            capabilities={c.value: ids for c, ids in self._providers.capabilities().items()} if self._providers else {},
            selected={s.capability.value: s.provider for s in self._selector.describe()} if self._selector else {},
            fallbacks={s.capability.value: list(s.fallbacks) for s in self._selector.describe() if s.fallbacks}
            if self._selector else {},
            offline=self._providers.offline if self._providers else False,
        )
