"""Composition root: builds every component from settings and wires the layers together.

This is the only place that knows concrete implementations. The API and the CLI demo both use it,
so they run the exact same application services.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.agents.diagnostic.agent import KnowledgeDiagnosticAgent
from app.agents.interpreter.agent import RequestInterpreterAgent
from app.agents.planner.agent import CurriculumPlannerAgent
from app.agents.registry import AgentRegistry
from app.agents.research.agent import KnowledgeResearchAgent
from app.agents.reviewer.agent import ContentReviewAgent
from app.agents.slides.agent import SlideGenerationAgent
from app.agents.teacher.agent import TeacherAgent
from app.artifacts.service import ArtifactService
from app.config.routing import ConfigError, load_routing
from app.config.settings import Settings
from app.learner.frameworks import FrameworkRegistry, default_frameworks
from app.learner.memory import LearnerMemoryService
from app.observability.events import EventBus
from app.observability.logging import log_event
from app.providers.image.mock import MockImageProvider
from app.providers.llm.base import LLMProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.llm.router import ModelRouter
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.search.mock import CorpusSearchProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.video.mock import MockVideoProvider
from app.runtime.orchestrator.orchestrator import NodeObserver, Orchestrator
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.runtime.workflow.engine import WorkflowEngine
from app.runtime.workflows.lesson_generation import LessonWorkflowOptions, lesson_template
from app.schemas.workflow import RevisionPolicy
from app.services.catalog import CatalogService
from app.services.learners import LearnerService
from app.services.tasks import TaskService
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import (
    SqlArtifactRepository,
    SqlEventRepository,
    SqlLearnerRepository,
    SqlTaskRepository,
)
from app.tools.artifacts.tools import StoreArtifactsTool
from app.tools.learner.tools import LearnerSnapshotTool, LearnerSummaryTool, RecordLessonTool
from app.tools.manager import ToolManager
from app.tools.media.tools import ImageGenerationTool, SpeechSynthesisTool, VideoRenderTool
from app.tools.rag.retrieve import ConceptMapTool, RetrievalTool
from app.tools.registry import ToolRegistry
from app.tools.web.search import WebSearchTool

LLM_PROVIDER_FACTORIES = {
    "mock": lambda settings: MockLLMProvider(default_responders()),
}


@dataclass
class Container:
    settings: Settings
    events: EventBus
    llm_providers: dict[str, LLMProvider]
    router: ModelRouter
    tools: ToolManager
    agents: AgentRegistry
    frameworks: FrameworkRegistry
    memory: LearnerMemoryService
    artifacts: ArtifactService
    orchestrator: Orchestrator
    task_service: TaskService
    learner_service: LearnerService
    catalog: CatalogService
    _sessions: object

    def close(self) -> None:
        dispose(self._sessions)


def build_llm_providers(settings: Settings) -> dict[str, LLMProvider]:
    providers = {}
    for name in settings.llm_providers:
        factory = LLM_PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ConfigError(f"LLM provider '{name}' has no adapter yet (available: {sorted(LLM_PROVIDER_FACTORIES)})")
        providers[name] = factory(settings)
    return providers


def build_container(
    settings: Settings | None = None,
    *,
    llm_providers: dict[str, LLMProvider] | None = None,
    observers: Sequence[NodeObserver] = (),
) -> Container:
    settings = settings or Settings()
    settings.validate_runtime()
    routing = load_routing(settings.routing_file)
    llm = llm_providers if llm_providers is not None else build_llm_providers(settings)
    router = ModelRouter(routing, llm)

    sessions = create_db(settings.resolved_database_url)
    task_repo = SqlTaskRepository(sessions)
    event_repo = SqlEventRepository(sessions)
    events = EventBus()
    events.subscribe(event_repo.append)
    events.subscribe(log_event)

    frameworks = default_frameworks()
    memory = LearnerMemoryService(SqlLearnerRepository(sessions), frameworks)
    artifacts = ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(settings.resolved_object_store_dir))

    retriever = LocalKnowledgeBase(settings.corpus_dir / "knowledge_base.json")
    registry = ToolRegistry()
    for tool in (
        WebSearchTool(CorpusSearchProvider(settings.corpus_dir / "web_corpus.json")),
        RetrievalTool(retriever),
        ConceptMapTool(retriever),
        LearnerSummaryTool(memory),
        LearnerSnapshotTool(memory),
        RecordLessonTool(memory),
        StoreArtifactsTool(artifacts),
        ImageGenerationTool(MockImageProvider(), artifacts),
        SpeechSynthesisTool(MockTTSProvider(), artifacts),
        VideoRenderTool(MockVideoProvider(), artifacts),
    ):
        registry.register(tool)
    tools = ToolManager(registry)

    agents = AgentRegistry()
    for agent in (RequestInterpreterAgent(), KnowledgeDiagnosticAgent(), KnowledgeResearchAgent(),
                  CurriculumPlannerAgent(), TeacherAgent(), ContentReviewAgent(), SlideGenerationAgent()):
        agents.register(agent)
    for agent_id in routing.agent_tiers:
        agents.get(agent_id)  # overrides must name real agents

    options = LessonWorkflowOptions(
        diagnostic_rounds=settings.diagnostic_max_rounds,
        memory_confidence=settings.diagnostic_memory_confidence,
        revision_policy=RevisionPolicy(max_revisions=settings.max_revisions,
                                       on_exhausted=settings.revision_exhausted_policy),
    )
    planner = WorkflowPlanner([lesson_template(options)], router, agents)
    orchestrator = Orchestrator(tasks=task_repo, engine=WorkflowEngine(agents, tools, router), planner=planner,
                                agents=agents, tools=tools, router=router, events=events, observers=observers)
    return Container(
        settings=settings, events=events, llm_providers=llm, router=router, tools=tools, agents=agents,
        frameworks=frameworks, memory=memory, artifacts=artifacts, orchestrator=orchestrator,
        task_service=TaskService(orchestrator, task_repo, event_repo, artifacts),
        learner_service=LearnerService(memory),
        catalog=CatalogService(agents, registry, router, planner, frameworks),
        _sessions=sessions,
    )
