"""Composition root: builds every component from settings and wires the layers together.

This is the only place that knows concrete implementations. The API and the CLI demo both use it,
so they run the exact same application services.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.agents.audio.agent import AudioPlannerAgent
from app.agents.diagnostic.agent import KnowledgeDiagnosticAgent
from app.agents.evaluator.agent import LearnerEvaluationAgent
from app.agents.interpreter.agent import RequestInterpreterAgent
from app.agents.planner.agent import CurriculumPlannerAgent
from app.agents.registry import AgentRegistry
from app.agents.research.agent import ResearchAgent
from app.agents.reviewer.agent import ContentReviewAgent
from app.agents.slides.agent import SlidePlannerAgent
from app.agents.teacher.agent import TeacherAgent
from app.agents.visual.agent import VisualAgent
from app.artifacts.service import ArtifactService
from app.config.routing import ConfigError, load_routing
from app.config.settings import Settings
from app.learner.frameworks import FrameworkRegistry, default_frameworks
from app.learner.memory import LearnerMemoryService
from app.observability.events import EventBus
from app.observability.logging import log_event
from app.providers.image.base import ImageGenerationProvider
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.image_search.base import ImageSearchProvider
from app.providers.image_search.mock import MockImageSearchProvider
from app.providers.llm.base import LLMProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.llm.router import ModelRouter
from app.providers.presentation.base import PresentationRenderer
from app.providers.presentation.mock import MockPresentationRenderer
from app.providers.presentation.pptx_renderer import PptxPresentationRenderer
from app.providers.retrieval.base import Retriever
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.ranking.heuristic import HeuristicRanker
from app.providers.search.base import SearchProvider
from app.providers.search.mock import MockSearchProvider
from app.providers.tts.base import TTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.video.mock import MockVideoProvider
from app.runtime.orchestrator.orchestrator import NodeObserver, Orchestrator
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.runtime.workflow.engine import WorkflowEngine
from app.runtime.workflows.lesson_evaluation import evaluation_template
from app.runtime.workflows.lesson_generation import LessonWorkflowOptions, lesson_template
from app.schemas.presentation import PresentationConfig
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
from app.tools.artifacts.tools import ReadArtifactsTool, StoreArtifactsTool
from app.tools.learner.tools import LearnerSnapshotTool, LearnerSummaryTool, RecordEvaluationTool, RecordLessonTool
from app.tools.manager import ToolManager
from app.tools.audio.assets import AudioAssetLookupTool, AudioAssetTool
from app.tools.audio.timeline import PresentationTimelineTool
from app.tools.audio.tts import TTSTool, VoiceCatalogTool
from app.tools.audio.validation import AudioPlanValidationTool
from app.tools.media.tools import VideoRenderTool
from app.tools.presentation.builder import PresentationBuildTool
from app.tools.presentation.render import PresentationRenderTool
from app.tools.presentation.validation import SlidePlanValidationTool
from app.tools.rag.retrieve import ConceptMapTool, RetrievalTool
from app.tools.registry import ToolRegistry
from app.tools.research.cache import InMemoryResearchCache
from app.tools.research.rank import RankSourcesTool
from app.tools.visual.assets import ImageAssetTool
from app.tools.visual.generate import ImageGenerationTool
from app.tools.visual.search import ImageFetchTool, ImageSearchTool
from app.tools.visual.selection import ImageSelectionTool
from app.tools.visual.validation import ImageValidationTool
from app.tools.web.search import SearchTool

PRESENTATION_RENDERER_FACTORIES = {
    "pptx": lambda artifacts: PptxPresentationRenderer(media=artifacts.read_object),
    "mock": lambda artifacts: MockPresentationRenderer(),
}

TTS_PROVIDER_FACTORIES = {
    "mock": lambda settings: MockTTSProvider(),
}

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
    search_provider: SearchProvider | None = None,
    retriever: Retriever | None = None,
    image_search_provider: ImageSearchProvider | None = None,
    image_generation_provider: ImageGenerationProvider | None = None,
    presentation_renderer: PresentationRenderer | None = None,
    tts_provider: TTSProvider | None = None,
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

    retriever = retriever or LocalKnowledgeBase(settings.corpus_dir / "knowledge_base.json")
    search_provider = search_provider or MockSearchProvider(settings.corpus_dir / "web_corpus.json")
    image_search = image_search_provider or MockImageSearchProvider(settings.corpus_dir / "image_catalog.json")
    image_generation = image_generation_provider or MockImageGenerationProvider()
    renderer = presentation_renderer or PRESENTATION_RENDERER_FACTORIES[settings.presentation_renderer](artifacts)
    tts = tts_provider or TTS_PROVIDER_FACTORIES[settings.tts_provider](settings)
    registry = ToolRegistry()
    for tool in (
        SearchTool(search_provider, cache=InMemoryResearchCache() if settings.research_cache else None),
        RankSourcesTool(HeuristicRanker()),
        RetrievalTool(retriever),
        ConceptMapTool(retriever),
        LearnerSummaryTool(memory),
        LearnerSnapshotTool(memory),
        RecordLessonTool(memory),
        RecordEvaluationTool(memory),
        StoreArtifactsTool(artifacts),
        ReadArtifactsTool(artifacts),
        ImageSearchTool(image_search),
        ImageFetchTool(image_search, artifacts),
        ImageSelectionTool(),
        ImageGenerationTool(image_generation, artifacts),
        ImageValidationTool(artifacts),
        ImageAssetTool(artifacts),
        SlidePlanValidationTool(),
        PresentationBuildTool(artifacts),
        PresentationRenderTool(renderer, artifacts),
        VoiceCatalogTool(tts),
        TTSTool(tts, artifacts),
        AudioPlanValidationTool(tts),
        AudioAssetLookupTool(artifacts),
        AudioAssetTool(artifacts),
        PresentationTimelineTool(artifacts),
        VideoRenderTool(MockVideoProvider(), artifacts),
    ):
        registry.register(tool)
    tools = ToolManager(registry)

    agents = AgentRegistry()
    for agent in (RequestInterpreterAgent(), KnowledgeDiagnosticAgent(), ResearchAgent(),
                  CurriculumPlannerAgent(), TeacherAgent(), ContentReviewAgent(), VisualAgent(),
                  SlidePlannerAgent(), AudioPlannerAgent(), LearnerEvaluationAgent()):
        agents.register(agent)
    for agent_id in routing.agent_tiers:
        agents.get(agent_id)  # overrides must name real agents

    options = LessonWorkflowOptions(
        diagnostic_rounds=settings.diagnostic_max_rounds,
        memory_confidence=settings.diagnostic_memory_confidence,
        revision_policy=RevisionPolicy(max_revisions=settings.max_revisions,
                                       on_exhausted=settings.revision_exhausted_policy),
        research_requirement=settings.research_requirement,
        research_max_results=settings.research_max_results,
        research_max_sources=settings.research_max_sources,
        research_min_reliability=settings.research_min_reliability,
        visual_failure_policy=settings.visual_failure_policy,
        visual_max_per_lesson=settings.visual_max_per_lesson,
        visual_max_candidates=settings.visual_max_candidates,
        presentation_config=PresentationConfig.for_aspect(settings.presentation_aspect_ratio),
        presentation_max_slides=settings.presentation_max_slides,
        audio_failure_policy=settings.audio_failure_policy,
        audio_language=settings.audio_language,
        audio_voice=settings.audio_voice,
        audio_speaking_rate=settings.audio_speaking_rate,
        audio_format=settings.audio_format,
        audio_sample_rate=settings.audio_sample_rate,
        audio_max_words_per_segment=settings.audio_max_words_per_segment,
        audio_silent_slide_seconds=settings.audio_silent_slide_seconds,
    )
    planner = WorkflowPlanner([lesson_template(options), evaluation_template()], router, agents)
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
