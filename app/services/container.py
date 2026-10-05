"""Composition root: builds every component from settings and wires the layers together.

This is the only place that knows concrete implementations. The API and the CLI demo both use it,
so they run the exact same application services.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.agents.audio.agent import AudioPlannerAgent
from app.agents.curriculum.agent import LearningPathPlannerAgent
from app.agents.diagnostic.agent import KnowledgeDiagnosticAgent
from app.agents.evaluator.agent import LearnerEvaluationAgent
from app.agents.interpreter.agent import RequestInterpreterAgent
from app.agents.planner.agent import CurriculumPlannerAgent
from app.agents.registry import AgentRegistry
from app.agents.research.agent import ResearchAgent
from app.agents.reviewer.agent import ContentReviewAgent
from app.agents.slides.agent import SlidePlannerAgent
from app.agents.teaching.agent import TeachingSessionAgent
from app.agents.teacher.agent import TeacherAgent
from app.agents.video.agent import VideoAgent
from app.agents.visual.agent import VisualAgent
from app.artifacts.service import ArtifactService
from app.config.providers import ProviderSettings, apply_llm_overrides
from app.config.routing import ConfigError, RoutingConfig, load_routing
from app.config.settings import Settings
from app.curriculum.engine import CurriculumEngine
from app.curriculum.tracker import CurriculumTracker
from app.learner.frameworks import FrameworkRegistry, default_frameworks
from app.learner.memory import LearnerMemoryService
from app.pedagogy.strategy import StrategyRegistry
from app.observability.events import EventBus
from app.observability.logging import log_event
from app.observability.redaction import register_secret
from app.providers.core.http import HttpClient
from app.providers.core.invoker import ProviderInvoker
from app.providers.core.registry import ProviderRegistry
from app.providers.core.selector import ProviderSelector
from app.providers.image.base import ImageGenerationProvider
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.image.openai import OpenAIImageGenerationProvider
from app.providers.image_search.base import ImageSearchProvider
from app.providers.image_search.mock import MockImageSearchProvider
from app.providers.llm.anthropic import AnthropicLLMProvider
from app.providers.llm.base import LLMProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.llm.router import ModelRouter
from app.providers.managed import (
    ManagedImageGenerationProvider,
    ManagedImageSearchProvider,
    ManagedLLMProvider,
    ManagedSearchProvider,
    ManagedTTSProvider,
    ManagedVideoGenerationProvider,
)
from app.providers.presentation.base import PresentationRenderer
from app.providers.presentation.mock import MockPresentationRenderer
from app.providers.presentation.pptx_renderer import PptxPresentationRenderer
from app.providers.retrieval.base import Retriever
from app.providers.retrieval.local import LocalKnowledgeBase
from app.providers.ranking.heuristic import HeuristicRanker
from app.providers.search.base import SearchProvider
from app.providers.search.mock import MockSearchProvider
from app.providers.search.tavily import TavilySearchProvider
from app.providers.tts.base import TTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.tts.openai import OpenAITTSProvider
from app.providers.video.base import VideoComposer, VideoNormalizer, VideoProber
from app.providers.video.ffmpeg import (
    FFmpegAdapter,
    FFmpegVideoComposer,
    FFmpegVideoNormalizer,
    FFprobeVideoProber,
)
from app.providers.video.mock import MockVideoComposer, MockVideoNormalizer, MockVideoProber
from app.providers.video_generation.base import VideoGenerationProvider
from app.providers.video_generation.minimax import DEFAULT_BASE_URL as MINIMAX_BASE_URL
from app.providers.video_generation.minimax import MiniMaxVideoGenerationProvider
from app.providers.video_generation.mock import MockVideoGenerationProvider
from app.runtime.interaction.teacher import TeachingRuntime
from app.runtime.orchestrator.orchestrator import NodeObserver, Orchestrator
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.runtime.workflow.engine import WorkflowEngine
from app.runtime.workflows.curriculum_planning import curriculum_template
from app.runtime.workflows.lesson_evaluation import evaluation_template
from app.runtime.workflows.lesson_generation import LessonWorkflowOptions, lesson_template
from app.schemas.presentation import PresentationConfig
from app.schemas.providers import Capability
from app.schemas.workflow import RevisionPolicy
from app.services.catalog import CatalogService
from app.services.curriculum import CurriculumService
from app.services.learners import LearnerService
from app.services.production import ProductionService
from app.services.tasks import TaskService
from app.services.teaching import TeachingSessionService
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import (
    SqlArtifactRepository,
    SqlCurriculumRepository,
    SqlEventRepository,
    SqlEvidenceRepository,
    SqlGoalRepository,
    SqlLearnerRepository,
    SqlLearningEventRepository,
    SqlTaskRepository,
    SqlTeachingRepository,
)
from app.tools.artifacts.tools import ReadArtifactsTool, StoreArtifactsTool
from app.tools.curriculum.tools import (
    CurriculumDraftTool,
    CurriculumFinalizeTool,
    CurriculumSaveTool,
    CurriculumTrackTool,
)
from app.tools.knowledge.base import KnowledgeConceptsTool, RetrieverKnowledgeBase
from app.tools.learner.tools import (
    LearnerModelTool,
    LearnerSnapshotTool,
    LearnerSummaryTool,
    LearningGoalTool,
    RecordDiagnosticTool,
    RecordEvaluationTool,
    RecordLessonTool,
)
from app.tools.pedagogy.tools import FeedbackTool, GapAnalysisTool, PlanningTool, RecommendationTool
from app.tools.manager import ToolManager
from app.tools.audio.assets import AudioAssetLookupTool, AudioAssetTool
from app.tools.audio.timeline import PresentationTimelineTool
from app.tools.audio.tts import TTSTool, VoiceCatalogTool
from app.tools.audio.validation import AudioPlanValidationTool
from app.tools.presentation.builder import PresentationBuildTool
from app.tools.presentation.render import PresentationRenderTool
from app.tools.presentation.validation import SlidePlanValidationTool
from app.tools.rag.retrieve import ConceptMapTool, RetrievalTool
from app.tools.registry import ToolRegistry
from app.tools.video.generation import GeneratedVideoService, generation_tools
from app.tools.video.service import VideoService
from app.tools.video.tools import VideoArtifactTool, VideoComposeTool, VideoValidateTool
from app.tools.video.validation import VideoPlanValidationTool
from app.tools.research.cache import InMemoryResearchCache
from app.tools.research.rank import RankSourcesTool
from app.tools.teaching.grounding import TeachingGroundingTool
from app.tools.visual.assets import ImageAssetTool
from app.tools.visual.generate import ImageGenerationTool
from app.tools.visual.search import ImageFetchTool, ImageSearchTool
from app.tools.visual.selection import ImageSelectionTool
from app.tools.visual.validation import ImageValidationTool
from app.tools.visual.video_strategy import VideoStrategyTool
from app.tools.web.search import SearchTool
from app.utils.workspace import ScratchSpace

PRESENTATION_RENDERER_FACTORIES = {
    "pptx": lambda artifacts: PptxPresentationRenderer(media=artifacts.read_object),
    "mock": lambda artifacts: MockPresentationRenderer(),
}

def _ffmpeg(settings: Settings) -> FFmpegAdapter:
    return FFmpegAdapter(ffmpeg=settings.ffmpeg_path, ffprobe=settings.ffprobe_path,
                         timeout_seconds=settings.video_timeout_seconds)


VIDEO_COMPOSER_FACTORIES = {  # (composer, prober): a composer is always validated by the prober that can read it
    "ffmpeg": lambda settings, artifacts: (
        FFmpegVideoComposer(media=artifacts.read_object, adapter=_ffmpeg(settings),
                            font_path=settings.video_font_path),
        FFprobeVideoProber(_ffmpeg(settings))),
    "mock": lambda settings, artifacts: (MockVideoComposer(media=artifacts.read_object), MockVideoProber()),
}
# Generated clips are normalised by the same toolchain that composes and probes the video.
VIDEO_NORMALIZER_FACTORIES = {
    "ffmpeg": lambda settings: FFmpegVideoNormalizer(_ffmpeg(settings)),
    "mock": lambda settings: MockVideoNormalizer(),
}

# --- Provider factories: the only place that knows concrete provider classes ------------------------------------
# Each takes the application Settings; network providers get an HttpClient carrying their credential, the
# capability's timeout and the request limits. Which ones are built is decided by the provider configuration.

DEFAULT_BASE_URLS = {"tavily": "https://api.tavily.com", "minimax": MINIMAX_BASE_URL}


def _http(settings: Settings, capability: Capability, provider: str, base_url: str, headers: dict[str, str],
          secret: str, request_id_header: str = "X-Request-Id") -> HttpClient:
    p = settings.providers
    return HttpClient(provider=provider, base_url=base_url, headers=headers, secrets=(secret,),
                      timeout_seconds=p.policies()[capability].timeout_seconds, offline=p.offline,
                      max_request_bytes=p.provider_max_request_bytes, max_response_bytes=p.provider_max_response_bytes,
                      request_id_header=request_id_header)


def _credential(settings: Settings, capability: Capability, provider: str) -> str:
    secret, _ = settings.providers.credential(capability, provider)
    if secret is None:  # validate_runtime reports this first, with the variable names
        raise ConfigError(f"no credential configured for {capability.value} provider '{provider}'")
    return secret


def _openai_headers(p: ProviderSettings, key: str) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {key}"}
    if p.openai_organization:
        headers["OpenAI-Organization"] = p.openai_organization
    return headers


def _openai_llm(settings: Settings) -> LLMProvider:
    p, key = settings.providers, _credential(settings, Capability.LLM, "openai")
    http = _http(settings, Capability.LLM, "openai", p.openai_base_url, _openai_headers(p, key), key,
                 request_id_header="X-Client-Request-Id")
    return OpenAICompatibleLLMProvider(http, native_structured_output=p.llm_native_structured_output,
                                       max_tokens_param=p.openai_max_tokens_param)


def _anthropic_llm(settings: Settings) -> LLMProvider:
    p, key = settings.providers, _credential(settings, Capability.LLM, "anthropic")
    http = _http(settings, Capability.LLM, "anthropic", p.anthropic_base_url,
                 {"x-api-key": key, "anthropic-version": "2023-06-01"}, key, request_id_header="")
    return AnthropicLLMProvider(http, native_structured_output=p.llm_native_structured_output)


def _openai_tts(settings: Settings) -> TTSProvider:
    p, key = settings.providers, _credential(settings, Capability.TTS, "openai")
    http = _http(settings, Capability.TTS, "openai", p.tts_base_url or p.openai_base_url, _openai_headers(p, key),
                 key, request_id_header="X-Client-Request-Id")
    return OpenAITTSProvider(http, model=p.tts_model, languages=p.languages())


def _openai_image(settings: Settings) -> ImageGenerationProvider:
    p, key = settings.providers, _credential(settings, Capability.IMAGE, "openai")
    http = _http(settings, Capability.IMAGE, "openai", p.image_base_url or p.openai_base_url,
                 _openai_headers(p, key), key, request_id_header="X-Client-Request-Id")
    return OpenAIImageGenerationProvider(http, model=p.image_model)


def _tavily_search(settings: Settings) -> SearchProvider:
    p, key = settings.providers, _credential(settings, Capability.SEARCH, "tavily")
    http = _http(settings, Capability.SEARCH, "tavily", p.search_base_url or DEFAULT_BASE_URLS["tavily"],
                 {"Authorization": f"Bearer {key}"}, key)
    return TavilySearchProvider(http, search_depth=p.search_depth, include_raw_content=p.search_include_raw_content)


def _minimax_video(settings: Settings) -> VideoGenerationProvider:
    p, key = settings.providers, _credential(settings, Capability.VIDEO_GENERATION, "minimax")
    http = _http(settings, Capability.VIDEO_GENERATION, "minimax",
                 p.video_generation_base_url or DEFAULT_BASE_URLS["minimax"], {"Authorization": f"Bearer {key}"}, key,
                 request_id_header="")
    return MiniMaxVideoGenerationProvider(http, model=p.video_generation_model,
                                          max_download_bytes=p.video_generation_max_download_bytes)


LLM_PROVIDER_FACTORIES = {
    "mock": lambda settings: MockLLMProvider(default_responders()),
    "openai": _openai_llm,
    "anthropic": _anthropic_llm,
}
PROVIDER_FACTORIES = {
    Capability.TTS: {"mock": lambda settings: MockTTSProvider(), "openai": _openai_tts},
    Capability.IMAGE: {"mock": lambda settings: MockImageGenerationProvider(), "openai": _openai_image},
    Capability.IMAGE_SEARCH: {
        "mock": lambda settings: MockImageSearchProvider(settings.corpus_dir / "image_catalog.json")},
    Capability.SEARCH: {
        "mock": lambda settings: MockSearchProvider(settings.corpus_dir / "web_corpus.json"), "tavily": _tavily_search},
    Capability.VIDEO_GENERATION: {"mock": lambda settings: MockVideoGenerationProvider(), "minimax": _minimax_video},
}


@dataclass
class ProviderStack:
    """The provider layer as the rest of the application sees it: managed providers per capability, plus the
    registry, selector and invoker behind them."""

    registry: ProviderRegistry
    selector: ProviderSelector
    invoker: ProviderInvoker
    routing: RoutingConfig
    llm: dict[str, ManagedLLMProvider]
    tts: ManagedTTSProvider
    image_generation: ManagedImageGenerationProvider
    image_search: ManagedImageSearchProvider
    search: ManagedSearchProvider
    video_generation: ManagedVideoGenerationProvider


def build_llm_providers(settings: Settings, routing: RoutingConfig) -> dict[str, LLMProvider]:
    """One LLM provider per provider id the routing uses."""
    providers = {}
    for name in routing.providers():
        factory = LLM_PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ConfigError(f"LLM provider '{name}' has no adapter (available: {sorted(LLM_PROVIDER_FACTORIES)})")
        providers[name] = factory(settings)
    return providers


def build_providers(settings: Settings, events: EventBus, *, llm_providers: dict[str, LLMProvider] | None = None,
                    injected: dict[Capability, object] | None = None) -> ProviderStack:
    """Build, register and select every provider. `llm_providers` and `injected` replace configured providers
    (tests and demos); they are still registered and run under the invoker like any other."""
    p = settings.providers
    for secret in p.secret_values():
        register_secret(secret)
    routing = apply_llm_overrides(load_routing(settings.routing_file), p)
    invoker = ProviderInvoker(policies=p.policies(), events=events, offline=p.offline)
    registry = ProviderRegistry(offline=p.offline)

    raw_llm = llm_providers if llm_providers is not None else build_llm_providers(settings, routing)
    for provider in raw_llm.values():
        registry.register(provider, capability=Capability.LLM)
    llm = {name: ManagedLLMProvider(provider, invoker, cost=routing.cost) for name, provider in raw_llm.items()}

    chains: dict[Capability, list[str]] = {}
    injected = injected or {}
    for capability, factories in PROVIDER_FACTORIES.items():
        given = injected.get(capability)
        if given is not None:
            registry.register(given, capability=capability, default=True)
            chains[capability] = [given.name]
            continue
        chain = p.chain(capability)
        for provider_id in chain:
            factory = factories.get(provider_id)
            if factory is None:
                raise ConfigError(f"{capability.value} provider '{provider_id}' has no adapter "
                                  f"(available: {sorted(factories)})")
            registry.register(factory(settings), capability=capability)
        chains[capability] = chain
    selector = ProviderSelector(registry, chains=chains, routing=routing,
                                models={cap: p.model(cap) for cap in Capability})
    return ProviderStack(
        registry=registry, selector=selector, invoker=invoker, routing=routing, llm=llm,
        tts=ManagedTTSProvider(selector.chain(Capability.TTS), invoker),
        image_generation=ManagedImageGenerationProvider(selector.chain(Capability.IMAGE), invoker),
        image_search=ManagedImageSearchProvider(selector.chain(Capability.IMAGE_SEARCH)[0], invoker),
        search=ManagedSearchProvider(selector.chain(Capability.SEARCH), invoker),
        video_generation=ManagedVideoGenerationProvider(selector.chain(Capability.VIDEO_GENERATION), invoker,
                                                        price_per_second=p.video_generation_price_per_second),
    )


@dataclass
class Container:
    settings: Settings
    events: EventBus
    llm_providers: dict[str, LLMProvider]
    router: ModelRouter
    providers: ProviderStack
    tools: ToolManager
    agents: AgentRegistry
    frameworks: FrameworkRegistry
    memory: LearnerMemoryService
    artifacts: ArtifactService
    orchestrator: Orchestrator
    task_service: TaskService
    learner_service: LearnerService
    curriculum_service: CurriculumService
    teaching_service: TeachingSessionService
    catalog: CatalogService
    production: ProductionService
    _sessions: object

    def close(self) -> None:
        dispose(self._sessions)


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
    video_composer: VideoComposer | None = None,
    video_prober: VideoProber | None = None,
    video_generation_provider: VideoGenerationProvider | None = None,
    video_normalizer: VideoNormalizer | None = None,
) -> Container:
    settings = settings or Settings()
    settings.validate_runtime()

    sessions = create_db(settings.resolved_database_url)
    task_repo = SqlTaskRepository(sessions)
    event_repo = SqlEventRepository(sessions)
    events = EventBus()
    events.subscribe(event_repo.append)
    events.subscribe(log_event)

    injected = {Capability.TTS: tts_provider, Capability.IMAGE: image_generation_provider,
                Capability.IMAGE_SEARCH: image_search_provider, Capability.SEARCH: search_provider,
                Capability.VIDEO_GENERATION: video_generation_provider}
    providers = build_providers(settings, events, llm_providers=llm_providers,
                                injected={k: v for k, v in injected.items() if v is not None})
    routing = providers.routing
    router = ModelRouter(routing, providers.llm)

    frameworks = default_frameworks()
    pedagogy = settings.pedagogy_config()
    strategies = StrategyRegistry()
    memory = LearnerMemoryService(SqlLearnerRepository(sessions), frameworks,
                                  evidence=SqlEvidenceRepository(sessions), events=SqlLearningEventRepository(sessions),
                                  goals=SqlGoalRepository(sessions), config=pedagogy)
    artifacts = ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(settings.resolved_object_store_dir))
    curriculum = CurriculumEngine(SqlCurriculumRepository(sessions), settings.curriculum_config())

    retriever = retriever or LocalKnowledgeBase(settings.corpus_dir / "knowledge_base.json")
    search = providers.search
    image_search = providers.image_search
    image_generation = providers.image_generation
    renderer = presentation_renderer or PRESENTATION_RENDERER_FACTORIES[settings.presentation_renderer](artifacts)
    tts = providers.tts
    default_composer, default_prober = VIDEO_COMPOSER_FACTORIES[settings.video_composer](settings, artifacts)
    scratch = ScratchSpace(settings.resolved_video_work_dir, keep_failed=settings.video_keep_failed_work)
    video = VideoService(artifacts, video_composer or default_composer, video_prober or default_prober, scratch)
    generated_video = GeneratedVideoService(
        artifacts, providers.video_generation, video_prober or default_prober,
        video_normalizer or VIDEO_NORMALIZER_FACTORIES[settings.video_composer](settings), scratch,
        settings.generated_video_config())
    registry = ToolRegistry()
    for tool in (
        SearchTool(search, cache=InMemoryResearchCache() if settings.research_cache else None,
                   include_domains=settings.providers.include_domains(),
                   exclude_domains=settings.providers.exclude_domains()),
        RankSourcesTool(HeuristicRanker()),
        RetrievalTool(retriever),
        ConceptMapTool(retriever),
        TeachingGroundingTool(retriever),
        LearnerSummaryTool(memory),
        LearnerSnapshotTool(memory),
        RecordLessonTool(memory),
        RecordEvaluationTool(memory),
        RecordDiagnosticTool(memory),
        LearnerModelTool(memory),
        LearningGoalTool(memory),
        KnowledgeConceptsTool(RetrieverKnowledgeBase(retriever)),
        GapAnalysisTool(pedagogy),
        PlanningTool(pedagogy, strategies),
        RecommendationTool(pedagogy, strategies),
        FeedbackTool(pedagogy),
        CurriculumDraftTool(curriculum),
        CurriculumFinalizeTool(curriculum),
        CurriculumSaveTool(curriculum),
        CurriculumTrackTool(CurriculumTracker(memory, curriculum), artifacts),
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
        VideoPlanValidationTool(),
        VideoComposeTool(video),
        VideoValidateTool(video),
        VideoArtifactTool(video),
        VideoStrategyTool(limits=generated_video.limits),
        *generation_tools(generated_video),
    ):
        registry.register(tool)
    tools = ToolManager(registry)

    agents = AgentRegistry()
    for agent in (RequestInterpreterAgent(), KnowledgeDiagnosticAgent(), ResearchAgent(),
                  CurriculumPlannerAgent(), TeacherAgent(), ContentReviewAgent(), VisualAgent(),
                  SlidePlannerAgent(), AudioPlannerAgent(), VideoAgent(), LearnerEvaluationAgent(),
                  LearningPathPlannerAgent(), TeachingSessionAgent()):
        agents.register(agent)
    for agent_id in [*routing.agent_tiers, *routing.routes]:
        agents.get(agent_id)  # overrides and routes must name real agents

    options = LessonWorkflowOptions(
        diagnostic_rounds=settings.diagnostic_max_rounds,
        memory_confidence=settings.diagnostic_memory_confidence,
        questioning=pedagogy.questioning,
        lesson_minutes=settings.pedagogy_lesson_minutes,
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
        video_failure_policy=settings.video_failure_policy,
        video_config=settings.video_config(),
        generated_video_enabled=settings.generated_video_enabled,
        generated_video=settings.generated_video_config(),
    )
    planner = WorkflowPlanner([lesson_template(options), evaluation_template(), curriculum_template()], router, agents)
    orchestrator = Orchestrator(tasks=task_repo, engine=WorkflowEngine(agents, tools, router), planner=planner,
                                agents=agents, tools=tools, router=router, events=events, observers=observers)
    task_service = TaskService(orchestrator, task_repo, event_repo, artifacts)
    learner_service = LearnerService(memory, RetrieverKnowledgeBase(retriever), pedagogy, strategies)
    curriculum_service = CurriculumService(memory, RetrieverKnowledgeBase(retriever), curriculum, task_service, events)
    teaching_service = TeachingSessionService(
        SqlTeachingRepository(sessions), TeachingRuntime(agents, tools, router, events), artifacts=artifacts,
        memory=memory, knowledge=RetrieverKnowledgeBase(retriever), curriculum=curriculum_service,
        tasks=task_service, events=events, config=settings.teaching_config())
    return Container(
        settings=settings, events=events, llm_providers=providers.llm, router=router, providers=providers,
        tools=tools, agents=agents,
        frameworks=frameworks, memory=memory, artifacts=artifacts, orchestrator=orchestrator,
        task_service=task_service, learner_service=learner_service, curriculum_service=curriculum_service,
        teaching_service=teaching_service,
        catalog=CatalogService(agents, registry, router, planner, frameworks, providers.registry, providers.selector),
        production=ProductionService(settings=settings, events=events, registry=providers.registry,
                                     selector=providers.selector, router=router, agents=agents, tools=tools,
                                     planner=planner, frameworks=frameworks, tasks=task_service,
                                     learners=learner_service),
        _sessions=sessions,
    )
