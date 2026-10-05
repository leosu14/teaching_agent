"""Production runs: a real-provider lesson, from a structured user task to the VIDEO artifact, with explicit
configuration, budget, health checks, resume, idempotency and a machine-readable report.

This service adds no educational logic and no second pipeline: it creates an ordinary lesson task (the same
`lesson_generation` workflow, agents and tools as every other entry point) and drives it through the TaskService.
What it adds is around the run:

- preflight (`plan`): resolves the request, providers, models, budget and workflow stages; no provider call;
- readiness: production mode and a real provider for every capability the workflow uses;
- health: a cheap check of the required providers only, before anything is spent;
- identity (`run_key`): a deterministic hash of the task input, provider/model configuration and workflow options.
  An identical completed run is reused; an identical interrupted or failed run is repaired and resumed;
- repair: artifacts whose stored bytes are missing or corrupt send the nodes that made them (and everything
  downstream) back to run again; nothing else is regenerated;
- the report (`report`): artifact graph, node/request trace, usage, cost, warnings, errors and the final video.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.agents.registry import AgentRegistry, UnknownAgent
from app.agents.teaching.agent import AGENT_ID as TEACHING_AGENT
from app.config.production import production_readiness, required_capabilities
from app.config.routing import ConfigError
from app.config.settings import Settings
from app.learner.frameworks import FrameworkRegistry, UnknownFramework
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger, task_usage
from app.providers.core.registry import ProviderRegistry
from app.providers.core.selector import ProviderSelector
from app.providers.llm.router import ModelRouter
from app.runtime.orchestrator.planner import WorkflowPlanner
from app.runtime.workflows.curriculum_planning import CAPABILITY as CURRICULUM_CAPABILITY
from app.runtime.workflows.curriculum_planning import WORKFLOW_ID as CURRICULUM_WORKFLOW
from app.runtime.workflows.lesson_evaluation import LESSON_TASK
from app.runtime.workflows.lesson_generation import GENERATED_VIDEO_CAPABILITY
from app.runtime.workflows.lesson_generation import WORKFLOW_ID as LESSON_WORKFLOW
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.common import utcnow
from app.schemas.events import EventType
from app.schemas.learner import LearnerPreferences, LearnerProfileInput, SubjectState
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LessonRequest
from app.schemas.production import (
    PRODUCTION,
    REQUIRED_CHAIN,
    RUN_KEY,
    ArtifactNode,
    FinalVideo,
    GraphCheck,
    NodeTrace,
    InteractiveTeachingPlan,
    ProductionPlan,
    ProductionReport,
    ProductionTask,
    ProviderChoice,
    VideoGenerationPlan,
)
from app.schemas.providers import Capability, HealthStatus
from app.schemas.task import Task, TaskStatus
from app.schemas.usage import BUDGET_KEY, TaskBudget
from app.services.learners import LearnerService
from app.services.tasks import TaskService
from app.tools.base import ToolCaller
from app.tools.manager import ToolManager
from app.tools.rag.retrieve import ConceptMapOutput

RUN_KEY_VERSION = 1  # bump when the workflow changes what a run produces from the same inputs
# Subject of a language level framework (CEFR) when none is given: the language itself, by its English name.
LANGUAGE_SUBJECTS = {
    "ar": "arabic", "de": "german", "en": "english", "es": "spanish", "fr": "french", "hi": "hindi",
    "it": "italian", "ja": "japanese", "ko": "korean", "nl": "dutch", "pl": "polish", "pt": "portuguese",
    "ru": "russian", "sv": "swedish", "tr": "turkish", "zh": "chinese",
}
LANGUAGE_FRAMEWORKS = {"cefr"}
LESSON_CAPABILITIES = ["lesson.text", "lesson.review", "lesson.visuals", "slides.plan", "presentation.pptx",
                       "audio.narration", "presentation.timeline", "video.mp4"]
# Settings that change what the lesson workflow produces (part of a run's identity). Paths, logging and the
# database are not.
WORKFLOW_SETTINGS = {
    "max_revisions", "revision_exhausted_policy", "diagnostic_max_rounds", "diagnostic_memory_confidence",
    "research_requirement", "research_max_results", "research_max_sources", "research_min_reliability",
    "visual_failure_policy", "visual_max_per_lesson", "visual_max_candidates", "presentation_aspect_ratio",
    "presentation_max_slides", "presentation_renderer", "audio_failure_policy", "audio_language", "audio_voice",
    "audio_speaking_rate", "audio_format", "audio_sample_rate", "audio_max_words_per_segment",
    "audio_silent_slide_seconds", "video_failure_policy", "video_composer",
}

DiagnosticAnswerer = Callable[[DiagnosticQuestionSheet], Awaitable[DiagnosticAnswers]]


# The long-term learning loop a production lesson is part of (goals are optional: without one, a lesson is planned
# from the adaptive gaps alone).
LEARNING_LOOP = ("Goal", "Curriculum", "Next Action", "Lesson", "Evaluation", "Mastery Update")
# The opt-in interactive layer on a finished lesson (app/services/teaching.py).
INTERACTIVE_LOOP = ("Lesson", "Interactive Teaching Session", "Teacher Turn", "Learner Turn", "Adaptive Response",
                    "Evidence", "Evaluation", "Mastery")
INTERACTIVE_ENDPOINTS = ("POST /lessons/{lesson_id}/teaching-session", "GET /teaching-sessions/{session_id}",
                         "POST /teaching-sessions/{session_id}/answers", "POST /teaching-sessions/{session_id}/pause",
                         "POST /teaching-sessions/{session_id}/resume", "POST /teaching-sessions/{session_id}/cancel")


@dataclass
class RunOutcome:
    task: Task
    reused: bool = False  # an identical completed run was reused
    resumed: bool = False  # an identical interrupted or failed run was continued
    repaired_nodes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ProductionService:
    def __init__(self, *, settings: Settings, events: EventBus, registry: ProviderRegistry,
                 selector: ProviderSelector, router: ModelRouter, agents: AgentRegistry, tools: ToolManager,
                 planner: WorkflowPlanner, frameworks: FrameworkRegistry, tasks: TaskService,
                 learners: LearnerService) -> None:
        self._settings = settings
        self._events = events
        self._registry = registry
        self._selector = selector
        self._router = router
        self._agents = agents
        self._tools = tools
        self._planner = planner
        self._frameworks = frameworks
        self._tasks = tasks
        self._learners = learners

    # --- preflight (no provider calls) -------------------------------------------------------------------------

    def budget(self) -> TaskBudget:
        return self._settings.production.budget()

    def lesson_request(self, task: ProductionTask) -> LessonRequest:
        framework_id = self._framework_for(task.level)
        subject = task.subject
        if not subject:
            base = task.language.split("-")[0].lower()
            if framework_id not in LANGUAGE_FRAMEWORKS or base not in LANGUAGE_SUBJECTS:
                raise ConfigError(f"cannot tell the subject of a {task.level} lesson in '{task.language}': pass "
                                  "--subject")
            subject = LANGUAGE_SUBJECTS[base]
        return LessonRequest(
            raw_request=f"Create a {task.level} {subject} lesson about {task.topic}.", subject=subject,
            topic=task.topic, framework_id=framework_id, target_level=task.level,
            language_of_instruction=task.language,
            capabilities=[*LESSON_CAPABILITIES, *([GENERATED_VIDEO_CAPABILITY] if task.generated_video else [])],
        )

    def video_generation_plan(self, task: ProductionTask, budget: TaskBudget) -> VideoGenerationPlan:
        """The generated-video part of a plan: provider, limits, an upper-bound estimate and the fallback policy.
        Nothing is called: the price is the configured one (VIDEO_GENERATION_PRICE_PER_SECOND), never guessed."""
        config = self._settings.generated_video_config()
        selection = self._selector.select(Capability.VIDEO_GENERATION)
        provider = self._registry.get(Capability.VIDEO_GENERATION, selection.provider)
        segments = budget.max_generated_video_segments or 0
        seconds = min(budget.max_generated_video_seconds or 0.0, segments * config.max_segment_seconds)
        enabled = (task.generated_video and self._settings.generated_video_enabled and segments > 0
                   and seconds >= config.min_segment_seconds)
        price = config.price_per_second_usd if provider.requires_network else 0.0
        return VideoGenerationPlan(
            requested=task.generated_video, enabled=enabled, provider=selection.provider, model=selection.model,
            real=provider.requires_network, max_segments=segments, max_seconds=budget.max_generated_video_seconds or 0.0,
            segment_seconds=(config.min_segment_seconds, config.max_segment_seconds),
            estimated_seconds=round(seconds if enabled else 0.0, 3),
            estimated_cost_usd=None if price is None else round((seconds if enabled else 0.0) * price, 4),
            max_cost_usd=budget.max_video_generation_cost_usd, required=config.required,
            failure_policy=config.failure_policy)

    def _framework_for(self, level: str) -> str:
        for framework_id in self._frameworks.ids():
            try:
                self._frameworks.get(framework_id).index(level)
            except (ValueError, UnknownFramework):
                continue
            return framework_id
        raise ConfigError(f"level '{level}' is not a level of any known framework ({self._frameworks.ids()})")

    def provider_choices(self, required: set[Capability]) -> list[ProviderChoice]:
        choices = []
        for capability in Capability:
            selection = self._selector.select(capability)
            provider = self._registry.get(capability, selection.provider)
            choices.append(ProviderChoice(capability=capability.value, provider=selection.provider,
                                          model=selection.model, fallbacks=list(selection.fallbacks),
                                          required=capability in required, real=provider.requires_network))
        return choices

    def llm_routes(self) -> dict[str, str]:
        """agent id -> the provider/model chain it is routed to."""
        routes = {}
        for agent in self._agents.describe():
            targets = self._router.targets(self._router.tier_for(agent.id, agent.tier), agent.id)
            routes[agent.id] = " -> ".join(f"{t.provider}/{t.model}" for t in targets)
        return dict(sorted(routes.items()))

    def run_key(self, request: LessonRequest, learner_id: str) -> str:
        """Deterministic identity of a run: the same task input, provider and model configuration and workflow
        options give the same key; any change gives a new one."""
        body = {
            "version": RUN_KEY_VERSION, "workflow": LESSON_WORKFLOW, "learner_id": learner_id,
            "request": request.model_dump(mode="json", exclude={"raw_request"}),
            "providers": [c.model_dump(mode="json", exclude={"required", "real"})
                          for c in self.provider_choices(set())],
            "llm_routes": self.llm_routes(),
            "options": self._settings.model_dump(mode="json", include=WORKFLOW_SETTINGS),
            "video": self._settings.video_config().model_dump(mode="json"),
        }
        if GENERATED_VIDEO_CAPABILITY in request.capabilities:  # other runs keep their earlier keys
            body["generated_video"] = {"enabled": self._settings.generated_video_enabled,
                                       "config": self._settings.generated_video_config().model_dump(mode="json")}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return f"run_{digest[:32]}"

    def _estimate(self) -> float | None:
        """The planner's LLM estimate, or None when any routed model has no configured price (never guessed)."""
        template = self._planner.template(LESSON_WORKFLOW)
        unpriced = set(self._router.config.unpriced_models)
        for call in template.expected_calls:
            tier = self._router.tier_for(call.agent_id, self._agents.get(call.agent_id).spec.tier)
            if any(t.model in unpriced or t.model not in self._router.config.pricing
                   for t in self._router.targets(tier, call.agent_id)):
                return None
        return self._planner.estimate(template)

    async def plan(self, task: ProductionTask, *, require_production: bool = True) -> ProductionPlan:
        """Everything a run would do, checked before any provider call. `require_production=False` plans a
        rehearsal on whatever providers are configured (the mocks, offline) without the production rules."""
        problems: list[str] = []
        request = self.lesson_request(task)  # ConfigError when the level or subject cannot be resolved
        template = self._planner.template(LESSON_WORKFLOW)
        budget = self.budget()
        video_generation = self.video_generation_plan(task, budget)
        required = required_capabilities(template.provider_capabilities, budget,
                                         generated_video=video_generation.enabled)
        readiness_problems, warnings = production_readiness(self._settings.providers, required)
        if require_production:
            problems += readiness_problems
        else:
            problems += self._settings.providers.problems()
            warnings = []
        concepts = await self._concepts(request)
        if concepts == 0:
            problems.append(f"the knowledge base ({self._settings.corpus_dir / 'knowledge_base.json'}) has no concepts "
                            f"for subject '{request.subject}' and topic '{request.topic}': the diagnostic needs them "
                            "(add concept documents, or set TA_CORPUS_DIR to a knowledge base that has them)")
        definition = template.build(request)
        return ProductionPlan(
            mode=self._settings.providers.mode, ready=not problems, problems=problems, warnings=warnings, task=task,
            lesson_request=request, workflow_id=template.id, stages=list(definition.all_nodes),
            providers=self.provider_choices(required), llm_routes=self.llm_routes(),
            required_capabilities=sorted(c.value for c in required), budget=budget,
            estimated_llm_cost_usd=self._estimate(), knowledge_concepts=concepts,
            video_generation=video_generation, run_key=self.run_key(request, task.learner_id),
            learning_loop=list(LEARNING_LOOP), curriculum_stages=self._curriculum_stages(request),
            interactive_teaching=self._interactive_teaching(),
        )

    def _interactive_teaching(self) -> InteractiveTeachingPlan:
        """Whether interactive sessions can run with this configuration (checked, never run: no provider call)."""
        try:
            self._agents.get(TEACHING_AGENT)
            available = True
        except UnknownAgent:
            available = False
        return InteractiveTeachingPlan(
            available=available, loop=list(INTERACTIVE_LOOP), agent_id=TEACHING_AGENT,
            llm_route=self.llm_routes().get(TEACHING_AGENT, "-"), endpoints=list(INTERACTIVE_ENDPOINTS),
            policy=self._settings.teaching_config().model_dump())

    def _curriculum_stages(self, request: LessonRequest) -> list[str]:
        """The curriculum-planning workflow's nodes for the request's subject (built, never run)."""
        template = self._planner.template(CURRICULUM_WORKFLOW)
        return list(template.build(request.model_copy(update={"capabilities": [CURRICULUM_CAPABILITY]})).all_nodes)

    async def _concepts(self, request: LessonRequest) -> int:
        """How many concepts the (local) knowledge base holds for the request: a local lookup, no provider."""
        caller = ToolCaller(caller_id="production.preflight", allowed_tools=frozenset({"rag.concept_map"}),
                            permissions=frozenset({"knowledge:read"}))
        scope = ExecutionScope(events=self._events, usage=UsageLedger())
        found = await self._tools.call(caller, "rag.concept_map", {
            "subject": request.subject, "topic": request.topic, "level": request.target_level}, scope)
        assert isinstance(found, ConceptMapOutput)
        return len(found.concepts)

    async def check_health(self, required: list[str]) -> list[HealthStatus]:
        """Health of the providers serving the required capabilities only (primary and configured fallbacks)."""
        statuses: list[HealthStatus] = []
        timeout = self._settings.production.production_health_timeout_seconds
        for capability in sorted(required):
            statuses += await self._registry.check_health(Capability(capability), timeout_seconds=timeout)
        return statuses

    # --- execution --------------------------------------------------------------------------------------------

    async def run(self, plan: ProductionPlan, *, answer: DiagnosticAnswerer, fresh: bool = False) -> RunOutcome:
        """Run the planned lesson: reuse an identical completed run, continue an identical unfinished one, or start
        a new task. Diagnostic questions are answered through `answer`."""
        if not fresh:
            previous = self._previous(plan)
            if previous is not None and previous.status == TaskStatus.COMPLETED:
                invalid = self._invalid_artifacts(previous)
                if not invalid:
                    return RunOutcome(task=previous, reused=True)
                warning = (f"the identical earlier run {previous.task_id} has unusable artifacts "
                           f"({', '.join(sorted(invalid))}); a completed task cannot be repaired, so a new run starts")
                outcome = await self._start(plan, answer)
                outcome.warnings.append(warning)
                return outcome
            if previous is not None and previous.status in (TaskStatus.FAILED, TaskStatus.PAUSED, TaskStatus.RUNNING,
                                                            TaskStatus.REVIEWING, TaskStatus.WAITING):
                return await self.resume(previous.task_id, answer=answer)
        return await self._start(plan, answer)

    async def _start(self, plan: ProductionPlan, answer: DiagnosticAnswerer) -> RunOutcome:
        self._ensure_learner(plan)
        task = self._tasks.create_lesson(
            lesson_request=plan.lesson_request, learner_id=plan.task.learner_id, user_id=plan.task.user_id,
            metadata={RUN_KEY: plan.run_key, BUDGET_KEY: plan.budget.model_dump(mode="json"),
                      PRODUCTION: {"mode": plan.mode, "providers": [p.model_dump(mode="json") for p in plan.providers],
                                   "llm_routes": plan.llm_routes}})
        task = await self._drive(await self._tasks.run(task.task_id), answer)
        return RunOutcome(task=task)

    async def resume(self, task_id: str, *, answer: DiagnosticAnswerer) -> RunOutcome:
        """Continue an interrupted, paused or failed production task from its last checkpoint. Completed nodes are
        not run again unless an artifact they made is missing or corrupt; the current budget applies."""
        task = self._tasks.get(task_id)
        task.metadata[BUDGET_KEY] = self.budget().model_dump(mode="json")
        self._tasks.save(task)
        invalid = self._invalid_artifacts(task)
        repaired: list[str] = []
        if invalid:
            makers = self._makers(task_id)
            repaired = self._tasks.invalidate(task_id, sorted({makers[a] for a in invalid if a in makers}))
        if task.status == TaskStatus.WAITING:
            task = await self._drive(task, answer)
        else:
            task = await self._drive(await self._tasks.resume(task_id), answer)
        warnings = [f"repaired: artifacts {', '.join(sorted(invalid))} were unusable; re-ran {', '.join(repaired)}"
                    ] if invalid else []
        return RunOutcome(task=task, resumed=True, repaired_nodes=repaired, warnings=warnings)

    async def _drive(self, task: Task, answer: DiagnosticAnswerer) -> Task:
        while task.status == TaskStatus.WAITING:
            assert task.waiting is not None
            if task.waiting.kind != "diagnostic_answers":
                break
            sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
            task = await self._tasks.submit_assessment(task.task_id, await answer(sheet))
        return task

    def _previous(self, plan: ProductionPlan) -> Task | None:
        matches = [t for t in self._tasks.list_for_learner(plan.task.learner_id)
                   if t.metadata.get(RUN_KEY) == plan.run_key and t.status != TaskStatus.CANCELLED]
        return matches[-1] if matches else None

    def _ensure_learner(self, plan: ProductionPlan) -> None:
        learner_id = plan.task.learner_id
        try:
            self._learners.get(learner_id)
            return
        except KeyError:
            pass
        request = plan.lesson_request
        self._learners.upsert(learner_id, LearnerProfileInput(
            display_name=learner_id,
            subjects=[SubjectState(subject=request.subject, framework_id=request.framework_id,
                                   target_level=request.target_level)],
            preferences=LearnerPreferences(language_of_instruction=request.language_of_instruction)))

    def _invalid_artifacts(self, task: Task) -> dict[str, str]:
        """artifact id -> why its stored bytes are unusable. Corrupt objects are discarded so they can be rewritten."""
        invalid = {}
        for artifact in self._tasks.artifacts(task.task_id):
            problem = self._tasks.verify_artifact(artifact)
            if problem is not None:
                invalid[artifact.artifact_id] = problem
                if problem == "checksum mismatch":
                    self._tasks.discard_artifact_object(artifact)
        return invalid

    def _makers(self, task_id: str) -> dict[str, str]:
        """artifact id -> the workflow node that created it."""
        return {e.data["artifact_id"]: e.node_id for e in self._tasks.events(task_id)
                if e.type == EventType.ARTIFACT_CREATED and e.node_id and "artifact_id" in e.data}

    # --- evaluation -------------------------------------------------------------------------------------------

    async def start_evaluation(self, lesson_task_id: str) -> Task:
        """Create the post-lesson assessment (one LLM step). The task then waits for the learner's answers."""
        lesson = self._tasks.get(lesson_task_id)
        return await self._tasks.start_evaluation(lesson_task_id, user_id=lesson.user_id)

    def evaluation_for(self, lesson_task_id: str) -> Task | None:
        lesson = self._tasks.get(lesson_task_id)
        found = [t for t in self._tasks.list_for_learner(lesson.learner_id)
                 if t.plan is not None and t.plan.inputs.get(LESSON_TASK) == lesson_task_id]
        return found[-1] if found else None

    async def submit_evaluation(self, task_id: str, payload: dict) -> Task:
        return await self._tasks.submit_answers(task_id, payload)

    # --- report -----------------------------------------------------------------------------------------------

    def report(self, outcome: RunOutcome, plan: ProductionPlan, *, start_time: datetime,
               health: list[HealthStatus] | None = None, evaluation_task_id: str | None = None) -> ProductionReport:
        task = self._tasks.get(outcome.task.task_id)
        artifacts = self._tasks.artifacts(task.task_id)
        makers = self._makers(task.task_id)
        requests = task.cost.provider_requests
        by_node: dict[str | None, list[str]] = {}
        for r in requests:
            by_node.setdefault(r.node_id, []).append(r.request_id)
        graph = [ArtifactNode(artifact_id=a.artifact_id, type=a.type, name=a.name, version=a.version, uri=a.uri,
                              content_hash=a.content_hash, size_bytes=a.size_bytes, provider=a.provider,
                              parent_ids=a.parent_ids, node_id=makers.get(a.artifact_id),
                              request_ids=by_node.get(makers.get(a.artifact_id), []))
                 for a in artifacts]
        trace = [NodeTrace(node_id=s.node_id, status=s.status.value, attempts=s.attempts, duration_ms=s.duration_ms,
                           request_ids=by_node.get(s.node_id, []),
                           artifact_ids=[a.artifact_id for a in graph if a.node_id == s.node_id])
                 for s in task.steps()]
        usage = task_usage(task.cost)
        video = self._final_video(artifacts)
        result_warnings = task.result.warnings if task.result else []
        return ProductionReport(
            task_id=task.task_id, run_key=str(task.metadata.get(RUN_KEY, plan.run_key)), reused=outcome.reused,
            resumed=outcome.resumed, mode=plan.mode, level=plan.task.level, topic=plan.task.topic,
            language=plan.task.language, subject=plan.lesson_request.subject, learner_id=task.learner_id,
            start_time=start_time, end_time=utcnow(), status=task.status.value, providers=plan.providers,
            llm_routes=plan.llm_routes, budget=TaskBudget.of(task.metadata), health=health or [],
            artifact_graph=graph, graph_check=self._graph_check(artifacts, video), trace=trace, usage=usage,
            estimated_cost_usd=usage.estimated_cost_usd, cost_complete=usage.cost_complete,
            warnings=[*plan.warnings, *outcome.warnings, *result_warnings], errors=task.errors,
            final_video=video, summary=self._summary(task, artifacts), evaluation_task_id=evaluation_task_id,
        )

    @staticmethod
    def _final_video(artifacts: list[Artifact]) -> FinalVideo | None:
        videos = [a for a in artifacts if a.type == ArtifactType.VIDEO]
        if not videos:
            return None
        v = videos[-1]
        meta = v.metadata
        return FinalVideo(artifact_id=v.artifact_id, uri=v.uri, content_hash=v.content_hash, size_bytes=v.size_bytes,
                          duration=meta.get("duration"), width=meta.get("width"), height=meta.get("height"))

    @staticmethod
    def _graph_check(artifacts: list[Artifact], video: FinalVideo | None) -> GraphCheck:
        by_id = {a.artifact_id: a for a in artifacts}
        present = {a.type for a in artifacts}
        missing = [t.value for t in REQUIRED_CHAIN if t not in present]
        ancestors: set[str] = set()
        if video is not None:
            frontier = list(by_id[video.artifact_id].parent_ids)
            seen: set[str] = set()
            while frontier:
                pid = frontier.pop()
                if pid in seen or pid not in by_id:
                    continue
                seen.add(pid)
                ancestors.add(by_id[pid].type.value)
                frontier.extend(by_id[pid].parent_ids)
        chain = [t.value for t in REQUIRED_CHAIN if t != ArtifactType.VIDEO]
        missing += [f"{t} (not an ancestor of the VIDEO)" for t in chain
                    if video is not None and t not in ancestors and t not in missing]
        return GraphCheck(complete=not missing and video is not None, missing_types=missing,
                          video_ancestor_types=sorted(ancestors))

    def _summary(self, task: Task, artifacts: list[Artifact]) -> dict:
        def meta(kind: ArtifactType) -> dict:
            found = [a for a in artifacts if a.type == kind]
            return found[-1].metadata if found else {}

        research, lesson, presentation = (meta(ArtifactType.RESEARCH_BUNDLE), meta(ArtifactType.LESSON),
                                          meta(ArtifactType.PRESENTATION))
        timeline = meta(ArtifactType.PRESENTATION_TIMELINE)
        images = [a for a in artifacts if a.type == ArtifactType.IMAGE_ASSET]
        audio = [a for a in artifacts if a.type == ArtifactType.AUDIO_ASSET]
        return {
            "research": {"sources": research.get("sources", 0), "evidence": research.get("evidence", 0),
                         "citations": research.get("citations", 0), "status": research.get("status")},
            "lesson": {"title": lesson.get("title"), "sections": lesson.get("sections", 0),
                       "references": lesson.get("references", 0),
                       "review": task.result.review_verdict if task.result else None,
                       "revisions": task.result.revisions if task.result else None},
            "visual": {"images": len(images),
                       "generated": sum(a.metadata.get("origin") == "generated" for a in images),
                       "searched": sum(a.metadata.get("origin") == "search" for a in images)},
            "presentation": {"slides": presentation.get("slides", 0)},
            "audio": {"segments": len(audio),
                      "duration": round(sum(float(a.metadata.get("duration", 0.0)) for a in audio), 3),
                      "timeline_duration": timeline.get("duration")},
        }

    def export(self, report: ProductionReport, out_dir: Path) -> list[Path]:
        """Copy the final video, its presentation and subtitles into `out_dir` (copies; the canonical artifacts stay
        in the object store)."""
        wanted = {ArtifactType.VIDEO: "lesson.mp4", ArtifactType.PRESENTATION: "lesson.pptx",
                  ArtifactType.SUBTITLE: "lesson.vtt"}
        latest: dict[ArtifactType, ArtifactNode] = {}
        for node in report.artifact_graph:
            if node.type in wanted:
                latest[node.type] = node
        return [self._tasks.export_artifact(node.artifact_id, out_dir / wanted[kind]) for kind, node in latest.items()]
