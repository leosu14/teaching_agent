"""Lesson-generation workflow template: the adaptive lesson.

snapshot -> concept graph -> learning goal -> adaptive diagnostic (ask / wait for answers / re-assess, up to N
rounds) -> diagnostic evidence -> learner model -> knowledge gaps -> pedagogical plan -> adaptive artifacts ->
research -> research policy -> research artifact -> lesson plan (worded from the pedagogical plan) ->
teach/review/revise loop -> visuals (only for an approved
lesson) -> visual policy -> lesson artifacts -> slide planning -> slide plan validation -> presentation build ->
presentation render (only for an approved lesson) -> audio planning -> audio plan validation -> TTS, audio validation
and AUDIO_ASSETs -> audio policy -> optional generated video segments (only for a lesson that asks for them: video strategy, segment plan, generation
jobs, validated and normalised GENERATED_VIDEO_ASSETs) -> presentation timeline (only after the presentation is
rendered) -> video planning
-> video plan validation -> video composition, MP4 validation and the VIDEO artifact (only after the timeline) ->
learner memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.runtime.orchestrator.planner import ExpectedCall, WorkflowTemplate
from app.runtime.workflow.engine import WorkflowDefinition
from app.runtime.workflow.nodes import (
    AgentNode,
    ConditionalNode,
    HumanApprovalNode,
    Node,
    ReviewNode,
    StateView,
    ToolNode,
    TransformNode,
)
from app.runtime.workflows.audio import AudioFailurePolicy, NarrationNode, apply_audio_policy
from app.runtime.workflows.generative_video import GenerativeVideoNode
from app.runtime.workflows.research_policy import ResearchRequirement, apply_research_policy
from app.runtime.workflows.video import VideoFailurePolicy, VideoNode
from app.runtime.workflows.visual_policy import VisualFailurePolicy, apply_visual_policy
from app.schemas.artifact import Artifact, ArtifactBatch, ArtifactDraft, ArtifactType, StoredArtifacts
from app.schemas.audio import (
    AudioPlan,
    AudioPlanningRequest,
    AudioPlanValidationReport,
    NarrationRequest,
    NarrationResult,
    TimelineRequest,
    TimelineResult,
)
from app.schemas.generative_video import (
    GeneratedVideoConfig,
    GeneratedVideoResult,
    GeneratedVideoStageRequest,
    VideoSegmentPlanSet,
)
from app.schemas.learner import LearnerSnapshot, LearningGoal, MasteryChange, MasteryUpdate
from app.schemas.lesson import (
    DiagnosticAnswers,
    DiagnosticInput,
    DiagnosticOutcome,
    DiagnosticQuestionSheet,
    DiagnosticRound,
    DiagnosticStep,
    LessonContent,
    LessonOutcome,
    LessonPlan,
    LessonRequest,
    PlannerInput,
    ResearchRequest,
    ReviewerInput,
    RevisionContext,
    SectionVisual,
    TeacherInput,
    VisualRequest,
)
from app.schemas.presentation import (
    Presentation,
    PresentationBuildRequest,
    PresentationConfig,
    PresentationRenderRequest,
    SlideDeckPlan,
    SlidePlanningRequest,
    SlidePlanValidationReport,
)
from app.schemas.pedagogy import (
    AdaptiveQuestioningPolicy,
    ConceptSet,
    GapAnalysisRequest,
    KnowledgeGapSet,
    LearnerModel,
    LessonFocus,
    PedagogicalPlan,
    PlanningRequest,
    learner_context,
)
from app.schemas.research import ResearchBundle
from app.schemas.providers import Capability
from app.schemas.task import ArtifactSummary, TaskResult
from app.schemas.usage import TaskBudget
from app.schemas.video import (
    AudioAssetInput,
    ImageAssetInput,
    VideoConfig,
    VideoPlan,
    VideoPlanningRequest,
    VideoPlanValidationReport,
    VideoResult,
    GeneratedClipInput,
    VideoStageRequest,
)
from app.schemas.video_strategy import GapSignal, VideoStrategyRequest
from app.schemas.visual import VisualResult
from app.schemas.workflow import NodeStatus, ReviewOutcome, RevisionPolicy, RevisionRequest

WORKFLOW_ID = "lesson_generation"
GOAL_KEY = "learning_goal_id"  # Task.metadata key naming the goal to plan against (default: resolved per topic)
# Task.metadata key carrying a LessonFocus: the lesson serves one curriculum objective and next learning action.
FOCUS_KEY = "lesson_focus"
# Provider capabilities the lesson's agents and tools call (what a production run must configure).
# Video generation is only called for a lesson that asks for generated segments (see required_capabilities).
PROVIDER_CAPABILITIES = frozenset({Capability.LLM, Capability.SEARCH, Capability.IMAGE, Capability.IMAGE_SEARCH,
                                   Capability.TTS, Capability.VIDEO_GENERATION})
# A lesson request opts into generated video segments by asking for this capability.
GENERATED_VIDEO_CAPABILITY = "video.generated_segments"


@dataclass(frozen=True)
class LessonWorkflowOptions:
    diagnostic_rounds: int = 2
    memory_confidence: float = 0.6
    questioning: AdaptiveQuestioningPolicy = field(default_factory=AdaptiveQuestioningPolicy)
    lesson_minutes: int | None = None  # time available for a lesson; None: the learner's preferred session length
    revision_policy: RevisionPolicy = field(default_factory=RevisionPolicy)
    research_requirement: ResearchRequirement = "mandatory"
    research_max_results: int = 5
    research_max_sources: int = 6
    research_min_reliability: float = 0.5
    visual_failure_policy: VisualFailurePolicy = "fail"
    visual_max_per_lesson: int = 6
    visual_max_candidates: int = 3
    presentation_config: PresentationConfig = field(default_factory=PresentationConfig)
    presentation_max_slides: int = 20
    audio_failure_policy: AudioFailurePolicy = "fail"
    audio_language: str | None = None  # BCP 47 tag; None: the lesson's language of instruction
    audio_voice: str | None = None  # a provider voice id; None: the provider's first voice for the language
    audio_speaking_rate: float = 1.0
    audio_format: str = "wav"
    audio_sample_rate: int | None = None  # None: the provider's default
    audio_max_words_per_segment: int = 80
    audio_silent_slide_seconds: float = 3.0
    video_failure_policy: VideoFailurePolicy = "fail"  # fail: video is required; continue: optional, warn
    video_config: VideoConfig = field(default_factory=VideoConfig)
    generated_video_enabled: bool = True  # False: generated segments are never made, even when a lesson asks
    generated_video: GeneratedVideoConfig = field(default_factory=GeneratedVideoConfig)


def _request(v: StateView) -> LessonRequest:
    assert v.task.plan is not None
    return v.task.plan.lesson_request


def _concepts(v: StateView) -> ConceptSet:
    return v.output("knowledge_graph", ConceptSet)


def _topic_concept_ids(v: StateView) -> list[str]:
    topic = _request(v).topic.casefold()  # the knowledge base matches topics case-insensitively too
    return [c.concept_id for c in _concepts(v).concepts if (c.topic or "").casefold() == topic]


def _goal(v: StateView) -> LearningGoal:
    return v.output("load_goal", LearningGoal)


def _focus(v: StateView) -> LessonFocus | None:
    raw = v.task.metadata.get(FOCUS_KEY)
    return LessonFocus.model_validate(raw) if raw else None


def _model(v: StateView) -> LearnerModel:
    return v.output("learner_model", LearnerModel)


def _pedagogical_plan(v: StateView) -> PedagogicalPlan:
    return v.output("pedagogical_plan", PedagogicalPlan)


def _pedagogy_artifact_id(v: StateView, key: str) -> str:
    return v.output("store_pedagogy", StoredArtifacts).by_key[key]


def merge_changes(*updates: MasteryUpdate) -> list[MasteryChange]:
    """One change per concept across consecutive updates: the first `before`, the last `after`, every reason."""
    merged: dict[str, MasteryChange] = {}
    for update in updates:
        for c in update.changes:
            prev = merged.get(c.concept_id)
            if prev is None:
                merged[c.concept_id] = c
            else:
                reasons = list(dict.fromkeys([*prev.reason.split(", "), *c.reason.split(", ")]))
                merged[c.concept_id] = MasteryChange(concept_id=c.concept_id, before=prev.before, after=c.after,
                                                     reason=", ".join(reasons))
    return list(merged.values())


def _review(v: StateView) -> ReviewOutcome:
    return v.output("teach_review", ReviewOutcome)


def _research(v: StateView) -> ResearchBundle:
    return v.output("research_policy", ResearchBundle)


def _research_artifact_id(v: StateView) -> str:
    return v.output("store_research", StoredArtifacts).by_key["research_bundle"]


def _visuals(v: StateView) -> VisualResult | None:
    """The visuals after the visual policy; None before they exist or when they were skipped."""
    return v.maybe("visual_policy", VisualResult)


def _lesson(v: StateView) -> LessonContent:
    """The approved lesson with its references resolved from the research bundle and its sections' visuals
    attached from the visual result. Neither is ever model-written."""
    lesson = LessonContent.model_validate(_review(v).candidate)
    research = _research(v)
    visuals = _visuals(v)
    sections = [s.model_copy(update={"visuals": [
        SectionVisual(visual_id=a.visual_id, artifact_id=a.artifact_id, asset_id=a.asset_id,
                      visual_type=a.visual_type, origin=a.origin, purpose=a.purpose, description=a.description)
        for a in (visuals.for_section(s.section_id) if visuals else [])
    ]}) for s in lesson.sections]
    cited = {c for s in lesson.sections for c in s.citations}
    return lesson.model_copy(update={"sections": sections,
                                     "references": [c for c in research.citations if c.citation_id in cited]})


def _image_budget(v: StateView) -> dict:
    """The task's image budget (a production task carries one in its metadata); no limit otherwise."""
    budget = TaskBudget.of(v.task.metadata)
    if budget is None:
        return {}
    return {"max_generated_images": budget.max_generated_images, "max_searched_images": budget.max_searched_images}


def _lesson_artifact_id(v: StateView) -> str:
    return v.output("store_artifacts", StoredArtifacts).by_key["lesson"]


def _slide_planning(v: StateView, options: LessonWorkflowOptions) -> SlidePlanningRequest:
    visuals = _visuals(v)
    return SlidePlanningRequest(
        request=_request(v), plan=v.output("plan", LessonPlan), lesson=_lesson(v), research=_research(v),
        visual_plan=visuals.plan if visuals else None, image_assets=visuals.assets if visuals else [],
        max_slides=options.presentation_max_slides,
    )


def _deck(v: StateView) -> SlideDeckPlan:
    """The slide plan as validated by the gate; nothing downstream sees an unvalidated deck."""
    deck = v.output("validate_slide_plan", SlidePlanValidationReport).deck
    assert deck is not None
    return deck


def _slide_plan_artifact_id(v: StateView) -> str:
    return v.output("store_slide_plan", StoredArtifacts).by_key["slide_plan"]


def _presentation(v: StateView) -> Artifact | None:
    return v.maybe("render_presentation", Artifact)


def _presentation_artifact_id(v: StateView) -> str:
    presentation = _presentation(v)
    assert presentation is not None
    return presentation.artifact_id


def _audio_planning(v: StateView, options: LessonWorkflowOptions) -> AudioPlanningRequest:
    return AudioPlanningRequest(
        task_id=v.task.task_id, request=_request(v), lesson=_lesson(v), deck=_deck(v), research=_research(v),
        presentation_artifact_id=_presentation_artifact_id(v),
        language=options.audio_language or _request(v).language_of_instruction, voice_id=options.audio_voice,
        speaking_rate=options.audio_speaking_rate, max_words_per_segment=options.audio_max_words_per_segment,
    )


def _audio_plan(v: StateView) -> AudioPlan:
    """The audio plan as validated by the gate; nothing downstream sees an unvalidated plan."""
    plan = v.output("validate_audio_plan", AudioPlanValidationReport).plan
    assert plan is not None
    return plan


def _audio_plan_artifact_id(v: StateView) -> str:
    return v.output("store_audio_plan", StoredArtifacts).by_key["audio_plan"]


def _narration(v: StateView) -> NarrationResult | None:
    """The voiced segments after the audio policy; None before they exist or when audio was skipped."""
    return v.maybe("audio_policy", NarrationResult)


def _timeline(v: StateView) -> TimelineResult | None:
    return v.maybe("audio_timeline", TimelineResult)


def _image_inputs(v: StateView) -> list[ImageAssetInput]:
    visuals = _visuals(v)
    return [ImageAssetInput(artifact_id=a.artifact_id, asset_id=a.asset_id, uri=a.uri, checksum=a.checksum,
                            media_type=a.media_type, width=a.width, height=a.height, alt_text=a.description)
            for a in (visuals.assets if visuals else [])]


def wants_generated_video(request: LessonRequest, options: LessonWorkflowOptions) -> bool:
    return options.generated_video_enabled and GENERATED_VIDEO_CAPABILITY in request.capabilities


def _generated_video_config(v: StateView, options: LessonWorkflowOptions) -> GeneratedVideoConfig:
    """The configured limits, tightened by the task's budget (a production task carries one)."""
    config = options.generated_video
    budget = TaskBudget.of(v.task.metadata)
    if budget is None:
        return config
    update: dict = {}
    if budget.max_generated_video_segments is not None:
        update["max_segments"] = min(config.max_segments, budget.max_generated_video_segments)
    if budget.max_generated_video_seconds is not None:
        update["max_total_seconds"] = min(config.max_total_seconds, budget.max_generated_video_seconds)
    if budget.max_video_generation_cost_usd is not None:
        update["max_cost_usd"] = (budget.max_video_generation_cost_usd if config.max_cost_usd is None
                                  else min(config.max_cost_usd, budget.max_video_generation_cost_usd))
    return config.model_copy(update=update)


def _video_strategy(v: StateView, options: LessonWorkflowOptions) -> VideoStrategyRequest:
    """The approved lesson, its pedagogy (gaps carry no learner data), the slide plan, the images it has and how
    long each slide is narrated. The provider's limits are added by the strategy tool."""
    visuals = _visuals(v)
    narration = _narration(v)
    seconds: dict[str, float] = {}
    for a in (narration.assets if narration else []):
        seconds[a.slide_id] = round(seconds.get(a.slide_id, 0.0) + a.duration, 3)
    return VideoStrategyRequest(
        lesson=_lesson(v), lesson_artifact_id=_lesson_artifact_id(v), pedagogical_plan=_pedagogical_plan(v).brief(),
        gaps=GapSignal.from_gaps(v.output("knowledge_gaps", KnowledgeGapSet)), deck=_deck(v),
        visual_plan=visuals.plan if visuals else None, image_assets=_image_inputs(v), narration_seconds=seconds,
        language=_request(v).language_of_instruction, config=_generated_video_config(v, options),
        video=options.video_config)


def _segment_plan(v: StateView) -> VideoSegmentPlanSet:
    return v.output("video_segment_plan", VideoSegmentPlanSet)


def _segment_plan_artifact_id(v: StateView) -> str:
    return v.output("store_video_segment_plan", StoredArtifacts).by_key["video_segment_plan"]


def _generated(v: StateView) -> GeneratedVideoResult | None:
    """The generated clips after the failure policy; None when the lesson did not ask for any."""
    return v.maybe("generate_video_segments", GeneratedVideoResult)


def _clip_inputs(v: StateView) -> list[GeneratedClipInput]:
    generated = _generated(v)
    if generated is None or not generated.assets:
        return []
    plan = _segment_plan(v)
    out = []
    for asset in generated.assets:
        segment = plan.segment(asset.segment_id)
        out.append(GeneratedClipInput(
            segment_id=asset.segment_id, slide_id=segment.slide_id, artifact_id=asset.artifact_id, uri=asset.uri,
            checksum=asset.checksum, media_type=asset.media_type, duration=asset.duration, width=asset.width,
            height=asset.height, fps=asset.fps, has_audio=asset.has_audio, strategy=segment.insertion_strategy,
            audio=segment.audio))
    return out


def _clip_artifact_ids(v: StateView) -> list[str]:
    return [c.artifact_id for c in _clip_inputs(v)]


def _video_planning(v: StateView, options: LessonWorkflowOptions) -> VideoPlanningRequest:
    """The presentation, its timeline and the IMAGE_ASSET and AUDIO_ASSET references; never file paths."""
    timeline = _timeline(v)
    narration = _narration(v)
    assert timeline is not None and narration is not None
    meta = {a.artifact_id: a.metadata for a in narration.artifacts}
    return VideoPlanningRequest(
        task_id=v.task.task_id, presentation=v.output("build_presentation", Presentation),
        presentation_artifact_id=_presentation_artifact_id(v), timeline=timeline.timeline,
        timeline_artifact_id=timeline.artifact.artifact_id, timeline_checksum=timeline.artifact.content_hash,
        image_assets=_image_inputs(v),
        audio_assets=[AudioAssetInput(artifact_id=a.artifact_id, segment_id=a.segment_id, slide_id=a.slide_id,
                                      uri=a.uri, checksum=a.checksum, media_type=a.media_type, duration=a.duration,
                                      text=meta[a.artifact_id]["text"], language=meta[a.artifact_id]["language"])
                      for a in narration.assets],
        generated_clips=_clip_inputs(v), config=options.video_config,
    )


def _video_plan(v: StateView) -> VideoPlan:
    """The video plan as validated by the gate; nothing downstream sees an unvalidated plan."""
    plan = v.output("validate_video_plan", VideoPlanValidationReport).plan
    assert plan is not None
    return plan


def _video_plan_artifact_id(v: StateView) -> str:
    return v.output("store_video_plan", StoredArtifacts).by_key["video_plan"]


def _video(v: StateView) -> VideoResult | None:
    return v.maybe("compose_video", VideoResult)


def _generated_video_warnings(v: StateView) -> list[str]:
    generated = _generated(v)
    return generated.warnings if generated else []


def _video_warnings(v: StateView) -> list[str]:
    if v.status("video_plan") == NodeStatus.SKIPPED:
        return ["No video was generated: video is only composed from a narrated presentation timeline."]
    video = _video(v)
    return video.warnings if video else []


def _audio_warnings(v: StateView) -> list[str]:
    if v.status("audio_plan") == NodeStatus.SKIPPED:
        return ["No narration was generated: audio is only generated for the presentation of an approved lesson."]
    narration = _narration(v)
    return narration.warnings if narration else []


def _presentation_warnings(v: StateView) -> list[str]:
    if v.status("slide_plan") == NodeStatus.SKIPPED:
        return [f"No presentation was generated: the lesson was {_review(v).status.replace('_', ' ')}, "
                "and presentations are only generated for an approved lesson."]
    return []


def _visual_warnings(v: StateView) -> list[str]:
    if v.status("visual") == NodeStatus.SKIPPED:
        return [f"No visuals were generated: the lesson was {_review(v).status.replace('_', ' ')}, "
                "and visuals are only generated for an approved lesson."]
    visuals = _visuals(v)
    return visuals.warnings if visuals else []


def build_lesson_workflow(request: LessonRequest, options: LessonWorkflowOptions) -> WorkflowDefinition:
    rounds = options.diagnostic_rounds
    nodes: list[Node] = [
        ToolNode(
            id="learner_snapshot",
            tool="learner.snapshot",
            permissions=frozenset({"learner:read"}),
            build_input=lambda v: {
                "learner_id": v.task.learner_id, "subject": _request(v).subject,
                "framework_id": _request(v).framework_id, "target_level": _request(v).target_level,
            },
        ),
        ToolNode(id="knowledge_graph", tool="knowledge.concepts", permissions=frozenset({"knowledge:read"}),
                 depends_on=("learner_snapshot",), build_input=lambda v: {"domain": _request(v).subject}),
        ToolNode(id="load_goal", tool="learning_goal.resolve", permissions=frozenset({"learner:read"}),
                 depends_on=("knowledge_graph",),
                 build_input=lambda v: {
                     "learner_id": v.task.learner_id, "domain": _request(v).subject, "topic": _request(v).topic,
                     "topic_concepts": _topic_concept_ids(v),
                     "target_level": _request(v).target_level, "goal_id": v.task.metadata.get(GOAL_KEY)}),
    ]

    def diagnostic_input(round_number: int):
        def build(v: StateView) -> DiagnosticInput:
            history = [
                DiagnosticRound(items=v.output(f"diagnose_{k}", DiagnosticStep).items,
                                answers=v.output(f"answers_{k}", DiagnosticAnswers).answers)
                for k in range(1, round_number)
            ]
            return DiagnosticInput(
                request=_request(v), snapshot=v.output("learner_snapshot", LearnerSnapshot).for_provider(),
                rounds=history, round_number=round_number, max_rounds=rounds,
                memory_confidence_threshold=options.memory_confidence, questioning=options.questioning,
            )
        return build

    def question_sheet(round_number: int):
        def build(v: StateView) -> DiagnosticQuestionSheet:
            step = v.output(f"diagnose_{round_number}", DiagnosticStep)
            return DiagnosticQuestionSheet(round_number=round_number, questions=[i.question for i in step.items])
        return build

    def asks(round_number: int):
        return lambda v: v.output(f"diagnose_{round_number}", DiagnosticStep).status == "ask"

    for r in range(1, rounds + 1):
        nodes += [
            AgentNode(id=f"diagnose_{r}", agent="knowledge_diagnostic", build_input=diagnostic_input(r),
                      depends_on=("learner_snapshot",) if r == 1 else (f"answers_{r - 1}",),
                      after=("load_goal",) if r == 1 else ()),
            ConditionalNode(id=f"diagnostic_gate_{r}", depends_on=(f"diagnose_{r}",), predicate=asks(r),
                            when_true=(f"answers_{r}",)),
            HumanApprovalNode(id=f"answers_{r}", depends_on=(f"diagnostic_gate_{r}",), wait_kind="diagnostic_answers",
                              build_request=question_sheet(r), response_model=DiagnosticAnswers),
        ]
    nodes.append(AgentNode(id=f"diagnose_{rounds + 1}", agent="knowledge_diagnostic",
                           build_input=diagnostic_input(rounds + 1), depends_on=(f"answers_{rounds}",)))
    diagnose_ids = [f"diagnose_{r}" for r in range(1, rounds + 2)]

    def final_diagnostic(v: StateView) -> DiagnosticStep:
        steps = [v.maybe(nid, DiagnosticStep) for nid in diagnose_ids]
        final = [s for s in steps if s is not None and s.status == "complete"]
        if not final:
            raise ValueError("diagnostic did not conclude")
        return final[-1]

    def diagnostic(v: StateView) -> DiagnosticStep:
        return v.output("diagnostic", DiagnosticStep)

    def teacher_input(v: StateView, revision: RevisionRequest | None) -> TeacherInput:
        plan = v.output("plan", LessonPlan)
        focus = [c.concept_id for c in plan.concepts] + plan.review_concepts
        pedagogy = _pedagogical_plan(v)
        return TeacherInput(
            request=_request(v), plan=plan, research=_research(v).focused(focus),
            pedagogical_plan=pedagogy.brief(), learner=_learner_context(v),
            gaps=v.output("knowledge_gaps", KnowledgeGapSet).for_concepts(pedagogy.concept_ids()),
            revision=None if revision is None else RevisionContext(
                revision_number=revision.revision_number, issues=revision.issues,
                previous=LessonContent.model_validate(revision.previous),
            ),
        )

    def reviewer_input(v: StateView, candidate, revision_number: int) -> ReviewerInput:
        return ReviewerInput(request=_request(v), plan=v.output("plan", LessonPlan),
                             research=_research(v), content=candidate,
                             revision_number=revision_number)

    # Optional generated video segments: only in the workflow of a lesson that asks for them (any other lesson runs
    # exactly the nodes it always did), after the narration (the strategy sizes clips to it) and before the
    # timeline. Agents never wait on a provider: the generation node does.
    generated_video_nodes: list[Node] = [
        ToolNode(id="video_segment_plan", tool="visual.video_strategy", depends_on=("audio_policy",),
                 build_input=lambda v: _video_strategy(v, options)),
        ToolNode(id="store_video_segment_plan", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("video_segment_plan",), build_input=_segment_plan_batch),
        GenerativeVideoNode(id="generate_video_segments", depends_on=("store_video_segment_plan",),
                            build_input=lambda v: GeneratedVideoStageRequest(
                                plan=_segment_plan(v), plan_artifact_id=_segment_plan_artifact_id(v),
                                config=_generated_video_config(v, options))),
    ] if wants_generated_video(request, options) else []
    generated_video_ids = tuple(n.id for n in generated_video_nodes)

    nodes += [
        TransformNode(id="diagnostic", fn=final_diagnostic, depends_on=("diagnose_1",),
                      after=tuple(diagnose_ids[1:]) + tuple(f"answers_{r}" for r in range(1, rounds + 1))),
        # Deterministic learner state: the diagnostic's graded answers become evidence, mastery is updated from it
        # by code, and the gaps and the pedagogical plan are computed from the resulting learner model.
        ToolNode(id="record_diagnostic", tool="learner.record_diagnostic", permissions=frozenset({"learner:write"}),
                 depends_on=("diagnostic",),
                 build_input=lambda v: DiagnosticOutcome(
                     task_id=v.task.task_id, learner_id=v.task.learner_id, request=_request(v),
                     diagnostic=diagnostic(v).result, concepts=diagnostic(v).concepts)),
        ToolNode(id="learner_model", tool="learner.model", permissions=frozenset({"learner:read"}),
                 depends_on=("record_diagnostic",),
                 build_input=lambda v: {
                     "learner_id": v.task.learner_id, "domain": _request(v).subject,
                     "framework_id": _request(v).framework_id, "target_level": _request(v).target_level,
                     "concept_ids": [c.concept_id for c in _concepts(v).concepts]}),
        ToolNode(id="knowledge_gaps", tool="pedagogy.analyze_gaps", permissions=frozenset({"learner:read"}),
                 depends_on=("learner_model", "load_goal"),
                 build_input=lambda v: GapAnalysisRequest(model=_model(v), goal=_goal(v),
                                                          concepts=_concepts(v).concepts)),
        ToolNode(id="pedagogical_plan", tool="pedagogy.plan", permissions=frozenset({"learner:read"}),
                 depends_on=("knowledge_gaps",),
                 build_input=lambda v: PlanningRequest(
                     model=_model(v), gaps=v.output("knowledge_gaps", KnowledgeGapSet), goal=_goal(v),
                     concepts=_concepts(v).concepts, available_minutes=options.lesson_minutes, focus=_focus(v))),
        ToolNode(id="store_pedagogy", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("pedagogical_plan",), build_input=_pedagogy_batch),
        AgentNode(id="research", agent="research", depends_on=("store_pedagogy",),
                  build_input=lambda v: ResearchRequest(
                      request=_request(v), diagnostic=diagnostic(v).result, concepts=diagnostic(v).concepts,
                      max_results_per_query=options.research_max_results, max_sources=options.research_max_sources,
                      min_reliability=options.research_min_reliability)),
        TransformNode(id="research_policy", depends_on=("research",),
                      fn=lambda v: apply_research_policy(v.output("research", ResearchBundle),
                                                         options.research_requirement)),
        ToolNode(id="store_research", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("research_policy",), build_input=_research_batch),
        AgentNode(id="plan", agent="curriculum_planner", depends_on=("store_research",),
                  build_input=lambda v: PlannerInput(
                      request=_request(v), learner=_learner_context(v), pedagogical_plan=_pedagogical_plan(v).brief(),
                      diagnostic=diagnostic(v).result, research=_research(v), concepts=_plan_concepts(v))),
        ReviewNode(id="teach_review", depends_on=("plan",), generator="teacher", reviewer="content_reviewer",
                   candidate_model=LessonContent, build_generator_input=teacher_input,
                   build_reviewer_input=reviewer_input, policy=options.revision_policy),
        # Visuals only for a lesson that passed review: never for a rejected draft, nor for one accepted with warnings.
        ConditionalNode(id="visual_gate", depends_on=("teach_review",),
                        predicate=lambda v: _review(v).status == "approved", when_true=("visual",)),
        AgentNode(id="visual", agent="visual", depends_on=("visual_gate",),
                  build_input=lambda v: VisualRequest(
                      plan=v.output("plan", LessonPlan), lesson=_lesson(v), research=_research(v),
                      language=_request(v).language_of_instruction, max_visuals=options.visual_max_per_lesson,
                      max_candidates=options.visual_max_candidates, parent_artifact_ids=[_research_artifact_id(v)],
                      **_image_budget(v))),
        TransformNode(id="visual_policy", depends_on=("visual",),
                      fn=lambda v: apply_visual_policy(v.output("visual", VisualResult),
                                                       options.visual_failure_policy)),
        TransformNode(id="package_artifacts", depends_on=("teach_review",), after=("visual_policy",), fn=_package),
        ToolNode(id="store_artifacts", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("package_artifacts",), build_input=lambda v: v.output("package_artifacts", ArtifactBatch)),
        # The presentation, like the visuals, is only made for a lesson that passed review.
        ConditionalNode(id="presentation_gate", depends_on=("store_artifacts",),
                        predicate=lambda v: _review(v).status == "approved", when_true=("slide_plan",)),
        AgentNode(id="slide_plan", agent="slide_planner", depends_on=("presentation_gate",),
                  build_input=lambda v: _slide_planning(v, options)),
        ToolNode(id="validate_slide_plan", tool="slide_plan.validate", depends_on=("slide_plan",),
                 build_input=lambda v: _slide_planning(v, options).validation_request(
                     v.output("slide_plan", SlideDeckPlan), enforce=True)),
        ToolNode(id="store_slide_plan", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("validate_slide_plan",), build_input=_slide_plan_batch),
        ToolNode(id="build_presentation", tool="presentation.build", permissions=frozenset({"artifact:read"}),
                 depends_on=("store_slide_plan",),
                 build_input=lambda v: PresentationBuildRequest(
                     deck=_deck(v), slide_plan_artifact_id=_slide_plan_artifact_id(v),
                     config=options.presentation_config.model_copy(
                         update={"language": _request(v).language_of_instruction}),
                     research=_research(v))),
        ToolNode(id="render_presentation", tool="presentation.render", permissions=frozenset({"artifact:write"}),
                 depends_on=("build_presentation",), build_input=_render_request),
        # Audio only for a rendered presentation (so only for an approved lesson); a skipped or failed presentation
        # means no audio.
        AgentNode(id="audio_plan", agent="audio_planner", depends_on=("render_presentation",),
                  build_input=lambda v: _audio_planning(v, options)),
        ToolNode(id="validate_audio_plan", tool="audio_plan.validate", depends_on=("audio_plan",),
                 build_input=lambda v: _audio_planning(v, options).validation_request(
                     v.output("audio_plan", AudioPlan), enforce=True)),
        ToolNode(id="store_audio_plan", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("validate_audio_plan",), build_input=_audio_plan_batch),
        NarrationNode(id="synthesize_audio", depends_on=("store_audio_plan",),
                      build_input=lambda v: NarrationRequest(
                          plan=_audio_plan(v), audio_plan_artifact_id=_audio_plan_artifact_id(v),
                          output_format=options.audio_format, sample_rate=options.audio_sample_rate)),
        TransformNode(id="audio_policy", depends_on=("synthesize_audio",),
                      fn=lambda v: apply_audio_policy(v.output("synthesize_audio", NarrationResult),
                                                      options.audio_failure_policy)),
        *generated_video_nodes,
        ToolNode(id="audio_timeline", tool="audio.timeline", permissions=frozenset({"artifact:write"}),
                 depends_on=("audio_policy",), after=generated_video_ids,
                 build_input=lambda v: TimelineRequest(
                     plan=_audio_plan(v), narration=v.output("audio_policy", NarrationResult),
                     slide_ids=[s.slide_id for s in _deck(v).slides],
                     presentation_artifact_id=_presentation_artifact_id(v),
                     audio_plan_artifact_id=_audio_plan_artifact_id(v),
                     silent_slide_seconds=options.audio_silent_slide_seconds)),
        # Video only once the timeline exists (so only for an approved, rendered, narrated presentation).
        AgentNode(id="video_plan", agent="video", depends_on=("audio_timeline",),
                  build_input=lambda v: _video_planning(v, options)),
        ToolNode(id="validate_video_plan", tool="video_plan.validate", depends_on=("video_plan",),
                 build_input=lambda v: _video_planning(v, options).validation_request(
                     v.output("video_plan", VideoPlan), enforce=True)),
        ToolNode(id="store_video_plan", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("validate_video_plan",), build_input=_video_plan_batch),
        VideoNode(id="compose_video", depends_on=("store_video_plan",), policy=options.video_failure_policy,
                  build_input=lambda v: VideoStageRequest(
                      plan=_video_plan(v), parent_ids=[_video_plan_artifact_id(v), _video_plan(v).timeline_ref,
                                                       _video_plan(v).presentation_ref,
                                                       *(c.artifact_id for c in _video_plan(v).generated_clips())])),
        ToolNode(id="update_learner", tool="learner.record_lesson", permissions=frozenset({"learner:write"}),
                 depends_on=("store_artifacts",), after=("render_presentation", "audio_timeline", "compose_video"),
                 build_input=_lesson_outcome),
    ]
    return WorkflowDefinition(id=WORKFLOW_ID, nodes=tuple(nodes), summarize=_summarize,
                              description="Diagnose, research, plan, teach with review, illustrate, store, plan, "
                                          "build and render the presentation, narrate it, time it, compose the "
                                          "video, remember.")


def _learner_context(v: StateView):
    return learner_context(_model(v), _pedagogical_plan(v).brief(),
                           v.output("learner_snapshot", LearnerSnapshot).framework_levels)


def _plan_concepts(v: StateView):
    """The topic's concepts plus any concept the pedagogical plan brought in from the goal's prerequisites."""
    topic = v.output("diagnostic", DiagnosticStep).concepts
    known = {c.concept_id for c in topic}
    by_id = {c.concept_id: c for c in _concepts(v).concepts}
    return [*topic, *(by_id[c].ref() for c in _pedagogical_plan(v).concept_ids() if c not in known)]


def _pedagogy_batch(v: StateView) -> ArtifactBatch:
    """Diagnostic -> LearningEvidence -> LearnerModel -> KnowledgeGapSet -> PedagogicalPlan."""
    evidence = v.output("record_diagnostic", MasteryUpdate).evidence
    model, gaps, plan = _model(v), v.output("knowledge_gaps", KnowledgeGapSet), _pedagogical_plan(v)
    drafts = []
    if evidence:
        drafts.append(ArtifactDraft(
            key="learning_evidence", name="learning_evidence", type=ArtifactType.LEARNING_EVIDENCE,
            media_type="application/json", content=_json_list(evidence),
            metadata={"source": "diagnostic", "items": len(evidence),
                      "concepts": sorted({e.concept_id for e in evidence})}))
    drafts += [
        ArtifactDraft(key="learner_model", name="learner_model", type=ArtifactType.LEARNER_MODEL,
                      media_type="application/json", content=_json(model),
                      parent_keys=["learning_evidence"] if evidence else [],
                      metadata={"domain": model.domain, "mastered": model.mastered_concepts,
                                "developing": model.developing_concepts, "weak": model.weak_concepts,
                                "evidence": model.evidence_count}),
        ArtifactDraft(key="knowledge_gaps", name="knowledge_gaps", type=ArtifactType.KNOWLEDGE_GAPS,
                      media_type="application/json", content=_json(gaps), parent_keys=["learner_model"],
                      metadata={"gap_set_id": gaps.gap_set_id, "goal_id": gaps.goal_id,
                                "gaps": [g.concept.concept_id for g in gaps.gaps]}),
        ArtifactDraft(key="pedagogical_plan", name="pedagogical_plan", type=ArtifactType.PEDAGOGICAL_PLAN,
                      media_type="application/json", content=_json(plan),
                      parent_keys=["knowledge_gaps", *(["learning_action"] if plan.focus else [])],
                      metadata={"plan_id": plan.plan_id, "targets": plan.target_concepts,
                                "prerequisites": plan.prerequisite_concepts, "reviews": plan.review_concepts,
                                "minutes": plan.estimated_duration, "strategy": plan.strategy_id}),
    ]
    if plan.focus is not None:
        # The curriculum action the lesson serves, derived from the LEARNING_OBJECTIVE artifact it was chosen from.
        focus = plan.focus
        drafts.insert(-1, ArtifactDraft(
            key="learning_action", name="learning_action", type=ArtifactType.LEARNING_ACTION,
            media_type="application/json", content=_json(focus),
            parent_ids=[focus.objective_artifact_id] if focus.objective_artifact_id else [],
            metadata={"action": focus.action, "action_id": focus.action_id, "goal_id": focus.goal_id,
                      "objective_id": focus.objective_id, "concept_id": focus.concept_id,
                      "curriculum_version": focus.curriculum_version}))
    return ArtifactBatch(drafts=drafts)


def _json_list(items) -> str:
    return "[\n" + ",\n".join(i.model_dump_json(indent=2) for i in items) + "\n]"


def _research_batch(v: StateView) -> ArtifactBatch:
    research = _research(v)
    return ArtifactBatch(drafts=[ArtifactDraft(
        key="research_bundle", name="research_bundle", type=ArtifactType.RESEARCH_BUNDLE,
        media_type="application/json", content=_json(research),
        metadata={"research_id": research.research_id, "status": research.status, "sources": len(research.sources),
                  "evidence": len(research.evidence), "citations": len(research.citations),
                  "warnings": len(research.warnings)},
    )])


def _slide_plan_batch(v: StateView) -> ArtifactBatch:
    deck = _deck(v)
    return ArtifactBatch(drafts=[ArtifactDraft(
        key="slide_plan", name="slide_plan", type=ArtifactType.SLIDE_PLAN, media_type="application/json",
        content=_json(deck), parent_ids=[_lesson_artifact_id(v)],
        metadata={"deck_id": deck.deck_id, "slides": len(deck.slides),
                  "slide_types": [s.slide_type.value for s in deck.slides],
                  "image_artifact_ids": deck.image_artifact_ids(), "citation_ids": deck.citation_ids()},
    )])


def _audio_plan_batch(v: StateView) -> ArtifactBatch:
    plan = _audio_plan(v)
    return ArtifactBatch(drafts=[ArtifactDraft(
        key="audio_plan", name="audio_plan", type=ArtifactType.AUDIO_PLAN, media_type="application/json",
        content=_json(plan),
        parent_ids=[_presentation_artifact_id(v), _slide_plan_artifact_id(v), _lesson_artifact_id(v)],
        metadata={"audio_plan_id": plan.audio_plan_id, "deck_id": plan.deck_id, "language": plan.language,
                  "voice": plan.voice, "segments": len(plan.segments),
                  "required": sum(s.required for s in plan.segments), "expected_duration": plan.expected_duration()},
    )])


def _video_plan_batch(v: StateView) -> ArtifactBatch:
    plan = _video_plan(v)
    drafts = [ArtifactDraft(
        key="video_plan", name="video_plan", type=ArtifactType.VIDEO_PLAN, media_type="application/json",
        content=_json(plan), parent_ids=[plan.timeline_ref, plan.presentation_ref, *plan.image_artifact_ids(),
                                         *(c.artifact_id for c in plan.generated_clips())],
        metadata={"video_plan_id": plan.video_plan_id, "segments": len(plan.slides),
                  **({"generated_clips": [c.artifact_id for c in plan.generated_clips()]}
                     if plan.generated_clips() else {}),
                  "audio_tracks": len(plan.audio_tracks), "duration": plan.duration,
                  "width": plan.resolution.width, "height": plan.resolution.height, "fps": plan.fps,
                  "transition": plan.config.transition.value, "timeline_ref": plan.timeline_ref,
                  "presentation_ref": plan.presentation_ref,
                  "subtitles": len(plan.subtitle_track.subtitles) if plan.subtitle_track else 0},
    )]
    if plan.subtitle_track is not None:
        drafts.append(ArtifactDraft(
            key="subtitles", name="subtitles", type=ArtifactType.SUBTITLE, media_type="text/vtt",
            content=plan.subtitle_track.to_webvtt(), parent_keys=["video_plan"],
            metadata={"format": "webvtt", "language": plan.subtitle_track.language,
                      "cues": len(plan.subtitle_track.subtitles), "source": plan.subtitle_track.source,
                      "burned_in": plan.subtitle_track.burned_in}))
    return ArtifactBatch(drafts=drafts)


def _segment_plan_batch(v: StateView) -> ArtifactBatch:
    plan = _segment_plan(v)
    return ArtifactBatch(drafts=[ArtifactDraft(
        key="video_segment_plan", name="video_segment_plan", type=ArtifactType.VIDEO_SEGMENT_PLAN,
        media_type="application/json", content=_json(plan),
        parent_ids=[_lesson_artifact_id(v), _slide_plan_artifact_id(v), _pedagogy_artifact_id(v, "pedagogical_plan")],
        metadata={"plan_id": plan.plan_id, "provider": plan.provider, "strategy": plan.strategy,
                  "segments": [s.segment_id for s in plan.segments], "seconds": plan.budget.seconds,
                  "estimated_cost_usd": plan.budget.estimated_cost_usd, "cost_known": plan.budget.cost_known,
                  "decisions": {d.lesson_section_id: d.skip_reason or f"selected:{d.purpose.value}"
                                for d in plan.decisions if d.selected or d.skip_reason}},
    )])


def _render_request(v: StateView) -> PresentationRenderRequest:
    presentation = v.output("build_presentation", Presentation)
    slide_plan, lesson = _slide_plan_artifact_id(v), _lesson_artifact_id(v)
    return PresentationRenderRequest(presentation=presentation, slide_plan_artifact_id=slide_plan,
                                     lesson_artifact_id=lesson,
                                     parent_ids=[slide_plan, lesson, *presentation.image_artifact_ids()])


def _json(model) -> str:
    return model.model_dump_json(indent=2)


def _script(lesson: LessonContent) -> str:
    lines = [f"# Narration script: {lesson.title}", "", lesson.introduction, ""]
    for s in lesson.sections:
        lines += [f"## [{s.section_id}] {s.heading}", "", s.narration, ""]
    lines += ["## Summary", "", lesson.summary, ""]
    return "\n".join(lines)


def _package(v: StateView) -> ArtifactBatch:
    lesson = _lesson(v)
    review = _review(v)
    research = _research(v)
    research_artifact = _research_artifact_id(v)
    visuals = _visuals(v)
    image_ids = [a.artifact_id for a in visuals.assets] if visuals else []
    focus = _pedagogical_plan(v).focus
    objective_ids = [focus.objective_artifact_id] if focus is not None and focus.objective_artifact_id else []
    return ArtifactBatch(drafts=[
        ArtifactDraft(key="lesson_plan", name="lesson_plan", type=ArtifactType.LESSON_PLAN,
                      media_type="application/json", content=_json(v.output("plan", LessonPlan)),
                      parent_ids=[research_artifact, _pedagogy_artifact_id(v, "pedagogical_plan")],
                      metadata={"objectives": len(v.output("plan", LessonPlan).objectives),
                                "pedagogical_plan_id": _pedagogical_plan(v).plan_id}),
        ArtifactDraft(key="lesson", name="lesson", type=ArtifactType.LESSON, media_type="application/json",
                      content=_json(lesson), parent_keys=["lesson_plan"],
                      parent_ids=[research_artifact, *image_ids, *objective_ids],
                      metadata={"title": lesson.title, "level": lesson.level, "sections": len(lesson.sections),
                                "objectives": [o.objective_id for o in lesson.objectives],
                                "section_purposes": {s.section_id: s.purpose for s in lesson.sections},
                                "research_id": research.research_id, "research_status": research.status,
                                "references": len(lesson.references), "visuals": len(image_ids),
                                "visual_status": visuals.status if visuals else "skipped",
                                "visual_warnings": _visual_warnings(v)}),
        ArtifactDraft(key="script", name="narration_script", type=ArtifactType.SCRIPT, media_type="text/markdown",
                      content=_script(lesson), parent_keys=["lesson"]),
        ArtifactDraft(key="review", name="review_report", type=ArtifactType.REPORT, media_type="application/json",
                      content=review.model_dump_json(indent=2, exclude={"candidate"}), parent_keys=["lesson"],
                      metadata={"kind": "content_review", "status": review.status, "revisions": review.revisions}),
    ])


def _lesson_outcome(v: StateView) -> LessonOutcome:
    step = v.output("diagnostic", DiagnosticStep)
    plan = v.output("plan", LessonPlan)
    stored = v.output("store_artifacts", StoredArtifacts)
    assert step.result is not None
    return LessonOutcome(
        task_id=v.task.task_id, learner_id=v.task.learner_id, request=_request(v), diagnostic=step.result,
        concepts=step.concepts, lesson_title=_lesson(v).title,
        taught_concept_ids=[c.concept_id for c in plan.concepts],
        artifact_ids=[*(a.artifact_id for a in v.output("store_pedagogy", StoredArtifacts).artifacts),
                      _research_artifact_id(v), *_visual_artifact_ids(v),
                      *(a.artifact_id for a in stored.artifacts), *_presentation_artifact_ids(v),
                      *(a.artifact_id for a in _audio_artifacts(v)),
                      *(a.artifact_id for a in _generated_video_artifacts(v)),
                      *(a.artifact_id for a in _video_artifacts(v))],
    )


def _visual_artifact_ids(v: StateView) -> list[str]:
    visuals = _visuals(v)
    return [a.artifact_id for a in visuals.artifacts] if visuals else []


def _presentation_artifact_ids(v: StateView) -> list[str]:
    presentation = _presentation(v)
    if presentation is None:
        return []
    return [_slide_plan_artifact_id(v), presentation.artifact_id]


def _audio_artifacts(v: StateView) -> list[Artifact]:
    """The audio plan, every AUDIO_ASSET and the timeline, once the timeline exists."""
    timeline = _timeline(v)
    if timeline is None:
        return []
    plan = v.output("store_audio_plan", StoredArtifacts).artifacts
    narration = _narration(v)
    assert narration is not None
    return [*plan, *narration.artifacts, timeline.artifact]


def _generated_video_artifacts(v: StateView) -> list[Artifact]:
    """The video segment plan and every GENERATED_VIDEO_ASSET, when the lesson asked for generated clips."""
    plan = v.maybe("store_video_segment_plan", StoredArtifacts)
    generated = _generated(v)
    return [*(plan.artifacts if plan else []), *(generated.artifacts if generated else [])]


def _video_artifacts(v: StateView) -> list[Artifact]:
    """The video plan, its subtitles and the VIDEO artifact, once they exist."""
    plan = v.maybe("store_video_plan", StoredArtifacts)
    video = _video(v)
    return [*(plan.artifacts if plan else []), *([video.artifact] if video and video.artifact else [])]


def _summarize(v: StateView) -> TaskResult:
    pedagogy = v.output("store_pedagogy", StoredArtifacts)
    research = v.output("store_research", StoredArtifacts)
    visuals = _visuals(v)
    stored = v.output("store_artifacts", StoredArtifacts)
    slide_plan = v.maybe("store_slide_plan", StoredArtifacts)
    presentation = _presentation(v)
    presentation_artifacts = [*(slide_plan.artifacts if slide_plan else []), *([presentation] if presentation else [])]
    update = v.output("update_learner", MasteryUpdate)
    review = _review(v)
    return TaskResult(
        title=_lesson(v).title,
        artifacts=[ArtifactSummary(artifact_id=a.artifact_id, type=a.type, name=a.name, version=a.version,
                                   uri=a.uri, parent_ids=a.parent_ids)
                   for a in [*pedagogy.artifacts, *research.artifacts, *(visuals.artifacts if visuals else []),
                             *stored.artifacts,
                             *presentation_artifacts, *_audio_artifacts(v), *_generated_video_artifacts(v),
                             *_video_artifacts(v)]],
        mastery_changes=merge_changes(v.output("record_diagnostic", MasteryUpdate), update),
        review_verdict=review.final_review.verdict.value if review.status == "approved" else review.status,
        revisions=review.revisions,
        estimated_level=update.estimated_level,
        warnings=[*_research(v).warnings, *_visual_warnings(v), *_presentation_warnings(v), *_audio_warnings(v),
                  *_generated_video_warnings(v), *_video_warnings(v)],
    )


def lesson_template(options: LessonWorkflowOptions) -> WorkflowTemplate:
    rounds = options.diagnostic_rounds
    return WorkflowTemplate(
        id=WORKFLOW_ID,
        description="Adaptive lesson: diagnostic evidence, learner model, knowledge gaps and a deterministic "
                    "pedagogical plan, then research, a reviewed lesson, visuals, a presentation, its narration and "
                    "the video.",
        provides=frozenset({"lesson.text", "learner.model", "pedagogy.plan", "lesson.review", "lesson.visuals", "slides.plan", "presentation.pptx",
                            "audio.narration", "presentation.timeline", "video.mp4",
                            GENERATED_VIDEO_CAPABILITY}),
        build=lambda request: build_lesson_workflow(request, options),
        provider_capabilities=PROVIDER_CAPABILITIES,
        expected_calls=(
            ExpectedCall("request_interpreter", 700, 200),
            ExpectedCall("knowledge_diagnostic", 3000, 1200, calls=rounds),
            ExpectedCall("research", 5000, 2500),
            ExpectedCall("curriculum_planner", 7000, 1800),
            ExpectedCall("teacher", 8000, 3500, calls=2),
            ExpectedCall("content_reviewer", 10000, 900, calls=2),
            ExpectedCall("visual", 9000, 1500),
            ExpectedCall("slide_planner", 9000, 3000),
            ExpectedCall("audio_planner", 6000, 2500),
        ),
    )
