"""Lesson-generation workflow template (the first vertical slice).

snapshot -> adaptive diagnostic (ask / wait for answers / re-assess, up to N rounds) -> research
-> research policy -> research artifact -> plan -> teach/review/revise loop -> visuals (only for an approved
lesson) -> visual policy -> lesson artifacts -> slide planning -> slide plan validation -> presentation build ->
presentation render (only for an approved lesson) -> audio planning -> audio plan validation -> TTS, audio validation
and AUDIO_ASSETs -> audio policy -> presentation timeline (only after the presentation is rendered) -> learner memory.
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
from app.runtime.workflows.research_policy import ResearchRequirement, apply_research_policy
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
from app.schemas.learner import LearnerSnapshot, MasteryUpdate
from app.schemas.lesson import (
    DiagnosticAnswers,
    DiagnosticInput,
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
from app.schemas.research import ResearchBundle
from app.schemas.task import ArtifactSummary, TaskResult
from app.schemas.visual import VisualResult
from app.schemas.workflow import NodeStatus, ReviewOutcome, RevisionPolicy, RevisionRequest

WORKFLOW_ID = "lesson_generation"


@dataclass(frozen=True)
class LessonWorkflowOptions:
    diagnostic_rounds: int = 2
    memory_confidence: float = 0.6
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


def _request(v: StateView) -> LessonRequest:
    assert v.task.plan is not None
    return v.task.plan.lesson_request


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
        )
    ]

    def diagnostic_input(round_number: int):
        def build(v: StateView) -> DiagnosticInput:
            history = [
                DiagnosticRound(items=v.output(f"diagnose_{k}", DiagnosticStep).items,
                                answers=v.output(f"answers_{k}", DiagnosticAnswers).answers)
                for k in range(1, round_number)
            ]
            return DiagnosticInput(
                request=_request(v), snapshot=v.output("learner_snapshot", LearnerSnapshot), rounds=history,
                round_number=round_number, max_rounds=rounds, memory_confidence_threshold=options.memory_confidence,
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
                      depends_on=("learner_snapshot",) if r == 1 else (f"answers_{r - 1}",)),
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
        return TeacherInput(
            request=_request(v), plan=plan, research=_research(v).focused(focus),
            snapshot=v.output("learner_snapshot", LearnerSnapshot),
            revision=None if revision is None else RevisionContext(
                revision_number=revision.revision_number, issues=revision.issues,
                previous=LessonContent.model_validate(revision.previous),
            ),
        )

    def reviewer_input(v: StateView, candidate, revision_number: int) -> ReviewerInput:
        return ReviewerInput(request=_request(v), plan=v.output("plan", LessonPlan),
                             research=_research(v), content=candidate,
                             revision_number=revision_number)

    nodes += [
        TransformNode(id="diagnostic", fn=final_diagnostic, depends_on=("diagnose_1",),
                      after=tuple(diagnose_ids[1:]) + tuple(f"answers_{r}" for r in range(1, rounds + 1))),
        AgentNode(id="research", agent="research", depends_on=("diagnostic",),
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
                      request=_request(v), snapshot=v.output("learner_snapshot", LearnerSnapshot),
                      diagnostic=diagnostic(v).result, research=_research(v), concepts=diagnostic(v).concepts)),
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
                      max_candidates=options.visual_max_candidates, parent_artifact_ids=[_research_artifact_id(v)])),
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
        ToolNode(id="audio_timeline", tool="audio.timeline", permissions=frozenset({"artifact:write"}),
                 depends_on=("audio_policy",),
                 build_input=lambda v: TimelineRequest(
                     plan=_audio_plan(v), narration=v.output("audio_policy", NarrationResult),
                     slide_ids=[s.slide_id for s in _deck(v).slides],
                     presentation_artifact_id=_presentation_artifact_id(v),
                     audio_plan_artifact_id=_audio_plan_artifact_id(v),
                     silent_slide_seconds=options.audio_silent_slide_seconds)),
        ToolNode(id="update_learner", tool="learner.record_lesson", permissions=frozenset({"learner:write"}),
                 depends_on=("store_artifacts",), after=("render_presentation", "audio_timeline"),
                 build_input=_lesson_outcome),
    ]
    return WorkflowDefinition(id=WORKFLOW_ID, nodes=tuple(nodes), summarize=_summarize,
                              description="Diagnose, research, plan, teach with review, illustrate, store, plan, "
                                          "build and render the presentation, narrate it, time it, remember.")


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
    return ArtifactBatch(drafts=[
        ArtifactDraft(key="lesson_plan", name="lesson_plan", type=ArtifactType.LESSON_PLAN,
                      media_type="application/json", content=_json(v.output("plan", LessonPlan)),
                      parent_ids=[research_artifact], metadata={"objectives": len(v.output("plan", LessonPlan).objectives)}),
        ArtifactDraft(key="lesson", name="lesson", type=ArtifactType.LESSON, media_type="application/json",
                      content=_json(lesson), parent_keys=["lesson_plan"], parent_ids=[research_artifact, *image_ids],
                      metadata={"title": lesson.title, "level": lesson.level, "sections": len(lesson.sections),
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
        artifact_ids=[_research_artifact_id(v), *_visual_artifact_ids(v),
                      *(a.artifact_id for a in stored.artifacts), *_presentation_artifact_ids(v),
                      *(a.artifact_id for a in _audio_artifacts(v))],
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


def _summarize(v: StateView) -> TaskResult:
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
                   for a in [*research.artifacts, *(visuals.artifacts if visuals else []), *stored.artifacts,
                             *presentation_artifacts, *_audio_artifacts(v)]],
        mastery_changes=update.changes,
        review_verdict=review.final_review.verdict.value if review.status == "approved" else review.status,
        revisions=review.revisions,
        estimated_level=update.estimated_level,
        warnings=[*_research(v).warnings, *_visual_warnings(v), *_presentation_warnings(v), *_audio_warnings(v)],
    )


def lesson_template(options: LessonWorkflowOptions) -> WorkflowTemplate:
    rounds = options.diagnostic_rounds
    return WorkflowTemplate(
        id=WORKFLOW_ID,
        description="Personalised text lesson with diagnostic, research, review loop, visuals, a presentation and "
                    "its narration.",
        provides=frozenset({"lesson.text", "lesson.review", "lesson.visuals", "slides.plan", "presentation.pptx",
                            "audio.narration", "presentation.timeline"}),
        build=lambda request: build_lesson_workflow(request, options),
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
