"""Lesson-generation workflow template (the first vertical slice).

snapshot -> adaptive diagnostic (ask / wait for answers / re-assess, up to N rounds) -> research -> plan
-> teach/review/revise loop -> slide plan -> artifacts -> learner memory.
"""

from __future__ import annotations

import json
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
from app.schemas.artifact import ArtifactBatch, ArtifactDraft, ArtifactType, StoredArtifacts
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
    ResearchBundle,
    ResearchRequest,
    ReviewerInput,
    RevisionContext,
    SlideDeckPlan,
    SlideInput,
    TeacherInput,
)
from app.schemas.task import ArtifactSummary, TaskResult
from app.schemas.workflow import ReviewOutcome, RevisionPolicy, RevisionRequest

WORKFLOW_ID = "lesson_generation"


@dataclass(frozen=True)
class LessonWorkflowOptions:
    diagnostic_rounds: int = 2
    memory_confidence: float = 0.6
    revision_policy: RevisionPolicy = field(default_factory=RevisionPolicy)


def _request(v: StateView) -> LessonRequest:
    assert v.task.plan is not None
    return v.task.plan.lesson_request


def _review(v: StateView) -> ReviewOutcome:
    return v.output("teach_review", ReviewOutcome)


def _lesson(v: StateView) -> LessonContent:
    return LessonContent.model_validate(_review(v).candidate)


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
        return TeacherInput(
            request=_request(v), plan=v.output("plan", LessonPlan), research=v.output("research", ResearchBundle),
            snapshot=v.output("learner_snapshot", LearnerSnapshot),
            revision=None if revision is None else RevisionContext(
                revision_number=revision.revision_number, issues=revision.issues,
                previous=LessonContent.model_validate(revision.previous),
            ),
        )

    def reviewer_input(v: StateView, candidate, revision_number: int) -> ReviewerInput:
        return ReviewerInput(request=_request(v), plan=v.output("plan", LessonPlan),
                             research=v.output("research", ResearchBundle), content=candidate,
                             revision_number=revision_number)

    nodes += [
        TransformNode(id="diagnostic", fn=final_diagnostic, depends_on=("diagnose_1",),
                      after=tuple(diagnose_ids[1:]) + tuple(f"answers_{r}" for r in range(1, rounds + 1))),
        AgentNode(id="research", agent="knowledge_research", depends_on=("diagnostic",),
                  build_input=lambda v: ResearchRequest(request=_request(v), diagnostic=diagnostic(v).result,
                                                        concepts=diagnostic(v).concepts)),
        AgentNode(id="plan", agent="curriculum_planner", depends_on=("research",),
                  build_input=lambda v: PlannerInput(
                      request=_request(v), snapshot=v.output("learner_snapshot", LearnerSnapshot),
                      diagnostic=diagnostic(v).result, research=v.output("research", ResearchBundle),
                      concepts=diagnostic(v).concepts)),
        ReviewNode(id="teach_review", depends_on=("plan",), generator="teacher", reviewer="content_reviewer",
                   candidate_model=LessonContent, build_generator_input=teacher_input,
                   build_reviewer_input=reviewer_input, policy=options.revision_policy),
        AgentNode(id="slides", agent="slide_generation", depends_on=("teach_review",),
                  build_input=lambda v: SlideInput(lesson=_lesson(v), plan=v.output("plan", LessonPlan))),
        TransformNode(id="package_artifacts", depends_on=("slides",), fn=_package),
        ToolNode(id="store_artifacts", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("package_artifacts",), build_input=lambda v: v.output("package_artifacts", ArtifactBatch)),
        ToolNode(id="update_learner", tool="learner.record_lesson", permissions=frozenset({"learner:write"}),
                 depends_on=("store_artifacts",), build_input=_lesson_outcome),
    ]
    return WorkflowDefinition(id=WORKFLOW_ID, nodes=tuple(nodes), summarize=_summarize,
                              description="Diagnose, research, plan, teach with review, plan slides, store, remember.")


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
    research = v.output("research", ResearchBundle)
    slides = v.output("slides", SlideDeckPlan)
    sources = {"sources": [s.model_dump(mode="json") for s in research.sources],
               "facts": [f.model_dump(mode="json") for f in research.facts]}
    return ArtifactBatch(drafts=[
        ArtifactDraft(key="sources", name="sources", type=ArtifactType.REPORT, media_type="application/json",
                      content=json.dumps(sources, ensure_ascii=False, indent=2),
                      metadata={"kind": "research_sources", "reliable": sum(s.reliable for s in research.sources)}),
        ArtifactDraft(key="lesson", name="lesson", type=ArtifactType.LESSON, media_type="application/json",
                      content=_json(lesson), parent_keys=["sources"],
                      metadata={"title": lesson.title, "level": lesson.level, "sections": len(lesson.sections)}),
        ArtifactDraft(key="script", name="narration_script", type=ArtifactType.SCRIPT, media_type="text/markdown",
                      content=_script(lesson), parent_keys=["lesson"]),
        ArtifactDraft(key="slide_plan", name="slide_plan", type=ArtifactType.SLIDE_PLAN,
                      media_type="application/json", content=_json(slides), parent_keys=["lesson"],
                      metadata={"slides": len(slides.slides)}),
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
        artifact_ids=[a.artifact_id for a in stored.artifacts],
    )


def _summarize(v: StateView) -> TaskResult:
    stored = v.output("store_artifacts", StoredArtifacts)
    update = v.output("update_learner", MasteryUpdate)
    review = _review(v)
    return TaskResult(
        title=_lesson(v).title,
        artifacts=[ArtifactSummary(artifact_id=a.artifact_id, type=a.type, name=a.name, version=a.version,
                                   uri=a.uri, parent_ids=a.parent_ids) for a in stored.artifacts],
        mastery_changes=update.changes,
        review_verdict=review.final_review.verdict.value if review.status == "approved" else review.status,
        revisions=review.revisions,
        estimated_level=update.estimated_level,
    )


def lesson_template(options: LessonWorkflowOptions) -> WorkflowTemplate:
    rounds = options.diagnostic_rounds
    return WorkflowTemplate(
        id=WORKFLOW_ID,
        description="Personalised text lesson with diagnostic, research, review loop and slide plan.",
        provides=frozenset({"lesson.text", "lesson.review", "slides.plan"}),
        build=lambda request: build_lesson_workflow(request, options),
        expected_calls=(
            ExpectedCall("request_interpreter", 700, 200),
            ExpectedCall("knowledge_diagnostic", 3000, 1200, calls=rounds),
            ExpectedCall("knowledge_research", 5000, 2500),
            ExpectedCall("curriculum_planner", 7000, 1800),
            ExpectedCall("teacher", 8000, 3500, calls=2),
            ExpectedCall("content_reviewer", 10000, 900, calls=2),
            ExpectedCall("slide_generation", 4000, 2000),
        ),
    )
