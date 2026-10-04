"""Curriculum-planning workflow template: a learning goal -> a validated, versioned curriculum.

concept graph -> learning goal -> learner model -> deterministic draft (scope, prerequisite order, targets,
feasibility) -> learning-path planner (the model words the objectives) -> validation and version decision ->
[only for a new version] curriculum artifacts (LEARNING_GOAL -> CURRICULUM_VERSION -> LEARNING_OBJECTIVE, CURRICULUM)
-> stored version.

Every node is checkpointed: a task resumed after a crash reuses what it already computed and stores the version
once (the version id is derived from its content). An unchanged curriculum stores nothing.
"""

from __future__ import annotations

from datetime import datetime

from app.runtime.orchestrator.planner import ExpectedCall, WorkflowTemplate
from app.runtime.workflow.engine import WorkflowDefinition
from app.runtime.workflow.nodes import AgentNode, ConditionalNode, StateView, ToolNode, TransformNode
from app.schemas.artifact import ArtifactBatch, StoredArtifacts
from app.schemas.curriculum import CurriculumDraft, CurriculumProposal, ReplanReason
from app.schemas.learner import LearningGoal
from app.schemas.lesson import LessonRequest
from app.schemas.pedagogy import ConceptSet, LearnerModel
from app.schemas.task import ArtifactSummary, TaskResult
from app.tools.curriculum.artifacts import artifact_ids, curriculum_drafts
from app.tools.curriculum.tools import DraftRequest, FinalizedPlan, FinalizeRequest, SaveRequest, SaveResult

WORKFLOW_ID = "curriculum_planning"
CAPABILITY = "curriculum.plan"
GOAL_INPUT = "goal_id"  # TaskPlan.inputs: the goal to plan
REASON_INPUT = "reason"  # TaskPlan.inputs: why (a ReplanReason value)
AS_OF_KEY = "curriculum_as_of"  # Task.metadata: the planning time, fixed at creation so a resumed task matches


def _request(v: StateView) -> LessonRequest:
    assert v.task.plan is not None
    return v.task.plan.lesson_request


def _inputs(v: StateView) -> dict[str, str]:
    assert v.task.plan is not None
    return v.task.plan.inputs


def _concepts(v: StateView) -> ConceptSet:
    return v.output("knowledge_graph", ConceptSet)


def _goal(v: StateView) -> LearningGoal:
    return v.output("load_goal", LearningGoal)


def _finalized(v: StateView) -> FinalizedPlan:
    return v.output("finalize", FinalizedPlan)


def _package(v: StateView) -> ArtifactBatch:
    finalized = _finalized(v)
    return ArtifactBatch(drafts=curriculum_drafts(finalized.plan, _goal(v), finalized.previous_artifact_id))


def _summarize(v: StateView) -> TaskResult:
    plan = _finalized(v).plan
    stored = v.maybe("store_artifacts", StoredArtifacts)
    saved = v.maybe("save", SaveResult)
    title = f"Curriculum: {plan.title} (version {plan.version}" + (")" if plan.changed else ", unchanged)")
    return TaskResult(
        title=title,
        artifacts=[ArtifactSummary(artifact_id=a.artifact_id, type=a.type, name=a.name, version=a.version,
                                   uri=a.uri, parent_ids=a.parent_ids) for a in (stored.artifacts if stored else [])],
        mastery_changes=[], warnings=[w.message for w in plan.warnings],
        curriculum=plan if saved is None else plan.model_copy(update={"changed": saved.created or plan.changed}))


def build_curriculum_workflow(request: LessonRequest) -> WorkflowDefinition:
    nodes = (
        ToolNode(id="knowledge_graph", tool="knowledge.concepts", permissions=frozenset({"knowledge:read"}),
                 build_input=lambda v: {"domain": _request(v).subject}),
        ToolNode(id="load_goal", tool="learning_goal.resolve", permissions=frozenset({"learner:read"}),
                 depends_on=("knowledge_graph",),
                 build_input=lambda v: {"learner_id": v.task.learner_id, "domain": _request(v).subject,
                                        "topic": _request(v).topic, "goal_id": _inputs(v)[GOAL_INPUT]}),
        ToolNode(id="learner_model", tool="learner.model", permissions=frozenset({"learner:read"}),
                 depends_on=("load_goal",),
                 build_input=lambda v: {"learner_id": v.task.learner_id, "domain": _request(v).subject,
                                        "framework_id": _request(v).framework_id,
                                        "target_level": _goal(v).target_level,
                                        "concept_ids": [c.concept_id for c in _concepts(v).concepts]}),
        ToolNode(id="draft", tool="curriculum.draft", permissions=frozenset({"learner:read"}),
                 depends_on=("learner_model",),
                 build_input=lambda v: DraftRequest(
                     goal=_goal(v), model=v.output("learner_model", LearnerModel), concepts=_concepts(v).concepts,
                     as_of=datetime.fromisoformat(v.task.metadata[AS_OF_KEY]),
                     language=_request(v).language_of_instruction)),
        # The model sees the brief only: concepts, prerequisites, coarse state bands; no ids, evidence or history.
        AgentNode(id="propose", agent="learning_path_planner", depends_on=("draft",),
                  build_input=lambda v: v.output("draft", CurriculumDraft).brief),
        ToolNode(id="finalize", tool="curriculum.finalize", permissions=frozenset({"learner:read"}),
                 depends_on=("propose",),
                 build_input=lambda v: FinalizeRequest(
                     draft=v.output("draft", CurriculumDraft), concepts=_concepts(v).concepts,
                     proposal=v.output("propose", CurriculumProposal),
                     reasons=[ReplanReason(_inputs(v)[REASON_INPUT])] if _inputs(v).get(REASON_INPUT) else [])),
        ConditionalNode(id="new_version", depends_on=("finalize",), predicate=lambda v: _finalized(v).plan.changed,
                        when_true=("package_artifacts",)),
        TransformNode(id="package_artifacts", depends_on=("new_version",), fn=_package),
        ToolNode(id="store_artifacts", tool="artifact.store", permissions=frozenset({"artifact:write"}),
                 depends_on=("package_artifacts",), build_input=lambda v: v.output("package_artifacts", ArtifactBatch)),
        ToolNode(id="save", tool="curriculum.save", permissions=frozenset({"learner:write"}),
                 depends_on=("store_artifacts",),
                 build_input=lambda v: SaveRequest(
                     plan=_finalized(v).plan, domain=_request(v).subject,
                     artifact_ids=artifact_ids(_finalized(v).plan, v.output("store_artifacts", StoredArtifacts)))),
    )
    return WorkflowDefinition(id=WORKFLOW_ID, nodes=nodes, summarize=_summarize,
                              description="Plan a learner's curriculum for a goal: deterministic structure, model "
                                          "wording, validation, versioned storage.")


def curriculum_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        id=WORKFLOW_ID,
        description="Learning goal -> validated, versioned curriculum (objectives, prerequisites, targets).",
        provides=frozenset({CAPABILITY}),
        build=build_curriculum_workflow,
        expected_calls=(ExpectedCall("learning_path_planner", 3000, 1500),),
    )
