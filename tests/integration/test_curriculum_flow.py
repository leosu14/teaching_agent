"""Learner goals and curricula end to end: goal -> curriculum planning task -> next action -> lesson from the
objective -> evaluation -> mastery -> curriculum progress -> next action. Plus resume, idempotency, the model's
boundary, privacy, events and backward compatibility for learners without curricula."""

from __future__ import annotations

import json

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.artifact import ArtifactType
from app.schemas.curriculum import GoalInput, LearningActionType
from app.schemas.events import Event, EventType
from app.schemas.learner import LearnerProfileInput, LearningGoal
from app.schemas.task import TaskStatus
from app.services.container import build_container
from app.services.curriculum import InvalidGoal
from scripts.run_curriculum_demo import FIXTURES, answer_diagnostic, placement_evidence, run_curriculum_demo

FIXTURE = json.loads((FIXTURES / "learner.json").read_text(encoding="utf-8"))
LEARNER = FIXTURE["learner_id"]
A, B, C, D = "es.present", "es.preterite", "es.past_contrast", "es.subjunctive"


class SimulatedCrash(BaseException):
    """The process dies: not an Exception, so nothing in the runtime catches it."""


def crash_after(node_id: str):
    def observer(task, finished: str) -> None:
        if finished == node_id:
            raise SimulatedCrash(node_id)
    return observer


@pytest.fixture
def curriculum_settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", corpus_dir=FIXTURES, log_json=False)


def make(settings: Settings, responders=None, observers=()):
    llm = MockLLMProvider(responders or default_responders())
    container = build_container(settings, llm_providers={"mock": llm}, observers=observers)
    seen: list[Event] = []
    container.events.subscribe(seen.append)
    return container, llm, seen


async def placed(container) -> LearningGoal:
    container.learner_service.upsert(LEARNER, LearnerProfileInput.model_validate(FIXTURE["profile"]))
    for domain, evidence in placement_evidence(LEARNER, FIXTURE["placement"]).items():
        await container.learner_service.record_evidence(LEARNER, domain, evidence)
    goal, _ = await container.curriculum_service.create_goal(LEARNER, GoalInput.model_validate(FIXTURE["goal"]))
    return goal


async def test_curriculum_demo_end_to_end(curriculum_settings) -> None:
    container, _, _ = make(curriculum_settings)
    lines: list[str] = []
    try:
        assert await run_curriculum_demo(container, out=lines.append), "\n".join(lines)
    finally:
        container.close()


async def test_planning_task_artifacts_events_and_privacy(curriculum_settings) -> None:
    container, llm, seen = make(curriculum_settings)
    goal = await placed(container)
    task = await container.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    assert task.status == TaskStatus.COMPLETED, task.errors
    plan = task.result.curriculum
    assert plan is not None and plan.version == 1 and plan.changed
    arts = container.task_service.artifacts(task.task_id)
    by_type = {}
    for a in arts:
        by_type.setdefault(a.type, []).append(a)
    assert len(by_type[ArtifactType.LEARNING_GOAL]) == 1 and len(by_type[ArtifactType.CURRICULUM]) == 1
    version_artifact = by_type[ArtifactType.CURRICULUM_VERSION][0]
    assert version_artifact.parent_ids == [by_type[ArtifactType.LEARNING_GOAL][0].artifact_id]
    assert len(by_type[ArtifactType.LEARNING_OBJECTIVE]) == 4
    assert all(o.parent_ids == [version_artifact.artifact_id] for o in by_type[ArtifactType.LEARNING_OBJECTIVE])
    stored = container.curriculum_service.versions(goal.goal_id)[0]
    assert stored.artifact_ids["version"] == version_artifact.artifact_id and stored.task_id == task.task_id

    types = [e.type for e in seen]
    assert EventType.GOAL_CREATED in types and types.count(EventType.CURRICULUM_CREATED) == 1
    created = next(e for e in seen if e.type == EventType.CURRICULUM_CREATED)
    assert created.data["version"] == 1 and "api_key" not in json.dumps(created.data)

    # The model saw the brief only: no learner, goal or task ids, no evidence, no numbers.
    request = next(r for r in llm.requests if r.agent_id == "learning_path_planner")
    payload = json.dumps(request.input_payload) + "".join(m.content for m in request.messages)
    for private in (LEARNER, goal.goal_id, task.task_id, "evidence", "mastery\":", "0.35"):
        assert private not in payload, private


async def test_crash_during_planning_resumes_without_duplicates(curriculum_settings) -> None:
    first, first_llm, _ = make(curriculum_settings, observers=[crash_after("store_artifacts")])
    goal = await placed(first)
    with pytest.raises(SimulatedCrash):
        await first.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    assert first_llm.calls["learning_path_planner"] == 1
    first.close()

    second, second_llm, _ = make(curriculum_settings)
    assert second.curriculum_service.versions(goal.goal_id) == []  # not committed before the crash
    tasks = second.task_service.list_for_learner(LEARNER)
    assert len(tasks) == 1 and tasks[0].status == TaskStatus.RUNNING
    # Building again resumes the unfinished task instead of starting a second one.
    resumed = await second.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    assert resumed.task_id == tasks[0].task_id and resumed.status == TaskStatus.COMPLETED, resumed.errors
    assert second_llm.calls["learning_path_planner"] == 0  # the proposal was checkpointed
    versions = second.curriculum_service.versions(goal.goal_id)
    assert len(versions) == 1
    arts = second.task_service.artifacts(resumed.task_id)
    assert sum(1 for a in arts if a.type == ArtifactType.CURRICULUM_VERSION) == 1
    assert len(second.task_service.list_for_learner(LEARNER)) == 1

    # A fresh, uninterrupted build of the same goal and state gives the same version id.
    ref_settings = curriculum_settings.model_copy(update={"data_dir": curriculum_settings.data_dir / "ref"})
    reference, _, _ = make(ref_settings)
    ref_goal = await placed(reference)
    ref_task = await reference.curriculum_service.build_curriculum(ref_goal.goal_id, user_id="u1")
    assert ref_goal.goal_id == goal.goal_id
    assert ref_task.result.curriculum.version_id == versions[0].version_id


async def test_crash_after_save_commits_once(curriculum_settings) -> None:
    first, _, _ = make(curriculum_settings, observers=[crash_after("save")])
    goal = await placed(first)
    with pytest.raises(SimulatedCrash):
        await first.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    first.close()
    second, _, seen = make(curriculum_settings)
    task = second.task_service.list_for_learner(LEARNER)[0]
    resumed = await second.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    assert len(second.curriculum_service.versions(goal.goal_id)) == 1
    assert EventType.CURRICULUM_CREATED not in [e.type for e in seen]  # not reported twice


async def test_idempotent_goal_creation_and_rebuild(curriculum_settings) -> None:
    container, _, _ = make(curriculum_settings)
    goal = await placed(container)
    again, created = await container.curriculum_service.create_goal(LEARNER, GoalInput.model_validate(FIXTURE["goal"]))
    assert not created and again == goal and len(container.curriculum_service.goals(LEARNER)) == 1
    with pytest.raises(InvalidGoal):  # the same key for a different goal
        await container.curriculum_service.create_goal(
            LEARNER, GoalInput.model_validate({**FIXTURE["goal"], "target_level": "B2"}))
    with pytest.raises(InvalidGoal):  # concepts the knowledge base does not know
        await container.curriculum_service.create_goal(
            LEARNER, GoalInput(title="t", domain="spanish", target_concepts=["es.invented"]))
    first = await container.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    second = await container.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    assert not second.result.curriculum.changed
    assert second.result.curriculum.version_id == first.result.curriculum.version_id
    assert len(container.curriculum_service.versions(goal.goal_id)) == 1
    assert not [a for a in container.task_service.artifacts(second.task_id)
                if a.type == ArtifactType.CURRICULUM_VERSION]


async def test_invalid_curriculum_fails_before_any_lesson(curriculum_settings) -> None:
    container, _, _ = make(curriculum_settings)
    await placed(container)
    # A goal stored directly (bypassing the service's checks) with a concept the knowledge base does not know.
    bad = container.memory.save_goal(LearningGoal(goal_id="goal-bad", learner_id=LEARNER, domain="spanish",
                                                  target_concepts=["es.invented"], title="Bad"))
    task = await container.curriculum_service.build_curriculum(bad.goal_id, user_id="u1")
    assert task.status == TaskStatus.FAILED and "does not know" in task.errors[-1].message
    assert container.curriculum_service.versions(bad.goal_id) == []
    assert (await container.curriculum_service.next_action(LEARNER)).action == LearningActionType.WAIT


def _mastery(container, concept_id: str) -> float:
    return container.memory.get(LEARNER).concepts[concept_id].mastery


@pytest.mark.parametrize("extra", [{"mastery": 1.0}, {"status": "MASTERED"}, {"goal_complete": True}])
async def test_model_cannot_write_mastery_status_or_completion(curriculum_settings, extra) -> None:
    def overreach(request):
        payload = default_responders()["learning_path_planner"](request)
        payload["objectives"][0].update(extra)
        return payload

    container, llm, _ = make(curriculum_settings, responders={**default_responders(),
                                                              "learning_path_planner": overreach})
    goal = await placed(container)
    before = {c: _mastery(container, c) for c in (A, B, C, D)}
    task = await container.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    assert task.status == TaskStatus.FAILED  # the proposal never validates
    assert llm.calls["learning_path_planner"] >= 2  # corrected, then given up
    assert container.curriculum_service.versions(goal.goal_id) == []
    assert {c: _mastery(container, c) for c in (A, B, C, D)} == before
    assert container.curriculum_service.goal(goal.goal_id).status.value == "ACTIVE"


async def test_unknown_concepts_in_the_proposal_are_rejected_not_stored(curriculum_settings) -> None:
    def invent(request):
        payload = default_responders()["learning_path_planner"](request)
        payload["objectives"].append({"concept_id": "es.invented", "description": "Invented"})
        return payload

    container, _, _ = make(curriculum_settings, responders={**default_responders(),
                                                            "learning_path_planner": invent})
    goal = await placed(container)
    task = await container.curriculum_service.build_curriculum(goal.goal_id, user_id="u1")
    # The agent asks the model to stay within the brief; the model insists, so the task fails without a version.
    assert task.status == TaskStatus.FAILED
    assert container.curriculum_service.versions(goal.goal_id) == []


async def test_learners_without_goals_keep_the_adaptive_loop(curriculum_settings) -> None:
    container, _, seen = make(curriculum_settings)
    container.learner_service.upsert(LEARNER, LearnerProfileInput.model_validate(FIXTURE["profile"]))
    for domain, evidence in placement_evidence(LEARNER, FIXTURE["placement"]).items():
        await container.learner_service.record_evidence(LEARNER, domain, evidence)
    task = await container.task_service.create_and_run(request="Create a B1 lesson about preterite.",
                                                       learner_id=LEARNER, user_id="u1")
    task = await answer_diagnostic(container, task, lambda *_: None)
    assert task.status == TaskStatus.COMPLETED, task.errors
    names = {a.name for a in container.task_service.artifacts(task.task_id)}
    assert "learning_action" not in names
    plan = next(a for a in container.task_service.artifacts(task.task_id) if a.name == "pedagogical_plan")
    assert json.loads(container.artifacts.read(plan.artifact_id)).get("focus") is None
    evaluation = await container.task_service.start_evaluation(task.task_id, user_id="u1")
    answers = [{"question_id": q["question_id"], "answer": "cenamos"} for q in evaluation.waiting.prompt["questions"]]
    evaluation = await container.task_service.submit_answers(evaluation.task_id, {"answers": answers})
    assert evaluation.status == TaskStatus.COMPLETED, evaluation.errors
    assert evaluation.result.learning_action is None and evaluation.result.next_recommendation is not None
    assert "learning_action" not in {a.name for a in container.task_service.artifacts(evaluation.task_id)}
    assert not [e for e in seen if e.type.startswith(("curriculum.", "objective.", "goal.completed"))]
