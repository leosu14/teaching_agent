"""Checkpoint/resume: a process killed after the Planner resumes without re-running earlier nodes."""

from __future__ import annotations

import pytest

from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.artifact import ArtifactType
from app.schemas.lesson import LessonContent
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import add_demo_learner, answers_for, run_lesson


class SimulatedCrash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing in the runtime catches it."""


def crash_after(node_id: str):
    def observer(task, finished: str) -> None:
        if finished == node_id:
            raise SimulatedCrash(node_id)
    return observer


async def test_kill_after_planner_then_resume(settings) -> None:
    first_llm = MockLLMProvider(default_responders())
    first = build_container(settings, llm_providers={"mock": first_llm}, observers=[crash_after("plan")])
    learner_id = add_demo_learner(first)
    task = await first.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                   learner_id=learner_id, user_id="u1")
    task = await first.task_service.submit_assessment(task.task_id, answers_for(task))
    with pytest.raises(SimulatedCrash):
        await first.task_service.submit_assessment(task.task_id, answers_for(task))
    first.close()

    # What survived the crash is exactly the persisted checkpoint.
    second_llm = MockLLMProvider(default_responders())
    second = build_container(settings, llm_providers={"mock": second_llm})
    crashed = second.task_service.get(task.task_id)
    assert crashed.status == TaskStatus.RUNNING
    assert crashed.workflow.node_states["plan"].status.value == "COMPLETED"
    assert crashed.workflow.node_states["teach_review"].status.value == "PENDING"
    # Only the research bundle was stored before the crash; it is part of the checkpointed past.
    assert [a.name for a in second.task_service.artifacts(task.task_id)] == ["research_bundle"]
    cost_before = crashed.cost.actual_cost_usd

    resumed = await second.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors

    # Nothing before the Planner ran again, in the new process.
    for agent_id in ("request_interpreter", "knowledge_diagnostic", "research", "curriculum_planner"):
        assert second_llm.calls[agent_id] == 0, agent_id
    assert second_llm.calls["teacher"] == 2 and second_llm.calls["slide_planner"] == 1
    events = second.task_service.events(task.task_id)
    for node in ("learner_snapshot", "diagnose_1", "diagnose_2", "diagnose_3", "research", "store_research", "plan"):
        assert sum(1 for e in events if e.type == "node.started" and e.node_id == node) == 1, node
    for node in ("teach_review", "slide_plan", "store_artifacts", "update_learner"):
        assert sum(1 for e in events if e.type == "node.started" and e.node_id == node) == 1, node

    # Final artifacts are correct, and the cost includes both processes' work.
    arts = {a.name: a for a in second.task_service.artifacts(task.task_id)}
    assert set(arts) == {"research_bundle", "lesson_plan", "lesson", "narration_script", "slide_plan",
                         "review_report", "visual_plan", "image_v1_photo", "image_v1_diagram",
                         "image_v2_illustration", "image_v2_diagram", "presentation"}
    assert arts["lesson"].type == ArtifactType.LESSON and arts["lesson"].version == 1
    assert arts["slide_plan"].parent_ids == [arts["lesson"].artifact_id]
    assert resumed.cost.actual_cost_usd > cost_before
    assert resumed.result.revisions == 1

    # The resumed result matches an uninterrupted run.
    reference = build_container(settings.model_copy(update={"data_dir": settings.data_dir / "ref"}),
                                llm_providers={"mock": MockLLMProvider(default_responders())})
    clean = await run_lesson(reference)
    clean_lesson = reference.artifacts.read(next(a.artifact_id for a in clean.result.artifacts if a.name == "lesson"))
    # Identical apart from retrieval timestamps in the resolved references and the (per-task) image artifact ids.
    resumed_lesson = LessonContent.model_validate_json(second.artifacts.read(arts["lesson"].artifact_id))
    clean_lesson = LessonContent.model_validate_json(clean_lesson)
    exclude = {"references": True, "sections": {"__all__": {"visuals": {"__all__": {"artifact_id"}}}}}
    assert resumed_lesson.model_dump(exclude=exclude) == clean_lesson.model_dump(exclude=exclude)
    assert ([r.model_dump(exclude={"retrieved_at"}) for r in resumed_lesson.references]
            == [r.model_dump(exclude={"retrieved_at"}) for r in clean_lesson.references])
    assert [c.model_dump() for c in resumed.result.mastery_changes] == [c.model_dump() for c in clean.result.mastery_changes]
    reference.close()
    second.close()


async def test_crash_inside_review_loop_resumes_mid_loop(settings) -> None:
    llm = MockLLMProvider(default_responders())
    first = build_container(settings, llm_providers={"mock": llm})
    add_demo_learner(first)

    class Stop(BaseException):
        pass

    def stop_on_review(event):
        if event.type == "review.completed" and event.data.get("revision") == 0:
            raise Stop()

    # Kill the process right after the first review round is checkpointed.
    first.events.subscribe(stop_on_review)
    task = await first.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                   learner_id="demo-learner", user_id="u1")
    task = await first.task_service.submit_assessment(task.task_id, answers_for(task))
    with pytest.raises(Stop):
        await first.task_service.submit_assessment(task.task_id, answers_for(task))
    first.close()

    llm2 = MockLLMProvider(default_responders())
    second = build_container(settings, llm_providers={"mock": llm2})
    saved = second.task_service.get(task.task_id)
    assert saved.status == TaskStatus.REVIEWING
    assert len(saved.workflow.node_states["teach_review"].progress["rounds"]) == 1
    done = await second.task_service.resume(task.task_id)
    assert done.status == TaskStatus.COMPLETED
    assert done.result.revisions == 1
    assert llm2.calls["teacher"] == 1 and llm2.calls["content_reviewer"] == 1  # only the revision round re-ran
    second.close()
