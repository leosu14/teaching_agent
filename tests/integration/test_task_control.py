"""Pause, resume, cancel and WAITING behaviour of tasks."""

from __future__ import annotations

import pytest

from app.runtime.tasks.state_machine import InvalidTransition
from app.schemas.lesson import DiagnosticAnswers, LearnerAnswer
from app.schemas.task import TaskStatus
from app.runtime.orchestrator.orchestrator import InvalidInput
from tests.conftest import add_demo_learner, answers_for


async def start(container):
    learner_id = add_demo_learner(container)
    return await container.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                       learner_id=learner_id, user_id="u1")


async def test_waiting_task_persists_questions_and_resumes_later(container) -> None:
    task = await start(container)
    assert task.status == TaskStatus.WAITING
    stored = container.task_service.get(task.task_id)
    assert stored.status == TaskStatus.WAITING and stored.waiting.node_id == "answers_1"
    assert stored.workflow.node_states["answers_1"].status.value == "WAITING"
    types = [e.type for e in container.task_service.events(task.task_id)]
    assert types[-1] == "task.waiting"


async def test_invalid_answers_are_rejected_and_task_keeps_waiting(container) -> None:
    task = await start(container)
    with pytest.raises(InvalidInput):
        await container.orchestrator.submit_input(task.task_id, "answers_1", {"answers": []})
    assert container.task_service.get(task.task_id).status == TaskStatus.WAITING
    with pytest.raises(InvalidTransition):
        await container.orchestrator.submit_input(task.task_id, "answers_2", {"answers": [{"question_id": "q", "answer": "a"}]})


async def test_pause_takes_effect_at_next_node_boundary_then_resume(container, mock_llm) -> None:
    task = await start(container)
    container.task_service.pause(task.task_id)
    paused = await container.task_service.submit_assessment(task.task_id, answers_for(task))
    assert paused.status == TaskStatus.PAUSED
    assert paused.workflow.node_states["answers_1"].human_input is not None  # the answers were recorded
    assert mock_llm.calls["knowledge_diagnostic"] == 1  # but nothing after it ran
    resumed = await container.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.WAITING and resumed.waiting.node_id == "answers_2"
    done = await container.task_service.submit_assessment(task.task_id, answers_for(resumed))
    assert done.status == TaskStatus.COMPLETED
    types = [e.type for e in container.task_service.events(task.task_id)]
    assert "task.paused" in types and "task.resumed" in types


async def test_cancel_waiting_task(container) -> None:
    task = await start(container)
    cancelled = container.task_service.cancel(task.task_id)
    assert cancelled.status == TaskStatus.CANCELLED
    with pytest.raises(InvalidTransition):
        await container.task_service.submit_assessment(task.task_id, DiagnosticAnswers(
            answers=[LearnerAnswer(question_id="x", answer="y")]))
    with pytest.raises(InvalidTransition):
        await container.task_service.resume(task.task_id)


async def test_failed_task_can_be_recovered(container, mock_llm) -> None:
    task = await start(container)
    task = await container.task_service.submit_assessment(task.task_id, answers_for(task))
    # Research fails on every attempt of every retry layer -> the task fails at the research node.
    mock_llm.inject("research", *(["not json"] * 6))
    failed = await container.task_service.submit_assessment(task.task_id, answers_for(task))
    assert failed.status == TaskStatus.FAILED and failed.errors[-1].node_id == "research"
    assert failed.workflow.node_states["research"].attempts == 2  # node retry policy
    recovered = await container.task_service.resume(task.task_id)
    assert recovered.status == TaskStatus.COMPLETED
    assert mock_llm.calls["knowledge_diagnostic"] == 3  # diagnostics were not repeated
