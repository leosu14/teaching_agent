"""The full deterministic vertical slice through TaskService (the same path the API and CLI use)."""

from __future__ import annotations

import json

from app.config.settings import Settings
from app.schemas.artifact import ArtifactType
from app.schemas.lesson import LessonContent, SlideDeckPlan
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import add_demo_learner, answers_for, run_lesson

REQUIRED_EVENTS = [
    "task.created", "task.started", "node.started", "node.finished", "agent.started", "agent.finished",
    "tool.started", "tool.finished", "review.completed", "artifact.created", "learner.updated", "task.completed",
]


async def test_request_to_completed_lesson(container, mock_llm) -> None:
    learner_id = add_demo_learner(container)
    task = await container.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                       learner_id=learner_id, user_id="u1")
    # The request was interpreted without any Spanish-specific code: the subject comes from the learner profile.
    assert task.plan is not None
    req = task.plan.lesson_request
    assert (req.subject, req.topic, req.framework_id, req.target_level) == ("spanish", "football", "cefr", "A2")

    # Adaptive diagnostic: round 1 asks one question per concept, round 2 only follows up on the misses.
    assert task.status == TaskStatus.WAITING
    first = task.waiting.prompt["questions"]
    assert len(first) == 4
    task = await container.task_service.submit_assessment(task.task_id, answers_for(task))
    assert task.status == TaskStatus.WAITING
    followups = task.waiting.prompt["questions"]
    assert {q["concept_id"] for q in followups} == {"es.football.preterite_match", "es.football.opinions"}
    assert all(q["difficulty"] < 0.5 for q in followups)
    task = await container.task_service.submit_assessment(task.task_id, answers_for(task))
    assert task.status == TaskStatus.COMPLETED, task.errors

    order = task.workflow.execution_order
    expected = ["learner_snapshot", "diagnose_1", "answers_1", "diagnose_2", "answers_2", "diagnose_3", "diagnostic",
                "research", "plan", "teach_review", "slides", "package_artifacts", "store_artifacts", "update_learner"]
    assert [n for n in order if not n.startswith("diagnostic_gate")] == expected

    # Artifacts and their dependency graph.
    arts = {a.name: a for a in container.task_service.artifacts(task.task_id)}
    assert arts["lesson"].type == ArtifactType.LESSON
    assert arts["lesson"].parent_ids == [arts["sources"].artifact_id]
    for child in ("narration_script", "slide_plan", "review_report"):
        assert arts[child].parent_ids == [arts["lesson"].artifact_id]
    lesson = LessonContent.model_validate_json(container.artifacts.read(arts["lesson"].artifact_id))
    slides = SlideDeckPlan.model_validate_json(container.artifacts.read(arts["slide_plan"].artifact_id))
    assert {s.narration_section_id for s in slides.slides if s.narration_section_id} == {s.section_id for s in lesson.sections}
    # Every section is traceable to reliable sources; the unreliable forum source was excluded.
    sources = json.loads(container.artifacts.read(arts["sources"].artifact_id))
    reliable = {s["source_id"] for s in sources["sources"] if s["reliable"]}
    assert any(not s["reliable"] and s["publisher"] == "Fan Forum" for s in sources["sources"])
    assert all(s.citations and set(s.citations) <= reliable for s in lesson.sections)
    # Gaps are taught; the concept the learner already knew is not.
    taught = {s.concept_id for s in lesson.sections}
    assert taught == {"es.football.preterite_match", "es.football.opinions"}

    # Learner memory.
    profile = container.learner_service.get(learner_id)
    assert profile.subjects["spanish"].estimated_level == "A2"
    assert len(profile.lessons) == 1 and len(profile.assessments) == 1
    assert {m.concept_id for m in profile.mistakes} == {"es.football.preterite_match", "es.football.opinions"}
    assert {c.concept_id for c in task.result.mastery_changes} == set(profile.concepts)
    assert all(c.after != c.before for c in task.result.mastery_changes)

    # Review loop ran once: the first draft lacked a citation, the revision fixed it.
    assert task.result.review_verdict == "APPROVED" and task.result.revisions == 1
    assert mock_llm.calls["teacher"] == 2 and mock_llm.calls["content_reviewer"] == 2

    # Cost and tokens.
    assert task.cost.estimated_cost_usd > 0
    assert task.cost.actual_cost_usd > 0
    assert task.cost.llm_calls == sum(mock_llm.calls.values())
    assert task.cost.token_usage.total_tokens > 0
    assert set(task.cost.by_agent) == set(mock_llm.calls)

    # Events.
    types = [e.type for e in container.task_service.events(task.task_id)]
    for required in REQUIRED_EVENTS:
        assert required in types, required
    assert types[0] == "task.created" and types[-1] == "task.completed"


async def test_second_lesson_uses_memory_instead_of_asking(tmp_path, mock_llm) -> None:
    settings = Settings(data_dir=tmp_path / "d", diagnostic_memory_confidence=0.3)
    container = build_container(settings, llm_providers={"mock": mock_llm})
    try:
        first = await run_lesson(container)
        assert first.status == TaskStatus.COMPLETED
        second = await container.task_service.create_and_run(request="Create an A2 lesson about football.",
                                                             learner_id=first.learner_id, user_id="u1")
        assert second.status == TaskStatus.COMPLETED  # no WAITING: the diagnostic concluded from memory
        states = second.workflow.node_states
        assert states["answers_1"].status.value == "SKIPPED"
        assert states["diagnose_2"].status.value == "SKIPPED"
        profile = container.learner_service.get(first.learner_id)
        assert len(profile.lessons) == 2 and len(profile.assessments) == 1
    finally:
        container.close()


async def test_malformed_model_output_goes_through_validation_retry(container, mock_llm) -> None:
    mock_llm.inject("curriculum_planner", "this is not json", '{"title": "missing everything"}')
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED
    assert mock_llm.calls["curriculum_planner"] == 3
    failures = [e for e in container.task_service.events(task.task_id) if e.type == "agent.validation_failed"]
    assert [e.agent_id for e in failures] == ["curriculum_planner", "curriculum_planner"]
    # The correction request carried the validation error back to the model.
    retry_request = [r for r in mock_llm.requests if r.agent_id == "curriculum_planner"][-1]
    assert "failed validation" in retry_request.messages[-1].content


async def test_unknown_topic_fails_cleanly(container) -> None:
    task = await run_lesson(container, request="Create an A2 lesson about astrophysics.")
    assert task.status == TaskStatus.FAILED
    assert task.errors and task.errors[-1].node_id == "diagnose_1"
    assert "no concepts known" in task.errors[-1].message
