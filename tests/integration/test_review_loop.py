"""ReviewNode as a workflow mechanism: verdicts, revision count, issue propagation, budget and policy."""

from __future__ import annotations

from app.config.settings import Settings
from app.providers.llm.base import LLMRequest
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.lesson import ReviewCriterion, ReviewResult, TeacherInput, Verdict
from app.schemas.task import TaskStatus
from app.services.container import build_container
from tests.conftest import run_lesson


def always_reject(request: LLMRequest) -> dict:
    return ReviewResult(
        verdict=Verdict.REVISION_REQUIRED,
        scores={c: 0.5 for c in ReviewCriterion},
        issues=[{"issue_id": "i1", "criterion": "level_appropriateness", "severity": "major",
                 "location": "sec_x", "problem": "Too advanced for the learner.", "suggested_fix": "Simplify."}],
        summary="Too hard.",
    ).model_dump(mode="json")


async def lesson_with(tmp_path, responders, **settings):
    llm = MockLLMProvider(responders)
    container = build_container(Settings(data_dir=tmp_path / "d", **settings), llm_providers={"mock": llm})
    try:
        return await run_lesson(container), llm, container.task_service
    finally:
        container.close()


async def test_revision_carries_reviewer_issues_to_the_teacher(tmp_path) -> None:
    task, llm, _ = await lesson_with(tmp_path, default_responders())
    assert task.result.revisions == 1
    teacher_calls = [TeacherInput.model_validate(r.input_payload) for r in llm.requests if r.agent_id == "teacher"]
    assert teacher_calls[0].revision is None
    revision = teacher_calls[1].revision
    assert revision is not None and revision.revision_number == 1
    assert [i.criterion for i in revision.issues] == [ReviewCriterion.SOURCE_QUALITY]
    assert revision.previous.sections[-1].citations == []  # the draft being revised


async def test_clean_first_draft_is_approved_without_revision(tmp_path) -> None:
    task, llm, _ = await lesson_with(tmp_path, default_responders(first_draft_defects=False))
    assert task.status == TaskStatus.COMPLETED
    assert task.result.revisions == 0 and task.result.review_verdict == "APPROVED"
    assert llm.calls["teacher"] == 1 and llm.calls["content_reviewer"] == 1


async def test_exhausted_revisions_fail_the_task(tmp_path) -> None:
    responders = {**default_responders(), "content_reviewer": always_reject}
    task, llm, service = await lesson_with(tmp_path, responders, max_revisions=2)
    assert task.status == TaskStatus.FAILED
    assert task.errors[-1].node_id == "teach_review"
    assert "ReviewRejected" in task.errors[-1].message and "Too advanced" in task.errors[-1].message
    assert llm.calls["teacher"] == 3 and llm.calls["content_reviewer"] == 3  # draft + 2 revisions, no node retry
    assert task.workflow.node_states["slide_plan"].status.value == "PENDING"


async def test_accept_with_warnings_policy_completes(tmp_path) -> None:
    responders = {**default_responders(), "content_reviewer": always_reject}
    task, llm, service = await lesson_with(tmp_path, responders, max_revisions=1,
                                           revision_exhausted_policy="accept_with_warnings")
    assert task.status == TaskStatus.COMPLETED
    assert task.result.review_verdict == "accepted_with_warnings" and task.result.revisions == 1
    assert llm.calls["content_reviewer"] == 2
