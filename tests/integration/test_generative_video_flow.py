"""Generated video segments inside the lesson workflow (mock LLM, TTS, video generation provider and composer): the
science lesson of fixtures/generative_video asks for them, the strategy selects two sections, the generation node
submits, polls and stores GENERATED_VIDEO_ASSET artifacts, and the VideoPlan places them. Covers lineage, the
per-segment fallback, required segments, waiting and resuming without resubmitting, cancellation while waiting,
the generation ledger across tasks, and lessons that do not ask for clips (no new nodes at all)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config.settings import Settings
from app.providers.video_generation.mock import MockVideoGenerationProvider
from app.schemas.artifact import ArtifactType
from app.schemas.generative_video import GeneratedVideoResult, VideoSegmentPlanSet
from app.schemas.learner import LearnerProfileInput
from app.schemas.lesson import DiagnosticAnswers, DiagnosticQuestionSheet, LearnerAnswer, LessonRequest
from app.schemas.task import TaskStatus
from app.schemas.video import VideoArtifactMetadata, VideoPlan
from app.services.container import build_container

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "generative_video"
GENERATED = "video.generated_segments"
GENERATION_NODES = ["video_segment_plan", "store_video_segment_plan", "generate_video_segments"]


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def request(*, generated: bool = True) -> LessonRequest:
    learner = fixture("learner.json")
    subject = learner["profile"]["subjects"][0]
    return LessonRequest(raw_request=learner["request"], subject=subject["subject"], topic="water cycle",
                         framework_id=subject["framework_id"], target_level=subject["target_level"],
                         language_of_instruction="en",
                         capabilities=["lesson.text", "video.mp4", *([GENERATED] if generated else [])])


@pytest.fixture
def make(tmp_path):
    made = []

    def build(provider: MockVideoGenerationProvider, *, data_dir: Path | None = None, **settings):
        settings = {"generated_video_poll_interval_seconds": 0.0, **settings}
        c = build_container(Settings(data_dir=data_dir or tmp_path / f"d{len(made)}", corpus_dir=FIXTURES,
                                     log_json=False, video_composer="mock", **settings),
                            video_generation_provider=provider)
        made.append(c)
        return c

    yield build
    for c in made:
        c.close()


async def start(container, *, learner_id: str = "science-learner", generated: bool = True):
    """Runs the lesson through the diagnostic; stops at the first wait that is not a diagnostic."""
    learner, answers = fixture("learner.json"), fixture("answers.json")
    container.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(learner["profile"]))
    task = container.task_service.create_lesson(lesson_request=request(generated=generated), learner_id=learner_id,
                                                user_id="test")
    task = await container.task_service.run(task.task_id)
    while task.status == TaskStatus.WAITING and task.waiting.kind != "video_generation":
        sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
        key = answers["rounds"][sheet.round_number - 1]
        task = await container.task_service.submit_assessment(task.task_id, DiagnosticAnswers(answers=[
            LearnerAnswer(question_id=q.question_id, answer=key.get(q.concept_id, "")) for q in sheet.questions]))
    return task


async def finish(container, task, limit: int = 50):
    for _ in range(limit):
        if task.status != TaskStatus.WAITING:
            return task
        task = await container.task_service.resume(task.task_id)
    raise AssertionError("the task never left WAITING")


def generation(task) -> GeneratedVideoResult:
    return GeneratedVideoResult.model_validate(task.workflow.node_states["generate_video_segments"].output)


def arts(container, task) -> dict:
    return {a.name: a for a in container.task_service.artifacts(task.task_id)}


def events(container, task, prefix: str) -> list:
    return [e for e in container.task_service.events(task.task_id) if e.type.startswith(prefix)]


async def test_clips_are_planned_generated_stored_and_composed_with_lineage(make) -> None:
    provider = MockVideoGenerationProvider(processing_polls=2)
    container = make(provider)
    task = await finish(container, await start(container))
    assert task.status == TaskStatus.COMPLETED, task.errors
    a = arts(container, task)
    plan = VideoSegmentPlanSet.model_validate_json(container.artifacts.read(a["video_segment_plan"].artifact_id))
    assert a["video_segment_plan"].type == ArtifactType.VIDEO_SEGMENT_PLAN and len(plan.segments) == 2
    assert not all(d.selected for d in plan.decisions)  # not every section gets a clip
    result = generation(task)
    assert result.status == "complete" and not result.failures and provider.submits == 2
    assets = [x for x in container.task_service.artifacts(task.task_id)
              if x.type == ArtifactType.GENERATED_VIDEO_ASSET]
    assert len(assets) == 2
    for asset in assets:
        assert a["video_segment_plan"].artifact_id in asset.parent_ids
        assert asset.metadata["validation"]["valid"] and asset.metadata["raw_validation"]["valid"]
        assert asset.metadata["has_audio"] is False and asset.content_hash == asset.metadata["checksum"]
    video_plan = VideoPlan.model_validate_json(container.artifacts.read(a["video_plan"].artifact_id))
    assert {c.artifact_id for c in video_plan.generated_clips()} == {x.artifact_id for x in assets}
    video = a["video"]
    assert {x.artifact_id for x in assets} <= set(video.parent_ids)
    meta = VideoArtifactMetadata.model_validate(video.metadata)
    assert meta.generated_segments == 2 and meta.validation.valid
    kinds = {e.type for e in events(container, task, "video_generation.")}
    assert {"video_generation.submitted", "video_generation.polled", "video_generation.completed"} <= kinds
    assert task.workflow.node_states["generate_video_segments"].status.value == "COMPLETED"


async def test_lessons_without_the_capability_run_exactly_as_before(make) -> None:
    provider = MockVideoGenerationProvider()
    container = make(provider)
    task = await finish(container, await start(container, generated=False))
    assert task.status == TaskStatus.COMPLETED
    assert not set(GENERATION_NODES) & set(task.workflow.node_states)
    assert provider.submits == 0 and "video_segment_plan" not in arts(container, task)
    assert "generated_segments" not in arts(container, task)["video"].metadata or \
        arts(container, task)["video"].metadata["generated_segments"] == 0


@pytest.mark.parametrize("broken", [{"fail_job": True}, {"corrupt": True}, {"fail_submit": True}])
async def test_an_optional_clip_that_fails_falls_back_and_the_lesson_completes(make, broken) -> None:
    container = make(MockVideoGenerationProvider(processing_polls=0, **broken))
    task = await finish(container, await start(container))
    assert task.status == TaskStatus.COMPLETED, task.errors
    result = generation(task)
    assert result.status == "failed" and len(result.failures) == 2 and not result.assets
    assert {f.stage for f in result.failures} <= {"poll", "download", "validate", "submit"}
    assert all(f.fallback.kind in {"slide", "image_asset"} for f in result.failures)
    assert len(events(container, task, "generated_video.fallback")) == 2
    meta = VideoArtifactMetadata.model_validate(arts(container, task)["video"].metadata)
    assert meta.generated_segments == 0 and meta.validation.valid
    assert any("generated" in w for w in task.result.warnings)


async def test_a_required_clip_that_fails_fails_the_task_unless_the_policy_continues(make) -> None:
    failing = make(MockVideoGenerationProvider(processing_polls=0, fail_job=True), generated_video_required=True)
    task = await finish(failing, await start(failing))
    assert task.status == TaskStatus.FAILED
    assert any("required generated clip" in e.message for e in task.errors)
    assert "video" not in arts(failing, task)
    lenient = make(MockVideoGenerationProvider(processing_polls=0, fail_job=True), generated_video_required=True,
                   generated_video_failure_policy="continue")
    task = await finish(lenient, await start(lenient))
    assert task.status == TaskStatus.COMPLETED and len(generation(task).failures) == 2


async def test_slow_jobs_wait_and_resume_without_being_submitted_again(make) -> None:
    provider = MockVideoGenerationProvider(processing_polls=6)
    container = make(provider, generated_video_poll_interval_seconds=0.02, generated_video_poll_timeout_seconds=0.05)
    task = await start(container)
    assert task.status == TaskStatus.WAITING and task.waiting.kind == "video_generation"
    assert events(container, task, "video_generation.waiting")
    resumes = 0
    while task.status == TaskStatus.WAITING:
        task = await container.task_service.resume(task.task_id)
        resumes += 1
        assert resumes < 50
    assert task.status == TaskStatus.COMPLETED and resumes >= 1
    assert provider.submits == 2 and provider.downloads == 2  # the jobs were followed, never started again
    assert len(generation(task).assets) == 2


async def test_jobs_that_never_finish_are_given_up_after_their_poll_budget(make) -> None:
    container = make(MockVideoGenerationProvider(processing_polls=10_000), generated_video_poll_max_attempts=3)
    task = await finish(container, await start(container))
    assert task.status == TaskStatus.COMPLETED
    failures = generation(task).failures
    assert len(failures) == 2 and {f.stage for f in failures} == {"poll"}


async def test_cancelling_a_waiting_task_releases_its_jobs_and_starts_nothing(make) -> None:
    provider = MockVideoGenerationProvider(processing_polls=10_000)
    container = make(provider, generated_video_poll_interval_seconds=0.01, generated_video_poll_timeout_seconds=0.03)
    task = await start(container)
    assert task.status == TaskStatus.WAITING and provider.submits == 2
    task = container.task_service.cancel(task.task_id)
    assert task.status == TaskStatus.CANCELLED
    assert provider.cancels == 2 and provider.submits == 2
    assert len(events(container, task, "video_generation.cancelled")) == 2
    assert not [a for a in container.task_service.artifacts(task.task_id)
                if a.type == ArtifactType.GENERATED_VIDEO_ASSET]


async def test_the_generation_ledger_makes_a_second_task_reuse_every_clip(make, tmp_path) -> None:
    first = MockVideoGenerationProvider(processing_polls=1)
    shared = tmp_path / "shared"
    container = make(first, data_dir=shared)
    task = await finish(container, await start(container))
    assert task.status == TaskStatus.COMPLETED and first.submits == 2
    container.close()
    second = MockVideoGenerationProvider()
    again = make(second, data_dir=shared)
    task2 = await finish(again, await start(again, learner_id="science-learner-2"))
    assert task2.status == TaskStatus.COMPLETED
    assert (second.submits, second.status_calls, second.downloads) == (0, 0, 0)
    assets = generation(task2).assets
    assert len(assets) == 2 and all(a.reused for a in assets)
    assert events(again, task2, "video_generation.reused")
    checksums = lambda t, c: sorted(a.content_hash for a in c.task_service.artifacts(t.task_id)  # noqa: E731
                                    if a.type == ArtifactType.GENERATED_VIDEO_ASSET)
    assert checksums(task2, again) == checksums(task, again)
