"""The video stages inside the lesson workflow (mock composer and prober): planning from the presentation, timeline
and asset artifacts, the plan validation gate, composition, MP4 validation and the VIDEO artifact; gating on the
timeline; the failure policy; events and usage; resume and idempotency; artifact lineage. The real FFmpeg path is
covered end to end by tests/e2e/test_demo_cli.py and in detail by tests/integration/test_video_ffmpeg.py."""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.tts.mock import MockTTSProvider
from app.providers.video.mock import MockVideoComposer, MockVideoProber
from app.schemas.artifact import ArtifactType
from app.schemas.audio import PresentationTimeline
from app.schemas.task import TaskStatus
from app.schemas.video import VideoArtifactMetadata, VideoComposeRequest, VideoPlan, VideoResult
from app.services.container import build_container
from app.tools.base import ToolCaller
from tests.conftest import run_lesson
from tests.integration.test_audio_flow import start_until_crash
from tests.integration.test_resume import SimulatedCrash, crash_after
from tests.integration.test_review_loop import always_reject
from tests.integration.test_visual_flow import GenerationFailing

VIDEO_NODES = ["video_plan", "validate_video_plan", "store_video_plan", "compose_video"]
INJECTED = ("image_generation_provider", "tts_provider", "video_composer", "video_prober")


@pytest.fixture
def make_container(tmp_path):
    made = []

    def make(*, llm=None, observers=(), data_dir=None, **kwargs):
        injected = {k: kwargs.pop(k) for k in INJECTED if k in kwargs}
        c = build_container(Settings(data_dir=data_dir or tmp_path / f"d{len(made)}", log_json=False,
                                     video_composer="mock", **kwargs),
                            llm_providers={"mock": llm or MockLLMProvider(default_responders())},
                            observers=observers, **injected)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def arts_of(container, task) -> dict:
    return {a.name: a for a in container.task_service.artifacts(task.task_id)}


def video_events(container, task) -> list:
    return [e for e in container.task_service.events(task.task_id)
            if e.type.startswith(("video_planning.", "video_plan.", "video_composition.", "video."))]


def states(task, *nodes) -> list[str]:
    return [task.workflow.node_states[n].status.value for n in nodes]


class BadProber(MockVideoProber):
    """Reports a resolution the plan did not ask for: the file must not become a VIDEO artifact."""

    def probe(self, path):
        probe = super().probe(path)
        return probe.model_copy(update={"video": probe.video.model_copy(update={"width": 640, "height": 360})})


class CrashingProber(MockVideoProber):
    def __init__(self) -> None:
        self.crashed = False

    def probe(self, path):
        if not self.crashed:
            self.crashed = True
            raise SimulatedCrash("killed during video validation")
        return super().probe(path)


async def test_approved_lesson_becomes_a_validated_video_artifact(make_container) -> None:
    composer = MockVideoComposer()
    container = make_container(video_composer=composer)
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    order = [n for n in task.workflow.execution_order if n in {"audio_timeline", *VIDEO_NODES, "update_learner"}]
    assert order == ["audio_timeline", *VIDEO_NODES, "update_learner"]
    assert composer.calls == 1

    arts = arts_of(container, task)
    timeline_art, pres = arts["presentation_timeline"], arts["presentation"]
    timeline = PresentationTimeline.model_validate_json(container.artifacts.read(timeline_art.artifact_id))
    plan = VideoPlan.model_validate_json(container.artifacts.read(arts["video_plan"].artifact_id))

    # The plan follows the timeline exactly and references the existing assets.
    assert [s.slide_id for s in plan.slides] == [s.slide_id for s in timeline.slides]
    assert [(s.start_time, s.end_time) for s in plan.slides] == [(s.start_time, s.end_time) for s in timeline.slides]
    assert [(t.track_id, t.start_time, t.end_time, t.artifact_id) for t in plan.audio_tracks] == [
        (s.segment_id, s.start_time, s.end_time, s.audio_artifact_id) for s in timeline.segments]
    images = {a.artifact_id for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET}
    assert plan.image_artifact_ids() and set(plan.image_artifact_ids()) <= images
    silent = [s for s in plan.slides if not s.audio_refs]
    assert [s.slide_id for s in silent] == [s.slide_id for s in timeline.slides if not s.audio_segment_refs]
    assert all(s.visual_ref.kind == "slide_card" for s in silent)
    assert (plan.timeline_ref, plan.presentation_ref) == (timeline_art.artifact_id, pres.artifact_id)
    assert plan.timeline_checksum == timeline_art.content_hash
    assert (plan.resolution.width, plan.resolution.height, plan.fps) == (1920, 1080, 30)

    # Artifacts and lineage.
    vp, subs, video = arts["video_plan"], arts["subtitles"], arts["video"]
    assert vp.type == ArtifactType.VIDEO_PLAN and vp.parent_ids[:2] == [timeline_art.artifact_id, pres.artifact_id]
    assert subs.type == ArtifactType.SUBTITLE and subs.media_type == "text/vtt" and subs.parent_ids == [vp.artifact_id]
    assert container.artifacts.read(subs.artifact_id).decode().startswith("WEBVTT")
    assert video.type == ArtifactType.VIDEO and video.media_type == "video/mp4"
    assert video.parent_ids == [vp.artifact_id, timeline_art.artifact_id, pres.artifact_id]
    meta = VideoArtifactMetadata.model_validate(video.metadata)
    assert abs(meta.duration - timeline.duration) <= 0.1 and meta.timeline_duration == timeline.duration
    assert (meta.width, meta.height, meta.fps, meta.codec, meta.audio_codec, meta.container) == (
        1920, 1080, 30.0, "h264", "aac", "mp4")
    assert meta.checksum == video.content_hash and meta.file_size == video.size_bytes and meta.audio_stream
    assert meta.object_key.startswith("objects/sha256/") and meta.object_key.endswith(".mp4")
    assert meta.validation.valid and meta.subtitles == len(plan.subtitle_track.subtitles) and meta.subtitles_burned
    lineage = {a.name for a in container.artifacts.lineage(video.artifact_id)}
    assert {"video_plan", "presentation_timeline", "presentation", "audio_plan", "slide_plan", "lesson",
            "research_bundle"} <= lineage and any(n.startswith("audio_s") for n in lineage)
    result_ids = {a.artifact_id for a in task.result.artifacts}
    assert {vp.artifact_id, subs.artifact_id, video.artifact_id} <= result_ids and video.artifact_id in task.artifact_ids
    assert task.result.warnings == []
    # The presentation was used, not regenerated or modified.
    assert [a.version for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.PRESENTATION] \
        == [1]


async def test_video_events_and_usage(make_container) -> None:
    container = make_container()
    task = await run_lesson(container)
    types = [e.type for e in video_events(container, task)]
    assert types == ["video_planning.started", "video_plan.validated", "video_plan.created", "video_plan.validated",
                     "video_composition.started", "video_composition.completed", "video.validation_started",
                     "video.validation_completed", "video.artifact_created"]
    completed = next(e for e in video_events(container, task) if e.type == "video_composition.completed")
    assert completed.node_id == "compose_video" and completed.data["reused"] is False
    line = task.cost.by_service["video_compose:mock-composer"]
    assert line.calls == 1 and line.cost_usd is None and line.estimated_cost_usd is None
    assert {"render_seconds", "output_bytes", "frames", "video_seconds"} <= set(line.units)
    assert "video" not in task.cost.by_agent  # the video planner makes no model call


async def test_no_video_without_a_timeline(make_container) -> None:
    llm = MockLLMProvider({**default_responders(), "content_reviewer": always_reject})
    composer = MockVideoComposer()
    container = make_container(llm=llm, video_composer=composer, max_revisions=1,
                               revision_exhausted_policy="accept_with_warnings")
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    assert states(task, *VIDEO_NODES) == ["SKIPPED"] * len(VIDEO_NODES)
    assert composer.calls == 0 and not video_events(container, task)
    assert any("video is only composed from a narrated presentation timeline" in w for w in task.result.warnings)


async def test_composition_failure_fails_the_task_and_keeps_earlier_artifacts(make_container) -> None:
    container = make_container(video_composer=MockVideoComposer(fail=True))
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED
    error = task.errors[-1]
    assert error.node_id == "compose_video" and "VideoRequired" in error.message and "compose" in error.message
    persisted = container.task_service.get(task.task_id)
    assert persisted.errors[-1].message == error.message  # the error is stored with the task
    arts = arts_of(container, task)
    assert "video" not in arts and not [a for a in arts.values() if a.type == ArtifactType.VIDEO]
    assert {"presentation", "presentation_timeline", "audio_plan", "video_plan", "subtitles", "lesson"} <= set(arts)
    failed = [e for e in video_events(container, task) if e.type == "video.failed"]
    assert failed and failed[0].data["stage"] == "compose"


async def test_an_invalid_mp4_never_becomes_a_video_artifact(make_container) -> None:
    container = make_container(video_prober=BadProber())
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED
    assert "resolution_mismatch" in task.errors[-1].message and "validate" in task.errors[-1].message
    assert not [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.VIDEO]
    completed = next(e for e in video_events(container, task) if e.type == "video.validation_completed")
    assert completed.data["valid"] is False


async def test_optional_video_completes_with_a_warning_and_no_fake_artifact(make_container) -> None:
    container = make_container(video_composer=MockVideoComposer(fail=True), video_failure_policy="continue")
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    result = VideoResult.model_validate(task.workflow.node_states["compose_video"].output)
    assert result.status == "failed" and result.failed_stage == "compose" and result.artifact is None
    assert any("No video was produced (video is optional)" in w for w in task.result.warnings)
    assert not [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.VIDEO]
    assert task.workflow.node_states["update_learner"].status.value == "COMPLETED"


async def test_invalid_plans_never_reach_the_composer(make_container, monkeypatch) -> None:
    from app.agents.video.agent import VideoAgent
    original = VideoAgent.plan

    def broken(self, data):
        plan = original(self, data)
        return plan.model_copy(update={"duration": plan.duration + 5})

    monkeypatch.setattr(VideoAgent, "plan", broken)
    composer = MockVideoComposer()
    container = make_container(video_composer=composer, video_failure_policy="continue")
    task = await run_lesson(container)
    assert task.status == TaskStatus.FAILED and task.errors[-1].node_id == "validate_video_plan"
    assert "duration_mismatch" in task.errors[-1].message and composer.calls == 0


async def test_kill_after_video_planning_resumes_from_the_plan(make_container, tmp_path) -> None:
    data_dir = tmp_path / "shared"
    first = make_container(observers=[crash_after("store_video_plan")], data_dir=data_dir)
    task = await start_until_crash(first)
    crashed = first.task_service.get(task.task_id)
    assert states(crashed, "store_video_plan", "compose_video") == ["COMPLETED", "PENDING"]

    llm = MockLLMProvider(default_responders())
    generation, tts, composer = GenerationFailing(prefix=""), MockTTSProvider(), MockVideoComposer()
    second = make_container(llm=llm, data_dir=data_dir, image_generation_provider=generation, tts_provider=tts,
                            video_composer=composer)
    resumed = await second.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    # No research, teaching, images, audio or presentation work again: only the composition.
    assert sum(llm.calls.values()) == 0 and generation.calls == 0 and tts.calls == 0 and composer.calls == 1
    started = [e.node_id for e in second.task_service.events(task.task_id) if e.type == "node.started"]
    for node in ("research", "teach_review", "visual", "render_presentation", "synthesize_audio", "audio_timeline",
                 *VIDEO_NODES):
        assert started.count(node) == 1, node
    arts = second.task_service.artifacts(task.task_id)
    for kind in (ArtifactType.PRESENTATION, ArtifactType.VIDEO_PLAN, ArtifactType.VIDEO,
                 ArtifactType.PRESENTATION_TIMELINE):
        assert len([a for a in arts if a.type == kind]) == 1, kind


async def test_kill_during_validation_resumes_without_composing_again(make_container, tmp_path) -> None:
    data_dir = tmp_path / "shared"
    composer = MockVideoComposer()
    first = make_container(data_dir=data_dir, video_composer=composer, video_prober=CrashingProber())
    task = await start_until_crash(first)
    crashed = first.task_service.get(task.task_id)
    assert composer.calls == 1 and "composed" in crashed.workflow.node_states["compose_video"].progress

    second_composer = MockVideoComposer()
    second = make_container(data_dir=data_dir, video_composer=second_composer)
    resumed = await second.task_service.resume(task.task_id)
    assert resumed.status == TaskStatus.COMPLETED, resumed.errors
    assert second_composer.calls == 0  # the composed MP4 from before the crash was validated and used
    started = [e for e in video_events(second, resumed) if e.type == "video_composition.started"]
    assert len(started) == 1
    assert len([a for a in second.task_service.artifacts(task.task_id) if a.type == ArtifactType.VIDEO]) == 1


async def test_composing_again_with_identical_inputs_reuses_the_video(make_container) -> None:
    composer = MockVideoComposer()
    container = make_container(video_composer=composer)
    task = await run_lesson(container)
    video = arts_of(container, task)["video"]
    plan = VideoPlan.model_validate_json(container.artifacts.read(arts_of(container, task)["video_plan"].artifact_id))
    from app.observability.scope import ExecutionScope, UsageLedger
    scope = ExecutionScope(events=container.events, usage=UsageLedger(), task_id=task.task_id)
    caller = ToolCaller(caller_id="test", allowed_tools=frozenset({"video.compose"}),
                        permissions=frozenset({"artifact:read", "artifact:write"}))
    again = await container.tools.call(caller, "video.compose", VideoComposeRequest(plan=plan), scope)
    assert again.reused and again.artifact.artifact_id == video.artifact_id and composer.calls == 1
    # A different configuration is a different composition.
    other = plan.model_copy(update={"video_plan_id": "vp_other"})
    fresh = await container.tools.call(caller, "video.compose", VideoComposeRequest(plan=other), scope)
    assert not fresh.reused and composer.calls == 2
    # Identical bytes are one object in the store.
    assert again.object.uri == video.uri
