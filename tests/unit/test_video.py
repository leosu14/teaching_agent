"""Video slice units: VideoConfig, the plan schemas, subtitles, the VideoAgent's deterministic plan, the
VideoPlanValidator, the mock composer and prober, VideoValidator, VideoService (artifact, checksum, deduplication)
and the video tools through the ToolManager. No FFmpeg here: see tests/integration/test_video_ffmpeg.py."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from app.agents.video.agent import VideoAgent, subtitle_cues, video_plan_id_for
from app.artifacts.service import ArtifactService
from app.config.settings import Settings
from app.providers.video.mock import MockVideoComposer, MockVideoProber
from app.schemas.artifact import ArtifactType
from app.schemas.video import (
    AudioTrack,
    SubtitleTrack,
    Transition,
    TransitionType,
    VideoArtifactRequest,
    VideoComposeRequest,
    VideoConfig,
    VideoPlan,
    VideoValidationInput,
)
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.tools.base import ToolCaller, ToolError
from app.tools.manager import ToolManager
from app.tools.registry import ToolRegistry
from app.tools.video.service import VideoService
from app.tools.video.tools import VideoArtifactTool, VideoComposeTool, VideoValidateTool
from app.tools.video.validation import VideoPlanInvalid, VideoPlanValidationTool, VideoPlanValidator
from app.utils.workspace import ScratchSpace
from tests.unit.helpers import scope
from tests.video_fixtures import SMALL, build_scenario


@pytest.fixture
def artifacts(tmp_path):
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(tmp_path / "objects"))
    dispose(sessions)


@pytest.fixture
def scenario(artifacts):
    return build_scenario(artifacts)


def make_plan(scenario, **config) -> VideoPlan:
    request = scenario.request
    if config:
        request = request.model_copy(update={"config": request.config.model_copy(update=config)})
    return VideoAgent().plan(request)


def check(scenario, plan: VideoPlan | dict):
    body = plan if isinstance(plan, dict) else plan.model_dump(mode="json")
    req = scenario.request.validation_request(VideoPlan.model_construct())
    return VideoPlanValidator().validate(req.model_copy(update={"plan": body}))


def codes(report) -> set[str]:
    return {e.code for e in report.errors}


# --- Configuration ------------------------------------------------------------------------------


def test_video_config_defaults_and_limits() -> None:
    c = VideoConfig()
    assert (c.width, c.height, c.fps, c.codec.value, c.audio_codec.value, c.container.value) == (
        1920, 1080, 30, "h264", "aac", "mp4")
    assert c.media_type == "video/mp4" and c.transition == TransitionType.CUT and c.duration_tolerance == 0.1
    assert c.pixel_format == "yuv420p" and c.subtitles.enabled
    hd = VideoConfig(width=1280, height=720, fps=25)
    assert (hd.resolution.width, hd.resolution.height, hd.fps) == (1280, 720, 25)
    for bad in ({"width": 1281}, {"fps": 0}, {"codec": "vp9"}, {"container": "mkv"}, {"background": "red"},
                {"duration_tolerance": 0}):
        with pytest.raises(ValidationError):
            VideoConfig(**bad)


def test_settings_override_only_what_is_set() -> None:
    assert Settings(log_json=False).video_config() == VideoConfig()
    config = Settings(video_width=1280, video_height=720, video_fps=24, video_transition="fade",
                      video_duration_tolerance=0.2, video_subtitles=False, video_subtitle_max_chars=30).video_config()
    assert (config.width, config.height, config.fps, config.transition, config.duration_tolerance) == (
        1280, 720, 24, TransitionType.FADE, 0.2)
    assert not config.subtitles.enabled and config.subtitles.max_chars_per_line == 30


def test_transition_and_visual_schemas() -> None:
    assert Transition().type == TransitionType.CUT
    with pytest.raises(ValidationError):
        Transition(type=TransitionType.CUT, duration=0.5)
    with pytest.raises(ValidationError):
        Transition(type=TransitionType.FADE, duration=0)


# --- The plan -----------------------------------------------------------------------------------


def test_agent_plans_one_segment_per_timeline_slide(scenario) -> None:
    plan = make_plan(scenario)
    request = scenario.request
    assert [s.slide_id for s in plan.slides] == ["s1", "s2", "s3"]
    assert [(s.start_time, s.end_time, s.duration) for s in plan.slides] == [(0.0, 3.0, 3.0), (3.0, 5.0, 2.0),
                                                                             (5.0, 7.0, 2.0)]
    assert plan.duration == request.timeline.duration == 7.0
    assert (plan.resolution.width, plan.resolution.height, plan.fps) == (320, 180, 10)
    assert (plan.presentation_ref, plan.timeline_ref, plan.timeline_id) == ("art_pres", "art_tl", "tl_test")

    # Visuals: the IMAGE_ASSET where the slide has one, a slide card (title + text) elsewhere.
    s1, s2, s3 = plan.slides
    assert s1.visual_ref.kind == "image_asset" and s1.visual_ref.image == scenario.image
    assert s1.visual_ref.card.title == "El partido" and s1.visual_ref.card.lines == ["Me gusta", "Me encanta"]
    assert s2.visual_ref.kind == "slide_card" and s2.visual_ref.image is None
    assert s2.visual_ref.card.lines == ["¿Quién ganó?"] and s2.visual_ref.card.deck_title == "Fútbol (A2)"

    # Audio: every clip at its timeline position; the silent slide has none.
    assert s1.audio_refs == ["s1_a1", "s1_a2"] and s2.audio_refs == [] and s3.audio_refs == ["s3_a1"]
    track = plan.track("s1_a2")
    assert (track.start_time, track.end_time, track.artifact_id) == (1.8, 2.6, "art_s1_a2")
    assert plan.audio_artifact_ids() == ["art_s1_a1", "art_s1_a2", "art_s3_a1"]
    assert plan.metadata["silent_segments"] == [s2.segment_id]

    # Default transition is a cut everywhere.
    assert all(s.transition.type == TransitionType.CUT for s in plan.slides)
    assert [(t.from_segment, t.to_segment) for t in plan.transitions] == [(s1.segment_id, s2.segment_id),
                                                                          (s2.segment_id, s3.segment_id)]
    # Nothing composer-specific in the plan.
    body = plan.model_dump_json()
    assert "ffmpeg" not in body.lower() and "-c:v" not in body and "libx264" not in body


def test_plan_identity_is_deterministic_and_input_sensitive(scenario) -> None:
    request = scenario.request
    assert make_plan(scenario) == make_plan(scenario)
    base = video_plan_id_for(request)
    assert video_plan_id_for(request.model_copy(update={"timeline_checksum": "1" * 64})) != base
    assert video_plan_id_for(request.model_copy(update={"config": SMALL.model_copy(update={"fps": 12})})) != base
    changed = [request.audio_assets[0].model_copy(update={"checksum": "f" * 64}), *request.audio_assets[1:]]
    assert video_plan_id_for(request.model_copy(update={"audio_assets": changed})) != base


def test_fade_transitions_are_deterministic_and_bounded(scenario) -> None:
    plan = make_plan(scenario, transition=TransitionType.FADE, fade_seconds=2.5)
    assert plan.slides[0].transition.type == TransitionType.CUT
    assert [s.transition.duration for s in plan.slides[1:]] == [2.0, 2.0]  # never longer than a joined segment
    assert check(scenario, plan).valid


def test_subtitles_come_from_narration_text_with_limited_lines(scenario) -> None:
    plan = make_plan(scenario)
    track = plan.subtitle_track
    assert isinstance(track, SubtitleTrack) and track.source == "narration_text" and track.language == "es"
    by_segment = {}
    for cue in track.subtitles:
        by_segment.setdefault(cue.segment_ref, []).append(cue)
        assert len(cue.lines) <= SMALL.subtitles.max_lines
        assert all(len(line) <= SMALL.subtitles.max_chars_per_line for line in cue.lines)
    for asset in scenario.audio:
        cues = by_segment[asset.segment_id]
        t = plan.track(asset.segment_id)
        assert " ".join(" ".join(c.lines) for c in cues) == asset.text  # all of the known text, nothing else
        assert cues[0].start_time == t.start_time and cues[-1].end_time == t.end_time
        assert all(a.end_time == b.start_time for a, b in zip(cues, cues[1:]))
    long = by_segment["s3_a1"]
    assert len(long) == 1 and len(long[0].lines) == 2  # 71 characters: one cue of two lines
    assert plan.slides[2].subtitle_refs == [c.subtitle_id for c in long]
    vtt = track.to_webvtt()
    assert vtt.startswith("WEBVTT") and "00:00:05.200 --> " in vtt and "fútbol" in vtt
    assert make_plan(scenario, subtitles=SMALL.subtitles.model_copy(update={"enabled": False})).subtitle_track is None


def test_subtitle_cues_split_time_in_proportion() -> None:
    track = AudioTrack(track_id="a", slide_id="s", artifact_id="x", uri="u", checksum="c", media_type="audio/wav",
                       start_time=1.0, end_time=3.0, duration=2.0, asset_duration=2.0)
    cues = subtitle_cues(track, "uno dos tres cuatro cinco seis siete ocho nueve diez once doce", "es",
                         VideoConfig(subtitles={"max_chars_per_line": 12, "max_lines": 1}))
    assert [c.text for c in cues][:2] == ["uno dos tres", "cuatro cinco"]
    assert cues[0].start_time == 1.0 and cues[-1].end_time == 3.0
    assert subtitle_cues(track, "   ", "es", VideoConfig()) == []


# --- Plan validation ----------------------------------------------------------------------------


def test_plan_validator_accepts_the_agent_plan(scenario) -> None:
    report = check(scenario, make_plan(scenario))
    assert report.valid and report.plan is not None and report.errors == []
    assert {"timeline", "audio_overlap", "image_refs", "audio_refs", "subtitles", "transitions", "config"} <= set(
        report.checks)


def mutate(plan: VideoPlan, fn) -> dict:
    body = plan.model_dump(mode="json")
    fn(body)
    return body


@pytest.mark.parametrize("change,code", [
    (lambda p: p["slides"][1].update(segment_id=p["slides"][0]["segment_id"]), "duplicate_segment_id"),
    (lambda p: p["slides"][1].update(slide_id="s9"), "unknown_slide"),
    (lambda p: p["slides"].reverse(), "segment_order"),
    (lambda p: p["slides"][1].update(start_time=3.2, duration=1.8), "segment_order"),
    (lambda p: p["slides"][1].update(end_time=2.0, duration=-1.0), "negative_duration"),
    (lambda p: p["slides"][2].update(end_time=6.0, duration=1.0), "timeline_mismatch"),
    (lambda p: p.update(duration=8.0), "duration_mismatch"),
    (lambda p: p["slides"][0]["visual_ref"]["image"].update(artifact_id="art_other"), "unknown_image"),
    (lambda p: p["slides"][0]["visual_ref"]["image"].update(checksum="0" * 64), "unknown_image"),
    (lambda p: p["slides"][0]["visual_ref"]["image"].update(media_type="image/svg+xml"), "unsupported_image"),
    (lambda p: p["audio_tracks"][0].update(artifact_id="art_nope"), "unknown_audio"),
    (lambda p: p["audio_tracks"][1].update(start_time=1.0, end_time=1.8), "audio_overlap"),
    (lambda p: p["audio_tracks"][2].update(start_time=4.0, end_time=5.5), "audio_outside_segment"),
    (lambda p: p["audio_tracks"][0].update(asset_duration=1.0) or p["audio_tracks"][0].update(end_time=1.6,
                                                                                             duration=1.3),
     "audio_duration_mismatch"),
    (lambda p: p["slides"][0]["audio_refs"].append("ghost"), "unknown_reference"),
    (lambda p: p["slides"][2].update(audio_refs=[]), "unknown_reference"),
    (lambda p: p["subtitle_track"]["subtitles"][0].update(end_time=2.9), "subtitle_timing"),
    (lambda p: p["slides"][0].update(transition={"type": "fade", "duration": 0.5}), "invalid_transition"),
    (lambda p: p["slides"][1].update(transition={"type": "fade", "duration": 2.5}), "invalid_transition"),
    (lambda p: p["transitions"].pop(), "invalid_transition"),
    (lambda p: p.update(fps=25), "unsupported_config"),
    (lambda p: p.update(timeline_ref="art_other_tl"), "reference_mismatch"),
])
def test_plan_validator_rejects(scenario, change, code) -> None:
    report = check(scenario, mutate(make_plan(scenario), change))
    assert not report.valid and code in codes(report), report.errors


def test_plan_validator_allows_overlap_only_when_configured(scenario) -> None:
    def overlap(p):
        p["audio_tracks"][1].update(start_time=1.2, end_time=2.0)
    assert "audio_overlap" in codes(check(scenario, mutate(make_plan(scenario), overlap)))
    allowed = mutate(make_plan(scenario, allow_audio_overlap=True), overlap)
    assert "audio_overlap" not in codes(check(scenario, allowed))


def test_plan_validator_reports_schema_errors_and_empty_plans(scenario) -> None:
    report = check(scenario, {"video_plan_id": "vp_x"})
    assert not report.valid and codes(report) == {"invalid_schema"} and report.video_plan_id == "vp_x"
    empty = check(scenario, mutate(make_plan(scenario), lambda p: p.update(slides=[], transitions=[])))
    assert "no_segments" in codes(empty)


async def test_plan_validation_tool_enforces_and_emits(scenario) -> None:
    sc, seen = scope("t1")
    tool = VideoPlanValidationTool()
    plan = make_plan(scenario)
    ok = await tool.run(scenario.request.validation_request(plan, enforce=True), sc)
    assert ok.valid and seen[-1].type == "video_plan.validated" and seen[-1].data["valid"]
    bad = plan.model_copy(update={"duration": 9.0})
    assert not (await tool.run(scenario.request.validation_request(bad), sc)).valid
    with pytest.raises(VideoPlanInvalid):
        await tool.run(scenario.request.validation_request(bad, enforce=True), sc)
    assert [e.type for e in seen[-2:]] == ["video_plan.validated", "video.failed"]


async def test_agent_emits_planning_events_and_validates_through_the_tool_manager(scenario) -> None:
    from app.agents.base import AgentContext
    registry = ToolRegistry()
    registry.register(VideoPlanValidationTool())
    sc, seen = scope("t1")
    agent = VideoAgent()
    plan = await agent.execute(scenario.request, AgentContext(router=None, tools=ToolManager(registry), scope=sc))
    assert isinstance(plan, VideoPlan) and plan == make_plan(scenario)
    types = [e.type for e in seen]
    assert types.index("video_planning.started") < types.index("video_plan.validated") < types.index(
        "video_plan.created")
    created = next(e for e in seen if e.type == "video_plan.created")
    assert created.data["valid"] and created.data["segments"] == 3 and created.data["audio_tracks"] == 3
    assert not any(e.type == "llm.call" for e in seen)  # deterministic: no model call
    assert agent.spec.tools == ("video_plan.validate",)


# --- Mock composer, prober, VideoValidator and VideoService --------------------------------------


@pytest.fixture
def service(artifacts, tmp_path):
    return VideoService(artifacts, MockVideoComposer(media=artifacts.read_object), MockVideoProber(),
                        ScratchSpace(tmp_path / "work"))


def test_mock_composer_is_deterministic_and_checks_inputs(scenario, artifacts, tmp_path) -> None:
    plan = make_plan(scenario)
    composer = MockVideoComposer(media=artifacts.read_object)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = composer.compose(plan, tmp_path / "a"), composer.compose(plan, tmp_path / "b")
    assert open(first.path, "rb").read() == open(second.path, "rb").read()
    assert first.frames == 70 and first.duration == 7.0 and first.subtitles_burned == len(plan.subtitle_track.subtitles)
    tampered = plan.model_copy(deep=True)
    tampered.audio_tracks[0].checksum = "0" * 64
    with pytest.raises(Exception, match="checksum"):
        composer.compose(tampered, tmp_path / "a")


def test_service_composes_validates_and_stores_a_video_artifact(scenario, service, artifacts, tmp_path) -> None:
    plan = make_plan(scenario)
    key = service.composition_key(plan)
    composed = service.compose(plan, key)
    assert composed.object.media_type == "video/mp4" and composed.object.key.endswith(".mp4")
    assert composed.object.checksum == hashlib.sha256(artifacts.read_object(composed.object.uri)).hexdigest()
    assert list((tmp_path / "work").iterdir()) == []  # scratch removed after success

    report = service.validate(plan, composed)
    assert report.valid, report.errors
    assert {"checksum", "container", "duration", "video_stream", "audio_stream", "narration", "frames"} <= set(
        report.checks)
    assert report.audio_levels["s1_a1:narration"] > -45 and report.audio_levels["v02_s2:silence"] < -50

    sc, _ = scope("task_v")
    artifact = service.store("task_v", VideoArtifactRequest(plan=plan, composed=composed, validation=report), sc)
    meta = artifact.metadata
    assert artifact.type == ArtifactType.VIDEO and artifact.media_type == "video/mp4"
    assert artifact.content_hash == meta["checksum"] == composed.object.checksum
    assert (meta["width"], meta["height"], meta["fps"], meta["codec"], meta["audio_codec"], meta["container"]) == (
        320, 180, 10.0, "h264", "aac", "mp4")
    assert meta["duration"] == 7.0 and meta["file_size"] == artifact.size_bytes and meta["audio_stream"]
    assert (meta["timeline_ref"], meta["presentation_ref"]) == ("art_tl", "art_pres")
    assert meta["composition_key"] == key and meta["subtitles_burned"]
    assert artifact.uri == composed.object.uri  # the artifact points at the stored object; nothing is copied

    # Deduplication: the same key finds it; a different configuration or tampered bytes do not.
    assert service.find("task_v", "video", key).artifact_id == artifact.artifact_id
    assert service.find("task_v", "video", "vc_other") is None
    with open(artifact.uri.removeprefix("file://"), "ab") as f:
        f.write(b"tamper")
    assert service.find("task_v", "video", key) is None


def test_invalid_validation_never_becomes_an_artifact(scenario, service) -> None:
    plan = make_plan(scenario)
    composed = service.compose(plan, service.composition_key(plan))
    report = service.validate(plan.model_copy(update={"config": plan.config.model_copy(update={"width": 640})}),
                              composed)
    assert not report.valid and "resolution_mismatch" in {e.code for e in report.errors}
    sc, _ = scope("task_v")
    with pytest.raises(ValueError):
        service.store("task_v", VideoArtifactRequest(plan=plan, composed=composed, validation=report), sc)


def test_video_validator_rejects_files_that_are_not_mp4(scenario, service, artifacts, tmp_path) -> None:
    plan = make_plan(scenario)
    composed = service.compose(plan, service.composition_key(plan))
    request = service.validation_request(plan, composed)
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not a video at all" * 10)
    report = service.validator.validate(request, fake, composed.object.checksum)
    assert not report.valid and {e.code for e in report.errors} >= {"checksum_mismatch", "not_mp4"}
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert {e.code for e in service.validator.validate(request, empty, "").errors} == {"empty_file"}
    # ftyp-shaped but not readable by the parser
    box = tmp_path / "box.mp4"
    box.write_bytes(b"\x00\x00\x00\x10ftypisom\x00\x00\x02\x00")
    assert "unreadable" in {e.code for e in service.validator.validate(request, box, "x").errors}


def test_video_validator_checks_duration_and_audio(scenario, service) -> None:
    plan = make_plan(scenario)
    composed = service.compose(plan, service.composition_key(plan))
    request = service.validation_request(plan, composed)
    local = service.scratch.root / "copy.mp4"
    checksum = service.artifacts.copy_object_to(composed.object.uri, local)
    off = request.model_copy(update={"expected_duration": 7.3})
    assert "duration_mismatch" in {e.code for e in service.validator.validate(off, local, checksum).errors}
    within = request.model_copy(update={"expected_duration": 7.05})
    assert service.validator.validate(within, local, checksum).valid
    strict = request.model_copy(update={"expected_duration": 7.05,
                                        "config": request.config.model_copy(update={"duration_tolerance": 0.01})})
    assert not service.validator.validate(strict, local, checksum).valid
    # Narration expected where the file has none, and sound where silence is expected.
    windows = [w.model_copy(update={"expect_sound": not w.expect_sound}) for w in request.audio_windows]
    errors = {e.code for e in service.validator.validate(request.model_copy(update={"audio_windows": windows}),
                                                         local, checksum).errors}
    assert errors == {"missing_narration", "unexpected_sound"}


# --- Tools through the ToolManager --------------------------------------------------------------


def manager(service) -> tuple[ToolManager, ToolCaller]:
    registry = ToolRegistry()
    for tool in (VideoComposeTool(service), VideoValidateTool(service), VideoArtifactTool(service)):
        registry.register(tool)
    return ToolManager(registry), ToolCaller(caller_id="t", allowed_tools=frozenset(registry.names()),
                                             permissions=frozenset({"artifact:read", "artifact:write"}))


async def test_tools_compose_validate_store_and_then_reuse(scenario, service) -> None:
    tools, caller = manager(service)
    sc, seen = scope("task_v")
    plan = make_plan(scenario)
    composed = await tools.call(caller, "video.compose", VideoComposeRequest(plan=plan), sc)
    report = await tools.call(caller, "video.validate", VideoValidationInput(plan=plan, composed=composed), sc)
    artifact = await tools.call(caller, "video.create_artifact",
                                VideoArtifactRequest(plan=plan, composed=composed, validation=report), sc)
    types = [e.type for e in seen if e.type.startswith("video")]
    assert types == ["video_composition.started", "video_composition.completed", "video.validation_started",
                     "video.validation_completed", "video.artifact_created"]
    usage = sc.usage.summary.by_service["video_compose:mock-composer"]
    assert usage.calls == 1 and usage.cost_usd is None and usage.units["output_bytes"] == composed.object.size_bytes
    assert sc.usage.summary.actual_cost_usd == 0

    # Idempotency: the same inputs again reuse the artifact without composing.
    calls = service.composer.calls
    again = await tools.call(caller, "video.compose", VideoComposeRequest(plan=plan), sc)
    assert again.reused and again.artifact.artifact_id == artifact.artifact_id and service.composer.calls == calls
    again_artifact = await tools.call(caller, "video.create_artifact",
                                      VideoArtifactRequest(plan=plan, composed=composed, validation=report), sc)
    assert again_artifact.artifact_id == artifact.artifact_id
    assert [a.artifact_id for a in service.artifacts.list_for_task("task_v") if a.type == ArtifactType.VIDEO] == [
        artifact.artifact_id]


async def test_compose_tool_maps_failures_and_keeps_diagnostics(scenario, artifacts, tmp_path) -> None:
    service = VideoService(artifacts, MockVideoComposer(fail=True), MockVideoProber(), ScratchSpace(tmp_path / "w"))
    tools, caller = manager(service)
    sc, seen = scope("task_v")
    with pytest.raises(ToolError, match="mock composer configured to fail"):
        await tools.call(caller, "video.compose", VideoComposeRequest(plan=make_plan(scenario)), sc)
    failed = next(e for e in seen if e.type == "video.failed")
    assert failed.data["stage"] == "compose"
    assert len(service.scratch.failed_sessions()) == 1  # the failed job's workspace is kept for diagnosis


def test_scratch_space_cleans_up_or_keeps_failures(tmp_path) -> None:
    space = ScratchSpace(tmp_path / "s", keep_failed=False)
    with space.session("ok/../../evil") as d:
        assert d.parent == space.root and ".." not in d.name
        (d / "f").write_text("x")
    assert not d.exists()
    with pytest.raises(RuntimeError):
        with space.session("bad") as d:
            raise RuntimeError("boom")
    assert not d.exists() and space.failed_sessions() == []


def test_vtt_and_plan_round_trip(scenario) -> None:
    plan = make_plan(scenario)
    assert VideoPlan.model_validate(json.loads(plan.model_dump_json())) == plan
