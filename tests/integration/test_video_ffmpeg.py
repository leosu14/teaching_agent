"""Real MP4s with FFmpeg, from the small deterministic fixture (320x180, 10 fps, 7 s), checked with ffprobe and by
decoding audio and sampled frames: container, duration, resolution, frame rate, streams; narration where the
timeline puts it and silence elsewhere; frames that are not empty, change between slides, carry subtitles and fade
when asked. Requires ffmpeg and ffprobe on PATH (CI installs them)."""

from __future__ import annotations

import math
from array import array

import pytest

from app.agents.video.agent import VideoAgent
from app.artifacts.service import ArtifactService
from app.providers.video.base import VideoCompositionError
from app.providers.video.ffmpeg import (
    FFmpegAdapter,
    FFmpegCommand,
    FFmpegError,
    FFmpegVideoComposer,
    FFprobeVideoProber,
)
from app.schemas.video import FrameSample, TransitionType, VideoPlan
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.tools.video.service import VideoService
from app.utils.workspace import ScratchSpace
from tests.video_fixtures import CLIPS, build_scenario

PROBER = FFprobeVideoProber()


class Rig:
    def __init__(self, root) -> None:
        self.sessions = create_db(f"sqlite:///{root / 'db.sqlite'}")
        self.artifacts = ArtifactService(SqlArtifactRepository(self.sessions), FilesystemObjectStore(root / "objects"))
        self.scenario = build_scenario(self.artifacts)
        self.scratch = ScratchSpace(root / "work")
        self.root = root

    def service(self, composer=None) -> VideoService:
        return VideoService(self.artifacts, composer or FFmpegVideoComposer(media=self.artifacts.read_object),
                            PROBER, self.scratch)

    def plan(self, **config) -> VideoPlan:
        request = self.scenario.request
        if config:
            request = request.model_copy(update={"config": request.config.model_copy(update=config)})
        return VideoAgent().plan(request)

    def compose(self, plan: VideoPlan, name: str):
        service = self.service()
        composed = service.compose(plan, service.composition_key(plan))
        local = self.root / f"{name}.mp4"
        self.artifacts.copy_object_to(composed.object.uri, local)
        return service, composed, local


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    r = Rig(tmp_path_factory.mktemp("video"))
    yield r
    dispose(r.sessions)


@pytest.fixture(scope="module")
def default_video(rig):
    plan = rig.plan()
    service, composed, local = rig.compose(plan, "default")
    return plan, service, composed, local


@pytest.fixture(scope="module")
def no_subtitles_video(rig):
    plan = rig.plan(subtitles=rig.scenario.request.config.subtitles.model_copy(update={"enabled": False}))
    return rig.compose(plan, "plain")[2]


def pcm(path, rate: int = 8000) -> array:
    adapter = FFmpegAdapter()
    raw = adapter.run(FFmpegCommand(workspace=path.parent, inputs=(((), path),), output=path.parent / "x.pcm",
                                    maps=("0:a:0",), output_options=("-ac", "1", "-ar", str(rate), "-f", "s16le"),
                                    pipe_output=True), log_name="test-pcm")
    samples = array("h")
    samples.frombytes(raw)
    return samples


def window(samples: array, start: float, end: float, rate: int = 8000) -> array:
    return samples[round(start * rate):round(end * rate)]


def peak_db(samples: array) -> float:
    peak = max((abs(s) for s in samples), default=0)
    return 20 * math.log10(peak / 32768) if peak else -math.inf


def frequency(samples: array, rate: int = 8000) -> float:
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))
    return crossings / 2 / (len(samples) / rate)


def test_real_mp4_container_duration_resolution_fps_and_streams(default_video) -> None:
    plan, service, composed, local = default_video
    assert local.read_bytes()[4:8] == b"ftyp"
    data = FFmpegAdapter().probe_json(local)
    fmt = data["format"]
    assert "mp4" in fmt["format_name"].split(",") and fmt["tags"]["major_brand"] == "isom"
    assert abs(float(fmt["duration"]) - plan.duration) <= 0.1
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    assert (video["codec_name"], video["width"], video["height"], video["pix_fmt"]) == ("h264", 320, 180, "yuv420p")
    assert video["avg_frame_rate"] == "10/1" and int(video["nb_frames"]) == 70 == composed.frames
    assert len(audio) == 1 and audio[0]["codec_name"] == "aac" and int(audio[0]["sample_rate"]) == 48000
    assert composed.composer == "ffmpeg-composer/1" and composed.render_seconds > 0

    report = service.validate(plan, composed)
    assert report.valid, report.errors
    assert report.probe.video.width == 320 and report.probe.audio[0].codec == "aac"
    assert set(report.checks) >= {"container", "duration", "video_stream", "audio_stream", "narration", "frames"}


def test_narration_is_placed_on_the_timeline_with_silence_preserved(default_video) -> None:
    plan, _, _, local = default_video
    samples = pcm(local)
    assert len(samples) / 8000 == pytest.approx(plan.duration, abs=0.1)  # the audio covers the whole timeline
    # Each clip plays at its own timeline slot (told apart by its tone), not concatenated.
    for seg_id, _, start, end, freq, _ in CLIPS:
        clip = window(samples, start + 0.1, end - 0.1)
        assert peak_db(clip) > -20, seg_id
        assert frequency(clip) == pytest.approx(freq, rel=0.05), seg_id
    # Silence before the first clip, in the pause between the two clips of slide 1, over the whole silent slide 2
    # and after the last clip.
    for start, end in ((0.0, 0.25), (1.4, 1.7), (3.05, 4.95), (6.8, 6.95)):
        assert peak_db(window(samples, start, end)) < -50, (start, end)


def test_sampled_frames_are_not_empty_and_change_with_the_slides(default_video, rig) -> None:
    _, _, _, local = default_video
    stats = PROBER.frame_stats(local, [FrameSample(label=str(t), time=t) for t in (1.5, 4.0, 6.0)])
    s1, s2, s3 = stats["1.5"], stats["4.0"], stats["6.0"]
    assert all(s.stddev > 5 for s in (s1, s2, s3))  # real content, not a flat frame
    assert s1.differs_from(s2) and s2.differs_from(s3) and s1.differs_from(s3)


def test_subtitles_are_burned_in_where_narration_plays(default_video, no_subtitles_video) -> None:
    _, _, _, with_subs = default_video
    times = [FrameSample(label="cue", time=0.8), FrameSample(label="silent", time=4.0)]
    subs, plain = PROBER.frame_stats(with_subs, times), PROBER.frame_stats(no_subtitles_video, times)
    w = 64

    def bottom(thumb: bytes) -> bytes:  # rows 26-33 of 36: the subtitle band
        return thumb[26 * w:34 * w]

    def diff(a: bytes, b: bytes) -> float:
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    assert diff(bottom(subs["cue"].thumbnail), bottom(plain["cue"].thumbnail)) > 10  # a subtitle box is there
    assert diff(subs["silent"].thumbnail, plain["silent"].thumbnail) < 2  # no narration, no subtitle
    assert diff(subs["cue"].thumbnail[:20 * w], plain["cue"].thumbnail[:20 * w]) < 2  # slide content unchanged


def test_fade_goes_through_the_background_and_cut_does_not(rig, default_video) -> None:
    _, _, _, cut = default_video
    plan = rig.plan(transition=TransitionType.FADE, fade_seconds=0.6)
    service, composed, faded = rig.compose(plan, "fade")
    assert service.validate(plan, composed).valid
    at = [FrameSample(label="boundary", time=3.0), FrameSample(label="mid", time=4.0)]
    f, c = PROBER.frame_stats(faded, at), PROBER.frame_stats(cut, at)
    assert f["boundary"].stddev < 1.0 < c["boundary"].stddev  # fully faded to the flat background colour
    assert not f["mid"].differs_from(c["mid"])  # after the fade the slide is the same as with a cut


def test_720p_and_other_frame_rates_are_supported(rig) -> None:
    plan = rig.plan(width=1280, height=720, fps=25)
    service, composed, local = rig.compose(plan, "hd")
    report = service.validate(plan, composed)
    assert report.valid, report.errors
    v = report.probe.video
    assert (v.width, v.height, v.fps, composed.frames) == (1280, 720, 25.0, 175)


def test_commands_are_typed_argument_lists_inside_the_workspace(tmp_path) -> None:
    (tmp_path / "ws").mkdir()
    inside = tmp_path / "ws" / "in.png"
    cmd = FFmpegCommand(workspace=tmp_path / "ws", inputs=((("-framerate", "10"), inside),),
                        output=tmp_path / "ws" / "out.mp4", filter_graph="[0:v]null[v]", maps=("[v]",))
    argv = cmd.argv("ffmpeg")
    assert argv[0] == "ffmpeg" and f"file:{inside.resolve()}" in argv and argv[-1].startswith("file:")
    assert "-nostdin" in argv and all(isinstance(a, str) for a in argv)
    outside = FFmpegCommand(workspace=tmp_path / "ws", inputs=(((), tmp_path / "secret.txt"),),
                            output=tmp_path / "ws" / "o.mp4")
    with pytest.raises(FFmpegError, match="outside the workspace"):
        outside.argv("ffmpeg")
    sneaky = FFmpegCommand(workspace=tmp_path / "ws", inputs=(), output=tmp_path / "ws" / ".." / "o.mp4")
    with pytest.raises(FFmpegError):
        sneaky.argv("ffmpeg")


def test_missing_ffmpeg_and_ffmpeg_errors_fail_with_kept_diagnostics(rig) -> None:
    plan = rig.plan()
    missing = rig.service(FFmpegVideoComposer(media=rig.artifacts.read_object,
                                              adapter=FFmpegAdapter(ffmpeg="/nonexistent/ffmpeg")))
    before = len(rig.scratch.failed_sessions())
    with pytest.raises(FFmpegError, match="not installed"):
        missing.compose(plan, "k")
    kept = rig.scratch.failed_sessions()
    assert len(kept) == before + 1 and (kept[-1] / "compose.args.json").exists()

    broken = rig.service(FFmpegVideoComposer(media=rig.artifacts.read_object, preset="no-such-preset"))
    with pytest.raises(FFmpegError, match="exited with"):
        broken.compose(plan, "k")
    newest = max(rig.scratch.failed_sessions(), key=lambda p: p.stat().st_mtime)
    assert (newest / "compose.stderr.log").read_bytes()  # FFmpeg's own error output is preserved


def test_composer_refuses_inputs_that_do_not_match_their_checksums(rig) -> None:
    plan = rig.plan()
    tampered = plan.model_copy(deep=True)
    tampered.slides[0].visual_ref.image.checksum = "0" * 64
    with pytest.raises(VideoCompositionError, match="checksum"):
        rig.service().compose(tampered, "k")
