"""End-to-end demo of the Video slice.

    python scripts/run_video_demo.py [--out-dir video/] [--width 1280 --height 720] [--transition fade]

Runs the same lesson as run_demo.py offline with the mock LLM, image and TTS providers and the real FFmpeg
composer (Diagnostic -> Research -> Planner -> Teacher -> Reviewer -> Visual -> Slide Planning -> Presentation
Build -> Presentation Render -> Audio Planning -> TTS -> Audio Assets -> Presentation Timeline -> Video Planning ->
Video Composition -> Video Validation -> VIDEO artifact), then walks through the video stages: the approved lesson,
the PRESENTATION, IMAGE_ASSET and AUDIO_ASSET artifacts, the PresentationTimeline, the VideoPlan and its
validation, the composed MP4, its validation and the VIDEO artifact. The MP4 is re-read from the object store and
measured again with ffprobe. It uses the same TaskService as the API. Needs ffmpeg and ffprobe on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.providers.video.ffmpeg import FFmpegAdapter, FFprobeVideoProber  # noqa: E402
from app.schemas.artifact import ArtifactType  # noqa: E402
from app.schemas.audio import PresentationTimeline  # noqa: E402
from app.schemas.lesson import LessonContent  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.schemas.video import (  # noqa: E402
    VideoArtifactMetadata,
    VideoPlan,
    VideoPlanValidationReport,
    VideoResult,
)
from app.services.container import Container, build_container  # noqa: E402
from scripts.run_demo import run_demo  # noqa: E402


def _clip(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _heading(out, title: str) -> None:
    out("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def report_video(container: Container, task: Task, out=print, save_to: Path | None = None) -> bool:
    arts = {a.name: a for a in container.task_service.artifacts(task.task_id)}
    all_arts = container.task_service.artifacts(task.task_id)
    nodes = task.workflow.node_states
    events = container.task_service.events(task.task_id)

    _heading(out, "1. Approved lesson")
    lesson = LessonContent.model_validate_json(container.artifacts.read(arts["lesson"].artifact_id))
    out(f"{lesson.title} ({lesson.level}); review: {task.result.review_verdict}; "
        f"lesson artifact {arts['lesson'].artifact_id}")

    _heading(out, "2. PRESENTATION artifact")
    pres = arts["presentation"]
    out(f"{pres.artifact_id} v{pres.version} {pres.media_type.split('.')[-1]}: {pres.metadata['slides']} slides, "
        f"sha256={pres.content_hash[:16]}... (the PPTX is not opened or changed by the video stage)")

    _heading(out, "3. IMAGE_ASSET artifacts")
    images = [a for a in all_arts if a.type == ArtifactType.IMAGE_ASSET]
    for a in images:
        out(f"  {a.artifact_id} {a.media_type} {a.metadata.get('width')}x{a.metadata.get('height')} "
            f"sha256={a.content_hash[:12]}...")

    _heading(out, "4. AUDIO_ASSET artifacts")
    audio = [a for a in all_arts if a.type == ArtifactType.AUDIO_ASSET]
    for a in audio:
        out(f"  {a.artifact_id} {a.metadata['segment_id']:<8} slide={a.metadata['slide_id']} "
            f"{a.metadata['duration']:.3f}s sha256={a.content_hash[:12]}...")

    _heading(out, "5. PresentationTimeline")
    tl_art = arts["presentation_timeline"]
    timeline = PresentationTimeline.model_validate_json(container.artifacts.read(tl_art.artifact_id))
    out(f"{timeline.timeline_id}: {len(timeline.slides)} slides, {len(timeline.segments)} narration segments, "
        f"{timeline.duration:.3f}s; artifact {tl_art.artifact_id}")

    _heading(out, "6. VideoPlan")
    plan = VideoPlan.model_validate_json(container.artifacts.read(arts["video_plan"].artifact_id))
    out(f"{plan.video_plan_id} ({plan.planner}): {plan.resolution.width}x{plan.resolution.height} @ {plan.fps} fps, "
        f"{plan.config.codec.value}/{plan.config.audio_codec.value} in {plan.config.container.value}, "
        f"transition={plan.config.transition.value}; artifact {arts['video_plan'].artifact_id}")
    for s in plan.slides:
        visual = f"image {s.visual_ref.image.artifact_id}" if s.visual_ref.image else "slide card"
        out(f"  {s.order:>2}. {s.slide_id} {s.start_time:7.3f} -> {s.end_time:7.3f} ({s.duration:6.3f}s) "
            f"{visual:<26} audio: {', '.join(s.audio_refs) or 'silence':<16} subtitles: {len(s.subtitle_refs)} "
            f"{_clip(s.visual_ref.card.title, 28)!r}")

    _heading(out, "7. VideoPlan validation")
    report = VideoPlanValidationReport.model_validate(nodes["validate_video_plan"].output)
    out(f"valid={report.valid} by {report.validator}; errors={len(report.errors)}")
    out(f"checks: {', '.join(report.checks)}")

    _heading(out, "8. Composition")
    result = VideoResult.model_validate(nodes["compose_video"].output)
    composed = result.composed
    for e in events:
        if e.type == "video_composition.completed":
            out(f"  composer={e.data.get('composer')} frames={e.data.get('frames')} "
                f"render={e.data.get('render_seconds')}s cpu={e.data.get('cpu_seconds')}s "
                f"size={e.data.get('size_bytes')} bytes reused={e.data.get('reused')}")
    line = task.cost.by_service.get(f"video_compose:{composed.composer.split('/')[0]}") if composed else None
    if line is not None:
        out(f"  usage: calls={line.calls} cost={line.cost_usd} units={line.units}")

    _heading(out, "9. MP4 validation")
    validation = result.validation
    out(f"valid={validation.valid} by {validation.validator}")
    out(f"checks: {', '.join(validation.checks)}")
    narrated = [k for k in validation.audio_levels if k.endswith(":narration")]
    silent = [k for k in validation.audio_levels if k.endswith(":silence")]
    out(f"narration audible in {len(narrated)} windows (min peak "
        f"{min(validation.audio_levels[k] for k in narrated):.1f} dBFS); silent windows: "
        f"{', '.join(f'{k} {validation.audio_levels[k]:.0f} dBFS' for k in silent) or '-'}")

    _heading(out, "10. VIDEO artifact")
    video = arts["video"]
    meta = VideoArtifactMetadata.model_validate(video.metadata)
    out(f"{video.artifact_id} {video.type.value} v{video.version} {video.media_type}; parents: "
        f"{', '.join(container.artifacts.get(p).name for p in video.parent_ids)}")
    out(f"object key: {meta.object_key}")

    _heading(out, "11. Result")
    with tempfile.TemporaryDirectory(prefix="video-demo-check-") as tmp:
        local = Path(tmp) / "video.mp4"
        checksum = container.artifacts.copy_object_to(video.uri, local)
        probe = FFprobeVideoProber(FFmpegAdapter()).probe(local)
        if save_to is not None:
            shutil.copyfile(local, save_to / "lesson.mp4")
            (save_to / "lesson.vtt").write_bytes(container.artifacts.read(arts["subtitles"].artifact_id))
    v = probe.video
    a = probe.audio[0] if probe.audio else None
    out(f"output:        {video.uri}")
    out(f"reference:     VIDEO artifact {video.artifact_id} (task {task.task_id})")
    out(f"duration:      {probe.duration:.3f}s (timeline {timeline.duration:.3f}s, "
        f"difference {abs(probe.duration - timeline.duration):.3f}s, tolerance {plan.config.duration_tolerance}s)")
    out(f"resolution:    {v.width}x{v.height}")
    out(f"fps:           {v.fps:g}")
    out(f"video stream:  {v.codec} {v.pixel_format}, container {probe.container} (brand {probe.brand})")
    out(f"audio stream:  {f'{a.codec} {a.sample_rate} Hz, {a.channels} channels' if a else 'MISSING'}")
    out(f"subtitles:     {meta.subtitles} cues burned in ({'yes' if meta.subtitles_burned else 'no'}); "
        f"WebVTT artifact {arts['subtitles'].artifact_id}")
    out(f"checksum:      sha256 {checksum} (matches artifact: {checksum == video.content_hash})")
    out(f"file size:     {meta.file_size} bytes")
    if save_to is not None:
        out(f"\nsaved lesson.mp4 and lesson.vtt to {save_to}")
    ok = (validation.valid and checksum == video.content_hash and a is not None
          and abs(probe.duration - timeline.duration) <= plan.config.duration_tolerance
          and hashlib.sha256(container.artifacts.read(pres.artifact_id)).hexdigest() == pres.content_hash)
    return ok


async def run_video_demo(container: Container, *, out=print, save_to: Path | None = None) -> tuple[Task, bool]:
    _heading(out, "Lesson -> Presentation -> Images + Audio -> Timeline -> VideoPlan -> MP4 -> VIDEO artifact")
    task = await run_demo(container, out=lambda line: None)
    out(f"task {task.task_id}: {task.status.value}")
    if task.status != TaskStatus.COMPLETED:
        for err in task.errors:
            out(f"error: [{err.kind}] {err.node_id}: {err.message}")
        return task, False
    return task, report_video(container, task, out, save_to)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--out-dir", type=Path, default=None, help="also save the MP4 and the WebVTT file here")
    parser.add_argument("--width", type=int, default=None, help="video width (default 1920)")
    parser.add_argument("--height", type=int, default=None, help="video height (default 1080)")
    parser.add_argument("--fps", type=int, default=None, help="frame rate (default 30)")
    parser.add_argument("--transition", choices=["cut", "fade"], default=None, help="slide transition (default cut)")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="teaching-agent-video-demo-") as tmp:
        data_dir = args.data_dir or Path(tmp)
        settings = Settings(data_dir=data_dir, video_composer="ffmpeg", video_width=args.width,
                            video_height=args.height, video_fps=args.fps, video_transition=args.transition)
        container = build_container(settings)
        try:
            task, ok = asyncio.run(run_video_demo(container, save_to=args.out_dir))
        finally:
            container.close()
        if args.data_dir is None and args.out_dir is None:
            print("\n(artifacts were written to a temporary directory; pass --out-dir video/ to keep the MP4)")
    return 0 if task.status == TaskStatus.COMPLETED and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
