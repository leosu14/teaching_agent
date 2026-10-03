"""End-to-end demo of generated video segments.

    python scripts/run_generative_video_demo.py [--out-dir video/]

A science lesson (the water cycle, fixtures/generative_video) that asks for generated video segments runs offline
with the mock LLM, image, TTS and video generation providers and the real FFmpeg normaliser and composer:

   1. the approved lesson;
   2. the video segment plan (VIDEO_SEGMENT_PLAN);
   3. why each section was or was not given a clip (deterministic rules);
   4. the generation jobs at the mock video provider (submitted, polled, completed);
   5. the GENERATED_VIDEO_ASSET artifacts;
   6. their validation, from the bytes (the provider's file, then the normalised clip);
   7. where each clip sits in the composition (VideoPlan);
   8. the final MP4, re-read from the object store and measured with ffprobe;
   9. its checksum;
  10. the timeline: narration stays authoritative, clips play inside their slides.

Then the same lesson runs again as a new task (for a second learner with the same profile and answers) on the same
store: the generation ledger is used, so nothing is submitted to the provider a second time. Needs ffmpeg and ffprobe on PATH. Exits 1 if anything does not hold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.providers.video.ffmpeg import FFmpegAdapter, FFprobeVideoProber  # noqa: E402
from app.providers.video_generation.mock import MockVideoGenerationProvider  # noqa: E402
from app.schemas.artifact import ArtifactType  # noqa: E402
from app.schemas.audio import PresentationTimeline  # noqa: E402
from app.schemas.generative_video import GeneratedVideoResult, VideoSegmentPlanSet  # noqa: E402
from app.schemas.learner import LearnerProfileInput  # noqa: E402
from app.schemas.lesson import (  # noqa: E402
    DiagnosticAnswers,
    DiagnosticQuestionSheet,
    LearnerAnswer,
    LessonContent,
    LessonRequest,
)
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.schemas.video import VideoArtifactMetadata, VideoPlan  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402

FIXTURES = REPO_ROOT / "fixtures" / "generative_video"
GENERATED_VIDEO = "video.generated_segments"  # the capability a lesson asks for to get generated segments
RULE = "=" * 72


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _heading(out, title: str) -> None:
    out(f"\n{RULE}\n{title}\n{RULE}")


def _clip(text: str, n: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def lesson_request() -> LessonRequest:
    learner = _fixture("learner.json")
    subject = learner["profile"]["subjects"][0]
    return LessonRequest(raw_request=learner["request"], subject=subject["subject"], topic="water cycle",
                         framework_id=subject["framework_id"], target_level=subject["target_level"],
                         language_of_instruction=learner["profile"]["preferences"]["language_of_instruction"],
                         capabilities=["lesson.text", "video.mp4", GENERATED_VIDEO])


async def run_lesson(container: Container, learner_id: str | None = None) -> Task:
    """The structured lesson request (no interpretation step), answering the diagnostic from the fixture."""
    learner, answers = _fixture("learner.json"), _fixture("answers.json")
    learner_id = learner_id or learner["learner_id"]
    container.learner_service.upsert(learner_id, LearnerProfileInput.model_validate(learner["profile"]))
    task = container.task_service.create_lesson(lesson_request=lesson_request(), learner_id=learner_id,
                                                user_id="demo")
    task = await container.task_service.run(task.task_id)
    while task.status == TaskStatus.WAITING and task.waiting is not None and task.waiting.kind != "video_generation":
        sheet = DiagnosticQuestionSheet.model_validate(task.waiting.prompt)
        key = answers["rounds"][sheet.round_number - 1]
        task = await container.task_service.submit_assessment(task.task_id, DiagnosticAnswers(answers=[
            LearnerAnswer(question_id=q.question_id, answer=key.get(q.concept_id, "")) for q in sheet.questions]))
    while task.status == TaskStatus.WAITING:  # clips still being generated: resume to check them again
        task = await container.task_service.resume(task.task_id)
    return task


def report(container: Container, task: Task, provider: MockVideoGenerationProvider, out=print,
           save_to: Path | None = None) -> bool:
    arts = {a.name: a for a in container.task_service.artifacts(task.task_id)}
    nodes = task.workflow.node_states
    checks: list[tuple[str, bool]] = []

    _heading(out, "1. Lesson")
    lesson = LessonContent.model_validate_json(container.artifacts.read(arts["lesson"].artifact_id))
    out(f"{lesson.title} (level {lesson.level}), language {lesson_request().language_of_instruction}; "
        f"generated segments requested: {GENERATED_VIDEO in lesson_request().capabilities}")
    for s in lesson.sections:
        out(f"  [{s.section_id}] {s.purpose:<16} {s.heading}: {_clip(s.explanation, 70)}")

    _heading(out, "2. Video segment plan")
    plan_art = arts["video_segment_plan"]
    plan = VideoSegmentPlanSet.model_validate_json(container.artifacts.read(plan_art.artifact_id))
    out(f"{plan_art.artifact_id} {plan_art.type.value}; plan {plan.plan_id}, strategy {plan.strategy}, "
        f"provider {plan.provider}")
    b = plan.budget
    out(f"budget: {b.segments}/{b.max_segments} segments, {b.seconds:g}/{b.max_seconds:g}s, estimated cost "
        f"{'unknown' if b.estimated_cost_usd is None else f'${b.estimated_cost_usd:.2f}'}")
    for seg in plan.segments:
        out(f"  {seg.segment_id}: slide {seg.slide_id}, {seg.duration:g}s, {seg.purpose.value}, "
            f"{seg.insertion_strategy.value}, audio {seg.audio}, required {seg.required}, fallback "
            f"{seg.fallback.kind}{f' {seg.fallback.artifact_id}' if seg.fallback.artifact_id else ''}")
        out(f"    prompt: {_clip(seg.prompt, 160)}")
    checks.append(("segments planned", bool(plan.segments)))

    _heading(out, "3. Why each section was (not) selected")
    for d in plan.decisions:
        verdict = f"SELECTED ({d.purpose.value}, score {d.score:g})" if d.selected else f"skipped: {d.skip_reason}"
        out(f"  {d.lesson_section_id}: {verdict}")
        if d.reasons:
            out(f"    because: {', '.join(d.reasons)}")
    for w in plan.warnings:
        out(f"  warning: {w}")

    _heading(out, "4. Generation jobs (mock video provider)")
    generated = GeneratedVideoResult.model_validate(nodes["generate_video_segments"].output)
    for job in generated.jobs:
        out(f"  {job.job_id} -> {job.provider}:{job.provider_job_id} ({job.model}) {job.status.value} after "
            f"{job.polls} polls, {job.request.duration:g}s {job.request.aspect_ratio}, seed {job.request.seed}")
    out(f"provider calls: {provider.submits} submitted, {provider.status_calls} polled, {provider.downloads} "
        f"downloaded; stage status: {generated.status}")
    checks.append(("every job completed", all(j.status.value == "completed" for j in generated.jobs)
                   and len(generated.jobs) == len(plan.segments)))

    _heading(out, "5. GENERATED_VIDEO_ASSET artifacts")
    assets = [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.GENERATED_VIDEO_ASSET]
    for a in assets:
        m = a.metadata
        out(f"  {a.artifact_id} {a.name} v{a.version}: {m['width']}x{m['height']} {m['fps']:g} fps "
            f"{m['duration']:.3f}s {m['format']}/{m['codec']}, audio {m['has_audio']}; job {m['generation_job_id']}; "
            f"parents: {', '.join(container.artifacts.get(p).name for p in a.parent_ids)}")
    checks.append(("an asset per segment", len(assets) == len(plan.segments)))

    _heading(out, "6. Validation from the bytes")
    for a in assets:
        raw, norm = a.metadata["raw_validation"], a.metadata["validation"]
        claims = a.metadata.get("provider_claims", {})
        out(f"  {a.metadata['segment_id']}:")
        out(f"    provider file  {raw['container']}/{raw['codec']} {raw['width']}x{raw['height']} {raw['fps']:g} fps "
            f"{raw['duration']:.3f}s -> {'valid' if raw['valid'] else 'INVALID'} ({', '.join(raw['checks'])})")
        out(f"    normalised     {norm['container']}/{norm['codec']} {norm['width']}x{norm['height']} "
            f"{norm['fps']:g} fps {norm['duration']:.3f}s -> {'valid' if norm['valid'] else 'INVALID'} "
            f"({a.metadata['normalizer']})")
        out(f"    provider claimed (recorded, not trusted): {claims or '-'}")
        checks.append((f"{a.metadata['segment_id']} valid", raw["valid"] and norm["valid"]))

    _heading(out, "7. Insertion into the composition")
    video_plan = VideoPlan.model_validate_json(container.artifacts.read(arts["video_plan"].artifact_id))
    timeline = PresentationTimeline.model_validate_json(
        container.artifacts.read(arts["presentation_timeline"].artifact_id))
    for seg in video_plan.slides:
        if seg.generated is not None:
            c = seg.generated
            out(f"  slide {seg.slide_id} ({seg.start_time:.3f}-{seg.end_time:.3f}s): {c.strategy.value} clip "
                f"{c.artifact_id} plays {c.start_time:.3f}-{c.end_time:.3f}s ({c.duration:.3f}s of "
                f"{c.asset_duration:.3f}s), clip audio {c.audio}; narration {', '.join(seg.audio_refs) or '-'} "
                f"and {len(seg.subtitle_refs)} subtitle cues unchanged")
    checks.append(("every asset placed", len(video_plan.generated_clips()) == len(assets)))

    _heading(out, "8. Final MP4")
    video = arts["video"]
    meta = VideoArtifactMetadata.model_validate(video.metadata)
    with tempfile.TemporaryDirectory(prefix="generative-video-demo-") as tmp:
        local = Path(tmp) / "lesson.mp4"
        checksum = container.artifacts.copy_object_to(video.uri, local)
        probe = FFprobeVideoProber(FFmpegAdapter()).probe(local)
        if save_to is not None:
            shutil.copyfile(local, save_to / "lesson.mp4")
    v, a = probe.video, probe.audio[0] if probe.audio else None
    out(f"output:      {video.uri}")
    out(f"artifact:    {video.artifact_id} {video.type.value}; parents: "
        f"{', '.join(container.artifacts.get(p).name for p in video.parent_ids)}")
    out(f"duration:    {probe.duration:.3f}s (timeline {timeline.duration:.3f}s)")
    out(f"streams:     {v.codec} {v.width}x{v.height} {v.fps:g} fps; "
        f"{f'{a.codec} {a.sample_rate} Hz' if a else 'NO AUDIO'}; container {probe.container}")
    out(f"generated:   {meta.generated_segments} clips composed ({', '.join(meta.generated_artifact_ids)})")
    out(f"validation:  {'valid' if meta.validation.valid else 'INVALID'} ({', '.join(meta.validation.checks)})")
    checks.append(("MP4 valid", meta.validation.valid and a is not None
                   and abs(probe.duration - timeline.duration) <= video_plan.config.duration_tolerance))
    checks.append(("clips composed", meta.generated_segments == len(assets)))

    _heading(out, "9. Checksum")
    out(f"sha256 {checksum} (matches the VIDEO artifact: {checksum == video.content_hash})")
    checks.append(("checksum", checksum == video.content_hash))

    _heading(out, "10. Timeline")
    clips = {seg.slide_id: seg.generated for seg in video_plan.slides if seg.generated}
    for s in timeline.slides:
        clip = clips.get(s.slide_id)
        out(f"  {s.order:>2}. {s.slide_id} {s.start_time:8.3f}-{s.end_time:8.3f}s narration "
            f"{len(s.audio_segment_refs)} segment(s){f'  + generated clip {clip.start_time:.3f}-{clip.end_time:.3f}s' if clip else ''}")
    if save_to is not None:
        out(f"\nsaved lesson.mp4 to {save_to}")

    _heading(out, "Checks")
    for name, ok in checks:
        out(f"  {'ok ' if ok else 'FAIL'} {name}")
    return all(ok for _, ok in checks)


async def check_reuse(data_dir: Path, out=print) -> bool:
    """The same lesson for a second learner with the same profile and answers, as a new task on the same store:
    the same clips are asked for, and they come from the generation ledger."""
    _heading(out, "Re-run: the same lesson as a new task (idempotency)")
    provider = MockVideoGenerationProvider()
    container = build_container(Settings(data_dir=data_dir, corpus_dir=FIXTURES, video_composer="mock"),
                                video_generation_provider=provider)
    try:
        task = await run_lesson(container, learner_id="science-learner-2")
        generated = GeneratedVideoResult.model_validate(task.workflow.node_states["generate_video_segments"].output)
    finally:
        container.close()
    reused = [a for a in generated.assets if a.reused]
    out(f"task {task.task_id}: {task.status.value}; {len(generated.assets)} assets, {len(reused)} reused from the "
        f"generation ledger; provider calls: {provider.submits} submitted, {provider.downloads} downloaded")
    return task.status == TaskStatus.COMPLETED and provider.submits == 0 and provider.downloads == 0 \
        and len(reused) == len(generated.assets) > 0


async def run(data_dir: Path, *, out=print, save_to: Path | None = None) -> bool:
    _heading(out, "Lesson -> segment plan -> generation jobs -> clips -> composition -> MP4")
    provider = MockVideoGenerationProvider(processing_polls=2)
    container = build_container(Settings(data_dir=data_dir, corpus_dir=FIXTURES, video_composer="ffmpeg",
                                         generated_video_poll_interval_seconds=0.0),
                                video_generation_provider=provider)
    try:
        task = await run_lesson(container)
        out(f"task {task.task_id}: {task.status.value}")
        if task.status != TaskStatus.COMPLETED:
            for err in task.errors:
                out(f"error: [{err.kind}] {err.node_id}: {err.message}")
            return False
        ok = report(container, task, provider, out, save_to)
    finally:
        container.close()
    return await check_reuse(data_dir, out) and ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--out-dir", type=Path, default=None, help="also save the final MP4 here")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="teaching-agent-generative-video-") as tmp:
        ok = asyncio.run(run(args.data_dir or Path(tmp), save_to=args.out_dir))
    print(f"\n{'All checks passed.' if ok else 'Some checks FAILED.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
