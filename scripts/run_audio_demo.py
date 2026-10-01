"""End-to-end demo of the Audio slice.

    python scripts/run_audio_demo.py [--out-dir narration/]

Runs the same lesson as run_demo.py offline with the mock providers (Diagnostic -> Research -> Planner -> Teacher ->
Reviewer -> Visual -> Slide Planning -> Presentation Build -> Presentation Render -> Audio Planning -> TTS -> Audio
Assets -> Presentation Timeline), then walks through the audio stages: the approved lesson, the presentation, the
AudioPlan and its validation, the mock TTS calls, the audio validation, the AUDIO_ASSET artifacts, the
PresentationTimeline with every slide's timing, the total audio duration and the artifact references. Every WAV
is opened with Python's `wave` module to show it is real audio. It uses the same TaskService as the API.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import sys
import tempfile
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import Settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.artifact import ArtifactType  # noqa: E402
from app.schemas.audio import (  # noqa: E402
    AudioAssetMetadata,
    AudioPlan,
    AudioPlanValidationReport,
    NarrationResult,
    PresentationTimeline,
)
from app.schemas.lesson import LessonContent  # noqa: E402
from app.schemas.presentation import PresentationArtifactMetadata, SlideDeckPlan  # noqa: E402
from app.schemas.task import Task, TaskStatus  # noqa: E402
from app.services.container import Container, build_container  # noqa: E402
from scripts.run_demo import run_demo  # noqa: E402


def _clip(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _heading(out, title: str) -> None:
    out("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def report_audio(container: Container, task: Task, out=print, save_to: Path | None = None) -> None:
    arts = {a.name: a for a in container.task_service.artifacts(task.task_id)}
    nodes = task.workflow.node_states
    events = container.task_service.events(task.task_id)

    _heading(out, "1. Approved lesson")
    lesson = LessonContent.model_validate_json(container.artifacts.read(arts["lesson"].artifact_id))
    out(f"{lesson.title} ({lesson.level}); review: {task.result.review_verdict}; "
        f"lesson artifact {arts['lesson'].artifact_id}")

    _heading(out, "2. Presentation")
    pres = arts["presentation"]
    meta = PresentationArtifactMetadata.model_validate(pres.metadata)
    deck = SlideDeckPlan.model_validate_json(container.artifacts.read(arts["slide_plan"].artifact_id))
    out(f"{pres.artifact_id} {pres.type.value} v{pres.version}: {meta.slides} slides, deck {meta.deck_id}, "
        f"sha256={pres.content_hash[:16]}...")

    _heading(out, "3. AudioPlan (audio planner)")
    plan = AudioPlan.model_validate_json(container.artifacts.read(arts["audio_plan"].artifact_id))
    out(f"{plan.audio_plan_id}: language={plan.language} voice={plan.voice} segments={len(plan.segments)} "
        f"estimated={plan.expected_duration():.1f}s; artifact {arts['audio_plan'].artifact_id}")
    for s in plan.segments:
        out(f"  {s.order:>2}. {s.segment_id:<8} slide={s.slide_id} {s.source_type.value:<22} "
            f"{'required' if s.required else 'optional'}  {_clip(s.text, 54)!r}")
    silent = [s.slide_id for s in deck.slides if not plan.for_slide(s.slide_id)]
    out(f"  slides without narration: {', '.join(silent) or '-'}")

    _heading(out, "4. Validation")
    report = AudioPlanValidationReport.model_validate(nodes["validate_audio_plan"].output)
    out(f"valid={report.valid} by {report.validator}; errors={len(report.errors)}")
    out(f"checks: {', '.join(report.checks)}")

    _heading(out, "5. Mock TTS")
    for e in events:
        if e.type == "tts.completed":
            out(f"  {e.data['segment_id']:<8} provider={e.data['provider']} model={e.data['model']} "
                f"duration={e.data['duration']:.3f}s chars={e.data['usage']['characters']} "
                f"sha256={e.data['checksum'][:12]}...")

    _heading(out, "6. Audio validation")
    assets = {a.metadata["segment_id"]: a for a in arts.values() if a.type == ArtifactType.AUDIO_ASSET}
    for seg_id, art in assets.items():
        m = AudioAssetMetadata.model_validate(art.metadata)
        out(f"  {seg_id:<8} valid={m.validation.valid} {m.format} {m.media_type} {m.sample_rate}Hz "
            f"{m.channels}ch {m.duration:.3f}s checks={len(m.validation.checks)}")
    rejected = [e for e in events if e.type == "audio.validation_failed"]
    out(f"  rejected: {len(rejected)}")

    _heading(out, "7. AUDIO_ASSET artifacts")
    readable = 0
    for seg_id, art in assets.items():
        data = container.artifacts.read(art.artifact_id)
        with wave.open(io.BytesIO(data)) as wav:  # a real WAV file, read by the standard library
            frames, rate = wav.getnframes(), wav.getframerate()
        readable += frames > 0
        out(f"  {art.artifact_id} {art.name:<16} {art.size_bytes:>7} bytes  wave: {frames} frames @ {rate}Hz  "
            f"bytes match: {hashlib.sha256(data).hexdigest() == art.content_hash}")
        if save_to is not None:
            (save_to / f"{art.name}.wav").write_bytes(data)
    out(f"  {readable}/{len(assets)} readable WAV files; "
        f"{len({a.content_hash for a in assets.values()})} distinct objects in the store")

    _heading(out, "8. PresentationTimeline")
    tl_art = arts["presentation_timeline"]
    timeline = PresentationTimeline.model_validate_json(container.artifacts.read(tl_art.artifact_id))
    out(f"{timeline.timeline_id} ({timeline.resolver}); presentation {timeline.presentation_artifact_id}; "
        f"artifact {tl_art.artifact_id} {tl_art.type.value} v{tl_art.version}")
    out(f"missing segments: {', '.join(timeline.missing_segments) or '-'}")

    _heading(out, "9. Slide timings")
    titles = {s.slide_id: s.title for s in deck.slides}
    for s in timeline.slides:
        out(f"  Slide {s.order:>2} {s.slide_id} {_clip(titles[s.slide_id], 30)!r:<34} "
            f"{s.start_time:7.3f} -> {s.end_time:7.3f}  ({s.duration:.3f}s)  "
            f"segments: {', '.join(s.audio_segment_refs) or 'silent'}")
        for seg in (x for x in timeline.segments if x.slide_id == s.slide_id):
            out(f"      {seg.segment_id:<8} {seg.start_time:7.3f} -> {seg.end_time:7.3f}")

    _heading(out, "10. Audio duration")
    spoken = sum(s.duration for s in timeline.segments)
    out(f"narration: {spoken:.3f}s in {len(timeline.segments)} segments; timeline: {timeline.duration:.3f}s "
        f"(with pauses and silent slides); planning estimate was {plan.expected_duration():.1f}s")

    _heading(out, "11. Artifact references")
    name_of = lambda aid: container.artifacts.get(aid).name  # noqa: E731
    out(f"PRESENTATION           {pres.artifact_id}")
    out(f"AUDIO_PLAN             {arts['audio_plan'].artifact_id}  parents: "
        f"{', '.join(name_of(p) for p in arts['audio_plan'].parent_ids)}")
    out(f"AUDIO_ASSET x{len(assets):<9} {', '.join(a.artifact_id for a in list(assets.values())[:2])}, ...  "
        f"parent: audio_plan")
    out(f"PRESENTATION_TIMELINE  {tl_art.artifact_id}  parents: presentation, audio_plan and "
        f"{len(tl_art.parent_ids) - 2} audio assets")
    narration = NarrationResult.model_validate(nodes["audio_policy"].output)
    out(f"narration status: {narration.status}; warnings: {len(narration.warnings)}")
    cost = task.cost
    planner = cost.by_agent["audio_planner"]
    tts = cost.by_service["tts:mock"]
    out(f"\naudio planner: calls={planner.calls} tokens={planner.usage.total_tokens} ${planner.cost_usd:.6f}; "
        f"tts: calls={tts.calls} cost=${tts.cost_usd} units={tts.units}")
    if save_to is not None:
        (save_to / "presentation_timeline.json").write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        out(f"\nsaved {len(assets)} WAV files and presentation_timeline.json to {save_to}")


async def run_audio_demo(container: Container, *, out=print, save_to: Path | None = None) -> Task:
    _heading(out, "Lesson -> ... -> Presentation -> AudioPlan -> TTS -> AudioAsset -> PresentationTimeline")
    task = await run_demo(container, out=lambda line: None)
    out(f"task {task.task_id}: {task.status.value}")
    if task.status != TaskStatus.COMPLETED:
        for err in task.errors:
            out(f"error: [{err.kind}] {err.node_id}: {err.message}")
        return task
    report_audio(container, task, out, save_to)
    return task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="where to keep the database and artifacts (default: a fresh temporary directory)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="also save the WAV files and the timeline JSON here")
    parser.add_argument("--verbose", action="store_true", help="also print structured event logs to stderr")
    args = parser.parse_args()
    configure_logging("INFO" if args.verbose else "WARNING")
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="teaching-agent-audio-demo-") as tmp:
        data_dir = args.data_dir or Path(tmp)
        container = build_container(Settings(data_dir=data_dir))
        try:
            task = asyncio.run(run_audio_demo(container, save_to=args.out_dir))
        finally:
            container.close()
        if args.data_dir is None and args.out_dir is None:
            print("\n(artifacts were written to a temporary directory; pass --out-dir narration/ to keep the audio)")
    return 0 if task.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
