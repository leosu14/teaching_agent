"""VideoService: the one place that drives a VideoComposer and a VideoProber against the object store.

    VideoAgent (plan) -> workflow -> video tools -> VideoService -> VideoComposer -> FFmpeg adapter

- compose: the composer writes into a scratch directory; the MP4 is streamed into the object store as a
  content-addressed object (never loaded whole, never in SQLite) and the scratch directory is removed. A failed
  composition keeps its directory (arguments, FFmpeg's log, the frames) for diagnosis.
- validate: a scratch copy of the stored object is measured by the prober; the checksum is recomputed while
  copying, so the bytes validated are the bytes stored.
- store: a VIDEO artifact is created only from a valid report and points at the stored object.
- Idempotency: a composition key (the plan id, which already covers every input checksum, the timeline checksum
  and the configuration, plus the composer version) identifies equivalent outputs. An existing VIDEO artifact with
  the same key whose bytes still match is reused instead of composing again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.video.base import VideoComposer, VideoProber
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.video import (
    AudioWindow,
    FrameSample,
    VideoArtifactMetadata,
    VideoArtifactRequest,
    VideoComposeResult,
    VideoPlan,
    VideoValidationReport,
    VideoValidationRequest,
)
from app.tools.video.validation import VideoValidator
from app.utils.workspace import ScratchSpace

WINDOW_MARGIN = 0.1  # seconds trimmed from each end of an audio window (encoder edges, fades)


class VideoArtifactRejected(ValueError):
    pass


class VideoService:
    def __init__(self, artifacts: ArtifactService, composer: VideoComposer, prober: VideoProber,
                 scratch: ScratchSpace) -> None:
        self.artifacts = artifacts
        self.composer = composer
        self.prober = prober
        self.scratch = scratch
        self.validator = VideoValidator(prober)

    def composition_key(self, plan: VideoPlan) -> str:
        body = json.dumps({"plan": plan.video_plan_id, "config": plan.config.model_dump(mode="json"),
                           "composer": self.composer.name}, sort_keys=True)
        return "vc_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]

    def find(self, task_id: str, name: str, key: str) -> Artifact | None:
        """This task's VIDEO artifact for the same composition key, if its stored bytes still match."""
        artifact = self.artifacts.find(task_id, name)
        if artifact is None or artifact.type != ArtifactType.VIDEO or artifact.metadata.get("composition_key") != key:
            return None
        try:
            if self.artifacts.object_checksum(artifact.uri) != artifact.content_hash:
                return None
        except (OSError, ValueError):
            return None
        return artifact

    def compose(self, plan: VideoPlan, key: str) -> VideoComposeResult:
        with self.scratch.session(f"compose-{plan.video_plan_id}") as workspace:
            composed = self.composer.compose(plan, workspace)
            stored = self.artifacts.put_object_file(Path(composed.path), composed.media_type)
        return VideoComposeResult(
            composition_key=key, object=stored, composer=composed.composer, frames=composed.frames,
            duration=composed.duration, render_seconds=composed.render_seconds, cpu_seconds=composed.cpu_seconds,
            subtitles_burned=composed.subtitles_burned)

    def validation_request(self, plan: VideoPlan, composed: VideoComposeResult) -> VideoValidationRequest:
        windows = []
        for seg in plan.slides:
            mixed = seg.generated is not None and seg.generated.audio == "mixed"  # the clip's own sound plays
            if not seg.audio_refs and not mixed and seg.duration > 3 * WINDOW_MARGIN:
                windows.append(AudioWindow(label=f"{seg.segment_id}:silence", start_time=seg.start_time + WINDOW_MARGIN,
                                           end_time=seg.end_time - WINDOW_MARGIN, expect_sound=False))
        for t in plan.audio_tracks:
            if t.duration > 3 * WINDOW_MARGIN:
                windows.append(AudioWindow(label=f"{t.track_id}:narration", start_time=t.start_time + WINDOW_MARGIN,
                                           end_time=t.end_time - WINDOW_MARGIN, expect_sound=True))
        samples, previous = [], None
        for seg in plan.slides:
            look = (seg.visual_ref.card.model_dump_json(), seg.visual_ref.image.checksum if seg.visual_ref.image else "",
                    seg.generated.checksum if seg.generated else "")
            samples.append(FrameSample(label=seg.segment_id, time=(seg.start_time + seg.end_time) / 2,
                                       expect_change=previous is not None and look != previous))
            previous = look
        return VideoValidationRequest(object=composed.object, config=plan.config, expected_duration=plan.duration,
                                      audio_windows=windows, frame_samples=samples)

    def validate(self, plan: VideoPlan, composed: VideoComposeResult) -> VideoValidationReport:
        request = self.validation_request(plan, composed)
        with self.scratch.session(f"validate-{plan.video_plan_id}") as workspace:
            local = workspace / "video.mp4"
            checksum = self.artifacts.copy_object_to(composed.object.uri, local)
            return self.validator.validate(request, local, checksum)

    def store(self, task_id: str, request: VideoArtifactRequest, scope: ExecutionScope) -> Artifact:
        plan, composed, report = request.plan, request.composed, request.validation
        if not report.valid or report.probe is None or report.probe.video is None:
            raise VideoArtifactRejected("a VIDEO artifact needs a valid validation report")
        probe, video = report.probe, report.probe.video
        obj = composed.object
        metadata = VideoArtifactMetadata(
            video_plan_id=plan.video_plan_id, composition_key=composed.composition_key, duration=probe.duration,
            width=video.width, height=video.height, fps=video.fps, codec=video.codec,
            audio_codec=probe.audio[0].codec if probe.audio else None, container=plan.config.container.value,
            media_type=obj.media_type, file_size=obj.size_bytes, checksum=obj.checksum, object_key=obj.key or "",
            timeline_ref=plan.timeline_ref, presentation_ref=plan.presentation_ref, timeline_duration=plan.duration,
            segments=len(plan.slides), audio_tracks=len(plan.audio_tracks), audio_stream=bool(probe.audio),
            subtitles=len(plan.subtitle_track.subtitles) if plan.subtitle_track else 0,
            subtitles_burned=bool(plan.subtitle_track and plan.subtitle_track.burned_in and composed.subtitles_burned),
            transition=plan.config.transition.value, composer=composed.composer,
            render_seconds=composed.render_seconds, validation=report,
            generated_segments=len(plan.generated_clips()),
            generated_artifact_ids=[c.artifact_id for c in plan.generated_clips()],
        )
        return self.artifacts.store_object(
            task_id=task_id, name=request.name, type=ArtifactType.VIDEO, obj=obj, provider=composed.composer,
            parent_ids=request.parent_ids, metadata=metadata.model_dump(mode="json"), scope=scope)
