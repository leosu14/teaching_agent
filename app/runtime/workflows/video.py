"""The video composition stage and the video failure policy.

VideoNode takes a validated VideoPlan through the ToolManager: compose (video.compose; reuses an equivalent VIDEO
artifact when one exists), validate the MP4 (video.validate), then create the VIDEO artifact
(video.create_artifact). Progress is checkpointed after composition and after validation, so a task interrupted
here resumes without composing again.

Video failure policy, applied inside the node:
- `fail` (the default: video is required): any failure fails the task with an explicit, persisted error. No VIDEO
  artifact is created; every artifact made earlier (lesson, images, presentation, audio, timeline, video plan)
  stays available.
- `continue` (video optional): the task completes with a warning and without a VIDEO artifact. Nothing fake is
  ever stored in its place.
An invalid VideoPlan is stopped earlier, by the plan validation gate, under either policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from app.runtime.workflow.nodes import Node, NodeFatal, NodeResult, NodeRuntime, StateView
from app.schemas.artifact import Artifact
from app.schemas.video import (
    VideoArtifactMetadata,
    VideoArtifactRequest,
    VideoComposeRequest,
    VideoComposeResult,
    VideoResult,
    VideoStageRequest,
    VideoValidationInput,
    VideoValidationReport,
)
from app.tools.base import ToolCaller

VideoFailurePolicy = Literal["fail", "continue"]

VIDEO_TOOLS = frozenset({"video.compose", "video.validate", "video.create_artifact"})
VIDEO_PERMISSIONS = frozenset({"artifact:read", "artifact:write"})


class VideoRequired(NodeFatal):
    """The video could not be made and the workflow requires it."""


class VideoRejected(Exception):
    """The composed file failed validation."""


@dataclass(frozen=True, kw_only=True)
class VideoNode(Node):
    build_input: Callable[[StateView], VideoStageRequest]
    policy: VideoFailurePolicy = "fail"
    kind: ClassVar[str] = "video"

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        data = self.build_input(rt.view)
        plan = data.plan
        progress = rt.node_state.progress
        caller = ToolCaller(caller_id=f"node:{self.id}", allowed_tools=VIDEO_TOOLS, permissions=VIDEO_PERMISSIONS)
        stage = "compose"
        try:
            if "composed" in progress:
                composed = VideoComposeResult.model_validate(progress["composed"])
            else:
                composed = await rt.tools.call(caller, "video.compose",
                                               VideoComposeRequest(plan=plan, name=data.name), rt.scope)
                assert isinstance(composed, VideoComposeResult)
                progress["composed"] = composed.model_dump(mode="json")
                rt.checkpoint()
            if composed.reused and composed.artifact is not None:
                existing = VideoArtifactMetadata.model_validate(composed.artifact.metadata)
                return NodeResult(output=VideoResult(status="complete", video_plan_id=plan.video_plan_id,
                                                     artifact=composed.artifact, composed=composed,
                                                     validation=existing.validation))
            stage = "validate"
            if "validation" in progress:
                report = VideoValidationReport.model_validate(progress["validation"])
            else:
                report = await rt.tools.call(caller, "video.validate",
                                             VideoValidationInput(plan=plan, composed=composed), rt.scope)
                assert isinstance(report, VideoValidationReport)
                progress["validation"] = report.model_dump(mode="json")
                rt.checkpoint()
            if not report.valid:
                raise VideoRejected("; ".join(f"{e.code} ({e.field}): {e.message}"
                                              + (f" expected {e.expected}, got {e.actual}" if e.expected else "")
                                              for e in report.errors))
            stage = "artifact"
            artifact = await rt.tools.call(caller, "video.create_artifact", VideoArtifactRequest(
                plan=plan, composed=composed, validation=report, parent_ids=data.parent_ids, name=data.name),
                rt.scope)
            assert isinstance(artifact, Artifact)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:1500]
            if self.policy == "fail":
                raise VideoRequired(f"video {plan.video_plan_id} failed at {stage}: {error}") from exc
            return NodeResult(output=VideoResult(
                status="failed", video_plan_id=plan.video_plan_id, failed_stage=stage, error=error,
                composed=VideoComposeResult.model_validate(progress["composed"]) if "composed" in progress else None,
                warnings=[f"No video was produced (video is optional): {stage} failed: {error}"]))
        return NodeResult(output=VideoResult(status="complete", video_plan_id=plan.video_plan_id, artifact=artifact,
                                             composed=composed, validation=report))
