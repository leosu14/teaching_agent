"""The generative video stage: optional short generated clips for a lesson video.

GenerativeVideoNode takes the stored VideoSegmentPlanSet through the ToolManager. For every planned segment it
submits a generation job (video_generation.submit; a job already known for the same generation key is reused, never
submitted twice), follows the open jobs with the generic bounded poller (video_generation.status), and turns each
completed job into a GENERATED_VIDEO_ASSET (video_generation.create_asset: download or reuse, validate the bytes,
normalise, validate again, store). Agents never wait on providers: this node, i.e. the workflow runtime, owns the
waiting.

Never blocking indefinitely: one run of the node polls for at most `poll_timeout_seconds`. If jobs are still running
then, their state is checkpointed and the task goes to WAITING (kind "video_generation"); resuming the task runs the
node again, which polls the same jobs (nothing is submitted again) and, once a job has used `poll_max_attempts`
polls in total, gives it up as failed.

Cancellation: the node checks the task's cancel request before every submission and between polls. Open jobs are
cancelled at the provider when it supports it, otherwise recorded as cancelled locally; no new job is started and
no artifact is written half-way. A task cancelled while WAITING here releases its jobs the same way (on_cancel).

Failure policy (GeneratedVideoConfig.failure_policy), applied per segment:
- an optional segment that fails (or does not fit the budget) is skipped with a warning; its slide keeps its
  fallback (the slide's existing IMAGE_ASSET, else the slide itself);
- a required segment that fails fails the task under `fail`, and falls back with a warning under `continue`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

from app.observability.budget import BudgetExceededError
from app.runtime.workflow.nodes import Node, NodeFatal, NodeResult, NodeRuntime, StateView
from app.schemas.events import EventType
from app.schemas.generative_video import (
    ClipAssetRequest,
    ClipAssetResult,
    GeneratedVideoResult,
    GeneratedVideoStageRequest,
    GenerationJobRequest,
    GenerationSubmitRequest,
    SegmentFailure,
    VideoGenerationJob,
    VideoGenerationStatus,
    VideoSegmentPlan,
)
from app.schemas.task import WaitRequest
from app.tools.base import ToolCaller, ToolError, ToolTransientError
from app.utils.polling import PollPolicy, poll_until

WAIT_KIND = "video_generation"
GENERATION_TOOLS = frozenset({"video_generation.submit", "video_generation.status", "video_generation.cancel",
                              "video_generation.create_asset"})
GENERATION_PERMISSIONS = frozenset({"media:generate", "artifact:read", "artifact:write"})


class GeneratedVideoRequired(NodeFatal):
    """A required generated clip could not be made and the failure policy is `fail`."""


@dataclass(frozen=True, kw_only=True)
class GenerativeVideoNode(Node):
    build_input: Callable[[StateView], GeneratedVideoStageRequest]
    sleep: Callable[[float], Awaitable[None]] | None = None  # injectable for tests; asyncio.sleep otherwise
    kind: ClassVar[str] = "generative_video"

    def _caller(self) -> ToolCaller:
        return ToolCaller(caller_id=f"node:{self.id}", allowed_tools=GENERATION_TOOLS,
                          permissions=GENERATION_PERMISSIONS)

    async def execute(self, rt: NodeRuntime) -> NodeResult:
        data = self.build_input(rt.view)
        plan, config = data.plan, data.config
        state = _Progress(rt)
        caller = self._caller()
        warnings = list(plan.warnings)

        if plan.over_budget_required and config.failure_policy == "fail":
            raise GeneratedVideoRequired(
                f"required generated clips for sections {', '.join(plan.over_budget_required)} do not fit the "
                "generated-video budget (MAX_GENERATED_VIDEO_SEGMENTS / MAX_GENERATED_VIDEO_SECONDS / "
                "MAX_VIDEO_GENERATION_COST_USD)")
        if not plan.segments:
            return NodeResult(output=GeneratedVideoResult(status="skipped", plan_id=plan.plan_id, warnings=warnings))

        # 1. Submit every segment that has no job yet (a resumed run skips this for submitted ones).
        for segment in plan.segments:
            sid = segment.segment_id
            if sid in state.assets or sid in state.failures or sid in state.jobs:
                continue
            if rt.cancel_requested():
                break
            try:
                job = await rt.tools.call(caller, "video_generation.submit", GenerationSubmitRequest(segment=segment),
                                          rt.scope)
            except BudgetExceededError as exc:
                state.fail(segment, "budget", str(exc))
                continue
            except ToolError as exc:
                state.fail(segment, "submit", str(exc))
                continue
            assert isinstance(job, VideoGenerationJob)
            state.jobs[sid] = job
            if job.status == VideoGenerationStatus.FAILED:
                state.fail(segment, "poll", job.error or "the provider reported the job as failed")
            state.save()

        # 2. Follow the open jobs, bounded; still running at the end of the window -> WAITING (resumable).
        if not rt.cancel_requested():
            waiting = await self._poll(rt, plan.segments, state, config, caller)
            if waiting is not None:
                return NodeResult(wait=waiting)

        if rt.cancel_requested():
            await self._cancel_open(rt, plan.segments, state, caller)
            return NodeResult(output=state.result(plan.plan_id, "cancelled", warnings))

        # 3. Turn every completed job into a GENERATED_VIDEO_ASSET.
        for segment in plan.segments:
            sid = segment.segment_id
            job = state.jobs.get(sid)
            if sid in state.assets or sid in state.failures or job is None:
                continue
            if job.status != VideoGenerationStatus.COMPLETED:
                stage = "cancelled" if job.status == VideoGenerationStatus.CANCELLED else "poll"
                state.fail(segment, stage, job.error or f"the job ended {job.status.value}")
                continue
            try:
                result = await rt.tools.call(caller, "video_generation.create_asset", ClipAssetRequest(
                    segment=segment, job=job, plan_id=plan.plan_id, parent_ids=[data.plan_artifact_id]), rt.scope)
            except ToolError as exc:
                state.fail(segment, getattr(exc, "stage", "artifact"), str(exc))
                continue
            assert isinstance(result, ClipAssetResult)
            state.assets[sid] = result
            state.save()

        # 4. The failure policy.
        for segment in plan.segments:
            failure = state.failures.get(segment.segment_id)
            if failure is None:
                continue
            rt.scope.emit(EventType.GENERATED_VIDEO_FALLBACK, segment_id=segment.segment_id,
                          slide_id=segment.slide_id, stage=failure.stage, required=segment.required,
                          fallback=failure.fallback.kind, fallback_artifact_id=failure.fallback.artifact_id,
                          error=failure.error[:500])
            if segment.required and config.failure_policy == "fail":
                raise GeneratedVideoRequired(f"the required generated clip {segment.segment_id} failed at "
                                             f"{failure.stage}: {failure.error}"[:1500])
            kind = "required" if segment.required else "optional"
            shown = ("its existing image" if failure.fallback.kind == "image_asset" else "the slide")
            warnings.append(f"The {kind} generated clip for slide {segment.slide_id} was not used ({failure.stage} "
                            f"failed: {failure.error[:200]}); {shown} is shown instead.")
        status = ("complete" if not state.failures else "partial" if state.assets else "failed")
        return NodeResult(output=state.result(plan.plan_id, status, warnings))

    async def _poll(self, rt: NodeRuntime, segments: list[VideoSegmentPlan], state: _Progress, config,
                    caller: ToolCaller) -> WaitRequest | None:
        by_id = {s.segment_id: s for s in segments}

        def open_ids() -> list[str]:
            return [sid for sid, job in state.jobs.items()
                    if sid not in state.failures and sid not in state.assets and not job.status.terminal]

        # jobs that already used every poll they are allowed fail now: the stage never waits forever
        for sid in open_ids():
            if state.jobs[sid].polls >= config.poll_max_attempts:
                state.fail(by_id[sid], "poll", f"the job was still {state.jobs[sid].status.value} after "
                                               f"{state.jobs[sid].polls} polls")
        if not open_ids():
            return None
        remaining = max(1, config.poll_max_attempts - max(state.jobs[sid].polls for sid in open_ids()))
        policy = PollPolicy(interval_seconds=config.poll_interval_seconds,
                            timeout_seconds=config.poll_timeout_seconds, max_attempts=remaining)

        async def fetch(_attempt: int) -> list[str]:
            for sid in open_ids():
                job = await rt.tools.call(caller, "video_generation.status", GenerationJobRequest(job=state.jobs[sid]),
                                          rt.scope)
                assert isinstance(job, VideoGenerationJob)
                state.jobs[sid] = job
                state.save()
            return open_ids()

        try:
            outcome = await poll_until(fetch, lambda still_open: not still_open, policy,
                                       cancelled=rt.cancel_requested,
                                       retryable=lambda exc: isinstance(exc, ToolTransientError),
                                       sleep=self.sleep or asyncio.sleep)
        except ToolError as exc:
            for sid in open_ids():
                state.fail(by_id[sid], "poll", str(exc))
            return None
        if outcome.stopped in ("timeout", "exhausted") and open_ids():
            still = open_ids()
            if all(state.jobs[sid].polls >= config.poll_max_attempts for sid in still):
                for sid in still:
                    state.fail(by_id[sid], "poll", f"the job was still {state.jobs[sid].status.value} after "
                                                   f"{state.jobs[sid].polls} polls")
                return None
            rt.scope.emit(EventType.VIDEO_GENERATION_WAITING, jobs=[state.jobs[sid].job_id for sid in still],
                          segments=still, polls=outcome.attempts, elapsed_seconds=round(outcome.elapsed_seconds, 3),
                          errors=outcome.errors[-3:])
            return WaitRequest(node_id=self.id, kind=WAIT_KIND, prompt={
                "message": "Generated video clips are still being produced; resume the task to check them again.",
                "jobs": [{"segment_id": sid, "job_id": state.jobs[sid].job_id, "provider": state.jobs[sid].provider,
                          "status": state.jobs[sid].status.value, "polls": state.jobs[sid].polls} for sid in still]})
        return None

    async def _cancel_open(self, rt: NodeRuntime, segments: list[VideoSegmentPlan], state: _Progress,
                           caller: ToolCaller) -> None:
        for segment in segments:
            sid = segment.segment_id
            if sid in state.assets or sid in state.failures:
                continue
            job = state.jobs.get(sid)
            if job is not None and not job.status.terminal:
                try:
                    job = await rt.tools.call(caller, "video_generation.cancel", GenerationJobRequest(job=job),
                                              rt.scope)
                    assert isinstance(job, VideoGenerationJob)
                    state.jobs[sid] = job
                except ToolError as exc:
                    state.jobs[sid] = job.advanced(VideoGenerationStatus.CANCELLED, error=str(exc)).model_copy(
                        update={"cancellation": "local"})
            state.fail(segment, "cancelled", "the task was cancelled"
                       + (f" (cancellation: {state.jobs[sid].cancellation})" if sid in state.jobs else ""))

    async def on_cancel(self, rt: NodeRuntime) -> None:
        """The task was cancelled while waiting here: cancel the jobs that are still open."""
        data = self.build_input(rt.view)
        state = _Progress(rt)
        await self._cancel_open(rt, data.plan.segments, state, self._caller())


class _Progress:
    """The node's checkpointed progress: jobs, finished assets and failures, by segment id."""

    def __init__(self, rt: NodeRuntime) -> None:
        self._rt = rt
        progress = rt.node_state.progress
        self.jobs = {k: VideoGenerationJob.model_validate(v) for k, v in progress.get("jobs", {}).items()}
        self.assets = {k: ClipAssetResult.model_validate(v) for k, v in progress.get("assets", {}).items()}
        self.failures = {k: SegmentFailure.model_validate(v) for k, v in progress.get("failures", {}).items()}

    def save(self) -> None:
        progress = self._rt.node_state.progress
        progress["jobs"] = {k: v.model_dump(mode="json") for k, v in self.jobs.items()}
        progress["assets"] = {k: v.model_dump(mode="json") for k, v in self.assets.items()}
        progress["failures"] = {k: v.model_dump(mode="json") for k, v in self.failures.items()}
        self._rt.checkpoint()

    def fail(self, segment: VideoSegmentPlan, stage: str, error: str) -> None:
        self.failures[segment.segment_id] = SegmentFailure(
            segment_id=segment.segment_id, lesson_section_id=segment.lesson_section_id, required=segment.required,
            stage=stage if stage in ("budget", "submit", "poll", "download", "validate", "normalize", "artifact",
                                     "cancelled") else "artifact",
            error=error[:1500], fallback=segment.fallback)
        self.save()

    def result(self, plan_id: str, status: str, warnings: list[str]) -> GeneratedVideoResult:
        return GeneratedVideoResult(
            status=status, plan_id=plan_id, assets=[a.asset for a in self.assets.values()],
            artifacts=[a.artifact for a in self.assets.values()], failures=list(self.failures.values()),
            jobs=list(self.jobs.values()), warnings=warnings)
