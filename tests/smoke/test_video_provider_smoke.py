"""Optional smoke test against the real configured video generation provider (MiniMax).

Skipped unless RUN_VIDEO_PROVIDER_SMOKE_TESTS=true, and skipped unless VIDEO_GENERATION_PROVIDER names a real provider
with credentials in production mode. It generates one short clip (a few cents with MiniMax's pricing), polls it with
the generic poller, downloads it and validates the bytes with ffprobe. CI never runs it.

    RUN_VIDEO_PROVIDER_SMOKE_TESTS=true TEACHING_AGENT_MODE=production VIDEO_GENERATION_PROVIDER=minimax \
        MINIMAX_API_KEY=... pytest tests/smoke/test_video_provider_smoke.py
"""

from __future__ import annotations

import hashlib
import os
import shutil

import pytest

from app.config.providers import MOCK
from app.config.settings import Settings
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.providers.video.ffmpeg import FFprobeVideoProber
from app.schemas.generative_video import ClipExpectation, VideoGenerationRequest, VideoGenerationStatus
from app.schemas.providers import Capability
from app.services.container import build_container
from app.tools.video.clips import ClipValidator
from app.utils.polling import PollPolicy

pytestmark = pytest.mark.skipif(os.environ.get("RUN_VIDEO_PROVIDER_SMOKE_TESTS", "").lower() != "true",
                                reason="set RUN_VIDEO_PROVIDER_SMOKE_TESTS=true to call the real video provider")


@pytest.fixture
def real(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", log_json=False)
    if settings.providers.offline:
        pytest.skip("offline mode: set TEACHING_AGENT_MODE=production (and not TEACHING_AGENT_OFFLINE=true)")
    container = build_container(settings)  # raises ConfigError naming any missing variable
    if container.providers.selector.select(Capability.VIDEO_GENERATION).provider == MOCK:
        container.close()
        pytest.skip("no real video generation provider configured (VIDEO_GENERATION_PROVIDER)")
    if shutil.which("ffprobe") is None:
        container.close()
        pytest.skip("ffprobe is needed to validate the clip")
    yield container
    container.close()


async def test_one_short_clip_is_generated_downloaded_and_valid(real, tmp_path) -> None:
    provider = real.providers.video_generation
    limits = provider.limits()
    request = VideoGenerationRequest(
        prompt="Educational video clip, 6 seconds: water evaporating from a warm lake and rising as vapour. Calm "
               "camera, natural light, no on-screen text.",
        duration=min(limits.durations or [limits.max_duration]), width=1920, height=1080, fps=24)
    provider.check(request)
    scope = ExecutionScope(events=real.events, usage=UsageLedger(), task_id="smoke")
    with active_scope(scope):
        result = await provider.generate(request, poll=PollPolicy(interval_seconds=10, timeout_seconds=900))
    assert result.status == VideoGenerationStatus.COMPLETED and result.content
    path = tmp_path / "clip.mp4"
    path.write_bytes(result.content)
    report = ClipValidator(FFprobeVideoProber()).validate(
        path, hashlib.sha256(result.content).hexdigest(),
        ClipExpectation(stage="raw", duration=request.duration, duration_tolerance=1.5, min_width=640,
                        min_height=360, aspect_ratio="16:9"))
    assert report.valid, report.errors
    assert scope.usage.usage().video_generations == 1
