"""Optional smoke tests against the real configured providers.

Skipped unless RUN_PROVIDER_SMOKE_TESTS=true, and each test is skipped unless its capability is configured with a
real provider and credentials. They send one minimal request each, so they cost a little; CI never runs them.

    RUN_PROVIDER_SMOKE_TESTS=true LLM_PROVIDER=anthropic LLM_MODEL=... ANTHROPIC_API_KEY=... pytest tests/smoke
"""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image
from pydantic import BaseModel

from app.config.providers import MOCK
from app.config.settings import Settings
from app.observability.scope import ExecutionScope, UsageLedger
from app.providers.image.base import ProviderImageRequest
from app.providers.llm.structured import StructuredLLM
from app.providers.search.base import ProviderSearchRequest
from app.providers.tts.base import ProviderSpeechRequest
from app.schemas.common import ModelTier
from app.schemas.providers import Capability
from app.services.container import build_container
from app.utils.audio import probe_audio

pytestmark = pytest.mark.skipif(os.environ.get("RUN_PROVIDER_SMOKE_TESTS", "").lower() != "true",
                                reason="set RUN_PROVIDER_SMOKE_TESTS=true to call the real providers")


class Answer(BaseModel):
    answer: str
    value: int


@pytest.fixture
def real(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", log_json=False)
    if settings.providers.offline:
        pytest.skip("TEACHING_AGENT_OFFLINE=true")
    container = build_container(settings)  # raises ConfigError naming any missing variable
    yield container
    container.close()


def require(container, capability: Capability) -> None:
    if container.providers.selector.select(capability).provider == MOCK:
        pytest.skip(f"no real {capability.value} provider configured")


async def test_llm_structured_output(real) -> None:
    require(real, Capability.LLM)
    scope = ExecutionScope(events=real.events, usage=UsageLedger(), task_id="smoke")
    result = await StructuredLLM(real.router).generate(
        Answer, agent_id="teacher", tier=ModelTier.CHEAP, system="You answer arithmetic questions in JSON.",
        prompt='What is 2 + 2? Return {"answer": "<the number in Spanish words>", "value": <the number>}.',
        scope=scope, max_output_tokens=200, timeout_seconds=120)
    assert result.value == 4
    assert scope.usage.summary.llm_calls >= 1


async def test_tts_synthesis(real) -> None:
    require(real, Capability.TTS)
    voice = (await real.providers.tts.voices())[0]
    speech = await real.providers.tts.synthesize(ProviderSpeechRequest(
        text="Hola.", language=voice.language, voice_id=voice.voice_id, format="wav"))
    assert probe_audio(speech.content).duration == pytest.approx(speech.duration, abs=0.05) > 0


async def test_image_generation(real) -> None:
    require(real, Capability.IMAGE)
    image = await real.providers.image_generation.generate(ProviderImageRequest(
        prompt="A simple flat diagram of a circle split into four equal parts", width=512, height=512))
    with Image.open(io.BytesIO(image.content)) as img:
        assert img.size == (512, 512)


async def test_web_search(real) -> None:
    require(real, Capability.SEARCH)
    page = await real.providers.search.search(ProviderSearchRequest(query="Pythagorean theorem", max_results=3))
    assert page.hits and all(h.url.startswith("https://") or h.url.startswith("http://") for h in page.hits)
