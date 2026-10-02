"""Contracts every provider of a capability meets, mock and production adapter alike.

The production adapters run over in-process fakes of the vendor APIs (tests/fake_vendors.py), so the same
assertions hold for the code path a real deployment uses, without a network or credentials.
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from app.providers.image.base import ImageGenerationProvider, ProviderImageRequest
from app.providers.image.mock import MockImageGenerationProvider
from app.providers.image.openai import OpenAIImageGenerationProvider
from app.providers.image_search.base import ImageSearchProvider, ProviderImageSearchRequest
from app.providers.image_search.mock import MockImageSearchProvider
from app.providers.llm.anthropic import AnthropicLLMProvider
from app.providers.llm.base import LLMMessage, LLMProvider, LLMRequest
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.search.base import ProviderSearchRequest, SearchProvider
from app.providers.search.mock import MockSearchProvider
from app.providers.search.tavily import TavilySearchProvider
from app.providers.tts.base import ProviderSpeechRequest, TTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.tts.openai import OpenAITTSProvider
from app.schemas.providers import Capability, HealthStatus
from app.utils.audio import probe_audio
from tests.conftest import FIXTURES
from tests.fake_vendors import FAKE_KEY, FakeAnthropic, FakeChatModel, FakeOpenAI, FakeTavily, http_client

OPENAI = "https://api.openai.example/v1"
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "n": {"type": "integer"}},
          "required": ["answer", "n"]}
RESPONDERS = {"contract": lambda request: {"answer": "cuatro", "n": 4}}
SYSTEM = "Answer with JSON."


@pytest.fixture(autouse=True)
def online(monkeypatch):
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)


def llm_providers() -> dict[str, LLMProvider]:
    chat = FakeChatModel(RESPONDERS, {SYSTEM: "contract"})
    return {
        "mock": MockLLMProvider(RESPONDERS),
        "openai": OpenAICompatibleLLMProvider(http_client("openai", OPENAI, FakeOpenAI(chat))),
        "anthropic": AnthropicLLMProvider(http_client("anthropic", "https://api.anthropic.example",
                                                      FakeAnthropic(chat), headers={"x-api-key": FAKE_KEY})),
    }


def tts_providers() -> dict[str, TTSProvider]:
    return {"mock": MockTTSProvider(),
            "openai": OpenAITTSProvider(http_client("openai", OPENAI, FakeOpenAI()), languages=["es-ES", "en-US"])}


def image_providers() -> dict[str, ImageGenerationProvider]:
    return {"mock": MockImageGenerationProvider(),
            "openai": OpenAIImageGenerationProvider(http_client("openai", OPENAI, FakeOpenAI()))}


def search_providers() -> dict[str, SearchProvider]:
    corpus = FIXTURES / "web_corpus.json"
    return {"mock": MockSearchProvider(corpus),
            "tavily": TavilySearchProvider(http_client("tavily", "https://api.tavily.example", FakeTavily(corpus)))}


def by_id(factory):
    return pytest.mark.parametrize("pid", list(factory()))


async def common_contract(provider, capability: Capability) -> None:
    assert provider.provider_id == provider.name and capability in provider.capabilities
    config = provider.configuration()
    assert isinstance(config, dict) and FAKE_KEY not in json.dumps(config, default=str)
    health = await provider.health_check()
    assert isinstance(health, HealthStatus) and health.provider == provider.name and health.available


@by_id(llm_providers)
async def test_llm_contract(pid) -> None:
    provider = llm_providers()[pid]
    await common_contract(provider, Capability.LLM)
    response = await provider.generate(LLMRequest(
        model="model-x", system=SYSTEM, messages=[LLMMessage(role="user", content="2+2?")], max_output_tokens=100,
        agent_id="contract", response_schema=SCHEMA))
    assert json.loads(response.text) == {"answer": "cuatro", "n": 4}
    assert response.provider == pid and response.model == "model-x"
    assert response.usage.input_tokens > 0 and response.usage.output_tokens > 0


@by_id(tts_providers)
async def test_tts_contract(pid) -> None:
    provider = tts_providers()[pid]
    await common_contract(provider, Capability.TTS)
    voices = await provider.voices()
    voice = next(v for v in voices if v.language == "es-ES")
    assert all(v.provider == pid for v in voices) and "wav" in provider.formats
    speech = await provider.synthesize(ProviderSpeechRequest(text="Hola, ¿qué tal?", language="es-ES",
                                                             voice_id=voice.voice_id, format="wav"))
    probe = probe_audio(speech.content)
    assert probe.format == "wav" and speech.format == "wav" and speech.media_type == "audio/wav"
    assert probe.sample_rate == speech.sample_rate and probe.channels == speech.channels
    assert speech.duration == pytest.approx(probe.duration, abs=0.01) and speech.duration > 0
    assert speech.usage.characters == len("Hola, ¿qué tal?")


@by_id(image_providers)
async def test_image_contract(pid) -> None:
    provider = image_providers()[pid]
    await common_contract(provider, Capability.IMAGE)
    image = await provider.generate(ProviderImageRequest(prompt="A number line from 0 to 1", width=1280,
                                                         height=720))
    with Image.open(io.BytesIO(image.content)) as img:
        assert img.size == (1280, 720) == (image.width, image.height)
    assert image.origin == "generated" and image.media_type == "image/png" and image.usage.images == 1


@by_id(search_providers)
async def test_search_contract(pid) -> None:
    provider = search_providers()[pid]
    await common_contract(provider, Capability.SEARCH)
    page = await provider.search(ProviderSearchRequest(query="football vocabulary spanish", max_results=4))
    assert 0 < len(page.hits) <= 4 and page.usage.results == len(page.hits)
    for hit in page.hits:
        assert hit.url.startswith("http") and hit.title and hit.snippet
    assert provider.supports_domain_filter
    domain = page.hits[0].url.split("/")[2]
    only = await provider.search(ProviderSearchRequest(query="football vocabulary spanish", max_results=4,
                                                       include_domains=[domain]))
    assert only.hits and all(h.url.split("/")[2] == domain for h in only.hits)
    excluded = await provider.search(ProviderSearchRequest(query="football vocabulary spanish", max_results=4,
                                                           exclude_domains=[domain]))
    assert all(h.url.split("/")[2] != domain for h in excluded.hits)


async def test_image_search_contract() -> None:
    provider: ImageSearchProvider = MockImageSearchProvider(FIXTURES / "image_catalog.json")
    await common_contract(provider, Capability.IMAGE_SEARCH)
    page = await provider.search(ProviderImageSearchRequest(query="football fans", max_results=3))
    assert page.hits and all(h.origin in ("searched", "external") for h in page.hits)
    assert all(h.source_url and h.url for h in page.hits)
