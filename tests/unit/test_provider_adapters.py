"""The production adapters and their HTTP client, against in-process fakes of the vendor APIs."""

from __future__ import annotations

import io
import json

import httpx
import pytest
from PIL import Image

from app.config.routing import ConfigError
from app.providers.core.base import bind_request_id, release_request_id
from app.providers.core.errors import (
    ProviderAuthenticationError,
    ProviderInvalidRequest,
    ProviderOfflineError,
    ProviderRateLimit,
    ProviderResponseError,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.providers.core.http import HttpClient, validate_base_url
from app.providers.image.base import ProviderImageRequest
from app.providers.image.openai import OpenAIImageGenerationProvider, closest_size, sizes_for
from app.providers.llm.anthropic import AnthropicLLMProvider
from app.providers.llm.base import LLMMessage, LLMRequest
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.search.base import ProviderSearchRequest
from app.providers.search.tavily import TavilySearchProvider
from app.providers.tts.base import ProviderSpeechRequest
from app.providers.tts.openai import OpenAITTSProvider
from app.utils.audio import probe_audio
from tests.conftest import FIXTURES
from tests.fake_vendors import FAKE_KEY, FakeAnthropic, FakeChatModel, FakeOpenAI, FakeTavily, http_client

BASE = "https://api.openai.example/v1"


@pytest.fixture(autouse=True)
def online(monkeypatch):
    """These tests exercise the HTTP path against fakes; the suite-wide offline flag would refuse it."""
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE", raising=False)


def chat_request(schema: dict | None = None, **kwargs) -> LLMRequest:
    return LLMRequest(model="m", system="You are terse.", messages=[LLMMessage(role="user", content="Say hi")],
                      max_output_tokens=50, agent_id="greeter", response_schema=schema, **kwargs)


def responders():
    return {"greeter": lambda request: {"greeting": "hola"}}


HELLO_SCHEMA = {"type": "object", "properties": {"greeting": {"type": "string", "minLength": 1}},
                "required": ["greeting"]}


# --- HTTP client ---------------------------------------------------------------------------------------------


def test_base_urls_must_be_https_except_loopback() -> None:
    assert validate_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"
    assert validate_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    for bad in ("http://api.example.com", "ftp://x", "api.example.com", "https://user:pw@api.example.com"):
        with pytest.raises(ConfigError):
            validate_base_url(bad)


@pytest.mark.parametrize("status, error, transient", [
    (401, ProviderAuthenticationError, False), (403, ProviderAuthenticationError, False),
    (429, ProviderRateLimit, True), (500, ProviderUnavailable, True), (503, ProviderUnavailable, True),
    (529, ProviderUnavailable, True), (408, ProviderUnavailable, True), (400, ProviderInvalidRequest, False),
    (404, ProviderInvalidRequest, False), (422, ProviderInvalidRequest, False),
    (302, ProviderResponseError, False),
])
async def test_status_codes_map_to_typed_errors(status, error, transient) -> None:
    def handler(request):
        return httpx.Response(status, headers={"retry-after": "7", "x-request-id": "vend-1", "location": "/x"},
                              json={"error": {"message": f"echo {FAKE_KEY}"}})

    client = http_client("acme", "https://api.acme.example", handler)
    with pytest.raises(error) as err:
        await client.request("POST", "/thing", json_body={"a": 1})
    assert err.value.transient is transient and err.value.status == status and err.value.provider == "acme"
    assert FAKE_KEY not in str(err.value) and "vend-1" in str(err.value)
    if status == 429:
        assert err.value.retry_after == 7.0


async def test_timeouts_connection_failures_and_size_limits() -> None:
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderTimeout):
        await http_client("acme", "https://a.example", slow).request("GET", "/")
    with pytest.raises(ProviderUnavailable, match="connection failed"):
        await http_client("acme", "https://a.example", refused).request("GET", "/")
    big = http_client("acme", "https://a.example", lambda r: httpx.Response(200, content=b"x" * 5000),
                      max_response_bytes=1024)
    with pytest.raises(ProviderResponseError, match="1024-byte limit"):
        await big.request("GET", "/")
    small = http_client("acme", "https://a.example", lambda r: httpx.Response(200, json={}), max_request_bytes=1024)
    with pytest.raises(ProviderInvalidRequest, match="over the 1024-byte limit"):
        await small.request("POST", "/", json_body={"text": "y" * 2000})


async def test_offline_mode_refuses_requests_before_any_io(monkeypatch) -> None:
    calls = []
    client = http_client("acme", "https://a.example", lambda r: calls.append(r) or httpx.Response(200, json={}))
    monkeypatch.setenv("TEACHING_AGENT_OFFLINE", "true")
    with pytest.raises(ProviderOfflineError):
        await client.request("GET", "/")
    monkeypatch.delenv("TEACHING_AGENT_OFFLINE")
    offline_client = http_client("acme", "https://a.example", lambda r: calls.append(r) or httpx.Response(200),
                                 offline=True)
    with pytest.raises(ProviderOfflineError):
        await offline_client.request("GET", "/")
    assert calls == []


async def test_request_id_header_and_vendor_request_id() -> None:
    seen = []
    client = http_client("acme", "https://a.example",
                         lambda r: seen.append(r) or httpx.Response(200, json={"ok": 1},
                                                                    headers={"request-id": "req_vendor"}),
                         request_id_header="X-Client-Request-Id")
    token = bind_request_id("preq_abc")
    try:
        response = await client.request("POST", "/x", json_body={})
    finally:
        release_request_id(token)
    assert seen[0].headers["X-Client-Request-Id"] == "preq_abc" and response.vendor_request_id == "req_vendor"
    assert FAKE_KEY not in repr(client)


# --- LLM adapters --------------------------------------------------------------------------------------------


async def test_openai_compatible_request_and_response() -> None:
    fake = FakeOpenAI(FakeChatModel(responders(), {"You are terse.": "greeter"}))
    provider = OpenAICompatibleLLMProvider(http_client("openai", BASE, fake))
    response = await provider.generate(chat_request(HELLO_SCHEMA, temperature=0.3))
    assert json.loads(response.text) == {"greeting": "hola"} and response.structured
    assert response.provider == "openai" and response.usage.input_tokens > 0 and response.stop_reason == "end_turn"
    [body] = fake.bodies("/v1/chat/completions")
    assert body["messages"][0] == {"role": "system", "content": "You are terse."}
    assert body["max_completion_tokens"] == 50 and body["temperature"] == 0.3
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False
    assert "minLength" not in json.dumps(rf)  # unsupported keyword dropped; validated after parsing instead
    assert fake.requests[0].headers["Authorization"] == f"Bearer {FAKE_KEY}"


async def test_openai_compatible_falls_back_to_json_mode_and_omits_unset_temperature() -> None:
    fake = FakeOpenAI(FakeChatModel(responders(), {"You are terse.": "greeter"}))
    provider = OpenAICompatibleLLMProvider(http_client("openai", BASE, fake), max_tokens_param="max_tokens")
    free_form = {"type": "object", "properties": {"meta": {"type": "object"}}}
    response = await provider.generate(chat_request(free_form))
    [body] = fake.bodies()
    assert body["response_format"] == {"type": "json_object"} and not response.structured
    assert "temperature" not in body and body["max_tokens"] == 50
    no_native = OpenAICompatibleLLMProvider(http_client("openai", BASE, fake), native_structured_output=False)
    await no_native.generate(chat_request(HELLO_SCHEMA))
    assert "response_format" not in fake.bodies()[-1]


async def test_openai_refusals_and_malformed_responses_are_typed() -> None:
    def refusing(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "refusal": "no"}}]})

    with pytest.raises(ProviderResponseError, match="refused"):
        await OpenAICompatibleLLMProvider(http_client("openai", BASE, refusing)).generate(chat_request())
    with pytest.raises(ProviderResponseError, match="no choices"):
        await OpenAICompatibleLLMProvider(http_client("openai", BASE, lambda r: httpx.Response(
            200, json={"choices": []}))).generate(chat_request())
    with pytest.raises(ProviderResponseError, match="not JSON"):
        await OpenAICompatibleLLMProvider(http_client("openai", BASE, lambda r: httpx.Response(
            200, content=b"<html>"))).generate(chat_request())


async def test_anthropic_request_and_response() -> None:
    fake = FakeAnthropic(FakeChatModel(responders(), {"You are terse.": "greeter"}))
    provider = AnthropicLLMProvider(http_client("anthropic", "https://api.anthropic.example", fake,
                                                headers={"x-api-key": FAKE_KEY, "anthropic-version": "2023-06-01"}))
    response = await provider.generate(chat_request(HELLO_SCHEMA))
    assert json.loads(response.text) == {"greeting": "hola"} and response.structured
    assert response.usage.cached_input_tokens == 2 and response.vendor_request_id == "req_1"
    [body] = fake.bodies()
    assert body["system"] == "You are terse." and body["max_tokens"] == 50 and "temperature" not in body
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert fake.requests[0].url.path == "/v1/messages" and fake.requests[0].headers["x-api-key"] == FAKE_KEY

    def refusal(request):
        return httpx.Response(200, json={"stop_reason": "refusal", "stop_details": {"category": "cyber"},
                                         "content": []})

    with pytest.raises(ProviderResponseError, match="declined"):
        await AnthropicLLMProvider(http_client("anthropic", "https://a.example", refusal)).generate(chat_request())


async def test_llm_health_probes_are_cheap_authenticated_gets() -> None:
    fake = FakeOpenAI()
    status = await OpenAICompatibleLLMProvider(http_client("openai", BASE, fake)).health_check()
    assert status.available and status.checked == "network" and fake.requests[0].method == "GET"
    denied = OpenAICompatibleLLMProvider(http_client("openai", BASE, lambda r: httpx.Response(401, json={})))
    status = await denied.health_check()
    assert not status.available and "ProviderAuthenticationError" in status.error


# --- TTS adapter ---------------------------------------------------------------------------------------------


async def test_openai_tts_returns_a_measured_wav() -> None:
    fake = FakeOpenAI()
    tts = OpenAITTSProvider(http_client("openai", BASE, fake), languages=["es-ES", "en-US"])
    voices = await tts.voices()
    assert {v.language for v in voices} == {"es-ES", "en-US"} and all(v.provider == "openai" for v in voices)
    speech = await tts.synthesize(ProviderSpeechRequest(text="Hola a todos.", language="es-ES", voice_id="nova",
                                                        format="wav", speaking_rate=1.25))
    probe = probe_audio(speech.content)
    assert probe.format == "wav" and probe.sample_rate == 24000 and probe.channels == 1
    assert probe.duration == pytest.approx(speech.duration)
    assert speech.usage.characters == len("Hola a todos.") and speech.usage.cost_usd is None  # not reported
    [body] = fake.bodies()
    assert body["response_format"] == "pcm" and body["speed"] == 1.25 and "es-ES" in body["instructions"]
    for bad in (dict(voice_id="nobody"), dict(format="mp3"), dict(speaking_rate=9.0)):
        request = ProviderSpeechRequest(**{"text": "x", "language": "es-ES", "voice_id": "nova", "format": "wav",
                                           **bad})
        with pytest.raises(ProviderInvalidRequest):
            await tts.synthesize(request)


# --- Image adapter -------------------------------------------------------------------------------------------


def test_closest_supported_size() -> None:
    assert closest_size(1600, 900, sizes_for("gpt-image-1")) == (1536, 1024)
    assert closest_size(900, 1600, sizes_for("gpt-image-1")) == (1024, 1536)
    assert closest_size(800, 800, sizes_for("dall-e-3")) == (1024, 1024)
    assert closest_size(1600, 900, sizes_for("dall-e-3")) == (1792, 1024)


async def test_openai_image_is_cropped_to_the_requested_size_and_recorded() -> None:
    fake = FakeOpenAI()
    provider = OpenAIImageGenerationProvider(http_client("openai", BASE, fake))
    image = await provider.generate(ProviderImageRequest(prompt="A fraction diagram", width=1600, height=900))
    with Image.open(io.BytesIO(image.content)) as img:
        assert img.size == (1600, 900) and img.format == "PNG"
    assert (image.width, image.height, image.origin) == (1600, 900, "generated")
    assert image.metadata["native_size"] == "1536x1024" and image.metadata["postprocess"] == "centre_crop_resize"
    assert image.usage.images == 1 and image.usage.cost_usd is None
    [body] = fake.bodies()
    assert body["size"] == "1536x1024" and "response_format" not in body
    with pytest.raises(ProviderResponseError, match="no base64 image"):
        await OpenAIImageGenerationProvider(http_client("openai", BASE, lambda r: httpx.Response(
            200, json={"data": [{"url": "https://x"}]}))).generate(ProviderImageRequest(prompt="p", width=10,
                                                                                         height=10))


# --- Search adapter ------------------------------------------------------------------------------------------


async def test_tavily_search_maps_results_without_inventing_fields() -> None:
    fake = FakeTavily(FIXTURES / "web_corpus.json")
    provider = TavilySearchProvider(http_client("tavily", "https://api.tavily.example", fake))
    page = await provider.search(ProviderSearchRequest(query="football vocabulary", max_results=3,
                                                       include_domains=["language-school.example.com"]))
    assert page.hits and all("language-school.example.com" in h.url for h in page.hits)
    assert all(h.publisher is None and h.author is None and h.language is None for h in page.hits)
    assert all(h.content for h in page.hits) and page.usage.results == len(page.hits)
    [body] = fake.bodies()
    assert body["include_domains"] == ["language-school.example.com"] and body["max_results"] == 3
    assert fake.requests[0].headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert (await provider.health_check()).checked == "local"  # no free endpoint: no credits spent on health


async def test_http_client_is_the_only_network_path() -> None:
    """A provider without a transport would need the real network: offline mode refuses it first."""
    client = HttpClient(provider="openai", base_url=BASE, headers={"Authorization": f"Bearer {FAKE_KEY}"},
                        offline=True)
    with pytest.raises(ProviderOfflineError):
        await OpenAICompatibleLLMProvider(client).generate(chat_request())
