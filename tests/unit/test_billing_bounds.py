"""The run-wide cap binds only where a configured price bounds a request before it is sent (PR #53 review, B2 and
B3): TTS and image prices in each OpenAI model's own billing units, the quality and size of an image pinned, a
combination no price can bound refused at startup, and an LLM answer that names a dated snapshot of the requested
model reconciled from its reported tokens. Prices here are arbitrary test values, not vendor rates."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from app.config.billing import billing_problems, image_option_problems
from app.config.providers import ProviderSettings
from app.config.routing import ConfigError
from app.observability.events import EventBus
from app.observability.scope import ExecutionScope, UsageLedger, active_scope
from app.observability.spend import task_account
from app.providers.core.http import HttpResponse
from app.providers.image.base import GeneratedImage, ImageGenerationProvider, ProviderImageRequest
from app.providers.image.openai import OpenAIImageGenerationProvider
from app.providers.llm.base import LLMMessage, LLMProvider, LLMRequest, LLMResponse
from app.providers.managed import (
    ManagedImageGenerationProvider,
    ManagedLLMProvider,
    ManagedTTSProvider,
    input_token_bound,
    pricing_model,
)
from app.providers.tts.base import ProviderSpeechRequest
from app.providers.tts.mock import MockTTSProvider
from app.providers.tts.openai import OpenAITTSProvider
from app.schemas.common import TokenUsage
from app.schemas.spend import SpendState
from app.schemas.visual import ImageUsage
from tests.unit.test_spend_run_cap import capped, invoker, production, store  # noqa: F401 - the store fixture

# --- B2: TTS -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["tts-1", "tts-1-hd", "tts-1-hd-1106"])
def test_a_tts_model_billed_per_character_is_bounded_by_its_price(model: str) -> None:
    assert production(tts_model=model).cost_cap_problems() == []


@pytest.mark.parametrize("model", [None, "gpt-4o-mini-tts", "gpt-4o-tts", "some-new-tts"])
def test_a_token_billed_or_unknown_tts_model_stops_startup_under_the_cap(model: str | None) -> None:
    settings = production(tts_model=model)
    with pytest.raises(ConfigError, match="cannot bound this request before it is sent") as stopped:
        settings.validate_runtime()
    assert f"TTS_MODEL='{model or 'gpt-4o-mini-tts'}' is billed by audio output tokens" in str(stopped.value)
    assert production(tts_model=model).model_copy(update={"budget_total_max_cost_usd": None}).cost_cap_problems() == []


def test_an_unknown_tts_provider_cannot_be_bounded() -> None:
    assert billing_problems(tts_chain=["mock", "acme"], tts_model=None, image_chain=["mock"], image_model=None,
                            image_quality=None, image_size=None, image_input_price_per_mtok=None) == [
        "TTS provider 'acme': its billing is not known here, so no price can bound it"]


@pytest.mark.parametrize("speed", [0.25, 1.0, 4.0])
async def test_a_per_character_reservation_does_not_depend_on_speed(store, speed: float) -> None:  # noqa: F811
    """Per-character billing ignores speed (it changes only the audio length, which a token-billed model would
    bill, and those are refused): the same text is reserved and charged the same at any speed."""
    account = task_account(store, capped(1.0), task_id="t1", learner_id=None, budget=None, fence=None)
    text = "Hola, me llamo Ana. ¿Y tú?"
    speech = ProviderSpeechRequest(text=text, language="es-ES", voice_id="mock-es-ES-1", format="wav",
                                   speaking_rate=speed)
    with active_scope(ExecutionScope(events=EventBus(), usage=UsageLedger(account=account), task_id="t1")):
        await ManagedTTSProvider([MockTTSProvider()], invoker(), price_per_mchar=30.0).synthesize(speech)
    [entry] = [e for e in store.entries(task_id="t1") if e.state == SpendState.SETTLED]
    assert entry.estimated_cost_microusd == entry.final_cost_microusd == round(len(text) * 30.0)


class FakeHttp:
    base_url = "https://api.example.test/v1"

    def __init__(self, content: bytes, content_type: str = "application/octet-stream") -> None:
        self.bodies: list[dict] = []
        self.content, self.content_type = content, content_type

    async def request(self, method: str, path: str, *, json_body: dict | None = None, **_kw) -> HttpResponse:
        self.bodies.append(json_body or {})
        return HttpResponse(status=200, content=self.content, content_type=self.content_type,
                            vendor_request_id="req_1")


async def test_a_per_character_openai_tts_request_carries_no_billed_instructions() -> None:
    http = FakeHttp(b"\x00\x00" * 2400)
    tts = OpenAITTSProvider(http, model="tts-1", languages=["es-ES"])
    await tts.synthesize(ProviderSpeechRequest(text="Hola", language="es-ES", voice_id="alloy", format="wav",
                                             speaking_rate=0.5))
    assert "instructions" not in http.bodies[0] and http.bodies[0]["model"] == "tts-1"


# --- B2: images ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("model", "quality", "size"), [
    ("gpt-image-1", "low", "1024x1024"), ("gpt-image-1", "medium", "1536x1024"), ("gpt-image-1", "high", "1024x1536"),
    ("dall-e-3", "standard", "1024x1024"), ("dall-e-3", "hd", "1792x1024"), ("dall-e-2", "standard", "512x512"),
])
def test_a_pinned_quality_and_size_the_model_supports_is_bounded(model: str, quality: str, size: str) -> None:
    assert production(image_model=model, image_quality=quality, image_size=size).cost_cap_problems() == []


@pytest.mark.parametrize(("change", "problem"), [
    ({"image_quality": None}, "IMAGE_QUALITY is not set"),
    ({"image_size": None}, "IMAGE_SIZE is not set"),
    ({"image_quality": "auto"}, "IMAGE_QUALITY='auto' is not a quality of gpt-image-1"),
    ({"image_quality": "hd"}, "IMAGE_QUALITY='hd' is not a quality of gpt-image-1"),
    ({"image_size": "1792x1024"}, "IMAGE_SIZE='1792x1024' is not a size gpt-image-1 generates"),
    ({"image_model": "dall-e-2", "image_quality": "hd", "image_size": "512x512"},
     "IMAGE_QUALITY='hd' is not a quality of dall-e-2"),
    ({"image_input_price_per_mtok": None}, "IMAGE_INPUT_PRICE_PER_MTOK is not set"),
    ({"image_model": None, "image_input_price_per_mtok": None}, "gpt-image-1 also bills the prompt"),
    ({"image_model": "imagen-9"}, "IMAGE_MODEL='imagen-9' is not a known OpenAI image model"),
])
def test_an_unpinned_invalid_or_unpriced_image_combination_stops_startup(change: dict, problem: str) -> None:
    settings = production(**change)
    with pytest.raises(ConfigError) as stopped:
        settings.validate_runtime()
    assert problem in str(stopped.value)


def test_dall_e_needs_no_prompt_price_and_an_invalid_option_is_a_problem_even_without_the_cap() -> None:
    assert production(image_model="dall-e-3", image_quality="hd", image_size="1024x1792",
                      image_input_price_per_mtok=None).cost_cap_problems() == []
    loose = ProviderSettings(llm_routes={}, image_provider="openai", openai_api_key="sk-test-0123456789",
                             image_model="gpt-image-1", image_quality="ultra")
    assert any("IMAGE_QUALITY='ultra'" in p for p in loose.problems())
    assert image_option_problems("gpt-image-1", None, None) == []


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


async def test_the_openai_adapter_sends_the_pinned_quality_and_size_whatever_the_aspect_ratio() -> None:
    body = {"data": [{"b64_json": base64.b64encode(png(1024, 1024)).decode()}],
            "usage": {"input_tokens": 12, "output_tokens": 272}}
    http = FakeHttp(json.dumps(body).encode(), "application/json")
    adapter = OpenAIImageGenerationProvider(http, model="gpt-image-1", quality="low", size="1024x1024")
    image = await adapter.generate(ProviderImageRequest(prompt="un mercado", width=1600, height=900))
    sent = http.bodies[0]
    assert (sent["quality"], sent["size"], sent["n"]) == ("low", "1024x1024", 1)
    assert (image.width, image.height) == (1600, 900) and image.metadata["vendor_usage"]["input_tokens"] == 12
    unpinned = FakeHttp(json.dumps(body).encode(), "application/json")
    await OpenAIImageGenerationProvider(unpinned, model="gpt-image-1").generate(
        ProviderImageRequest(prompt="x", width=1600, height=900))
    assert "quality" not in unpinned.bodies[0] and unpinned.bodies[0]["size"] == "1536x1024"


class TokenBilledImages(ImageGenerationProvider):
    """A GPT image model as the managed layer sees it: the vendor's usage in the metadata."""
    name = "fake-image"
    requires_network = False

    def __init__(self, input_tokens: int | None) -> None:
        self.input_tokens = input_tokens

    async def generate(self, request: ProviderImageRequest) -> GeneratedImage:
        metadata = {} if self.input_tokens is None else {"vendor_usage": {"input_tokens": self.input_tokens,
                                                                          "output_tokens": 272}}
        return GeneratedImage(content=png(8, 8), media_type="image/png", width=8, height=8, model="gpt-image-1",
                              usage=ImageUsage(requests=1, images=1), metadata=metadata)


@pytest.mark.parametrize("reported", [9, None])
async def test_an_image_reserves_the_prompt_bound_and_settles_at_the_reported_prompt_tokens(
        store, reported: int | None) -> None:  # noqa: F811
    account = task_account(store, capped(1.0), task_id="t1", learner_id=None, budget=None, fence=None)
    prompt = "Un mercado de Madrid, ilustración sencilla"
    bound = len(prompt.encode()) + 16
    managed = ManagedImageGenerationProvider([TokenBilledImages(reported)], invoker(), price_per_image=0.011,
                                             input_price_per_mtok=5.0)
    with active_scope(ExecutionScope(events=EventBus(), usage=UsageLedger(account=account), task_id="t1")):
        await managed.generate(ProviderImageRequest(prompt=prompt, width=8, height=8))
    [entry] = [e for e in store.entries(task_id="t1") if e.state == SpendState.SETTLED]
    assert entry.estimated_cost_microusd == round((0.011 + bound * 5.0 / 1e6) * 1e6)
    settled_tokens = reported if reported is not None else bound
    assert entry.final_cost_microusd == round((0.011 + settled_tokens * 5.0 / 1e6) * 1e6)
    assert entry.final_cost_microusd <= entry.estimated_cost_microusd and entry.cost_source == "priced"


# --- B3: dated model snapshots -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("requested", "reported", "priced_as"), [
    ("gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"),
    ("gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo"),
    ("gpt-4o-mini", None, "gpt-4o-mini"),
    ("gpt-4o", "gpt-4o-mini-2024-07-18", None),  # another model, not a snapshot of the requested one
    ("gpt-4o-mini", "gpt-4o-mini-preview", None),
    ("gpt-4o-mini", "gpt-4o", None),
])
def test_a_reply_is_priced_as_the_requested_model_or_not_at_all(requested, reported, priced_as) -> None:
    assert pricing_model(requested, reported) == priced_as


class Snapshotting(LLMProvider):
    """OpenAI's behaviour: the reply names a dated snapshot of the requested model, with its real usage."""
    name = "fake-openai"
    requires_network = False

    def __init__(self, reply_model: str) -> None:
        self.reply_model = reply_model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="Hola", provider=self.name, model=self.reply_model,
                           usage=TokenUsage(input_tokens=120, output_tokens=30))


def priced_cost(model: str, usage: TokenUsage) -> float | None:
    if model != "gpt-4o-mini":  # the only priced model, as LLM_MODEL + LLM_*_PRICE_PER_MTOK price one
        return None
    return (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000


@pytest.mark.parametrize(("reply_model", "source"), [("gpt-4o-mini-2024-07-18", "priced"),
                                                     ("gpt-4o-mini", "priced"),
                                                     ("gpt-5-unknown", "estimate")])
async def test_a_dated_snapshot_reply_is_reconciled_from_its_reported_tokens(store, reply_model, source) -> None:  # noqa: F811
    account = task_account(store, capped(1.0), task_id="t1", learner_id=None, budget=None, fence=None)
    llm = ManagedLLMProvider(Snapshotting(reply_model), invoker(), cost=priced_cost)
    request = LLMRequest(model="gpt-4o-mini", system="Eres un profesor.", agent_id="teacher",
                         messages=[LLMMessage(role="user", content="Hola")], max_output_tokens=400)
    with active_scope(ExecutionScope(events=EventBus(), usage=UsageLedger(account=account), task_id="t1")):
        await llm.generate(request)
    [entry] = [e for e in store.entries(task_id="t1") if e.state == SpendState.SETTLED]
    reserved = priced_cost("gpt-4o-mini", TokenUsage(input_tokens=input_token_bound(request), output_tokens=400))
    assert entry.estimated_cost_microusd == round(reserved * 1e6)
    assert entry.cost_source == source
    if source == "priced":  # reconciled: the reported tokens at the configured price, under the reservation
        assert entry.final_cost_microusd == round(priced_cost("gpt-4o-mini", TokenUsage(input_tokens=120,
                                                                                         output_tokens=30)) * 1e6)
        assert entry.final_cost_microusd < entry.estimated_cost_microusd
    else:  # an unknown model's usage is not priced: the upper-bound reservation stands (fail closed)
        assert entry.final_cost_microusd == entry.estimated_cost_microusd
