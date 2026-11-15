"""Managed providers: the capability interfaces as tools and the router see them.

Each wraps the configured provider (or chain: primary, then the explicitly configured fallback) and runs every call
through the ProviderInvoker (offline guard, request id, timeout, bounded retry, rate limit, events, usage). They
implement the same interfaces as the providers they wrap, so nothing above the provider layer changes.

Fallback moves to the next provider only on a transient failure that survived the provider-level retries, and it
is always recorded as a provider.fallback event. Results carry the provider that actually produced them.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, TypeVar

from app.providers.core.base import Provider
from app.providers.core.errors import ProviderError
from app.providers.core.invoker import ProviderInvoker
from app.providers.image.base import GeneratedImage, ImageGenerationProvider, ProviderImageRequest
from app.providers.image_search.base import (
    DownloadedImage,
    ImageSearchPage,
    ImageSearchProvider,
    ProviderImageSearchRequest,
)
from app.providers.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.providers.search.base import ProviderSearchRequest, SearchPage, SearchProvider
from app.providers.tts.base import ProviderSpeechRequest, SynthesizedSpeech, TTSProvider
from app.providers.video_generation.base import VideoGenerationProvider
from app.schemas.audio import Voice
from app.schemas.common import TokenUsage
from app.schemas.generative_video import (
    GeneratedVideoFile,
    ProviderVideoLimits,
    VideoGenerationJob,
    VideoGenerationRequest,
)
from app.schemas.providers import Capability, HealthStatus, ProviderUsage
from app.schemas.spend import Estimate

log = logging.getLogger(__name__)

P = TypeVar("P", bound=Provider)
T = TypeVar("T")
CostFn = Callable[[str, TokenUsage], float | None]  # (model, usage) -> estimated cost in USD, None when unpriced
# Conservative pre-call estimates, replaced by what the provider reports once it answers.
CHARS_PER_TOKEN = 3  # most tokenizers average 3-4 characters per token in European languages: over-estimate
SPEECH_CHARS_PER_SECOND = 12  # about 12-15 characters of speech per second: over-estimate the audio length


# A strict upper bound of a request's input tokens, for its cost reservation: a byte-level tokenizer never makes
# more tokens than the text has UTF-8 bytes, plus each chat message's framing tokens and the reply's priming.
TOKENS_PER_MESSAGE = 8
TOKENS_PER_REQUEST = 16


def input_token_bound(request: LLMRequest) -> int:
    """At least as many input tokens as the request can be billed for: the UTF-8 bytes of the system prompt, the
    messages and the response schema (sent with the request when native structured output is used), plus framing."""
    size = len(request.system.encode("utf-8")) + sum(len(m.content.encode("utf-8")) for m in request.messages)
    if request.response_schema is not None:
        size += len(json.dumps(request.response_schema, ensure_ascii=False).encode("utf-8"))
    return size + TOKENS_PER_MESSAGE * (len(request.messages) + 1) + TOKENS_PER_REQUEST


_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{4})$")  # OpenAI's dated snapshots: -2024-07-18, -0125


def pricing_model(requested: str, reported: str | None) -> str | None:
    """The model a response is priced as: the requested model (the routed target priced at startup) when the
    response names it or one of its dated snapshots (OpenAI answers `gpt-4o-mini` with `gpt-4o-mini-2024-07-18`).
    None when it names another model: its usage is not priced, so the attempt keeps its upper-bound reservation
    (fail closed, never a guess)."""
    if not reported or reported == requested:
        return requested
    match = _SNAPSHOT.search(reported)
    if match and reported[:match.start()] == requested:
        return requested
    return None


def llm_estimate(request: LLMRequest, cost: CostFn | None) -> Estimate:
    """What an LLM request is reserved at before it is sent. Tokens: estimated input plus the request's own output
    limit (for MAX_LLM_TOKENS). Cost: the configured price of an upper bound (`input_token_bound`, plus the output
    limit, which the provider never exceeds), so the settled cost of the request cannot exceed its reservation."""
    chars = len(request.system) + sum(len(m.content) for m in request.messages)
    input_tokens = -(-chars // CHARS_PER_TOKEN)
    bound = TokenUsage(input_tokens=input_token_bound(request), output_tokens=request.max_output_tokens)
    return Estimate(llm_tokens=input_tokens + request.max_output_tokens,
                    cost_usd=cost(request.model, bound) if cost else None)


def priced(units: float, price: float | None) -> float | None:
    """The configured price of `units` (characters, images, requests) in USD; None when no price is configured."""
    return None if price is None else round(units * price, 6)


def at_least_priced(reported: float | None, price: float | None) -> float | None:
    """The provider-reported cost to settle when a price is configured: kept only when it is at least the configured
    price (None otherwise, so the configured price is charged). A configured price is an upper bound by contract, so
    a lower reported cost (a mock's 0.0) never makes the run-wide cap charge less than that."""
    if price is None or reported is None:
        return reported
    return reported if reported >= price else None


def _currency(*costs: float | None) -> str | None:
    return "USD" if any(c is not None for c in costs) else None


class _Chain(Generic[P]):
    """Runs one operation on the first provider of the chain, falling back on transient failures."""

    def __init__(self, chain: Sequence[P], invoker: ProviderInvoker, capability: Capability) -> None:
        if not chain:
            raise ValueError("a provider chain needs at least one provider")
        self.providers = list(chain)
        self.invoker = invoker
        self.capability = capability

    async def run(self, operation: str, fn: Callable[[P], Awaitable[T]],
                  usage: Callable[[P, T, str], ProviderUsage], model: str | None = None,
                  estimate: Estimate | None = None, idempotency_key: str | None = None) -> tuple[P, T]:
        failure: ProviderError | None = None
        for index, provider in enumerate(self.providers):
            if failure is not None:
                self.invoker.fallback(self.capability, operation, from_provider=self.providers[index - 1].name,
                                      to_provider=provider.name, error=failure)
            try:
                result = await self.invoker.call(
                    provider, self.capability, operation, lambda p=provider: fn(p),
                    usage=lambda r, rid, p=provider: usage(p, r, rid), model=model, estimate=estimate,
                    idempotency_key=idempotency_key)
            except ProviderError as exc:
                if not exc.transient or index == len(self.providers) - 1:
                    raise
                failure = exc
                continue
            return provider, result
        raise AssertionError("unreachable")


class _Delegating:
    """Health, configuration and unknown attributes come from the primary provider."""

    _primary: Provider

    def __getattr__(self, item: str):  # e.g. a mock's call counters or fault injection, in tests
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._primary, item)

    def configuration(self) -> dict:
        return self._primary.configuration()

    async def probe(self) -> str:
        return await self._primary.probe()

    async def health_check(self, capability: Capability | None = None) -> HealthStatus:
        return await self._primary.health_check(capability)


class ManagedLLMProvider(_Delegating, LLMProvider):
    """One LLM provider under the invoker. Fallback between LLM targets is the router's (tier chains)."""

    def __init__(self, inner: LLMProvider, invoker: ProviderInvoker, *, cost: CostFn | None = None) -> None:
        self._primary = inner
        self.inner = inner
        self.name = inner.name
        self.structured_output = inner.structured_output
        self.requires_network = inner.requires_network
        self._invoker = invoker
        self._cost = cost

    @property
    def deadline_seconds(self) -> float:
        return self._invoker.policy(Capability.LLM).deadline_seconds()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        def usage(response: LLMResponse, request_id: str) -> ProviderUsage:
            priced_as = pricing_model(request.model, response.model)
            cost = self._cost(priced_as, response.usage) if self._cost and priced_as else None
            if priced_as is None:
                log.warning("LLM response names model %r, not %r or a dated snapshot of it: its usage is not "
                            "priced, the reservation stands", response.model, request.model)
            return ProviderUsage(
                provider=self.name, capability=Capability.LLM, operation="generate", model=response.model,
                request_id=request_id, vendor_request_id=response.vendor_request_id,
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
                cached_input_tokens=response.usage.cached_input_tokens, estimated_cost=cost, currency=_currency(cost),
            )

        return await self._invoker.call(self.inner, Capability.LLM, "generate", lambda: self.inner.generate(request),
                                        usage=usage, model=request.model, estimate=llm_estimate(request, self._cost))


class ManagedTTSProvider(_Delegating, TTSProvider):
    def __init__(self, chain: Sequence[TTSProvider], invoker: ProviderInvoker, *,
                 price_per_mchar: float | None = None) -> None:
        self._chain = _Chain(chain, invoker, Capability.TTS)
        self.price_per_char = None if price_per_mchar is None else price_per_mchar / 1_000_000
        primary = self._primary = chain[0]
        self.name = primary.name
        self.requires_network = primary.requires_network
        self.formats = primary.formats
        self.supports_speaking_rate = primary.supports_speaking_rate
        self.supports_pitch = primary.supports_pitch
        self.supports_sample_rate = primary.supports_sample_rate

    @property
    def providers(self) -> list[TTSProvider]:
        return list(self._chain.providers)

    async def voices(self) -> list[Voice]:
        def usage(p: TTSProvider, voices: list[Voice], request_id: str) -> ProviderUsage:
            return ProviderUsage(provider=p.name, capability=Capability.TTS, operation="voices",
                                 request_id=request_id, result_count=len(voices))

        _, voices = await self._chain.run("voices", lambda p: p.voices(), usage)
        return voices

    async def synthesize(self, request: ProviderSpeechRequest) -> SynthesizedSpeech:
        def usage(p: TTSProvider, speech: SynthesizedSpeech, request_id: str) -> ProviderUsage:
            u = speech.usage
            # the configured price of the characters billed (the provider's own estimate when none is configured)
            price = (priced(u.characters if u.characters is not None else len(request.text), self.price_per_char)
                     if self.price_per_char is not None else None)
            estimated = price if price is not None else u.estimated_cost_usd
            return ProviderUsage(
                provider=p.name, capability=Capability.TTS, operation="synthesize", model=speech.model,
                request_id=request_id, vendor_request_id=speech.metadata.get("vendor_request_id"),
                characters=u.characters, audio_seconds=u.seconds, input_tokens=u.tokens,
                estimated_cost=estimated, actual_cost=at_least_priced(u.cost_usd, price),
                currency=_currency(estimated, u.cost_usd),
            )

        chars = len(request.text)
        provider, speech = await self._chain.run("synthesize", lambda p: p.synthesize(request), usage,
                                                 estimate=Estimate(characters=chars,
                                                                   audio_seconds=chars / SPEECH_CHARS_PER_SECOND,
                                                                   cost_usd=priced(chars, self.price_per_char)))
        return speech.model_copy(update={"provider": provider.name})


class ManagedImageGenerationProvider(_Delegating, ImageGenerationProvider):
    """Priced per image at its pinned quality and size (`price_per_image`), plus, for a model that bills the prompt
    (GPT image models), its text input tokens (`input_price_per_mtok`): reserved at the prompt's UTF-8 bytes plus
    framing, an upper bound of its tokens, and settled at the input tokens the vendor reports when it does
    (app/config/billing.py)."""

    def __init__(self, chain: Sequence[ImageGenerationProvider], invoker: ProviderInvoker, *,
                 price_per_image: float | None = None, input_price_per_mtok: float | None = None) -> None:
        self._chain = _Chain(chain, invoker, Capability.IMAGE)
        self.price_per_image = price_per_image
        self.input_price_per_mtok = input_price_per_mtok
        primary = self._primary = chain[0]
        self.name = primary.name
        self.requires_network = primary.requires_network
        self.supports_negative_prompt = primary.supports_negative_prompt
        self.supports_seed = primary.supports_seed
        self.supports_style = primary.supports_style

    @property
    def providers(self) -> list[ImageGenerationProvider]:
        return list(self._chain.providers)

    def image_cost(self, images: int, prompt_tokens: int) -> float | None:
        cost = priced(max(images, 1), self.price_per_image)
        if cost is None or self.input_price_per_mtok is None:
            return cost
        return round(cost + prompt_tokens * self.input_price_per_mtok / 1_000_000, 6)

    async def generate(self, request: ProviderImageRequest) -> GeneratedImage:
        bound = len(request.prompt.encode("utf-8")) + TOKENS_PER_REQUEST

        def usage(p: ImageGenerationProvider, image: GeneratedImage, request_id: str) -> ProviderUsage:
            vendor = image.metadata.get("vendor_usage") or {}
            reported = vendor.get("input_tokens")
            tokens = reported if isinstance(reported, int) and reported >= 0 else bound
            price = self.image_cost(image.usage.images, tokens)
            return ProviderUsage(
                provider=p.name, capability=Capability.IMAGE, operation="generate", model=image.model,
                request_id=request_id, vendor_request_id=image.metadata.get("vendor_request_id"),
                image_count=image.usage.images, input_tokens=reported if tokens == reported else None,
                actual_cost=at_least_priced(image.usage.cost_usd, price), estimated_cost=price,
                currency=_currency(image.usage.cost_usd, self.price_per_image),
            )

        provider, image = await self._chain.run("generate", lambda p: p.generate(request), usage,
                                                estimate=Estimate(cost_usd=self.image_cost(1, bound)))
        return image.model_copy(update={"provider": provider.name})


class ManagedImageSearchProvider(_Delegating, ImageSearchProvider):
    """Image search has no fallback: a found image's id and download belong to the provider that found it."""

    def __init__(self, provider: ImageSearchProvider, invoker: ProviderInvoker) -> None:
        self._chain = _Chain([provider], invoker, Capability.IMAGE_SEARCH)
        self._primary = provider
        self.name = provider.name
        self.requires_network = provider.requires_network

    async def search(self, request: ProviderImageSearchRequest) -> ImageSearchPage:
        def usage(p: ImageSearchProvider, page: ImageSearchPage, request_id: str) -> ProviderUsage:
            return ProviderUsage(provider=p.name, capability=Capability.IMAGE_SEARCH, operation="search",
                                 request_id=request_id, result_count=len(page.hits), actual_cost=page.usage.cost_usd,
                                 currency=_currency(page.usage.cost_usd))

        _, page = await self._chain.run("search", lambda p: p.search(request), usage)
        return page

    async def download(self, provider_image_id: str, url: str) -> DownloadedImage:
        def usage(p: ImageSearchProvider, image: DownloadedImage, request_id: str) -> ProviderUsage:
            return ProviderUsage(provider=p.name, capability=Capability.IMAGE_SEARCH, operation="download",
                                 request_id=request_id, image_count=1)

        _, image = await self._chain.run("download", lambda p: p.download(provider_image_id, url), usage)
        return image


class ManagedSearchProvider(_Delegating, SearchProvider):
    def __init__(self, chain: Sequence[SearchProvider], invoker: ProviderInvoker, *,
                 price_per_request: float | None = None) -> None:
        self._chain = _Chain(chain, invoker, Capability.SEARCH)
        self.price_per_request = price_per_request
        primary = self._primary = chain[0]
        self.name = primary.name
        self.requires_network = primary.requires_network
        self.supports_domain_filter = primary.supports_domain_filter

    @property
    def providers(self) -> list[SearchProvider]:
        return list(self._chain.providers)

    async def search(self, request: ProviderSearchRequest) -> SearchPage:
        def usage(p: SearchProvider, page: SearchPage, request_id: str) -> ProviderUsage:
            return ProviderUsage(provider=p.name, capability=Capability.SEARCH, operation="search",
                                 request_id=request_id, result_count=len(page.hits),
                                 actual_cost=at_least_priced(page.usage.cost_usd, priced(1, self.price_per_request)),
                                 estimated_cost=priced(1, self.price_per_request),
                                 currency=_currency(page.usage.cost_usd, self.price_per_request))

        provider, page = await self._chain.run("search", lambda p: p.search(request), usage,
                                               estimate=Estimate(cost_usd=priced(1, self.price_per_request)))
        return page.model_copy(update={"provider": provider.name})


class ManagedVideoGenerationProvider(_Delegating, VideoGenerationProvider):
    """Video generation under the invoker. Submission may fall back to the next provider of the chain on a transient
    failure; a submitted job is then polled, downloaded and cancelled at the provider that owns it (job.provider).
    Only submission is billed: its known seconds (and, with a configured price, its cost) are admitted against the
    task's budget before the request is sent."""

    def __init__(self, chain: Sequence[VideoGenerationProvider], invoker: ProviderInvoker, *,
                 price_per_second: float | None = None) -> None:
        self._chain = _Chain(chain, invoker, Capability.VIDEO_GENERATION)
        self._invoker = invoker
        primary = self._primary = chain[0]
        self._by_name = {p.name: p for p in chain}
        self.name = primary.name
        self.model = primary.model
        self.requires_network = primary.requires_network
        self.supports_seed = primary.supports_seed
        self.supports_cancel = primary.supports_cancel
        self.supports_audio = primary.supports_audio
        self.durations = primary.durations
        self.max_duration = primary.max_duration
        self.aspect_ratios = primary.aspect_ratios
        self.price_per_second = price_per_second if primary.requires_network else 0.0

    @property
    def providers(self) -> list[VideoGenerationProvider]:
        return list(self._chain.providers)

    def limits(self) -> ProviderVideoLimits:
        return self._primary.limits()

    def estimated_cost(self, seconds: float) -> float | None:
        return None if self.price_per_second is None else round(seconds * self.price_per_second, 6)

    def _owner(self, job: VideoGenerationJob) -> VideoGenerationProvider:
        provider = self._by_name.get(job.provider)
        if provider is None:
            raise ProviderError(f"video job {job.job_id} belongs to provider '{job.provider}', which is not "
                                "configured any more", provider=job.provider, transient=False)
        return provider

    async def submit(self, request: VideoGenerationRequest, *, generation_key: str) -> VideoGenerationJob:
        cost = self.estimated_cost(request.duration)

        def usage(p: VideoGenerationProvider, job: VideoGenerationJob, request_id: str) -> ProviderUsage:
            return ProviderUsage(provider=p.name, capability=Capability.VIDEO_GENERATION, operation="submit",
                                 model=job.model, request_id=request_id, video_seconds=request.duration,
                                 estimated_cost=cost if p.name == self.name else None, currency=_currency(cost))

        _, job = await self._chain.run("submit", lambda p: p.submit(request, generation_key=generation_key), usage,
                                       model=self.model,
                                       estimate=Estimate(video_seconds=request.duration, cost_usd=cost),
                                       idempotency_key=generation_key)
        return job

    async def _on_owner(self, operation: str, job: VideoGenerationJob, fn) -> object:
        owner = self._owner(job)

        def usage(result, request_id: str) -> ProviderUsage:
            return ProviderUsage(provider=owner.name, capability=Capability.VIDEO_GENERATION, operation=operation,
                                 model=job.model, request_id=request_id)

        return await self._invoker.call(owner, Capability.VIDEO_GENERATION, operation, lambda: fn(owner),
                                        usage=usage, model=job.model)

    async def status(self, job: VideoGenerationJob) -> VideoGenerationJob:
        return await self._on_owner("status", job, lambda p: p.status(job))

    async def download(self, job: VideoGenerationJob) -> GeneratedVideoFile:
        return await self._on_owner("download", job, lambda p: p.download(job))

    async def cancel(self, job: VideoGenerationJob) -> VideoGenerationJob:
        return await self._on_owner("cancel", job, lambda p: p.cancel(job))
