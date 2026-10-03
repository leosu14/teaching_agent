"""Managed providers: the capability interfaces as tools and the router see them.

Each wraps the configured provider (or chain: primary, then the explicitly configured fallback) and runs every call
through the ProviderInvoker (offline guard, request id, timeout, bounded retry, rate limit, events, usage). They
implement the same interfaces as the providers they wrap, so nothing above the provider layer changes.

Fallback moves to the next provider only on a transient failure that survived the provider-level retries, and it
is always recorded as a provider.fallback event. Results carry the provider that actually produced them.
"""

from __future__ import annotations

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

P = TypeVar("P", bound=Provider)
T = TypeVar("T")
CostFn = Callable[[str, TokenUsage], float | None]  # (model, usage) -> estimated cost in USD, None when unpriced


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
                  units: dict[str, float] | None = None) -> tuple[P, T]:
        failure: ProviderError | None = None
        for index, provider in enumerate(self.providers):
            if failure is not None:
                self.invoker.fallback(self.capability, operation, from_provider=self.providers[index - 1].name,
                                      to_provider=provider.name, error=failure)
            try:
                result = await self.invoker.call(
                    provider, self.capability, operation, lambda p=provider: fn(p),
                    usage=lambda r, rid, p=provider: usage(p, r, rid), model=model, units=units)
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
            cost = self._cost(response.model, response.usage) if self._cost else None
            return ProviderUsage(
                provider=self.name, capability=Capability.LLM, operation="generate", model=response.model,
                request_id=request_id, vendor_request_id=response.vendor_request_id,
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
                cached_input_tokens=response.usage.cached_input_tokens, estimated_cost=cost, currency=_currency(cost),
            )

        return await self._invoker.call(self.inner, Capability.LLM, "generate", lambda: self.inner.generate(request),
                                        usage=usage, model=request.model)


class ManagedTTSProvider(_Delegating, TTSProvider):
    def __init__(self, chain: Sequence[TTSProvider], invoker: ProviderInvoker) -> None:
        self._chain = _Chain(chain, invoker, Capability.TTS)
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
            return ProviderUsage(
                provider=p.name, capability=Capability.TTS, operation="synthesize", model=speech.model,
                request_id=request_id, vendor_request_id=speech.metadata.get("vendor_request_id"),
                characters=u.characters, audio_seconds=u.seconds, input_tokens=u.tokens,
                estimated_cost=u.estimated_cost_usd, actual_cost=u.cost_usd,
                currency=_currency(u.estimated_cost_usd, u.cost_usd),
            )

        provider, speech = await self._chain.run("synthesize", lambda p: p.synthesize(request), usage,
                                                 units={"characters": len(request.text)})
        return speech.model_copy(update={"provider": provider.name})


class ManagedImageGenerationProvider(_Delegating, ImageGenerationProvider):
    def __init__(self, chain: Sequence[ImageGenerationProvider], invoker: ProviderInvoker) -> None:
        self._chain = _Chain(chain, invoker, Capability.IMAGE)
        primary = self._primary = chain[0]
        self.name = primary.name
        self.requires_network = primary.requires_network
        self.supports_negative_prompt = primary.supports_negative_prompt
        self.supports_seed = primary.supports_seed
        self.supports_style = primary.supports_style

    @property
    def providers(self) -> list[ImageGenerationProvider]:
        return list(self._chain.providers)

    async def generate(self, request: ProviderImageRequest) -> GeneratedImage:
        def usage(p: ImageGenerationProvider, image: GeneratedImage, request_id: str) -> ProviderUsage:
            return ProviderUsage(
                provider=p.name, capability=Capability.IMAGE, operation="generate", model=image.model,
                request_id=request_id, vendor_request_id=image.metadata.get("vendor_request_id"),
                image_count=image.usage.images, actual_cost=image.usage.cost_usd,
                currency=_currency(image.usage.cost_usd),
            )

        provider, image = await self._chain.run("generate", lambda p: p.generate(request), usage)
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
    def __init__(self, chain: Sequence[SearchProvider], invoker: ProviderInvoker) -> None:
        self._chain = _Chain(chain, invoker, Capability.SEARCH)
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
                                 actual_cost=page.usage.cost_usd, currency=_currency(page.usage.cost_usd))

        provider, page = await self._chain.run("search", lambda p: p.search(request), usage)
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
                                       model=self.model, units={"seconds": request.duration, "cost_usd": cost})
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
