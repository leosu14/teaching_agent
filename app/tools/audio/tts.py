"""TTS tools: the voice catalog and provider-independent speech synthesis with recorded usage."""

from __future__ import annotations

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.providers.tts.base import ProviderSpeechRequest, TTSProvider, TTSProviderError
from app.schemas.audio import TTSRequest, TTSResult, VoiceCatalog, VoiceQuery, voices_for
from app.schemas.common import RetryPolicy
from app.tools.base import Tool, ToolError, ToolTransientError


class VoiceCatalogTool(Tool[VoiceQuery, VoiceCatalog]):
    name = "tts.voices"
    description = "List the configured TTS provider's voices, optionally only those that speak a language."
    input_model = VoiceQuery
    output_model = VoiceCatalog

    def __init__(self, provider: TTSProvider) -> None:
        self._provider = provider

    async def run(self, data: VoiceQuery, scope: ExecutionScope) -> VoiceCatalog:
        try:
            voices = await self._provider.voices()
        except (TTSProviderError, ConnectionError, OSError) as exc:
            raise ToolTransientError(f"TTS provider '{self._provider.name}' voice list failed: {exc}") from exc
        return VoiceCatalog(provider=self._provider.name,
                            voices=voices_for(data.language, voices) if data.language else voices)


class TTSTool(Tool[TTSRequest, TTSResult]):
    name = "tts.synthesize"
    description = "Synthesize speech with the configured TTS provider and store the audio content-addressed; the " \
                  "result records format, declared duration, sample rate, channels, provider, model and usage."
    input_model = TTSRequest
    output_model = TTSResult
    permissions = frozenset({"media:generate", "artifact:write"})
    timeout_seconds = 120.0
    retry = RetryPolicy(max_attempts=2, backoff_seconds=0.1)

    def __init__(self, provider: TTSProvider, artifacts: ArtifactService) -> None:
        self._provider = provider
        self._artifacts = artifacts

    async def run(self, data: TTSRequest, scope: ExecutionScope) -> TTSResult:
        p = self._provider
        if data.output_format not in p.formats:
            raise ToolError(f"TTS provider '{p.name}' cannot produce {data.output_format} "
                            f"(supported: {sorted(p.formats)})")
        ignored = [name for name, given, supported in (
            ("speaking_rate", data.speaking_rate != 1.0, p.supports_speaking_rate),
            ("pitch", data.pitch is not None, p.supports_pitch),
            ("sample_rate", data.sample_rate is not None, p.supports_sample_rate),
        ) if given and not supported]
        request = ProviderSpeechRequest(
            text=data.text, language=data.language, voice_id=data.voice, format=data.output_format,
            speaking_rate=data.speaking_rate if p.supports_speaking_rate else None,
            pitch=data.pitch if p.supports_pitch else None,
            sample_rate=data.sample_rate if p.supports_sample_rate else None,
        )
        try:
            speech = await p.synthesize(request)
        except (TTSProviderError, ConnectionError, OSError) as exc:
            transient = getattr(exc, "transient", True)
            raise (ToolTransientError if transient else ToolError)(f"TTS provider '{p.name}' failed: {exc}") from exc
        if not speech.content:
            raise ToolError(f"TTS provider '{p.name}' returned no audio")
        obj = self._artifacts.put_object(speech.content, speech.media_type)
        usage = speech.usage
        units = {"requests": usage.requests, "characters": usage.characters}
        units |= {k: v for k, v in (("tokens", usage.tokens), ("seconds", usage.seconds)) if v is not None}
        scope.usage.record_service(service=f"tts:{p.name}", results=1, cost_usd=usage.cost_usd,
                                   estimated_cost_usd=usage.estimated_cost_usd, units=units)
        return TTSResult(
            audio=obj, duration=speech.duration, sample_rate=speech.sample_rate, channels=speech.channels,
            format=speech.format, media_type=speech.media_type, provider=p.name, model=speech.model,
            voice=data.voice, language=data.language, input_hash=data.fingerprint(),
            provider_metadata=speech.metadata, usage=usage, ignored_parameters=ignored,
        )
