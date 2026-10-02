"""TTS adapter for the OpenAI speech API (POST /audio/speech). Plain HTTP: no vendor SDK.

Audio is requested as raw PCM (16-bit little-endian mono at 24 kHz) and wrapped in a WAV header here, so the file
is a complete WAV whose duration is exact (OpenAI's own streamed WAV carries placeholder sizes). The declared
duration is measured from the samples; the audio validator measures the bytes again anyway.

OpenAI voices are multilingual: each voice is offered in every language listed in TTS_LANGUAGES, and the language
is passed as a speaking instruction to models that accept instructions.
"""

from __future__ import annotations

from typing import ClassVar

from app.providers.core.errors import ProviderInvalidRequest, ProviderResponseError
from app.providers.core.http import HttpClient
from app.providers.tts.base import ProviderSpeechRequest, SynthesizedSpeech, TTSProvider
from app.schemas.audio import AUDIO_MEDIA_TYPES, TTSUsage, Voice
from app.utils.audio import encode_wav

VOICES = ("alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse")
PCM_SAMPLE_RATE = 24000
SPEED_RANGE = (0.25, 4.0)
DEFAULT_MODEL = "gpt-4o-mini-tts"
INSTRUCTION_MODELS = ("gpt-4o",)  # model prefixes that accept `instructions`


class OpenAITTSProvider(TTSProvider):
    name = "openai"
    requires_network: ClassVar[bool] = True
    formats = frozenset({"wav"})
    supports_speaking_rate = True

    def __init__(self, http: HttpClient, *, model: str | None = None, languages: list[str] | None = None,
                 voices: tuple[str, ...] = VOICES) -> None:
        self._http = http
        self.model = model or DEFAULT_MODEL
        self._languages = languages or ["en-US"]
        self._voices = voices

    def configuration(self) -> dict:
        return {"base_url": self._http.base_url, "model": self.model, "languages": self._languages,
                "output": f"wav (pcm s16le {PCM_SAMPLE_RATE} Hz mono)"}

    async def probe(self) -> str:
        await self._http.request("GET", f"/models/{self.model}")  # authenticated, nothing synthesized
        return "network"

    async def voices(self) -> list[Voice]:
        return [Voice(voice_id=voice, language=language, display_name=f"OpenAI {voice} ({language})",
                      provider=self.name, metadata={"multilingual": True})
                for language in self._languages for voice in self._voices]

    async def synthesize(self, request: ProviderSpeechRequest) -> SynthesizedSpeech:
        if request.voice_id not in self._voices:
            raise ProviderInvalidRequest(f"unknown voice '{request.voice_id}'", provider=self.name)
        if request.format not in self.formats:
            raise ProviderInvalidRequest(f"format '{request.format}' is not supported", provider=self.name)
        speed = request.speaking_rate or 1.0
        if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise ProviderInvalidRequest(f"speaking rate {speed} outside {SPEED_RANGE}", provider=self.name)
        body = {"model": self.model, "input": request.text, "voice": request.voice_id, "response_format": "pcm",
                "speed": speed}
        if self.model.startswith(INSTRUCTION_MODELS):
            body["instructions"] = f"Speak in the language with BCP 47 tag {request.language}, clearly, " \
                                   "as a teacher narrating a lesson."
        response = await self._http.request("POST", "/audio/speech", json_body=body)
        pcm = response.content
        if not pcm or len(pcm) % 2:
            raise ProviderResponseError(f"{self.name}: speech response is not 16-bit PCM ({len(pcm)} bytes)",
                                        provider=self.name)
        duration = len(pcm) / 2 / PCM_SAMPLE_RATE
        return SynthesizedSpeech(
            content=encode_wav(pcm, sample_rate=PCM_SAMPLE_RATE), format="wav", media_type=AUDIO_MEDIA_TYPES["wav"],
            model=self.model, sample_rate=PCM_SAMPLE_RATE, channels=1, duration=duration,
            usage=TTSUsage(requests=1, characters=len(request.text), seconds=round(duration, 6)),
            metadata={"vendor_request_id": response.vendor_request_id, "native_format": "pcm_s16le_24000",
                      "language_instruction": "instructions" in body},
        )
