"""Text-to-speech provider contract. Adapters (OpenAI TTS, ElevenLabs, Azure Speech, Google Cloud TTS, ...) map their
API to these schemas and keep their SDKs inside their own module. A provider exposes its voice catalog, declares the
formats and optional parameters it supports, and reports usage; the TTS tool records unsupported parameters as
ignored and never trusts the declared duration (the audio validator measures the bytes)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from app.schemas.audio import TTSUsage, Voice
from app.schemas.common import Schema


class ProviderSpeechRequest(Schema):
    text: str = Field(min_length=1)
    language: str
    voice_id: str
    format: str
    speaking_rate: float | None = None  # None when the provider does not support it
    pitch: float | None = None
    sample_rate: int | None = None


class SynthesizedSpeech(Schema):
    content: bytes
    format: str
    media_type: str
    model: str
    sample_rate: int
    channels: int
    duration: float  # seconds, as the provider declares it
    usage: TTSUsage
    metadata: dict = Field(default_factory=dict)  # anything else the provider returned, verbatim


class TTSProviderError(Exception):
    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class TTSProvider(ABC):
    name: str
    formats: frozenset[str] = frozenset()
    supports_speaking_rate: bool = False
    supports_pitch: bool = False
    supports_sample_rate: bool = False

    @abstractmethod
    async def voices(self) -> list[Voice]: ...

    @abstractmethod
    async def synthesize(self, request: ProviderSpeechRequest) -> SynthesizedSpeech: ...
