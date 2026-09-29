"""Text-to-speech provider contract, including timing metadata for scene synchronisation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.common import Schema


class WordTiming(Schema):
    word: str
    start_ms: int
    end_ms: int


class SynthesizedAudio(Schema):
    content: bytes
    media_type: str
    provider: str
    duration_ms: int
    timings: list[WordTiming]


class TTSProvider(ABC):
    name: str

    @abstractmethod
    async def synthesize(self, text: str, voice: str, language: str) -> SynthesizedAudio: ...
