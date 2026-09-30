"""Deterministic TTS: silent WAV whose length and word timings follow the text."""

from __future__ import annotations

import io
import wave

from app.providers.tts.base import SynthesizedAudio, TTSProvider, WordTiming

MS_PER_WORD = 350
SAMPLE_RATE = 8000


class MockTTSProvider(TTSProvider):
    name = "mock"

    async def synthesize(self, text: str, voice: str, language: str) -> SynthesizedAudio:
        words = text.split()
        timings = [WordTiming(word=w, start_ms=i * MS_PER_WORD, end_ms=(i + 1) * MS_PER_WORD)
                   for i, w in enumerate(words)]
        duration = len(words) * MS_PER_WORD
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(1)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"\x80" * (SAMPLE_RATE * duration // 1000))
        return SynthesizedAudio(content=buf.getvalue(), media_type="audio/wav", provider=self.name,
                                duration_ms=duration, timings=timings)
