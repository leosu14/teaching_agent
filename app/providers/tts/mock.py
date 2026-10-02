"""Deterministic offline TTS: a real 16-bit PCM WAV file in which every word is a short tone and every gap silence.

Nothing is spoken, but the file is valid audio whose length follows the text and the speaking rate, whose tone
follows the voice and pitch, and whose bytes are identical for identical requests. Usage is characters and
seconds with zero cost, since no paid API is called.
"""

from __future__ import annotations

import hashlib
import math
import struct

from app.providers.core.errors import ProviderError
from app.providers.tts.base import ProviderSpeechRequest, SynthesizedSpeech, TTSProvider
from app.schemas.audio import AUDIO_MEDIA_TYPES, TTSUsage, Voice
from app.utils.audio import encode_wav

CHARACTERS_PER_SECOND = 15.0  # at speaking rate 1.0
SENTENCE_PAUSE = 0.25  # seconds after . ! ?
DEFAULT_SAMPLE_RATE = 16000
SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000})

# (voice id, language): the languages the tests and demos use, two voices where a test needs a choice.
VOICES = (
    ("mock-en-US-1", "en-US"), ("mock-en-US-2", "en-US"), ("mock-en-GB-1", "en-GB"),
    ("mock-es-ES-1", "es-ES"), ("mock-es-ES-2", "es-ES"), ("mock-es-MX-1", "es-MX"),
    ("mock-zh-CN-1", "zh-CN"), ("mock-fr-FR-1", "fr-FR"), ("mock-de-DE-1", "de-DE"),
)


def _digest(*parts: object) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()


class MockTTSProvider(TTSProvider):
    name = "mock"
    model = "mock-tts-1"
    formats = frozenset({"wav"})
    supports_speaking_rate = True
    supports_pitch = True
    supports_sample_rate = True

    def __init__(self) -> None:
        self._voices = [Voice(voice_id=vid, language=lang, display_name=f"Mock {lang} voice {vid[-1]}",
                              provider=self.name, metadata={"engine": "tone"}) for vid, lang in VOICES]
        self.calls = 0

    async def voices(self) -> list[Voice]:
        return list(self._voices)

    async def synthesize(self, request: ProviderSpeechRequest) -> SynthesizedSpeech:
        self.calls += 1
        voice = next((v for v in self._voices if v.voice_id == request.voice_id), None)
        if voice is None:
            raise ProviderError(f"unknown voice '{request.voice_id}'", transient=False)
        if not voice.supports(request.language):
            raise ProviderError(f"voice '{voice.voice_id}' does not speak {request.language}", transient=False)
        if request.format not in self.formats:
            raise ProviderError(f"format '{request.format}' is not supported", transient=False)
        rate = request.sample_rate or DEFAULT_SAMPLE_RATE
        if rate not in SAMPLE_RATES:
            raise ProviderError(f"sample rate {rate} is not supported", transient=False)
        speed = request.speaking_rate or 1.0
        base = 110 + _digest(voice.voice_id)[0] % 150  # 110..259 Hz per voice
        frequency = base * 2 ** ((request.pitch or 0.0) / 12)
        pcm = b"".join(self._word(word, i, rate, speed, frequency, voice.voice_id)
                       for i, word in enumerate(request.text.split()))
        frames = len(pcm) // 2
        return SynthesizedSpeech(
            content=encode_wav(pcm, sample_rate=rate), format="wav", media_type=AUDIO_MEDIA_TYPES["wav"],
            model=self.model, sample_rate=rate, channels=1, duration=frames / rate,
            usage=TTSUsage(requests=1, characters=len(request.text), seconds=round(frames / rate, 6), cost_usd=0.0),
            metadata={"engine": "tone", "base_frequency_hz": round(frequency, 3)},
        )

    @staticmethod
    def _word(word: str, index: int, rate: int, speed: float, frequency: float, voice_id: str) -> bytes:
        seconds = (len(word) + 1) / (CHARACTERS_PER_SECOND * speed)
        if word[-1] in ".!?":
            seconds += SENTENCE_PAUSE / speed
        frames = max(1, round(seconds * rate))
        tone = int(frames * 0.8)
        period = max(2, round(rate / frequency))
        amplitude = 6000 + _digest(voice_id, index, word)[0] * 40  # 6000..16200 of 32767
        cycle = b"".join(struct.pack("<h", round(amplitude * math.sin(2 * math.pi * k / period)))
                         for k in range(period))
        voiced = (cycle * (tone // period + 1))[: tone * 2]
        return voiced + b"\x00\x00" * (frames - tone)
