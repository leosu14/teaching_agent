"""Audio slice units: schemas, the mock TTS provider and its WAV output, the TTS tool, audio validation, AUDIO_ASSET
creation and deduplication, the audio plan validator, the timing resolver and the presentation timeline."""

from __future__ import annotations

import hashlib
import io
import wave

import pytest
from pydantic import ValidationError

from app.artifacts.service import ArtifactService
from app.providers.tts.base import ProviderSpeechRequest, SynthesizedSpeech, TTSProvider, TTSProviderError
from app.providers.tts.mock import MockTTSProvider
from app.schemas.artifact import ArtifactDraft, ArtifactType, StoredObject
from app.schemas.audio import (
    AudioAssetLookup,
    AudioAssetMetadata,
    AudioAssetRef,
    AudioAssetRequest,
    AudioPlan,
    AudioPlanValidationRequest,
    AudioSegment,
    AudioSourceType,
    AudioValidationRequest,
    NarrationResult,
    PresentationTimeline,
    SlideTiming,
    TimelineRequest,
    TTSRequest,
    TTSResult,
    TTSUsage,
    Voice,
    concise,
    language_matches,
    normalize_language,
    voices_for,
)
from app.schemas.common import CostSummary
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import SqlArtifactRepository
from app.tools.audio.assets import AudioAssetLookupTool, AudioAssetRejected, AudioAssetTool
from app.tools.audio.timeline import PresentationTimelineTool, resolve_timing
from app.tools.audio.tts import TTSTool, VoiceCatalogTool
from app.tools.audio.validation import AudioPlanInvalid, AudioPlanValidationTool, AudioPlanValidator, AudioValidator
from app.tools.base import ToolCaller, ToolError, ToolTransientError
from app.tools.manager import ToolManager
from app.tools.registry import ToolRegistry
from app.utils.audio import AudioProbeError, encode_wav, probe_audio
from tests.unit.helpers import scope

CALLER = ToolCaller(caller_id="test", allowed_tools=frozenset({
    "tts.voices", "tts.synthesize", "audio.create_asset", "audio.find_asset", "audio_plan.validate", "audio.timeline"}),
    permissions=frozenset({"media:generate", "artifact:write", "artifact:read"}))


@pytest.fixture
def artifacts(tmp_path):
    sessions = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(tmp_path / "objects"))
    dispose(sessions)


def manager_for(artifacts: ArtifactService, provider: TTSProvider | None = None) -> ToolManager:
    provider = provider or MockTTSProvider()
    registry = ToolRegistry()
    for tool in (VoiceCatalogTool(provider), TTSTool(provider, artifacts), AudioAssetTool(artifacts),
                 AudioAssetLookupTool(artifacts), AudioPlanValidationTool(provider), PresentationTimelineTool(artifacts)):
        registry.register(tool)
    return ToolManager(registry)


def segment(segment_id="s01_a1", slide_id="s01", order=1, text="Hola, bienvenidos.", **kw) -> AudioSegment:
    fields = dict(segment_id=segment_id, order=order, slide_id=slide_id, source_type="speaker_notes",
                  source_ref=f"{slide_id}.notes", text=text, language="es-ES", voice="mock-es-ES-1",
                  expected_duration=1.0)
    return AudioSegment(**{**fields, **kw})


def plan(*segments: AudioSegment, **kw) -> AudioPlan:
    fields = dict(audio_plan_id="ap_1", task_id="t1", deck_id="deck_1", language="es-ES", voice="mock-es-ES-1",
                  segments=list(segments))
    return AudioPlan(**{**fields, **kw})


def wav_bytes(seconds=0.5, rate=16000, channels=1) -> bytes:
    return encode_wav(b"\x10\x00" * int(seconds * rate) * channels, sample_rate=rate, channels=channels)


def stored(data: bytes, media_type="audio/wav") -> StoredObject:
    return StoredObject(uri="file:///nowhere", checksum=hashlib.sha256(data).hexdigest(), media_type=media_type,
                        size_bytes=len(data))


# --- schemas -----------------------------------------------------------------------------------


def test_language_tags_are_matched_without_hard_coding_a_language() -> None:
    assert normalize_language("zh-hans-cn") == "zh-Hans-CN" and normalize_language("es_es") == "es-ES"
    assert language_matches("en", "en-US") and language_matches("es-ES", "es-es")
    assert not language_matches("en-US", "en-GB") and not language_matches("es", "en-US")
    voices = [Voice(voice_id=v, language=lang, display_name=v, provider="p")
              for v, lang in (("b", "en-GB"), ("a", "en-US"), ("c", "zh-CN"))]
    assert [v.voice_id for v in voices_for("en-GB", voices)] == ["b"]
    assert [v.voice_id for v in voices_for("en", voices)] == ["b", "a"]  # the provider's order
    assert [v.voice_id for v in voices_for("en-US", voices)] == ["a"]
    assert [v.voice_id for v in voices_for("zh-CN", voices)] == ["c"]


def test_voice_schema_keeps_gender_optional_and_rejects_bad_tags() -> None:
    voice = Voice(voice_id="v1", language="es-ES", display_name="Lucía", provider="acme")
    assert voice.gender is None and voice.metadata == {}
    assert Voice(voice_id="v2", language="en-US", display_name="x", provider="acme", gender="female").gender == "female"
    with pytest.raises(ValidationError):
        Voice(voice_id="v3", language="not a tag", display_name="x", provider="acme")
    with pytest.raises(ValidationError):
        Voice(voice_id="v4", language="en", display_name="x", provider="acme", gender="robot")


def test_audio_segment_and_plan_schema() -> None:
    seg = segment(pitch=2.0, pause_before=0.3, pause_after=0.5)
    assert seg.start_time is None and seg.duration is None and seg.required  # timing is resolved later
    assert seg.source_type == AudioSourceType.SPEAKER_NOTES
    request = seg.tts_request("wav", 16000)
    assert (request.text, request.voice, request.language, request.pitch) == (seg.text, seg.voice, "es-ES", 2.0)
    for bad in ({"speaking_rate": 0}, {"pitch": 40}, {"pause_after": -1}, {"expected_duration": 0},
                {"source_type": "image"}, {"segment_id": "has space"}, {"language": "español"}):
        with pytest.raises(ValidationError):
            segment(**bad)
    p = plan(seg, segment("s02_a1", "s02", 2, "Adiós."))
    assert [s.segment_id for s in p.for_slide("s02")] == ["s02_a1"]
    assert p.expected_duration() == pytest.approx(0.3 + 1.0 + 0.5 + 1.0)
    with pytest.raises(ValidationError):
        AudioPlan(audio_plan_id="ap", task_id="t", deck_id="d", language="es-ES", voice="v", segments=[],
                  provider_specific={"x": 1})  # no provider details in a plan


def test_concise_keeps_whole_sentences() -> None:
    text = "One two three. Four five six. Seven eight nine ten."
    assert concise(text, 6) == "One two three. Four five six."
    assert concise("a b c d e f g", 3) == "a b c"


def test_fingerprint_changes_with_every_input() -> None:
    base = TTSRequest(text="Hola", language="es-ES", voice="v1")
    assert base.fingerprint() == TTSRequest(text="Hola", language="es-es", voice="v1").fingerprint()
    for change in ({"text": "Adiós"}, {"voice": "v2"}, {"speaking_rate": 1.2}, {"pitch": 1.0},
                   {"output_format": "mp3"}, {"sample_rate": 8000}, {"language": "es-MX"}):
        assert base.model_copy(update=change).fingerprint() != base.fingerprint(), change


# --- mock provider and WAV ---------------------------------------------------------------------


async def test_mock_provider_produces_real_deterministic_wav() -> None:
    provider = MockTTSProvider()
    request = ProviderSpeechRequest(text="Me gusta el fútbol.", language="es-ES", voice_id="mock-es-ES-1",
                                    format="wav", speaking_rate=1.0)
    first, second = await provider.synthesize(request), await provider.synthesize(request)
    assert first.content == second.content and first.content[:4] == b"RIFF"  # same request, same bytes
    with wave.open(io.BytesIO(first.content)) as wav:  # readable by the standard library parser
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        frames = wav.getnframes()
        samples = wav.readframes(frames)
    assert frames > 0 and any(samples)  # real samples, not silence padding
    assert first.duration == pytest.approx(frames / 16000) and first.sample_rate == 16000 and first.channels == 1
    assert first.usage == TTSUsage(requests=1, characters=len(request.text), seconds=round(first.duration, 6),
                                   cost_usd=0.0)
    assert first.media_type == "audio/wav" and first.model == "mock-tts-1"


async def test_mock_provider_bytes_follow_voice_text_rate_and_pitch() -> None:
    provider = MockTTSProvider()
    base = ProviderSpeechRequest(text="Me gusta el fútbol.", language="es-ES", voice_id="mock-es-ES-1", format="wav")
    variants = {
        "voice": base.model_copy(update={"voice_id": "mock-es-ES-2"}),
        "text": base.model_copy(update={"text": "Me gusta el tenis."}),
        "pitch": base.model_copy(update={"pitch": 3.0}),
    }
    reference = await provider.synthesize(base)
    for name, request in variants.items():
        assert (await provider.synthesize(request)).content != reference.content, name
    faster = await provider.synthesize(base.model_copy(update={"speaking_rate": 2.0}))
    assert faster.duration < reference.duration * 0.6
    low = await provider.synthesize(base.model_copy(update={"sample_rate": 8000}))
    assert low.sample_rate == 8000 and low.duration == pytest.approx(reference.duration, abs=0.01)


async def test_mock_provider_languages_and_errors() -> None:
    provider = MockTTSProvider()
    languages = {v.language for v in await provider.voices()}
    assert {"es-ES", "en-US", "zh-CN"} <= languages
    zh = await provider.synthesize(ProviderSpeechRequest(text="我 喜欢 足球。", language="zh-CN",
                                                         voice_id="mock-zh-CN-1", format="wav"))
    assert probe_audio(zh.content).duration > 0
    for request, message in (
        ({"voice_id": "nobody"}, "unknown voice"),
        ({"language": "fr-FR"}, "does not speak"),
        ({"format": "mp3"}, "not supported"),
        ({"sample_rate": 11025}, "sample rate"),
    ):
        with pytest.raises(TTSProviderError, match=message) as err:
            await provider.synthesize(ProviderSpeechRequest(**{
                "text": "Hola", "language": "es-ES", "voice_id": "mock-es-ES-1", "format": "wav", **request}))
        assert err.value.transient is False


# --- TTS tool --------------------------------------------------------------------------------


class PlainProvider(TTSProvider):
    """A provider without rate, pitch or sample-rate control that fails on demand."""

    name = "plain"
    formats = frozenset({"wav"})

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.requests: list[ProviderSpeechRequest] = []

    async def voices(self):
        return [Voice(voice_id="p1", language="en-US", display_name="P1", provider=self.name)]

    async def synthesize(self, request):
        self.requests.append(request)
        if self.fail:
            raise self.fail
        data = wav_bytes(0.25)
        return SynthesizedSpeech(content=data, format="wav", media_type="audio/wav", model="plain-1",
                                 sample_rate=16000, channels=1, duration=0.25,
                                 usage=TTSUsage(characters=len(request.text), tokens=7, estimated_cost_usd=0.002))


async def test_tts_tool_stores_audio_content_addressed_and_records_usage(artifacts) -> None:
    tools = manager_for(artifacts)
    sc, _ = scope("t1")
    request = TTSRequest(text="Hello and welcome.", language="en-US", voice="mock-en-US-1")
    first = await tools.call(CALLER, "tts.synthesize", request, sc)
    second = await tools.call(CALLER, "tts.synthesize", request, sc)
    assert isinstance(first, TTSResult)
    assert first.audio.checksum == second.audio.checksum and second.audio.reused and not first.audio.reused
    assert first.audio.key == f"objects/sha256/{first.audio.checksum[:2]}/{first.audio.checksum}.wav"
    assert first.input_hash == request.fingerprint() and first.provider == "mock" and first.model == "mock-tts-1"
    assert (first.format, first.media_type, first.sample_rate, first.channels) == ("wav", "audio/wav", 16000, 1)
    assert first.duration > 0 and first.usage.characters == len(request.text)
    line = sc.usage.summary.by_service["tts:mock"]
    assert line.calls == 2 and line.cost_usd == 0.0 and line.estimated_cost_usd is None
    assert line.units["characters"] == 2 * len(request.text) and line.units["seconds"] == pytest.approx(2 * first.duration, abs=1e-5)
    assert sc.usage.summary.llm_calls == 0  # speech is a service, not a model call


async def test_tts_tool_reports_ignored_parameters_and_provider_usage(artifacts) -> None:
    provider = PlainProvider()
    tools = manager_for(artifacts, provider)
    sc, _ = scope("t1")
    result = await tools.call(CALLER, "tts.synthesize", TTSRequest(
        text="Hi", language="en-US", voice="p1", speaking_rate=1.5, pitch=2.0, sample_rate=22050), sc)
    assert result.ignored_parameters == ["speaking_rate", "pitch", "sample_rate"]
    assert provider.requests[0].speaking_rate is None and provider.requests[0].pitch is None
    line = sc.usage.summary.by_service["tts:plain"]
    assert line.units["tokens"] == 7 and line.estimated_cost_usd == 0.002 and line.cost_usd is None
    assert sc.usage.summary.actual_cost_usd == 0  # an estimate is not an actual cost
    with pytest.raises(ToolError, match="cannot produce mp3"):
        await tools.call(CALLER, "tts.synthesize", TTSRequest(text="Hi", language="en-US", voice="p1",
                                                               output_format="mp3"), sc)


async def test_tts_tool_maps_provider_errors(artifacts) -> None:
    sc, _ = scope("t1")
    transient = PlainProvider(fail=TTSProviderError("rate limited"))
    with pytest.raises(ToolTransientError):
        await manager_for(artifacts, transient).call(CALLER, "tts.synthesize",
                                                     TTSRequest(text="Hi", language="en-US", voice="p1"), sc)
    assert len(transient.requests) == 2  # retried once
    fatal = PlainProvider(fail=TTSProviderError("bad voice", transient=False))
    with pytest.raises(ToolError, match="bad voice"):
        await manager_for(artifacts, fatal).call(CALLER, "tts.synthesize",
                                                 TTSRequest(text="Hi", language="en-US", voice="p1"), sc)
    assert len(fatal.requests) == 1


async def test_voice_catalog_tool_filters_by_language(artifacts) -> None:
    tools = manager_for(artifacts)
    sc, _ = scope("t1")
    catalog = await tools.call(CALLER, "tts.voices", {"language": "es-ES"}, sc)
    assert catalog.provider == "mock" and [v.voice_id for v in catalog.voices] == ["mock-es-ES-1", "mock-es-ES-2"]
    assert all(v.gender is None for v in catalog.voices)  # the mock exposes no gender, so none is invented
    everything = await tools.call(CALLER, "tts.voices", {}, sc)
    assert len(everything.voices) > len(catalog.voices)


# --- audio validation ------------------------------------------------------------------------


def validation_request(data: bytes, **kw) -> AudioValidationRequest:
    try:
        duration = probe_audio(data).duration
    except AudioProbeError:
        duration = 1.0
    fields = dict(object=stored(data), declared_format="wav", declared_duration=duration,
                  declared_sample_rate=16000, declared_channels=1)
    return AudioValidationRequest(**{**fields, **kw})


def codes(report) -> set[str]:
    return {e.code for e in report.errors}


def test_audio_validator_accepts_valid_wav_and_measures_it() -> None:
    data = wav_bytes(0.5)
    report = AudioValidator().validate(validation_request(data, expected_sample_rate=16000, expected_channels=1), data)
    assert report.valid, report.errors
    m = report.measured
    assert (m.format, m.media_type, m.sample_rate, m.channels, m.sample_width) == ("wav", "audio/wav", 16000, 1, 2)
    assert m.duration == pytest.approx(0.5) and m.frames == 8000 and m.checksum == hashlib.sha256(data).hexdigest()
    assert report.checks == ["content", "checksum", "media_type", "container", "duration", "sample_rate", "channels"]


def test_audio_validator_rejects_invalid_audio() -> None:
    v = AudioValidator()
    good = wav_bytes(0.5)
    assert codes(v.validate(validation_request(good), b"")) == {"empty_audio"}
    fake = b"this is not audio, only a .wav name" * 10  # arbitrary bytes posing as WAV
    assert "unreadable_audio" in codes(v.validate(validation_request(fake), fake))
    truncated = good[:-100]
    assert "unreadable_audio" in codes(v.validate(validation_request(truncated), truncated))
    silent = wav_bytes(0)
    assert "zero_duration" in codes(v.validate(validation_request(silent, declared_duration=0.0), silent))
    assert codes(v.validate(validation_request(good), good + b"x")) >= {"checksum_mismatch"}
    assert codes(v.validate(validation_request(good, declared_duration=2.0), good)) == {"duration_mismatch"}
    assert codes(v.validate(validation_request(good, expected_sample_rate=24000), good)) == {"sample_rate_mismatch"}
    assert codes(v.validate(validation_request(good, declared_channels=2), good)) == {"channel_mismatch"}
    stereo = wav_bytes(0.5, channels=2)
    assert codes(v.validate(validation_request(stereo, expected_channels=1), stereo)) == {"channel_mismatch"}
    assert "unsupported_media_type" in codes(v.validate(validation_request(good, object=stored(good, "text/plain")),
                                                        good))
    assert codes(v.validate(validation_request(good, declared_format="mp3"), good)) == {"format_mismatch"}
    mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 64  # a recognised container without a reader yet
    report = v.validate(validation_request(mp3, object=stored(mp3, "audio/mpeg"), declared_format="mp3"), mp3)
    assert "unreadable_audio" in codes(report) and "cannot be measured" in report.errors[-1].message


# --- AUDIO_ASSET -----------------------------------------------------------------------------


async def synthesize(tools, sc, seg: AudioSegment) -> TTSResult:
    return await tools.call(CALLER, "tts.synthesize", seg.tts_request(), sc)


async def test_audio_asset_records_measured_metadata(artifacts) -> None:
    tools = manager_for(artifacts)
    sc, events = scope("t1")
    seg = segment()
    tts = await synthesize(tools, sc, seg)
    art = await tools.call(CALLER, "audio.create_asset", AudioAssetRequest(
        name="audio_s01_a1", segment=seg, audio_plan_id="ap_1", tts=tts, expected_sample_rate=16000), sc)
    assert art.type == ArtifactType.AUDIO_ASSET and art.media_type == "audio/wav" and art.uri == tts.audio.uri
    meta = AudioAssetMetadata.model_validate(art.metadata)
    assert (meta.segment_id, meta.slide_id, meta.language, meta.voice, meta.provider, meta.model) == (
        "s01_a1", "s01", "es-ES", "mock-es-ES-1", "mock", "mock-tts-1")
    assert meta.duration == pytest.approx(tts.duration) and meta.sample_rate == 16000 and meta.channels == 1
    assert meta.checksum == art.content_hash == hashlib.sha256(artifacts.read(art.artifact_id)).hexdigest()
    assert meta.object_key == tts.audio.key and meta.input_hash == seg.tts_request().fingerprint()
    assert meta.validation.valid and meta.format == "wav" and meta.media_type == "audio/wav"
    assert [e.type for e in events if e.type == "artifact.created"] == ["artifact.created"]


async def test_invalid_audio_never_becomes_an_asset(artifacts) -> None:
    tools = manager_for(artifacts)
    sc, _ = scope("t1")
    seg = segment()
    tts = await synthesize(tools, sc, seg)
    lying = tts.model_copy(update={"duration": tts.duration + 5})  # declared duration does not match the bytes
    with pytest.raises(AudioAssetRejected) as err:
        await tools.call(CALLER, "audio.create_asset", AudioAssetRequest(
            name="audio_s01_a1", segment=seg, audio_plan_id="ap_1", tts=lying), sc)
    assert {e.code for e in err.value.report.errors} == {"duration_mismatch"}
    fake = artifacts.put_object(b"not a wav at all" * 8, "audio/wav")
    with pytest.raises(AudioAssetRejected):
        await tools.call(CALLER, "audio.create_asset", AudioAssetRequest(
            name="audio_s01_a1", segment=seg, audio_plan_id="ap_1", tts=tts.model_copy(update={"audio": fake})), sc)
    assert artifacts.list_for_task("t1") == []


async def test_identical_audio_is_stored_once_and_reused(artifacts, tmp_path) -> None:
    tools = manager_for(artifacts)
    sc, _ = scope("t1")
    a = segment("s01_a1", "s01", 1, "Practice")
    b = segment("s05_a1", "s05", 2, "Practice")  # identical text, same voice, another slide
    other_voice = segment("s06_a1", "s06", 3, "Practice", voice="mock-es-ES-2")
    other_text = segment("s07_a1", "s07", 4, "Practise")
    made = {}
    for seg in (a, b, other_voice, other_text, a):  # `a` twice: repeated execution
        tts = await synthesize(tools, sc, seg)
        made.setdefault(seg.segment_id, []).append(await tools.call(CALLER, "audio.create_asset", AudioAssetRequest(
            name=f"audio_{seg.segment_id}", segment=seg, audio_plan_id="ap_1", tts=tts), sc))
    first, again = made["s01_a1"]
    assert first.artifact_id == again.artifact_id and first.version == 1  # repeated execution reuses the asset
    assert made["s05_a1"][0].content_hash == first.content_hash  # identical bytes...
    assert made["s05_a1"][0].artifact_id != first.artifact_id  # ...one asset per segment, one object
    assert made["s06_a1"][0].content_hash != first.content_hash  # another voice is not deduplicated
    assert made["s07_a1"][0].content_hash != first.content_hash  # other content is not deduplicated
    files = sorted(p.name for p in (tmp_path / "objects").rglob("*.wav"))
    assert len(files) == 3 and len(artifacts.list_for_task("t1")) == 4


async def test_asset_lookup_reuses_only_matching_inputs_and_bytes(artifacts, tmp_path) -> None:
    tools = manager_for(artifacts)
    sc, _ = scope("t1")
    seg = segment()
    tts = await synthesize(tools, sc, seg)
    art = await tools.call(CALLER, "audio.create_asset", AudioAssetRequest(
        name="audio_s01_a1", segment=seg, audio_plan_id="ap_1", tts=tts), sc)
    found = await tools.call(CALLER, "audio.find_asset", AudioAssetLookup(name="audio_s01_a1",
                                                                         input_hash=tts.input_hash), sc)
    assert found.artifact.artifact_id == art.artifact_id
    changed = await tools.call(CALLER, "audio.find_asset", AudioAssetLookup(
        name="audio_s01_a1", input_hash=seg.model_copy(update={"text": "Otra"}).tts_request().fingerprint()), sc)
    assert changed.artifact is None and changed.reason == "inputs changed"
    assert (await tools.call(CALLER, "audio.find_asset", AudioAssetLookup(name="audio_x", input_hash="h"), sc)).artifact is None
    next((tmp_path / "objects").rglob("*.wav")).write_bytes(b"corrupted")
    tampered = await tools.call(CALLER, "audio.find_asset", AudioAssetLookup(name="audio_s01_a1",
                                                                            input_hash=tts.input_hash), sc)
    assert tampered.artifact is None and tampered.reason == "checksum mismatch"


# --- audio plan validation ---------------------------------------------------------------------

VOICES = [Voice(voice_id=v, language=lang, display_name=v, provider="mock")
          for v, lang in (("mock-es-ES-1", "es-ES"), ("mock-en-US-1", "en-US"), ("mock-zh-CN-1", "zh-CN"))]
SLIDES = ["s01", "s02", "s03"]


def check(p: AudioPlan | dict, voices=VOICES, **kw):
    raw = p if isinstance(p, dict) else p.model_dump(mode="json")
    return AudioPlanValidator().validate(AudioPlanValidationRequest(plan=raw, slide_ids=SLIDES, **kw), voices)


def test_plan_validator_accepts_a_valid_plan() -> None:
    report = check(plan(segment(), segment("s02_a1", "s02", 2, "Dos.")))
    assert report.valid and report.plan is not None and "voice" in report.checks and "timing" not in report.checks


@pytest.mark.parametrize("change, code", [
    (lambda s: [s[0], s[0].model_copy(update={"order": 2})], "duplicate_segment_id"),
    (lambda s: [s[0].model_copy(update={"slide_id": "s99"}), s[1]], "unknown_slide"),
    (lambda s: [s[0], s[1].model_copy(update={"order": 5})], "segment_order"),
    (lambda s: [s[1].model_copy(update={"order": 1}), s[0].model_copy(update={"order": 2})], "slide_order"),
    (lambda s: [s[0].model_copy(update={"text": "   "}), s[1]], "empty_text"),
    (lambda s: [s[0].model_copy(update={"text": "palabra " * 30}), s[1]], "too_long"),
    (lambda s: [s[0].model_copy(update={"language": "fr-FR"}), s[1]], "unsupported_language"),
    (lambda s: [s[0].model_copy(update={"voice": "mock-xx-1"}), s[1]], "unknown_voice"),
    (lambda s: [s[0].model_copy(update={"voice": "mock-en-US-1"}), s[1]], "voice_language_mismatch"),
    (lambda s: [s[0].model_copy(update={"text": "As cite_web_3 shows"}), s[1]], "spoken_metadata"),
    (lambda s: [s[0].model_copy(update={"text": "See https://example.org"}), s[1]], "spoken_metadata"),
])
def test_plan_validator_rejects(change, code) -> None:
    segs = [segment(), segment("s02_a1", "s02", 2, "Dos.")]
    report = check(plan(*change(segs)), unspoken_terms=["cite_web_3"], max_words_per_segment=20)
    assert not report.valid and code in {e.code for e in report.errors}, report.errors


def test_plan_validator_reports_schema_errors_and_empty_plans() -> None:
    report = check({"audio_plan_id": "ap", "segments": [{"text": "x"}]})
    assert not report.valid and {e.code for e in report.errors} == {"invalid_schema"}
    assert {e.code for e in check(plan()).errors} == {"no_segments"}


def timed(seg: AudioSegment, start, end, duration) -> AudioSegment:
    return seg.model_copy(update={"start_time": start, "end_time": end, "duration": duration})


@pytest.mark.parametrize("segments, code", [
    ([timed(segment(), 0, 2, 2), timed(segment("s02_a1", "s02", 2), 1.5, 3, 1.5)], "overlapping_timing"),
    ([timed(segment(), 0, -1, -1)], "negative_duration"),
    ([timed(segment(), -1, 1, 2)], "negative_duration"),
    ([timed(segment(), 0, 2, 1)], "timing_mismatch"),
    ([segment().model_copy(update={"start_time": 0.0})], "incomplete_timing"),
])
def test_plan_validator_checks_generated_timing(segments, code) -> None:
    report = check(plan(*segments), voices=None)
    assert "timing" in report.checks and code in {e.code for e in report.errors}


async def test_plan_validator_tool_enforces_and_emits(artifacts) -> None:
    tools = manager_for(artifacts)
    sc, events = scope("t1")
    good = AudioPlanValidationRequest(plan=plan(segment()).model_dump(mode="json"), slide_ids=SLIDES, enforce=True)
    report = await tools.call(CALLER, "audio_plan.validate", good, sc)
    assert report.valid and events[-2].type == "audio_plan.validated"
    bad = good.model_copy(update={"plan": plan(segment(voice="nobody")).model_dump(mode="json")})
    with pytest.raises(AudioPlanInvalid):
        await tools.call(CALLER, "audio_plan.validate", bad, sc)
    failed = [e for e in events if e.type == "audio.failed"]
    assert failed[-1].data["stage"] == "validation" and failed[-1].data["errors"][0]["code"] == "unknown_voice"


# --- timing --------------------------------------------------------------------------------------


def test_timing_resolver_uses_measured_durations_in_slide_order() -> None:
    p = plan(segment("s01_a1", "s01", 1, expected_duration=9.0),
             segment("s02_a1", "s02", 2, pause_before=0.25, pause_after=0.5, expected_duration=9.0),
             segment("s02_a2", "s02", 3, expected_duration=9.0))
    measured = {"s01_a1": 4.2, "s02_a1": 3.1, "s02_a2": 1.0004}
    timed_plan, slides = resolve_timing(p, measured, ["s01", "s02", "s03"], silent_slide_seconds=2.0)
    spans = [(s.segment_id, s.start_time, s.end_time, s.duration) for s in timed_plan.segments]
    assert spans == [("s01_a1", 0.0, 4.2, 4.2), ("s02_a1", 4.45, 7.55, 3.1), ("s02_a2", 8.05, 9.05, 1.0)]
    assert [(s.slide_id, s.start_time, s.end_time, s.audio_segment_refs) for s in slides] == [
        ("s01", 0.0, 4.2, ["s01_a1"]), ("s02", 4.2, 9.05, ["s02_a1", "s02_a2"]), ("s03", 9.05, 11.05, [])]
    assert resolve_timing(p, measured, ["s01", "s02", "s03"], 2.0) == (timed_plan, slides)  # deterministic
    assert check(timed_plan, voices=None).valid  # no overlap, no negative durations


def test_timing_resolver_leaves_unvoiced_segments_untimed() -> None:
    p = plan(segment("s01_a1", "s01", 1), segment("s01_a2", "s01", 2, required=False))
    timed_plan, slides = resolve_timing(p, {"s01_a1": 1.5}, ["s01"])
    assert timed_plan.segments[1].start_time is None and slides[0].audio_segment_refs == ["s01_a1"]
    assert slides[0].duration == 1.5


def test_presentation_timeline_must_be_contiguous() -> None:
    slide = SlideTiming(slide_id="s01", order=1, start_time=0, end_time=2, duration=2)
    fields = dict(timeline_id="tl", presentation_artifact_id="art_p", deck_id="d", audio_plan_id="ap",
                  audio_plan_artifact_id="art_a", language="es-ES", voice="v", resolver="r")
    PresentationTimeline(**fields, duration=2, slides=[slide])
    gap = SlideTiming(slide_id="s02", order=2, start_time=3, end_time=4, duration=1)
    with pytest.raises(ValidationError, match="contiguous"):
        PresentationTimeline(**fields, duration=4, slides=[slide, gap])
    with pytest.raises(ValidationError, match="ordered"):
        PresentationTimeline(**fields, duration=2, slides=[slide.model_copy(update={"order": 2})])


async def test_timeline_tool_stores_the_timeline_linked_to_presentation_and_audio(artifacts) -> None:
    tools = manager_for(artifacts)
    sc, events = scope("t1")
    parents = artifacts.store_batch("t1", [
        ArtifactDraft(key="p", name="presentation", type=ArtifactType.PRESENTATION, media_type="application/json",
                      content="{}"),
        ArtifactDraft(key="a", name="audio_plan", type=ArtifactType.AUDIO_PLAN, media_type="application/json",
                      content="{}")], sc).by_key
    seg = segment()
    tts = await synthesize(tools, sc, seg)
    asset = await tools.call(CALLER, "audio.create_asset", AudioAssetRequest(
        name="audio_s01_a1", segment=seg, audio_plan_id="ap_1", tts=tts, parent_ids=[parents["a"]]), sc)
    ref = AudioAssetRef(segment_id=seg.segment_id, slide_id="s01", artifact_id=asset.artifact_id,
                        asset_id=asset.metadata["asset_id"], checksum=asset.content_hash, uri=asset.uri,
                        media_type="audio/wav", duration=asset.metadata["duration"])
    request = TimelineRequest(plan=plan(seg), narration=NarrationResult(status="complete", audio_plan_id="ap_1",
                                                                        assets=[ref]),
                              slide_ids=["s01", "s02"], presentation_artifact_id=parents["p"],
                              audio_plan_artifact_id=parents["a"], silent_slide_seconds=1.0)
    result = await tools.call(CALLER, "audio.timeline", request, sc)
    tl = result.timeline
    assert tl.presentation_artifact_id == parents["p"] and [s.slide_id for s in tl.slides] == ["s01", "s02"]
    assert tl.segments[0].audio_artifact_id == asset.artifact_id and tl.duration == pytest.approx(
        round(ref.duration, 3) + 1.0)
    assert result.artifact.type == ArtifactType.PRESENTATION_TIMELINE
    assert result.artifact.parent_ids == [parents["p"], parents["a"], asset.artifact_id]
    assert PresentationTimeline.model_validate_json(artifacts.read(result.artifact.artifact_id)) == tl
    assert result.plan.segments[0].duration == tl.segments[0].duration
    again = await tools.call(CALLER, "audio.timeline", request, sc)
    assert again.artifact.artifact_id == result.artifact.artifact_id  # identical timeline: reused
    assert [e.type for e in events if e.type in {"timeline.created", "audio.completed"}] == [
        "timeline.created", "audio.completed"] * 2


def test_cost_summary_keeps_estimates_apart_from_actual_cost() -> None:
    summary = CostSummary()
    summary.record_service(service="tts:x", results=1, units={"characters": 10, "seconds": 1.5},
                           estimated_cost_usd=0.01)
    summary.record_service(service="tts:x", results=1, units={"characters": 5}, cost_usd=0.02,
                           estimated_cost_usd=0.01)
    line = summary.by_service["tts:x"]
    assert line.units == {"characters": 15, "seconds": 1.5} and line.estimated_cost_usd == 0.02
    assert line.cost_usd == 0.02 and summary.actual_cost_usd == 0.02

