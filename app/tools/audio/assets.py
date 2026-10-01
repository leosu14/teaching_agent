"""AUDIO_ASSET creation and lookup. The asset tool re-validates the audio bytes itself, so no invalid audio can
become an asset; identical bytes are one object in the store, however many assets point at it."""

from __future__ import annotations

import hashlib
import json

from app.artifacts.service import ArtifactService
from app.observability.scope import ExecutionScope
from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.audio import (
    AudioAssetLookup,
    AudioAssetLookupResult,
    AudioAssetMetadata,
    AudioAssetRequest,
    AudioValidationReport,
    AudioValidationRequest,
)
from app.tools.audio.validation import AudioValidator
from app.tools.base import Tool, ToolError


class AudioAssetRejected(ToolError):
    """The audio failed validation. `report` holds the structured errors."""

    def __init__(self, report: AudioValidationReport) -> None:
        super().__init__("audio failed validation: " + json.dumps(
            [e.model_dump(exclude_none=True) for e in report.errors]))
        self.report = report


class AudioAssetTool(Tool[AudioAssetRequest, Artifact]):
    name = "audio.create_asset"
    description = "Validate synthesized audio against its bytes and create an AUDIO_ASSET artifact pointing at the " \
                  "stored object; its metadata keeps the segment, slide, measured format and the TTS provenance."
    input_model = AudioAssetRequest
    output_model = Artifact
    permissions = frozenset({"artifact:write", "artifact:read"})

    def __init__(self, artifacts: ArtifactService, validator: AudioValidator | None = None) -> None:
        self._artifacts = artifacts
        self._validator = validator or AudioValidator()

    async def run(self, data: AudioAssetRequest, scope: ExecutionScope) -> Artifact:
        if scope.task_id is None:
            raise ToolError("audio assets can only be created inside a task")
        tts, seg = data.tts, data.segment
        try:
            content = self._artifacts.read_object(tts.audio.uri)
        except (OSError, ValueError) as exc:
            raise ToolError(f"cannot read audio object {tts.audio.uri}: {exc}") from exc
        report = self._validator.validate(AudioValidationRequest(
            object=tts.audio, declared_format=tts.format, declared_duration=tts.duration,
            declared_sample_rate=tts.sample_rate, declared_channels=tts.channels,
            expected_sample_rate=data.expected_sample_rate, expected_channels=data.expected_channels,
        ), content)
        if not report.valid:
            raise AudioAssetRejected(report)
        assert report.measured is not None
        m = report.measured
        metadata = AudioAssetMetadata(
            asset_id="aud_" + m.checksum[:16], segment_id=seg.segment_id, slide_id=seg.slide_id,
            audio_plan_id=data.audio_plan_id, source_type=seg.source_type, source_ref=seg.source_ref, text=seg.text,
            input_hash=tts.input_hash, duration=m.duration, format=m.format, media_type=m.media_type,
            sample_rate=m.sample_rate, channels=m.channels, language=tts.language, voice=tts.voice,
            provider=tts.provider, model=tts.model, checksum=m.checksum, size_bytes=m.size_bytes,
            object_key=tts.audio.key or "", usage=tts.usage, validation=report,
        )
        return self._artifacts.store_object(
            task_id=scope.task_id, name=data.name, type=ArtifactType.AUDIO_ASSET, obj=tts.audio,
            provider=tts.provider, parent_ids=data.parent_ids, metadata=metadata.model_dump(mode="json"), scope=scope,
        )


class AudioAssetLookupTool(Tool[AudioAssetLookup, AudioAssetLookupResult]):
    name = "audio.find_asset"
    description = "Find this task's AUDIO_ASSET of a segment that was made from the same TTS inputs and whose " \
                  "stored bytes still match its checksum, so it can be reused instead of synthesized again."
    input_model = AudioAssetLookup
    output_model = AudioAssetLookupResult
    permissions = frozenset({"artifact:read"})

    def __init__(self, artifacts: ArtifactService) -> None:
        self._artifacts = artifacts

    async def run(self, data: AudioAssetLookup, scope: ExecutionScope) -> AudioAssetLookupResult:
        if scope.task_id is None:
            raise ToolError("audio assets can only be looked up inside a task")
        artifact = self._artifacts.find(scope.task_id, data.name)
        if artifact is None or artifact.type != ArtifactType.AUDIO_ASSET:
            return AudioAssetLookupResult(reason="no asset")
        if artifact.metadata.get("input_hash") != data.input_hash:
            return AudioAssetLookupResult(reason="inputs changed")
        try:
            content = self._artifacts.read_object(artifact.uri)
        except (OSError, ValueError) as exc:
            return AudioAssetLookupResult(reason=f"object unreadable: {exc}")
        if hashlib.sha256(content).hexdigest() != artifact.content_hash:
            return AudioAssetLookupResult(reason="checksum mismatch")
        return AudioAssetLookupResult(artifact=artifact, reason="inputs and checksum match")
