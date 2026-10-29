"""Issue #28: `ArtifactView` serves artifact metadata through the per-type allowlist `PUBLIC_METADATA` only.

The metadata below has the shape the pipeline really stores (an AUDIO_ASSET from `audio.create_asset`, an IMAGE_ASSET
from a searched image), with a canary answer in every field a model or an image provider writes."""

from __future__ import annotations

import json

import pytest

from app.schemas import artifact as artifact_schema
from app.schemas.artifact import PUBLIC_METADATA, Artifact, ArtifactType, ArtifactView, public_metadata, server_only

CANARY = "quillvexdrum"


def audio_metadata(source_type: str = "answer_explanation") -> dict:
    return {
        "asset_id": "aud_0123456789abcdef", "segment_id": f"s07_{CANARY}_a1", "slide_id": f"s07_{CANARY}",
        "audio_plan_id": "ap_4b52e2ee55201678", "source_type": source_type, "source_ref": f"ex.{CANARY}",
        "text": f"{CANARY}. Compare with the lesson examples.", "input_hash": "f" * 64, "duration": 3.3,
        "format": "wav", "media_type": "audio/wav", "sample_rate": 16000, "channels": 1, "language": "en",
        "voice": "mock-en-US-1", "provider": "mock", "model": "mock-tts-1", "checksum": "c" * 64, "size_bytes": 105644,
        "object_key": "objects/sha256/cc/" + "c" * 64 + ".wav",
        "usage": {"requests": 1, "characters": 41, "tokens": None, "seconds": 3.3, "estimated_cost_usd": None,
                  "cost_usd": 0.0},
        "validation": {"valid": True, "errors": [], "checks": ["format"], "measured": {"duration": 3.3, "frames": 52800,
                                                                                    "checksum": "c" * 64}},
        "transcript": CANARY,  # a field added later: not on the allowlist, so not served
    }


def image_metadata() -> dict:
    return {
        "asset_id": "img_0123456789abcdef", "visual_id": f"v_{CANARY}", "plan_id": "vp_0123456789abcdef",
        "lesson_section_id": f"sec_{CANARY}", "visual_type": "photo", "purpose": f"shows {CANARY}",
        "description": f"a match where they said {CANARY}", "origin": "search", "width": 640, "height": 480,
        "format": "png",
        "object": {"uri": "file:///srv/o.png", "checksum": "d" * 64, "key": "objects/sha256/dd/d.png",
                   "media_type": "image/png", "size_bytes": 9, "reused": False},
        "attribution": {"kind": "search", "provider": "mock", "title": f"The answer is {CANARY}",
                        "creator": CANARY, "attribution_text": f"{CANARY} by {CANARY}", "source_url": "https://x"},
        "selection": {"selector": "s", "rank": 1, "rejected": [{"image_id": "i", "stage": "title",
                                                                 "reason": f"title states {CANARY}"}]},
    }


def view(type_: ArtifactType, metadata: dict) -> dict:
    a = Artifact(artifact_id="art_1", task_id="t", type=type_, name="n", uri="file:///srv/a", media_type="audio/wav",
                 content_hash="h" * 64, size_bytes=1, version=1, provider="mock", metadata=metadata)
    return ArtifactView.of(a).model_dump(mode="json")["metadata"]


def test_every_artifact_type_has_an_explicit_allowlist_entry() -> None:
    assert set(PUBLIC_METADATA) == set(ArtifactType)


def test_no_allowlisted_key_is_a_storage_location_digest_or_length() -> None:
    for type_, keys in PUBLIC_METADATA.items():
        assert not {k for k in keys if server_only(k)}, type_


@pytest.mark.parametrize("source_type", ["answer_explanation", "exercise_instructions", "speaker_notes"])
def test_audio_asset_serves_no_narration_text_length_or_model_written_id(source_type) -> None:
    served = view(ArtifactType.AUDIO_ASSET, audio_metadata(source_type))
    assert CANARY not in json.dumps(served)
    assert not {"text", "transcript", "duration", "usage", "source_ref", "segment_id", "slide_id", "asset_id",
                "validation", "input_hash", "checksum", "size_bytes", "object_key"} & set(served)
    assert served == {"audio_plan_id": "ap_4b52e2ee55201678", "source_type": source_type, "format": "wav",
                      "media_type": "audio/wav", "sample_rate": 16000, "channels": 1, "language": "en",
                      "voice": "mock-en-US-1", "provider": "mock", "model": "mock-tts-1"}


def test_image_asset_serves_no_provider_title_credit_selection_or_planned_text() -> None:
    served = view(ArtifactType.IMAGE_ASSET, image_metadata())
    assert CANARY not in json.dumps(served)
    assert served == {"asset_id": "img_0123456789abcdef", "plan_id": "vp_0123456789abcdef", "visual_type": "photo",
                      "origin": "search", "width": 640, "height": 480, "format": "png"}


def test_an_allowlisted_key_with_a_nested_value_is_not_served() -> None:
    """Only flat values are served: a dict or a list of dicts under an allowlisted key could carry anything."""
    served = view(ArtifactType.LEARNER_EVALUATION, {"score": 0.5, "action": {"text": CANARY},
                                                    "remaining_gaps": [{"answer": CANARY}]})
    assert served == {"score": 0.5}


def test_a_type_without_an_entry_serves_no_metadata(monkeypatch) -> None:
    monkeypatch.setattr(artifact_schema, "PUBLIC_METADATA",
                        {t: k for t, k in PUBLIC_METADATA.items() if t != ArtifactType.LEARNER_EVALUATION})
    assert view(ArtifactType.LEARNER_EVALUATION, {"score": 0.5, "remaining_gaps": ["c"]}) == {}
    assert public_metadata(ArtifactType.LEARNER_EVALUATION, {"score": 0.5}) == {}


def test_the_internal_artifact_keeps_its_metadata() -> None:
    metadata = audio_metadata()
    a = Artifact(artifact_id="art_1", task_id="t", type=ArtifactType.AUDIO_ASSET, name="n", uri="file:///srv/a",
                 media_type="audio/wav", content_hash="h" * 64, size_bytes=1, version=1, provider="mock",
                 metadata=metadata)
    ArtifactView.of(a)
    assert a.metadata == audio_metadata() and a.metadata["text"].startswith(CANARY)
