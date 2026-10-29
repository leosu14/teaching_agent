"""Issue #28: learner-facing artifact listings must not serve answer-bearing metadata.

Every AUDIO_ASSET kept the narration `text` it speaks in its metadata, and `GET /tasks/{id}/artifacts` served it:
the `answer_explanation` segments read the exercise answers aloud, so the listing gave the learner the answers (and
TTS `usage.characters` and the clip `duration` gave their lengths). Artifact metadata is now served through a
per-type allowlist (`PUBLIC_METADATA`). These tests run a real lesson whose first exercise answer is a canary and check
both listings, the learning-cycle child task, the content endpoint, and that nothing off the allowlist is served."""

from __future__ import annotations

import json

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders, make_teacher
from app.schemas.artifact import PUBLIC_METADATA, ArtifactType
from app.services.container import build_container
from tests.auth import TestAuth, client_for
from tests.conftest import LEARNER, run_lesson
from tests.learning_cycle_fixtures import LEARNER as CYCLE_LEARNER

CANARY = "quillvexdrum"
INTRUDER = "intruder-learner"
NOT_FOUND = {"error": "NotFound", "detail": "not found"}
# Content that speaks or shows the narration: never served (the content policy blocks every type today).
NARRATION_TYPES = {ArtifactType.AUDIO_ASSET, ArtifactType.SUBTITLE, ArtifactType.AUDIO_PLAN,
                   ArtifactType.PRESENTATION_TIMELINE, ArtifactType.SCRIPT, ArtifactType.VIDEO}
# Narration metadata that gives an answer or its length.
NARRATION_SECRETS = {"text", "usage", "duration", "source_ref", "segment_id", "slide_id", "asset_id", "validation"}


def canary_teacher(request):
    lesson = make_teacher(True)(request)
    lesson["exercises"][0]["answer"] = CANARY
    return lesson


def lesson_answers(container, task_id: str) -> list[str]:
    lesson = json.loads(container.artifacts.read(container.artifacts.find(task_id, "lesson").artifact_id))
    answers = [e["answer"] for e in lesson["exercises"]] + [q["answer"] for q in lesson.get("check_questions", [])]
    return [a.rstrip(".").strip() for a in answers if a and a.strip()]


def assert_listing_is_safe(listing: list[dict], answers: list[str], body: str) -> None:
    for answer in answers:
        assert answer.lower() not in body.lower(), f"answer {answer!r} is served in the listing"
    for view in listing:
        allowed = PUBLIC_METADATA[ArtifactType(view["type"])]
        assert set(view["metadata"]) <= allowed, (view["type"], set(view["metadata"]) - allowed)
        assert all(not isinstance(v, dict) for v in view["metadata"].values()), view
        if view["type"] == ArtifactType.AUDIO_ASSET.value:
            assert not set(view["metadata"]) & NARRATION_SECRETS, view["metadata"]


@pytest.fixture(scope="module")
def canary_lesson(tmp_path_factory):
    """A completed lesson (with narration) whose first exercise answer is CANARY."""
    import asyncio

    responders = {**default_responders(), "teacher": canary_teacher}
    settings = Settings(data_dir=tmp_path_factory.mktemp("canary-lesson") / "data", log_json=False)
    container = build_container(settings, llm_providers={"mock": MockLLMProvider(responders)})
    task = asyncio.run(run_lesson(container))
    yield container, task
    container.close()


def test_the_lesson_narrates_the_canary_internally(canary_lesson) -> None:
    """The precondition: the stored narration does speak the canary, so the listing checks below can fail."""
    container, task = canary_lesson
    audio = [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.AUDIO_ASSET]
    spoken = [a for a in audio if CANARY in a.metadata["text"]]
    assert spoken and all(a.metadata["source_type"] == "answer_explanation" for a in spoken)
    assert CANARY in lesson_answers(container, task.task_id)
    # the internal artifact keeps everything the pipeline uses
    assert all({"text", "duration", "usage", "checksum", "object_key"} <= set(a.metadata) for a in audio)


def test_task_artifact_listing_serves_no_answer_and_only_allowlisted_metadata(canary_lesson) -> None:
    container, task = canary_lesson
    with client_for(container, LEARNER["learner_id"], auth=TestAuth()) as a:
        response = a.get(f"/tasks/{task.task_id}/artifacts")
    assert response.status_code == 200
    listing = response.json()
    assert CANARY not in response.text
    assert_listing_is_safe(listing, lesson_answers(container, task.task_id), response.text)
    # the listing still describes the narration: every clip, its kind, format and voice
    audio = [v for v in listing if v["type"] == ArtifactType.AUDIO_ASSET.value]
    assert audio and {"answer_explanation", "speaker_notes"} <= {v["metadata"]["source_type"] for v in audio}
    assert all(v["media_type"] == "audio/wav" and v["metadata"]["voice"] for v in audio)
    # every type the lesson produced was checked
    assert {ArtifactType.AUDIO_ASSET, ArtifactType.IMAGE_ASSET, ArtifactType.LESSON, ArtifactType.VIDEO} <= {
        ArtifactType(v["type"]) for v in listing}


def test_narration_content_stays_blocked_for_the_owner_and_others(canary_lesson) -> None:
    container, task = canary_lesson
    ids = [a.artifact_id for a in container.task_service.artifacts(task.task_id) if a.type in NARRATION_TYPES]
    assert {a.type for a in container.task_service.artifacts(task.task_id)} >= NARRATION_TYPES
    auth = TestAuth()
    for learner in (LEARNER["learner_id"], INTRUDER):
        with client_for(container, learner, auth=auth) as c:
            for artifact_id in ids:
                r = c.get(f"/artifacts/{artifact_id}/content")
                assert (r.status_code, r.json()) == (404, NOT_FOUND), (learner, artifact_id)
                assert CANARY not in r.text
    with client_for(container, INTRUDER, auth=auth) as c:
        assert c.get(f"/tasks/{task.task_id}/artifacts").status_code == 404  # ownership is unchanged


def test_learning_cycle_and_its_child_lesson_listings_serve_no_answer(cycle_env_at) -> None:
    """A cycle's lesson runs as a child task the learner lists through /tasks/{child}/artifacts."""
    env = cycle_env_at("session")
    c = env.container
    [cycle] = c.learning_cycle_service.repository.for_learner(CYCLE_LEARNER)
    lesson_task = next(s.child_id for s in cycle.steps if s.kind.value == "LESSON")
    answers = lesson_answers(c, lesson_task)
    narrated = [a for a in c.task_service.artifacts(lesson_task) if a.type == ArtifactType.AUDIO_ASSET
                and any(x.lower() in a.metadata["text"].lower() for x in answers)]
    assert narrated, "precondition: the child lesson narrates an answer"
    with client_for(c, CYCLE_LEARNER, auth=TestAuth()) as a:
        for path in (f"/tasks/{lesson_task}/artifacts", f"/learning-cycles/{cycle.cycle_id}/artifacts"):
            r = a.get(path)
            assert r.status_code == 200, path
            assert_listing_is_safe(r.json(), answers, r.text)
