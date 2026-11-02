"""GET /artifacts/{id}/content: ownership, authentication, the block-by-default content policy, response headers,
streaming and missing objects (issue #24).

The production policy serves only raster IMAGE_ASSETs planned from the answer-free visual brief (issue #26; their tests
are in test_image_content.py). The generic serving path is exercised with a test-only policy injected into the
container; every type is checked under both."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.schemas.artifact import ArtifactType
from app.services.artifact_content import BLOCKED, CHUNK, PROVENANCE, SERVABLE, ArtifactContentService
from app.storage.object_store import FilesystemObjectStore
from tests.auth import TestAuth, client_for
from tests.unit.helpers import scope

OWNER, INTRUDER = "owner-learner", "intruder-learner"
NOT_FOUND = {"error": "NotFound", "detail": "not found"}
UNAVAILABLE = {"error": "ContentUnavailable", "detail": "artifact content is unavailable"}
MEDIA = "application/octet-stream"
ALL_TYPES = list(ArtifactType)


@pytest.fixture
def env(container):
    """A task of OWNER's with one stored artifact per type, each with distinct content."""
    task = container.orchestrator.create_task(request="r", learner_id=OWNER, user_id="u")
    s, _ = scope(task.task_id)
    stored = {t: container.artifacts.store(task_id=task.task_id, name=f"a_{t.value.lower()}", type=t, media_type=MEDIA,
                                           content=f"content of {t.value}".encode(), provider="test", parent_ids=[],
                                           metadata={}, scope=s)
              for t in ALL_TYPES}
    auth = TestAuth()
    return container, stored, auth


def allow(container, *types: ArtifactType, media: str = MEDIA) -> None:
    """A test-only policy: these types are servable as `media`, with no provenance requirement."""
    container.artifact_content = ArtifactContentService(container.artifacts, {t: frozenset({media}) for t in types},
                                                        provenance={})


def object_path(artifact) -> Path:
    return Path(artifact.uri.removeprefix("file://"))


# --- the policy itself -------------------------------------------------------------------------------------------


def test_every_artifact_type_is_classified_exactly_once() -> None:
    assert set(BLOCKED) | set(SERVABLE) == set(ArtifactType)
    assert not set(BLOCKED) & set(SERVABLE)
    assert all(reason.strip() for reason in BLOCKED.values())


def test_only_raster_images_are_servable_once_proven_free_of_answer_keys() -> None:
    """Allowing a type is a deliberate change: update this test, BLOCKED, SERVABLE and docs/security.md together."""
    assert dict(SERVABLE) == {
        ArtifactType.IMAGE_ASSET: frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})}
    assert set(PROVENANCE) == {ArtifactType.IMAGE_ASSET}  # and only with the answer-isolated planning marker
    for answer_bearing in (ArtifactType.LESSON, ArtifactType.EXERCISE, ArtifactType.ASSESSMENT_ITEM,
                           ArtifactType.PRESENTATION, ArtifactType.SLIDE_PLAN, ArtifactType.LEARNER_EVALUATION,
                           ArtifactType.TEACHING_SESSION, ArtifactType.VISUAL_PLAN):
        assert answer_bearing in BLOCKED


# --- every type, under the production policy and when allowed ----------------------------------------------------


@pytest.mark.parametrize("type_", ALL_TYPES, ids=lambda t: t.value)
def test_production_policy_serves_no_type_stored_as_octet_stream(env, type_) -> None:
    container, stored, auth = env
    with client_for(container, OWNER, auth=auth) as a:
        theirs = a.get(f"/artifacts/{stored[type_].artifact_id}/content")
        nothing = a.get("/artifacts/art_does_not_exist/content")
    assert (theirs.status_code, theirs.json()) == (nothing.status_code, nothing.json()) == (404, NOT_FOUND)
    assert f"content of {type_.value}" not in theirs.text


@pytest.mark.parametrize("type_", ALL_TYPES, ids=lambda t: t.value)
def test_an_allowed_type_is_served_and_every_other_type_is_not(env, type_) -> None:
    container, stored, auth = env
    allow(container, type_)
    with client_for(container, OWNER, auth=auth) as a:
        for t, artifact in stored.items():
            r = a.get(f"/artifacts/{artifact.artifact_id}/content")
            if t == type_:
                assert r.status_code == 200, r.text
                assert r.content == f"content of {t.value}".encode()
            else:
                assert (r.status_code, r.json()) == (404, NOT_FOUND), t


def test_an_allowed_type_in_a_media_type_not_allowed_is_not_served(env) -> None:
    container, stored, auth = env
    allow(container, ArtifactType.IMAGE_ASSET, media="image/png")  # stored as application/octet-stream
    with client_for(container, OWNER, auth=auth) as a:
        r = a.get(f"/artifacts/{stored[ArtifactType.IMAGE_ASSET].artifact_id}/content")
    assert (r.status_code, r.json()) == (404, NOT_FOUND)


# --- ownership and authentication --------------------------------------------------------------------------------


def test_another_learners_artifact_is_indistinguishable_from_an_unknown_id(env) -> None:
    container, stored, auth = env
    allow(container, *ALL_TYPES)  # even a servable type: ownership is checked first
    with client_for(container, INTRUDER, auth=auth) as b:
        for artifact in stored.values():
            theirs = b.get(f"/artifacts/{artifact.artifact_id}/content")
            assert (theirs.status_code, theirs.json()) == (404, NOT_FOUND)
            assert "content of" not in theirs.text
        nothing = b.get("/artifacts/art_does_not_exist/content")
        assert (nothing.status_code, nothing.json()) == (404, NOT_FOUND)
        assert theirs.headers.get("content-type") == nothing.headers.get("content-type")


def test_unauthenticated_requests_get_the_existing_401(env) -> None:
    container, stored, auth = env
    allow(container, *ALL_TYPES)
    path = f"/artifacts/{stored[ArtifactType.LESSON].artifact_id}/content"
    with TestClient(create_app(container, authenticator=auth.authenticator)) as anonymous:
        for headers in ({}, {"Authorization": "Bearer not-a-real-token-at-all"}, {"Authorization": "Bearer "},
                        {"Authorization": "Basic Zm9vOmJhcg=="}, {"X-Auth-Subject": OWNER}):
            for p in (path, "/artifacts/art_does_not_exist/content"):
                r = anonymous.get(p, headers=headers)
                assert r.status_code == 401, (p, headers)
                assert r.json() == {"error": "Unauthenticated", "detail": "authentication required"}
                assert r.headers["www-authenticate"] == "Bearer"


# --- response shape ----------------------------------------------------------------------------------------------


def test_served_content_is_a_download_without_storage_details(env) -> None:
    container, stored, auth = env
    artifact = stored[ArtifactType.IMAGE_ASSET]
    allow(container, ArtifactType.IMAGE_ASSET)
    with client_for(container, OWNER, auth=auth) as a:
        r = a.get(f"/artifacts/{artifact.artifact_id}/content")
    assert r.status_code == 200
    h = r.headers
    assert h["content-disposition"] == f'attachment; filename="{artifact.artifact_id}.bin"'
    assert h["x-content-type-options"] == "nosniff"
    assert h["cache-control"] == "no-store"
    assert h["content-type"] == MEDIA
    # no byte length, digest or validator: each could check a guessed answer key
    for name in ("content-length", "etag", "last-modified", "content-md5", "digest", "location"):
        assert name not in h, name
    everything = " ".join(f"{k}: {v}" for k, v in h.items())
    for secret in (artifact.uri, artifact.content_hash, str(object_path(artifact)), artifact.name):
        assert secret not in everything


def test_content_is_streamed_in_chunks_from_the_object_store(env, monkeypatch) -> None:
    container, stored, auth = env
    s, _ = scope(stored[ArtifactType.VIDEO].task_id)
    big = bytes(range(256)) * (3 * CHUNK // 256) + b"tail"  # more than three chunks
    artifact = container.artifacts.store(task_id=s.task_id, name="big_video", type=ArtifactType.VIDEO,
                                         media_type="video/mp4", content=big, provider="test", parent_ids=[],
                                         metadata={}, scope=s)
    allow(container, ArtifactType.VIDEO, media="video/mp4")

    def whole_read(*_args, **_kwargs):
        raise AssertionError("content must be streamed, not read whole")

    monkeypatch.setattr(FilesystemObjectStore, "get", whole_read)
    opened = []
    real_open = FilesystemObjectStore.open

    def tracking_open(self, uri):
        f = real_open(self, uri)
        opened.append(f)
        return f

    monkeypatch.setattr(FilesystemObjectStore, "open", tracking_open)
    content = container.artifact_content.open(artifact.artifact_id)
    sizes = [len(chunk) for chunk in content.chunks()]
    assert len(sizes) == 4 and max(sizes) <= CHUNK and sum(sizes) == len(big)
    assert opened[-1].closed  # the stream is closed once it has been read
    with client_for(container, OWNER, auth=auth) as a:
        r = a.get(f"/artifacts/{artifact.artifact_id}/content")
    assert r.status_code == 200 and r.content == big
    assert r.headers["content-disposition"] == f'attachment; filename="{artifact.artifact_id}.mp4"'
    assert all(f.closed for f in opened)


# --- missing objects ---------------------------------------------------------------------------------------------


def test_a_missing_object_gets_a_fixed_sanitized_error(env) -> None:
    container, stored, auth = env
    artifact = stored[ArtifactType.IMAGE_ASSET]
    allow(container, ArtifactType.IMAGE_ASSET)
    object_path(artifact).unlink()
    with client_for(container, OWNER, auth=auth) as a:
        r = a.get(f"/artifacts/{artifact.artifact_id}/content")
    assert (r.status_code, r.json()) == (500, UNAVAILABLE)
    for secret in (artifact.uri, artifact.content_hash, str(object_path(artifact)), "objects", "file://"):
        assert secret not in r.text
    assert "content-disposition" not in r.headers


def test_a_missing_object_is_still_404_when_not_yours_or_not_servable(env) -> None:
    """The missing-object answer is reached only after ownership and the policy: it reveals nothing to others."""
    container, stored, auth = env
    allowed, blocked = stored[ArtifactType.IMAGE_ASSET], stored[ArtifactType.LESSON]
    allow(container, ArtifactType.IMAGE_ASSET)
    object_path(allowed).unlink()
    object_path(blocked).unlink()
    with client_for(container, INTRUDER, auth=auth) as b, client_for(container, OWNER, auth=auth) as a:
        assert b.get(f"/artifacts/{allowed.artifact_id}/content").json() == NOT_FOUND
        assert a.get(f"/artifacts/{blocked.artifact_id}/content").json() == NOT_FOUND


def test_an_object_outside_the_store_is_unavailable_not_served(env) -> None:
    container, stored, auth = env
    artifact = stored[ArtifactType.IMAGE_ASSET]
    allow(container, ArtifactType.IMAGE_ASSET)
    container.artifacts.get = lambda _id: artifact.model_copy(update={"uri": "file:///etc/passwd"})
    with client_for(container, OWNER, auth=auth) as a:
        r = a.get(f"/artifacts/{artifact.artifact_id}/content")
    assert (r.status_code, r.json()) == (500, UNAVAILABLE)
    assert "passwd" not in r.text
