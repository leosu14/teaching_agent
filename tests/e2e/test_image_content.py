"""GET /artifacts/{id}/content for learner images under the production policy (issue #26).

Only an IMAGE_ASSET planned from the answer-free visual brief, stored as PNG, JPEG, WebP or GIF, is served to the
learner who owns it. An SVG, an image made before the fix (no planning marker), another learner's image and every
other type get the same 404 as an unknown id."""

from __future__ import annotations

import pytest

from app.schemas.artifact import ArtifactType
from app.schemas.task import TaskStatus
from app.schemas.visual import ANSWER_ISOLATED_PLANNING
from tests.auth import TestAuth, client_for
from tests.conftest import LEARNER, run_lesson
from tests.unit.helpers import scope

OWNER, INTRUDER = "owner-learner", "intruder-learner"
NOT_FOUND = {"error": "NotFound", "detail": "not found"}
RASTER = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
MARKED = {"planning": ANSWER_ISOLATED_PLANNING}


@pytest.fixture
def env(container):
    task = container.orchestrator.create_task(request="r", learner_id=OWNER, user_id="u")
    s, _ = scope(task.task_id)

    def store(media_type: str, metadata: dict, type_: ArtifactType = ArtifactType.IMAGE_ASSET, name: str = "img"):
        return container.artifacts.store(task_id=task.task_id, name=name, type=type_, media_type=media_type,
                                         content=f"bytes of {name} {media_type}".encode(), provider="test",
                                         parent_ids=[], metadata=metadata, scope=s)

    return container, store, TestAuth()


def get(container, auth, learner, artifact_id):
    with client_for(container, learner, auth=auth) as client:
        return client.get(f"/artifacts/{artifact_id}/content")


@pytest.mark.parametrize("media_type", sorted(RASTER))
def test_an_answer_isolated_raster_image_is_served_to_its_owner(env, media_type) -> None:
    container, store, auth = env
    image = store(media_type, MARKED)
    r = get(container, auth, OWNER, image.artifact_id)
    assert r.status_code == 200, r.text
    assert r.content == f"bytes of img {media_type}".encode()
    assert r.headers["content-type"] == media_type
    assert r.headers["content-disposition"] == f'attachment; filename="{image.artifact_id}{RASTER[media_type]}"'
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert "content-length" not in r.headers


@pytest.mark.parametrize("media_type, metadata", [
    ("image/svg+xml", MARKED),  # text: it could spell an answer out (and carries script)
    ("image/png", {}),  # planned before the fix, from the whole lesson
    ("image/png", {"planning": "something-else"}),
    ("application/octet-stream", MARKED),
], ids=["svg", "legacy", "unknown-marker", "octet-stream"])
def test_svg_and_unproven_images_are_indistinguishable_from_unknown_ids(env, media_type, metadata) -> None:
    container, store, auth = env
    image = store(media_type, metadata)
    theirs = get(container, auth, OWNER, image.artifact_id)
    nothing = get(container, auth, OWNER, "art_does_not_exist")
    assert (theirs.status_code, theirs.json()) == (nothing.status_code, nothing.json()) == (404, NOT_FOUND)
    assert "bytes of" not in theirs.text


def test_another_learners_image_is_not_served(env) -> None:
    container, store, auth = env
    image = store("image/png", MARKED)
    r = get(container, auth, INTRUDER, image.artifact_id)
    assert (r.status_code, r.json()) == (404, NOT_FOUND)


@pytest.mark.parametrize("type_", [t for t in ArtifactType if t != ArtifactType.IMAGE_ASSET], ids=lambda t: t.value)
def test_no_other_type_is_served_even_as_a_marked_png(env, type_) -> None:
    container, store, auth = env
    artifact = store("image/png", MARKED, type_=type_, name=type_.value.lower())
    r = get(container, auth, OWNER, artifact.artifact_id)
    assert (r.status_code, r.json()) == (404, NOT_FOUND)


async def test_images_from_a_lesson_run_are_downloadable_by_their_learner_only(container) -> None:
    """End to end: the visual stage marks every image it makes, and the owner can download each one."""
    task = await run_lesson(container)
    assert task.status == TaskStatus.COMPLETED, task.errors
    images = [a for a in container.task_service.artifacts(task.task_id) if a.type == ArtifactType.IMAGE_ASSET]
    assert images and all(a.metadata["planning"] == ANSWER_ISOLATED_PLANNING for a in images)
    auth = TestAuth()
    with client_for(container, LEARNER["learner_id"], auth=auth) as a, client_for(container, INTRUDER, auth=auth) as b:
        served = 0
        for image in images:
            mine = a.get(f"/artifacts/{image.artifact_id}/content")
            if image.media_type == "image/svg+xml":
                assert (mine.status_code, mine.json()) == (404, NOT_FOUND)
                continue
            assert mine.status_code == 200 and mine.content == container.artifacts.read(image.artifact_id)
            assert b.get(f"/artifacts/{image.artifact_id}/content").json() == NOT_FOUND
            served += 1
    assert served
