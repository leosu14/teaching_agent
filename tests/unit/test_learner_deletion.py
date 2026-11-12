"""MVP item 9: the learner data deletion's repository, on its own (SQLite).

Every table is either erased with a learner or kept on purpose, so a new table cannot hold learner data without a
decision; the events that name a learner outside any task are matched exactly, LIKE wildcards in the id included; a
content-addressed object another learner's artifact holds is kept, every other object of the learner is deleted; work
in progress blocks the deletion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.artifact import Artifact, ArtifactType
from app.schemas.events import Event
from app.schemas.privacy import RETAINED_RECORDS, LearnerDataBusy
from app.storage.db import create_db, dispose
from app.storage.orm import Base, LearningCycleRow, TaskRow
from app.storage.repositories import (
    ERASED_TABLES,
    KEPT_TABLES,
    SqlArtifactRepository,
    SqlEventRepository,
    SqlLearnerDataRepository,
    _learner_pattern,
)

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def sessions(tmp_path):
    factory = create_db(f"sqlite:///{tmp_path / 'del.db'}")
    yield factory
    dispose(factory)


def add_task(sessions, task_id: str, learner_id: str, *, lease_until: datetime | None = None) -> None:
    with sessions.begin() as s:
        s.add(TaskRow(task_id=task_id, user_id="u", learner_id=learner_id, status="COMPLETED", created_at=NOW,
                      updated_at=NOW, body="{}", lease_owner="w" if lease_until else None,
                      lease_expires_ms=int(lease_until.timestamp() * 1000) if lease_until else None))


def add_artifact(sessions, task_id: str, uri: str, content_hash: str) -> None:
    SqlArtifactRepository(sessions).add(Artifact(
        artifact_id=f"art_{task_id}_{content_hash}", task_id=task_id, type=ArtifactType.IMAGE_ASSET,
        name=f"img-{content_hash}",
        uri=uri, media_type="image/png", content_hash=content_hash, size_bytes=1, version=1, provider="mock"))


def test_every_table_is_either_erased_or_kept_on_purpose() -> None:
    erased, kept = set(ERASED_TABLES), set(KEPT_TABLES)
    assert not erased & kept
    assert erased | kept == set(Base.metadata.tables)  # a new table must be put in one of the two
    assert set(RETAINED_RECORDS) <= kept and "spend_counters" in kept  # the caps keep counting after a deletion
    for name, table in Base.metadata.tables.items():
        if "learner_id" in table.c and name not in RETAINED_RECORDS:
            assert name in erased, name  # a table keyed by learner is erased


@pytest.mark.parametrize("learner_id", ["a_b", "a%b", 'a"b', "a\\b", "ä b"])
def test_events_naming_a_learner_are_matched_exactly(sessions, learner_id) -> None:
    events = SqlEventRepository(sessions)
    near = [learner_id.replace("_", "x").replace("%", "xx").replace('"', "").replace("\\", "") + "!",
            learner_id + "2", "z" + learner_id]
    for i, lid in enumerate([learner_id, *near]):
        events.append(Event(event_id=f"e{i}", type="GOAL_CREATED", data={"learner_id": lid}))
    repo = SqlLearnerDataRepository(sessions, clock=lambda: NOW)
    assert repo.holds_data(learner_id)
    assert repo.erase(learner_id) == {"events": 1}
    assert not repo.holds_data(learner_id)
    assert all(repo.holds_data(lid) for lid in near)


def test_the_pattern_escapes_like_wildcards() -> None:
    assert _learner_pattern("a_b%c") == '%"learner\\_id":"a\\_b\\%c"%'


def test_shared_content_addressed_objects_are_kept_and_the_rest_deleted(sessions) -> None:
    add_task(sessions, "t-a", "learner-a")
    add_task(sessions, "t-b", "learner-b")
    shared, own = "object:objects/sha256/aa/aa.png", "object:objects/sha256/bb/bb.png"
    keyed = "object:t-a/lesson/v1.json"
    add_artifact(sessions, "t-a", shared, "aa")
    add_artifact(sessions, "t-b", shared, "aa")  # B's lesson holds the identical image
    add_artifact(sessions, "t-a", own, "bb")
    add_artifact(sessions, "t-a", keyed, "cc")
    add_artifact(sessions, "t-b", "object:t-b/lesson/v1.json", "cc")  # same bytes under B's own key: not shared
    repo = SqlLearnerDataRepository(sessions, clock=lambda: NOW)
    found = repo.objects("learner-a")
    assert sorted(found.delete) == sorted([own, keyed]) and found.shared == 1
    assert repo.erase("learner-a") == {"artifacts": 3, "tasks": 1}
    assert repo.objects("learner-b").delete == (shared, "object:t-b/lesson/v1.json")  # B's still all there


def test_a_live_task_lease_or_a_driven_cycle_blocks_and_an_expired_one_does_not(sessions) -> None:
    add_task(sessions, "t-a", "learner-a", lease_until=NOW + timedelta(minutes=5))
    repo = SqlLearnerDataRepository(sessions, clock=lambda: NOW)
    for call in (repo.objects, repo.erase):
        with pytest.raises(LearnerDataBusy):
            call("learner-a")
    assert repo.holds_data("learner-a")
    later = SqlLearnerDataRepository(sessions, clock=lambda: NOW + timedelta(minutes=6))
    assert later.erase("learner-a") == {"tasks": 1}

    with sessions.begin() as s:
        s.add(LearningCycleRow(cycle_id="c-a", learner_id="learner-a", status="RUNNING", version=1, created_at=NOW,
                               body="{}", lease_token="tok", lease_expires_ms=int(NOW.timestamp() * 1000) + 60_000))
    with pytest.raises(LearnerDataBusy):
        repo.erase("learner-a")
    assert later.erase("learner-a") == {"learning_cycles": 1}


def test_a_learner_without_data_erases_nothing(sessions) -> None:
    repo = SqlLearnerDataRepository(sessions, clock=lambda: NOW)
    assert repo.erase("nobody") == {} and repo.objects("nobody").delete == ()


def test_the_privacy_notice_is_static_text_linked_from_both_apps() -> None:
    """MVP item 9: the notice is served like the apps' files (tests/unit/test_student_app.py checks the headers) but
    has no script at all, and both apps link it."""
    import re

    from app.api.routes import web

    notice = (web.WEB_DIR / "privacy.html").read_text(encoding="utf-8")
    assert "privacy.html" in web.FILES
    assert "<script" not in notice and " style=" not in notice and not re.search(r"\son[a-z]+\s*=", notice)
    assert not re.search(r"(src|href)=\"(https?:)?//", notice)  # no third-party origin
    for section in ("who", "what", "who-sees", "providers", "how-long", "delete"):
        assert f'id="{section}"' in notice, section
    for page in ("index.html", "teacher.html"):
        assert 'href="privacy.html"' in (web.WEB_DIR / page).read_text(encoding="utf-8"), page
    assert "privacy.html" in (web.WEB_DIR / "app.js").read_text(encoding="utf-8")
