from __future__ import annotations

import pytest

from app.artifacts.service import ArtifactGraphError, ArtifactService
from app.schemas.artifact import ArtifactDraft, ArtifactType
from app.schemas.events import Event
from app.schemas.learner import LearnerProfile
from app.schemas.task import Task, TaskStatus
from app.storage.db import create_db, dispose
from app.storage.object_store import FilesystemObjectStore
from app.storage.repositories import (
    NotFound,
    SqlArtifactRepository,
    SqlEventRepository,
    SqlLearnerRepository,
    SqlTaskRepository,
)
from tests.unit.helpers import scope


@pytest.fixture
def sessions(tmp_path):
    factory = create_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    yield factory
    dispose(factory)


@pytest.fixture
def artifacts(sessions, tmp_path):
    return ArtifactService(SqlArtifactRepository(sessions), FilesystemObjectStore(tmp_path / "objects"))


def draft(key, content="x", parents=(), name=None, type=ArtifactType.LESSON):
    return ArtifactDraft(key=key, name=name or key, type=type, media_type="text/plain", content=content,
                         parent_keys=list(parents))


def test_batch_builds_dependency_graph_and_stores_blobs_outside_db(artifacts, tmp_path) -> None:
    sc, events = scope("t1")
    stored = artifacts.store_batch("t1", [draft("script", parents=["lesson"], type=ArtifactType.SCRIPT),
                                          draft("lesson", content="body"),
                                          draft("review", parents=["lesson"], type=ArtifactType.REPORT)], sc)
    ids = stored.by_key
    assert artifacts.graph("t1") == {ids["lesson"]: [], ids["script"]: [ids["lesson"]], ids["review"]: [ids["lesson"]]}
    assert [a.name for a in artifacts.lineage(ids["script"])] == ["lesson"]
    lesson = artifacts.get(ids["lesson"])
    assert lesson.uri.startswith("file://") and artifacts.read(ids["lesson"]) == b"body"
    assert (tmp_path / "objects" / "t1" / "lesson" / "v1.txt").read_bytes() == b"body"
    assert [e.type for e in events] == ["artifact.created"] * 3


def test_versioning_and_deduplication(artifacts) -> None:
    sc, _ = scope("t1")
    v1 = artifacts.store_batch("t1", [draft("lesson", content="a")], sc).artifacts[0]
    same = artifacts.store_batch("t1", [draft("lesson", content="a")], sc).artifacts[0]
    v2 = artifacts.store_batch("t1", [draft("lesson", content="b")], sc).artifacts[0]
    assert same.artifact_id == v1.artifact_id
    assert (v1.version, v2.version) == (1, 2)
    assert len(artifacts.list_for_task("t1")) == 2


def test_graph_errors(artifacts) -> None:
    sc, _ = scope("t1")
    with pytest.raises(ArtifactGraphError, match="unknown parent"):
        artifacts.store_batch("t1", [draft("a", parents=["ghost"])], sc)
    with pytest.raises(ArtifactGraphError, match="cycle"):
        artifacts.store_batch("t1", [draft("a", parents=["b"]), draft("b", parents=["a"])], sc)


def test_object_store_rejects_paths_outside_root(tmp_path) -> None:
    store = FilesystemObjectStore(tmp_path / "o")
    with pytest.raises(ValueError):
        store.put("../escape.txt", b"x")
    with pytest.raises(ValueError):
        store.get((tmp_path / "elsewhere.txt").as_uri())


def test_repositories_round_trip(sessions) -> None:
    tasks = SqlTaskRepository(sessions)
    task = Task(task_id="t1", user_id="u", learner_id="l", request="r")
    tasks.save(task)
    task.status = TaskStatus.PLANNING
    tasks.save(task)
    assert tasks.get("t1").status == TaskStatus.PLANNING and len(tasks.list_for_learner("l")) == 1
    with pytest.raises(NotFound):
        tasks.get("missing")

    events = SqlEventRepository(sessions)
    for i in range(3):
        events.append(Event(event_id=f"e{i}", type=f"x.{i}", task_id="t1"))
    assert [e.type for e in events.list_for_task("t1")] == ["x.0", "x.1", "x.2"]

    learners = SqlLearnerRepository(sessions)
    assert learners.get("l") is None
    learners.save(LearnerProfile(learner_id="l", display_name="L"))
    assert learners.get("l").display_name == "L"
