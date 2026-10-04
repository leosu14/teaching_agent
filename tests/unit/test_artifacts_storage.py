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
    SqlEvidenceRepository,
    SqlGoalRepository,
    SqlLearnerRepository,
    SqlLearningEventRepository,
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


def test_learner_history_repositories(sessions) -> None:
    from datetime import timedelta

    from app.schemas.learner import EvidenceConflict, GoalStatus, LearningEvent, LearningEvidence, LearningGoal
    from tests.unit.helpers import NOW

    def ev(ref: str, at, concept: str = "c1") -> LearningEvidence:
        return LearningEvidence(evidence_id=LearningEvidence.id_for("l1", "exercise", ref, concept), learner_id="l1",
                                concept_id=concept, source_type="exercise", source_ref=ref, correctness="correct",
                                score=1.0, difficulty=0.5, timestamp=at)

    evidence = SqlEvidenceRepository(sessions)
    later, earlier = ev("b", NOW + timedelta(hours=1)), ev("a", NOW, "c2")
    assert evidence.add(later) and evidence.add(earlier)
    assert evidence.add(later) is False  # append-only, idempotent
    with pytest.raises(EvidenceConflict):
        evidence.add(later.model_copy(update={"difficulty": 0.9}))
    assert evidence.for_learner("l1") == [later, earlier]  # recording order
    assert evidence.for_learner("l1", "c2") == [earlier] and evidence.for_learner("other") == []

    events = SqlLearningEventRepository(sessions)
    e = LearningEvent.create("l1", "lesson_completed", NOW, key="t1", subject="spanish", concept_ids=["c1"])
    assert events.add(e) and not events.add(e)
    assert events.for_learner("l1") == [e]

    goals = SqlGoalRepository(sessions)
    goal = LearningGoal(goal_id="g1", learner_id="l1", domain="spanish", target_concepts=["c1"])
    goals.save(goal)
    goals.save(goal.model_copy(update={"status": GoalStatus.COMPLETED}))
    assert goals.get("g1").status == GoalStatus.COMPLETED and goals.get("missing") is None
    # Goals stored with the earlier status names and `deadline` still load.
    legacy = LearningGoal.model_validate({"goal_id": "g0", "learner_id": "l1", "domain": "spanish",
                                          "target_concepts": ["c1"], "status": "achieved",
                                          "deadline": "2030-01-01T00:00:00Z"})
    assert legacy.status == GoalStatus.COMPLETED and legacy.target_date.year == 2030
    assert [g.goal_id for g in goals.for_learner("l1")] == ["g1"]
