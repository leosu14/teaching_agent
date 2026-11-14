"""MVP item 9: two learners' whole lessons on the S3 object store, then each deletes their data.

On mock providers the two lessons are identical, so their media (images, narration, slides, video) are the same
content-addressed objects in the bucket. Deleting the first learner removes their own objects (the lesson text and
plans under their task's keys, and media only their lesson holds) and keeps the shared media the second learner's lesson still uses, which keeps
serving. Deleting the second learner then removes the rest: no object any record of either lesson points at is left
in the bucket."""

from __future__ import annotations

import json
import sys
import threading

import pytest

from app.api.auth import AUTHOR
from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.storage.s3 import S3ObjectStore
from app.services import container as container_module
from app.services.container import build_container
from app.storage import repositories
from tests.auth import client_for, drain
from tests.conftest import ANSWERS, LEARNER
from tests.fake_s3 import FakeS3

FIRST, SECOND = "deleting-learner-1", "deleting-learner-2"


@pytest.fixture
def s3_container(tmp_path, monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(container_module, "S3ObjectStore",
                        lambda config: S3ObjectStore(config, transport=fake.transport()))
    settings = Settings(data_dir=tmp_path / "data", log_json=False, object_store="s3", s3_bucket=fake.bucket,
                        s3_endpoint_url=f"https://{fake.host}", s3_access_key_id=fake.access_key,
                        s3_secret_access_key=fake.secret_key, s3_prefix="pilot")
    c = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    yield c, fake
    c.close()


def lesson(c, client, learner_id: str) -> str:
    client.put(f"/learners/{learner_id}", json=LEARNER["profile"])
    created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": learner_id})
    drain(c)
    task = client.get(f"/tasks/{created.json()['task_id']}").json()
    while task["status"] == "WAITING":
        sheet = task["waiting"]["prompt"]
        key = ANSWERS["rounds"][sheet["round_number"] - 1]
        answers = [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]
        client.post(f"/tasks/{task['task_id']}/answers", json={"answers": answers})
        drain(c)
        task = client.get(f"/tasks/{task['task_id']}").json()
    assert task["status"] == "COMPLETED"
    return task["task_id"]


def keys(c, task_id: str) -> set[str]:
    return {f"pilot/{a.uri.removeprefix('object:')}" for a in c.artifacts.list_for_task(task_id)}


def test_each_learner_deletes_their_lesson_from_the_bucket(s3_container) -> None:
    c, fake = s3_container
    with client_for(c, FIRST) as one, client_for(c, SECOND) as two:
        first, second = lesson(c, one, FIRST), lesson(c, two, SECOND)
        mine, theirs = keys(c, first), keys(c, second)
        shared = mine & theirs
        assert shared and all("/objects/sha256/" in k for k in shared)  # identical media, stored once
        assert any(k.startswith(f"pilot/{first}/") for k in mine - shared)  # the lesson text under its task
        assert (mine | theirs) <= set(fake.objects)

        r = one.delete(f"/learners/{FIRST}")
        assert r.status_code == 200, r.text
        assert r.json()["objects_deleted"] == len(mine - shared)
        assert r.json()["objects_kept_shared"] == len(shared)
        assert not (mine - shared) & set(fake.objects)  # the first learner's own objects are gone
        assert theirs <= set(fake.objects)  # the second learner's lesson is whole
        served = [a for a in two.get(f"/tasks/{second}/artifacts").json()
                  if two.get(f"/artifacts/{a['artifact_id']}/content").status_code == 200]
        assert served and one.get(f"/tasks/{first}").status_code == 404

        r = two.delete(f"/learners/{SECOND}")
        assert r.status_code == 200 and r.json()["objects_kept_shared"] == 0, r.text
        assert not (mine | theirs) & set(fake.objects)
        # Nothing any record of either lesson pointed at is left. What remains is image candidates the visual step
        # stored but did not select: content-addressed pictures of the topic that no row ties to anyone
        # (docs/privacy.md, known limits).
        left = [k for k in fake.objects if not k.startswith("pilot/records/")]
        assert all(k.startswith("pilot/objects/sha256/") and k.endswith((".png", ".jpg", ".webp", ".svg"))
                   for k in left), left


ANSWER = "respuesta-carrera-s3"  # no object in the bucket may hold it after the deletion


def registered_item(c, client, task_id: str) -> str:
    lesson = c.artifacts.find(task_id, "lesson")
    concept = json.loads(c.artifacts.read(lesson.artifact_id))["sections"][0]["concept_id"]
    item = {"item": {"assessment_item_id": "s3-race-item", "lesson_id": task_id, "concept_id": concept,
                     "prompt": "Ayer yo ___ (hablar) con María.", "expected_answer": "hablé",
                     "response_type": "SHORT_TEXT", "language": "es"}}
    r = client.post("/assessment-items", json=item)
    assert r.status_code == 201, r.text
    return "s3-race-item"


def left_in_bucket(fake, task_id: str) -> list[str]:
    """Objects under the learner's task key, or holding their answer."""
    return sorted(k for k, data in fake.objects.items() if k.startswith(f"pilot/{task_id}/") or ANSWER.encode() in data)


@pytest.mark.parametrize("order", ["deletion first", "grading first"])
def test_a_deletion_racing_a_graded_answer_leaves_nothing_in_the_bucket(s3_container, monkeypatch, order) -> None:
    """MVP item 9 on S3: the grading writes its grade and feedback files to the bucket. Deletion first (it runs while
    the answer is being graded): every later write is refused, and a file put just before its refused row is
    removed again. Grading first (its attempt write holds the locks, the deletion waits): the deletion then removes
    what the grading wrote, files included."""
    c, fake = s3_container
    with client_for(c, FIRST, roles=(AUTHOR,)) as one:
        task_id = lesson(c, one, FIRST)
        item = registered_item(c, one, task_id)
        receipts = []
        if order == "deletion first":
            real = c.assessment_service.engine.grade

            async def grade_after_a_deletion(*args, **kwargs):
                receipts.append(c.learner_deletion.delete(FIRST))
                return await real(*args, **kwargs)

            monkeypatch.setattr(c.assessment_service.engine, "grade", grade_after_a_deletion)
        else:
            deletion = []
            guard = repositories._guard

            def guard_then_delete(s) -> None:
                if deletion and not receipts:  # the grading's next write waits for the deletion it let in
                    deletion[0].join(30)
                    receipts.append(deletion[0].receipt)
                guard(s)
                if not deletion and sys._getframe(1).f_code.co_name == "add_attempt":
                    thread = threading.Thread(target=lambda: setattr(thread, "receipt", c.learner_deletion.delete(FIRST)))
                    deletion.append(thread)
                    thread.start()
                    thread.join(0.3)
                    assert thread.is_alive(), "the deletion did not wait for the grading's write"

            monkeypatch.setattr(repositories, "_guard", guard_then_delete)
        r = one.post(f"/assessment-items/{item}/attempts",
                     json={"learner_id": FIRST, "answer": ANSWER, "attempt_id": "s3-race-attempt"})
        assert r.status_code == 404, r.text
        assert receipts and receipts[0].objects_deleted > 0
        if order == "grading first":
            assert receipts[0].records["assessment_attempts"] == 1  # the attempt it waited for
    assert left_in_bucket(fake, task_id) == []
    assert not c.learner_deletion.repository.holds_data(FIRST)
