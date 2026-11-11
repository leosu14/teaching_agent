"""A whole lesson on the S3 object store (MVP item 8): every artifact's bytes go to the bucket under store-relative
keys, nothing is written to the local object directory, and the API serves content back from the bucket."""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.storage.s3 import S3ObjectStore
from app.services import container as container_module
from app.services.container import build_container
from tests.auth import client_for, drain
from tests.conftest import ANSWERS, LEARNER
from tests.fake_s3 import FakeS3

LID = LEARNER["learner_id"]


@pytest.fixture
def s3_container(tmp_path, monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(container_module, "S3ObjectStore",
                        lambda config: S3ObjectStore(config, transport=fake.transport()))
    settings = Settings(data_dir=tmp_path / "data", log_json=False, object_store="s3", s3_bucket=fake.bucket,
                        s3_endpoint_url=f"https://{fake.host}", s3_access_key_id=fake.access_key,
                        s3_secret_access_key=fake.secret_key, s3_prefix="pilot")
    c = build_container(settings, llm_providers={"mock": MockLLMProvider(default_responders())})
    yield c, fake, tmp_path
    c.close()


def test_a_lesson_lives_in_the_bucket(s3_container) -> None:
    c, fake, tmp_path = s3_container
    with client_for(c, LID) as client:
        client.put(f"/learners/{LID}", json=LEARNER["profile"])
        created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID})
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
        stored = c.artifacts.list_for_task(task["task_id"])
        assert stored and all(a.uri.startswith("object:") for a in stored)
        assert all(f"pilot/{a.uri.removeprefix('object:')}" in fake.objects for a in stored)
        assert {"LESSON", "PRESENTATION", "AUDIO_ASSET", "VIDEO"} <= {a.type.value for a in stored}
        served = [a for a in client.get(f"/tasks/{task['task_id']}/artifacts").json()
                  if client.get(f"/artifacts/{a['artifact_id']}/content").status_code == 200]
        assert served
        body = client.get(f"/artifacts/{served[0]['artifact_id']}/content")
        assert body.content == fake.objects[f"pilot/{c.artifacts.get(served[0]['artifact_id']).uri[7:]}"]
        assert "s3.test" not in str(body.headers) and "pilot/" not in str(body.headers)
    objects_dir = tmp_path / "data" / "objects"
    assert not objects_dir.exists() or not any(p.is_file() for p in objects_dir.rglob("*"))
