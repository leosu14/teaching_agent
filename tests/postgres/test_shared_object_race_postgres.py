"""A learner's data deletion racing another learner's reuse of a shared stored object, on PostgreSQL with two app
instances and one S3 bucket (MVP item 9; the second review's B1, probe_shared_object_race.py).

Lesson media are content-addressed: two learners' lessons with the same media share one object in the bucket. The
second learner's lesson runs on instance 1 (its worker); the first learner deletes their data on instance 2. The two
meet on the object's lock (a PostgreSQL advisory lock, `_lock_object`), and both orders are forced with real lock waits,
each one seen waiting in `pg_stat_activity` before it is let go:

- the reuse first: the second lesson's hold on the object is written and not yet committed; the deletion's rows go,
  and its object purge waits for the object lock. Once the hold commits, the purge sees it and keeps the object.
- the deletion first: the purge holds the object lock (nothing references the object any more); the second lesson's
  hold waits for it. The purge deletes the object and commits; the hold then goes in and the lesson writes the object
  afresh (it is no longer in the bucket), then adds its row.

Either way the second lesson completes (paused and resumed on the other instance once the race is over), every file
of it is stored with the bytes its row names and serves from both instances, another lesson of that learner reuses
the same media, and no artifact row of either learner points at a missing object. Their own deletion then empties
the bucket."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest
from sqlalchemy import create_engine, select, text

from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.providers.storage.s3 import S3ObjectStore
from app.schemas.common import new_id
from app.services import container as container_module
from app.services.container import build_container
from app.storage import repositories
from app.storage.orm import ArtifactRow, ObjectHoldRow
from tests.auth import client_for, drain
from tests.conftest import ANSWERS, LEARNER
from tests.fake_s3 import FakeS3

URL = os.environ.get("TA_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set TA_TEST_POSTGRES_URL to run the PostgreSQL deletion tests")


@pytest.fixture
def instances(tmp_path, monkeypatch, fresh_db):
    """(instance 1, instance 2, the bucket, its prefix): two app instances, each with its own engine and connection
    pool, on one PostgreSQL database and one S3 bucket. A database of the test's own: other tests' learners, on a
    shared one, would hold the same media."""
    fake = FakeS3()
    monkeypatch.setattr(container_module, "S3ObjectStore",
                        lambda config: S3ObjectStore(config, transport=fake.transport()))
    prefix = new_id("pg")  # this test's objects only, in a database other tests share

    def instance(n: int):
        return build_container(
            Settings(data_dir=tmp_path / f"data{n}", database_url=fresh_db, log_json=False, object_store="s3",
                     s3_bucket=fake.bucket, s3_endpoint_url=f"https://{fake.host}", s3_access_key_id=fake.access_key,
                     s3_secret_access_key=fake.secret_key, s3_prefix=prefix),
            llm_providers={"mock": MockLLMProvider(default_responders())})

    first, second = instance(1), instance(2)
    try:
        yield first, second, fake, prefix
    finally:
        first.close()
        second.close()


def lesson(c, client, learner_id: str, *, other=None) -> tuple[str, int]:
    """Make a lesson on `c`'s worker, answering its diagnostic. A task paused meanwhile is resumed through `other` (the
    other instance's client). Returns (the task, how many times it was resumed)."""
    client.put(f"/learners/{learner_id}", json=LEARNER["profile"])
    created = client.post("/tasks", json={"request": LEARNER["request"], "learner_id": learner_id})
    assert created.status_code == 202, created.text
    task_id, resumed = created.json()["task_id"], 0
    drain(c)
    for _ in range(50):
        task = client.get(f"/tasks/{task_id}").json()
        if task["status"] == "COMPLETED":
            return task_id, resumed
        if task["status"] == "PAUSED":
            assert (other or client).post(f"/tasks/{task_id}/resume").status_code == 202
            resumed += 1
        else:
            assert task["status"] == "WAITING", task
            sheet = task["waiting"]["prompt"]
            key = ANSWERS["rounds"][sheet["round_number"] - 1]
            answers = [{"question_id": q["question_id"], "answer": key[q["concept_id"]]} for q in sheet["questions"]]
            assert client.post(f"/tasks/{task_id}/answers", json={"answers": answers}).status_code == 202
        drain(c)
    raise AssertionError(f"the lesson never completed: {task}")


def holders(c, uri: str) -> set[str]:
    with c._sessions() as s:
        return set(s.scalars(select(ObjectHoldRow.owner_id).where(ObjectHoldRow.uri == uri)))


def dangling(c, task_ids) -> list[str]:
    """The tasks' artifact rows whose stored object is missing (none may be, ever)."""
    with c._sessions() as s:
        uris = {json.loads(body)["uri"] for body in s.scalars(select(ArtifactRow.body)
                                                              .where(ArtifactRow.task_id.in_(list(task_ids))))}
    return sorted(u for u in uris if not c.artifacts.object_exists(u))


def whole(c, client, task_id: str, *, serves: bool = True) -> None:
    """The lesson is complete and every file of it is stored with the bytes its row names, and (`serves`) its media
    serve."""
    assert client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"
    for a in c.artifacts.list_for_task(task_id):
        assert c.artifacts.verify(a) is None, (a.name, c.artifacts.verify(a))
    served = [a for a in client.get(f"/tasks/{task_id}/artifacts").json()
              if client.get(f"/artifacts/{a['artifact_id']}/content").status_code == 200]
    assert served or not serves
    assert client.get(f"/lessons/{c.artifacts.find(task_id, 'lesson').artifact_id}/reading").status_code == 200
    assert dangling(c, [task_id]) == []


def wait_for_an_object_lock_waiter(url: str, timeout: float = 20.0) -> None:
    """Until some session waits on an advisory lock (the object's: nothing else here takes one exclusively)."""
    engine = create_engine(url)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with engine.connect() as conn:
                if conn.execute(text("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                                     "AND wait_event = 'advisory' AND datname = current_database()")).scalar():
                    return
            time.sleep(0.05)
        raise AssertionError("nothing waited for the object lock")
    finally:
        engine.dispose()


class Background(threading.Thread):
    def __init__(self, call) -> None:
        super().__init__(daemon=True)
        self._call, self.result, self.error = call, None, None
        self.start()

    def outcome(self):
        self.join(timeout=60)
        assert not self.is_alive(), "the deletion never finished (a lock wait that never ends)"
        if self.error is not None:
            raise self.error
        return self.result

    def run(self) -> None:
        try:
            self.result = self._call()
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            self.error = exc


@pytest.mark.parametrize("order", ["reuse first", "deletion first"])
def test_a_deletion_on_one_instance_racing_a_reuse_of_shared_media_on_another(instances, fresh_db, monkeypatch,
                                                                            order) -> None:
    one_c, two_c, fake, prefix = instances
    first, second = new_id("pgshare1"), new_id("pgshare2")
    with client_for(one_c, first) as one, client_for(one_c, second) as two, client_for(two_c, second) as two_there:
        first_task, _ = lesson(one_c, one, first)
        mine = {a.uri for a in one_c.artifacts.list_for_task(first_task) if a.uri.startswith("object:objects/")}
        only_mine = {u for u in mine if holders(one_c, u) == {first_task}}  # deleted by the deletion, unless held
        assert only_mine, "the first lesson stored no content-addressed media"
        deletion, raced, deleted = [], [], []
        real_delete = two_c.artifacts.delete_object
        two_c.artifacts.delete_object = lambda uri: (deleted.append(uri), real_delete(uri))[1]

        pausing = []

        def pause_second_lesson() -> None:
            """Ask, on the other instance, for the second lesson to pause (it is resumed there once the race is over).
            In a thread of its own: the request waits for the task row, which the hold's transaction may have locked."""
            [task] = one_c.task_service.list_for_learner(second)
            pausing.append(Background(lambda: two_c.task_service.pause(task.task_id)))

        if order == "reuse first":
            real_hold = repositories._hold

            def hold_then_let_the_deletion_in(s, uri, owner_id, now) -> None:
                real_hold(s, uri, owner_id, now)
                if not deletion and uri in only_mine:  # the hold is written, under the object lock, not committed
                    raced.append(uri)
                    deletion.append(Background(lambda: two_c.learner_deletion.delete(first)))
                    wait_for_an_object_lock_waiter(fresh_db)  # the purge, on the other instance, waits for this commit
                    pause_second_lesson()

            monkeypatch.setattr(repositories, "_hold", hold_then_let_the_deletion_in)
        else:
            real_hold = repositories.SqlArtifactRepository.hold
            real_referenced = repositories.SqlLearnerDataRepository._referenced
            locked = threading.Event()

            def referenced_while_the_hold_waits(s, uri, *args, **kwargs):
                if raced and uri == raced[0] and sys._getframe(1).f_code.co_name == "_purge_one":
                    locked.set()  # the purge holds the object lock
                    wait_for_an_object_lock_waiter(fresh_db)  # and the second lesson's hold waits for it
                return real_referenced(s, uri, *args, **kwargs)

            def let_the_deletion_in_then_hold(self, uri, owner_id) -> None:
                if not deletion and uri in only_mine:
                    raced.append(uri)
                    deletion.append(Background(lambda: two_c.learner_deletion.delete(first)))
                    assert locked.wait(30), (deletion[0].error, "the purge never took the object lock")
                    pause_second_lesson()
                return real_hold(self, uri, owner_id)

            monkeypatch.setattr(repositories.SqlLearnerDataRepository, "_referenced",
                                staticmethod(referenced_while_the_hold_waits))
            monkeypatch.setattr(repositories.SqlArtifactRepository, "hold", let_the_deletion_in_then_hold)

        second_task, resumed = lesson(one_c, two, second, other=two_there)
        assert raced, "the second lesson reused none of the first's media"
        receipt = deletion[0].outcome()
        pausing[0].outcome()
        assert receipt.records["learners"] == 1 and receipt.objects_pending == 0, receipt
        if order == "reuse first":
            assert raced[0] not in deleted and receipt.objects_kept_shared > 0  # it saw the committed hold
        else:
            assert raced[0] in deleted  # gone with the first learner, then written afresh by the second lesson
        assert resumed, "the second lesson was not paused and resumed"
        monkeypatch.undo()

        whole(one_c, two, second_task)
        whole(two_c, two_there, second_task)  # the other instance serves it too
        assert holders(one_c, raced[0]) == {second_task}
        assert not one_c.learner_deletion.repository.holds_data(first)
        again, _ = lesson(two_c, two_there, second)  # another lesson of theirs reuses the same media
        shared = {a.uri for a in one_c.artifacts.list_for_task(second_task) if a.uri.startswith("object:objects/")}
        assert shared & {a.uri for a in two_c.artifacts.list_for_task(again)}
        whole(one_c, two, again, serves=False)  # every file stored, with the bytes its row names
        assert dangling(one_c, [second_task, again]) == []

        assert two.delete(f"/learners/{second}").status_code == 200
    assert [k for k in fake.objects if k.startswith(f"{prefix}/") and not k.startswith(f"{prefix}/records/")] == []
