"""Issue #54: the task and learning-cycle progress streams over real HTTP (uvicorn on a free port: the TestClient
buffers a streamed body until it ends, so it cannot show frames arriving while work runs).

Covered: frames and their order, the cursor across reconnects with both sources (the cycle's and its lesson task's)
including a reconnect across a step change and a source absent from the cursor, no repeated progress after `end`,
termination (task answer POST, cycle response POST with `after_version`, a refused POST, the maximum lifetime),
access loss (learner deletion, roster removal, a failing poll) with no byte after it and a 404 on reconnect, and the
limits and errors (400, 404, 429)."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from app.api.app import create_app
from app.api.auth import TEACHER
from app.config.settings import Settings
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.schemas.learning_cycle import LearnerPrompt
from app.services.container import build_container
from app.services.progress_stream import StreamCursor
from tests import sse
from tests.auth import TestAuth, drain
from tests.conftest import ANSWERS, add_demo_learner
from tests.conftest import LEARNER as DEMO
from tests.learning_cycle_fixtures import LEARNER, ScriptedLearner

FAST = {"stream_settle_seconds": 0.2, "stream_poll_seconds": 0.05}


class Api:
    """A live server over a container, and an HTTP client for one principal."""

    def __init__(self, container, *learners: str, auth: TestAuth | None = None, **principal) -> None:
        self.auth = auth or TestAuth()
        self.container = container
        self.served = sse.serve(create_app(container, authenticator=self.auth.authenticator))
        self.headers = self.auth.headers(*learners, **principal)
        self.http = httpx.Client(base_url=self.served.url, headers=self.headers, timeout=300)

    def stream(self, path: str, *, cursor: str | None = None, headers: dict | None = None) -> sse.LiveStream:
        return sse.LiveStream(self.served.url, path, headers or self.headers, cursor=cursor)

    def close(self) -> None:
        self.http.close()
        self.served.stop()


@pytest.fixture
def demo(tmp_path):
    c = build_container(Settings(data_dir=tmp_path / "data", log_json=False, **FAST),
                        llm_providers={"mock": MockLLMProvider(default_responders())})
    add_demo_learner(c)
    api = Api(c, DEMO["learner_id"])
    yield api
    api.close()
    c.close()


@pytest.fixture
def cycle_api(cycle_env_at):
    made = []

    def open_at(stage: str, **overrides) -> Api:
        env = cycle_env_at(stage, **{**FAST, **overrides})
        api = Api(env.container, LEARNER)
        made.append(api)
        return api

    yield open_at
    for api in made:
        api.close()


def background(fn, *args):
    out: dict = {}

    def target() -> None:
        try:
            out["value"] = fn(*args)
        except BaseException as exc:  # re-raised by join()
            out["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    def join(timeout: float = 300):
        thread.join(timeout)
        if "error" in out:
            raise out["error"]
        return out.get("value")

    return join


def sheet_answers(task: dict) -> dict:
    sheet = task["waiting"]["prompt"]
    key = ANSWERS["rounds"][sheet["round_number"] - 1]
    return {"answers": [{"question_id": q["question_id"], "answer": key.get(q["concept_id"], "")}
                        for q in sheet["questions"]]}


def stored_ids(container, key: str) -> list[str]:
    """The stored events of a task, or of a learning cycle, in `seq` order."""
    if container.learning_cycle_service.repository.get(key) is not None:
        return [e.event_id for e in container.learning_cycle_service.events(key)]
    return [e.event_id for e in container.task_service.events(key)]


# --- the task stream -------------------------------------------------------------------------------------------------


def test_a_task_stream_follows_the_worker_to_each_stop_and_resumes_without_repeats(demo) -> None:
    created = demo.http.post("/tasks", json={"request": DEMO["request"], "learner_id": DEMO["learner_id"]})
    assert created.status_code == 202
    tid = created.json()["task_id"]
    stream = demo.stream(f"/tasks/{tid}/stream")
    assert stream.status == 200 and stream.headers["content-type"].startswith("text/event-stream")
    assert stream.headers["x-accel-buffering"] == "no" and "no-cache" in stream.headers["cache-control"]
    first = stream.next()
    assert first["event"] == "status" and first["data"]["job"]["status"] == "QUEUED" and "id" not in first
    drain(demo.container)  # the worker runs the job while the stream is open
    frames = [first] + stream.rest()
    end = frames[-1]
    assert end["event"] == "end" and end["data"]["status"] in ("WAITING", "COMPLETED")
    assert all(f["data"]["source"] == f"task:{tid}" for f in frames if f.get("event") == "progress")
    shown = sse.progress_ids(frames)
    assert shown == stored_ids(demo.container, tid)  # every stored event once, in order
    assert all(f.get("id") for f in frames if f.get("event") in ("progress", "end"))
    cursor = end["id"]
    task = end["data"]
    while task["status"] == "WAITING":
        posted = demo.http.post(f"/tasks/{tid}/answers", json=sheet_answers(task))
        assert posted.status_code == 202
        stream = demo.stream(f"/tasks/{tid}/stream", cursor=cursor)
        drain(demo.container)
        frames = stream.rest()
        assert frames[-1]["event"] == "end"
        new = sse.progress_ids(frames)
        assert not set(new) & set(shown)  # resumed after the cursor: no repeated progress
        shown += new
        cursor, task = frames[-1]["id"], frames[-1]["data"]
    assert task["status"] == "COMPLETED"
    assert shown == stored_ids(demo.container, tid)
    again = demo.stream(f"/tasks/{tid}/stream", cursor=cursor).rest()  # reopened after `end`
    assert [f["event"] for f in again] == ["status", "end"]


def test_stream_errors_and_limits(demo, tmp_path) -> None:
    created = demo.http.post("/tasks", json={"request": DEMO["request"], "learner_id": DEMO["learner_id"]})
    tid = created.json()["task_id"]
    for bad in ("v9.e30", "v1.!!", "nonsense", "v1." + "A" * 5000):
        r = demo.http.get(f"/tasks/{tid}/stream", headers={"Last-Event-ID": bad})
        assert r.status_code == 400 and r.json()["error"] == "InvalidCursor"
    other = demo.auth.headers("someone-else")
    for path in (f"/tasks/{tid}/stream", "/tasks/does-not-exist/stream", "/learning-cycles/nope/stream"):
        r = demo.http.get(path, headers=other)
        assert r.status_code == 404 and r.json() == {"error": "NotFound", "detail": "not found"}
    assert demo.http.get(f"/tasks/{tid}/stream", headers={"Authorization": "Bearer nope"}).status_code == 401
    # A queued task never stops on its own: the streams stay open, and the cap counts them.
    capped = build_container(Settings(data_dir=tmp_path / "capped", log_json=False, stream_max_per_principal=2,
                                      stream_max_streams=3, **FAST),
                             llm_providers={"mock": MockLLMProvider(default_responders())})
    add_demo_learner(capped)
    api = Api(capped, DEMO["learner_id"], auth=demo.auth)
    try:
        queued = api.http.post("/tasks", json={"request": DEMO["request"], "learner_id": DEMO["learner_id"]})
        tid = queued.json()["task_id"]
        open_ = [api.stream(f"/tasks/{tid}/stream") for _ in range(2)]
        assert all(s.next()["event"] == "status" for s in open_)
        refused = api.http.get(f"/tasks/{tid}/stream")
        assert refused.status_code == 429 and refused.json()["error"] == "TooManyStreams"
        guardian = demo.auth.headers(DEMO["learner_id"], user_id="guardian")  # another principal: its own cap
        third = api.stream(f"/tasks/{tid}/stream", headers=guardian)
        assert third.next()["event"] == "status"
        assert api.http.get(f"/tasks/{tid}/stream", headers=demo.auth.headers(
            DEMO["learner_id"], user_id="fourth")).status_code == 429  # the process-wide cap
        open_[0].close()
        deadline = time.monotonic() + 10
        while capped.progress_streams.slots.open_count() > 2 and time.monotonic() < deadline:
            time.sleep(0.05)  # a dropped client is noticed at the stream's next write (a heartbeat at the latest)
        assert capped.progress_streams.slots.open_count() <= 2
    finally:
        for s in open_ + [third]:
            s.close()
        api.close()
        capped.close()


# --- the cycle stream ------------------------------------------------------------------------------------------------


def respond(api: Api, view: dict, learner: ScriptedLearner):
    body = learner.respond(LearnerPrompt.model_validate(view["prompt"])).model_dump(mode="json", exclude_none=True)
    return background(lambda: api.http.post(f"/learning-cycles/{view['cycle_id']}/responses", json=body))


def test_cycle_stream_reconnects_across_both_sources_and_a_step_change(cycle_api) -> None:
    api = cycle_api("diagnostic", stream_settle_seconds=0.3, stream_poll_seconds=0.02)
    view = api.http.get(f"/learners/{LEARNER}/learning-cycles").json()[0]
    check_reconnects(api, api.http.get(f"/learning-cycles/{view['cycle_id']}").json())


def check_reconnects(api: Api, view: dict) -> None:
    """From a cycle WAITING on its lesson's diagnostic: answer it (the POST generates the lesson and moves the cycle
    to the session step) while the stream is dropped and resumed at three points. Also run on PostgreSQL."""
    assert view["status"] == "WAITING" and view["prompt"]["kind"] == "DIAGNOSTIC_QUESTIONS"
    cid, version = view["cycle_id"], view["version"]
    lesson = next(s["child_id"] for s in view["steps"] if s["kind"] == "LESSON")
    path = f"/learning-cycles/{cid}/stream?after_version={version}"
    seen: list[dict] = []

    # 1: disconnect before any new child event (only what was stored before the POST has been read)
    first = api.stream(path)
    assert first.next()["event"] == "status"
    first.until(lambda f: f.get("event") == "progress")
    before_post = first.cursor()
    seen += first.received
    first.close()
    positions = StreamCursor.decode(before_post)
    assert set(positions) <= {"cycle", f"task:{lesson}"}

    # 2: the response POST generates the lesson; reconnect, then drop in the middle of the lesson's events
    posted = respond(api, view, ScriptedLearner(prefix="d"))
    middle = api.stream(path, cursor=before_post)
    middle.until(lambda f: f.get("event") == "progress" and f["data"]["source"] == f"task:{lesson}"
                 and f["data"]["event"]["event_id"] not in {g["data"]["event"]["event_id"] for g in seen
                                                             if g.get("event") == "progress"})
    for _ in range(2):
        middle.until(lambda f: f.get("event") == "progress")
    seen += middle.received
    mid_cursor = middle.cursor()
    middle.close()

    # 3: reconnect from there and read to the end: the cycle moved on to the session step (a step change)
    last = api.stream(path, cursor=mid_cursor)
    frames = last.rest()
    seen += frames
    reply = posted()
    assert reply.status_code == 200, reply.text
    end = frames[-1]
    assert end["event"] == "end" and end["data"]["status"] == "WAITING" and end["data"]["version"] > version
    assert end["data"]["prompt"]["kind"] == "SESSION_QUESTION"
    assert any(f.get("event") == "status" and f["data"]["status"] == "RUNNING" for f in seen)

    # every stored event of both sources was received at least once, in per-source seq order per connection
    progress = [f["data"] for f in seen if f.get("event") == "progress"]
    stored = {"cycle": stored_ids(api.container, cid), f"task:{lesson}": stored_ids(api.container, lesson)}
    assert {p["event"]["event_id"] for p in progress} == set(stored["cycle"]) | set(stored[f"task:{lesson}"])
    for connection in (first.received, middle.received, frames):
        for source, ids in stored.items():
            got = [f["data"]["event"]["event_id"] for f in connection
                   if f.get("event") == "progress" and f["data"]["source"] == source]
            assert got == [i for i in ids if i in got]  # in the stored (seq) order
    final = StreamCursor.decode(end["id"])
    assert set(final) == {"cycle", f"task:{lesson}"}

    # 4: a cursor issued before the child existed names only `cycle`: the child is read from its start
    only_cycle = sse.LiveStream(api.served.url, path, api.headers,
                                cursor=StreamCursor.encode({"cycle": positions.get("cycle", 0)}))
    rest = only_cycle.rest()
    lesson_ids = [p["event"]["event_id"] for p in (f["data"] for f in rest if f.get("event") == "progress")
                  if p["source"] == f"task:{lesson}"]
    assert lesson_ids == stored[f"task:{lesson}"]

    # 5: reopened after `end` with its cursor (nothing in flight, no after_version): no repeated progress
    reopened = api.stream(f"/learning-cycles/{cid}/stream", cursor=end["id"]).rest()
    assert [f["event"] for f in reopened] == ["status", "end"]


def test_a_refused_post_does_not_end_the_stream_and_the_lifetime_is_the_backstop(cycle_api) -> None:
    api = cycle_api("session", stream_max_seconds=1.5, stream_heartbeat_seconds=0.3)
    view = api.http.get(f"/learners/{LEARNER}/learning-cycles").json()[0]
    cid, version = view["cycle_id"], view["version"]
    stream = api.stream(f"/learning-cycles/{cid}/stream?after_version={version}")
    refused = api.http.post(f"/learning-cycles/{cid}/responses", json={"client_response_id": "x",
                                                                       "answers": [{"question_id": "q", "answer": "a"}]})
    assert refused.status_code == 422
    frames = stream.rest()
    kinds = [f.get("event") or "comment" for f in frames]
    assert "end" not in kinds and kinds[-1] == "reconnect" and "comment" in kinds
    assert StreamCursor.decode(frames[-1]["id"])  # the reconnect frame carries the cursor
    # Opened without after_version, the same WAITING state is a stop (once the stored events are delivered).
    plain = [f["event"] for f in api.stream(f"/learning-cycles/{cid}/stream").rest()]
    assert plain[0] == "status" and plain[-1] == "end" and set(plain[1:-1]) == {"progress"}


def test_a_running_cycle_that_nobody_drives_keeps_the_stream_open(cycle_api) -> None:
    api = cycle_api("session", stream_max_seconds=1.0)
    [cycle] = api.container.learning_cycle_service.repository.for_learner(LEARNER)
    from app.curriculum import cycle as engine
    from app.schemas.common import utcnow

    running = engine.receive(cycle, utcnow())  # RUNNING with no lease: the gap between two requests
    api.container.learning_cycle_service.repository.apply(running, expected_version=cycle.version)
    original = api.container.learning_cycle_service.read

    async def no_fold(cycle_id):  # the session still waits: a read would fold it back; keep it RUNNING
        record = api.container.learning_cycle_service.cycle(cycle_id)
        return record, await api.container.learning_cycle_service._view(record)

    api.container.learning_cycle_service.read = no_fold
    try:
        frames = api.stream(f"/learning-cycles/{cycle.cycle_id}/stream").rest()
    finally:
        api.container.learning_cycle_service.read = original
    assert frames[0]["data"]["status"] == "RUNNING" and frames[0]["data"]["busy_until"] is None
    assert [f.get("event") for f in frames if f.get("event")][-1] == "reconnect"  # never `end`


# --- access loss -----------------------------------------------------------------------------------------------------


def wait_closed(stream: sse.LiveStream, timeout: float = 10) -> None:
    assert stream.closed.wait(timeout), "the stream did not close"


def test_learner_deletion_closes_the_stream_with_nothing_more(cycle_api) -> None:
    api = cycle_api("session")
    view = api.http.get(f"/learners/{LEARNER}/learning-cycles").json()[0]
    cid = view["cycle_id"]
    lesson = next(s["child_id"] for s in view["steps"] if s["kind"] == "LESSON")
    cycle_stream = api.stream(f"/learning-cycles/{cid}/stream?after_version={view['version']}")
    task_stream = api.stream(f"/tasks/{lesson}/stream")
    cycle_stream.until(lambda f: f.get("event") == "progress")
    time.sleep(1.0)  # every settled event delivered: the cycle stream is idle (nothing changes, no heartbeat yet)
    task_frames = task_stream.rest()  # the lesson task is COMPLETED: its stream ended normally
    assert task_frames[-1]["event"] == "end"
    before = len(cycle_stream.raw)
    deleted = api.http.delete(f"/learners/{LEARNER}")
    assert deleted.status_code == 200, deleted.text
    wait_closed(cycle_stream)
    assert bytes(cycle_stream.raw[before:]) == b""  # no end, no status, no frame naming the cycle
    for path in (f"/learning-cycles/{cid}/stream", f"/tasks/{lesson}/stream", f"/learning-cycles/{cid}"):
        r = api.http.get(path)
        assert r.status_code == 404 and r.json() == {"error": "NotFound", "detail": "not found"}, path


def test_removal_from_the_teachers_roster_closes_the_teachers_stream(cycle_api) -> None:
    api = cycle_api("session")
    teacher = api.auth.headers(user_id="teacher-1", roles=(TEACHER,))
    made = api.http.post("/classes", json={"name": "Spanish"}, headers=teacher)
    assert made.status_code == 201, made.text
    joined = api.http.post(f"/learners/{LEARNER}/classes", json={"join_code": made.json()["join_code"]})
    assert joined.status_code == 201, joined.text
    view = api.http.get(f"/learners/{LEARNER}/learning-cycles").json()[0]
    path = f"/learning-cycles/{view['cycle_id']}/stream?after_version={view['version']}"
    stream = api.stream(path, headers=teacher)
    assert stream.status == 200 and stream.next()["event"] == "status"
    stream.until(lambda f: f.get("event") == "progress")
    time.sleep(1.0)
    before = len(stream.raw)
    removed = api.http.delete(f"/classes/{made.json()['class_id']}/learners/{LEARNER}", headers=teacher)
    assert removed.status_code in (200, 204), removed.text
    wait_closed(stream)
    assert bytes(stream.raw[before:]) == b""
    assert api.http.get(path, headers=teacher).status_code == 404
    own = api.stream(path)  # the learner's own stream still opens
    assert own.status == 200 and own.next()["event"] == "status"
    own.close()


def test_a_failing_poll_closes_the_stream_without_a_frame(cycle_api, caplog) -> None:
    api = cycle_api("session")
    view = api.http.get(f"/learners/{LEARNER}/learning-cycles").json()[0]
    stream = api.stream(f"/learning-cycles/{view['cycle_id']}/stream?after_version={view['version']}")
    stream.until(lambda f: f.get("event") == "progress")
    time.sleep(1.0)
    before = len(stream.raw)
    service = api.container.learning_cycle_service

    async def broken(cycle_id):
        raise RuntimeError("SECRET-DETAIL from a failed read")

    service.read = broken
    with caplog.at_level("INFO"):
        wait_closed(stream)
    assert bytes(stream.raw[before:]) == b""
    assert "SECRET-DETAIL" not in caplog.text and "RuntimeError" in caplog.text
