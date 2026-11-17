"""Issue #54: the progress stream through the identity gateway, both on live servers (uvicorn), as deployed. Frames
arrive as they are written (nothing buffers them), the heartbeat keeps a stream open past a short upstream read
timeout, the gateway passes `Last-Event-ID` through, and a client that goes away frees the API's stream slot."""

from __future__ import annotations

import time

import pytest

from app.api.app import create_app
from app.api.auth import GatewayAuthenticator
from app.config.settings import Settings
from app.gateway.app import create_gateway
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.mock_responders import default_responders
from app.services.container import build_container
from tests import sse
from tests.auth import drain
from tests.conftest import LEARNER, add_demo_learner
from tests.gateway_fixtures import GATEWAY_SECRET, IdP, gateway_settings

LID = LEARNER["learner_id"]


@pytest.fixture
def deployed(tmp_path):
    idp = IdP()
    jwks = tmp_path / "jwks.json"
    idp.write_jwks(jwks)
    container = build_container(Settings(data_dir=tmp_path / "data", log_json=False, stream_settle_seconds=0.2,
                                         stream_poll_seconds=0.05, stream_heartbeat_seconds=0.4),
                                llm_providers={"mock": MockLLMProvider(default_responders())})
    add_demo_learner(container)
    api = sse.serve(create_app(container, authenticator=GatewayAuthenticator(GATEWAY_SECRET)))
    # An upstream read timeout far below the stream's life: only the heartbeat keeps it open.
    gateway = sse.serve(create_gateway(gateway_settings(jwks, upstream_url=api.url, timeout_seconds=1.5)))
    yield container, gateway.url, {"Authorization": f"Bearer {idp.token(LID)}"}
    gateway.stop()
    api.stop()
    container.close()


def test_frames_arrive_unbuffered_and_heartbeats_outlive_the_upstream_timeout(deployed) -> None:
    import httpx

    container, url, me = deployed
    with httpx.Client(base_url=url, headers=me, timeout=30) as http:
        tid = http.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}).json()["task_id"]
    stream = sse.LiveStream(url, f"/tasks/{tid}/stream", me)
    assert stream.status == 200 and stream.headers["content-type"].startswith("text/event-stream")
    opened = time.monotonic()
    first = stream.next(timeout=5)
    assert first["event"] == "status" and time.monotonic() - opened < 1.5  # before the response ends
    beats = []
    while time.monotonic() - opened < 4:  # a queued task: no frame but heartbeats, past the 1.5 s timeout
        f = stream.next(timeout=3)
        assert f is not None, "the gateway cut the stream"
        beats.append(f)
    assert any("comment" in f for f in beats) and not stream.closed.is_set()
    drain(container)
    rest = stream.rest()
    assert rest[-1]["event"] == "end" and sse.progress_ids(rest)
    cursor = rest[-1]["id"]
    again = sse.LiveStream(url, f"/tasks/{tid}/stream", me, cursor=cursor).rest()  # Last-Event-ID passes through
    assert [f["event"] for f in again] == ["status", "end"]


def test_a_client_that_goes_away_frees_the_stream_slot(deployed) -> None:
    import httpx

    container, url, me = deployed
    with httpx.Client(base_url=url, headers=me, timeout=30) as http:
        tid = http.post("/tasks", json={"request": LEARNER["request"], "learner_id": LID}).json()["task_id"]
    stream = sse.LiveStream(url, f"/tasks/{tid}/stream", me)
    assert stream.next(timeout=5)["event"] == "status"
    assert container.progress_streams.slots.open_count() == 1
    stream.close()
    deadline = time.monotonic() + 10
    while container.progress_streams.slots.open_count() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert container.progress_streams.slots.open_count() == 0
