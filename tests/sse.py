"""Test helpers for the progress streams (issue #54): an SSE parser, a live uvicorn server (the TestClient buffers a
streamed body until it ends) and an incremental stream reader over real HTTP."""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass

import httpx


def parse(text: str) -> list[dict]:
    """Frames of an SSE body: {"event", "id", "data"} (data parsed as JSON), or {"comment": text}."""
    frames = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        frame: dict = {}
        for line in block.split("\n"):
            if line.startswith(":"):
                frame["comment"] = line[1:].strip()
                continue
            key, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            frame[key] = value
        if "data" in frame:
            frame["data"] = json.loads(frame["data"])
        frames.append(frame)
    return frames


def progress_ids(frames: list[dict]) -> list[str]:
    return [f["data"]["event"]["event_id"] for f in frames if f.get("event") == "progress"]


@dataclass
class Served:
    url: str
    server: object
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def serve(app) -> Served:
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the test server did not start")
        time.sleep(0.05)
    return Served(url=f"http://127.0.0.1:{port}", server=server, thread=thread)


class LiveStream:
    """One stream read in a background thread, frame by frame as the bytes arrive. `raw` keeps every byte received,
    so a test can prove nothing at all arrived after a point."""

    def __init__(self, base_url: str, path: str, headers: dict, *, cursor: str | None = None,
                 timeout: float = 60) -> None:
        self.frames: queue.Queue = queue.Queue()
        self.received: list[dict] = []
        self.raw = bytearray()
        self.status: int | None = None
        self.headers: dict = {}
        self.closed = threading.Event()
        self.opened = threading.Event()
        self._stop = threading.Event()
        self._response: httpx.Response | None = None
        h = dict(headers)
        if cursor:
            h["Last-Event-ID"] = cursor
        self._thread = threading.Thread(target=self._read, args=(base_url + path, h, timeout), daemon=True)
        self._thread.start()
        self.opened.wait(timeout)

    def _read(self, url: str, headers: dict, timeout: float) -> None:
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=10)) as client, \
                    client.stream("GET", url, headers=headers) as response:
                self._response = response
                self.status = response.status_code
                self.headers = dict(response.headers)
                self.opened.set()
                buffer = ""
                for chunk in response.iter_raw():
                    self.raw += chunk
                    buffer += chunk.decode("utf-8")
                    while "\n\n" in buffer:
                        block, buffer = buffer.split("\n\n", 1)
                        for f in parse(block + "\n\n"):
                            self.frames.put(f)
                    if self._stop.is_set():
                        return
        except (httpx.HTTPError, OSError, ValueError, RuntimeError):
            pass  # closed by `close()` or by the server
        finally:
            self.opened.set()
            self.closed.set()

    def next(self, timeout: float = 30) -> dict | None:
        """The next frame, or None once the stream closed with nothing more."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                f = self.frames.get(timeout=0.05)
                self.received.append(f)
                return f
            except queue.Empty:
                if self.closed.is_set() and self.frames.empty():
                    return None
                if time.monotonic() > deadline:
                    raise AssertionError(f"no frame within {timeout}s; received {self.received[-3:]}")

    def until(self, predicate, timeout: float = 60) -> dict | None:
        deadline = time.monotonic() + timeout
        while True:
            f = self.next(max(0.1, deadline - time.monotonic()))
            if f is None or predicate(f):
                return f

    def rest(self, timeout: float = 60) -> list[dict]:
        """Every frame until the stream closes."""
        out = []
        while (f := self.next(timeout)) is not None:
            out.append(f)
        return out

    def cursor(self) -> str | None:
        """The last cursor received (what a client sends back as Last-Event-ID)."""
        ids = [f["id"] for f in self.received if "id" in f]
        return ids[-1] if ids else None

    def close(self) -> None:
        """Drop the connection, as a client that goes away does."""
        self._stop.set()
        if self._response is not None:
            try:
                self._response.close()
            except Exception:  # noqa: BLE001 - closing from another thread can race the reader
                pass
        self._thread.join(timeout=5)


def is_event(name: str):
    return lambda f: f.get("event") == name
