"""Browser tests of the student web app (app/api/web/): a real Chromium drives the page against the real API, served
by uvicorn on a free local port over a test container (mock providers). Skipped unless Playwright and its Chromium are
installed (`pip install -e ".[browser]"` and `python -m playwright install chromium`); CI's `browser` job installs
both and fails when the suite skips."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")


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


@pytest.fixture(scope="module")
def browser(cycle_stages):
    """Chromium for one test module. Sync Playwright keeps an event loop running on this thread while it is open, and
    asyncio.run (used by other tests and fixtures) cannot run meanwhile: so it is closed after the module, and the
    learning-cycle stages (built with asyncio.run) are built before it starts."""
    with playwright_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - Playwright is installed but its browser is not
            pytest.skip(f"Chromium for Playwright is not installed: {exc}")
        yield b
        b.close()
