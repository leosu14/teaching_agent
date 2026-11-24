"""Development only: live reload of the student app and the teacher dashboard (TA_DEV_LIVE_RELOAD=true).

Off by default, and refused under TA_DEPLOYMENT=production (`Settings.deployment_problems`). When it is off, nothing in
this module is mounted: the apps are served by `web.router` exactly as in production, from the bytes read at startup.

When it is on, `create_app` mounts `router(...)` ahead of `web.router`, and the same files are served instead:

- read from disk on every request, so an edit to app/api/web/ is served at once (no server restart, so nothing
  interrupts a running lesson or the embedded worker);
- each HTML page gets one extra `<script src="dev-reload.js">`, carrying the version of the files it was served with.
  The script is a separate same-origin file, so the page keeps the same Content-Security-Policy and headers;
- the script polls `/app/dev-reload/version` (a hash of the served files, nothing else) and reloads the page once the
  server serves a different version. `location.reload()` keeps the `#/` route and the tab's sign-in. A server that is
  restarting (a Python edit under `uvicorn --reload`) only makes the polls fail until it is back; the version is a
  hash of the files' contents, so a restart alone never reloads the page, and a reloaded page carries the new
  version, so there is no reload loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from app.api.routes import web

SCRIPT_NAME = "dev-reload.js"
VERSION_PATH = "/app/dev-reload/version"
POLL_MS = 700  # how often an open page asks for the version
RETRY_MS = 1500  # how often while the server does not answer (restarting)

# Dependency-free, no DOM writes: it only reads its own data-version attribute and calls location.reload().
SCRIPT = f"""\
/* Development live reload (TA_DEV_LIVE_RELOAD=true). Never served otherwise. */
(() => {{
  "use strict";
  const loaded = document.currentScript && document.currentScript.dataset.version;
  if (!loaded) return;
  const url = new URL("dev-reload/version", document.baseURI).pathname;
  async function poll() {{
    let wait = {POLL_MS};
    try {{
      const r = await fetch(url, {{ cache: "no-store", credentials: "omit" }});
      if (r.ok) {{
        const body = await r.json();
        if (body.version && body.version !== loaded) {{
          location.reload();
          return;
        }}
      }} else {{
        wait = {RETRY_MS};
      }}
    }} catch (err) {{
      wait = {RETRY_MS};  // the server is restarting: keep the page and try again
    }}
    setTimeout(poll, wait);
  }}
  setTimeout(poll, {POLL_MS});
}})();
"""

DEV_HEADERS = {**web.HEADERS, "Cache-Control": "no-store"}


def version(web_dir: Path) -> str:
    """A hash of every served file's contents: it changes exactly when what the apps are served changes."""
    digest = hashlib.sha256()
    for name in sorted(web.FILES):
        digest.update(name.encode() + b"\0")
        try:
            digest.update((web_dir / name).read_bytes())
        except OSError:  # mid-save (an editor replacing the file): the next poll sees it
            digest.update(b"\0missing")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def inject(html: bytes, file_version: str) -> bytes:
    tag = f'<script src="{SCRIPT_NAME}" data-version="{file_version}" defer></script>\n'.encode()
    head = html.lower().rfind(b"</head>")
    return html[:head] + tag + html[head:] if head >= 0 else html + tag


class _QuietPolls(logging.Filter):
    """Keeps uvicorn's access log readable: the version polls are not logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(isinstance(a, str) and a.startswith(VERSION_PATH) for a in (record.args or ()))


def quiet_access_log() -> None:
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _QuietPolls) for f in access.filters):
        access.addFilter(_QuietPolls())


def router(web_dir: Path | None = None) -> APIRouter:
    """The development routes. They shadow `web.router`'s file routes and serve the same files (`web.FILES` only,
    never a path from the request, from `web.WEB_DIR` unless told otherwise), with the same headers."""
    web_dir = web_dir or web.WEB_DIR
    dev = APIRouter(tags=["web"], include_in_schema=False)

    def serve(name: str) -> Response:
        body = (web_dir / name).read_bytes()
        if name.endswith(".html"):
            body = inject(body, version(web_dir))
        return Response(content=body, media_type=web.FILES[name], headers=DEV_HEADERS)

    @dev.get(VERSION_PATH, include_in_schema=False)
    def file_version() -> Response:
        return Response(content=json.dumps({"version": version(web_dir)}), media_type="application/json",
                        headers=DEV_HEADERS)

    @dev.get("/app/", include_in_schema=False)
    def dev_index() -> Response:
        return serve("index.html")

    @dev.get("/app/{name}", include_in_schema=False)
    def dev_file(name: str) -> Response:
        if name == SCRIPT_NAME:
            return Response(content=SCRIPT, media_type="text/javascript; charset=utf-8", headers=DEV_HEADERS)
        if name not in web.FILES:
            return JSONResponse(status_code=404, content={"error": "NotFound", "detail": "not found"})
        return serve(name)

    return dev
