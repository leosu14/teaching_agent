"""The student web app: static files only, served without authentication.

The app (app/api/web/) is HTML, CSS and one script with no data in it: everything it shows comes from the
authenticated JSON API, called from the page (same origin). So these routes are public like /health, and nothing
else is: they serve exactly the files listed in `FILES` and never a path from the request. Every response carries a
strict Content-Security-Policy (no inline script, no third-party origin, blob: only for media the page fetched
itself)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, RedirectResponse, Response

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
FILES: dict[str, str] = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}
_CONTENT = {name: (WEB_DIR / name).read_bytes() for name in FILES}

router = APIRouter(tags=["web"], include_in_schema=False)


def _file(name: str) -> Response:
    return Response(content=_CONTENT[name], media_type=FILES[name], headers=HEADERS)


@router.get("/app", include_in_schema=False)
def app_root() -> RedirectResponse:
    return RedirectResponse("/app/", status_code=308)


@router.get("/app/", include_in_schema=False)
def app_index() -> Response:
    return _file("index.html")


@router.get("/app/{name}", include_in_schema=False)
def app_file(name: str) -> Response:
    if name not in FILES:
        return JSONResponse(status_code=404, content={"error": "NotFound", "detail": "not found"})
    return _file(name)
