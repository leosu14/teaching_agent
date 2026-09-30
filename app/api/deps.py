"""Request-scoped access to application services."""

from __future__ import annotations

from fastapi import Request

from app.services.container import Container


def container(request: Request) -> Container:
    return request.app.state.container
