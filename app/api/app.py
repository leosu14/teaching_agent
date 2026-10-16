"""FastAPI application factory. Routes only validate input, call services, and shape responses.

Every router except `system.public` (/health) is mounted behind the authentication dependency, so a new route is
protected by default; routes that touch learner data also check ownership (app/api/authorization.py)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.auth import AuthenticationFailed, Authenticator, authenticator_from_settings
from app.api.authorization import Forbidden, ResourceNotFound, principal
from app.api.routes import assessment, curriculum, learners, learning_cycles, system, tasks, teaching
from app.config.routing import ConfigError
from app.config.settings import Settings
from app.learner.frameworks import UnknownFramework
from app.learner.memory import UnknownGoal, UnknownLearner
from app.runtime.orchestrator.orchestrator import InvalidInput
from app.runtime.tasks.state_machine import InvalidTransition
from app.services.assessment import (
    AssessmentItemNotFound,
    AttemptConflict,
    AttemptNotFound,
    InvalidAssessmentRequest,
    ItemConflict,
)
from app.services.container import Container, build_container
from app.services.curriculum import InvalidGoal
from app.services.learning_cycles import CycleConflict, CycleNotFound, InvalidCycleRequest, InvalidCycleTransition
from app.services.teaching import (
    InvalidSessionTransition,
    InvalidTeachingRequest,
    SessionConflict,
    TeacherUnavailable,
    TeachingSessionNotFound,
)
from app.storage.repositories import NotFound


def create_app(container: Container | None = None, *, authenticator: Authenticator | None = None) -> FastAPI:
    """`authenticator` defaults to the one the settings configure (TA_AUTH_MODE); a configuration that cannot
    authenticate anyone raises ConfigError here, so the API never starts open."""
    if authenticator is None:
        authenticator = authenticator_from_settings(container.settings if container is not None else Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = container is None
        app.state.container = container or build_container()
        yield
        if owned:
            app.state.container.close()

    app = FastAPI(title="Teaching Agent", version="0.1.0", lifespan=lifespan)
    if container is not None:
        app.state.container = container
    app.state.authenticator = authenticator

    for exc_type, status in ((NotFound, 404), (UnknownLearner, 404), (UnknownGoal, 404), (InvalidTransition, 409),
                             (InvalidInput, 422), (UnknownFramework, 422), (InvalidGoal, 422), (ConfigError, 500),
                             (TeachingSessionNotFound, 404), (InvalidSessionTransition, 409), (SessionConflict, 409),
                             (InvalidTeachingRequest, 422), (TeacherUnavailable, 503),
                             (AssessmentItemNotFound, 404), (AttemptNotFound, 404), (AttemptConflict, 409),
                             (ItemConflict, 409), (InvalidAssessmentRequest, 422),
                             (CycleNotFound, 404), (CycleConflict, 409), (InvalidCycleTransition, 409),
                             (InvalidCycleRequest, 422)):
        app.add_exception_handler(exc_type, _handler(status))
    app.add_exception_handler(AuthenticationFailed, _unauthenticated)
    app.add_exception_handler(Forbidden, _handler(403))
    app.add_exception_handler(ResourceNotFound, _handler(404, error="NotFound"))

    app.include_router(system.public)
    for module in (system, tasks, learners, curriculum, teaching, assessment, learning_cycles):
        app.include_router(module.router, dependencies=[Depends(principal)])
    return app


def _handler(status: int, *, error: str | None = None):
    async def handle(_request: Request, exc: Exception) -> JSONResponse:
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return JSONResponse(status_code=status, content={"error": error or type(exc).__name__, "detail": str(detail)})
    return handle


async def _unauthenticated(_request: Request, _exc: Exception) -> JSONResponse:
    """One answer for every authentication failure: why it failed is not the caller's business."""
    return JSONResponse(status_code=401, content={"error": "Unauthenticated", "detail": "authentication required"},
                        headers={"WWW-Authenticate": "Bearer"})
