"""FastAPI application factory. Routes only validate input, call services, and shape responses.

Every router except `system.public` (/health) and `web` (the student app's static files, /app/) is mounted behind the
authentication dependency, so a new route is protected by default; routes that touch learner data also check
ownership (app/api/authorization.py)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.auth import AuthenticationFailed, Authenticator, authenticator_from_settings
from app.api.authorization import Forbidden, ResourceNotFound, principal
from app.api.routes import (
    artifacts,
    assessment,
    classes,
    curriculum,
    dev_reload,
    learners,
    learning_cycles,
    lessons,
    system,
    tasks,
    teaching,
    web,
)
from app.config.routing import ConfigError
from app.config.settings import Settings
from app.learner.frameworks import UnknownFramework
from app.learner.memory import UnknownGoal, UnknownLearner
from app.runtime.orchestrator.orchestrator import InvalidInput
from app.runtime.tasks.state_machine import InvalidTransition
from app.schemas.lease import LearnerDataGone, LeaseLost, WriteConflict
from app.schemas.privacy import LearnerDataBusy
from app.services.artifact_content import ArtifactContentNotFound, ArtifactContentUnavailable
from app.services.assessment import (
    AssessmentItemNotFound,
    AttemptConflict,
    AttemptNotFound,
    InvalidAssessmentRequest,
    ItemConflict,
)
from app.services.budgets import BudgetExhausted
from app.schemas.assignment import AssignmentConflict, AssignmentNotFound, GoalLocked, InvalidAssignment
from app.services.classes import ClassNotFound
from app.services.container import Container, build_container
from app.services.curriculum import InvalidGoal
from app.services.lessons import LessonNotReady
from app.services.releases import ReleaseNotFound
from app.schemas.review import LessonHeld, OverrideConflict, ReleaseConflict
from app.services.progress_stream import InvalidCursor, StreamSlotsUnavailable, TooManyStreams
from app.services.tasks import JobConflict
from app.services.learning_cycles import CycleConflict, CycleNotFound, InvalidCycleRequest, InvalidCycleTransition
from app.services.teaching import (
    InvalidSessionTransition,
    InvalidTeachingRequest,
    SessionConflict,
    TeacherUnavailable,
    TeachingSessionNotFound,
    TeachingTurnBusy,
)
from app.storage.repositories import NotFound


def create_app(container: Container | None = None, *, authenticator: Authenticator | None = None) -> FastAPI:
    """`authenticator` defaults to the one the settings configure (TA_AUTH_MODE); a configuration that cannot
    authenticate anyone raises ConfigError here, so the API never starts open."""
    settings = container.settings if container is not None else Settings()
    if authenticator is None:
        authenticator = authenticator_from_settings(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = container is None
        app.state.container = container or build_container()
        worker = None
        if app.state.container.settings.worker_embedded:  # development: run the task worker in this process
            stop = asyncio.Event()
            worker = asyncio.create_task(app.state.container.task_worker.run_forever(stop))
        yield
        if worker is not None:
            stop.set()
            await worker
        if owned:
            app.state.container.close()

    app = FastAPI(title="Teaching Agent", version="0.1.0", lifespan=lifespan)
    if container is not None:
        app.state.container = container
    app.state.authenticator = authenticator

    for exc_type, status in ((NotFound, 404), (UnknownLearner, 404), (UnknownGoal, 404), (InvalidTransition, 409),
                             (InvalidInput, 422), (UnknownFramework, 422), (InvalidGoal, 422), (ConfigError, 500),
                             (TeachingSessionNotFound, 404), (InvalidSessionTransition, 409), (SessionConflict, 409),
                             (InvalidTeachingRequest, 422), (TeacherUnavailable, 503), (BudgetExhausted, 429),
                             (AssessmentItemNotFound, 404), (AttemptNotFound, 404), (AttemptConflict, 409),
                             (ItemConflict, 409), (InvalidAssessmentRequest, 422),
                             (CycleNotFound, 404), (CycleConflict, 409), (InvalidCycleTransition, 409),
                             (InvalidCycleRequest, 422), (LeaseLost, 409), (WriteConflict, 409), (JobConflict, 409),
                             (LessonNotReady, 409), (LessonHeld, 409), (ReleaseConflict, 409), (OverrideConflict, 409),
                             (LearnerDataBusy, 409), (InvalidCursor, 400), (TooManyStreams, 429), (StreamSlotsUnavailable, 503),
                             (AssignmentConflict, 409), (InvalidAssignment, 422), (GoalLocked, 409)):
        app.add_exception_handler(exc_type, _handler(status))
    app.add_exception_handler(TeachingTurnBusy, _turn_busy)
    app.add_exception_handler(AuthenticationFailed, _unauthenticated)
    app.add_exception_handler(Forbidden, _handler(403))
    app.add_exception_handler(ResourceNotFound, _handler(404, error="NotFound"))
    app.add_exception_handler(ArtifactContentNotFound, _handler(404, error="NotFound"))
    app.add_exception_handler(ClassNotFound, _handler(404, error="NotFound"))
    app.add_exception_handler(ReleaseNotFound, _handler(404, error="NotFound"))
    app.add_exception_handler(AssignmentNotFound, _handler(404, error="NotFound"))
    app.add_exception_handler(LearnerDataGone, _handler(404, error="NotFound"))
    app.add_exception_handler(ArtifactContentUnavailable, _handler(500, error="ContentUnavailable"))

    app.include_router(system.public)
    if settings.dev_live_reload:  # development only, off by default (app/api/routes/dev_reload.py)
        dev_reload.quiet_access_log()
        app.include_router(dev_reload.router())
    app.include_router(web.router)
    for module in (system, tasks, artifacts, learners, curriculum, teaching, lessons, assessment, learning_cycles,
                   classes):
        app.include_router(module.router, dependencies=[Depends(principal)])
    return app


def _handler(status: int, *, error: str | None = None):
    async def handle(_request: Request, exc: Exception) -> JSONResponse:
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return JSONResponse(status_code=status, content={"error": error or type(exc).__name__, "detail": str(detail)})
    return handle


async def _turn_busy(_request: Request, exc: Exception) -> JSONResponse:
    """409 with Retry-After: another request is producing this session's teacher reply (decision A4, issue #66)."""
    assert isinstance(exc, TeachingTurnBusy)
    return JSONResponse(status_code=409, content={"error": "TeachingTurnBusy", "detail": str(exc)},
                        headers={"Retry-After": str(exc.retry_after)})


async def _unauthenticated(_request: Request, _exc: Exception) -> JSONResponse:
    """One answer for every authentication failure: why it failed is not the caller's business."""
    return JSONResponse(status_code=401, content={"error": "Unauthenticated", "detail": "authentication required"},
                        headers={"WWW-Authenticate": "Bearer"})
