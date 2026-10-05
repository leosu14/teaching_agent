"""FastAPI application factory. Routes only validate input, call services, and shape responses."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import assessment, curriculum, learners, learning_cycles, system, tasks, teaching
from app.config.routing import ConfigError
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


def create_app(container: Container | None = None) -> FastAPI:
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

    for exc_type, status in ((NotFound, 404), (UnknownLearner, 404), (UnknownGoal, 404), (InvalidTransition, 409),
                             (InvalidInput, 422), (UnknownFramework, 422), (InvalidGoal, 422), (ConfigError, 500),
                             (TeachingSessionNotFound, 404), (InvalidSessionTransition, 409), (SessionConflict, 409),
                             (InvalidTeachingRequest, 422), (TeacherUnavailable, 503),
                             (AssessmentItemNotFound, 404), (AttemptNotFound, 404), (AttemptConflict, 409),
                             (ItemConflict, 409), (InvalidAssessmentRequest, 422),
                             (CycleNotFound, 404), (CycleConflict, 409), (InvalidCycleTransition, 409),
                             (InvalidCycleRequest, 422)):
        app.add_exception_handler(exc_type, _handler(status))

    app.include_router(system.router)
    app.include_router(tasks.router)
    app.include_router(learners.router)
    app.include_router(curriculum.router)
    app.include_router(teaching.router)
    app.include_router(assessment.router)
    app.include_router(learning_cycles.router)
    return app


def _handler(status: int):
    async def handle(_request: Request, exc: Exception) -> JSONResponse:
        detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return JSONResponse(status_code=status, content={"error": type(exc).__name__, "detail": str(detail)})
    return handle
