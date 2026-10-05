"""Interactive teaching sessions on a lesson (opt-in: lessons work exactly as before without one)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.deps import container
from app.schemas.teaching import (
    AnswerResult,
    LearnerInput,
    SessionStarted,
    StartTeachingSession,
    TeachingSessionView,
)
from app.services.container import Container

router = APIRouter(tags=["teaching"])


@router.post("/lessons/{lesson_id}/teaching-session", response_model=SessionStarted, status_code=201)
async def start_session(lesson_id: str, response: Response, body: StartTeachingSession | None = None,
                        c: Container = Depends(container)) -> SessionStarted:
    """`lesson_id`: the LESSON artifact id or the id of the completed lesson task. Objective and action default to
    the lesson's curriculum focus (else its first objective, LEARN). The same idempotency key returns the session."""
    started = await c.teaching_service.start(lesson_id, body or StartTeachingSession())
    if not started.created:
        response.status_code = 200
    return started


@router.get("/teaching-sessions/{session_id}", response_model=TeachingSessionView)
async def get_session(session_id: str, c: Container = Depends(container)) -> TeachingSessionView:
    return await c.teaching_service.view(session_id)


@router.post("/teaching-sessions/{session_id}/answers", response_model=AnswerResult)
async def submit_answer(session_id: str, body: LearnerInput, c: Container = Depends(container)) -> AnswerResult:
    """An answer (`kind: answer`), a question to the teacher (`question`) or a request to stop (`stop`). Idempotent
    per `client_turn_id`."""
    return await c.teaching_service.submit(session_id, body)


@router.post("/teaching-sessions/{session_id}/pause", response_model=TeachingSessionView)
async def pause_session(session_id: str, c: Container = Depends(container)) -> TeachingSessionView:
    return await c.teaching_service.pause(session_id)


@router.post("/teaching-sessions/{session_id}/resume", response_model=TeachingSessionView)
async def resume_session(session_id: str, c: Container = Depends(container)) -> TeachingSessionView:
    return await c.teaching_service.resume(session_id)


@router.post("/teaching-sessions/{session_id}/cancel", response_model=TeachingSessionView)
async def cancel_session(session_id: str, c: Container = Depends(container)) -> TeachingSessionView:
    return await c.teaching_service.cancel(session_id)
