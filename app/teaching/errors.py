"""Errors of the interactive teaching layer. The API maps them to status codes."""

from __future__ import annotations

from app.schemas.teaching import SessionConflict

__all__ = ["InvalidSessionTransition", "InvalidTeachingRequest", "SessionConflict", "TeacherUnavailable",
           "TeachingSessionNotFound"]


class TeachingSessionNotFound(KeyError):
    """No such session (404)."""


class InvalidSessionTransition(ValueError):
    """The session's status does not allow the request, e.g. an answer to a paused or cancelled session (409)."""


class InvalidTeachingRequest(ValueError):
    """The request cannot be served: an unknown objective, a lesson without teachable material (422)."""


class TeacherUnavailable(RuntimeError):
    """The teacher's reply could not be generated (the model failed validation or every provider failed). What the
    learner sent is stored; resuming the session generates the reply (503).

    Its text is served to learners, so it is always the fixed `MESSAGE`: the cause (validation errors quote the model's
    output, which can hold an answer key) stays on `__cause__` and in the teacher's internal agent events."""

    MESSAGE = "The teacher could not produce a response. You can retry the session."

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)

    def __str__(self) -> str:
        return self.MESSAGE
