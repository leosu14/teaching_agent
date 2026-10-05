"""Errors of the assessment layer. The API maps them to status codes."""

from __future__ import annotations

from app.schemas.assessment import AttemptConflict, AttemptExists, ItemConflict

__all__ = ["AssessmentItemNotFound", "AttemptConflict", "AttemptExists", "AttemptNotFound", "InvalidAssessmentRequest",
           "ItemConflict"]


class AssessmentItemNotFound(KeyError):
    """No such assessment item (404)."""


class AttemptNotFound(KeyError):
    """No such attempt (404)."""


class InvalidAssessmentRequest(ValueError):
    """The request cannot be served: an unknown lesson or rubric, a lesson without the item's concept (422)."""
