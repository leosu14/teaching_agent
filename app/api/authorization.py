"""The authorization boundary: every protected route depends on one of these before it calls a service.

- `principal`: the request's authenticated principal (401 without one).
- `require_learner`, `owned_*`: the resource's owning learner, read from stored data (`OwnershipService`), must be in
  the principal's learner scope. A resource outside the scope gets exactly the same 404 as one that does not exist,
  so ids cannot be probed for existence.
- `require_role`: a capability beyond learner scope (403), e.g. registering assessment items.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.api.auth import AuthenticationFailed, Authenticator, Principal
from app.api.deps import container
from app.services.container import Container


class ResourceNotFound(Exception):
    """Not found, or not visible to this principal: the two are deliberately indistinguishable (404)."""

    def __init__(self) -> None:
        super().__init__("not found")


class Forbidden(Exception):
    """Authenticated, but missing a role the action needs (403)."""


def principal(request: Request) -> Principal:
    authenticator: Authenticator | None = getattr(request.app.state, "authenticator", None)
    if authenticator is None:  # never anonymous: an app without an authenticator rejects everything
        raise AuthenticationFailed("no authenticator configured")
    found = authenticator.authenticate(request)
    request.state.principal = found
    return found


def authorize(p: Principal, owner: str | None) -> str:
    """The owning learner id when `p` may act for it; ResourceNotFound otherwise (also when there is no owner)."""
    if owner is None or not p.may_access(owner):
        raise ResourceNotFound()
    return owner


def require_role(p: Principal, role: str) -> None:
    if not p.has_role(role):
        raise Forbidden(f"this action needs the {role!r} role")


def require_learner(learner_id: str, p: Principal = Depends(principal)) -> str:
    return authorize(p, learner_id)


def owned_task(task_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.task(task_id))


def owned_goal(goal_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.goal(goal_id))


def owned_lesson(lesson_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.lesson(lesson_id))


def owned_session(session_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.teaching_session(session_id))


def owned_item(item_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.assessment_item(item_id))


def owned_attempt(attempt_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.assessment_attempt(attempt_id))


def owned_cycle(cycle_id: str, p: Principal = Depends(principal), c: Container = Depends(container)) -> str:
    return authorize(p, c.ownership.learning_cycle(cycle_id))
